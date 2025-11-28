"""
File Manager for Multi-Modal Content

SPECIFICATION:
- File upload and storage with organization
- Hash-based deduplication
- Automatic cleanup of old files
- Session and type-based organization
- Secure file operations

ARCHITECTURE:
- Async file I/O operations
- Integration with metadata store
- Automatic directory management
- Error handling and validation
"""

import hashlib
import mimetypes
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, BinaryIO, Any
import aiofiles
import aiofiles.os

from app.storage.config import storage_config
from app.storage.metadata_store import metadata_store
from app.logger import get_logger
from app.resilience import with_retry

logger = get_logger(__name__)


class FileManager:
    """
    File manager for multi-modal content storage

    PATTERN: Facade pattern for file operations
    WHY: Simplified interface for complex file management

    Features:
    - Hierarchical directory organization (year/month/session)
    - SHA256-based deduplication
    - Automatic cleanup (configurable retention)
    - Size and type validation
    - Metadata tracking

    Example:
        manager = FileManager()
        await manager.initialize()

        # Save file
        result = await manager.save_file(
            file_data=file_bytes,
            original_filename="photo.jpg",
            file_type="image",
            session_id="session_123",
            metadata={"source": "upload", "user_id": "user_456"}
        )

        # Retrieve file
        file_data = await manager.get_file(result["file_id"])

        # List files
        files = await manager.list_files(session_id="session_123")

        # Cleanup old files
        deleted = await manager.cleanup_old_files()
    """

    def __init__(self, config=None):
        self.config = config or storage_config
        self.metadata_store = metadata_store
        self._initialized = False

    async def initialize(self):
        """Initialize file manager and metadata store"""
        if self._initialized:
            return

        try:
            logger.info("file_manager_initialization_started")

            # Initialize metadata store
            await self.metadata_store.initialize()

            # Ensure base directory exists
            base_path = Path(self.config.base_directory)
            base_path.mkdir(parents=True, exist_ok=True)

            self._initialized = True
            logger.info(
                "file_manager_initialized",
                base_dir=str(base_path),
                deduplication=self.config.deduplication_enabled
            )

        except Exception as e:
            logger.error(
                "file_manager_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    @with_retry(max_attempts=3, min_wait=0.5)
    async def save_file(
        self,
        file_data: bytes,
        original_filename: str,
        file_type: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save file with organization and metadata

        PATTERN: Hash-first deduplication with metadata linking
        WHY: Save storage space, fast duplicate detection

        Args:
            file_data: File content as bytes
            original_filename: Original filename
            file_type: File type (image, pdf, docx, txt, audio, video)
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            Dictionary with file_id, stored_path, file_hash, deduplicated

        Raises:
            ValueError: If file validation fails
        """
        try:
            # Validate file
            file_size = len(file_data)
            is_valid, error_msg = self.config.validate_file(
                file_type, original_filename, file_size
            )
            if not is_valid:
                logger.warning("file_validation_failed", error=error_msg)
                raise ValueError(error_msg)

            # Calculate file hash for deduplication
            file_hash = self._calculate_hash(file_data)

            # Check for existing file (deduplication)
            deduplicated = False
            if self.config.deduplication_enabled:
                existing = await self.metadata_store.get_file_by_hash(file_hash)
                if existing:
                    logger.info(
                        "file_deduplicated",
                        file_hash=file_hash[:16],
                        existing_file_id=existing["file_id"]
                    )

                    # Create new metadata entry pointing to same file
                    file_id = self._generate_file_id(session_id, file_hash)
                    await self.metadata_store.save_file_metadata(
                        file_id=file_id,
                        session_id=session_id,
                        file_type=file_type,
                        original_filename=original_filename,
                        stored_path=existing["stored_path"],  # Reuse existing path
                        file_size=file_size,
                        mime_type=self._guess_mime_type(original_filename),
                        file_hash=file_hash,
                        metadata={**(metadata or {}), "deduplicated": True}
                    )

                    return {
                        "file_id": file_id,
                        "stored_path": existing["stored_path"],
                        "file_hash": file_hash,
                        "file_size": file_size,
                        "deduplicated": True,
                        "original_file_id": existing["file_id"]
                    }

            # Generate file ID and path
            file_id = self._generate_file_id(session_id, file_hash)
            storage_path = self._get_storage_path(file_type, session_id, file_id, original_filename)

            # Save file to disk
            await self._write_file(storage_path, file_data)

            # Save metadata
            await self.metadata_store.save_file_metadata(
                file_id=file_id,
                session_id=session_id,
                file_type=file_type,
                original_filename=original_filename,
                stored_path=str(storage_path),
                file_size=file_size,
                mime_type=self._guess_mime_type(original_filename),
                file_hash=file_hash,
                metadata=metadata
            )

            logger.info(
                "file_saved",
                file_id=file_id,
                file_type=file_type,
                size_kb=file_size // 1024,
                session_id=session_id
            )

            return {
                "file_id": file_id,
                "stored_path": str(storage_path),
                "file_hash": file_hash,
                "file_size": file_size,
                "deduplicated": False
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "save_file_failed",
                filename=original_filename,
                file_type=file_type,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def get_file(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve file content by ID

        Args:
            file_id: File identifier

        Returns:
            File content as bytes, or None if not found
        """
        try:
            # Get metadata
            metadata = await self.metadata_store.get_file_by_id(file_id)
            if not metadata:
                logger.warning("file_not_found", file_id=file_id)
                return None

            # Read file
            stored_path = Path(metadata["stored_path"])
            if not stored_path.exists():
                logger.error("file_missing_on_disk", file_id=file_id, path=str(stored_path))
                return None

            file_data = await self._read_file(stored_path)

            logger.debug(
                "file_retrieved",
                file_id=file_id,
                size_kb=len(file_data) // 1024
            )

            return file_data

        except Exception as e:
            logger.error(
                "get_file_failed",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            return None

    async def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path without reading content"""
        try:
            metadata = await self.metadata_store.get_file_by_id(file_id)
            return metadata["stored_path"] if metadata else None
        except Exception as e:
            logger.error("get_file_path_failed", file_id=file_id, error=str(e))
            return None

    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata"""
        return await self.metadata_store.get_file_by_id(file_id)

    async def list_files(
        self,
        session_id: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List files with optional filters

        Args:
            session_id: Filter by session
            file_type: Filter by type
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of file metadata dictionaries
        """
        return await self.metadata_store.list_files(
            session_id=session_id,
            file_type=file_type,
            limit=limit,
            offset=offset
        )

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete file and metadata

        PATTERN: Metadata-first deletion with error recovery
        WHY: Consistent state even if disk deletion fails

        Args:
            file_id: File identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            # Get metadata first
            metadata = await self.metadata_store.get_file_by_id(file_id)
            if not metadata:
                logger.warning("delete_file_not_found", file_id=file_id)
                return False

            stored_path = Path(metadata["stored_path"])

            # Check if other files reference this path (deduplication)
            file_hash = metadata.get("file_hash")
            if file_hash:
                # Check if other files share this hash
                existing = await self.metadata_store.get_file_by_hash(file_hash)
                if existing and existing["file_id"] != file_id:
                    # Other files reference this, don't delete from disk
                    logger.info(
                        "file_shared_skipping_disk_delete",
                        file_id=file_id,
                        file_hash=file_hash[:16]
                    )
                    # Only delete metadata
                    await self.metadata_store.delete_file(file_id)
                    return True

            # Delete from metadata store
            await self.metadata_store.delete_file(file_id)

            # Delete from disk
            if stored_path.exists():
                await aiofiles.os.remove(stored_path)
                logger.info("file_deleted", file_id=file_id, path=str(stored_path))
            else:
                logger.warning("file_not_on_disk", file_id=file_id, path=str(stored_path))

            return True

        except Exception as e:
            logger.error(
                "delete_file_failed",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def cleanup_old_files(
        self,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up files older than retention policy

        PATTERN: Batch cleanup with progress tracking
        WHY: Efficient resource management, prevent disk exhaustion

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            logger.info("cleanup_started", dry_run=dry_run)

            deleted_count = 0
            freed_bytes = 0
            errors = []

            # Get old files by type
            for file_type, type_config in self.config.file_types.items():
                retention_days = type_config.retention_days
                old_files = await self.metadata_store.get_old_files(retention_days)

                logger.info(
                    "cleanup_processing_type",
                    file_type=file_type,
                    retention_days=retention_days,
                    old_files_count=len(old_files)
                )

                for file_info in old_files:
                    try:
                        if not dry_run:
                            success = await self.delete_file(file_info["file_id"])
                            if success:
                                deleted_count += 1
                                freed_bytes += file_info.get("file_size", 0)
                        else:
                            deleted_count += 1
                            freed_bytes += file_info.get("file_size", 0)
                            logger.debug(
                                "cleanup_would_delete",
                                file_id=file_info["file_id"],
                                age_days=(datetime.utcnow() - datetime.fromisoformat(file_info["uploaded_at"])).days
                            )
                    except Exception as e:
                        errors.append({
                            "file_id": file_info["file_id"],
                            "error": str(e)
                        })

            stats = {
                "deleted_count": deleted_count,
                "freed_bytes": freed_bytes,
                "freed_mb": freed_bytes / (1024 * 1024),
                "errors_count": len(errors),
                "errors": errors[:10],  # First 10 errors
                "dry_run": dry_run
            }

            logger.info(
                "cleanup_complete",
                **{k: v for k, v in stats.items() if k != "errors"}
            )

            return stats

        except Exception as e:
            logger.error(
                "cleanup_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        try:
            # Get metadata stats
            stats = await self.metadata_store.get_storage_stats()

            # Add configuration info
            stats["config"] = {
                "base_directory": self.config.base_directory,
                "max_storage_gb": self.config.max_storage_per_user_gb,
                "retention_days": self.config.retention_days,
                "deduplication_enabled": self.config.deduplication_enabled
            }

            # Calculate storage percentage
            if stats.get("total_bytes"):
                max_bytes = self.config.max_total_storage_gb * 1024 * 1024 * 1024
                stats["storage_used_percent"] = (stats["total_bytes"] / max_bytes) * 100

            return stats

        except Exception as e:
            logger.error("get_storage_stats_failed", error=str(e))
            return {}

    # Private helper methods

    def _calculate_hash(self, data: bytes) -> str:
        """Calculate SHA256 hash of file data"""
        hash_obj = hashlib.sha256(data)
        return hash_obj.hexdigest()

    def _generate_file_id(self, session_id: str, file_hash: str) -> str:
        """Generate unique file ID"""
        # Use session + hash prefix for uniqueness
        return f"{session_id}_{file_hash[:16]}"

    def _get_storage_path(
        self,
        file_type: str,
        session_id: str,
        file_id: str,
        original_filename: str
    ) -> Path:
        """Get full storage path for file"""
        # Get organized directory path
        dir_path = self.config.get_storage_path(file_type, session_id)

        # Determine extension
        extension = Path(original_filename).suffix or self._guess_extension(file_type)

        # Create filename: {file_id}{extension}
        filename = f"{file_id}{extension}"

        return dir_path / filename

    def _guess_extension(self, file_type: str) -> str:
        """Guess file extension from type"""
        type_config = self.config.get_file_type_config(file_type)
        if type_config and type_config.allowed_extensions:
            return type_config.allowed_extensions[0]
        return ".bin"

    def _guess_mime_type(self, filename: str) -> Optional[str]:
        """Guess MIME type from filename"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    async def _write_file(self, path: Path, data: bytes):
        """Write file to disk asynchronously"""
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)

    async def _read_file(self, path: Path) -> bytes:
        """Read file from disk asynchronously"""
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()


# Global file manager instance
file_manager = FileManager()
