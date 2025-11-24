"""
File Manager - Upload and Storage Handling

SPECIFICATION:
- Validate file types (images: PNG, JPEG, GIF, WebP; documents: PDF, DOCX, TXT, MD)
- Enforce file size limits (images: 5MB, documents: 10MB)
- Store files with unique IDs
- Retrieve files by ID
- Clean up old files

ARCHITECTURE:
- File validation with magic bytes
- UUID-based file naming
- Directory organization by type
- Async file operations

PATTERN: Service layer with validation and storage abstraction
WHY: Centralized file handling with security checks
RESILIENCE: File type validation, size limits, error handling
"""

import os
import uuid
import hashlib
import magic
import aiofiles
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from app.logger import api_logger
from app.config import settings


class FileManager:
    """
    File upload and storage manager

    PATTERN: Service class with validation and storage
    WHY: Centralized file handling with security
    """

    # Supported file types and their MIME types
    SUPPORTED_IMAGES = {
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'image/gif': '.gif',
        'image/webp': '.webp'
    }

    SUPPORTED_DOCUMENTS = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'text/plain': '.txt',
        'text/markdown': '.md'
    }

    # Size limits (in bytes)
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, base_upload_dir: str = "uploads"):
        """
        Initialize file manager

        Args:
            base_upload_dir: Base directory for uploads
        """
        self.base_upload_dir = Path(base_upload_dir)
        self.images_dir = self.base_upload_dir / "images"
        self.documents_dir = self.base_upload_dir / "documents"

        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        api_logger.info(
            "file_manager_initialized",
            base_dir=str(self.base_upload_dir),
            images_dir=str(self.images_dir),
            documents_dir=str(self.documents_dir)
        )

    def validate_file_type(
        self,
        file_data: bytes,
        file_type: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate file type using magic bytes

        PATTERN: Security validation with magic bytes
        WHY: Prevent file type spoofing

        Args:
            file_data: Raw file bytes
            file_type: Expected type ('image' or 'document')

        Returns:
            (is_valid, mime_type, extension)
        """
        try:
            # Detect MIME type from file content
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_buffer(file_data)

            api_logger.debug(
                "file_type_detection",
                detected_mime=detected_mime,
                expected_type=file_type
            )

            # Check against supported types
            if file_type == "image":
                if detected_mime in self.SUPPORTED_IMAGES:
                    return True, detected_mime, self.SUPPORTED_IMAGES[detected_mime]
            elif file_type == "document":
                if detected_mime in self.SUPPORTED_DOCUMENTS:
                    return True, detected_mime, self.SUPPORTED_DOCUMENTS[detected_mime]

            api_logger.warning(
                "file_type_validation_failed",
                detected_mime=detected_mime,
                expected_type=file_type
            )
            return False, detected_mime, None

        except Exception as e:
            api_logger.error(
                "file_type_detection_error",
                error=str(e),
                exc_info=True
            )
            return False, None, None

    def validate_file_size(
        self,
        file_size: int,
        file_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate file size

        Args:
            file_size: File size in bytes
            file_type: File type ('image' or 'document')

        Returns:
            (is_valid, error_message)
        """
        max_size = self.MAX_IMAGE_SIZE if file_type == "image" else self.MAX_DOCUMENT_SIZE

        if file_size > max_size:
            max_size_mb = max_size / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_size_mb}MB"

        if file_size == 0:
            return False, "File is empty"

        return True, None

    async def save_file(
        self,
        file_data: bytes,
        original_filename: str,
        file_type: str,
        session_id: str
    ) -> Dict:
        """
        Save uploaded file

        ALGORITHM:
        1. Validate file type
        2. Validate file size
        3. Generate unique file ID
        4. Calculate file hash
        5. Save to disk
        6. Return metadata

        Args:
            file_data: Raw file bytes
            original_filename: Original filename
            file_type: 'image' or 'document'
            session_id: Session ID for organization

        Returns:
            File metadata dictionary

        Raises:
            ValueError: If validation fails
        """
        # Validate file type
        is_valid, mime_type, extension = self.validate_file_type(file_data, file_type)
        if not is_valid:
            raise ValueError(f"Unsupported file type: {mime_type}")

        # Validate file size
        is_valid, error_msg = self.validate_file_size(len(file_data), file_type)
        if not is_valid:
            raise ValueError(error_msg)

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(file_data).hexdigest()

        # Determine storage directory
        storage_dir = self.images_dir if file_type == "image" else self.documents_dir

        # Generate stored filename
        stored_filename = f"{file_id}{extension}"
        stored_path = storage_dir / stored_filename

        # Save file
        try:
            async with aiofiles.open(stored_path, 'wb') as f:
                await f.write(file_data)

            api_logger.info(
                "file_saved",
                file_id=file_id,
                file_type=file_type,
                mime_type=mime_type,
                size_bytes=len(file_data),
                session_id=session_id
            )

            # Return metadata
            return {
                "file_id": file_id,
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "stored_path": str(stored_path),
                "file_type": file_type,
                "mime_type": mime_type,
                "file_size": len(file_data),
                "file_hash": file_hash,
                "session_id": session_id,
                "upload_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            api_logger.error(
                "file_save_error",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_file(self, file_id: str, file_type: str) -> Optional[Tuple[bytes, str]]:
        """
        Retrieve file by ID

        Args:
            file_id: Unique file ID
            file_type: 'image' or 'document'

        Returns:
            (file_data, mime_type) or None if not found
        """
        try:
            # Search for file with any extension
            storage_dir = self.images_dir if file_type == "image" else self.documents_dir

            for file_path in storage_dir.glob(f"{file_id}.*"):
                async with aiofiles.open(file_path, 'rb') as f:
                    file_data = await f.read()

                # Detect MIME type
                mime = magic.Magic(mime=True)
                mime_type = mime.from_buffer(file_data)

                api_logger.debug(
                    "file_retrieved",
                    file_id=file_id,
                    mime_type=mime_type,
                    size_bytes=len(file_data)
                )

                return file_data, mime_type

            api_logger.warning("file_not_found", file_id=file_id, file_type=file_type)
            return None

        except Exception as e:
            api_logger.error(
                "file_retrieval_error",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            return None

    async def delete_file(self, file_id: str, file_type: str) -> bool:
        """
        Delete file by ID

        Args:
            file_id: Unique file ID
            file_type: 'image' or 'document'

        Returns:
            True if deleted, False if not found
        """
        try:
            storage_dir = self.images_dir if file_type == "image" else self.documents_dir

            for file_path in storage_dir.glob(f"{file_id}.*"):
                file_path.unlink()
                api_logger.info("file_deleted", file_id=file_id, file_type=file_type)
                return True

            return False

        except Exception as e:
            api_logger.error(
                "file_deletion_error",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            return False


# Singleton instance
file_manager = FileManager()
