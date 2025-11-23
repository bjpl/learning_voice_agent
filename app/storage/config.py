"""
Storage Configuration

SPECIFICATION:
- File storage paths and organization
- Size limits and retention policies
- Cleanup schedules
- File type configurations

ARCHITECTURE:
- Dataclass-based configuration
- Environment variable overrides
- Type-safe settings
- Validation on initialization
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import os


@dataclass
class FileTypeConfig:
    """Configuration for specific file type"""
    max_size_mb: int
    allowed_extensions: list[str]
    storage_path: str
    retention_days: int
    enable_deduplication: bool = True
    enable_compression: bool = False


@dataclass
class StorageConfig:
    """
    Storage system configuration

    PATTERN: Centralized configuration with defaults
    WHY: Single source of truth, easy testing

    Attributes:
        base_directory: Root directory for all uploads
        max_storage_per_user_gb: Maximum storage per user in GB
        cleanup_interval_hours: Hours between cleanup runs
        deduplication_enabled: Enable hash-based deduplication
        retention_days: Default retention for all files
        file_types: Configuration per file type
    """

    # Base paths
    base_directory: str = "./data/uploads"
    metadata_db_path: str = "./data/storage_metadata.db"

    # Storage limits
    max_storage_per_user_gb: float = 1.0
    max_total_storage_gb: float = 100.0

    # Cleanup settings
    cleanup_interval_hours: int = 24
    retention_days: int = 30
    enable_auto_cleanup: bool = True

    # Deduplication
    deduplication_enabled: bool = True
    hash_algorithm: str = "sha256"

    # File type configurations
    file_types: Dict[str, FileTypeConfig] = field(default_factory=lambda: {
        "image": FileTypeConfig(
            max_size_mb=10,
            allowed_extensions=[".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"],
            storage_path="images",
            retention_days=30,
            enable_deduplication=True,
            enable_compression=False
        ),
        "pdf": FileTypeConfig(
            max_size_mb=25,
            allowed_extensions=[".pdf"],
            storage_path="documents",
            retention_days=90,
            enable_deduplication=True,
            enable_compression=False
        ),
        "docx": FileTypeConfig(
            max_size_mb=20,
            allowed_extensions=[".docx", ".doc"],
            storage_path="documents",
            retention_days=90,
            enable_deduplication=True,
            enable_compression=False
        ),
        "txt": FileTypeConfig(
            max_size_mb=5,
            allowed_extensions=[".txt", ".md", ".rst"],
            storage_path="documents",
            retention_days=60,
            enable_deduplication=True,
            enable_compression=True
        ),
        "audio": FileTypeConfig(
            max_size_mb=50,
            allowed_extensions=[".mp3", ".wav", ".m4a", ".ogg", ".flac"],
            storage_path="audio",
            retention_days=30,
            enable_deduplication=True,
            enable_compression=False
        ),
        "video": FileTypeConfig(
            max_size_mb=100,
            allowed_extensions=[".mp4", ".webm", ".mov", ".avi"],
            storage_path="video",
            retention_days=14,
            enable_deduplication=True,
            enable_compression=False
        )
    })

    # Indexing settings
    enable_vector_indexing: bool = True
    enable_fulltext_indexing: bool = True
    enable_knowledge_graph: bool = True

    # Analysis settings
    enable_vision_analysis: bool = True
    enable_ocr: bool = True
    enable_document_extraction: bool = True

    def __post_init__(self):
        """Validate configuration and apply environment overrides"""
        # Apply environment variable overrides
        self.base_directory = os.getenv("STORAGE_BASE_DIR", self.base_directory)
        self.metadata_db_path = os.getenv("STORAGE_METADATA_DB", self.metadata_db_path)

        if env_retention := os.getenv("STORAGE_RETENTION_DAYS"):
            self.retention_days = int(env_retention)

        if env_max_storage := os.getenv("STORAGE_MAX_USER_GB"):
            self.max_storage_per_user_gb = float(env_max_storage)

        # Create base directory if it doesn't exist
        Path(self.base_directory).mkdir(parents=True, exist_ok=True)

    def get_file_type_config(self, file_type: str) -> Optional[FileTypeConfig]:
        """Get configuration for specific file type"""
        return self.file_types.get(file_type.lower())

    def get_storage_path(self, file_type: str, session_id: str) -> Path:
        """
        Get organized storage path for file

        PATTERN: Hierarchical organization by date and session
        WHY: Efficient filesystem operations, easy cleanup

        Returns:
            Path: {base_dir}/{year}/{month}/{session_id}/
        """
        from datetime import datetime

        now = datetime.utcnow()
        year = str(now.year)
        month = f"{now.month:02d}"

        type_config = self.get_file_type_config(file_type)
        subdir = type_config.storage_path if type_config else "misc"

        path = Path(self.base_directory) / subdir / year / month / session_id
        path.mkdir(parents=True, exist_ok=True)

        return path

    def is_file_type_allowed(self, file_type: str, extension: str) -> bool:
        """Check if file extension is allowed for type"""
        type_config = self.get_file_type_config(file_type)
        if not type_config:
            return False

        return extension.lower() in type_config.allowed_extensions

    def is_file_size_allowed(self, file_type: str, size_bytes: int) -> bool:
        """Check if file size is within limits"""
        type_config = self.get_file_type_config(file_type)
        if not type_config:
            return False

        max_bytes = type_config.max_size_mb * 1024 * 1024
        return size_bytes <= max_bytes

    def get_max_size_mb(self, file_type: str) -> int:
        """Get maximum file size in MB for type"""
        type_config = self.get_file_type_config(file_type)
        return type_config.max_size_mb if type_config else 10

    def get_retention_days(self, file_type: str) -> int:
        """Get retention period in days for file type"""
        type_config = self.get_file_type_config(file_type)
        return type_config.retention_days if type_config else self.retention_days

    def validate_file(self, file_type: str, filename: str, size_bytes: int) -> tuple[bool, Optional[str]]:
        """
        Validate file against configuration rules

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file type is configured
        type_config = self.get_file_type_config(file_type)
        if not type_config:
            return False, f"Unsupported file type: {file_type}"

        # Check extension
        extension = Path(filename).suffix
        if not self.is_file_type_allowed(file_type, extension):
            return False, f"File extension {extension} not allowed for type {file_type}"

        # Check size
        if not self.is_file_size_allowed(file_type, size_bytes):
            max_mb = type_config.max_size_mb
            actual_mb = size_bytes / (1024 * 1024)
            return False, f"File size {actual_mb:.2f}MB exceeds limit of {max_mb}MB"

        return True, None


# Global storage configuration instance
storage_config = StorageConfig()
