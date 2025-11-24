"""
Sync Configuration - Constants and Settings for Sync/Backup
PATTERN: Centralized configuration with environment overrides
WHY: Easy configuration management and deployment flexibility

Configuration:
- Backup format version
- Size limits and intervals
- File naming conventions
- Encryption settings
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ============================================================================
# CORE CONSTANTS
# ============================================================================

# Backup format version - increment when schema changes
BACKUP_VERSION: str = "1.0"

# Maximum backup file size in megabytes
MAX_BACKUP_SIZE_MB: int = 100

# Auto-backup interval in hours (0 to disable)
AUTO_BACKUP_INTERVAL_HOURS: int = 24

# Enable encryption for backup files
BACKUP_ENCRYPTION_ENABLED: bool = False

# Compression level (0-9, 0=none, 9=max)
BACKUP_COMPRESSION_LEVEL: int = 6

# Maximum number of backup files to retain
MAX_BACKUP_RETENTION_COUNT: int = 10

# Maximum age of backup files in days (0 = no limit)
MAX_BACKUP_AGE_DAYS: int = 90


# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================

def get_backup_filename(
    device_id: str,
    timestamp: Optional[datetime] = None,
    compressed: bool = True,
    encrypted: bool = False
) -> str:
    """
    Generate backup filename following naming convention.

    Format: backup_{device_id}_{timestamp}.{ext}

    Args:
        device_id: Device identifier
        timestamp: Backup timestamp (defaults to now)
        compressed: Whether backup is compressed
        encrypted: Whether backup is encrypted

    Returns:
        Formatted filename string
    """
    ts = timestamp or datetime.utcnow()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")

    # Determine extension
    if encrypted:
        ext = "enc.json.gz" if compressed else "enc.json"
    elif compressed:
        ext = "json.gz"
    else:
        ext = "json"

    return f"backup_{device_id}_{ts_str}.{ext}"


def parse_backup_filename(filename: str) -> Optional[dict]:
    """
    Parse backup filename to extract metadata.

    Args:
        filename: Backup filename to parse

    Returns:
        Dict with device_id, timestamp, compressed, encrypted or None if invalid
    """
    try:
        # Remove directory path if present
        name = filename.split("/")[-1].split("\\")[-1]

        # Check prefix
        if not name.startswith("backup_"):
            return None

        # Remove prefix
        name = name[7:]  # len("backup_") = 7

        # Determine flags from extension
        encrypted = ".enc." in name
        compressed = name.endswith(".gz")

        # Extract device_id and timestamp
        parts = name.split("_")
        if len(parts) < 3:
            return None

        device_id = parts[0]
        timestamp_str = f"{parts[1]}_{parts[2].split('.')[0]}"

        # Parse timestamp
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

        return {
            "device_id": device_id,
            "timestamp": timestamp,
            "compressed": compressed,
            "encrypted": encrypted,
        }
    except (ValueError, IndexError):
        return None


# ============================================================================
# SYNC CONFIGURATION CLASS
# ============================================================================

@dataclass
class SyncConfig:
    """
    Comprehensive sync configuration.

    PATTERN: Configuration dataclass with environment overrides
    WHY: Centralized, type-safe configuration
    """
    # Version control
    backup_version: str = BACKUP_VERSION

    # Size limits
    max_backup_size_mb: int = MAX_BACKUP_SIZE_MB
    max_backup_size_bytes: int = field(init=False)

    # Timing
    auto_backup_interval_hours: int = AUTO_BACKUP_INTERVAL_HOURS
    sync_timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 5

    # Retention
    max_backup_retention_count: int = MAX_BACKUP_RETENTION_COUNT
    max_backup_age_days: int = MAX_BACKUP_AGE_DAYS

    # Security
    encryption_enabled: bool = BACKUP_ENCRYPTION_ENABLED
    compression_level: int = BACKUP_COMPRESSION_LEVEL

    # Storage paths
    backup_directory: str = "data/backups"
    sync_database_path: str = "data/sync.db"

    # Sync behavior
    auto_sync_on_change: bool = True
    sync_debounce_seconds: int = 30
    conflict_auto_resolve: bool = True

    # Device limits
    max_devices_per_user: int = 10

    def __post_init__(self):
        """Calculate derived values and apply environment overrides."""
        # Calculate max size in bytes
        self.max_backup_size_bytes = self.max_backup_size_mb * 1024 * 1024

        # Apply environment overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "BACKUP_VERSION": ("backup_version", str),
            "MAX_BACKUP_SIZE_MB": ("max_backup_size_mb", int),
            "AUTO_BACKUP_INTERVAL_HOURS": ("auto_backup_interval_hours", int),
            "BACKUP_ENCRYPTION_ENABLED": ("encryption_enabled", lambda x: x.lower() == "true"),
            "BACKUP_COMPRESSION_LEVEL": ("compression_level", int),
            "BACKUP_DIRECTORY": ("backup_directory", str),
            "SYNC_DATABASE_PATH": ("sync_database_path", str),
            "MAX_DEVICES_PER_USER": ("max_devices_per_user", int),
        }

        for env_key, (attr, converter) in env_mappings.items():
            env_value = os.environ.get(env_key)
            if env_value is not None:
                try:
                    setattr(self, attr, converter(env_value))
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails

        # Recalculate derived values
        self.max_backup_size_bytes = self.max_backup_size_mb * 1024 * 1024

    def get_backup_path(self, filename: str) -> str:
        """Get full path for a backup file."""
        return f"{self.backup_directory}/{filename}"

    def validate(self) -> list[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.max_backup_size_mb <= 0:
            errors.append("max_backup_size_mb must be positive")

        if self.max_backup_size_mb > 1024:
            errors.append("max_backup_size_mb should not exceed 1024 MB")

        if self.auto_backup_interval_hours < 0:
            errors.append("auto_backup_interval_hours cannot be negative")

        if not 0 <= self.compression_level <= 9:
            errors.append("compression_level must be between 0 and 9")

        if self.max_devices_per_user <= 0:
            errors.append("max_devices_per_user must be positive")

        if self.sync_timeout_seconds <= 0:
            errors.append("sync_timeout_seconds must be positive")

        return errors


# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Global sync configuration instance
sync_config = SyncConfig()


# ============================================================================
# MIME TYPES AND FILE FORMATS
# ============================================================================

BACKUP_MIME_TYPES = {
    "json": "application/json",
    "json.gz": "application/gzip",
    "enc.json": "application/octet-stream",
    "enc.json.gz": "application/octet-stream",
}

SUPPORTED_IMPORT_VERSIONS = ["1.0"]


# ============================================================================
# SYNC EVENT TYPES
# ============================================================================

class SyncEventType:
    """Constants for sync event types."""
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    DEVICE_REGISTERED = "device_registered"
    DEVICE_REMOVED = "device_removed"
