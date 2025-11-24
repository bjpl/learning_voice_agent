"""
Sync Models - Pydantic Models for Sync and Backup Data
PATTERN: Contract-first development with type safety
WHY: Ensure data integrity for sync/backup operations

Models:
- SyncMetadata: Sync state tracking (last_sync, device_id, version, checksum)
- BackupData: Complete backup payload with all user data
- ExportRequest: Configuration for data export
- ImportRequest: Configuration for data import with merge strategy
- SyncConflict: Conflict detection and resolution
- DeviceInfo: Device registration and tracking
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import uuid
import hashlib


class MergeStrategy(str, Enum):
    """Strategy for handling data during import."""
    REPLACE = "replace"     # Replace all local data with imported data
    MERGE = "merge"         # Merge imported data with local data
    KEEP_NEWER = "keep_newer"  # Keep the newer version based on timestamps


class SyncStatus(str, Enum):
    """Status of sync operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class ConflictResolution(str, Enum):
    """How a conflict was resolved."""
    LOCAL = "local"         # Kept local value
    REMOTE = "remote"       # Used remote value
    MERGED = "merged"       # Values were merged
    MANUAL = "manual"       # User manually resolved


# ============================================================================
# CORE SYNC MODELS
# ============================================================================

class SyncMetadata(BaseModel):
    """
    Metadata for tracking sync state.

    PATTERN: Optimistic concurrency with checksums
    WHY: Enable efficient delta syncing and conflict detection
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    last_sync: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of last successful sync"
    )
    device_id: str = Field(..., description="Unique device identifier")
    version: str = Field(..., description="Backup format version")
    checksum: str = Field(
        ...,
        description="SHA-256 checksum of backup data for integrity verification"
    )
    sync_status: SyncStatus = Field(
        default=SyncStatus.COMPLETED,
        description="Current sync status"
    )
    data_size_bytes: int = Field(
        default=0,
        ge=0,
        description="Size of synced data in bytes"
    )
    item_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of items by type (conversations, feedback, etc.)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('checksum')
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        """Validate checksum format (SHA-256 hex)."""
        if len(v) != 64:
            raise ValueError('Checksum must be 64 character SHA-256 hex string')
        try:
            int(v, 16)
        except ValueError:
            raise ValueError('Checksum must be valid hexadecimal')
        return v.lower()

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sync_abc123",
                "last_sync": "2025-11-21T10:00:00Z",
                "device_id": "device_xyz789",
                "version": "1.0",
                "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "sync_status": "completed",
                "data_size_bytes": 1024000,
                "item_counts": {
                    "conversations": 50,
                    "feedback": 120,
                    "goals": 5
                }
            }
        }


class ConversationBackup(BaseModel):
    """Backup format for a conversation."""
    id: str = Field(..., description="Conversation/session ID")
    user_text: str = Field(..., description="User's message")
    agent_text: str = Field(..., description="Agent's response")
    intent: str = Field(default="statement", description="Detected intent")
    topics: List[str] = Field(default_factory=list, description="Extracted topics")
    timestamp: datetime = Field(..., description="When the exchange occurred")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class FeedbackBackup(BaseModel):
    """Backup format for feedback data."""
    id: str = Field(..., description="Feedback ID")
    session_id: str = Field(..., description="Related session")
    feedback_type: str = Field(..., description="explicit/implicit/correction")
    rating: Optional[int] = Field(None, ge=1, le=5)
    helpful: Optional[bool] = None
    comment: Optional[str] = None
    timestamp: datetime = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class GoalBackup(BaseModel):
    """Backup format for goals."""
    id: str = Field(..., description="Goal ID")
    title: str = Field(...)
    description: Optional[str] = None
    goal_type: str = Field(...)
    target_value: float = Field(...)
    current_value: float = Field(default=0)
    status: str = Field(default="active")
    deadline: Optional[date] = None
    created_at: datetime = Field(...)
    completed_at: Optional[datetime] = None
    milestones: List[Dict[str, Any]] = Field(default_factory=list)


class AchievementBackup(BaseModel):
    """Backup format for achievements."""
    id: str = Field(..., description="Achievement ID")
    title: str = Field(...)
    unlocked: bool = Field(default=False)
    unlocked_at: Optional[datetime] = None
    progress: float = Field(default=0)
    points: int = Field(default=0)


class SettingsBackup(BaseModel):
    """Backup format for user settings."""
    theme: str = Field(default="system", description="UI theme preference")
    notifications_enabled: bool = Field(default=True)
    auto_backup_enabled: bool = Field(default=True)
    backup_frequency_hours: int = Field(default=24)
    language: str = Field(default="en")
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class BackupData(BaseModel):
    """
    Complete backup payload containing all user data.

    PATTERN: Comprehensive data export with version control
    WHY: Enable full data portability and backup/restore
    """
    metadata: SyncMetadata = Field(..., description="Sync metadata")
    conversations: List[ConversationBackup] = Field(
        default_factory=list,
        description="Conversation history"
    )
    feedback: List[FeedbackBackup] = Field(
        default_factory=list,
        description="Feedback data"
    )
    goals: List[GoalBackup] = Field(
        default_factory=list,
        description="Learning goals"
    )
    achievements: List[AchievementBackup] = Field(
        default_factory=list,
        description="Unlocked achievements"
    )
    settings: SettingsBackup = Field(
        default_factory=SettingsBackup,
        description="User settings and preferences"
    )
    progress_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Progress tracking data (streaks, daily progress, etc.)"
    )
    topic_mastery: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Topic mastery scores"
    )
    export_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this backup was created"
    )

    @property
    def total_items(self) -> int:
        """Calculate total number of backed up items."""
        return (
            len(self.conversations) +
            len(self.feedback) +
            len(self.goals) +
            len(self.achievements)
        )

    def get_item_counts(self) -> Dict[str, int]:
        """Get counts of each item type."""
        return {
            "conversations": len(self.conversations),
            "feedback": len(self.feedback),
            "goals": len(self.goals),
            "achievements": len(self.achievements),
        }

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "device_id": "device_xyz789",
                    "version": "1.0",
                    "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                },
                "conversations": [],
                "feedback": [],
                "goals": [],
                "achievements": [],
                "settings": {
                    "theme": "dark",
                    "notifications_enabled": True
                },
                "export_timestamp": "2025-11-21T10:00:00Z"
            }
        }


# ============================================================================
# EXPORT/IMPORT REQUEST MODELS
# ============================================================================

class DateRange(BaseModel):
    """Date range for filtering exports."""
    start_date: Optional[date] = Field(None, description="Start date (inclusive)")
    end_date: Optional[date] = Field(None, description="End date (inclusive)")

    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v: Optional[date], info) -> Optional[date]:
        """Validate end_date is after start_date."""
        start = info.data.get('start_date')
        if v is not None and start is not None and v < start:
            raise ValueError('end_date must be after start_date')
        return v


class ExportRequest(BaseModel):
    """
    Request model for data export configuration.

    PATTERN: Selective export with filtering options
    WHY: Allow users to export specific data subsets
    """
    include_conversations: bool = Field(
        default=True,
        description="Include conversation history"
    )
    include_feedback: bool = Field(
        default=True,
        description="Include feedback data"
    )
    include_goals: bool = Field(
        default=True,
        description="Include goals and milestones"
    )
    include_achievements: bool = Field(
        default=True,
        description="Include achievements"
    )
    include_settings: bool = Field(
        default=True,
        description="Include user settings"
    )
    include_progress: bool = Field(
        default=True,
        description="Include progress tracking data"
    )
    date_range: Optional[DateRange] = Field(
        default=None,
        description="Optional date range filter"
    )
    format: str = Field(
        default="json",
        pattern="^(json|encrypted_json)$",
        description="Export format"
    )
    compress: bool = Field(
        default=True,
        description="Compress the export file"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "include_conversations": True,
                "include_feedback": True,
                "include_goals": True,
                "include_achievements": True,
                "include_settings": True,
                "date_range": {
                    "start_date": "2025-01-01",
                    "end_date": "2025-11-21"
                },
                "format": "json",
                "compress": True
            }
        }


class ImportRequest(BaseModel):
    """
    Request model for data import configuration.

    PATTERN: Flexible import with conflict resolution
    WHY: Handle various import scenarios safely
    """
    data: BackupData = Field(..., description="Backup data to import")
    merge_strategy: MergeStrategy = Field(
        default=MergeStrategy.KEEP_NEWER,
        description="How to handle existing data"
    )
    validate_checksum: bool = Field(
        default=True,
        description="Verify data integrity before import"
    )
    dry_run: bool = Field(
        default=False,
        description="Preview changes without applying them"
    )
    skip_settings: bool = Field(
        default=False,
        description="Skip importing settings"
    )
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.REMOTE,
        description="Default conflict resolution strategy"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "metadata": {
                        "device_id": "device_xyz789",
                        "version": "1.0",
                        "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    },
                    "conversations": [],
                    "feedback": [],
                    "goals": [],
                    "achievements": [],
                    "settings": {}
                },
                "merge_strategy": "keep_newer",
                "validate_checksum": True,
                "dry_run": False
            }
        }


# ============================================================================
# CONFLICT MODELS
# ============================================================================

class SyncConflict(BaseModel):
    """
    Represents a sync conflict between local and remote data.

    PATTERN: Conflict detection and resolution tracking
    WHY: Enable users to review and resolve data conflicts
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    field: str = Field(..., description="Field/path where conflict occurred")
    item_type: str = Field(..., description="Type of item (conversation, goal, etc.)")
    item_id: str = Field(..., description="ID of the conflicting item")
    local_value: Any = Field(..., description="Current local value")
    remote_value: Any = Field(..., description="Incoming remote value")
    local_timestamp: Optional[datetime] = Field(
        None,
        description="When local value was last modified"
    )
    remote_timestamp: Optional[datetime] = Field(
        None,
        description="When remote value was last modified"
    )
    resolved_value: Optional[Any] = Field(
        None,
        description="Final resolved value (if resolved)"
    )
    resolution: Optional[ConflictResolution] = Field(
        None,
        description="How the conflict was resolved"
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="When the conflict was resolved"
    )
    auto_resolved: bool = Field(
        default=False,
        description="Whether conflict was auto-resolved"
    )

    @property
    def is_resolved(self) -> bool:
        """Check if conflict has been resolved."""
        return self.resolved_value is not None

    def resolve(
        self,
        value: Any,
        resolution: ConflictResolution,
        auto: bool = False
    ) -> None:
        """Resolve the conflict with the given value."""
        self.resolved_value = value
        self.resolution = resolution
        self.resolved_at = datetime.utcnow()
        self.auto_resolved = auto

    class Config:
        json_schema_extra = {
            "example": {
                "id": "conflict_abc123",
                "field": "title",
                "item_type": "goal",
                "item_id": "goal_xyz789",
                "local_value": "Learn Python",
                "remote_value": "Master Python Programming",
                "local_timestamp": "2025-11-20T10:00:00Z",
                "remote_timestamp": "2025-11-21T08:00:00Z",
                "resolved_value": "Master Python Programming",
                "resolution": "remote",
                "auto_resolved": True
            }
        }


# ============================================================================
# DEVICE MODELS
# ============================================================================

class DeviceInfo(BaseModel):
    """
    Information about a registered sync device.

    PATTERN: Multi-device sync management
    WHY: Track and manage devices for sync operations
    """
    device_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique device identifier"
    )
    device_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable device name"
    )
    platform: str = Field(
        ...,
        description="Device platform (ios, android, web, desktop)"
    )
    platform_version: Optional[str] = Field(
        None,
        description="OS/platform version"
    )
    app_version: Optional[str] = Field(
        None,
        description="App version on this device"
    )
    last_seen: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last time device synced"
    )
    last_sync_status: SyncStatus = Field(
        default=SyncStatus.PENDING,
        description="Status of last sync attempt"
    )
    registered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When device was registered"
    )
    is_primary: bool = Field(
        default=False,
        description="Whether this is the primary device"
    )
    push_enabled: bool = Field(
        default=False,
        description="Whether push notifications are enabled"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional device metadata"
    )

    def update_last_seen(self, status: SyncStatus = SyncStatus.COMPLETED) -> None:
        """Update last seen timestamp and status."""
        self.last_seen = datetime.utcnow()
        self.last_sync_status = status

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "device_xyz789",
                "device_name": "iPhone 15 Pro",
                "platform": "ios",
                "platform_version": "17.1",
                "app_version": "2.0.0",
                "last_seen": "2025-11-21T10:00:00Z",
                "last_sync_status": "completed",
                "registered_at": "2025-10-01T12:00:00Z",
                "is_primary": True,
                "push_enabled": True
            }
        }


# ============================================================================
# API RESPONSE MODELS
# ============================================================================

class ExportResponse(BaseModel):
    """Response model for export operation."""
    success: bool = Field(...)
    backup_data: Optional[BackupData] = Field(None)
    file_size_bytes: int = Field(default=0)
    checksum: str = Field(...)
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str = Field(default="Export completed successfully")


class ImportResponse(BaseModel):
    """Response model for import operation."""
    success: bool = Field(...)
    items_imported: Dict[str, int] = Field(default_factory=dict)
    items_skipped: Dict[str, int] = Field(default_factory=dict)
    conflicts: List[SyncConflict] = Field(default_factory=list)
    conflicts_resolved: int = Field(default=0)
    import_timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str = Field(default="Import completed successfully")


class SyncStatusResponse(BaseModel):
    """Response model for sync status check."""
    device_id: str = Field(...)
    sync_status: SyncStatus = Field(...)
    last_sync: Optional[datetime] = Field(None)
    pending_changes: int = Field(default=0)
    conflicts_count: int = Field(default=0)
    devices: List[DeviceInfo] = Field(default_factory=list)


class DeviceListResponse(BaseModel):
    """Response model for listing devices."""
    devices: List[DeviceInfo] = Field(...)
    total_count: int = Field(...)
    primary_device_id: Optional[str] = Field(None)
    current_device_id: Optional[str] = Field(None)


# ============================================================================
# API REQUEST/RESPONSE MODELS FOR ROUTES
# ============================================================================

class DeviceRegistrationRequest(BaseModel):
    """Request to register a new device."""
    device_name: str = Field(..., min_length=1, max_length=100)
    device_type: str = Field("unknown", pattern="^(desktop|mobile|tablet|unknown)$")
    platform: Optional[str] = Field(None)
    app_version: Optional[str] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "device_name": "My iPhone",
                "device_type": "mobile",
                "platform": "ios",
                "app_version": "1.0.0"
            }
        }


class ValidationResponse(BaseModel):
    """
    Response for backup file validation.

    PATTERN: Validate before import
    WHY: Prevent data corruption from invalid backups
    """
    valid: bool = Field(True)
    version: Optional[str] = Field(None, description="Backup format version")
    created_at: Optional[datetime] = Field(None, description="When backup was created")
    source_device: Optional[str] = Field(None, description="Device that created backup")
    record_count: int = Field(0)
    file_size_bytes: int = Field(0)
    checksum_valid: bool = Field(True, description="Checksum verification passed")
    schema_compatible: bool = Field(True, description="Schema is compatible")
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    sections: Dict[str, int] = Field(
        default_factory=dict,
        description="Record counts by section"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ConflictListResponse(BaseModel):
    """Response containing list of pending conflicts."""
    conflicts: List[SyncConflict] = Field(default_factory=list)
    total_count: int = Field(0)
    by_type: Dict[str, int] = Field(default_factory=dict, description="Count by conflict type")
    by_entity: Dict[str, int] = Field(default_factory=dict, description="Count by entity type")


class ConflictResolutionRequest(BaseModel):
    """Request to resolve one or more conflicts."""
    conflict_id: Optional[str] = Field(None, description="Specific conflict to resolve")
    conflict_ids: Optional[List[str]] = Field(None, description="Multiple conflicts to resolve")
    strategy: ConflictResolution = Field(..., description="Resolution strategy to apply")
    custom_value: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom merged value for manual resolution"
    )

    class Config:
        use_enum_values = True


class ConflictResolutionResponse(BaseModel):
    """Response for conflict resolution."""
    success: bool = Field(True)
    resolved_count: int = Field(0)
    failed_count: int = Field(0)
    remaining_conflicts: int = Field(0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class ScheduleBackupRequest(BaseModel):
    """Request to schedule auto-backup."""
    interval_hours: int = Field(
        12,
        ge=1,
        le=168,
        description="Backup interval in hours (1-168)"
    )
    enabled: bool = Field(True)

    class Config:
        json_schema_extra = {
            "example": {
                "interval_hours": 12,
                "enabled": True
            }
        }


class ScheduleBackupResponse(BaseModel):
    """Response for backup scheduling."""
    success: bool = Field(True)
    enabled: bool = Field(True)
    interval_hours: int = Field(12)
    next_backup: Optional[datetime] = Field(None)
    message: str = Field("")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ExtendedSyncStatusResponse(BaseModel):
    """
    Extended sync status response with backup scheduling info.

    PATTERN: Status endpoint for sync health monitoring
    WHY: Allow clients to check sync state before operations
    """
    status: SyncStatus = Field(SyncStatus.PENDING, description="Current sync status")
    last_sync: Optional[datetime] = Field(None, description="Last successful sync timestamp")
    next_backup: Optional[datetime] = Field(None, description="Next scheduled backup time")
    device_count: int = Field(0, description="Number of registered devices")
    data_size_bytes: int = Field(0, description="Total data size in bytes")
    data_size_human: str = Field("0 B", description="Human-readable data size")
    pending_changes: int = Field(0, description="Number of unsynced changes")
    conflicts_count: int = Field(0, description="Number of pending conflicts")
    backup_enabled: bool = Field(False, description="Whether auto-backup is enabled")
    backup_interval_hours: Optional[int] = Field(None, description="Backup interval in hours")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
