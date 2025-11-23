"""
Sync Module - Data Synchronization and Backup Functionality
============================================================

PATTERN: Modular sync/backup system with cross-device support
WHY: Enable data portability and backup/restore capabilities

Components:
- models: Pydantic models for sync/backup data structures
- config: Sync configuration and constants
- store: SQLite persistence for sync metadata
- import_service: Data import/restore with transaction support
- export_service: Data export and backup creation
- conflict_resolver: Merge conflict detection and resolution
- validators: Backup data validation
- serializers: Data serialization/deserialization helpers
"""
from app.sync.models import (
    SyncMetadata,
    BackupData,
    ExportRequest,
    ImportRequest,
    SyncConflict,
    DeviceInfo,
    MergeStrategy,
    SyncStatus,
)
from app.sync.config import (
    BACKUP_VERSION,
    MAX_BACKUP_SIZE_MB,
    AUTO_BACKUP_INTERVAL_HOURS,
    BACKUP_ENCRYPTION_ENABLED,
    sync_config,
)
from app.sync.store import SyncStore, sync_store
from app.sync.import_service import (
    ImportService,
    ImportResult,
    ImportStrategy,
    ImportStatus,
    import_service,
)
from app.sync.validators import (
    BackupValidator,
    ValidationResult,
    ValidationError,
    ValidationErrorCode,
    backup_validator,
)
from app.sync.conflict_resolver import (
    ConflictResolver,
    SyncConflict as ResolverSyncConflict,
    ConflictType,
    ResolutionStrategy,
    ResolutionResult,
    conflict_resolver,
)
from app.sync.scheduler import (
    BackupScheduler,
    backup_scheduler,
    schedule_auto_backup,
    run_scheduled_backup,
    get_next_backup_time,
    cancel_scheduled_backup,
)
from app.sync.routes import (
    router as sync_router,
    setup_sync_routes,
)
from app.sync.service import (
    SyncService,
    sync_service,
)
from app.sync.models import (
    # Additional response models
    ExportResponse,
    ImportResponse,
    SyncStatusResponse,
    DeviceListResponse,
    ValidationResponse,
    ConflictListResponse,
    ConflictResolutionResponse,
    ScheduleBackupResponse,
    ExtendedSyncStatusResponse,
    DeviceRegistrationRequest,
    ConflictResolutionRequest,
    ScheduleBackupRequest,
    ConflictResolution,
)
from app.sync.export_service import (
    DataExportService,
    BackupData as ExportBackupData,
    BackupMetadata,
    DateRange,
    ExportScope,
    EXPORT_VERSION,
    EXPORT_FORMAT,
    data_export_service,
)
from app.sync.serializers import (
    serialize_conversation,
    serialize_feedback,
    serialize_goal,
    serialize_achievement,
    deserialize_conversation,
    deserialize_feedback,
    deserialize_goal,
    deserialize_achievement,
    serialize_conversations_batch,
    serialize_feedback_batch,
    serialize_goals_batch,
    serialize_achievements_batch,
    ConversationData,
    FeedbackData,
    SettingsData,
)

__all__ = [
    # Models
    "SyncMetadata",
    "BackupData",
    "ExportRequest",
    "ImportRequest",
    "SyncConflict",
    "DeviceInfo",
    "MergeStrategy",
    "SyncStatus",
    # Config
    "BACKUP_VERSION",
    "MAX_BACKUP_SIZE_MB",
    "AUTO_BACKUP_INTERVAL_HOURS",
    "BACKUP_ENCRYPTION_ENABLED",
    "sync_config",
    # Store
    "SyncStore",
    "sync_store",
    # Import Service
    "ImportService",
    "ImportResult",
    "ImportStrategy",
    "ImportStatus",
    "import_service",
    # Validators
    "BackupValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationErrorCode",
    "backup_validator",
    # Conflict Resolver
    "ConflictResolver",
    "ResolverSyncConflict",
    "ConflictType",
    "ResolutionStrategy",
    "ResolutionResult",
    "conflict_resolver",
    # Scheduler
    "BackupScheduler",
    "backup_scheduler",
    "schedule_auto_backup",
    "run_scheduled_backup",
    "get_next_backup_time",
    "cancel_scheduled_backup",
    # Routes
    "sync_router",
    "setup_sync_routes",
    # Service
    "SyncService",
    "sync_service",
    # Response Models
    "ExportResponse",
    "ImportResponse",
    "SyncStatusResponse",
    "DeviceListResponse",
    "ValidationResponse",
    "ConflictListResponse",
    "ConflictResolutionResponse",
    "ScheduleBackupResponse",
    "ExtendedSyncStatusResponse",
    # Request Models
    "DeviceRegistrationRequest",
    "ConflictResolutionRequest",
    "ScheduleBackupRequest",
    "ConflictResolution",
    # Export Service
    "DataExportService",
    "ExportBackupData",
    "BackupMetadata",
    "DateRange",
    "ExportScope",
    "EXPORT_VERSION",
    "EXPORT_FORMAT",
    "data_export_service",
    # Serializers
    "serialize_conversation",
    "serialize_feedback",
    "serialize_goal",
    "serialize_achievement",
    "deserialize_conversation",
    "deserialize_feedback",
    "deserialize_goal",
    "deserialize_achievement",
    "serialize_conversations_batch",
    "serialize_feedback_batch",
    "serialize_goals_batch",
    "serialize_achievements_batch",
    "ConversationData",
    "FeedbackData",
    "SettingsData",
]
