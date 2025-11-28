"""
Tests for sync data models and schemas.
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.sync.models import (
    SyncMetadata,
    BackupData,
    ExportRequest,
    ImportRequest,
    SyncConflict,
    DeviceInfo,
    MergeStrategy,
    SyncStatus,
    DateRange,
)


class TestSyncMetadata:
    """Tests for SyncMetadata model."""

    def test_create_metadata(self):
        """Test creating sync metadata."""
        valid_checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        metadata = SyncMetadata(
            last_sync=datetime.utcnow(),
            device_id=str(uuid4()),
            version="1.0",
            checksum=valid_checksum,
        )
        assert metadata.version == "1.0"
        assert metadata.checksum == valid_checksum

    def test_metadata_defaults(self):
        """Test metadata default values."""
        valid_checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        metadata = SyncMetadata(
            device_id=str(uuid4()),
            version="1.0",
            checksum=valid_checksum,
        )
        assert metadata.last_sync is None or isinstance(metadata.last_sync, datetime)


class TestBackupData:
    """Tests for BackupData model."""

    def test_create_empty_backup(self):
        """Test creating empty backup data."""
        valid_checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        backup = BackupData(
            metadata=SyncMetadata(
                device_id=str(uuid4()),
                version="1.0",
                checksum=valid_checksum,
            ),
            conversations=[],
            feedback=[],
            goals=[],
            achievements=[],
            settings={},
        )
        assert len(backup.conversations) == 0
        assert len(backup.goals) == 0

    def test_backup_with_data(self):
        """Test backup with actual data."""
        valid_checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        from app.sync.models import ConversationBackup, FeedbackBackup, GoalBackup, AchievementBackup, SettingsBackup
        from datetime import date

        backup = BackupData(
            metadata=SyncMetadata(
                device_id=str(uuid4()),
                version="1.0",
                checksum=valid_checksum,
            ),
            conversations=[
                ConversationBackup(
                    id="1",
                    user_text="Hello",
                    agent_text="Hi there",
                    timestamp=datetime.utcnow()
                )
            ],
            feedback=[
                FeedbackBackup(
                    id="1",
                    session_id="session_1",
                    feedback_type="explicit",
                    rating=5,
                    timestamp=datetime.utcnow()
                )
            ],
            goals=[
                GoalBackup(
                    id="1",
                    title="Learn Python",
                    goal_type="learning",
                    target_value=100.0,
                    created_at=datetime.utcnow()
                )
            ],
            achievements=[
                AchievementBackup(
                    id="1",
                    title="First Steps"
                )
            ],
            settings=SettingsBackup(theme="dark"),
        )
        assert len(backup.conversations) == 1
        assert backup.settings.theme == "dark"


class TestExportRequest:
    """Tests for ExportRequest model."""

    def test_export_all(self):
        """Test export request for all data."""
        request = ExportRequest(
            include_conversations=True,
            include_feedback=True,
            include_goals=True,
        )
        assert request.include_conversations is True

    def test_export_with_date_range(self):
        """Test export with date range."""
        request = ExportRequest(
            include_conversations=True,
            date_range=DateRange(
                start=datetime.utcnow() - timedelta(days=30),
                end=datetime.utcnow(),
            ),
        )
        assert request.date_range is not None


class TestImportRequest:
    """Tests for ImportRequest model."""

    def test_import_replace_strategy(self):
        """Test import with replace strategy."""
        valid_checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        backup_data = BackupData(
            metadata=SyncMetadata(
                device_id=str(uuid4()),
                version="1.0",
                checksum=valid_checksum,
            )
        )
        request = ImportRequest(
            data=backup_data,
            merge_strategy=MergeStrategy.REPLACE,
        )
        assert request.merge_strategy == MergeStrategy.REPLACE

    def test_import_merge_strategy(self):
        """Test import with merge strategy."""
        valid_checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        backup_data = BackupData(
            metadata=SyncMetadata(
                device_id=str(uuid4()),
                version="1.0",
                checksum=valid_checksum,
            )
        )
        request = ImportRequest(
            data=backup_data,
            merge_strategy=MergeStrategy.MERGE,
        )
        assert request.merge_strategy == MergeStrategy.MERGE


class TestSyncConflict:
    """Tests for SyncConflict model."""

    def test_create_conflict(self):
        """Test creating a sync conflict."""
        conflict = SyncConflict(
            field="title",
            item_type="goal",
            item_id="goal_123",
            local_value="Local Title",
            remote_value="Remote Title",
        )
        assert conflict.field == "title"
        assert conflict.item_type == "goal"
        assert conflict.item_id == "goal_123"
        assert conflict.resolved_value is None

    def test_resolve_conflict(self):
        """Test resolving a conflict."""
        conflict = SyncConflict(
            field="title",
            item_type="goal",
            item_id="goal_456",
            local_value="Local",
            remote_value="Remote",
            resolved_value="Merged",
        )
        assert conflict.resolved_value == "Merged"


class TestDeviceInfo:
    """Tests for DeviceInfo model."""

    def test_create_device(self):
        """Test creating device info."""
        device = DeviceInfo(
            device_id=str(uuid4()),
            device_name="My Phone",
            platform="android",
            last_seen=datetime.utcnow(),
        )
        assert device.device_name == "My Phone"
        assert device.platform == "android"


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_strategy_values(self):
        """Test merge strategy values exist."""
        assert MergeStrategy.REPLACE.value == "replace"
        assert MergeStrategy.MERGE.value == "merge"
        assert MergeStrategy.KEEP_NEWER.value == "keep_newer"


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_status_values(self):
        """Test sync status values exist."""
        assert SyncStatus.PENDING is not None
        assert SyncStatus.IN_PROGRESS is not None
        assert SyncStatus.COMPLETED is not None
        assert SyncStatus.FAILED is not None
        assert SyncStatus.CONFLICT is not None
