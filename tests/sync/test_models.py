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
        metadata = SyncMetadata(
            last_sync=datetime.utcnow(),
            device_id=str(uuid4()),
            version="1.0",
            checksum="abc123",
        )
        assert metadata.version == "1.0"
        assert metadata.checksum == "abc123"

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = SyncMetadata(
            device_id=str(uuid4()),
            version="1.0",
        )
        assert metadata.last_sync is None or isinstance(metadata.last_sync, datetime)


class TestBackupData:
    """Tests for BackupData model."""

    def test_create_empty_backup(self):
        """Test creating empty backup data."""
        backup = BackupData(
            metadata=SyncMetadata(
                device_id=str(uuid4()),
                version="1.0",
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
        backup = BackupData(
            metadata=SyncMetadata(
                device_id=str(uuid4()),
                version="1.0",
            ),
            conversations=[{"id": "1", "text": "Hello"}],
            feedback=[{"id": "1", "rating": 5}],
            goals=[{"id": "1", "title": "Learn Python"}],
            achievements=[{"id": "1", "name": "First Steps"}],
            settings={"theme": "dark"},
        )
        assert len(backup.conversations) == 1
        assert backup.settings["theme"] == "dark"


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
        request = ImportRequest(
            data=b"backup_data",
            merge_strategy=MergeStrategy.REPLACE,
        )
        assert request.merge_strategy == MergeStrategy.REPLACE

    def test_import_merge_strategy(self):
        """Test import with merge strategy."""
        request = ImportRequest(
            data=b"backup_data",
            merge_strategy=MergeStrategy.MERGE,
        )
        assert request.merge_strategy == MergeStrategy.MERGE


class TestSyncConflict:
    """Tests for SyncConflict model."""

    def test_create_conflict(self):
        """Test creating a sync conflict."""
        conflict = SyncConflict(
            field="title",
            local_value="Local Title",
            remote_value="Remote Title",
        )
        assert conflict.field == "title"
        assert conflict.resolved_value is None

    def test_resolve_conflict(self):
        """Test resolving a conflict."""
        conflict = SyncConflict(
            field="title",
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
        assert SyncStatus.IDLE is not None
        assert SyncStatus.SYNCING is not None
        assert SyncStatus.ERROR is not None
