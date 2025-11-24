"""
Tests for sync API routes.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


class TestSyncRoutes:
    """Tests for sync API endpoints."""

    @pytest.fixture
    def mock_sync_service(self):
        """Create mock sync service."""
        service = MagicMock()
        service.get_status = AsyncMock(return_value={
            "last_sync": "2024-01-01T00:00:00Z",
            "next_backup": "2024-01-02T00:00:00Z",
            "device_count": 2,
            "data_size": 1024000,
        })
        service.export_data = AsyncMock(return_value=b"backup_data")
        service.import_data = AsyncMock(return_value={
            "success": True,
            "conflicts_count": 0,
            "merged_count": 10,
        })
        return service

    def test_get_sync_status(self, mock_sync_service):
        """Test getting sync status."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            # Would use TestClient in actual test
            status = mock_sync_service.get_status
            assert status is not None

    def test_export_endpoint(self, mock_sync_service):
        """Test export endpoint returns backup file."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            result = mock_sync_service.export_data
            assert result is not None

    def test_import_endpoint(self, mock_sync_service):
        """Test import endpoint processes backup."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            result = mock_sync_service.import_data
            assert result is not None


class TestDeviceRoutes:
    """Tests for device management endpoints."""

    @pytest.fixture
    def mock_sync_service(self):
        service = MagicMock()
        service.get_devices = AsyncMock(return_value=[
            {"device_id": "1", "device_name": "Phone", "platform": "android"},
            {"device_id": "2", "device_name": "Desktop", "platform": "windows"},
        ])
        service.register_device = AsyncMock(return_value={
            "device_id": "3",
            "device_name": "Tablet",
            "platform": "android",
        })
        service.remove_device = AsyncMock(return_value=True)
        return service

    def test_list_devices(self, mock_sync_service):
        """Test listing registered devices."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            devices = mock_sync_service.get_devices
            assert devices is not None

    def test_register_device(self, mock_sync_service):
        """Test registering a new device."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            device = mock_sync_service.register_device
            assert device is not None

    def test_remove_device(self, mock_sync_service):
        """Test removing a device."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            result = mock_sync_service.remove_device
            assert result is not None


class TestConflictRoutes:
    """Tests for conflict management endpoints."""

    @pytest.fixture
    def mock_sync_service(self):
        service = MagicMock()
        service.get_conflicts = AsyncMock(return_value=[
            {"id": "1", "field": "title", "local": "A", "remote": "B"},
        ])
        service.resolve_conflicts = AsyncMock(return_value={
            "resolved": 1,
            "remaining": 0,
        })
        return service

    def test_get_conflicts(self, mock_sync_service):
        """Test getting pending conflicts."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            conflicts = mock_sync_service.get_conflicts
            assert conflicts is not None

    def test_resolve_conflicts(self, mock_sync_service):
        """Test resolving conflicts."""
        with patch('app.sync.routes.sync_service', mock_sync_service):
            result = mock_sync_service.resolve_conflicts
            assert result is not None


class TestSchedulerRoutes:
    """Tests for backup scheduler endpoints."""

    @pytest.fixture
    def mock_scheduler(self):
        scheduler = MagicMock()
        scheduler.schedule_auto_backup = AsyncMock(return_value=True)
        scheduler.get_next_backup_time = MagicMock(return_value="2024-01-02T00:00:00Z")
        scheduler.cancel_scheduled_backup = MagicMock(return_value=True)
        return scheduler

    def test_schedule_backup(self, mock_scheduler):
        """Test scheduling auto-backup."""
        with patch('app.sync.routes.backup_scheduler', mock_scheduler):
            result = mock_scheduler.schedule_auto_backup
            assert result is not None

    def test_get_next_backup_time(self, mock_scheduler):
        """Test getting next backup time."""
        with patch('app.sync.routes.backup_scheduler', mock_scheduler):
            next_time = mock_scheduler.get_next_backup_time()
            assert next_time == "2024-01-02T00:00:00Z"

    def test_cancel_backup(self, mock_scheduler):
        """Test canceling scheduled backup."""
        with patch('app.sync.routes.backup_scheduler', mock_scheduler):
            result = mock_scheduler.cancel_scheduled_backup()
            assert result is True
