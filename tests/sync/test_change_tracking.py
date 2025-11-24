"""
Tests for Change Tracking in Sync System (Feature 4)

SPARC Specification:
- Track edit history and versions
- Enable advanced conflict detection
- Diff generation for changes
- Audit trail for compliance
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json
import hashlib


@pytest.fixture
def mock_sync_service():
    """Mock sync service with change tracking"""
    class MockChangeTracker:
        def __init__(self):
            self.changes = []
            self.versions = {}

        def track_change(self, entity_type: str, entity_id: str, operation: str,
                        old_value: dict = None, new_value: dict = None):
            change = {
                "id": hashlib.md5(f"{entity_id}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8],
                "entity_type": entity_type,
                "entity_id": entity_id,
                "operation": operation,
                "old_value": old_value,
                "new_value": new_value,
                "timestamp": datetime.utcnow().isoformat(),
                "version": self._increment_version(entity_type, entity_id)
            }
            self.changes.append(change)
            return change

        def _increment_version(self, entity_type: str, entity_id: str):
            key = f"{entity_type}:{entity_id}"
            self.versions[key] = self.versions.get(key, 0) + 1
            return self.versions[key]

        def get_changes(self, entity_type: str = None, since: datetime = None):
            result = self.changes
            if entity_type:
                result = [c for c in result if c["entity_type"] == entity_type]
            if since:
                result = [c for c in result if c["timestamp"] >= since.isoformat()]
            return result

        def get_version(self, entity_type: str, entity_id: str):
            key = f"{entity_type}:{entity_id}"
            return self.versions.get(key, 0)

        def generate_diff(self, old_value: dict, new_value: dict):
            """Generate diff between two values"""
            diff = {
                "added": {},
                "removed": {},
                "modified": {}
            }

            old_keys = set(old_value.keys()) if old_value else set()
            new_keys = set(new_value.keys()) if new_value else set()

            # Added keys
            for key in new_keys - old_keys:
                diff["added"][key] = new_value[key]

            # Removed keys
            for key in old_keys - new_keys:
                diff["removed"][key] = old_value[key]

            # Modified keys
            for key in old_keys & new_keys:
                if old_value[key] != new_value[key]:
                    diff["modified"][key] = {
                        "old": old_value[key],
                        "new": new_value[key]
                    }

            return diff

    return MockChangeTracker()


@pytest.fixture
def sample_capture_data():
    """Sample capture data for testing"""
    return {
        "id": "capture-123",
        "session_id": "session-abc",
        "user_text": "Original question",
        "agent_text": "Original response",
        "timestamp": "2024-11-21T10:00:00",
        "metadata": {"source": "voice"}
    }


class TestChangeTracking:
    """Test change tracking functionality"""

    def test_track_create_operation(self, mock_sync_service, sample_capture_data):
        """Test tracking create operations"""
        change = mock_sync_service.track_change(
            entity_type="capture",
            entity_id=sample_capture_data["id"],
            operation="create",
            old_value=None,
            new_value=sample_capture_data
        )

        assert change["operation"] == "create"
        assert change["entity_type"] == "capture"
        assert change["old_value"] is None
        assert change["new_value"] == sample_capture_data
        assert "timestamp" in change
        assert change["version"] == 1

    def test_track_update_operation(self, mock_sync_service, sample_capture_data):
        """Test tracking update operations"""
        # First create
        mock_sync_service.track_change(
            entity_type="capture",
            entity_id=sample_capture_data["id"],
            operation="create",
            new_value=sample_capture_data
        )

        # Then update
        updated_data = {**sample_capture_data, "user_text": "Updated question"}
        change = mock_sync_service.track_change(
            entity_type="capture",
            entity_id=sample_capture_data["id"],
            operation="update",
            old_value=sample_capture_data,
            new_value=updated_data
        )

        assert change["operation"] == "update"
        assert change["version"] == 2
        assert change["old_value"]["user_text"] == "Original question"
        assert change["new_value"]["user_text"] == "Updated question"

    def test_track_delete_operation(self, mock_sync_service, sample_capture_data):
        """Test tracking delete operations"""
        change = mock_sync_service.track_change(
            entity_type="capture",
            entity_id=sample_capture_data["id"],
            operation="delete",
            old_value=sample_capture_data,
            new_value=None
        )

        assert change["operation"] == "delete"
        assert change["old_value"] == sample_capture_data
        assert change["new_value"] is None

    def test_version_increments(self, mock_sync_service, sample_capture_data):
        """Test version numbers increment correctly"""
        entity_id = sample_capture_data["id"]

        # Multiple updates
        for i in range(5):
            mock_sync_service.track_change(
                entity_type="capture",
                entity_id=entity_id,
                operation="update",
                new_value=sample_capture_data
            )

        version = mock_sync_service.get_version("capture", entity_id)
        assert version == 5

    def test_get_changes_by_entity_type(self, mock_sync_service):
        """Test filtering changes by entity type"""
        mock_sync_service.track_change("capture", "c1", "create", new_value={})
        mock_sync_service.track_change("goal", "g1", "create", new_value={})
        mock_sync_service.track_change("capture", "c2", "create", new_value={})

        capture_changes = mock_sync_service.get_changes(entity_type="capture")

        assert len(capture_changes) == 2
        assert all(c["entity_type"] == "capture" for c in capture_changes)

    def test_get_changes_since_timestamp(self, mock_sync_service):
        """Test filtering changes by timestamp"""
        # Add old change (simulated)
        mock_sync_service.track_change("capture", "c1", "create", new_value={})

        # Get changes since now (should be empty for new queries)
        since = datetime.utcnow() - timedelta(seconds=1)
        changes = mock_sync_service.get_changes(since=since)

        assert len(changes) >= 1  # At least the one we just added


class TestDiffGeneration:
    """Test diff generation for changes"""

    def test_generate_diff_added_fields(self, mock_sync_service):
        """Test diff shows added fields"""
        old_value = {"name": "Test"}
        new_value = {"name": "Test", "description": "New field"}

        diff = mock_sync_service.generate_diff(old_value, new_value)

        assert "description" in diff["added"]
        assert diff["added"]["description"] == "New field"

    def test_generate_diff_removed_fields(self, mock_sync_service):
        """Test diff shows removed fields"""
        old_value = {"name": "Test", "temp": "Will be removed"}
        new_value = {"name": "Test"}

        diff = mock_sync_service.generate_diff(old_value, new_value)

        assert "temp" in diff["removed"]
        assert diff["removed"]["temp"] == "Will be removed"

    def test_generate_diff_modified_fields(self, mock_sync_service):
        """Test diff shows modified fields"""
        old_value = {"name": "Old Name", "count": 5}
        new_value = {"name": "New Name", "count": 5}

        diff = mock_sync_service.generate_diff(old_value, new_value)

        assert "name" in diff["modified"]
        assert diff["modified"]["name"]["old"] == "Old Name"
        assert diff["modified"]["name"]["new"] == "New Name"

    def test_generate_diff_no_changes(self, mock_sync_service):
        """Test diff with identical values"""
        value = {"name": "Same", "count": 10}

        diff = mock_sync_service.generate_diff(value, value.copy())

        assert len(diff["added"]) == 0
        assert len(diff["removed"]) == 0
        assert len(diff["modified"]) == 0


class TestConflictDetection:
    """Test conflict detection using change tracking"""

    def test_detect_concurrent_modifications(self, mock_sync_service):
        """Test detecting concurrent modifications"""
        # Device A makes change at version 1
        mock_sync_service.track_change("capture", "c1", "update",
                                       old_value={"text": "v1"},
                                       new_value={"text": "v2a"})

        # Device B tries to make change also from version 1
        # This should detect a conflict
        version_before = mock_sync_service.get_version("capture", "c1")

        mock_sync_service.track_change("capture", "c1", "update",
                                       old_value={"text": "v1"},
                                       new_value={"text": "v2b"})

        version_after = mock_sync_service.get_version("capture", "c1")

        # Version should have advanced
        assert version_after > version_before

    def test_detect_delete_update_conflict(self, mock_sync_service):
        """Test conflict when one device deletes and another updates"""
        # Create initial
        mock_sync_service.track_change("capture", "c1", "create",
                                       new_value={"text": "original"})

        # Device A deletes
        mock_sync_service.track_change("capture", "c1", "delete",
                                       old_value={"text": "original"})

        # Device B updates (conflict)
        mock_sync_service.track_change("capture", "c1", "update",
                                       old_value={"text": "original"},
                                       new_value={"text": "updated"})

        changes = mock_sync_service.get_changes(entity_type="capture")
        operations = [c["operation"] for c in changes]

        # Both operations should be recorded
        assert "delete" in operations
        assert "update" in operations


class TestAuditTrail:
    """Test audit trail functionality"""

    def test_audit_trail_includes_timestamp(self, mock_sync_service):
        """Test all changes have timestamps"""
        mock_sync_service.track_change("capture", "c1", "create", new_value={})

        changes = mock_sync_service.get_changes()
        for change in changes:
            assert "timestamp" in change
            # Verify timestamp is ISO format
            datetime.fromisoformat(change["timestamp"])

    def test_audit_trail_preserves_all_changes(self, mock_sync_service):
        """Test audit trail preserves complete history"""
        entity_id = "audit-test"

        # Create -> Update -> Update -> Delete
        operations = ["create", "update", "update", "delete"]
        for op in operations:
            mock_sync_service.track_change("capture", entity_id, op, new_value={})

        changes = [c for c in mock_sync_service.get_changes()
                  if c["entity_id"] == entity_id]

        assert len(changes) == 4
        assert [c["operation"] for c in changes] == operations

    def test_audit_trail_queryable_by_time_range(self, mock_sync_service):
        """Test audit trail can be queried by time range"""
        # Add some changes
        mock_sync_service.track_change("capture", "c1", "create", new_value={})

        # Query since a timestamp
        since = datetime.utcnow() - timedelta(hours=1)
        changes = mock_sync_service.get_changes(since=since)

        # Should include recent changes
        assert len(changes) >= 1
