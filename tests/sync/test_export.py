"""
Tests for sync export service.
"""
import pytest
import gzip
import json
import hashlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.sync.export_service import DataExportService


class TestDataExportService:
    """Tests for DataExportService."""

    @pytest.fixture
    def export_service(self):
        """Create export service instance."""
        return DataExportService()

    def test_calculate_checksum(self, export_service):
        """Test checksum calculation."""
        test_data = {"key": "value", "number": 123}
        checksum = export_service.calculate_checksum(test_data)

        # Verify it's a valid SHA-256 hash
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)

    def test_checksum_consistency(self, export_service):
        """Test checksum is consistent for same data."""
        data = {"test": "data"}
        checksum1 = export_service.calculate_checksum(data)
        checksum2 = export_service.calculate_checksum(data)
        assert checksum1 == checksum2

    def test_checksum_different_data(self, export_service):
        """Test different data produces different checksums."""
        data1 = {"test": "data1"}
        data2 = {"test": "data2"}
        checksum1 = export_service.calculate_checksum(data1)
        checksum2 = export_service.calculate_checksum(data2)
        assert checksum1 != checksum2

    @pytest.mark.asyncio
    async def test_export_statistics(self, export_service):
        """Test getting export statistics."""
        with patch.object(export_service, 'get_export_statistics', new_callable=AsyncMock) as mock_stats:
            mock_stats.return_value = {
                'conversations': 10,
                'feedback': 25,
                'goals': 5,
                'achievements': 3,
            }

            stats = await export_service.get_export_statistics()

            assert 'conversations' in stats
            assert 'feedback' in stats
            assert 'goals' in stats

    @pytest.mark.asyncio
    async def test_export_all_data(self, export_service):
        """Test exporting all data."""
        with patch.object(export_service, 'export_all_data', new_callable=AsyncMock) as mock_export:
            mock_export.return_value = MagicMock(
                metadata=MagicMock(version="1.0"),
                conversations=[],
                feedback=[],
                goals=[],
                achievements=[],
                settings={},
            )

            result = await export_service.export_all_data()

            assert result.metadata.version == "1.0"

    @pytest.mark.asyncio
    async def test_create_backup_file_compressed(self, export_service):
        """Test creating compressed backup file."""
        with patch.object(export_service, 'create_backup_file', new_callable=AsyncMock) as mock_backup:
            # Simulate compressed data
            test_data = json.dumps({"test": "data"}).encode()
            compressed = gzip.compress(test_data)
            mock_backup.return_value = compressed

            result = await export_service.create_backup_file(compress=True)

            # Verify it's gzip compressed
            assert result[:2] == b'\x1f\x8b'  # gzip magic number

    def test_verify_checksum_valid(self, export_service):
        """Test checksum verification with valid data."""
        data = {"test": "data"}
        checksum = export_service.calculate_checksum(data)

        assert export_service.verify_checksum(data, checksum) is True

    def test_verify_checksum_invalid(self, export_service):
        """Test checksum verification with invalid checksum."""
        data = {"test": "data"}
        wrong_checksum = "a" * 64

        assert export_service.verify_checksum(data, wrong_checksum) is False


class TestExportPartial:
    """Tests for partial export functionality."""

    @pytest.fixture
    def export_service(self):
        return DataExportService()

    @pytest.mark.asyncio
    async def test_export_conversations_only(self, export_service):
        """Test exporting only conversations."""
        with patch.object(export_service, 'export_conversations', new_callable=AsyncMock) as mock_export:
            mock_export.return_value = [
                {"id": "1", "text": "Hello"},
                {"id": "2", "text": "World"},
            ]

            result = await export_service.export_conversations()

            assert len(result) == 2
            assert result[0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_export_with_date_range(self, export_service):
        """Test export with date range filter."""
        date_range = {
            'start': datetime.utcnow() - timedelta(days=7),
            'end': datetime.utcnow(),
        }

        with patch.object(export_service, 'export_conversations', new_callable=AsyncMock) as mock_export:
            mock_export.return_value = []

            result = await export_service.export_conversations(date_range=date_range)

            mock_export.assert_called_once()
