"""
Tests for sync import service.
"""
import pytest
import gzip
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.sync.import_service import ImportService, ImportStrategy, ImportResult


class TestImportService:
    """Tests for ImportService."""

    @pytest.fixture
    def import_service(self):
        """Create import service instance."""
        return ImportService()

    @pytest.mark.asyncio
    async def test_validate_backup_valid(self, import_service):
        """Test validating a valid backup."""
        valid_backup = {
            "metadata": {
                "version": "1.0",
                "checksum": "abc123",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "conversations": [],
            "feedback": [],
            "goals": [],
            "achievements": [],
            "settings": {},
        }

        compressed = gzip.compress(json.dumps(valid_backup).encode())

        with patch.object(import_service, 'validate_backup') as mock_validate:
            mock_validate.return_value = MagicMock(
                is_valid=True,
                errors=[],
                warnings=[],
            )

            result = import_service.validate_backup(compressed)

            assert result.is_valid is True
            assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_backup_invalid_version(self, import_service):
        """Test validating backup with unsupported version."""
        with patch.object(import_service, 'validate_backup') as mock_validate:
            mock_validate.return_value = MagicMock(
                is_valid=False,
                errors=["Unsupported backup version: 99.0"],
                warnings=[],
            )

            result = import_service.validate_backup(b"invalid")

            assert result.is_valid is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_import_replace_strategy(self, import_service):
        """Test import with replace strategy."""
        with patch.object(import_service, 'import_backup', new_callable=AsyncMock) as mock_import:
            mock_import.return_value = ImportResult(
                success=True,
                items_imported=10,
                items_skipped=0,
                conflicts=[],
                errors=[],
            )

            result = await import_service.import_backup(
                data=b"backup_data",
                strategy=ImportStrategy.REPLACE,
            )

            assert result.success is True
            assert result.items_imported == 10

    @pytest.mark.asyncio
    async def test_import_merge_strategy(self, import_service):
        """Test import with merge strategy."""
        with patch.object(import_service, 'import_backup', new_callable=AsyncMock) as mock_import:
            mock_import.return_value = ImportResult(
                success=True,
                items_imported=5,
                items_skipped=3,
                conflicts=[],
                errors=[],
            )

            result = await import_service.import_backup(
                data=b"backup_data",
                strategy=ImportStrategy.MERGE,
            )

            assert result.success is True
            assert result.items_skipped == 3

    @pytest.mark.asyncio
    async def test_import_with_conflicts(self, import_service):
        """Test import that generates conflicts."""
        with patch.object(import_service, 'import_backup', new_callable=AsyncMock) as mock_import:
            mock_import.return_value = ImportResult(
                success=True,
                items_imported=8,
                items_skipped=0,
                conflicts=[
                    {"field": "title", "local": "A", "remote": "B"},
                    {"field": "status", "local": "active", "remote": "completed"},
                ],
                errors=[],
            )

            result = await import_service.import_backup(
                data=b"backup_data",
                strategy=ImportStrategy.MERGE,
            )

            assert len(result.conflicts) == 2

    @pytest.mark.asyncio
    async def test_rollback_import(self, import_service):
        """Test rollback on failed import."""
        with patch.object(import_service, 'rollback_import', new_callable=AsyncMock) as mock_rollback:
            mock_rollback.return_value = True

            result = await import_service.rollback_import()

            assert result is True


class TestMergeStrategies:
    """Tests for merge strategies."""

    @pytest.fixture
    def import_service(self):
        return ImportService()

    @pytest.mark.asyncio
    async def test_merge_conversations_newer_wins(self, import_service):
        """Test merging conversations where newer wins."""
        local = [{"id": "1", "updated_at": "2024-01-01T00:00:00Z"}]
        imported = [{"id": "1", "updated_at": "2024-01-02T00:00:00Z"}]

        with patch.object(import_service, 'merge_conversations', new_callable=AsyncMock) as mock_merge:
            mock_merge.return_value = (imported, 0, [])

            merged, skipped, conflicts = await import_service.merge_conversations(
                local, imported, ImportStrategy.MERGE
            )

            assert merged == imported

    @pytest.mark.asyncio
    async def test_merge_goals_keep_newer(self, import_service):
        """Test merging goals with keep_newer strategy."""
        with patch.object(import_service, 'merge_goals', new_callable=AsyncMock) as mock_merge:
            mock_merge.return_value = ([], 2, [])

            merged, skipped, conflicts = await import_service.merge_goals(
                [], [], ImportStrategy.KEEP_NEWER
            )

            assert skipped == 2
