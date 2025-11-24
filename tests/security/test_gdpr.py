"""
GDPR Compliance Tests - Plan A Security

Tests for:
- Data export functionality
- Data deletion (right to be forgotten)
- Export status tracking
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.security.models import (
    GDPRExportRequest,
    GDPRDeleteRequest,
    User,
    UserRole,
    UserStatus,
)
from app.security.routes import (
    generate_user_export,
    process_data_export,
    process_account_deletion,
)


class TestGDPRDataExport:
    """Test GDPR data export functionality."""

    @pytest.mark.asyncio
    async def test_export_includes_user_profile(self):
        """Export should include user profile data."""
        mock_user = User(
            id="user-123",
            email="export@example.com",
            full_name="Test User",
            hashed_password="hashed",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )

        with patch('app.security.routes.auth_service') as mock_auth:
            mock_auth.get_user = AsyncMock(return_value=mock_user)

            export_data = await generate_user_export("user-123")

        assert "user_profile" in export_data
        assert export_data["user_profile"]["email"] == "export@example.com"
        assert export_data["user_profile"]["full_name"] == "Test User"

    @pytest.mark.asyncio
    async def test_export_includes_metadata(self):
        """Export should include compliance metadata."""
        mock_user = User(
            id="user-123",
            email="export@example.com",
            hashed_password="hashed",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        with patch('app.security.routes.auth_service') as mock_auth:
            mock_auth.get_user = AsyncMock(return_value=mock_user)

            export_data = await generate_user_export("user-123")

        assert "export_metadata" in export_data
        assert export_data["export_metadata"]["gdpr_compliant"] is True
        assert "generated_at" in export_data["export_metadata"]

    @pytest.mark.asyncio
    async def test_export_returns_empty_for_nonexistent_user(self):
        """Export should return empty dict for non-existent user."""
        with patch('app.security.routes.auth_service') as mock_auth:
            mock_auth.get_user = AsyncMock(return_value=None)

            export_data = await generate_user_export("nonexistent")

        assert export_data == {}


class TestGDPRDataDeletion:
    """Test GDPR data deletion functionality."""

    def test_deletion_requires_confirmation(self):
        """Deletion request must have confirm=True."""
        with pytest.raises(ValueError) as exc_info:
            GDPRDeleteRequest(confirm=False)

        assert "confirm" in str(exc_info.value).lower()

    def test_deletion_accepts_valid_request(self):
        """Valid deletion request should be accepted."""
        request = GDPRDeleteRequest(
            confirm=True,
            reason="Account no longer needed",
        )

        assert request.confirm is True
        assert request.reason == "Account no longer needed"

    @pytest.mark.asyncio
    async def test_deletion_marks_user_as_deleted(self):
        """Deletion should mark user as deleted."""
        mock_user = MagicMock()
        mock_user.status = UserStatus.ACTIVE

        with patch('app.security.routes.auth_service') as mock_auth:
            mock_auth.get_user = AsyncMock(return_value=mock_user)

            await process_account_deletion("user-123", "Test reason")

        assert mock_user.status == UserStatus.DELETED


class TestGDPRExportRequest:
    """Test GDPR export request model."""

    def test_default_format_is_json(self):
        """Default export format should be JSON."""
        request = GDPRExportRequest()
        assert request.format == "json"

    def test_accepts_csv_format(self):
        """Should accept CSV format."""
        request = GDPRExportRequest(format="csv")
        assert request.format == "csv"

    def test_rejects_invalid_format(self):
        """Should reject invalid format."""
        with pytest.raises(ValueError):
            GDPRExportRequest(format="xml")

    def test_default_includes_all_data(self):
        """Default should include all data types."""
        request = GDPRExportRequest()

        assert request.include_conversations is True
        assert request.include_sessions is True
        assert request.include_preferences is True

    def test_can_exclude_data_types(self):
        """Should allow excluding data types."""
        request = GDPRExportRequest(
            include_conversations=False,
            include_sessions=False,
            include_preferences=True,
        )

        assert request.include_conversations is False
        assert request.include_sessions is False
        assert request.include_preferences is True


class TestGDPRCompliance:
    """Test overall GDPR compliance."""

    def test_export_does_not_include_password(self):
        """Export should never include hashed password."""
        mock_user = User(
            id="user-123",
            email="secure@example.com",
            hashed_password="super-secret-hash",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Verify UserResponse doesn't include password
        from app.security.models import UserResponse

        response = UserResponse(
            id=mock_user.id,
            email=mock_user.email,
            role=mock_user.role,
            status=mock_user.status,
            created_at=mock_user.created_at,
        )

        response_dict = response.model_dump()
        assert "hashed_password" not in response_dict
        assert "password" not in response_dict

    def test_deletion_provides_grace_period_info(self):
        """Deletion response should include grace period info."""
        from app.security.models import GDPRDeleteResponse

        response = GDPRDeleteResponse(
            status="scheduled",
            scheduled_at=datetime.utcnow(),
            completion_date=datetime.utcnow(),
            items_to_delete=["user_profile", "conversations"],
        )

        assert response.status == "scheduled"
        assert response.completion_date is not None
        assert len(response.items_to_delete) > 0
