"""
WebSocket Authentication Tests - Plan A Security

Tests for:
- Token validation before handshake
- Session ownership verification
- Connection rejection on invalid token
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.security.dependencies import websocket_auth, websocket_auth_optional
from app.security.models import TokenData, UserRole, TokenType
from datetime import datetime, timedelta


class TestWebSocketAuth:
    """Test WebSocket authentication."""

    @pytest.mark.asyncio
    async def test_valid_token_allows_connection(self):
        """Valid token should allow WebSocket connection."""
        mock_websocket = MagicMock()
        mock_websocket.headers = {}

        mock_token_data = TokenData(
            sub="user-123",
            email="test@example.com",
            role=UserRole.USER,
            type=TokenType.ACCESS,
            jti="token-id",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
        )

        with patch('app.security.dependencies.auth_service') as mock_auth:
            mock_auth.verify_token = AsyncMock(return_value=mock_token_data)

            result = await websocket_auth(
                websocket=mock_websocket,
                token="valid-token",
            )

        assert result.sub == "user-123"
        assert result.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_missing_token_closes_connection(self):
        """Missing token should close WebSocket connection."""
        mock_websocket = AsyncMock()
        mock_websocket.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await websocket_auth(
                websocket=mock_websocket,
                token=None,
            )

        assert exc_info.value.status_code == 401
        mock_websocket.close.assert_called_once_with(
            code=4001,
            reason="Authentication required",
        )

    @pytest.mark.asyncio
    async def test_invalid_token_closes_connection(self):
        """Invalid token should close WebSocket connection."""
        mock_websocket = AsyncMock()
        mock_websocket.headers = {}

        with patch('app.security.dependencies.auth_service') as mock_auth:
            mock_auth.verify_token = AsyncMock(
                side_effect=HTTPException(status_code=401, detail="Invalid token")
            )

            with pytest.raises(HTTPException):
                await websocket_auth(
                    websocket=mock_websocket,
                    token="invalid-token",
                )

        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_from_header_fallback(self):
        """Should try Authorization header if query param not provided."""
        mock_websocket = MagicMock()
        mock_websocket.headers = {"Authorization": "Bearer header-token"}

        mock_token_data = TokenData(
            sub="user-456",
            email="header@example.com",
            role=UserRole.USER,
            type=TokenType.ACCESS,
            jti="token-id",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
        )

        with patch('app.security.dependencies.auth_service') as mock_auth:
            mock_auth.verify_token = AsyncMock(return_value=mock_token_data)

            result = await websocket_auth(
                websocket=mock_websocket,
                token=None,
            )

        assert result.sub == "user-456"


class TestWebSocketAuthOptional:
    """Test optional WebSocket authentication."""

    @pytest.mark.asyncio
    async def test_returns_none_without_token(self):
        """Should return None when no token provided."""
        mock_websocket = MagicMock()
        mock_websocket.headers = {}

        result = await websocket_auth_optional(
            websocket=mock_websocket,
            token=None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_token_data_with_valid_token(self):
        """Should return token data when valid token provided."""
        mock_websocket = MagicMock()
        mock_websocket.headers = {}

        mock_token_data = TokenData(
            sub="user-789",
            email="optional@example.com",
            role=UserRole.USER,
            type=TokenType.ACCESS,
            jti="token-id",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
        )

        with patch('app.security.dependencies.auth_service') as mock_auth:
            mock_auth.verify_token = AsyncMock(return_value=mock_token_data)

            result = await websocket_auth_optional(
                websocket=mock_websocket,
                token="valid-token",
            )

        assert result is not None
        assert result.sub == "user-789"

    @pytest.mark.asyncio
    async def test_returns_none_with_invalid_token(self):
        """Should return None when invalid token provided (not raise)."""
        mock_websocket = MagicMock()
        mock_websocket.headers = {}

        with patch('app.security.dependencies.auth_service') as mock_auth:
            mock_auth.verify_token = AsyncMock(
                side_effect=HTTPException(status_code=401, detail="Invalid")
            )

            result = await websocket_auth_optional(
                websocket=mock_websocket,
                token="invalid-token",
            )

        assert result is None


class TestSessionOwnership:
    """Test session ownership verification."""

    @pytest.mark.asyncio
    async def test_owner_can_access_session(self):
        """Session owner should be able to access their session."""
        from app.security.dependencies import SessionOwnershipChecker

        mock_store = AsyncMock()
        mock_store.get_session = AsyncMock(return_value={"user_id": "user-123"})

        checker = SessionOwnershipChecker(session_store=mock_store)

        mock_user = MagicMock()
        mock_user.id = "user-123"

        with patch('app.security.dependencies.get_current_active_user', return_value=mock_user):
            result = await checker.verify(
                session_id="session-abc",
                user=mock_user,
            )

        assert result.id == "user-123"

    @pytest.mark.asyncio
    async def test_non_owner_cannot_access_session(self):
        """Non-owner should not be able to access session."""
        from app.security.dependencies import SessionOwnershipChecker

        mock_store = AsyncMock()
        mock_store.get_session = AsyncMock(return_value={"user_id": "user-123"})

        checker = SessionOwnershipChecker(session_store=mock_store)

        mock_user = MagicMock()
        mock_user.id = "user-456"  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await checker.verify(
                session_id="session-abc",
                user=mock_user,
            )

        assert exc_info.value.status_code == 403
        assert "access" in str(exc_info.value.detail).lower()
