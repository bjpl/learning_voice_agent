"""
Security Test Fixtures - Plan A Implementation

Common fixtures for security tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from app.security.auth import AuthService
from app.security.models import User, UserRole, UserStatus, UserCreate


@pytest.fixture
def auth_service():
    """Fresh AuthService instance for testing."""
    return AuthService()


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return User(
        id="test-user-123",
        email="test@example.com",
        full_name="Test User",
        hashed_password="$2b$12$hashedpassword",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user for testing."""
    return User(
        id="admin-user-456",
        email="admin@example.com",
        full_name="Admin User",
        hashed_password="$2b$12$hashedpassword",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock()
    request.url.path = "/api/test"
    request.client.host = "127.0.0.1"
    request.headers = {}
    return request


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    websocket = AsyncMock()
    websocket.headers = {}
    return websocket


@pytest.fixture
def valid_user_create():
    """Valid user creation data."""
    return UserCreate(
        email="newuser@example.com",
        password="ValidPass123",
        full_name="New User",
    )

