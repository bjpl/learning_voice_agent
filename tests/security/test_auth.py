"""
JWT Authentication Tests - Plan A Security

Tests for:
- User registration
- Login/logout
- Token refresh
- Password validation
- Account lockout
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.security.auth import (
    AuthService,
    SecurityConfig,
    auth_service,
    verify_token,
    get_current_user,
)
from app.security.models import (
    UserCreate,
    UserLogin,
    UserRole,
    UserStatus,
    TokenType,
)


@pytest.fixture
def auth_svc():
    """Fresh AuthService instance for testing."""
    return AuthService()


class TestPasswordHashing:
    """Test password hashing functionality."""

    def test_hash_password(self, auth_svc):
        """Password should be hashed."""
        password = "TestPassword123"
        hashed = auth_svc.hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_verify_correct_password(self, auth_svc):
        """Correct password should verify."""
        password = "TestPassword123"
        hashed = auth_svc.hash_password(password)

        assert auth_svc.verify_password(password, hashed) is True

    def test_verify_wrong_password(self, auth_svc):
        """Wrong password should not verify."""
        password = "TestPassword123"
        hashed = auth_svc.hash_password(password)

        assert auth_svc.verify_password("WrongPassword123", hashed) is False


class TestUserRegistration:
    """Test user registration."""

    @pytest.mark.asyncio
    async def test_create_user_success(self, auth_svc):
        """Valid user data should create user."""
        user_data = UserCreate(
            email="test@example.com",
            password="ValidPass123",
            full_name="Test User",
        )

        user = await auth_svc.create_user(user_data)

        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.id is not None
        assert user.role == UserRole.USER

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, auth_svc):
        """Duplicate email should fail."""
        user_data = UserCreate(
            email="duplicate@example.com",
            password="ValidPass123",
        )

        await auth_svc.create_user(user_data)

        with pytest.raises(Exception) as exc_info:
            await auth_svc.create_user(user_data)

        assert "already registered" in str(exc_info.value.detail)

    def test_password_validation_too_short(self):
        """Short password should fail validation."""
        with pytest.raises(ValueError) as exc_info:
            UserCreate(
                email="test@example.com",
                password="Short1",
            )

        assert "8 characters" in str(exc_info.value)

    def test_password_validation_no_uppercase(self):
        """Password without uppercase should fail."""
        with pytest.raises(ValueError) as exc_info:
            UserCreate(
                email="test@example.com",
                password="lowercase123",
            )

        assert "uppercase" in str(exc_info.value)

    def test_password_validation_no_lowercase(self):
        """Password without lowercase should fail."""
        with pytest.raises(ValueError) as exc_info:
            UserCreate(
                email="test@example.com",
                password="UPPERCASE123",
            )

        assert "lowercase" in str(exc_info.value)

    def test_password_validation_no_digit(self):
        """Password without digit should fail."""
        with pytest.raises(ValueError) as exc_info:
            UserCreate(
                email="test@example.com",
                password="NoDigitsHere",
            )

        assert "digit" in str(exc_info.value)


class TestUserAuthentication:
    """Test user login."""

    @pytest.mark.asyncio
    async def test_login_success(self, auth_svc):
        """Valid credentials should return tokens."""
        # Create user first
        await auth_svc.create_user(UserCreate(
            email="login@example.com",
            password="ValidPass123",
        ))

        # Login
        credentials = UserLogin(
            email="login@example.com",
            password="ValidPass123",
        )
        user, token = await auth_svc.authenticate_user(credentials)

        assert user.email == "login@example.com"
        assert token.access_token is not None
        assert token.refresh_token is not None
        assert token.token_type == "bearer"
        assert token.expires_in > 0

    @pytest.mark.asyncio
    async def test_login_wrong_email(self, auth_svc):
        """Non-existent email should fail."""
        credentials = UserLogin(
            email="nonexistent@example.com",
            password="ValidPass123",
        )

        with pytest.raises(Exception) as exc_info:
            await auth_svc.authenticate_user(credentials)

        assert "Invalid email or password" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, auth_svc):
        """Wrong password should fail."""
        await auth_svc.create_user(UserCreate(
            email="wrongpass@example.com",
            password="ValidPass123",
        ))

        credentials = UserLogin(
            email="wrongpass@example.com",
            password="WrongPass456",
        )

        with pytest.raises(Exception) as exc_info:
            await auth_svc.authenticate_user(credentials)

        assert "Invalid email or password" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_account_lockout_after_failed_attempts(self, auth_svc):
        """Account should lock after max failed attempts."""
        await auth_svc.create_user(UserCreate(
            email="lockout@example.com",
            password="ValidPass123",
        ))

        wrong_credentials = UserLogin(
            email="lockout@example.com",
            password="WrongPassword",
        )

        # Attempt login with wrong password multiple times
        for i in range(SecurityConfig.PASSWORD_MAX_ATTEMPTS):
            with pytest.raises(Exception):
                await auth_svc.authenticate_user(wrong_credentials)

        # Next attempt should show locked message
        with pytest.raises(Exception) as exc_info:
            await auth_svc.authenticate_user(wrong_credentials)

        assert "locked" in str(exc_info.value.detail).lower()


class TestTokenOperations:
    """Test JWT token operations."""

    @pytest.mark.asyncio
    async def test_verify_valid_token(self, auth_svc):
        """Valid token should verify."""
        await auth_svc.create_user(UserCreate(
            email="token@example.com",
            password="ValidPass123",
        ))

        _, token = await auth_svc.authenticate_user(UserLogin(
            email="token@example.com",
            password="ValidPass123",
        ))

        token_data = await auth_svc.verify_token(token.access_token)

        assert token_data.email == "token@example.com"
        assert token_data.type == TokenType.ACCESS

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, auth_svc):
        """Expired token should fail verification."""
        # Create token with very short expiry
        with patch.object(SecurityConfig, 'ACCESS_TOKEN_EXPIRE_MINUTES', -1):
            token, _ = auth_svc._create_token(
                user_id="test-id",
                email="test@example.com",
                role=UserRole.USER,
                token_type=TokenType.ACCESS,
                expires_delta=timedelta(minutes=-1),  # Already expired
            )

        with pytest.raises(Exception) as exc_info:
            await auth_svc.verify_token(token)

        assert "Invalid or expired" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_refresh_token(self, auth_svc):
        """Refresh token should return new access token."""
        await auth_svc.create_user(UserCreate(
            email="refresh@example.com",
            password="ValidPass123",
        ))

        _, original_token = await auth_svc.authenticate_user(UserLogin(
            email="refresh@example.com",
            password="ValidPass123",
        ))

        new_token = await auth_svc.refresh_access_token(original_token.refresh_token)

        assert new_token.access_token is not None
        assert new_token.access_token != original_token.access_token

    @pytest.mark.asyncio
    async def test_logout_blacklists_token(self, auth_svc):
        """Logout should blacklist token."""
        await auth_svc.create_user(UserCreate(
            email="logout@example.com",
            password="ValidPass123",
        ))

        _, token = await auth_svc.authenticate_user(UserLogin(
            email="logout@example.com",
            password="ValidPass123",
        ))

        # Logout
        await auth_svc.logout(token.access_token, token.refresh_token)

        # Token should now be invalid
        with pytest.raises(Exception) as exc_info:
            await auth_svc.verify_token(token.access_token)

        assert "revoked" in str(exc_info.value.detail)


class TestInactiveUser:
    """Test handling of inactive users."""

    @pytest.mark.asyncio
    async def test_inactive_user_cannot_login(self, auth_svc):
        """Inactive user should not be able to login."""
        user_response = await auth_svc.create_user(UserCreate(
            email="inactive@example.com",
            password="ValidPass123",
        ))

        # Mark user as inactive
        user = auth_svc._users[user_response.id]
        user.status = UserStatus.INACTIVE

        credentials = UserLogin(
            email="inactive@example.com",
            password="ValidPass123",
        )

        with pytest.raises(Exception) as exc_info:
            await auth_svc.authenticate_user(credentials)

        assert "inactive" in str(exc_info.value.detail).lower()
