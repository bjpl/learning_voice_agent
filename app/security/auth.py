"""
JWT Authentication Service - Plan A Security Implementation

SPARC Implementation:
- Specification: Token-based auth for all API endpoints
- Pseudocode: See SECURITY_ARCHITECTURE.md
- Architecture: Stateless JWT with Redis blacklist
- Refinement: bcrypt password hashing, RS256/HS256 tokens
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.security.models import (
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    UserRole,
    UserStatus,
    Token,
    TokenData,
    TokenType,
    TokenBlacklist,
)
from app.config import settings

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


class SecurityConfig:
    """Security configuration from environment."""
    # JWT Settings
    SECRET_KEY: str = getattr(settings, 'jwt_secret_key', 'dev-secret-key-change-in-production')
    ALGORITHM: str = getattr(settings, 'jwt_algorithm', 'HS256')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = getattr(settings, 'jwt_access_expire_minutes', 15)
    REFRESH_TOKEN_EXPIRE_DAYS: int = getattr(settings, 'jwt_refresh_expire_days', 7)

    # Password Policy
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_MAX_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15


class AuthService:
    """
    Authentication service handling user management and token operations.

    PATTERN: Service class with dependency injection
    WHY: Testable, maintainable, single responsibility
    """

    def __init__(self):
        self._users: dict[str, User] = {}  # In-memory for now, replace with DB
        self._blacklist: dict[str, TokenBlacklist] = {}
        self._refresh_tokens: dict[str, str] = {}  # jti -> user_id mapping

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """
        Create a new user account.

        Args:
            user_data: User registration data

        Returns:
            Created user (without sensitive data)

        Raises:
            HTTPException: If email already exists
        """
        # Check for existing user
        for user in self._users.values():
            if user.email == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )

        # Create user
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()

        user = User(
            id=user_id,
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=self.hash_password(user_data.password),
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

        self._users[user_id] = user

        logger.info(f"Created user: {user_id} ({user_data.email})")

        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            status=user.status,
            created_at=user.created_at,
        )

    async def authenticate_user(self, credentials: UserLogin) -> Tuple[User, Token]:
        """
        Authenticate a user and return tokens.

        Args:
            credentials: Login credentials

        Returns:
            Tuple of (User, Token)

        Raises:
            HTTPException: If authentication fails
        """
        # Find user by email
        user = None
        for u in self._users.values():
            if u.email == credentials.email:
                user = u
                break

        if not user:
            logger.warning(f"Login attempt for non-existent user: {credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            remaining = (user.locked_until - datetime.utcnow()).seconds // 60
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked. Try again in {remaining} minutes.",
            )

        # Check if account is active
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is {user.status.value}",
            )

        # Verify password
        if not self.verify_password(credentials.password, user.hashed_password):
            # Increment failed attempts
            user.failed_login_attempts += 1

            if user.failed_login_attempts >= SecurityConfig.PASSWORD_MAX_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(
                    minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES
                )
                logger.warning(f"Account locked due to failed attempts: {user.id}")

            logger.warning(f"Failed login attempt for user: {user.id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()

        # Generate tokens
        access_token, access_jti = self._create_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            token_type=TokenType.ACCESS,
            expires_delta=timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
        )

        refresh_token, refresh_jti = self._create_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            token_type=TokenType.REFRESH,
            expires_delta=timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS),
        )

        # Store refresh token mapping
        self._refresh_tokens[refresh_jti] = user.id

        logger.info(f"User logged in: {user.id}")

        return user, Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    def _create_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        token_type: TokenType,
        expires_delta: timedelta,
    ) -> Tuple[str, str]:
        """
        Create a JWT token.

        Returns:
            Tuple of (token_string, jti)
        """
        jti = str(uuid.uuid4())
        now = datetime.utcnow()
        expire = now + expires_delta

        payload = {
            "sub": user_id,
            "email": email,
            "role": role.value,
            "type": token_type.value,
            "jti": jti,
            "exp": expire,
            "iat": now,
        }

        token = jwt.encode(payload, SecurityConfig.SECRET_KEY, algorithm=SecurityConfig.ALGORITHM)
        return token, jti

    async def refresh_access_token(self, refresh_token: str) -> Token:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New Token with fresh access token

        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            payload = jwt.decode(
                refresh_token,
                SecurityConfig.SECRET_KEY,
                algorithms=[SecurityConfig.ALGORITHM]
            )

            # Validate token type
            if payload.get("type") != TokenType.REFRESH.value:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )

            jti = payload.get("jti")
            user_id = payload.get("sub")

            # Check if token is blacklisted
            if jti in self._blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            # Verify token is in our records
            if jti not in self._refresh_tokens or self._refresh_tokens[jti] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token",
                )

            # Get user
            user = self._users.get(user_id)
            if not user or user.status != UserStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive",
                )

            # Generate new access token
            access_token, _ = self._create_token(
                user_id=user.id,
                email=user.email,
                role=user.role,
                token_type=TokenType.ACCESS,
                expires_delta=timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
            )

            return Token(
                access_token=access_token,
                refresh_token=refresh_token,  # Return same refresh token
                token_type="bearer",
                expires_in=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

        except JWTError as e:
            logger.warning(f"Invalid refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

    async def logout(self, access_token: str, refresh_token: Optional[str] = None) -> None:
        """
        Logout user by blacklisting tokens.

        Args:
            access_token: Current access token
            refresh_token: Optional refresh token to revoke
        """
        try:
            # Blacklist access token
            payload = jwt.decode(
                access_token,
                SecurityConfig.SECRET_KEY,
                algorithms=[SecurityConfig.ALGORITHM]
            )

            self._blacklist[payload["jti"]] = TokenBlacklist(
                jti=payload["jti"],
                user_id=payload["sub"],
                token_type=TokenType.ACCESS,
                blacklisted_at=datetime.utcnow(),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                reason="logout",
            )

            # Blacklist refresh token if provided
            if refresh_token:
                try:
                    refresh_payload = jwt.decode(
                        refresh_token,
                        SecurityConfig.SECRET_KEY,
                        algorithms=[SecurityConfig.ALGORITHM]
                    )

                    self._blacklist[refresh_payload["jti"]] = TokenBlacklist(
                        jti=refresh_payload["jti"],
                        user_id=refresh_payload["sub"],
                        token_type=TokenType.REFRESH,
                        blacklisted_at=datetime.utcnow(),
                        expires_at=datetime.fromtimestamp(refresh_payload["exp"]),
                        reason="logout",
                    )

                    # Remove from refresh tokens
                    if refresh_payload["jti"] in self._refresh_tokens:
                        del self._refresh_tokens[refresh_payload["jti"]]

                except JWTError:
                    pass  # Ignore invalid refresh token

            logger.info(f"User logged out: {payload['sub']}")

        except JWTError as e:
            logger.warning(f"Logout with invalid token: {e}")

    async def verify_token(self, token: str) -> TokenData:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token data

        Raises:
            HTTPException: If token is invalid or blacklisted
        """
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.SECRET_KEY,
                algorithms=[SecurityConfig.ALGORITHM]
            )

            jti = payload.get("jti")

            # Check blacklist
            if jti in self._blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            return TokenData(
                sub=payload["sub"],
                email=payload.get("email"),
                role=UserRole(payload.get("role", "user")),
                type=TokenType(payload.get("type", "access")),
                jti=jti,
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
            )

        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None


# Singleton instance
auth_service = AuthService()


# FastAPI Dependencies

async def verify_token(token: str = Depends(oauth2_scheme)) -> TokenData:
    """Dependency to verify JWT token."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await auth_service.verify_token(token)


async def get_current_user(token_data: TokenData = Depends(verify_token)) -> User:
    """Dependency to get current authenticated user."""
    user = await auth_service.get_user(token_data.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    """Dependency to get current active user."""
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status.value}",
        )
    return user


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """Dependency to optionally get current user (for public endpoints)."""
    if not token:
        return None
    try:
        token_data = await auth_service.verify_token(token)
        return await auth_service.get_user(token_data.sub)
    except HTTPException:
        return None


def create_access_token(user_id: str, email: str, role: UserRole = UserRole.USER) -> str:
    """Helper to create access token."""
    token, _ = auth_service._create_token(
        user_id=user_id,
        email=email,
        role=role,
        token_type=TokenType.ACCESS,
        expires_delta=timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return token


def create_refresh_token(user_id: str, email: str, role: UserRole = UserRole.USER) -> str:
    """Helper to create refresh token."""
    token, jti = auth_service._create_token(
        user_id=user_id,
        email=email,
        role=role,
        token_type=TokenType.REFRESH,
        expires_delta=timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    auth_service._refresh_tokens[jti] = user_id
    return token
