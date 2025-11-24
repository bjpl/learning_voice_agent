"""
Security Models - User, Token, and Session Management

SPARC Implementation:
- Specification: Define all security-related data models
- Architecture: Pydantic models for validation, SQLAlchemy-compatible
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, field_validator
import re


class UserRole(str, Enum):
    """User roles for RBAC."""
    USER = "user"
    ADMIN = "admin"
    API_CLIENT = "api_client"


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class UserBase(BaseModel):
    """Base user model with common fields."""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """User registration model."""
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login credentials."""
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """User response model (excludes sensitive data)."""
    id: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """User profile update model."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class PasswordChange(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Ensure new password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class User(UserBase):
    """Full user model for internal use."""
    id: str
    hashed_password: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenType(str, Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token expiry in seconds")


class TokenData(BaseModel):
    """Decoded token payload."""
    sub: str  # User ID
    email: Optional[str] = None
    role: UserRole = UserRole.USER
    type: TokenType = TokenType.ACCESS
    jti: str  # JWT ID for blacklisting
    exp: datetime
    iat: datetime


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class TokenBlacklist(BaseModel):
    """Blacklisted token record."""
    jti: str
    user_id: str
    token_type: TokenType
    blacklisted_at: datetime
    expires_at: datetime
    reason: Optional[str] = None


class GDPRExportRequest(BaseModel):
    """GDPR data export request."""
    format: str = Field(default="json", pattern="^(json|csv)$")
    include_conversations: bool = True
    include_sessions: bool = True
    include_preferences: bool = True


class GDPRExportResponse(BaseModel):
    """GDPR data export response."""
    export_id: str
    status: str
    created_at: datetime
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class GDPRDeleteRequest(BaseModel):
    """GDPR data deletion request."""
    reason: Optional[str] = None
    confirm: bool = Field(..., description="Must be True to confirm deletion")

    @field_validator('confirm')
    @classmethod
    def must_confirm(cls, v: bool) -> bool:
        if not v:
            raise ValueError('You must confirm the deletion request')
        return v


class GDPRDeleteResponse(BaseModel):
    """GDPR data deletion response."""
    status: str
    scheduled_at: datetime
    completion_date: datetime
    items_to_delete: List[str]


class SessionInfo(BaseModel):
    """Active session information."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_current: bool = False


class SecurityAuditLog(BaseModel):
    """Security audit log entry."""
    id: str
    user_id: Optional[str]
    action: str
    resource: str
    ip_address: str
    user_agent: Optional[str]
    timestamp: datetime
    success: bool
    details: Optional[dict] = None
