"""
Security Module - Plan A Security-First Implementation

This module provides comprehensive security features:
- JWT Authentication with access/refresh tokens
- Rate limiting with Redis backend
- CORS configuration management
- WebSocket authentication
- GDPR compliance utilities
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
"""

from app.security.auth import (
    AuthService,
    get_current_user,
    get_current_active_user,
    get_optional_user,
    verify_token,
    create_access_token,
    create_refresh_token,
)
from app.security.models import (
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    TokenData,
    TokenBlacklist,
)
from app.security.rate_limit import (
    RateLimiter,
    rate_limit,
    get_rate_limiter,
)
from app.security.dependencies import (
    require_auth,
    require_admin,
    websocket_auth,
)
from app.security.headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    WebSocketOriginValidator,
    get_websocket_origin_validator,
    validate_websocket_origin,
    get_security_headers_config,
)

__all__ = [
    # Auth
    "AuthService",
    "get_current_user",
    "get_current_active_user",
    "get_optional_user",
    "verify_token",
    "create_access_token",
    "create_refresh_token",
    # Models
    "User",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "Token",
    "TokenData",
    "TokenBlacklist",
    # Rate Limiting
    "RateLimiter",
    "rate_limit",
    "get_rate_limiter",
    # Dependencies
    "require_auth",
    "require_admin",
    "websocket_auth",
    # Security Headers
    "SecurityHeadersMiddleware",
    "SecurityHeadersConfig",
    "WebSocketOriginValidator",
    "get_websocket_origin_validator",
    "validate_websocket_origin",
    "get_security_headers_config",
]
