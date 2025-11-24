"""
Security Headers Middleware - Plan A Security Implementation

SPARC Implementation:
- Specification: Add defense-in-depth security headers
- Architecture: Starlette middleware with configurable headers
- Features: CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy

Security Headers Implemented:
1. Content-Security-Policy (CSP) - Prevent XSS and data injection
2. Strict-Transport-Security (HSTS) - Enforce HTTPS
3. X-Frame-Options - Prevent clickjacking
4. X-Content-Type-Options - Prevent MIME sniffing
5. Referrer-Policy - Control referrer information
6. X-XSS-Protection - Legacy XSS protection
7. Permissions-Policy - Control browser features
"""

import os
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


@dataclass
class SecurityHeadersConfig:
    """
    Configuration for security headers.

    All values can be overridden via environment variables.
    """

    # Content-Security-Policy
    csp_enabled: bool = True
    csp_default_src: str = "'self'"
    csp_script_src: str = "'self'"
    csp_style_src: str = "'self' 'unsafe-inline'"  # Some frameworks need inline styles
    csp_img_src: str = "'self' data: https:"
    csp_font_src: str = "'self' https:"
    csp_connect_src: str = "'self'"
    csp_frame_ancestors: str = "'none'"
    csp_base_uri: str = "'self'"
    csp_form_action: str = "'self'"
    csp_report_uri: Optional[str] = None
    csp_report_only: bool = False  # Use Content-Security-Policy-Report-Only

    # Strict-Transport-Security (HSTS)
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year in seconds
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False  # Only enable after thorough testing

    # X-Frame-Options
    frame_options_enabled: bool = True
    frame_options_value: str = "DENY"  # DENY, SAMEORIGIN, or ALLOW-FROM uri

    # X-Content-Type-Options
    content_type_options_enabled: bool = True

    # Referrer-Policy
    referrer_policy_enabled: bool = True
    referrer_policy_value: str = "strict-origin-when-cross-origin"

    # X-XSS-Protection (legacy, but still useful for older browsers)
    xss_protection_enabled: bool = True
    xss_protection_value: str = "1; mode=block"

    # Permissions-Policy (formerly Feature-Policy)
    permissions_policy_enabled: bool = True
    permissions_policy_value: str = (
        "accelerometer=(), "
        "camera=(), "
        "geolocation=(), "
        "gyroscope=(), "
        "magnetometer=(), "
        "microphone=(self), "  # Allow microphone for voice features
        "payment=(), "
        "usb=()"
    )

    # Cache-Control for security-sensitive responses
    cache_control_enabled: bool = True
    cache_control_value: str = "no-store, no-cache, must-revalidate, private"

    # Paths to exclude from certain headers (e.g., static files)
    excluded_paths: Set[str] = field(default_factory=lambda: {"/static", "/health"})

    @classmethod
    def from_environment(cls) -> "SecurityHeadersConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - SECURITY_CSP_ENABLED: Enable/disable CSP (default: true)
        - SECURITY_CSP_DEFAULT_SRC: CSP default-src directive
        - SECURITY_CSP_SCRIPT_SRC: CSP script-src directive
        - SECURITY_CSP_REPORT_URI: CSP report-uri for violation reports
        - SECURITY_HSTS_ENABLED: Enable/disable HSTS (default: true)
        - SECURITY_HSTS_MAX_AGE: HSTS max-age in seconds
        - SECURITY_HSTS_PRELOAD: Enable HSTS preload
        - SECURITY_FRAME_OPTIONS: X-Frame-Options value
        - SECURITY_REFERRER_POLICY: Referrer-Policy value
        - SECURITY_PERMISSIONS_POLICY: Permissions-Policy value
        """
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, "").lower()
            if value in ("true", "1", "yes"):
                return True
            elif value in ("false", "0", "no"):
                return False
            return default

        def get_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default

        environment = os.getenv("ENVIRONMENT", "development").lower()
        is_production = environment == "production"

        # In production, enable stricter defaults
        config = cls(
            # CSP
            csp_enabled=get_bool("SECURITY_CSP_ENABLED", True),
            csp_default_src=os.getenv("SECURITY_CSP_DEFAULT_SRC", "'self'"),
            csp_script_src=os.getenv("SECURITY_CSP_SCRIPT_SRC", "'self'"),
            csp_style_src=os.getenv("SECURITY_CSP_STYLE_SRC", "'self' 'unsafe-inline'"),
            csp_img_src=os.getenv("SECURITY_CSP_IMG_SRC", "'self' data: https:"),
            csp_connect_src=os.getenv("SECURITY_CSP_CONNECT_SRC", "'self'"),
            csp_report_uri=os.getenv("SECURITY_CSP_REPORT_URI"),
            csp_report_only=get_bool("SECURITY_CSP_REPORT_ONLY", not is_production),

            # HSTS
            hsts_enabled=get_bool("SECURITY_HSTS_ENABLED", is_production),
            hsts_max_age=get_int("SECURITY_HSTS_MAX_AGE", 31536000),
            hsts_include_subdomains=get_bool("SECURITY_HSTS_INCLUDE_SUBDOMAINS", True),
            hsts_preload=get_bool("SECURITY_HSTS_PRELOAD", False),

            # Frame Options
            frame_options_enabled=get_bool("SECURITY_FRAME_OPTIONS_ENABLED", True),
            frame_options_value=os.getenv("SECURITY_FRAME_OPTIONS", "DENY"),

            # Content Type Options
            content_type_options_enabled=get_bool("SECURITY_CONTENT_TYPE_OPTIONS_ENABLED", True),

            # Referrer Policy
            referrer_policy_enabled=get_bool("SECURITY_REFERRER_POLICY_ENABLED", True),
            referrer_policy_value=os.getenv(
                "SECURITY_REFERRER_POLICY",
                "strict-origin-when-cross-origin"
            ),

            # XSS Protection
            xss_protection_enabled=get_bool("SECURITY_XSS_PROTECTION_ENABLED", True),

            # Permissions Policy
            permissions_policy_enabled=get_bool("SECURITY_PERMISSIONS_POLICY_ENABLED", True),
            permissions_policy_value=os.getenv(
                "SECURITY_PERMISSIONS_POLICY",
                "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                "magnetometer=(), microphone=(self), payment=(), usb=()"
            ),

            # Cache Control
            cache_control_enabled=get_bool("SECURITY_CACHE_CONTROL_ENABLED", True),
        )

        return config


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all HTTP responses.

    PATTERN: Defense in depth through HTTP headers
    WHY: Protect against common web vulnerabilities

    Usage:
        app.add_middleware(SecurityHeadersMiddleware)
        # or with custom config
        app.add_middleware(SecurityHeadersMiddleware, config=custom_config)
    """

    def __init__(self, app, config: Optional[SecurityHeadersConfig] = None):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig.from_environment()
        self._headers_cache: Optional[Dict[str, str]] = None
        logger.info("Security headers middleware initialized")
        self._log_config()

    def _log_config(self) -> None:
        """Log configuration for debugging."""
        logger.debug(f"CSP enabled: {self.config.csp_enabled}")
        logger.debug(f"HSTS enabled: {self.config.hsts_enabled}")
        logger.debug(f"X-Frame-Options: {self.config.frame_options_value}")
        logger.debug(f"Referrer-Policy: {self.config.referrer_policy_value}")

    def _build_csp_header(self) -> str:
        """Build Content-Security-Policy header value."""
        directives = [
            f"default-src {self.config.csp_default_src}",
            f"script-src {self.config.csp_script_src}",
            f"style-src {self.config.csp_style_src}",
            f"img-src {self.config.csp_img_src}",
            f"font-src {self.config.csp_font_src}",
            f"connect-src {self.config.csp_connect_src}",
            f"frame-ancestors {self.config.csp_frame_ancestors}",
            f"base-uri {self.config.csp_base_uri}",
            f"form-action {self.config.csp_form_action}",
        ]

        if self.config.csp_report_uri:
            directives.append(f"report-uri {self.config.csp_report_uri}")

        return "; ".join(directives)

    def _build_hsts_header(self) -> str:
        """Build Strict-Transport-Security header value."""
        parts = [f"max-age={self.config.hsts_max_age}"]

        if self.config.hsts_include_subdomains:
            parts.append("includeSubDomains")

        if self.config.hsts_preload:
            parts.append("preload")

        return "; ".join(parts)

    def _get_security_headers(self) -> Dict[str, str]:
        """
        Get all security headers to add.

        Caches headers for performance since they don't change per-request.
        """
        if self._headers_cache is not None:
            return self._headers_cache

        headers: Dict[str, str] = {}

        # Content-Security-Policy
        if self.config.csp_enabled:
            csp_value = self._build_csp_header()
            if self.config.csp_report_only:
                headers["Content-Security-Policy-Report-Only"] = csp_value
            else:
                headers["Content-Security-Policy"] = csp_value

        # Strict-Transport-Security (HSTS)
        if self.config.hsts_enabled:
            headers["Strict-Transport-Security"] = self._build_hsts_header()

        # X-Frame-Options
        if self.config.frame_options_enabled:
            headers["X-Frame-Options"] = self.config.frame_options_value

        # X-Content-Type-Options
        if self.config.content_type_options_enabled:
            headers["X-Content-Type-Options"] = "nosniff"

        # Referrer-Policy
        if self.config.referrer_policy_enabled:
            headers["Referrer-Policy"] = self.config.referrer_policy_value

        # X-XSS-Protection
        if self.config.xss_protection_enabled:
            headers["X-XSS-Protection"] = self.config.xss_protection_value

        # Permissions-Policy
        if self.config.permissions_policy_enabled:
            headers["Permissions-Policy"] = self.config.permissions_policy_value

        self._headers_cache = headers
        return headers

    def _should_add_headers(self, path: str) -> bool:
        """Check if headers should be added for this path."""
        for excluded in self.config.excluded_paths:
            if path.startswith(excluded):
                return False
        return True

    def _should_add_cache_control(self, path: str, content_type: str) -> bool:
        """Check if Cache-Control should be added."""
        # Don't add cache control for static assets
        if path.startswith("/static"):
            return False

        # Add for API responses and HTML
        if "application/json" in content_type or "text/html" in content_type:
            return True

        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        path = request.url.path

        # Add security headers
        if self._should_add_headers(path):
            for header_name, header_value in self._get_security_headers().items():
                response.headers[header_name] = header_value

        # Add Cache-Control for sensitive responses
        if self.config.cache_control_enabled:
            content_type = response.headers.get("content-type", "")
            if self._should_add_cache_control(path, content_type):
                response.headers["Cache-Control"] = self.config.cache_control_value
                response.headers["Pragma"] = "no-cache"

        return response


class WebSocketOriginValidator:
    """
    Validator for WebSocket Origin headers.

    PATTERN: Origin validation for WebSocket connections
    WHY: Prevent Cross-Site WebSocket Hijacking (CSWSH)

    Usage:
        validator = WebSocketOriginValidator()
        if not validator.is_valid_origin(websocket):
            await websocket.close(code=4003)
    """

    def __init__(self, allowed_origins: Optional[List[str]] = None):
        """
        Initialize validator.

        Args:
            allowed_origins: List of allowed origins. If None, uses CORS_ORIGINS.
        """
        self._allowed_origins = allowed_origins

    @property
    def allowed_origins(self) -> List[str]:
        """Get allowed origins, loading from environment if needed."""
        if self._allowed_origins is not None:
            return self._allowed_origins

        # Import here to avoid circular imports
        from app.security.cors import get_cors_origins
        return get_cors_origins()

    def is_valid_origin(self, websocket: WebSocket) -> bool:
        """
        Check if WebSocket origin is valid.

        Args:
            websocket: WebSocket connection to validate

        Returns:
            True if origin is allowed, False otherwise
        """
        origin = websocket.headers.get("origin")

        # No origin header - might be same-origin or non-browser client
        # Be permissive for development but stricter for production
        if not origin:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                logger.warning("WebSocket connection without Origin header in production")
                return False
            return True

        allowed = self.allowed_origins

        # If no origins configured, reject in production
        if not allowed:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                logger.warning(f"WebSocket rejected - no allowed origins configured")
                return False
            # In development, allow localhost
            return origin.startswith(("http://localhost", "http://127.0.0.1"))

        # Check if origin is in allowed list
        is_allowed = origin in allowed

        if not is_allowed:
            logger.warning(f"WebSocket rejected - origin not allowed: {origin}")

        return is_allowed

    async def validate_or_reject(self, websocket: WebSocket) -> bool:
        """
        Validate origin and close connection if invalid.

        Args:
            websocket: WebSocket connection to validate

        Returns:
            True if valid, raises HTTPException if invalid

        Raises:
            HTTPException: 403 if origin is not allowed
        """
        if not self.is_valid_origin(websocket):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="WebSocket origin not allowed"
            )
        return True


# Singleton instance
_websocket_origin_validator: Optional[WebSocketOriginValidator] = None


def get_websocket_origin_validator() -> WebSocketOriginValidator:
    """Get or create WebSocket origin validator singleton."""
    global _websocket_origin_validator
    if _websocket_origin_validator is None:
        _websocket_origin_validator = WebSocketOriginValidator()
    return _websocket_origin_validator


async def validate_websocket_origin(websocket: WebSocket) -> bool:
    """
    FastAPI dependency for WebSocket origin validation.

    Usage:
        @app.websocket("/ws/{session_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            session_id: str,
            origin_valid: bool = Depends(validate_websocket_origin)
        ):
            ...
    """
    validator = get_websocket_origin_validator()
    return await validator.validate_or_reject(websocket)


def get_security_headers_config() -> SecurityHeadersConfig:
    """Get current security headers configuration."""
    return SecurityHeadersConfig.from_environment()
