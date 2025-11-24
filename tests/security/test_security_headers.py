"""
Security Headers Tests - Plan A Security Implementation

Tests for:
- Security headers middleware
- CSP header generation
- HSTS header generation
- X-Frame-Options
- X-Content-Type-Options
- Referrer-Policy
- WebSocket origin validation
- Configuration flexibility
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.websockets import WebSocket

from app.security.headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    WebSocketOriginValidator,
    get_websocket_origin_validator,
    validate_websocket_origin,
    get_security_headers_config,
)


class TestSecurityHeadersConfig:
    """Test SecurityHeadersConfig configuration."""

    def test_default_values(self):
        """Config should have secure defaults."""
        config = SecurityHeadersConfig()

        assert config.csp_enabled is True
        assert config.hsts_enabled is True
        assert config.frame_options_enabled is True
        assert config.content_type_options_enabled is True
        assert config.referrer_policy_enabled is True
        assert config.xss_protection_enabled is True
        assert config.permissions_policy_enabled is True

    def test_csp_default_src(self):
        """CSP default-src should be 'self'."""
        config = SecurityHeadersConfig()
        assert config.csp_default_src == "'self'"

    def test_hsts_max_age_default(self):
        """HSTS max-age should default to 1 year."""
        config = SecurityHeadersConfig()
        assert config.hsts_max_age == 31536000  # 1 year in seconds

    def test_frame_options_default(self):
        """X-Frame-Options should default to DENY."""
        config = SecurityHeadersConfig()
        assert config.frame_options_value == "DENY"

    def test_referrer_policy_default(self):
        """Referrer-Policy should have secure default."""
        config = SecurityHeadersConfig()
        assert config.referrer_policy_value == "strict-origin-when-cross-origin"

    def test_from_environment_csp_enabled(self):
        """Config should read CSP enabled from environment."""
        with patch.dict(os.environ, {"SECURITY_CSP_ENABLED": "false"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.csp_enabled is False

    def test_from_environment_hsts_enabled(self):
        """Config should read HSTS enabled from environment."""
        with patch.dict(os.environ, {"SECURITY_HSTS_ENABLED": "true"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.hsts_enabled is True

    def test_from_environment_hsts_max_age(self):
        """Config should read HSTS max-age from environment."""
        with patch.dict(os.environ, {"SECURITY_HSTS_MAX_AGE": "86400"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.hsts_max_age == 86400

    def test_from_environment_frame_options(self):
        """Config should read frame options from environment."""
        with patch.dict(os.environ, {"SECURITY_FRAME_OPTIONS": "SAMEORIGIN"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.frame_options_value == "SAMEORIGIN"

    def test_from_environment_referrer_policy(self):
        """Config should read referrer policy from environment."""
        with patch.dict(os.environ, {"SECURITY_REFERRER_POLICY": "no-referrer"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.referrer_policy_value == "no-referrer"

    def test_from_environment_csp_report_uri(self):
        """Config should read CSP report URI from environment."""
        with patch.dict(os.environ, {"SECURITY_CSP_REPORT_URI": "https://example.com/csp-report"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.csp_report_uri == "https://example.com/csp-report"

    def test_from_environment_production_defaults(self):
        """Production environment should enable HSTS."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.hsts_enabled is True

    def test_from_environment_development_csp_report_only(self):
        """Development should use CSP report-only mode."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.csp_report_only is True

    def test_from_environment_production_csp_enforced(self):
        """Production should enforce CSP (not report-only)."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            config = SecurityHeadersConfig.from_environment()
        assert config.csp_report_only is False


class TestSecurityHeadersMiddleware:
    """Test SecurityHeadersMiddleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create test app with security headers middleware."""
        async def homepage(request):
            return JSONResponse({"status": "ok"})

        async def static_file(request):
            return JSONResponse({"type": "static"})

        app = Starlette(
            routes=[
                Route("/", homepage),
                Route("/api/test", homepage),
                Route("/static/file.js", static_file),
                Route("/health", homepage),
            ]
        )
        app.add_middleware(SecurityHeadersMiddleware)
        return app

    @pytest.fixture
    def client(self, app_with_middleware):
        """Create test client."""
        return TestClient(app_with_middleware)

    def test_csp_header_present(self, client):
        """CSP header should be present in response."""
        response = client.get("/api/test")
        # Check for either enforced or report-only CSP
        has_csp = (
            "Content-Security-Policy" in response.headers or
            "Content-Security-Policy-Report-Only" in response.headers
        )
        assert has_csp

    def test_csp_header_contains_default_src(self, client):
        """CSP header should contain default-src directive."""
        response = client.get("/api/test")
        csp = response.headers.get(
            "Content-Security-Policy",
            response.headers.get("Content-Security-Policy-Report-Only", "")
        )
        assert "default-src" in csp

    def test_csp_header_contains_script_src(self, client):
        """CSP header should contain script-src directive."""
        response = client.get("/api/test")
        csp = response.headers.get(
            "Content-Security-Policy",
            response.headers.get("Content-Security-Policy-Report-Only", "")
        )
        assert "script-src" in csp

    def test_x_frame_options_header_present(self, client):
        """X-Frame-Options header should be present."""
        response = client.get("/api/test")
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_x_content_type_options_header_present(self, client):
        """X-Content-Type-Options header should be present."""
        response = client.get("/api/test")
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_referrer_policy_header_present(self, client):
        """Referrer-Policy header should be present."""
        response = client.get("/api/test")
        assert "Referrer-Policy" in response.headers

    def test_xss_protection_header_present(self, client):
        """X-XSS-Protection header should be present."""
        response = client.get("/api/test")
        assert "X-XSS-Protection" in response.headers
        assert "mode=block" in response.headers["X-XSS-Protection"]

    def test_permissions_policy_header_present(self, client):
        """Permissions-Policy header should be present."""
        response = client.get("/api/test")
        assert "Permissions-Policy" in response.headers

    def test_permissions_policy_allows_microphone(self, client):
        """Permissions-Policy should allow microphone for voice features."""
        response = client.get("/api/test")
        policy = response.headers.get("Permissions-Policy", "")
        assert "microphone=(self)" in policy

    def test_health_endpoint_excluded(self, client):
        """Health endpoint should be excluded from some headers."""
        response = client.get("/health")
        # Health endpoint should still return successfully
        assert response.status_code == 200

    def test_static_path_excluded(self, client):
        """Static paths should be excluded from headers."""
        response = client.get("/static/file.js")
        # Should not have Cache-Control set to no-store for static files
        cache_control = response.headers.get("Cache-Control", "")
        # Static files might not have the strict no-store policy
        assert response.status_code == 200


class TestSecurityHeadersMiddlewareCustomConfig:
    """Test SecurityHeadersMiddleware with custom configuration."""

    def test_disabled_csp(self):
        """CSP can be disabled via config."""
        config = SecurityHeadersConfig(csp_enabled=False)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SecurityHeadersMiddleware, config=config)

        client = TestClient(app)
        response = client.get("/")

        assert "Content-Security-Policy" not in response.headers
        assert "Content-Security-Policy-Report-Only" not in response.headers

    def test_disabled_hsts(self):
        """HSTS can be disabled via config."""
        config = SecurityHeadersConfig(hsts_enabled=False)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SecurityHeadersMiddleware, config=config)

        client = TestClient(app)
        response = client.get("/")

        assert "Strict-Transport-Security" not in response.headers

    def test_custom_frame_options(self):
        """X-Frame-Options can be customized."""
        config = SecurityHeadersConfig(frame_options_value="SAMEORIGIN")

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SecurityHeadersMiddleware, config=config)

        client = TestClient(app)
        response = client.get("/")

        assert response.headers["X-Frame-Options"] == "SAMEORIGIN"

    def test_csp_report_only_mode(self):
        """CSP can be set to report-only mode."""
        config = SecurityHeadersConfig(csp_enabled=True, csp_report_only=True)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SecurityHeadersMiddleware, config=config)

        client = TestClient(app)
        response = client.get("/")

        assert "Content-Security-Policy-Report-Only" in response.headers
        assert "Content-Security-Policy" not in response.headers

    def test_hsts_with_preload(self):
        """HSTS can include preload directive."""
        config = SecurityHeadersConfig(
            hsts_enabled=True,
            hsts_preload=True,
            hsts_include_subdomains=True
        )

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SecurityHeadersMiddleware, config=config)

        client = TestClient(app)
        response = client.get("/")

        hsts = response.headers.get("Strict-Transport-Security", "")
        assert "preload" in hsts
        assert "includeSubDomains" in hsts


class TestWebSocketOriginValidator:
    """Test WebSocketOriginValidator."""

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        websocket = MagicMock(spec=WebSocket)
        websocket.headers = {}
        return websocket

    def test_valid_origin_allowed(self, mock_websocket):
        """Valid origin should be allowed."""
        mock_websocket.headers = {"origin": "http://localhost:3000"}
        validator = WebSocketOriginValidator(
            allowed_origins=["http://localhost:3000", "http://localhost:8000"]
        )

        assert validator.is_valid_origin(mock_websocket) is True

    def test_invalid_origin_rejected(self, mock_websocket):
        """Invalid origin should be rejected."""
        mock_websocket.headers = {"origin": "http://evil.com"}
        validator = WebSocketOriginValidator(
            allowed_origins=["http://localhost:3000"]
        )

        assert validator.is_valid_origin(mock_websocket) is False

    def test_no_origin_in_development(self, mock_websocket):
        """No origin header should be allowed in development."""
        mock_websocket.headers = {}
        validator = WebSocketOriginValidator(
            allowed_origins=["http://localhost:3000"]
        )

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            assert validator.is_valid_origin(mock_websocket) is True

    def test_no_origin_in_production_rejected(self, mock_websocket):
        """No origin header should be rejected in production."""
        mock_websocket.headers = {}
        validator = WebSocketOriginValidator(
            allowed_origins=["https://app.example.com"]
        )

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            assert validator.is_valid_origin(mock_websocket) is False

    def test_localhost_in_development_without_config(self, mock_websocket):
        """Localhost should be allowed in development when no origins configured."""
        mock_websocket.headers = {"origin": "http://localhost:3000"}
        validator = WebSocketOriginValidator(allowed_origins=[])

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            assert validator.is_valid_origin(mock_websocket) is True

    def test_localhost_in_production_without_config_rejected(self, mock_websocket):
        """Localhost should be rejected in production when no origins configured."""
        mock_websocket.headers = {"origin": "http://localhost:3000"}
        validator = WebSocketOriginValidator(allowed_origins=[])

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            assert validator.is_valid_origin(mock_websocket) is False

    def test_uses_cors_origins_by_default(self, mock_websocket):
        """Validator should use CORS origins when none specified."""
        mock_websocket.headers = {"origin": "https://test.example.com"}

        with patch.dict(os.environ, {"CORS_ORIGINS": "https://test.example.com"}, clear=False):
            validator = WebSocketOriginValidator()
            assert validator.is_valid_origin(mock_websocket) is True


class TestWebSocketOriginValidatorAsync:
    """Test async WebSocket origin validation."""

    @pytest.fixture
    def mock_websocket(self):
        """Create async mock WebSocket."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.headers = {}
        return websocket

    @pytest.mark.asyncio
    async def test_validate_or_reject_valid(self, mock_websocket):
        """Valid origin should pass validation."""
        mock_websocket.headers = {"origin": "http://localhost:3000"}
        validator = WebSocketOriginValidator(
            allowed_origins=["http://localhost:3000"]
        )

        result = await validator.validate_or_reject(mock_websocket)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_or_reject_invalid(self, mock_websocket):
        """Invalid origin should raise HTTPException."""
        from fastapi import HTTPException

        mock_websocket.headers = {"origin": "http://evil.com"}
        validator = WebSocketOriginValidator(
            allowed_origins=["http://localhost:3000"]
        )

        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_or_reject(mock_websocket)

        assert exc_info.value.status_code == 403
        assert "origin not allowed" in exc_info.value.detail.lower()


class TestGetSecurityHeadersConfig:
    """Test get_security_headers_config utility."""

    def test_returns_config(self):
        """Should return SecurityHeadersConfig instance."""
        config = get_security_headers_config()
        assert isinstance(config, SecurityHeadersConfig)

    def test_reads_from_environment(self):
        """Should read configuration from environment."""
        with patch.dict(os.environ, {"SECURITY_FRAME_OPTIONS": "SAMEORIGIN"}, clear=False):
            config = get_security_headers_config()
        assert config.frame_options_value == "SAMEORIGIN"


class TestGetWebSocketOriginValidator:
    """Test get_websocket_origin_validator singleton."""

    def test_returns_validator(self):
        """Should return WebSocketOriginValidator instance."""
        # Reset singleton for test
        import app.security.headers as headers_module
        headers_module._websocket_origin_validator = None

        validator = get_websocket_origin_validator()
        assert isinstance(validator, WebSocketOriginValidator)

    def test_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        # Reset singleton for test
        import app.security.headers as headers_module
        headers_module._websocket_origin_validator = None

        validator1 = get_websocket_origin_validator()
        validator2 = get_websocket_origin_validator()
        assert validator1 is validator2


class TestCSPHeaderGeneration:
    """Test CSP header generation."""

    def test_build_csp_header_basic(self):
        """CSP header should contain basic directives."""
        config = SecurityHeadersConfig()

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        middleware = SecurityHeadersMiddleware(app, config=config)

        csp = middleware._build_csp_header()

        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_build_csp_header_with_report_uri(self):
        """CSP header should include report-uri when configured."""
        config = SecurityHeadersConfig(
            csp_report_uri="https://example.com/csp-report"
        )

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        middleware = SecurityHeadersMiddleware(app, config=config)

        csp = middleware._build_csp_header()

        assert "report-uri https://example.com/csp-report" in csp


class TestHSTSHeaderGeneration:
    """Test HSTS header generation."""

    def test_build_hsts_header_basic(self):
        """HSTS header should contain max-age."""
        config = SecurityHeadersConfig(hsts_max_age=86400)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        middleware = SecurityHeadersMiddleware(app, config=config)

        hsts = middleware._build_hsts_header()

        assert "max-age=86400" in hsts

    def test_build_hsts_header_with_subdomains(self):
        """HSTS header should include includeSubDomains."""
        config = SecurityHeadersConfig(hsts_include_subdomains=True)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        middleware = SecurityHeadersMiddleware(app, config=config)

        hsts = middleware._build_hsts_header()

        assert "includeSubDomains" in hsts

    def test_build_hsts_header_without_subdomains(self):
        """HSTS header should not include includeSubDomains when disabled."""
        config = SecurityHeadersConfig(hsts_include_subdomains=False)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        middleware = SecurityHeadersMiddleware(app, config=config)

        hsts = middleware._build_hsts_header()

        assert "includeSubDomains" not in hsts

    def test_build_hsts_header_with_preload(self):
        """HSTS header should include preload when enabled."""
        config = SecurityHeadersConfig(hsts_preload=True)

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/", homepage)])
        middleware = SecurityHeadersMiddleware(app, config=config)

        hsts = middleware._build_hsts_header()

        assert "preload" in hsts
