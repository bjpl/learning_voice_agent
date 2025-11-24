"""
CORS Configuration Tests - Plan A Security

Tests for:
- Environment-based origins
- Origin validation
- Secure defaults
"""

import pytest
import os
from unittest.mock import patch

from app.security.cors import CORSConfig, get_cors_origins


class TestCORSOrigins:
    """Test CORS origin configuration."""

    def test_development_defaults(self):
        """Development should allow localhost origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            # Clear CORS_ORIGINS to use defaults
            with patch.dict(os.environ, {"CORS_ORIGINS": ""}, clear=False):
                origins = CORSConfig.get_allowed_origins()

        assert "http://localhost:3000" in origins
        assert "http://localhost:8000" in origins

    def test_production_requires_explicit_origins(self):
        """Production should require explicit CORS_ORIGINS."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "CORS_ORIGINS": "",
        }, clear=False):
            # Clear default production origins
            original = CORSConfig.DEFAULT_PRODUCTION_ORIGINS
            CORSConfig.DEFAULT_PRODUCTION_ORIGINS = []

            origins = CORSConfig.get_allowed_origins()

            CORSConfig.DEFAULT_PRODUCTION_ORIGINS = original

        assert origins == []

    def test_explicit_origins_override_defaults(self):
        """CORS_ORIGINS env var should override defaults."""
        with patch.dict(os.environ, {
            "CORS_ORIGINS": "https://app.example.com,https://api.example.com",
        }, clear=False):
            origins = CORSConfig.get_allowed_origins()

        assert "https://app.example.com" in origins
        assert "https://api.example.com" in origins
        assert "http://localhost:3000" not in origins


class TestOriginValidation:
    """Test origin validation."""

    def test_wildcard_rejected(self):
        """Wildcard origin should be rejected."""
        assert CORSConfig._is_valid_origin("*") is False

    def test_http_origins_accepted(self):
        """HTTP origins should be accepted."""
        assert CORSConfig._is_valid_origin("http://localhost:3000") is True
        assert CORSConfig._is_valid_origin("https://example.com") is True

    def test_invalid_scheme_rejected(self):
        """Non-HTTP schemes should be rejected."""
        assert CORSConfig._is_valid_origin("ftp://example.com") is False
        assert CORSConfig._is_valid_origin("example.com") is False

    def test_localhost_rejected_in_production(self):
        """Localhost should be rejected in production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            assert CORSConfig._is_valid_origin("http://localhost:3000") is False


class TestCORSConfigOutput:
    """Test complete CORS configuration."""

    def test_credentials_enabled_for_specific_origins(self):
        """Credentials should be enabled when specific origins configured."""
        with patch.dict(os.environ, {
            "CORS_ORIGINS": "https://app.example.com",
        }, clear=False):
            config = CORSConfig.get_cors_config()

        assert config["allow_credentials"] is True

    def test_credentials_disabled_for_empty_origins(self):
        """Credentials should be disabled when no origins configured."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "CORS_ORIGINS": "",
        }, clear=False):
            original = CORSConfig.DEFAULT_PRODUCTION_ORIGINS
            CORSConfig.DEFAULT_PRODUCTION_ORIGINS = []

            config = CORSConfig.get_cors_config()

            CORSConfig.DEFAULT_PRODUCTION_ORIGINS = original

        assert config["allow_credentials"] is False

    def test_production_has_longer_preflight_cache(self):
        """Production should cache preflight longer."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "CORS_ORIGINS": "https://app.example.com",
        }, clear=False):
            prod_config = CORSConfig.get_cors_config()

        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
        }, clear=False):
            dev_config = CORSConfig.get_cors_config()

        assert prod_config["max_age"] > dev_config["max_age"]

    def test_allowed_methods_include_standard_methods(self):
        """Allowed methods should include standard HTTP methods."""
        config = CORSConfig.get_cors_config()
        methods = config["allow_methods"]

        assert "GET" in methods
        assert "POST" in methods
        assert "PUT" in methods
        assert "DELETE" in methods
        assert "OPTIONS" in methods

    def test_allowed_headers_include_auth(self):
        """Allowed headers should include Authorization."""
        config = CORSConfig.get_cors_config()
        headers = config["allow_headers"]

        assert "Authorization" in headers
        assert "Content-Type" in headers


class TestGetCorsOrigins:
    """Test the utility function."""

    def test_returns_current_origins(self):
        """get_cors_origins should return configured origins."""
        with patch.dict(os.environ, {
            "CORS_ORIGINS": "https://test.example.com",
        }, clear=False):
            origins = get_cors_origins()

        assert "https://test.example.com" in origins
