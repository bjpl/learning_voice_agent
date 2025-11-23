"""
Unit Tests for Configuration Module
Tests settings loading, validation, and defaults
"""
import pytest
import os
from unittest.mock import patch, MagicMock


class TestSettings:
    """Test suite for Settings class"""

    @pytest.mark.unit
    def test_settings_loads_defaults(self):
        """Test that settings loads with default values"""
        from app.config import Settings

        # Create settings with minimal env
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-openai"
        }, clear=False):
            settings = Settings()

            assert settings.anthropic_api_key == "test-key"
            assert settings.openai_api_key == "test-openai"
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.claude_model == "claude-3-haiku-20240307"
            assert settings.claude_max_tokens == 150
            assert settings.claude_temperature == 0.7

    @pytest.mark.unit
    def test_settings_database_default(self):
        """Test default database URL"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key"
        }, clear=False):
            settings = Settings()
            assert "sqlite" in settings.database_url

    @pytest.mark.unit
    def test_settings_redis_defaults(self):
        """Test Redis configuration defaults"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key"
        }, clear=False):
            settings = Settings()
            assert settings.redis_url == "redis://localhost:6379"
            assert settings.redis_ttl == 1800  # 30 minutes

    @pytest.mark.unit
    def test_settings_session_defaults(self):
        """Test session management defaults"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key"
        }, clear=False):
            settings = Settings()
            assert settings.session_timeout == 180  # 3 minutes
            assert settings.max_context_exchanges == 5

    @pytest.mark.unit
    def test_settings_cors_origins_default(self):
        """Test CORS origins default"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key"
        }, clear=False):
            settings = Settings()
            assert "*" in settings.cors_origins

    @pytest.mark.unit
    def test_settings_twilio_optional(self):
        """Test that Twilio settings are optional"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key"
        }, clear=False):
            settings = Settings()
            # Should not raise even without Twilio credentials
            assert settings.twilio_account_sid is None or settings.twilio_account_sid == ""
            assert settings.twilio_auth_token is None or settings.twilio_auth_token == ""

    @pytest.mark.unit
    def test_settings_audio_defaults(self):
        """Test audio processing defaults"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key"
        }, clear=False):
            settings = Settings()
            assert settings.whisper_model == "whisper-1"
            assert settings.max_audio_duration == 60

    @pytest.mark.unit
    def test_settings_env_override(self):
        """Test that environment variables override defaults"""
        from app.config import Settings

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "custom-key",
            "PORT": "9000",
            "CLAUDE_MODEL": "claude-3-opus-20240229",
            "CLAUDE_MAX_TOKENS": "500"
        }, clear=False):
            settings = Settings()
            assert settings.anthropic_api_key == "custom-key"
            assert settings.port == 9000
            assert settings.claude_model == "claude-3-opus-20240229"
            assert settings.claude_max_tokens == 500

    @pytest.mark.unit
    def test_settings_singleton_import(self):
        """Test that settings is available as singleton"""
        from app.config import settings

        assert settings is not None
        assert hasattr(settings, 'anthropic_api_key')
        assert hasattr(settings, 'database_url')
