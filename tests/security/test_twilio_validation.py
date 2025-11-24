"""
Twilio Validation Tests - Plan A Security

Tests for:
- Fail-closed behavior when validator not configured
- Proper signature validation
- Environment-based behavior
"""

import pytest
import os
from unittest.mock import MagicMock, patch

# Check if twilio is available
try:
    from app.twilio_handler import TwilioHandler
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioHandler = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not TWILIO_AVAILABLE,
    reason="Twilio package not installed"
)


class TestTwilioValidationFailClosed:
    """Test that Twilio validation fails closed."""

    def test_rejects_when_no_validator_in_production(self):
        """Should reject requests when validator not configured in production."""
        handler = TwilioHandler()
        handler.validator = None

        mock_request = MagicMock()
        mock_request.headers = {"X-Twilio-Signature": "test-sig"}
        mock_request.url = "https://example.com/twilio/voice"

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            result = handler.validate_request(mock_request, "body")

        assert result is False

    def test_rejects_by_default_in_development(self):
        """Should reject by default even in development."""
        handler = TwilioHandler()
        handler.validator = None

        mock_request = MagicMock()

        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "TWILIO_ALLOW_UNVALIDATED": "false",
        }, clear=False):
            result = handler.validate_request(mock_request, "body")

        assert result is False

    def test_allows_with_explicit_opt_in_in_development(self):
        """Should allow with explicit opt-in in development."""
        handler = TwilioHandler()
        handler.validator = None

        mock_request = MagicMock()

        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "TWILIO_ALLOW_UNVALIDATED": "true",
        }, clear=False):
            result = handler.validate_request(mock_request, "body")

        assert result is True

    def test_never_allows_unvalidated_in_production(self):
        """Should never allow unvalidated requests in production."""
        handler = TwilioHandler()
        handler.validator = None

        mock_request = MagicMock()

        # Even with TWILIO_ALLOW_UNVALIDATED=true, production should reject
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "TWILIO_ALLOW_UNVALIDATED": "true",
        }, clear=False):
            result = handler.validate_request(mock_request, "body")

        # Production check happens before the opt-in check
        assert result is False


class TestTwilioSignatureValidation:
    """Test Twilio signature validation."""

    def test_valid_signature_accepted(self):
        """Valid Twilio signature should be accepted."""
        handler = TwilioHandler()

        # Mock a valid validator
        handler.validator = MagicMock()
        handler.validator.validate.return_value = True

        mock_request = MagicMock()
        mock_request.headers = {"X-Twilio-Signature": "valid-signature"}
        mock_request.url = "https://example.com/twilio/voice"

        result = handler.validate_request(mock_request, "body")

        assert result is True
        handler.validator.validate.assert_called_once()

    def test_invalid_signature_rejected(self):
        """Invalid Twilio signature should be rejected."""
        handler = TwilioHandler()

        # Mock an invalid validator response
        handler.validator = MagicMock()
        handler.validator.validate.return_value = False

        mock_request = MagicMock()
        mock_request.headers = {"X-Twilio-Signature": "invalid-signature"}
        mock_request.url = "https://example.com/twilio/voice"

        result = handler.validate_request(mock_request, "body")

        assert result is False

    def test_missing_signature_rejected(self):
        """Missing signature should be rejected."""
        handler = TwilioHandler()

        # Mock validator
        handler.validator = MagicMock()
        handler.validator.validate.return_value = False

        mock_request = MagicMock()
        mock_request.headers = {}  # No signature
        mock_request.url = "https://example.com/twilio/voice"

        result = handler.validate_request(mock_request, "body")

        assert result is False


class TestTwilioHandlerInitialization:
    """Test TwilioHandler initialization."""

    def test_initializes_validator_with_token(self):
        """Should initialize validator when token is set."""
        with patch.dict(os.environ, {
            "TWILIO_AUTH_TOKEN": "test-auth-token",
        }, clear=False):
            with patch('app.twilio_handler.settings') as mock_settings:
                mock_settings.twilio_auth_token = "test-auth-token"

                handler = TwilioHandler()

        assert handler.validator is not None

    def test_no_validator_without_token(self):
        """Should not initialize validator when token not set."""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            handler = TwilioHandler()

        assert handler.validator is None
