"""
Unit Tests for Twilio Handler Module
Tests TwiML generation, webhook handling, and voice flow

Note: These tests require the 'twilio' package to be installed.
Tests are skipped if twilio is not available.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from httpx import AsyncClient, ASGITransport
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    AsyncClient = None
    ASGITransport = None

# Check if twilio is available
try:
    import twilio
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TWILIO_AVAILABLE, reason="twilio package not installed")


class TestTwilioHandler:
    """Test suite for TwilioHandler class"""

    @pytest.mark.unit
    def test_twilio_handler_initialization(self):
        """Test TwilioHandler initialization"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            assert handler.validator is None  # No validator without auth token

    @pytest.mark.unit
    def test_twilio_handler_with_auth_token(self):
        """Test TwilioHandler with auth token"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = "test-auth-token"

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            assert handler.validator is not None

    @pytest.mark.unit
    def test_validate_request_no_validator(self):
        """Test request validation without validator (dev mode)"""
        import os
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            # Set TWILIO_ALLOW_UNVALIDATED to allow in dev mode
            with patch.dict(os.environ, {"TWILIO_ALLOW_UNVALIDATED": "true"}):
                from app.twilio_handler import TwilioHandler

                handler = TwilioHandler()
                mock_request = MagicMock()

                result = handler.validate_request(mock_request, "body")
                assert result is True  # Skip validation in dev when explicitly allowed


class TestCreateGatherResponse:
    """Test suite for TwiML Gather response generation"""

    @pytest.mark.unit
    def test_create_gather_response_basic(self):
        """Test basic gather response generation"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            response = handler.create_gather_response(
                prompt="Hello, how can I help?",
                session_id="test-session"
            )

            assert "<Response>" in response
            assert "<Gather" in response
            assert "<Say" in response
            assert "Hello, how can I help?" in response
            assert "session_id=test-session" in response

    @pytest.mark.unit
    def test_create_gather_response_custom_timeout(self):
        """Test gather response with custom timeout"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            response = handler.create_gather_response(
                prompt="Test prompt",
                session_id="test",
                timeout=10
            )

            assert 'timeout="10"' in response

    @pytest.mark.unit
    def test_create_gather_response_includes_fallback(self):
        """Test that gather response includes fallback"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            response = handler.create_gather_response(
                prompt="Test",
                session_id="test"
            )

            assert "<Redirect>" in response
            assert "didn't catch that" in response.lower()


class TestCreateSayResponse:
    """Test suite for TwiML Say response generation"""

    @pytest.mark.unit
    def test_create_say_response_basic(self):
        """Test basic say response generation"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            response = handler.create_say_response("Hello there!")

            assert "<Response>" in response
            assert "<Say" in response
            assert "Hello there!" in response

    @pytest.mark.unit
    def test_create_say_response_with_hangup(self):
        """Test say response with hangup"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            response = handler.create_say_response("Goodbye!", end_call=True)

            assert "<Hangup" in response

    @pytest.mark.unit
    def test_create_say_response_without_hangup(self):
        """Test say response without hangup"""
        with patch('app.twilio_handler.settings') as mock_settings:
            mock_settings.twilio_auth_token = None

            from app.twilio_handler import TwilioHandler

            handler = TwilioHandler()
            response = handler.create_say_response("Continue talking")

            assert "<Hangup" not in response


@pytest.fixture
def mock_twilio_dependencies():
    """Mock dependencies for Twilio endpoint tests"""
    with patch('app.twilio_handler.state_manager.get_conversation_context', new_callable=AsyncMock) as mock_ctx, \
         patch('app.twilio_handler.state_manager.update_conversation_context', new_callable=AsyncMock) as mock_update, \
         patch('app.twilio_handler.state_manager.update_session_metadata', new_callable=AsyncMock) as mock_meta, \
         patch('app.twilio_handler.conversation_handler.generate_response', new_callable=AsyncMock) as mock_gen, \
         patch('app.twilio_handler.conversation_handler.detect_intent') as mock_intent, \
         patch('app.twilio_handler.conversation_handler.create_summary') as mock_summary, \
         patch('app.twilio_handler.db.save_exchange', new_callable=AsyncMock) as mock_save, \
         patch('app.twilio_handler.twilio_handler.validate_request') as mock_validate:

        mock_ctx.return_value = []
        mock_gen.return_value = "Test response"
        mock_intent.return_value = "statement"
        mock_summary.return_value = "Test summary"
        mock_validate.return_value = True

        yield {
            "ctx": mock_ctx,
            "gen": mock_gen,
            "intent": mock_intent,
            "summary": mock_summary,
            "validate": mock_validate
        }


class TestVoiceWebhook:
    """Test suite for voice webhook endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_webhook_ringing(self, mock_twilio_dependencies):
        """Test voice webhook for ringing call"""
        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/voice",
                data={
                    "CallSid": "CA123456",
                    "From": "+15551234567",
                    "CallStatus": "ringing"
                }
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        assert "<Response>" in response.text
        assert "learning companion" in response.text.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_webhook_in_progress(self, mock_twilio_dependencies):
        """Test voice webhook for in-progress call"""
        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/voice",
                data={
                    "CallSid": "CA123456",
                    "From": "+15551234567",
                    "CallStatus": "in-progress"
                }
            )

        assert response.status_code == 200
        assert "<Gather" in response.text


class TestProcessSpeech:
    """Test suite for speech processing endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_speech_success(self, mock_twilio_dependencies):
        """Test successful speech processing"""
        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/process-speech?session_id=test-session",
                data={
                    "SpeechResult": "I am learning about AI",
                    "Confidence": "0.95"
                }
            )

        assert response.status_code == 200
        assert "<Response>" in response.text
        mock_twilio_dependencies["gen"].assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_speech_no_speech(self, mock_twilio_dependencies):
        """Test speech processing with no speech result"""
        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/process-speech?session_id=test-session",
                data={
                    "SpeechResult": "",
                    "Confidence": "0"
                }
            )

        assert response.status_code == 200
        assert "didn't hear" in response.text.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_speech_low_confidence(self, mock_twilio_dependencies):
        """Test speech processing with low confidence"""
        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/process-speech?session_id=test-session",
                data={
                    "SpeechResult": "some text",
                    "Confidence": "0.3"  # Below 0.5 threshold
                }
            )

        assert response.status_code == 200
        assert "not sure" in response.text.lower() or "say that again" in response.text.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_speech_end_conversation(self, mock_twilio_dependencies):
        """Test speech processing with end conversation intent"""
        mock_twilio_dependencies["intent"].return_value = "end_conversation"

        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/process-speech?session_id=test-session",
                data={
                    "SpeechResult": "goodbye",
                    "Confidence": "0.95"
                }
            )

        assert response.status_code == 200
        assert "<Hangup" in response.text
        mock_twilio_dependencies["summary"].assert_called_once()


class TestRecordingWebhook:
    """Test suite for recording webhook endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recording_webhook(self, mock_twilio_dependencies):
        """Test recording webhook"""
        from app.twilio_handler import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/twilio/recording",
                data={
                    "RecordingUrl": "https://api.twilio.com/recording.mp3",
                    "CallSid": "CA123456"
                }
            )

        assert response.status_code == 200
        assert "<Response>" in response.text


class TestHelperFunctions:
    """Test suite for Twilio helper functions"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_twilio_session(self, mock_twilio_dependencies):
        """Test update_twilio_session helper"""
        with patch('app.twilio_handler.state_manager.update_session_metadata', new_callable=AsyncMock) as mock_update:
            from app.twilio_handler import update_twilio_session

            await update_twilio_session(
                "twilio_CA123456",
                "+15551234567",
                "ringing"
            )

            mock_update.assert_called_once()
            call_args = mock_update.call_args
            metadata = call_args[0][1]
            assert metadata["source"] == "twilio"
            assert metadata["from_number"] == "+15551234567"
            assert metadata["call_status"] == "ringing"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_twilio_exchange(self, mock_twilio_dependencies):
        """Test save_twilio_exchange helper"""
        with patch('app.twilio_handler.state_manager.update_conversation_context', new_callable=AsyncMock) as mock_ctx, \
             patch('app.twilio_handler.db.save_exchange', new_callable=AsyncMock) as mock_save:

            from app.twilio_handler import save_twilio_exchange

            await save_twilio_exchange(
                "twilio_CA123456",
                "User said this",
                "Agent responded"
            )

            mock_ctx.assert_called_once()
            mock_save.assert_called_once()
            save_args = mock_save.call_args
            # metadata is passed as a keyword argument
            assert save_args.kwargs["metadata"]["source"] == "twilio"


class TestSetupTwilioRoutes:
    """Test suite for route setup"""

    @pytest.mark.unit
    def test_setup_twilio_routes(self):
        """Test that Twilio routes are set up correctly"""
        from app.twilio_handler import setup_twilio_routes
        from fastapi import FastAPI

        app = FastAPI()
        setup_twilio_routes(app)

        # Check routes are added
        route_paths = [route.path for route in app.routes]
        assert any("/twilio/voice" in path for path in route_paths)
        assert any("/twilio/process-speech" in path for path in route_paths)
        assert any("/twilio/recording" in path for path in route_paths)
