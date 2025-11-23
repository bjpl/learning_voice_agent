"""
Unit Tests for Pydantic Models Module
Tests request/response model validation
"""
import pytest
from datetime import datetime
from pydantic import ValidationError


class TestConversationRequest:
    """Test suite for ConversationRequest model"""

    @pytest.mark.unit
    def test_conversation_request_with_text(self):
        """Test ConversationRequest with text input"""
        from app.models import ConversationRequest

        request = ConversationRequest(
            session_id="test-session",
            text="Hello, I'm learning Python"
        )

        assert request.session_id == "test-session"
        assert request.text == "Hello, I'm learning Python"
        assert request.audio_base64 is None

    @pytest.mark.unit
    def test_conversation_request_with_audio(self):
        """Test ConversationRequest with audio input"""
        from app.models import ConversationRequest

        request = ConversationRequest(
            session_id="test-session",
            audio_base64="base64encodedaudio"
        )

        assert request.audio_base64 == "base64encodedaudio"
        assert request.text is None

    @pytest.mark.unit
    def test_conversation_request_optional_session(self):
        """Test ConversationRequest with optional session_id"""
        from app.models import ConversationRequest

        request = ConversationRequest(text="Hello")

        assert request.session_id is None
        assert request.text == "Hello"

    @pytest.mark.unit
    def test_conversation_request_all_optional(self):
        """Test ConversationRequest with all optional fields"""
        from app.models import ConversationRequest

        request = ConversationRequest()

        assert request.session_id is None
        assert request.text is None
        assert request.audio_base64 is None


class TestConversationResponse:
    """Test suite for ConversationResponse model"""

    @pytest.mark.unit
    def test_conversation_response_required_fields(self):
        """Test ConversationResponse with required fields"""
        from app.models import ConversationResponse

        response = ConversationResponse(
            session_id="test-session",
            user_text="Hello",
            agent_text="Hi there!"
        )

        assert response.session_id == "test-session"
        assert response.user_text == "Hello"
        assert response.agent_text == "Hi there!"
        assert response.intent == "statement"  # default
        assert response.timestamp is not None

    @pytest.mark.unit
    def test_conversation_response_with_intent(self):
        """Test ConversationResponse with custom intent"""
        from app.models import ConversationResponse

        response = ConversationResponse(
            session_id="test-session",
            user_text="Why does this work?",
            agent_text="Great question!",
            intent="question"
        )

        assert response.intent == "question"

    @pytest.mark.unit
    def test_conversation_response_missing_required(self):
        """Test ConversationResponse raises error for missing fields"""
        from app.models import ConversationResponse

        with pytest.raises(ValidationError):
            ConversationResponse(session_id="test")


class TestSearchRequest:
    """Test suite for SearchRequest model"""

    @pytest.mark.unit
    def test_search_request_valid(self):
        """Test SearchRequest with valid query"""
        from app.models import SearchRequest

        request = SearchRequest(query="machine learning")

        assert request.query == "machine learning"
        assert request.limit == 20  # default

    @pytest.mark.unit
    def test_search_request_custom_limit(self):
        """Test SearchRequest with custom limit"""
        from app.models import SearchRequest

        request = SearchRequest(query="python", limit=50)

        assert request.limit == 50

    @pytest.mark.unit
    def test_search_request_empty_query(self):
        """Test SearchRequest rejects empty query"""
        from app.models import SearchRequest

        with pytest.raises(ValidationError):
            SearchRequest(query="")

    @pytest.mark.unit
    def test_search_request_limit_bounds(self):
        """Test SearchRequest enforces limit bounds"""
        from app.models import SearchRequest

        # Too small
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)

        # Too large
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=200)


class TestSearchResponse:
    """Test suite for SearchResponse model"""

    @pytest.mark.unit
    def test_search_response_with_results(self):
        """Test SearchResponse with results"""
        from app.models import SearchResponse

        response = SearchResponse(
            query="python",
            results=[
                {"id": 1, "text": "Python basics"},
                {"id": 2, "text": "Python advanced"}
            ],
            count=2
        )

        assert response.query == "python"
        assert len(response.results) == 2
        assert response.count == 2

    @pytest.mark.unit
    def test_search_response_empty_results(self):
        """Test SearchResponse with no results"""
        from app.models import SearchResponse

        response = SearchResponse(
            query="nonexistent",
            results=[],
            count=0
        )

        assert response.results == []
        assert response.count == 0


class TestTwilioVoiceRequest:
    """Test suite for TwilioVoiceRequest model"""

    @pytest.mark.unit
    def test_twilio_voice_request_required(self):
        """Test TwilioVoiceRequest with required fields"""
        from app.models import TwilioVoiceRequest

        request = TwilioVoiceRequest(
            CallSid="CA123456789",
            From="+15551234567",
            To="+15559876543",
            CallStatus="ringing"
        )

        assert request.CallSid == "CA123456789"
        assert request.From == "+15551234567"
        assert request.CallStatus == "ringing"

    @pytest.mark.unit
    def test_twilio_voice_request_with_speech(self):
        """Test TwilioVoiceRequest with speech result"""
        from app.models import TwilioVoiceRequest

        request = TwilioVoiceRequest(
            CallSid="CA123456789",
            From="+15551234567",
            To="+15559876543",
            CallStatus="in-progress",
            SpeechResult="Hello world"
        )

        assert request.SpeechResult == "Hello world"

    @pytest.mark.unit
    def test_twilio_voice_request_with_recording(self):
        """Test TwilioVoiceRequest with recording URL"""
        from app.models import TwilioVoiceRequest

        request = TwilioVoiceRequest(
            CallSid="CA123456789",
            From="+15551234567",
            To="+15559876543",
            CallStatus="completed",
            RecordingUrl="https://api.twilio.com/recording.mp3"
        )

        assert request.RecordingUrl is not None


class TestWebSocketMessage:
    """Test suite for WebSocketMessage model"""

    @pytest.mark.unit
    def test_websocket_message_audio(self):
        """Test WebSocketMessage with audio type"""
        from app.models import WebSocketMessage

        message = WebSocketMessage(
            type="audio",
            audio="base64audiodata"
        )

        assert message.type == "audio"
        assert message.audio == "base64audiodata"

    @pytest.mark.unit
    def test_websocket_message_text(self):
        """Test WebSocketMessage with text type"""
        from app.models import WebSocketMessage

        message = WebSocketMessage(
            type="text",
            text="Hello world"
        )

        assert message.type == "text"
        assert message.text == "Hello world"

    @pytest.mark.unit
    def test_websocket_message_end(self):
        """Test WebSocketMessage with end type"""
        from app.models import WebSocketMessage

        message = WebSocketMessage(type="end")

        assert message.type == "end"

    @pytest.mark.unit
    def test_websocket_message_ping(self):
        """Test WebSocketMessage with ping type"""
        from app.models import WebSocketMessage

        message = WebSocketMessage(type="ping")

        assert message.type == "ping"

    @pytest.mark.unit
    def test_websocket_message_invalid_type(self):
        """Test WebSocketMessage rejects invalid type"""
        from app.models import WebSocketMessage

        with pytest.raises(ValidationError):
            WebSocketMessage(type="invalid")

    @pytest.mark.unit
    def test_websocket_message_with_session(self):
        """Test WebSocketMessage with session_id"""
        from app.models import WebSocketMessage

        message = WebSocketMessage(
            type="audio",
            session_id="test-session"
        )

        assert message.session_id == "test-session"
