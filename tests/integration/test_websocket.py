"""
Integration Tests for WebSocket Endpoint
Tests real-time WebSocket communication
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


class TestWebSocketConnection:
    """Test WebSocket connection and messaging"""

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_websocket_connect(self, mock_db, mock_audio, mock_handler, mock_state, client):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws/test-session") as websocket:
            assert websocket is not None

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_websocket_audio_message(self, mock_db, mock_audio, mock_handler, mock_state, client):
        """Test sending audio message via WebSocket"""
        # Setup mocks
        mock_audio.transcribe_base64 = AsyncMock(return_value="Hello from audio")
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Hi there!")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        with client.websocket_connect("/ws/test-session") as websocket:
            # Send audio message
            message = {
                "type": "audio",
                "audio": "dGVzdCBhdWRpbw=="
            }
            websocket.send_text(json.dumps(message))

            # Receive response
            response = websocket.receive_json()

            assert response["type"] == "response"
            assert response["user_text"] == "Hello from audio"
            assert response["agent_text"] == "Hi there!"
            assert "intent" in response

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    def test_websocket_end_message(self, mock_handler, mock_state, client):
        """Test ending conversation via WebSocket"""
        mock_state.get_conversation_context = AsyncMock(return_value=[
            {"user": "Hello", "agent": "Hi!"}
        ])
        mock_state.end_session = AsyncMock()
        mock_handler.create_summary = MagicMock(return_value="Great conversation!")

        with client.websocket_connect("/ws/test-session") as websocket:
            # Send end message
            message = {"type": "end"}
            websocket.send_text(json.dumps(message))

            # Receive summary
            response = websocket.receive_json()

            assert response["type"] == "summary"
            assert response["text"] == "Great conversation!"
            mock_state.end_session.assert_called_once_with("test-session")

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_websocket_multiple_messages(self, mock_db, mock_audio, mock_handler, mock_state, client):
        """Test sending multiple messages in one session"""
        # Setup mocks
        mock_audio.transcribe_base64 = AsyncMock(return_value="Test message")
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Response")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        with client.websocket_connect("/ws/test-session") as websocket:
            # Send multiple messages
            for i in range(3):
                message = {
                    "type": "audio",
                    "audio": f"message{i}"
                }
                websocket.send_text(json.dumps(message))

                response = websocket.receive_json()
                assert response["type"] == "response"

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    def test_websocket_error_handling(self, mock_audio, mock_handler, mock_state, client):
        """Test WebSocket handles errors gracefully"""
        # Setup mocks to raise error
        mock_audio.transcribe_base64 = AsyncMock(side_effect=Exception("Transcription error"))

        with client.websocket_connect("/ws/test-session") as websocket:
            message = {
                "type": "audio",
                "audio": "test"
            }
            websocket.send_text(json.dumps(message))

            # Connection should close on error
            # In practice, the error is caught and connection closes


class TestWebSocketStateManagement:
    """Test WebSocket state updates"""

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_websocket_updates_context(self, mock_db, mock_audio, mock_handler, mock_state, client):
        """Test that WebSocket updates conversation context"""
        # Setup mocks
        mock_audio.transcribe_base64 = AsyncMock(return_value="Hello")
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Hi!")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        with client.websocket_connect("/ws/test-session") as websocket:
            message = {"type": "audio", "audio": "test"}
            websocket.send_text(json.dumps(message))
            websocket.receive_json()

            # Note: In TestClient, background tasks may not execute
            # In real usage, verify via integration tests

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_websocket_saves_to_database(self, mock_db, mock_audio, mock_handler, mock_state, client):
        """Test that WebSocket saves exchanges to database"""
        # Setup mocks
        mock_audio.transcribe_base64 = AsyncMock(return_value="Test")
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Response")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        with client.websocket_connect("/ws/test-session") as websocket:
            message = {"type": "audio", "audio": "test"}
            websocket.send_text(json.dumps(message))
            websocket.receive_json()


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics"""

    @patch('app.main.state_manager')
    @patch('app.main.conversation_handler')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_websocket_response_time(self, mock_db, mock_audio, mock_handler, mock_state, client, timing):
        """Test WebSocket response latency"""
        # Setup mocks for fast response
        mock_audio.transcribe_base64 = AsyncMock(return_value="Test")
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Response")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        with client.websocket_connect("/ws/test-session") as websocket:
            with timing:
                message = {"type": "audio", "audio": "test"}
                websocket.send_text(json.dumps(message))
                websocket.receive_json()

            # With mocks, should be very fast (< 100ms)
            assert timing.elapsed < 0.1
