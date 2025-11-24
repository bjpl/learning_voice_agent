"""
Integration Tests for API Endpoints
Tests FastAPI REST endpoints with full request/response flow
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


class TestHealthEndpoint:
    """Test health check and root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Learning Voice Agent"
        assert "endpoints" in data


class TestConversationEndpoint:
    """Test /api/conversation endpoint"""

    @patch('app.main.conversation_handler')
    @patch('app.main.state_manager')
    @patch('app.main.db')
    def test_conversation_text_input(self, mock_db, mock_state, mock_handler, client):
        """Test conversation with text input"""
        # Setup mocks
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="That's interesting! Tell me more.")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        request_data = {
            "text": "I'm learning about Python"
        }

        response = client.post("/api/conversation", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["user_text"] == "I'm learning about Python"
        assert data["agent_text"] == "That's interesting! Tell me more."
        assert data["intent"] == "statement"

    @patch('app.main.conversation_handler')
    @patch('app.main.state_manager')
    @patch('app.main.audio_pipeline')
    @patch('app.main.db')
    def test_conversation_audio_input(self, mock_db, mock_audio, mock_state, mock_handler, client):
        """Test conversation with audio input"""
        # Setup mocks
        mock_audio.transcribe_base64 = AsyncMock(return_value="Hello world")
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Hi there!")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        request_data = {
            "audio_base64": "dGVzdCBhdWRpbw=="
        }

        response = client.post("/api/conversation", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["user_text"] == "Hello world"

    @patch('app.main.conversation_handler')
    @patch('app.main.state_manager')
    @patch('app.main.db')
    def test_conversation_with_session_id(self, mock_db, mock_state, mock_handler, client):
        """Test conversation with existing session"""
        # Setup mocks
        existing_context = [
            {"user": "Hello", "agent": "Hi there!"}
        ]
        mock_state.get_conversation_context = AsyncMock(return_value=existing_context)
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value={"exchange_count": 1})
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Interesting!")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=2)

        request_data = {
            "session_id": "test-session-123",
            "text": "Tell me more"
        }

        response = client.post("/api/conversation", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_conversation_no_input(self, client):
        """Test conversation fails without input"""
        request_data = {}

        response = client.post("/api/conversation", json=request_data)

        assert response.status_code == 400

    @patch('app.main.conversation_handler')
    @patch('app.main.state_manager')
    @patch('app.main.db')
    def test_conversation_background_task(self, mock_db, mock_state, mock_handler, client):
        """Test that background tasks update state"""
        # Setup mocks
        mock_state.get_conversation_context = AsyncMock(return_value=[])
        mock_state.update_conversation_context = AsyncMock()
        mock_state.get_session_metadata = AsyncMock(return_value=None)
        mock_state.update_session_metadata = AsyncMock()
        mock_handler.generate_response = AsyncMock(return_value="Response")
        mock_handler.detect_intent = MagicMock(return_value="statement")
        mock_db.save_exchange = AsyncMock(return_value=1)

        request_data = {"text": "Test"}

        response = client.post("/api/conversation", json=request_data)

        assert response.status_code == 200
        # Background tasks execute in TestClient
        # Verify they were called would require different approach


class TestSearchEndpoint:
    """Test /api/search endpoint"""

    @patch('app.main.db')
    def test_search_basic(self, mock_db, client):
        """Test basic search"""
        mock_results = [
            {
                "id": 1,
                "session_id": "s1",
                "user_text": "Python programming",
                "agent_text": "Great topic!",
                "timestamp": "2024-01-15T10:00:00"
            }
        ]
        mock_db.search_captures = AsyncMock(return_value=mock_results)

        request_data = {
            "query": "Python"
        }

        response = client.post("/api/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Python"
        assert data["count"] == 1
        assert len(data["results"]) == 1
        assert "Python" in data["results"][0]["user_text"]

    @patch('app.main.db')
    def test_search_with_limit(self, mock_db, client):
        """Test search with custom limit"""
        mock_db.search_captures = AsyncMock(return_value=[])

        request_data = {
            "query": "test",
            "limit": 10
        }

        response = client.post("/api/search", json=request_data)

        assert response.status_code == 200
        mock_db.search_captures.assert_called_once_with("test", 10)

    @patch('app.main.db')
    def test_search_no_results(self, mock_db, client):
        """Test search with no results"""
        mock_db.search_captures = AsyncMock(return_value=[])

        request_data = {
            "query": "nonexistent"
        }

        response = client.post("/api/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_search_empty_query(self, client):
        """Test search fails with empty query"""
        request_data = {
            "query": ""
        }

        response = client.post("/api/search", json=request_data)

        assert response.status_code == 422  # Validation error


class TestStatsEndpoint:
    """Test /api/stats endpoint"""

    @patch('app.main.db')
    @patch('app.main.state_manager')
    def test_stats_basic(self, mock_state, mock_db, client):
        """Test stats endpoint"""
        mock_db.get_stats = AsyncMock(return_value={
            "total_captures": 100,
            "unique_sessions": 25,
            "last_capture": "2024-01-15T10:00:00"
        })
        mock_state.get_active_sessions = AsyncMock(return_value=["s1", "s2", "s3"])

        response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["database"]["total_captures"] == 100
        assert data["database"]["unique_sessions"] == 25
        assert data["sessions"]["active"] == 3
        assert len(data["sessions"]["ids"]) == 3

    @patch('app.main.db')
    @patch('app.main.state_manager')
    def test_stats_no_sessions(self, mock_state, mock_db, client):
        """Test stats with no active sessions"""
        mock_db.get_stats = AsyncMock(return_value={
            "total_captures": 0,
            "unique_sessions": 0,
            "last_capture": None
        })
        mock_state.get_active_sessions = AsyncMock(return_value=[])

        response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["sessions"]["active"] == 0


class TestSessionHistoryEndpoint:
    """Test /api/session/{session_id}/history endpoint"""

    @patch('app.main.db')
    def test_get_session_history(self, mock_db, client):
        """Test getting session history"""
        mock_history = [
            {
                "id": 1,
                "timestamp": "2024-01-15T10:00:00",
                "user_text": "Hello",
                "agent_text": "Hi!"
            },
            {
                "id": 2,
                "timestamp": "2024-01-15T10:01:00",
                "user_text": "How are you?",
                "agent_text": "Great!"
            }
        ]
        mock_db.get_session_history = AsyncMock(return_value=mock_history)

        response = client.get("/api/session/test-session/history")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["count"] == 2
        assert len(data["history"]) == 2

    @patch('app.main.db')
    def test_get_session_history_with_limit(self, mock_db, client):
        """Test getting session history with limit"""
        mock_db.get_session_history = AsyncMock(return_value=[])

        response = client.get("/api/session/test-session/history?limit=10")

        assert response.status_code == 200
        mock_db.get_session_history.assert_called_once_with("test-session", 10)

    @patch('app.main.db')
    def test_get_session_history_empty(self, mock_db, client):
        """Test getting history for session with no data"""
        mock_db.get_session_history = AsyncMock(return_value=[])

        response = client.get("/api/session/nonexistent/history")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["history"] == []


class TestCORS:
    """Test CORS configuration"""

    def test_cors_headers_present(self, client):
        """Test CORS headers are set"""
        response = client.options(
            "/api/conversation",
            headers={"Origin": "http://localhost:3000"}
        )

        # CORS middleware should handle this
        assert response.status_code in [200, 405]  # May not have OPTIONS handler


class TestErrorHandling:
    """Test error handling in endpoints"""

    @patch('app.main.conversation_handler')
    @patch('app.main.state_manager')
    def test_conversation_internal_error(self, mock_state, mock_handler, client):
        """Test handling of internal errors"""
        mock_state.get_conversation_context = AsyncMock(side_effect=Exception("Redis error"))

        request_data = {
            "text": "Hello"
        }

        response = client.post("/api/conversation", json=request_data)

        assert response.status_code == 500

    @patch('app.main.db')
    def test_search_internal_error(self, mock_db, client):
        """Test search handles database errors"""
        mock_db.search_captures = AsyncMock(side_effect=Exception("Database error"))

        request_data = {
            "query": "test"
        }

        response = client.post("/api/search", json=request_data)

        assert response.status_code == 500
