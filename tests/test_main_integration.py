"""
Integration Tests for FastAPI Main Application
Tests API endpoints, middleware, and application lifecycle
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
import json


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for integration tests"""
    with patch('app.database.db.initialize', new_callable=AsyncMock) as mock_db_init, \
         patch('app.state_manager.state_manager.initialize', new_callable=AsyncMock) as mock_state_init, \
         patch('app.state_manager.state_manager.close', new_callable=AsyncMock) as mock_state_close, \
         patch('app.state_manager.state_manager.get_conversation_context', new_callable=AsyncMock) as mock_get_ctx, \
         patch('app.state_manager.state_manager.update_conversation_context', new_callable=AsyncMock) as mock_update_ctx, \
         patch('app.state_manager.state_manager.get_session_metadata', new_callable=AsyncMock) as mock_get_meta, \
         patch('app.state_manager.state_manager.update_session_metadata', new_callable=AsyncMock) as mock_update_meta, \
         patch('app.state_manager.state_manager.get_active_sessions', new_callable=AsyncMock) as mock_active, \
         patch('app.database.db.save_exchange', new_callable=AsyncMock) as mock_save, \
         patch('app.database.db.search_captures', new_callable=AsyncMock) as mock_search, \
         patch('app.database.db.get_stats', new_callable=AsyncMock) as mock_stats, \
         patch('app.database.db.get_session_history', new_callable=AsyncMock) as mock_history, \
         patch('app.conversation_handler.conversation_handler.generate_response', new_callable=AsyncMock) as mock_gen, \
         patch('app.conversation_handler.conversation_handler.detect_intent') as mock_intent, \
         patch('app.audio_pipeline.audio_pipeline.transcribe_base64', new_callable=AsyncMock) as mock_transcribe:

        # Configure mock returns
        mock_get_ctx.return_value = []
        mock_get_meta.return_value = None
        mock_active.return_value = ["session-1", "session-2"]
        mock_save.return_value = 1
        mock_search.return_value = []
        mock_stats.return_value = {"total_captures": 10, "unique_sessions": 5, "last_capture": "2024-01-01"}
        mock_history.return_value = []
        mock_gen.return_value = "Test response from Claude"
        mock_intent.return_value = "statement"
        mock_transcribe.return_value = "Transcribed text"

        yield {
            "db_init": mock_db_init,
            "state_init": mock_state_init,
            "get_ctx": mock_get_ctx,
            "gen_response": mock_gen,
            "detect_intent": mock_intent,
            "transcribe": mock_transcribe,
            "search": mock_search,
            "stats": mock_stats,
            "history": mock_history,
            "active_sessions": mock_active
        }


class TestRootEndpoint:
    """Test suite for root endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_root_returns_health_info(self, mock_dependencies):
        """Test that root endpoint returns health and API info"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Learning Voice Agent"
        assert "endpoints" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_root_lists_endpoints(self, mock_dependencies):
        """Test that root lists available endpoints"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        data = response.json()
        endpoints = data["endpoints"]
        assert "websocket" in endpoints
        assert "twilio" in endpoints
        assert "search" in endpoints


class TestConversationEndpoint:
    """Test suite for conversation API endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_with_text(self, mock_dependencies):
        """Test conversation endpoint with text input"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                json={
                    "session_id": "test-session",
                    "text": "Hello, I'm learning Python"
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["user_text"] == "Hello, I'm learning Python"
        assert data["agent_text"] == "Test response from Claude"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_generates_session_id(self, mock_dependencies):
        """Test that conversation generates session ID if not provided"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                json={"text": "Hello"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_with_audio(self, mock_dependencies):
        """Test conversation endpoint with audio input"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                json={
                    "session_id": "test-session",
                    "audio_base64": "base64encodedaudio"
                }
            )

        assert response.status_code == 200
        mock_dependencies["transcribe"].assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_no_input_error(self, mock_dependencies):
        """Test conversation returns error when no input"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                json={"session_id": "test-session"}
            )

        assert response.status_code == 400

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_includes_intent(self, mock_dependencies):
        """Test that conversation response includes intent"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                json={"text": "Hello"}
            )

        data = response.json()
        assert "intent" in data
        assert data["intent"] == "statement"


class TestSearchEndpoint:
    """Test suite for search API endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_valid_query(self, mock_dependencies):
        """Test search endpoint with valid query"""
        mock_dependencies["search"].return_value = [
            {"id": 1, "user_text": "Python learning", "agent_text": "Great topic!"}
        ]

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/search",
                json={"query": "Python"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Python"
        assert data["count"] == 1
        assert len(data["results"]) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_custom_limit(self, mock_dependencies):
        """Test search endpoint with custom limit"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/search",
                json={"query": "test", "limit": 5}
            )

        assert response.status_code == 200
        mock_dependencies["search"].assert_called_with("test", 5)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_dependencies):
        """Test search endpoint with no results"""
        mock_dependencies["search"].return_value = []

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/search",
                json={"query": "nonexistent"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []


class TestStatsEndpoint:
    """Test suite for stats API endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stats_returns_data(self, mock_dependencies):
        """Test stats endpoint returns database and session info"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert "sessions" in data
        assert data["database"]["total_captures"] == 10

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stats_includes_active_sessions(self, mock_dependencies):
        """Test stats includes active session count"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/stats")

        data = response.json()
        assert data["sessions"]["active"] == 2
        assert len(data["sessions"]["ids"]) == 2


class TestSessionHistoryEndpoint:
    """Test suite for session history endpoint"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_session_history(self, mock_dependencies):
        """Test getting session history"""
        mock_dependencies["history"].return_value = [
            {"id": 1, "user_text": "Hello", "agent_text": "Hi!"},
            {"id": 2, "user_text": "How are you?", "agent_text": "I'm good!"}
        ]

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/session/test-session/history")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["count"] == 2
        assert len(data["history"]) == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_session_history_with_limit(self, mock_dependencies):
        """Test getting session history with limit parameter"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/session/test-session/history?limit=5")

        assert response.status_code == 200
        mock_dependencies["history"].assert_called_with("test-session", 5)


class TestCORSMiddleware:
    """Test suite for CORS middleware"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, mock_dependencies):
        """Test that CORS headers are present"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.options(
                "/api/conversation",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST"
                }
            )

        # CORS preflight should be handled
        assert response.status_code in [200, 204]


class TestErrorHandling:
    """Test suite for error handling"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_error_handling(self, mock_dependencies):
        """Test that conversation errors are handled gracefully"""
        mock_dependencies["gen_response"].side_effect = Exception("Test error")

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                json={"text": "Hello"}
            )

        assert response.status_code == 500

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, mock_dependencies):
        """Test handling of invalid JSON"""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/conversation",
                content="invalid json",
                headers={"Content-Type": "application/json"}
            )

        assert response.status_code == 422  # Validation error
