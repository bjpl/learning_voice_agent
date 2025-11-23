"""
Unit Tests for State Manager Module
Tests Redis operations, session management, and conversation context
"""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta


class TestStateManagerInitialization:
    """Test suite for state manager initialization"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_creates_redis_client(self, mock_redis_client):
        """Test that initialization creates Redis client"""
        with patch('app.state_manager.redis.from_url', new_callable=AsyncMock) as mock_from_url:
            mock_from_url.return_value = mock_redis_client

            from app.state_manager import StateManager

            manager = StateManager()
            await manager.initialize()

            assert manager.redis_client is not None
            mock_from_url.assert_called_once()

    @pytest.mark.unit
    def test_state_manager_default_ttl(self):
        """Test that state manager has default TTL from settings"""
        from app.state_manager import StateManager

        manager = StateManager()
        assert manager.ttl == 1800  # 30 minutes default


class TestGetConversationContext:
    """Test suite for getting conversation context"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_context_empty(self, mock_redis_client):
        """Test getting context when none exists"""
        mock_redis_client.get = AsyncMock(return_value=None)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        context = await manager.get_conversation_context("test-session")

        assert context == []
        mock_redis_client.get.assert_called_once_with("session:test-session:context")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_context_with_data(self, mock_redis_client, sample_context):
        """Test getting context with existing data"""
        mock_redis_client.get = AsyncMock(return_value=json.dumps(sample_context))

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        context = await manager.get_conversation_context("test-session")

        assert len(context) == 2
        assert context[0]['user'] == "I'm learning about machine learning"


class TestUpdateConversationContext:
    """Test suite for updating conversation context"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_context_new_session(self, mock_redis_client):
        """Test updating context for new session"""
        mock_redis_client.get = AsyncMock(return_value=None)
        mock_redis_client.setex = AsyncMock(return_value=True)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        await manager.update_conversation_context(
            "new-session",
            "Hello",
            "Hi there!"
        )

        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert "session:new-session:context" in call_args[0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_context_appends_to_existing(self, mock_redis_client, sample_context):
        """Test that update appends to existing context"""
        mock_redis_client.get = AsyncMock(return_value=json.dumps(sample_context))
        mock_redis_client.setex = AsyncMock(return_value=True)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        await manager.update_conversation_context(
            "existing-session",
            "New message",
            "New response"
        )

        # Verify the saved context includes new exchange
        call_args = mock_redis_client.setex.call_args
        saved_context = json.loads(call_args[0][2])
        assert len(saved_context) == 3
        assert saved_context[-1]['user'] == "New message"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_context_trims_to_max_exchanges(self, mock_redis_client):
        """Test that context is trimmed to max exchanges"""
        # Create context with 10 exchanges
        large_context = [
            {"timestamp": datetime.utcnow().isoformat(), "user": f"msg{i}", "agent": f"resp{i}"}
            for i in range(10)
        ]
        mock_redis_client.get = AsyncMock(return_value=json.dumps(large_context))
        mock_redis_client.setex = AsyncMock(return_value=True)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        await manager.update_conversation_context(
            "session",
            "New message",
            "New response"
        )

        # Verify trimming (default max is 5)
        call_args = mock_redis_client.setex.call_args
        saved_context = json.loads(call_args[0][2])
        assert len(saved_context) <= 5


class TestSessionMetadata:
    """Test suite for session metadata operations"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_metadata_none(self, mock_redis_client):
        """Test getting metadata when none exists"""
        mock_redis_client.get = AsyncMock(return_value=None)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        metadata = await manager.get_session_metadata("test-session")

        assert metadata is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_metadata_with_data(self, mock_redis_client, sample_session_metadata):
        """Test getting existing metadata"""
        mock_redis_client.get = AsyncMock(return_value=json.dumps(sample_session_metadata))

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        metadata = await manager.get_session_metadata("test-session")

        assert metadata is not None
        assert metadata['exchange_count'] == 2
        assert metadata['source'] == 'test'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_session_metadata(self, mock_redis_client):
        """Test updating session metadata"""
        mock_redis_client.setex = AsyncMock(return_value=True)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        metadata = {"created_at": datetime.utcnow().isoformat(), "exchange_count": 1}
        await manager.update_session_metadata("test-session", metadata)

        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        saved_metadata = json.loads(call_args[0][2])
        assert 'last_activity' in saved_metadata


class TestIsSessionActive:
    """Test suite for session activity check"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_is_session_active_no_metadata(self, mock_redis_client):
        """Test inactive when no metadata exists"""
        mock_redis_client.get = AsyncMock(return_value=None)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        is_active = await manager.is_session_active("test-session")

        assert is_active is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_is_session_active_recent_activity(self, mock_redis_client):
        """Test active when recent activity"""
        recent_metadata = {
            "last_activity": datetime.utcnow().isoformat()
        }
        mock_redis_client.get = AsyncMock(return_value=json.dumps(recent_metadata))

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        is_active = await manager.is_session_active("test-session")

        assert is_active is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_is_session_active_old_activity(self, mock_redis_client):
        """Test inactive when old activity"""
        old_time = datetime.utcnow() - timedelta(minutes=10)  # 10 minutes ago
        old_metadata = {
            "last_activity": old_time.isoformat()
        }
        mock_redis_client.get = AsyncMock(return_value=json.dumps(old_metadata))

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        is_active = await manager.is_session_active("test-session")

        assert is_active is False  # Default timeout is 3 minutes


class TestEndSession:
    """Test suite for ending sessions"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_end_session_deletes_keys(self, mock_redis_client):
        """Test that end session deletes all session keys"""
        mock_redis_client.delete = AsyncMock(return_value=1)

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        await manager.end_session("test-session")

        assert mock_redis_client.delete.call_count == 2  # context and metadata


class TestGetActiveSessions:
    """Test suite for getting active sessions"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_active_sessions_empty(self, mock_redis_client):
        """Test getting active sessions when none exist"""
        async def empty_iter(*args, **kwargs):
            return
            yield  # Make it an async generator

        mock_redis_client.scan_iter = empty_iter

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        sessions = await manager.get_active_sessions()

        assert sessions == []


class TestStateManagerClose:
    """Test suite for closing state manager"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_closes_redis(self, mock_redis_client):
        """Test that close closes Redis connection"""
        mock_redis_client.close = AsyncMock()

        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = mock_redis_client

        await manager.close()

        mock_redis_client.close.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_handles_no_client(self):
        """Test that close handles case when no client exists"""
        from app.state_manager import StateManager

        manager = StateManager()
        manager.redis_client = None

        # Should not raise
        await manager.close()


class TestStateManagerSingleton:
    """Test state manager singleton"""

    @pytest.mark.unit
    def test_state_manager_singleton(self):
        """Test that state_manager singleton is available"""
        from app.state_manager import state_manager

        assert state_manager is not None
        assert hasattr(state_manager, 'initialize')
        assert hasattr(state_manager, 'get_conversation_context')
        assert hasattr(state_manager, 'end_session')
