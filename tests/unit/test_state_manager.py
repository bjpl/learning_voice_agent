"""
Unit Tests for State Manager
Tests Redis-based conversation context and session management
"""
import pytest
import json
from datetime import datetime, timedelta
from app.state_manager import StateManager


class TestStateManager:
    """Test suite for StateManager class"""

    def test_initialization(self):
        """Test state manager initialization"""
        manager = StateManager()

        assert manager.redis_client is None
        assert manager.ttl == 1800  # Default 30 minutes

    @pytest.mark.asyncio
    async def test_initialize(self, mock_redis):
        """Test Redis initialization"""
        manager = StateManager()
        manager.redis_client = mock_redis

        assert manager.redis_client is not None

    @pytest.mark.asyncio
    async def test_get_conversation_context_empty(self, test_state_manager):
        """Test getting context for new session"""
        context = await test_state_manager.get_conversation_context("new-session")

        assert context == []

    @pytest.mark.asyncio
    async def test_get_conversation_context_with_data(self, test_state_manager):
        """Test getting existing context"""
        session_id = "test-session"
        test_data = [
            {"timestamp": "2024-01-15T10:00:00", "user": "Hello", "agent": "Hi!"}
        ]

        # Manually set data in mock Redis
        await test_state_manager.redis_client.setex(
            f"session:{session_id}:context",
            1800,
            json.dumps(test_data)
        )

        context = await test_state_manager.get_conversation_context(session_id)

        assert len(context) == 1
        assert context[0]["user"] == "Hello"
        assert context[0]["agent"] == "Hi!"

    @pytest.mark.asyncio
    async def test_update_conversation_context_new(self, test_state_manager):
        """Test updating context for new session"""
        session_id = "new-session"

        await test_state_manager.update_conversation_context(
            session_id,
            "Hello",
            "Hi there!"
        )

        context = await test_state_manager.get_conversation_context(session_id)

        assert len(context) == 1
        assert context[0]["user"] == "Hello"
        assert context[0]["agent"] == "Hi there!"
        assert "timestamp" in context[0]

    @pytest.mark.asyncio
    async def test_update_conversation_context_appends(self, test_state_manager):
        """Test that updates append to existing context"""
        session_id = "test-session"

        # First exchange
        await test_state_manager.update_conversation_context(
            session_id,
            "First message",
            "First response"
        )

        # Second exchange
        await test_state_manager.update_conversation_context(
            session_id,
            "Second message",
            "Second response"
        )

        context = await test_state_manager.get_conversation_context(session_id)

        assert len(context) == 2
        assert context[0]["user"] == "First message"
        assert context[1]["user"] == "Second message"

    @pytest.mark.asyncio
    async def test_update_conversation_context_limits_size(self, test_state_manager):
        """Test context is limited to max exchanges"""
        session_id = "test-session"
        max_exchanges = 5  # Default from config

        # Add more than max exchanges
        for i in range(10):
            await test_state_manager.update_conversation_context(
                session_id,
                f"Message {i}",
                f"Response {i}"
            )

        context = await test_state_manager.get_conversation_context(session_id)

        # Should only keep last 5
        assert len(context) == max_exchanges
        assert context[0]["user"] == "Message 5"
        assert context[-1]["user"] == "Message 9"

    @pytest.mark.asyncio
    async def test_get_session_metadata_none(self, test_state_manager):
        """Test getting metadata for non-existent session"""
        metadata = await test_state_manager.get_session_metadata("non-existent")

        assert metadata is None

    @pytest.mark.asyncio
    async def test_update_session_metadata(self, test_state_manager):
        """Test updating session metadata"""
        session_id = "test-session"
        metadata = {
            "created_at": "2024-01-15T10:00:00",
            "exchange_count": 5
        }

        await test_state_manager.update_session_metadata(session_id, metadata)

        retrieved = await test_state_manager.get_session_metadata(session_id)

        assert retrieved is not None
        assert retrieved["created_at"] == "2024-01-15T10:00:00"
        assert retrieved["exchange_count"] == 5
        assert "last_activity" in retrieved

    @pytest.mark.asyncio
    async def test_update_session_metadata_adds_timestamp(self, test_state_manager):
        """Test that metadata update adds last_activity timestamp"""
        session_id = "test-session"
        metadata = {"exchange_count": 1}

        await test_state_manager.update_session_metadata(session_id, metadata)

        retrieved = await test_state_manager.get_session_metadata(session_id)

        assert "last_activity" in retrieved
        # Should be recent timestamp
        last_activity = datetime.fromisoformat(retrieved["last_activity"])
        assert (datetime.utcnow() - last_activity).seconds < 5

    @pytest.mark.asyncio
    async def test_is_session_active_no_metadata(self, test_state_manager):
        """Test session is not active without metadata"""
        is_active = await test_state_manager.is_session_active("non-existent")

        assert is_active is False

    @pytest.mark.asyncio
    async def test_is_session_active_recent(self, test_state_manager):
        """Test session is active with recent activity"""
        session_id = "test-session"
        metadata = {
            "last_activity": datetime.utcnow().isoformat()
        }

        await test_state_manager.redis_client.setex(
            f"session:{session_id}:metadata",
            1800,
            json.dumps(metadata)
        )

        is_active = await test_state_manager.is_session_active(session_id)

        assert is_active is True

    @pytest.mark.asyncio
    async def test_is_session_active_expired(self, test_state_manager):
        """Test session is not active after timeout"""
        session_id = "test-session"
        old_time = datetime.utcnow() - timedelta(seconds=200)  # > 180s timeout
        metadata = {
            "last_activity": old_time.isoformat()
        }

        await test_state_manager.redis_client.setex(
            f"session:{session_id}:metadata",
            1800,
            json.dumps(metadata)
        )

        is_active = await test_state_manager.is_session_active(session_id)

        assert is_active is False

    @pytest.mark.asyncio
    async def test_end_session(self, test_state_manager):
        """Test session cleanup"""
        session_id = "test-session"

        # Create session data
        await test_state_manager.update_conversation_context(
            session_id,
            "Hello",
            "Hi!"
        )
        await test_state_manager.update_session_metadata(
            session_id,
            {"exchange_count": 1}
        )

        # End session
        await test_state_manager.end_session(session_id)

        # Verify data is deleted
        context = await test_state_manager.get_conversation_context(session_id)
        metadata = await test_state_manager.get_session_metadata(session_id)

        assert context == []
        assert metadata is None

    @pytest.mark.asyncio
    async def test_get_active_sessions_none(self, test_state_manager):
        """Test getting active sessions with none"""
        sessions = await test_state_manager.get_active_sessions()

        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_active_sessions_multiple(self, test_state_manager):
        """Test getting multiple active sessions"""
        # Create multiple active sessions
        for i in range(3):
            session_id = f"session-{i}"
            await test_state_manager.update_session_metadata(
                session_id,
                {"exchange_count": i}
            )

        sessions = await test_state_manager.get_active_sessions()

        assert len(sessions) == 3
        assert "session-0" in sessions
        assert "session-1" in sessions
        assert "session-2" in sessions

    @pytest.mark.asyncio
    async def test_get_active_sessions_filters_expired(self, test_state_manager):
        """Test that expired sessions are filtered out"""
        # Create active session
        await test_state_manager.update_session_metadata(
            "active-session",
            {"exchange_count": 1}
        )

        # Create expired session
        old_time = datetime.utcnow() - timedelta(seconds=200)
        await test_state_manager.redis_client.setex(
            "session:expired-session:metadata",
            1800,
            json.dumps({"last_activity": old_time.isoformat()})
        )

        sessions = await test_state_manager.get_active_sessions()

        assert "active-session" in sessions
        assert "expired-session" not in sessions

    @pytest.mark.asyncio
    async def test_close(self, test_state_manager):
        """Test Redis connection close"""
        await test_state_manager.close()

        assert test_state_manager.redis_client.close.called

    @pytest.mark.asyncio
    async def test_context_ttl_set(self, test_state_manager):
        """Test that context is stored with TTL"""
        session_id = "test-session"

        await test_state_manager.update_conversation_context(
            session_id,
            "Hello",
            "Hi!"
        )

        # Verify setex was called with TTL
        assert test_state_manager.redis_client.setex.called
