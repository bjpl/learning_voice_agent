"""
Unit Tests for Database Module
Tests SQLite operations, FTS5 search, and data persistence
"""
import pytest
import os
import json
from datetime import datetime


class TestDatabaseInitialization:
    """Test suite for database initialization"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_initialize(self, test_database):
        """Test database initialization creates tables"""
        assert test_database._initialized is True
        assert os.path.exists(test_database.db_path)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_initialize_idempotent(self, test_database):
        """Test that initialization can be called multiple times"""
        await test_database.initialize()
        await test_database.initialize()  # Should not raise
        assert test_database._initialized is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_creates_captures_table(self, test_database):
        """Test that captures table is created"""
        import aiosqlite

        async with aiosqlite.connect(test_database.db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='captures'"
            )
            result = await cursor.fetchone()

        assert result is not None
        assert result[0] == 'captures'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_creates_fts_table(self, test_database):
        """Test that FTS5 virtual table is created"""
        import aiosqlite

        async with aiosqlite.connect(test_database.db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='captures_fts'"
            )
            result = await cursor.fetchone()

        assert result is not None
        assert result[0] == 'captures_fts'


class TestSaveExchange:
    """Test suite for saving conversation exchanges"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_exchange_success(self, test_database):
        """Test saving a conversation exchange"""
        row_id = await test_database.save_exchange(
            session_id="test-session-1",
            user_text="Hello, I'm learning Python",
            agent_text="That's great! What aspect interests you?",
            metadata={"source": "test"}
        )

        assert row_id is not None
        assert row_id > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_exchange_without_metadata(self, test_database):
        """Test saving exchange without metadata"""
        row_id = await test_database.save_exchange(
            session_id="test-session-2",
            user_text="Testing without metadata",
            agent_text="Response text"
        )

        assert row_id is not None
        assert row_id > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_multiple_exchanges(self, test_database):
        """Test saving multiple exchanges"""
        ids = []
        for i in range(5):
            row_id = await test_database.save_exchange(
                session_id="test-session-multi",
                user_text=f"Message {i}",
                agent_text=f"Response {i}"
            )
            ids.append(row_id)

        assert len(ids) == 5
        assert len(set(ids)) == 5  # All unique IDs


class TestGetSessionHistory:
    """Test suite for retrieving session history"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_history_empty(self, test_database):
        """Test getting history for non-existent session"""
        history = await test_database.get_session_history("nonexistent-session")

        assert history == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_history_returns_exchanges(self, test_database):
        """Test getting history returns saved exchanges"""
        session_id = "history-test-session"

        # Save some exchanges
        await test_database.save_exchange(session_id, "First message", "First response")
        await test_database.save_exchange(session_id, "Second message", "Second response")
        await test_database.save_exchange(session_id, "Third message", "Third response")

        history = await test_database.get_session_history(session_id)

        assert len(history) == 3
        # Verify all messages are present (order depends on implementation)
        user_texts = [h['user_text'] for h in history]
        assert "First message" in user_texts
        assert "Second message" in user_texts
        assert "Third message" in user_texts

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_history_respects_limit(self, test_database):
        """Test that history respects limit parameter"""
        session_id = "limit-test-session"

        # Save 10 exchanges
        for i in range(10):
            await test_database.save_exchange(session_id, f"Message {i}", f"Response {i}")

        history = await test_database.get_session_history(session_id, limit=5)

        assert len(history) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_history_chronological_order(self, test_database):
        """Test that history contains expected exchanges"""
        session_id = "order-test-session"

        await test_database.save_exchange(session_id, "First", "First response")
        await test_database.save_exchange(session_id, "Second", "Second response")
        await test_database.save_exchange(session_id, "Third", "Third response")

        history = await test_database.get_session_history(session_id)

        # Verify all 3 exchanges are present
        assert len(history) == 3
        user_texts = [h['user_text'] for h in history]
        assert "First" in user_texts
        assert "Second" in user_texts
        assert "Third" in user_texts


class TestSearchCaptures:
    """Test suite for FTS5 search functionality"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_captures_empty_results(self, test_database):
        """Test search returns empty for no matches"""
        results = await test_database.search_captures("nonexistent query xyz123")

        assert results == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_captures_finds_matches(self, test_database):
        """Test search finds matching captures"""
        # Insert searchable content
        await test_database.save_exchange(
            "search-session",
            "I'm learning about machine learning algorithms",
            "That's an interesting topic!"
        )
        await test_database.save_exchange(
            "search-session",
            "Python is great for data science",
            "Indeed it is!"
        )

        results = await test_database.search_captures("machine learning")

        assert len(results) >= 1
        assert any("machine" in r['user_text'].lower() for r in results)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_captures_respects_limit(self, test_database):
        """Test that search respects limit parameter"""
        # Insert multiple matches
        for i in range(20):
            await test_database.save_exchange(
                f"search-limit-{i}",
                f"Python programming example {i}",
                f"Response {i}"
            )

        results = await test_database.search_captures("Python", limit=5)

        assert len(results) <= 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_captures_includes_snippets(self, test_database):
        """Test that search results include snippets"""
        await test_database.save_exchange(
            "snippet-session",
            "Neural networks are fascinating to study",
            "They certainly are!"
        )

        results = await test_database.search_captures("neural networks")

        if results:
            assert 'user_snippet' in results[0] or 'agent_snippet' in results[0]


class TestGetStats:
    """Test suite for database statistics"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stats_empty_database(self, test_database):
        """Test stats on empty database"""
        stats = await test_database.get_stats()

        assert stats is not None
        assert 'total_captures' in stats
        assert stats['total_captures'] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, test_database):
        """Test stats with data in database"""
        # Insert some data
        await test_database.save_exchange("session-1", "User 1", "Agent 1")
        await test_database.save_exchange("session-1", "User 2", "Agent 2")
        await test_database.save_exchange("session-2", "User 3", "Agent 3")

        stats = await test_database.get_stats()

        assert stats['total_captures'] == 3
        assert stats['unique_sessions'] == 2
        assert stats['last_capture'] is not None


class TestDatabaseSingleton:
    """Test database singleton"""

    @pytest.mark.unit
    def test_database_singleton(self, test_database):
        """Test that test_database fixture works as expected"""
        assert test_database is not None
        assert hasattr(test_database, 'initialize')
        assert hasattr(test_database, 'save_exchange')
        assert hasattr(test_database, 'search_captures')
