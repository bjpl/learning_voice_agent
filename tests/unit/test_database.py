"""
Unit Tests for Database Layer
Tests SQLite operations and FTS5 search functionality
"""
import pytest
import json
from datetime import datetime
from app.database import Database


class TestDatabase:
    """Test suite for Database class"""

    def test_initialization(self):
        """Test database initialization"""
        db = Database(":memory:")

        assert db.db_path == ":memory:"
        assert db._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, test_db):
        """Test that initialization creates tables"""
        async with test_db.get_connection() as conn:
            # Check captures table exists
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='captures'"
            )
            result = await cursor.fetchone()
            assert result is not None

            # Check FTS table exists
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='captures_fts'"
            )
            result = await cursor.fetchone()
            assert result is not None

    @pytest.mark.asyncio
    async def test_initialize_creates_indexes(self, test_db):
        """Test that initialization creates indexes"""
        async with test_db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_session_timestamp'"
            )
            result = await cursor.fetchone()
            assert result is not None

    @pytest.mark.asyncio
    async def test_initialize_creates_triggers(self, test_db):
        """Test that initialization creates FTS triggers"""
        async with test_db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'captures_a%'"
            )
            results = await cursor.fetchall()

            # Should have 3 triggers: insert, update, delete
            assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, test_db):
        """Test that initialize can be called multiple times"""
        await test_db.initialize()
        await test_db.initialize()

        assert test_db._initialized is True

    @pytest.mark.asyncio
    async def test_save_exchange_basic(self, test_db):
        """Test saving a basic exchange"""
        row_id = await test_db.save_exchange(
            session_id="test-session",
            user_text="Hello",
            agent_text="Hi there!"
        )

        assert row_id > 0

    @pytest.mark.asyncio
    async def test_save_exchange_with_metadata(self, test_db):
        """Test saving exchange with metadata"""
        metadata = {
            "source": "api",
            "confidence": 0.95
        }

        row_id = await test_db.save_exchange(
            session_id="test-session",
            user_text="Hello",
            agent_text="Hi!",
            metadata=metadata
        )

        assert row_id > 0

        # Verify metadata was saved
        async with test_db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT metadata FROM captures WHERE id = ?",
                (row_id,)
            )
            result = await cursor.fetchone()
            saved_metadata = json.loads(result[0])
            assert saved_metadata["source"] == "api"
            assert saved_metadata["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_save_exchange_updates_fts(self, test_db):
        """Test that saving updates FTS index"""
        await test_db.save_exchange(
            session_id="test-session",
            user_text="Python programming",
            agent_text="Great topic!"
        )

        # Search should find it
        results = await test_db.search_captures("Python")

        assert len(results) > 0
        assert "Python" in results[0]["user_text"]

    @pytest.mark.asyncio
    async def test_get_session_history_empty(self, test_db):
        """Test getting history for non-existent session"""
        history = await test_db.get_session_history("non-existent")

        assert history == []

    @pytest.mark.asyncio
    async def test_get_session_history_basic(self, db_with_data):
        """Test getting session history"""
        history = await db_with_data.get_session_history("session1")

        assert len(history) == 2
        assert history[0]["user_text"] == "Tell me about Python"
        assert history[1]["user_text"] == "Its syntax"

    @pytest.mark.asyncio
    async def test_get_session_history_limit(self, db_with_data):
        """Test history respects limit"""
        # Add more exchanges
        for i in range(10):
            await db_with_data.save_exchange(
                "test-session",
                f"Message {i}",
                f"Response {i}"
            )

        history = await db_with_data.get_session_history("test-session", limit=3)

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_session_history_order(self, test_db):
        """Test history is in correct chronological order"""
        # Add exchanges
        await test_db.save_exchange("session1", "First", "Response 1")
        await test_db.save_exchange("session1", "Second", "Response 2")
        await test_db.save_exchange("session1", "Third", "Response 3")

        history = await test_db.get_session_history("session1")

        # Should be in order (oldest first due to reversed())
        assert history[0]["user_text"] == "First"
        assert history[1]["user_text"] == "Second"
        assert history[2]["user_text"] == "Third"

    @pytest.mark.asyncio
    async def test_search_captures_basic(self, db_with_data):
        """Test basic FTS5 search"""
        results = await db_with_data.search_captures("Python")

        assert len(results) >= 1
        # Should find the exchange about Python
        assert any("Python" in r["user_text"] for r in results)

    @pytest.mark.asyncio
    async def test_search_captures_multiple_results(self, db_with_data):
        """Test search with multiple results"""
        # Add more Python-related exchanges
        await db_with_data.save_exchange("session3", "Python is great", "Indeed!")
        await db_with_data.save_exchange("session4", "Learning Python", "Wonderful!")

        results = await db_with_data.search_captures("Python")

        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_search_captures_limit(self, test_db):
        """Test search respects limit"""
        # Add many results
        for i in range(30):
            await test_db.save_exchange(
                f"session{i}",
                f"Python topic {i}",
                f"Response {i}"
            )

        results = await test_db.search_captures("Python", limit=10)

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_search_captures_no_results(self, db_with_data):
        """Test search with no matching results"""
        results = await db_with_data.search_captures("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_captures_snippets(self, db_with_data):
        """Test search returns snippets with highlighting"""
        results = await db_with_data.search_captures("Python")

        if results:
            # Should have snippet fields
            assert "user_snippet" in results[0] or "agent_snippet" in results[0]

    @pytest.mark.asyncio
    async def test_search_captures_ranking(self, test_db):
        """Test search results are ranked"""
        # Add exchanges with varying relevance
        await test_db.save_exchange("s1", "Python Python Python", "Very relevant")
        await test_db.save_exchange("s2", "Python", "Less relevant")

        results = await test_db.search_captures("Python")

        # More matches should rank higher (first result)
        assert len(results) >= 2
        assert "Python Python Python" in results[0]["user_text"]

    @pytest.mark.asyncio
    async def test_search_captures_multi_word(self, test_db):
        """Test search with multiple words"""
        await test_db.save_exchange("s1", "Python programming language", "Great choice!")
        await test_db.save_exchange("s2", "Learning Python syntax", "Good topic!")

        results = await test_db.search_captures("Python programming")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, test_db):
        """Test stats for empty database"""
        stats = await test_db.get_stats()

        assert stats["total_captures"] == 0
        assert stats["unique_sessions"] == 0
        assert stats["last_capture"] is None

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, db_with_data):
        """Test stats with data"""
        stats = await db_with_data.get_stats()

        assert stats["total_captures"] == 3
        assert stats["unique_sessions"] == 2  # session1 and session2
        assert stats["last_capture"] is not None

    @pytest.mark.asyncio
    async def test_get_stats_counts_correctly(self, test_db):
        """Test stats counts are accurate"""
        # Add known data
        await test_db.save_exchange("session1", "Test 1", "Response 1")
        await test_db.save_exchange("session1", "Test 2", "Response 2")
        await test_db.save_exchange("session2", "Test 3", "Response 3")

        stats = await test_db.get_stats()

        assert stats["total_captures"] == 3
        assert stats["unique_sessions"] == 2

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, test_db):
        """Test database connection context manager"""
        async with test_db.get_connection() as conn:
            assert conn is not None
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1

    @pytest.mark.asyncio
    async def test_row_factory_returns_dict_like(self, test_db):
        """Test that row factory returns dict-like objects"""
        await test_db.save_exchange("s1", "Test", "Response")

        async with test_db.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM captures LIMIT 1")
            row = await cursor.fetchone()

            # Should be able to access by column name
            assert "session_id" in dict(row)
            assert "user_text" in dict(row)
            assert "agent_text" in dict(row)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, test_db):
        """Test multiple concurrent writes"""
        import asyncio

        async def save_exchange(i):
            return await test_db.save_exchange(
                f"session{i}",
                f"User {i}",
                f"Agent {i}"
            )

        # Save 10 exchanges concurrently
        results = await asyncio.gather(*[save_exchange(i) for i in range(10)])

        assert len(results) == 10
        assert all(r > 0 for r in results)

        # Verify all saved
        stats = await test_db.get_stats()
        assert stats["total_captures"] == 10
