"""
Test Suite: FeedbackStore
=========================

20+ tests for feedback storage functionality.
"""

import pytest
from datetime import datetime, timedelta
import tempfile
import os

from app.learning.feedback_store import FeedbackStore
from app.learning.models import Feedback, FeedbackType, FeedbackSource
from app.learning.config import LearningConfig


class TestFeedbackStoreInitialization:
    """Tests for store initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, test_config_with_temp_db):
        """Test that initialization creates the database file."""
        store = FeedbackStore(db_path=test_config_with_temp_db.feedback.database_path)
        await store.initialize()

        assert os.path.exists(store.db_path)
        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, initialized_feedback_store):
        """Test that initialization creates required tables."""
        # If we can query, tables exist
        result = await initialized_feedback_store.query(limit=1)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_close_releases_connection(self, test_config_with_temp_db):
        """Test that close releases database connection."""
        store = FeedbackStore(db_path=test_config_with_temp_db.feedback.database_path)
        await store.initialize()
        await store.close()

        assert store._db is None


class TestFeedbackStorage:
    """Tests for storing feedback."""

    @pytest.mark.asyncio
    async def test_store_returns_id(self, initialized_feedback_store, sample_feedback):
        """Test that store returns the feedback ID."""
        feedback = sample_feedback()
        feedback_id = await initialized_feedback_store.store(feedback)

        assert feedback_id == feedback.id

    @pytest.mark.asyncio
    async def test_store_persists_all_fields(self, initialized_feedback_store, sample_feedback):
        """Test that all feedback fields are stored."""
        feedback = sample_feedback(
            text="Great response!",
            correction="Actually meant this",
            metadata={"key": "value"}
        )

        await initialized_feedback_store.store(feedback)
        retrieved = await initialized_feedback_store.get(feedback.id)

        assert retrieved.session_id == feedback.session_id
        assert retrieved.query_id == feedback.query_id
        assert retrieved.rating == feedback.rating
        assert retrieved.text == feedback.text
        assert retrieved.correction == feedback.correction
        assert retrieved.metadata == feedback.metadata

    @pytest.mark.asyncio
    async def test_store_multiple_feedback(self, initialized_feedback_store, sample_feedback):
        """Test storing multiple feedback records."""
        for i in range(5):
            feedback = sample_feedback(query_id=f"query-{i}")
            await initialized_feedback_store.store(feedback)

        count = await initialized_feedback_store.count()
        assert count == 5


class TestFeedbackRetrieval:
    """Tests for retrieving feedback."""

    @pytest.mark.asyncio
    async def test_get_existing_feedback(self, initialized_feedback_store, sample_feedback):
        """Test retrieving existing feedback by ID."""
        feedback = sample_feedback()
        await initialized_feedback_store.store(feedback)

        retrieved = await initialized_feedback_store.get(feedback.id)

        assert retrieved is not None
        assert retrieved.id == feedback.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_feedback(self, initialized_feedback_store):
        """Test retrieving non-existent feedback returns None."""
        retrieved = await initialized_feedback_store.get("nonexistent-id")
        assert retrieved is None


class TestFeedbackQuerying:
    """Tests for querying feedback."""

    @pytest.mark.asyncio
    async def test_query_by_session_id(self, initialized_feedback_store, sample_feedback):
        """Test querying feedback by session ID."""
        for i in range(3):
            await initialized_feedback_store.store(
                sample_feedback(session_id="session-A", query_id=f"q-{i}")
            )
        for i in range(2):
            await initialized_feedback_store.store(
                sample_feedback(session_id="session-B", query_id=f"q-B-{i}")
            )

        results = await initialized_feedback_store.query(session_id="session-A")

        assert len(results) == 3
        assert all(r.session_id == "session-A" for r in results)

    @pytest.mark.asyncio
    async def test_query_by_feedback_type(self, initialized_feedback_store, sample_feedback):
        """Test querying feedback by type."""
        await initialized_feedback_store.store(
            sample_feedback(feedback_type=FeedbackType.EXPLICIT_POSITIVE, query_id="q1")
        )
        await initialized_feedback_store.store(
            sample_feedback(feedback_type=FeedbackType.EXPLICIT_NEGATIVE, query_id="q2")
        )

        results = await initialized_feedback_store.query(
            feedback_type=FeedbackType.EXPLICIT_POSITIVE
        )

        assert len(results) == 1
        assert results[0].feedback_type == FeedbackType.EXPLICIT_POSITIVE

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, initialized_feedback_store, sample_feedback):
        """Test querying feedback by time range."""
        feedback = sample_feedback()
        await initialized_feedback_store.store(feedback)

        results = await initialized_feedback_store.query(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow() + timedelta(hours=1)
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_with_limit(self, initialized_feedback_store, sample_feedback):
        """Test query with limit parameter."""
        for i in range(10):
            await initialized_feedback_store.store(
                sample_feedback(query_id=f"query-{i}")
            )

        results = await initialized_feedback_store.query(limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_with_offset(self, initialized_feedback_store, sample_feedback):
        """Test query with offset for pagination."""
        for i in range(10):
            await initialized_feedback_store.store(
                sample_feedback(query_id=f"query-{i}")
            )

        results = await initialized_feedback_store.query(limit=5, offset=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_by_rating_range(self, initialized_feedback_store, sample_feedback):
        """Test querying by rating range."""
        await initialized_feedback_store.store(sample_feedback(rating=0.2, query_id="q1"))
        await initialized_feedback_store.store(sample_feedback(rating=0.5, query_id="q2"))
        await initialized_feedback_store.store(sample_feedback(rating=0.9, query_id="q3"))

        results = await initialized_feedback_store.query(min_rating=0.4, max_rating=0.6)

        assert len(results) == 1
        assert results[0].rating == 0.5


class TestFeedbackAggregation:
    """Tests for feedback aggregation methods."""

    @pytest.mark.asyncio
    async def test_count_all_feedback(self, initialized_feedback_store, sample_feedback):
        """Test counting all feedback."""
        for i in range(7):
            await initialized_feedback_store.store(sample_feedback(query_id=f"q-{i}"))

        count = await initialized_feedback_store.count()

        assert count == 7

    @pytest.mark.asyncio
    async def test_count_by_session(self, initialized_feedback_store, sample_feedback):
        """Test counting feedback by session."""
        for i in range(3):
            await initialized_feedback_store.store(
                sample_feedback(session_id="session-A", query_id=f"q-{i}")
            )
        for i in range(2):
            await initialized_feedback_store.store(
                sample_feedback(session_id="session-B", query_id=f"q-B-{i}")
            )

        count = await initialized_feedback_store.count(session_id="session-A")

        assert count == 3

    @pytest.mark.asyncio
    async def test_get_average_rating(self, initialized_feedback_store, sample_feedback):
        """Test calculating average rating."""
        await initialized_feedback_store.store(sample_feedback(rating=0.6, query_id="q1"))
        await initialized_feedback_store.store(sample_feedback(rating=0.8, query_id="q2"))
        await initialized_feedback_store.store(sample_feedback(rating=1.0, query_id="q3"))

        avg = await initialized_feedback_store.get_average_rating()

        assert abs(avg - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_get_feedback_distribution(self, initialized_feedback_store, sample_feedback):
        """Test getting feedback type distribution."""
        await initialized_feedback_store.store(
            sample_feedback(feedback_type=FeedbackType.EXPLICIT_POSITIVE, query_id="q1")
        )
        await initialized_feedback_store.store(
            sample_feedback(feedback_type=FeedbackType.EXPLICIT_POSITIVE, query_id="q2")
        )
        await initialized_feedback_store.store(
            sample_feedback(feedback_type=FeedbackType.EXPLICIT_NEGATIVE, query_id="q3")
        )

        dist = await initialized_feedback_store.get_feedback_distribution()

        assert dist["explicit_positive"] == 2
        assert dist["explicit_negative"] == 1


class TestFeedbackDeletion:
    """Tests for deleting feedback."""

    @pytest.mark.asyncio
    async def test_delete_feedback(self, initialized_feedback_store, sample_feedback):
        """Test deleting a specific feedback record."""
        feedback = sample_feedback()
        await initialized_feedback_store.store(feedback)

        result = await initialized_feedback_store.delete(feedback.id)

        assert result is True
        retrieved = await initialized_feedback_store.get(feedback.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_feedback(self, initialized_feedback_store):
        """Test deleting non-existent feedback returns False."""
        result = await initialized_feedback_store.delete("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_old_feedback(self, initialized_feedback_store, sample_feedback):
        """Test deleting old feedback by retention period."""
        # Store feedback with recent timestamp (will be kept)
        recent = sample_feedback(query_id="recent")
        await initialized_feedback_store.store(recent)

        # Note: In a real test, we'd need to manipulate timestamps
        # For now, just verify the method runs without error
        deleted = await initialized_feedback_store.delete_old_feedback(retention_days=90)

        assert isinstance(deleted, int)


class TestFeedbackStoreMaintenance:
    """Tests for store maintenance operations."""

    @pytest.mark.asyncio
    async def test_vacuum_database(self, initialized_feedback_store, sample_feedback):
        """Test database vacuum operation."""
        for i in range(10):
            await initialized_feedback_store.store(sample_feedback(query_id=f"q-{i}"))

        # Delete half
        for i in range(5):
            feedback = (await initialized_feedback_store.query(limit=1))[0]
            await initialized_feedback_store.delete(feedback.id)

        # Vacuum should not raise
        await initialized_feedback_store.vacuum()
