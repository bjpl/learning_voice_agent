"""
Feedback Store - SQLite Persistence Layer
PATTERN: Repository pattern with async operations
WHY: Separation of concerns and efficient data access

Tables:
- explicit_feedback: User ratings and comments
- implicit_feedback: Engagement metrics
- corrections: User corrections and rephrases

Features:
- Async SQLite operations with aiosqlite
- Efficient indexing for common queries
- JSON metadata columns for flexibility
- Aggregate query support
"""
import aiosqlite
import json
import uuid
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from contextlib import asynccontextmanager

from app.learning.config import feedback_config
from app.learning.feedback_models import (
    ExplicitFeedback,
    ImplicitFeedback,
    CorrectionFeedback,
    SessionFeedback,
    FeedbackStats,
    CorrectionType,
    FeedbackSentiment,
)
from app.logger import get_logger

# Module logger
logger = get_logger("feedback_store")


class FeedbackStore:
    """
    SQLite persistence layer for feedback data.

    PATTERN: Repository with async operations
    WHY: Clean data access with proper error handling
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the feedback store.

        Args:
            db_path: Path to SQLite database. Uses config default if not provided.
        """
        self.db_path = db_path or feedback_config.database_path
        self._initialized = False

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection with row factory."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates all required tables and indexes if they don't exist.
        """
        if self._initialized:
            return

        logger.info("feedback_store_initialization_started", db_path=self.db_path)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Explicit feedback table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS explicit_feedback (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        exchange_id TEXT NOT NULL,
                        rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                        helpful INTEGER NOT NULL,
                        comment TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Implicit feedback table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS implicit_feedback (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        response_time_ms INTEGER NOT NULL,
                        user_response_time_ms INTEGER,
                        engagement_duration_seconds INTEGER,
                        follow_up_count INTEGER DEFAULT 0,
                        scroll_depth REAL,
                        copy_action INTEGER DEFAULT 0,
                        share_action INTEGER DEFAULT 0,
                        engagement_score REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Corrections table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS corrections (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        original_text TEXT NOT NULL,
                        corrected_text TEXT NOT NULL,
                        correction_type TEXT NOT NULL,
                        edit_distance INTEGER,
                        edit_distance_ratio REAL,
                        context TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Indexes for efficient queries
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_explicit_session
                    ON explicit_feedback(session_id, timestamp DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_explicit_timestamp
                    ON explicit_feedback(timestamp DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_explicit_rating
                    ON explicit_feedback(rating)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_implicit_session
                    ON implicit_feedback(session_id, timestamp DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_implicit_timestamp
                    ON implicit_feedback(timestamp DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_corrections_session
                    ON corrections(session_id, timestamp DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_corrections_type
                    ON corrections(correction_type)
                """)

                await db.commit()

            self._initialized = True
            logger.info("feedback_store_initialization_complete", db_path=self.db_path)

        except Exception as e:
            logger.error(
                "feedback_store_initialization_failed",
                db_path=self.db_path,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    # ========================================================================
    # EXPLICIT FEEDBACK OPERATIONS
    # ========================================================================

    async def save_explicit(self, feedback: ExplicitFeedback) -> str:
        """
        Save explicit feedback to the database.

        Args:
            feedback: ExplicitFeedback model instance

        Returns:
            Generated feedback ID
        """
        feedback_id = feedback.id or str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO explicit_feedback
                    (id, session_id, exchange_id, rating, helpful, comment, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        feedback.session_id,
                        feedback.exchange_id,
                        feedback.rating,
                        1 if feedback.helpful else 0,
                        feedback.comment,
                        feedback.timestamp.isoformat(),
                        json.dumps(feedback.metadata) if feedback.metadata else None
                    )
                )
                await db.commit()

            logger.info(
                "explicit_feedback_saved",
                feedback_id=feedback_id,
                session_id=feedback.session_id,
                rating=feedback.rating,
                helpful=feedback.helpful
            )

            return feedback_id

        except Exception as e:
            logger.error(
                "explicit_feedback_save_failed",
                session_id=feedback.session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def get_explicit_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[ExplicitFeedback]:
        """
        Get explicit feedback for a session.

        Args:
            session_id: Session identifier
            limit: Maximum results to return

        Returns:
            List of ExplicitFeedback instances
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT id, session_id, exchange_id, rating, helpful,
                           comment, timestamp, metadata
                    FROM explicit_feedback
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit)
                )
                rows = await cursor.fetchall()

                return [
                    ExplicitFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        exchange_id=row['exchange_id'],
                        rating=row['rating'],
                        helpful=bool(row['helpful']),
                        comment=row['comment'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(
                "get_explicit_by_session_failed",
                session_id=session_id,
                error=str(e)
            )
            return []

    # ========================================================================
    # IMPLICIT FEEDBACK OPERATIONS
    # ========================================================================

    async def save_implicit(self, feedback: ImplicitFeedback) -> str:
        """
        Save implicit feedback to the database.

        Args:
            feedback: ImplicitFeedback model instance

        Returns:
            Generated feedback ID
        """
        feedback_id = feedback.id or str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO implicit_feedback
                    (id, session_id, response_time_ms, user_response_time_ms,
                     engagement_duration_seconds, follow_up_count, scroll_depth,
                     copy_action, share_action, engagement_score, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        feedback.session_id,
                        feedback.response_time_ms,
                        feedback.user_response_time_ms,
                        feedback.engagement_duration_seconds,
                        feedback.follow_up_count,
                        feedback.scroll_depth,
                        1 if feedback.copy_action else 0,
                        1 if feedback.share_action else 0,
                        feedback.engagement_score,
                        feedback.timestamp.isoformat(),
                        json.dumps(feedback.metadata) if feedback.metadata else None
                    )
                )
                await db.commit()

            logger.debug(
                "implicit_feedback_saved",
                feedback_id=feedback_id,
                session_id=feedback.session_id,
                engagement_score=feedback.engagement_score
            )

            return feedback_id

        except Exception as e:
            logger.error(
                "implicit_feedback_save_failed",
                session_id=feedback.session_id,
                error=str(e)
            )
            raise

    async def save_implicit_batch(
        self,
        feedback_list: List[ImplicitFeedback]
    ) -> List[str]:
        """
        Save multiple implicit feedback items in a batch.

        Args:
            feedback_list: List of ImplicitFeedback instances

        Returns:
            List of generated feedback IDs
        """
        if not feedback_list:
            return []

        feedback_ids = []

        try:
            async with self.get_connection() as db:
                for feedback in feedback_list:
                    feedback_id = feedback.id or str(uuid.uuid4())
                    feedback_ids.append(feedback_id)

                    await db.execute(
                        """
                        INSERT INTO implicit_feedback
                        (id, session_id, response_time_ms, user_response_time_ms,
                         engagement_duration_seconds, follow_up_count, scroll_depth,
                         copy_action, share_action, engagement_score, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            feedback_id,
                            feedback.session_id,
                            feedback.response_time_ms,
                            feedback.user_response_time_ms,
                            feedback.engagement_duration_seconds,
                            feedback.follow_up_count,
                            feedback.scroll_depth,
                            1 if feedback.copy_action else 0,
                            1 if feedback.share_action else 0,
                            feedback.engagement_score,
                            feedback.timestamp.isoformat(),
                            json.dumps(feedback.metadata) if feedback.metadata else None
                        )
                    )

                await db.commit()

            logger.info(
                "implicit_feedback_batch_saved",
                count=len(feedback_ids)
            )

            return feedback_ids

        except Exception as e:
            logger.error(
                "implicit_feedback_batch_save_failed",
                count=len(feedback_list),
                error=str(e)
            )
            raise

    async def get_implicit_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[ImplicitFeedback]:
        """Get implicit feedback for a session."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM implicit_feedback
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit)
                )
                rows = await cursor.fetchall()

                return [
                    ImplicitFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        response_time_ms=row['response_time_ms'],
                        user_response_time_ms=row['user_response_time_ms'],
                        engagement_duration_seconds=row['engagement_duration_seconds'],
                        follow_up_count=row['follow_up_count'],
                        scroll_depth=row['scroll_depth'],
                        copy_action=bool(row['copy_action']),
                        share_action=bool(row['share_action']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(
                "get_implicit_by_session_failed",
                session_id=session_id,
                error=str(e)
            )
            return []

    # ========================================================================
    # CORRECTION OPERATIONS
    # ========================================================================

    async def save_correction(self, correction: CorrectionFeedback) -> str:
        """
        Save a correction to the database.

        Args:
            correction: CorrectionFeedback model instance

        Returns:
            Generated correction ID
        """
        correction_id = correction.id or str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO corrections
                    (id, session_id, original_text, corrected_text, correction_type,
                     edit_distance, edit_distance_ratio, context, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        correction_id,
                        correction.session_id,
                        correction.original_text,
                        correction.corrected_text,
                        correction.correction_type.value,
                        correction.edit_distance,
                        correction.edit_distance_ratio,
                        correction.context,
                        correction.timestamp.isoformat(),
                        json.dumps(correction.metadata) if correction.metadata else None
                    )
                )
                await db.commit()

            logger.info(
                "correction_saved",
                correction_id=correction_id,
                session_id=correction.session_id,
                correction_type=correction.correction_type.value
            )

            return correction_id

        except Exception as e:
            logger.error(
                "correction_save_failed",
                session_id=correction.session_id,
                error=str(e)
            )
            raise

    async def get_corrections_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[CorrectionFeedback]:
        """Get corrections for a session."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM corrections
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit)
                )
                rows = await cursor.fetchall()

                return [
                    CorrectionFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        original_text=row['original_text'],
                        corrected_text=row['corrected_text'],
                        correction_type=CorrectionType(row['correction_type']),
                        edit_distance=row['edit_distance'],
                        edit_distance_ratio=row['edit_distance_ratio'],
                        context=row['context'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(
                "get_corrections_by_session_failed",
                session_id=session_id,
                error=str(e)
            )
            return []

    # ========================================================================
    # AGGREGATION QUERIES
    # ========================================================================

    async def get_session_feedback(self, session_id: str) -> SessionFeedback:
        """
        Get aggregated feedback for a session.

        Args:
            session_id: Session identifier

        Returns:
            SessionFeedback with aggregated metrics
        """
        try:
            async with self.get_connection() as db:
                # Get explicit feedback stats
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        AVG(rating) as avg_rating,
                        SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                        MIN(timestamp) as first_at,
                        MAX(timestamp) as last_at
                    FROM explicit_feedback
                    WHERE session_id = ?
                    """,
                    (session_id,)
                )
                explicit_stats = dict(await cursor.fetchone())

                # Get implicit feedback stats
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        AVG(response_time_ms) as avg_response_time,
                        SUM(follow_up_count) as total_follow_ups,
                        AVG(engagement_score) as avg_engagement
                    FROM implicit_feedback
                    WHERE session_id = ?
                    """,
                    (session_id,)
                )
                implicit_stats = dict(await cursor.fetchone())

                # Get correction stats
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        correction_type
                    FROM corrections
                    WHERE session_id = ?
                    GROUP BY correction_type
                    """,
                    (session_id,)
                )
                correction_rows = await cursor.fetchall()

                correction_count = sum(row['count'] for row in correction_rows)
                correction_types = {row['correction_type']: row['count'] for row in correction_rows}

                # Compute overall sentiment
                overall_sentiment = FeedbackSentiment.NEUTRAL
                if explicit_stats['avg_rating']:
                    if explicit_stats['avg_rating'] >= 4:
                        overall_sentiment = FeedbackSentiment.POSITIVE
                    elif explicit_stats['avg_rating'] <= 2:
                        overall_sentiment = FeedbackSentiment.NEGATIVE

                # Compute helpful percentage
                helpful_percentage = None
                if explicit_stats['count'] > 0:
                    helpful_percentage = (explicit_stats['helpful_count'] / explicit_stats['count']) * 100

                return SessionFeedback(
                    session_id=session_id,
                    explicit_feedback_count=explicit_stats['count'] or 0,
                    average_rating=explicit_stats['avg_rating'],
                    helpful_percentage=helpful_percentage,
                    implicit_feedback_count=implicit_stats['count'] or 0,
                    average_response_time_ms=implicit_stats['avg_response_time'],
                    total_follow_ups=implicit_stats['total_follow_ups'] or 0,
                    engagement_score=implicit_stats['avg_engagement'],
                    correction_count=correction_count,
                    correction_types=correction_types,
                    first_feedback_at=datetime.fromisoformat(explicit_stats['first_at']) if explicit_stats['first_at'] else None,
                    last_feedback_at=datetime.fromisoformat(explicit_stats['last_at']) if explicit_stats['last_at'] else None,
                    overall_sentiment=overall_sentiment
                )

        except Exception as e:
            logger.error(
                "get_session_feedback_failed",
                session_id=session_id,
                error=str(e)
            )
            # Return empty session feedback
            return SessionFeedback(session_id=session_id)

    async def get_aggregate_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> FeedbackStats:
        """
        Get aggregate feedback statistics for a time range.

        Args:
            start_time: Start of time range (defaults to 24 hours ago)
            end_time: End of time range (defaults to now)

        Returns:
            FeedbackStats with aggregate metrics
        """
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=feedback_config.default_time_range_hours))

        try:
            async with self.get_connection() as db:
                # Get explicit feedback stats
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(DISTINCT session_id) as sessions,
                        COUNT(*) as count,
                        AVG(rating) as avg_rating,
                        SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count
                    FROM explicit_feedback
                    WHERE timestamp BETWEEN ? AND ?
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                )
                explicit_stats = dict(await cursor.fetchone())

                # Get rating distribution
                cursor = await db.execute(
                    """
                    SELECT rating, COUNT(*) as count
                    FROM explicit_feedback
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY rating
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                )
                rating_rows = await cursor.fetchall()
                rating_distribution = {row['rating']: row['count'] for row in rating_rows}

                # Get implicit feedback stats
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        AVG(response_time_ms) as avg_response_time,
                        AVG(engagement_score) as avg_engagement,
                        AVG(follow_up_count) as avg_follow_ups
                    FROM implicit_feedback
                    WHERE timestamp BETWEEN ? AND ?
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                )
                implicit_stats = dict(await cursor.fetchone())

                # Get response time percentiles
                cursor = await db.execute(
                    """
                    SELECT response_time_ms
                    FROM implicit_feedback
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY response_time_ms
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                )
                response_times = [row['response_time_ms'] for row in await cursor.fetchall()]

                p50 = p95 = p99 = None
                if response_times:
                    n = len(response_times)
                    p50 = response_times[int(n * 0.5)] if n > 0 else None
                    p95 = response_times[int(n * 0.95)] if n > 0 else None
                    p99 = response_times[int(n * 0.99)] if n > 0 else None

                # Get correction stats
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        correction_type
                    FROM corrections
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY correction_type
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                )
                correction_rows = await cursor.fetchall()

                correction_count = sum(row['count'] for row in correction_rows)
                correction_type_dist = {row['correction_type']: row['count'] for row in correction_rows}

                # Compute helpful percentage
                helpful_percentage = None
                if explicit_stats['count'] and explicit_stats['count'] > 0:
                    helpful_percentage = (explicit_stats['helpful_count'] / explicit_stats['count']) * 100

                # Compute sentiment distribution
                sentiment_dist = {}
                for rating, count in rating_distribution.items():
                    if rating >= 4:
                        sentiment_dist['positive'] = sentiment_dist.get('positive', 0) + count
                    elif rating <= 2:
                        sentiment_dist['negative'] = sentiment_dist.get('negative', 0) + count
                    else:
                        sentiment_dist['neutral'] = sentiment_dist.get('neutral', 0) + count

                return FeedbackStats(
                    time_range_start=start_time,
                    time_range_end=end_time,
                    total_sessions=explicit_stats['sessions'] or 0,
                    total_explicit_feedback=explicit_stats['count'] or 0,
                    total_implicit_feedback=implicit_stats['count'] or 0,
                    total_corrections=correction_count,
                    average_rating=explicit_stats['avg_rating'],
                    rating_distribution=rating_distribution,
                    helpful_percentage=helpful_percentage,
                    average_response_time_ms=implicit_stats['avg_response_time'],
                    p50_response_time_ms=p50,
                    p95_response_time_ms=p95,
                    p99_response_time_ms=p99,
                    average_engagement_score=implicit_stats['avg_engagement'],
                    average_follow_ups_per_session=implicit_stats['avg_follow_ups'],
                    correction_type_distribution=correction_type_dist,
                    sentiment_distribution=sentiment_dist
                )

        except Exception as e:
            logger.error(
                "get_aggregate_stats_failed",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                error=str(e)
            )
            # Return empty stats
            return FeedbackStats(
                time_range_start=start_time,
                time_range_end=end_time
            )

    async def get_feedback_count(
        self,
        session_id: Optional[str] = None,
        feedback_type: Optional[str] = None
    ) -> int:
        """
        Get count of feedback items.

        Args:
            session_id: Filter by session (optional)
            feedback_type: Filter by type ('explicit', 'implicit', 'correction')

        Returns:
            Count of matching feedback items
        """
        try:
            async with self.get_connection() as db:
                if feedback_type == 'explicit':
                    table = 'explicit_feedback'
                elif feedback_type == 'implicit':
                    table = 'implicit_feedback'
                elif feedback_type == 'correction':
                    table = 'corrections'
                else:
                    # Count all
                    total = 0
                    for t in ['explicit_feedback', 'implicit_feedback', 'corrections']:
                        if session_id:
                            cursor = await db.execute(
                                f"SELECT COUNT(*) as count FROM {t} WHERE session_id = ?",
                                (session_id,)
                            )
                        else:
                            cursor = await db.execute(f"SELECT COUNT(*) as count FROM {t}")
                        total += (await cursor.fetchone())['count']
                    return total

                if session_id:
                    cursor = await db.execute(
                        f"SELECT COUNT(*) as count FROM {table} WHERE session_id = ?",
                        (session_id,)
                    )
                else:
                    cursor = await db.execute(f"SELECT COUNT(*) as count FROM {table}")

                return (await cursor.fetchone())['count']

        except Exception as e:
            logger.error("get_feedback_count_failed", error=str(e))
            return 0

    async def close(self) -> None:
        """Close the store (cleanup if needed)."""
        self._initialized = False
        self._db = None
        logger.info("feedback_store_closed", db_path=self.db_path)

    # ========================================================================
    # GENERIC FEEDBACK API (Legacy compatibility)
    # ========================================================================

    async def store(self, feedback: 'Feedback') -> str:
        """
        Store a generic feedback item (legacy API compatibility).

        Args:
            feedback: Generic Feedback model instance

        Returns:
            Feedback ID
        """
        from app.learning.models import Feedback as GenericFeedback, FeedbackType

        try:
            async with self.get_connection() as db:
                # Ensure table exists
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS generic_feedback (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        query_id TEXT NOT NULL,
                        feedback_type TEXT NOT NULL,
                        source TEXT,
                        rating REAL,
                        text TEXT,
                        correction TEXT,
                        original_query TEXT,
                        original_response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT,
                        metadata TEXT
                    )
                """)

                await db.execute(
                    """
                    INSERT INTO generic_feedback
                    (id, session_id, query_id, feedback_type, source, rating, text,
                     correction, original_query, original_response, timestamp, user_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.id,
                        feedback.session_id,
                        feedback.query_id,
                        feedback.feedback_type.value,
                        feedback.source.value if feedback.source else None,
                        feedback.rating,
                        feedback.text,
                        feedback.correction,
                        feedback.original_query,
                        feedback.original_response,
                        feedback.timestamp.isoformat(),
                        feedback.user_id,
                        json.dumps(feedback.metadata) if feedback.metadata else None
                    )
                )
                await db.commit()

            return feedback.id

        except Exception as e:
            logger.error("store_generic_feedback_failed", error=str(e))
            raise

    async def get(self, feedback_id: str) -> 'Optional[Feedback]':
        """
        Get a feedback item by ID (legacy API compatibility).

        Args:
            feedback_id: Feedback ID

        Returns:
            Feedback instance or None
        """
        from app.learning.models import Feedback as GenericFeedback, FeedbackType, FeedbackSource

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM generic_feedback WHERE id = ?",
                    (feedback_id,)
                )
                row = await cursor.fetchone()

                if not row:
                    return None

                return GenericFeedback(
                    id=row['id'],
                    session_id=row['session_id'],
                    query_id=row['query_id'],
                    feedback_type=FeedbackType(row['feedback_type']),
                    source=FeedbackSource(row['source']) if row['source'] else FeedbackSource.USER_BUTTON,
                    rating=row['rating'],
                    text=row['text'],
                    correction=row['correction'],
                    original_query=row['original_query'] or "",
                    original_response=row['original_response'] or "",
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )

        except Exception as e:
            logger.error("get_generic_feedback_failed", error=str(e))
            return None

    async def query(
        self,
        session_id: Optional[str] = None,
        query_id: Optional[str] = None,
        feedback_type: Optional['FeedbackType'] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        **kwargs
    ) -> 'List[Feedback]':
        """
        Query feedback with filters (legacy API compatibility).

        Returns:
            List of matching Feedback instances
        """
        from app.learning.models import Feedback as GenericFeedback, FeedbackType as FT, FeedbackSource

        try:
            async with self.get_connection() as db:
                conditions = []
                params = []

                if session_id:
                    conditions.append("session_id = ?")
                    params.append(session_id)

                if query_id:
                    conditions.append("query_id = ?")
                    params.append(query_id)

                if feedback_type:
                    conditions.append("feedback_type = ?")
                    params.append(feedback_type.value if hasattr(feedback_type, 'value') else feedback_type)

                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(start_time.isoformat())

                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(end_time.isoformat())

                if min_rating is not None:
                    conditions.append("rating >= ?")
                    params.append(min_rating)

                if max_rating is not None:
                    conditions.append("rating <= ?")
                    params.append(max_rating)

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                query_sql = f"""
                    SELECT * FROM generic_feedback
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cursor = await db.execute(query_sql, params)
                rows = await cursor.fetchall()

                return [
                    GenericFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        query_id=row['query_id'],
                        feedback_type=FT(row['feedback_type']),
                        source=FeedbackSource(row['source']) if row['source'] else FeedbackSource.USER_BUTTON,
                        rating=row['rating'],
                        text=row['text'],
                        correction=row['correction'],
                        original_query=row['original_query'] or "",
                        original_response=row['original_response'] or "",
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        user_id=row['user_id'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error("query_generic_feedback_failed", error=str(e))
            return []

    async def count(
        self,
        session_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Count feedback items (legacy API compatibility).

        Returns:
            Count of matching items
        """
        try:
            async with self.get_connection() as db:
                if session_id:
                    cursor = await db.execute(
                        "SELECT COUNT(*) as count FROM generic_feedback WHERE session_id = ?",
                        (session_id,)
                    )
                else:
                    cursor = await db.execute("SELECT COUNT(*) as count FROM generic_feedback")

                row = await cursor.fetchone()
                return row['count'] if row else 0

        except Exception as e:
            logger.error("count_generic_feedback_failed", error=str(e))
            return 0

    async def get_average_rating(
        self,
        session_id: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Get average rating (legacy API compatibility).

        Returns:
            Average rating or 0.0
        """
        try:
            async with self.get_connection() as db:
                if session_id:
                    cursor = await db.execute(
                        "SELECT AVG(rating) as avg_rating FROM generic_feedback WHERE session_id = ? AND rating IS NOT NULL",
                        (session_id,)
                    )
                else:
                    cursor = await db.execute(
                        "SELECT AVG(rating) as avg_rating FROM generic_feedback WHERE rating IS NOT NULL"
                    )

                row = await cursor.fetchone()
                return row['avg_rating'] if row and row['avg_rating'] else 0.0

        except Exception as e:
            logger.error("get_average_rating_failed", error=str(e))
            return 0.0

    async def get_feedback_distribution(
        self,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, int]:
        """
        Get feedback type distribution (legacy API compatibility).

        Returns:
            Dict mapping feedback type to count
        """
        try:
            async with self.get_connection() as db:
                if session_id:
                    cursor = await db.execute(
                        """
                        SELECT feedback_type, COUNT(*) as count
                        FROM generic_feedback
                        WHERE session_id = ?
                        GROUP BY feedback_type
                        """,
                        (session_id,)
                    )
                else:
                    cursor = await db.execute(
                        """
                        SELECT feedback_type, COUNT(*) as count
                        FROM generic_feedback
                        GROUP BY feedback_type
                        """
                    )

                rows = await cursor.fetchall()
                return {row['feedback_type']: row['count'] for row in rows}

        except Exception as e:
            logger.error("get_feedback_distribution_failed", error=str(e))
            return {}

    async def delete(self, feedback_id: str) -> bool:
        """
        Delete a feedback item by ID (legacy API compatibility).

        Returns:
            True if deleted, False otherwise
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "DELETE FROM generic_feedback WHERE id = ?",
                    (feedback_id,)
                )
                await db.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error("delete_generic_feedback_failed", error=str(e))
            return False

    async def delete_old_feedback(self, retention_days: int = 90) -> int:
        """
        Delete feedback older than retention period (legacy API compatibility).

        Returns:
            Number of deleted items
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=retention_days)
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "DELETE FROM generic_feedback WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )
                await db.commit()
                return cursor.rowcount

        except Exception as e:
            logger.error("delete_old_feedback_failed", error=str(e))
            return 0

    async def vacuum(self) -> None:
        """
        Vacuum the database (legacy API compatibility).
        """
        try:
            async with self.get_connection() as db:
                await db.execute("VACUUM")

        except Exception as e:
            logger.error("vacuum_failed", error=str(e))


# Singleton instance
feedback_store = FeedbackStore()
