"""
Progress Store - SQLite Persistence Layer
PATTERN: Repository pattern with async operations
WHY: Efficient data access for progress tracking with pre-computed aggregations

Tables:
- daily_progress: Day-by-day learning progress
- weekly_progress: Weekly aggregated metrics
- monthly_progress: Monthly aggregated metrics
- streaks: Learning streak tracking
- topic_mastery: Topic-specific progress and confidence
- milestones: Achievement milestone tracking

Features:
- Async SQLite operations with aiosqlite
- Pre-computed aggregations for fast dashboard loading
- Efficient indexing for time-range queries
- JSON metadata columns for flexibility
"""
import aiosqlite
import json
import uuid
import statistics
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager

from app.analytics.config import analytics_config
from app.analytics.progress_models import (
    ProgressMetrics,
    DailyProgress,
    WeeklyProgress,
    MonthlyProgress,
    LearningStreak,
    TopicMastery,
    ProgressLevel,
    TrendDirection,
)
from app.logger import get_logger

# Module logger
logger = get_logger("progress_store")


class ProgressStore:
    """
    SQLite persistence layer for progress tracking data.

    PATTERN: Repository with async operations and pre-computed aggregations
    WHY: Fast dashboard loading with < 500ms target
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the progress store.

        Args:
            db_path: Path to SQLite database. Uses config default if not provided.
        """
        self.db_path = db_path or analytics_config.progress.database_path
        self._initialized = False
        self._init_lock = asyncio.Lock()

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
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("progress_store_initialization_started", db_path=self.db_path)

            try:
                async with aiosqlite.connect(self.db_path) as db:
                    # Daily progress table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS daily_progress (
                            id TEXT PRIMARY KEY,
                            date TEXT NOT NULL UNIQUE,
                            user_id TEXT DEFAULT 'default',

                            -- Core metrics
                            sessions_count INTEGER DEFAULT 0,
                            total_exchanges INTEGER DEFAULT 0,
                            total_duration_minutes REAL DEFAULT 0,
                            avg_quality_score REAL DEFAULT 0,

                            -- Topic metrics
                            topics_explored INTEGER DEFAULT 0,
                            topics_list TEXT DEFAULT '[]',

                            -- Feedback metrics
                            positive_feedback INTEGER DEFAULT 0,
                            negative_feedback INTEGER DEFAULT 0,
                            corrections_made INTEGER DEFAULT 0,
                            clarifications_requested INTEGER DEFAULT 0,

                            -- Computed metrics
                            learning_velocity REAL DEFAULT 0,
                            engagement_depth REAL DEFAULT 0,
                            quality_trend REAL DEFAULT 0,

                            -- Session tracking
                            session_ids TEXT DEFAULT '[]',
                            quality_scores TEXT DEFAULT '[]',

                            -- Metadata
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Weekly progress table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS weekly_progress (
                            id TEXT PRIMARY KEY,
                            week_start TEXT NOT NULL,
                            week_end TEXT NOT NULL,
                            user_id TEXT DEFAULT 'default',

                            -- Aggregated metrics
                            sessions_count INTEGER DEFAULT 0,
                            total_exchanges INTEGER DEFAULT 0,
                            total_duration_minutes REAL DEFAULT 0,
                            avg_quality_score REAL DEFAULT 0,

                            -- Topic metrics
                            topics_explored INTEGER DEFAULT 0,
                            topics_list TEXT DEFAULT '[]',

                            -- Activity metrics
                            active_days INTEGER DEFAULT 0,
                            best_day TEXT,
                            quality_by_day TEXT DEFAULT '{}',

                            -- Trend metrics
                            quality_improvement REAL DEFAULT 0,
                            consistency_score REAL DEFAULT 0,

                            -- Metadata
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                            UNIQUE(week_start, user_id)
                        )
                    """)

                    # Monthly progress table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS monthly_progress (
                            id TEXT PRIMARY KEY,
                            year INTEGER NOT NULL,
                            month INTEGER NOT NULL,
                            user_id TEXT DEFAULT 'default',

                            -- Aggregated metrics
                            sessions_count INTEGER DEFAULT 0,
                            total_exchanges INTEGER DEFAULT 0,
                            total_duration_hours REAL DEFAULT 0,
                            avg_quality_score REAL DEFAULT 0,

                            -- Topic metrics
                            topics_explored INTEGER DEFAULT 0,
                            topics_list TEXT DEFAULT '[]',

                            -- Activity metrics
                            active_days INTEGER DEFAULT 0,
                            total_weeks INTEGER DEFAULT 0,
                            best_week TEXT,
                            quality_by_week TEXT DEFAULT '{}',

                            -- Trend metrics
                            trend_direction TEXT DEFAULT 'stable',
                            activity_rate REAL DEFAULT 0,

                            -- Metadata
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                            UNIQUE(year, month, user_id)
                        )
                    """)

                    # Streaks table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS streaks (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL UNIQUE,

                            -- Streak data
                            current_streak INTEGER DEFAULT 0,
                            longest_streak INTEGER DEFAULT 0,
                            total_active_days INTEGER DEFAULT 0,
                            last_active_date TEXT,
                            streak_start_date TEXT,

                            -- History
                            streak_broken_count INTEGER DEFAULT 0,
                            avg_streak_length REAL DEFAULT 0,
                            streak_history TEXT DEFAULT '[]',

                            -- Metadata
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Topic mastery table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS topic_mastery (
                            id TEXT PRIMARY KEY,
                            topic TEXT NOT NULL,
                            user_id TEXT DEFAULT 'default',

                            -- Mastery metrics
                            exchanges_count INTEGER DEFAULT 0,
                            avg_quality REAL DEFAULT 0,
                            mastery_score REAL DEFAULT 0,
                            confidence_level TEXT DEFAULT 'beginner',
                            confidence_score REAL DEFAULT 0,

                            -- Temporal tracking
                            first_discussed DATETIME,
                            last_discussed DATETIME,
                            discussion_frequency REAL DEFAULT 0,

                            -- Quality tracking
                            quality_scores TEXT DEFAULT '[]',
                            quality_trend TEXT DEFAULT 'stable',

                            -- Related topics
                            related_topics TEXT DEFAULT '[]',

                            -- Metadata
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                            UNIQUE(topic, user_id)
                        )
                    """)

                    # Milestones table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS milestones (
                            id TEXT PRIMARY KEY,
                            milestone_id TEXT NOT NULL,
                            user_id TEXT DEFAULT 'default',

                            -- Milestone data
                            milestone_type TEXT NOT NULL,
                            title TEXT NOT NULL,
                            description TEXT,
                            requirement_value INTEGER NOT NULL,
                            current_value INTEGER DEFAULT 0,

                            -- Status
                            achieved INTEGER DEFAULT 0,
                            achieved_at DATETIME,

                            -- Metadata
                            icon TEXT DEFAULT '',
                            tier INTEGER DEFAULT 1,
                            points INTEGER DEFAULT 0,

                            -- Timestamps
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                            UNIQUE(milestone_id, user_id)
                        )
                    """)

                    # Create indexes for efficient queries
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_daily_progress_date
                        ON daily_progress(date DESC, user_id)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_weekly_progress_week
                        ON weekly_progress(week_start DESC, user_id)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monthly_progress_month
                        ON monthly_progress(year DESC, month DESC, user_id)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_topic_mastery_topic
                        ON topic_mastery(topic, user_id)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_topic_mastery_score
                        ON topic_mastery(mastery_score DESC)
                    """)

                    await db.commit()

                self._initialized = True
                logger.info("progress_store_initialization_complete", db_path=self.db_path)

            except Exception as e:
                logger.error(
                    "progress_store_initialization_failed",
                    db_path=self.db_path,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

    # ========================================================================
    # DAILY PROGRESS OPERATIONS
    # ========================================================================

    async def save_daily_progress(self, progress: DailyProgress) -> str:
        """
        Save or update daily progress.

        Args:
            progress: DailyProgress model instance

        Returns:
            Progress ID
        """
        if not self._initialized:
            await self.initialize()

        progress_id = str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO daily_progress
                    (id, date, user_id, sessions_count, total_exchanges,
                     total_duration_minutes, avg_quality_score, topics_explored,
                     topics_list, positive_feedback, negative_feedback,
                     corrections_made, clarifications_requested, learning_velocity,
                     engagement_depth, quality_trend, session_ids, quality_scores,
                     updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress_id,
                    progress.date.isoformat(),
                    progress.user_id if hasattr(progress, 'user_id') else 'default',
                    progress.total_sessions,
                    progress.total_exchanges,
                    progress.total_time_minutes,
                    progress.avg_quality_score,
                    len(set(progress.topics_covered)) if hasattr(progress, 'topics_covered') else 0,
                    json.dumps(progress.topics_covered if hasattr(progress, 'topics_covered') else []),
                    progress.positive_feedback_count if hasattr(progress, 'positive_feedback_count') else 0,
                    progress.negative_feedback_count if hasattr(progress, 'negative_feedback_count') else 0,
                    progress.corrections_made if hasattr(progress, 'corrections_made') else 0,
                    progress.clarifications_needed if hasattr(progress, 'clarifications_needed') else 0,
                    progress.learning_velocity if hasattr(progress, 'learning_velocity') else 0,
                    progress.engagement_depth if hasattr(progress, 'engagement_depth') else 0,
                    0,  # quality_trend
                    json.dumps(progress.session_ids if hasattr(progress, 'session_ids') else []),
                    json.dumps(progress.quality_scores if hasattr(progress, 'quality_scores') else []),
                    datetime.utcnow().isoformat()
                ))
                await db.commit()

            logger.debug("daily_progress_saved", date=progress.date.isoformat())
            return progress_id

        except Exception as e:
            logger.error(
                "save_daily_progress_failed",
                date=progress.date.isoformat(),
                error=str(e)
            )
            raise

    async def get_daily_progress(
        self,
        target_date: date,
        user_id: str = "default"
    ) -> Optional[DailyProgress]:
        """Get daily progress for a specific date."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM daily_progress
                    WHERE date = ? AND user_id = ?
                """, (target_date.isoformat(), user_id))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_daily_progress(row)

        except Exception as e:
            logger.error(
                "get_daily_progress_failed",
                date=target_date.isoformat(),
                error=str(e)
            )
            return None

    async def get_daily_progress_range(
        self,
        start_date: date,
        end_date: date,
        user_id: str = "default"
    ) -> List[DailyProgress]:
        """Get daily progress for a date range."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM daily_progress
                    WHERE date >= ? AND date <= ? AND user_id = ?
                    ORDER BY date ASC
                """, (start_date.isoformat(), end_date.isoformat(), user_id))
                rows = await cursor.fetchall()

                return [self._row_to_daily_progress(row) for row in rows]

        except Exception as e:
            logger.error(
                "get_daily_progress_range_failed",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                error=str(e)
            )
            return []

    async def get_all_daily_progress(
        self,
        user_id: str = "default",
        limit: int = 365
    ) -> List[DailyProgress]:
        """Get all daily progress records."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM daily_progress
                    WHERE user_id = ?
                    ORDER BY date DESC
                    LIMIT ?
                """, (user_id, limit))
                rows = await cursor.fetchall()

                return [self._row_to_daily_progress(row) for row in rows]

        except Exception as e:
            logger.error("get_all_daily_progress_failed", error=str(e))
            return []

    # ========================================================================
    # WEEKLY PROGRESS OPERATIONS
    # ========================================================================

    async def save_weekly_progress(self, progress: WeeklyProgress) -> str:
        """Save or update weekly progress."""
        if not self._initialized:
            await self.initialize()

        progress_id = str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO weekly_progress
                    (id, week_start, week_end, user_id, sessions_count,
                     total_exchanges, total_duration_minutes, avg_quality_score,
                     topics_explored, topics_list, active_days, best_day,
                     quality_by_day, quality_improvement, consistency_score,
                     updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress_id,
                    progress.week_start.isoformat(),
                    progress.week_end.isoformat(),
                    progress.user_id if hasattr(progress, 'user_id') else 'default',
                    progress.total_sessions,
                    progress.total_exchanges,
                    progress.total_time_minutes,
                    progress.avg_quality_score,
                    len(set(progress.topics_covered)),
                    json.dumps(progress.topics_covered),
                    progress.active_days,
                    progress.best_day.isoformat() if progress.best_day else None,
                    json.dumps(progress.quality_by_day if hasattr(progress, 'quality_by_day') else {}),
                    progress.quality_improvement if hasattr(progress, 'quality_improvement') else 0,
                    progress.consistency_score if hasattr(progress, 'consistency_score') else 0,
                    datetime.utcnow().isoformat()
                ))
                await db.commit()

            logger.debug("weekly_progress_saved", week_start=progress.week_start.isoformat())
            return progress_id

        except Exception as e:
            logger.error(
                "save_weekly_progress_failed",
                week_start=progress.week_start.isoformat(),
                error=str(e)
            )
            raise

    async def get_weekly_progress(
        self,
        week_start: date,
        user_id: str = "default"
    ) -> Optional[WeeklyProgress]:
        """Get weekly progress for a specific week."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM weekly_progress
                    WHERE week_start = ? AND user_id = ?
                """, (week_start.isoformat(), user_id))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_weekly_progress(row)

        except Exception as e:
            logger.error(
                "get_weekly_progress_failed",
                week_start=week_start.isoformat(),
                error=str(e)
            )
            return None

    async def get_weekly_progress_range(
        self,
        start_date: date,
        end_date: date,
        user_id: str = "default"
    ) -> List[WeeklyProgress]:
        """Get weekly progress for a date range."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM weekly_progress
                    WHERE week_start >= ? AND week_start <= ? AND user_id = ?
                    ORDER BY week_start ASC
                """, (start_date.isoformat(), end_date.isoformat(), user_id))
                rows = await cursor.fetchall()

                return [self._row_to_weekly_progress(row) for row in rows]

        except Exception as e:
            logger.error(
                "get_weekly_progress_range_failed",
                error=str(e)
            )
            return []

    # ========================================================================
    # MONTHLY PROGRESS OPERATIONS
    # ========================================================================

    async def save_monthly_progress(self, progress: MonthlyProgress) -> str:
        """Save or update monthly progress."""
        if not self._initialized:
            await self.initialize()

        progress_id = str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO monthly_progress
                    (id, year, month, user_id, sessions_count, total_exchanges,
                     total_duration_hours, avg_quality_score, topics_explored,
                     topics_list, active_days, total_weeks, best_week,
                     quality_by_week, trend_direction, activity_rate, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress_id,
                    progress.year,
                    progress.month,
                    progress.user_id if hasattr(progress, 'user_id') else 'default',
                    progress.total_sessions,
                    progress.total_exchanges,
                    progress.total_time_hours,
                    progress.avg_quality_score,
                    len(set(progress.topics_covered if hasattr(progress, 'topics_covered') else [])),
                    json.dumps(progress.topics_covered if hasattr(progress, 'topics_covered') else []),
                    progress.active_days,
                    progress.total_weeks if hasattr(progress, 'total_weeks') else 0,
                    progress.best_week.isoformat() if hasattr(progress, 'best_week') and progress.best_week else None,
                    json.dumps(progress.quality_by_week if hasattr(progress, 'quality_by_week') else {}),
                    progress.trend_direction if hasattr(progress, 'trend_direction') else 'stable',
                    progress.activity_rate if hasattr(progress, 'activity_rate') else 0,
                    datetime.utcnow().isoformat()
                ))
                await db.commit()

            logger.debug("monthly_progress_saved", year=progress.year, month=progress.month)
            return progress_id

        except Exception as e:
            logger.error(
                "save_monthly_progress_failed",
                year=progress.year,
                month=progress.month,
                error=str(e)
            )
            raise

    async def get_monthly_progress(
        self,
        year: int,
        month: int,
        user_id: str = "default"
    ) -> Optional[MonthlyProgress]:
        """Get monthly progress for a specific month."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM monthly_progress
                    WHERE year = ? AND month = ? AND user_id = ?
                """, (year, month, user_id))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_monthly_progress(row)

        except Exception as e:
            logger.error(
                "get_monthly_progress_failed",
                year=year,
                month=month,
                error=str(e)
            )
            return None

    async def get_monthly_progress_range(
        self,
        months: int = 12,
        user_id: str = "default"
    ) -> List[MonthlyProgress]:
        """Get monthly progress for the last N months."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM monthly_progress
                    WHERE user_id = ?
                    ORDER BY year DESC, month DESC
                    LIMIT ?
                """, (user_id, months))
                rows = await cursor.fetchall()

                return [self._row_to_monthly_progress(row) for row in rows]

        except Exception as e:
            logger.error("get_monthly_progress_range_failed", error=str(e))
            return []

    # ========================================================================
    # STREAK OPERATIONS
    # ========================================================================

    async def save_streak(self, streak: LearningStreak) -> str:
        """Save or update learning streak."""
        if not self._initialized:
            await self.initialize()

        streak_id = str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO streaks
                    (id, user_id, current_streak, longest_streak, total_active_days,
                     last_active_date, streak_start_date, streak_broken_count,
                     avg_streak_length, streak_history, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    streak_id,
                    streak.user_id if hasattr(streak, 'user_id') else 'default',
                    streak.current_streak,
                    streak.longest_streak,
                    streak.total_active_days if hasattr(streak, 'total_active_days') else 0,
                    streak.last_active_date.isoformat() if streak.last_active_date else None,
                    streak.streak_start_date.isoformat() if streak.streak_start_date else None,
                    streak.streak_broken_count if hasattr(streak, 'streak_broken_count') else 0,
                    streak.avg_streak_length if hasattr(streak, 'avg_streak_length') else 0,
                    json.dumps(streak.streak_history if hasattr(streak, 'streak_history') else []),
                    datetime.utcnow().isoformat()
                ))
                await db.commit()

            logger.debug("streak_saved", user_id=streak.user_id if hasattr(streak, 'user_id') else 'default')
            return streak_id

        except Exception as e:
            logger.error("save_streak_failed", error=str(e))
            raise

    async def get_streak(self, user_id: str = "default") -> Optional[LearningStreak]:
        """Get learning streak for a user."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM streaks WHERE user_id = ?
                """, (user_id,))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_streak(row)

        except Exception as e:
            logger.error("get_streak_failed", user_id=user_id, error=str(e))
            return None

    async def update_streak(self, user_id: str, activity_date: date) -> LearningStreak:
        """Update streak for a user based on activity date."""
        if not self._initialized:
            await self.initialize()

        streak = await self.get_streak(user_id)

        if streak is None:
            streak = LearningStreak(user_id=user_id)

        streak.update(activity_date)
        await self.save_streak(streak)

        return streak

    # ========================================================================
    # TOPIC MASTERY OPERATIONS
    # ========================================================================

    async def save_topic_mastery(self, mastery: TopicMastery) -> str:
        """Save or update topic mastery."""
        if not self._initialized:
            await self.initialize()

        mastery_id = str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO topic_mastery
                    (id, topic, user_id, exchanges_count, avg_quality,
                     mastery_score, confidence_level, confidence_score,
                     first_discussed, last_discussed, discussion_frequency,
                     quality_scores, quality_trend, related_topics, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mastery_id,
                    mastery.topic,
                    mastery.user_id if hasattr(mastery, 'user_id') else 'default',
                    mastery.total_interactions,
                    mastery.avg_quality_score,
                    mastery.mastery_score,
                    mastery.level.value if hasattr(mastery, 'level') else 'beginner',
                    mastery.confidence,
                    mastery.first_interaction.isoformat() if mastery.first_interaction else None,
                    mastery.last_interaction.isoformat() if mastery.last_interaction else None,
                    mastery.discussion_frequency if hasattr(mastery, 'discussion_frequency') else 0,
                    json.dumps(mastery.quality_scores if hasattr(mastery, 'quality_scores') else []),
                    mastery.quality_trend.value if hasattr(mastery, 'quality_trend') else 'stable',
                    json.dumps(mastery.related_topics if hasattr(mastery, 'related_topics') else []),
                    datetime.utcnow().isoformat()
                ))
                await db.commit()

            logger.debug("topic_mastery_saved", topic=mastery.topic)
            return mastery_id

        except Exception as e:
            logger.error("save_topic_mastery_failed", topic=mastery.topic, error=str(e))
            raise

    async def get_topic_mastery(
        self,
        topic: str,
        user_id: str = "default"
    ) -> Optional[TopicMastery]:
        """Get mastery for a specific topic."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM topic_mastery
                    WHERE topic = ? AND user_id = ?
                """, (topic, user_id))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_topic_mastery(row)

        except Exception as e:
            logger.error("get_topic_mastery_failed", topic=topic, error=str(e))
            return None

    async def get_all_topic_mastery(
        self,
        user_id: str = "default",
        limit: int = 50
    ) -> List[TopicMastery]:
        """Get all topic mastery records, ordered by mastery score."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM topic_mastery
                    WHERE user_id = ?
                    ORDER BY mastery_score DESC
                    LIMIT ?
                """, (user_id, limit))
                rows = await cursor.fetchall()

                return [self._row_to_topic_mastery(row) for row in rows]

        except Exception as e:
            logger.error("get_all_topic_mastery_failed", error=str(e))
            return []

    async def get_topic_exchanges(
        self,
        topic: str,
        user_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """Get exchange history for a topic (placeholder - would need session store integration)."""
        # This would need integration with the feedback store
        # For now, return based on stored quality scores
        mastery = await self.get_topic_mastery(topic, user_id)
        if not mastery:
            return []

        # Return quality scores as exchange proxies
        return [
            {"quality": score, "topic": topic}
            for score in (mastery.quality_scores if hasattr(mastery, 'quality_scores') else [])
        ]

    # ========================================================================
    # AGGREGATION HELPERS
    # ========================================================================

    async def compute_overall_metrics(
        self,
        user_id: str = "default"
    ) -> ProgressMetrics:
        """Compute overall progress metrics from stored data."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                # Get totals from daily progress
                cursor = await db.execute("""
                    SELECT
                        COUNT(*) as total_days,
                        SUM(sessions_count) as total_sessions,
                        SUM(total_exchanges) as total_exchanges,
                        SUM(total_duration_minutes) as total_duration,
                        AVG(avg_quality_score) as avg_quality,
                        SUM(positive_feedback) as total_positive,
                        SUM(negative_feedback) as total_negative,
                        SUM(corrections_made) as total_corrections
                    FROM daily_progress
                    WHERE user_id = ?
                """, (user_id,))
                row = await cursor.fetchone()

                if not row or row['total_days'] == 0:
                    return ProgressMetrics()

                # Get topic count
                cursor = await db.execute("""
                    SELECT COUNT(DISTINCT topic) as topic_count
                    FROM topic_mastery
                    WHERE user_id = ?
                """, (user_id,))
                topic_row = await cursor.fetchone()

                # Get streak
                streak = await self.get_streak(user_id)

                # Calculate metrics
                total_days = row['total_days'] or 1
                total_sessions = row['total_sessions'] or 0

                return ProgressMetrics(
                    sessions_count=total_sessions,
                    total_exchanges=row['total_exchanges'] or 0,
                    total_time_hours=(row['total_duration'] or 0) / 60,
                    avg_quality_score=row['avg_quality'] or 0,
                    topics_explored=topic_row['topic_count'] or 0,
                    current_streak=streak.current_streak if streak else 0,
                    longest_streak=streak.longest_streak if streak else 0,
                    learning_velocity=(row['total_exchanges'] or 0) / total_days,
                )

        except Exception as e:
            logger.error("compute_overall_metrics_failed", error=str(e))
            return ProgressMetrics()

    # ========================================================================
    # ROW CONVERSION HELPERS
    # ========================================================================

    def _row_to_daily_progress(self, row) -> DailyProgress:
        """Convert database row to DailyProgress model."""
        return DailyProgress(
            id=row['id'],
            date=date.fromisoformat(row['date']),
            total_sessions=row['sessions_count'],
            total_exchanges=row['total_exchanges'],
            total_time_minutes=row['total_duration_minutes'],
            avg_quality_score=row['avg_quality_score'],
            topics_covered=json.loads(row['topics_list'] or '[]'),
        )

    def _row_to_weekly_progress(self, row) -> WeeklyProgress:
        """Convert database row to WeeklyProgress model."""
        return WeeklyProgress(
            id=row['id'],
            week_start=date.fromisoformat(row['week_start']),
            week_end=date.fromisoformat(row['week_end']),
            total_sessions=row['sessions_count'],
            total_exchanges=row['total_exchanges'],
            total_time_minutes=row['total_duration_minutes'],
            avg_quality_score=row['avg_quality_score'],
            topics_covered=json.loads(row['topics_list'] or '[]'),
            active_days=row['active_days'],
            best_day=date.fromisoformat(row['best_day']) if row['best_day'] else None,
        )

    def _row_to_monthly_progress(self, row) -> MonthlyProgress:
        """Convert database row to MonthlyProgress model."""
        return MonthlyProgress(
            id=row['id'],
            year=row['year'],
            month=row['month'],
            total_sessions=row['sessions_count'],
            total_exchanges=row['total_exchanges'],
            total_time_hours=row['total_duration_hours'],
            avg_quality_score=row['avg_quality_score'],
            active_days=row['active_days'],
        )

    def _row_to_streak(self, row) -> LearningStreak:
        """Convert database row to LearningStreak model."""
        return LearningStreak(
            id=row['id'],
            user_id=row['user_id'],
            current_streak=row['current_streak'],
            longest_streak=row['longest_streak'],
            last_active_date=date.fromisoformat(row['last_active_date']) if row['last_active_date'] else None,
            streak_start_date=date.fromisoformat(row['streak_start_date']) if row['streak_start_date'] else None,
            streak_history=json.loads(row['streak_history'] or '[]'),
        )

    def _row_to_topic_mastery(self, row) -> TopicMastery:
        """Convert database row to TopicMastery model."""
        return TopicMastery(
            id=row['id'],
            topic=row['topic'],
            user_id=row['user_id'],
            total_interactions=row['exchanges_count'],
            avg_quality_score=row['avg_quality'],
            mastery_score=row['mastery_score'],
            level=ProgressLevel(row['confidence_level']),
            confidence=row['confidence_score'],
            first_interaction=datetime.fromisoformat(row['first_discussed']) if row['first_discussed'] else None,
            last_interaction=datetime.fromisoformat(row['last_discussed']) if row['last_discussed'] else None,
            quality_trend=TrendDirection(row['quality_trend']),
        )

    async def close(self) -> None:
        """Close the store (cleanup if needed)."""
        self._initialized = False
        logger.info("progress_store_closed", db_path=self.db_path)


# Global store instance
progress_store = ProgressStore()
