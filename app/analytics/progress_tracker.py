"""
Progress Tracker
================

Tracks and aggregates learning progress across sessions.

PATTERN: Event-driven progress aggregation
WHY: Provide real-time progress updates and historical tracking
SPARC: Specification-driven metrics with comprehensive coverage
"""

import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import uuid

from app.logger import db_logger
from app.analytics.analytics_config import AnalyticsEngineConfig, analytics_config
from app.analytics.progress_models import (
    ProgressMetrics,
    LearningStreak,
    TopicMastery,
    SessionProgress,
    DailyProgress,
    WeeklyProgress,
    MonthlyProgress,
    ProgressSnapshot,
    TrendDirection,
    ProgressLevel,
)


class ProgressTracker:
    """
    Track and aggregate learning progress.

    PATTERN: Event-driven aggregation with caching
    WHY: Efficient progress tracking with minimal latency

    USAGE:
        tracker = ProgressTracker()
        await tracker.initialize()

        # Record session progress
        await tracker.record_session_progress(session_progress)

        # Get overall progress
        metrics = await tracker.get_overall_progress()

        # Get streak information
        streak = await tracker.get_learning_streak()
    """

    def __init__(
        self,
        config: Optional[AnalyticsEngineConfig] = None,
        progress_store: Optional[Any] = None,
        feedback_store: Optional[Any] = None,
        quality_store: Optional[Any] = None
    ):
        """
        Initialize progress tracker.

        Args:
            config: Analytics configuration
            progress_store: Store for progress data
            feedback_store: Store for feedback data
            quality_store: Store for quality scores
        """
        self.config = config or analytics_config
        self.progress_store = progress_store
        self.feedback_store = feedback_store
        self.quality_store = quality_store

        # Internal caches
        self._streak_cache: Dict[str, LearningStreak] = {}
        self._mastery_cache: Dict[str, Dict[str, TopicMastery]] = {}
        self._metrics_cache: Dict[str, Tuple[datetime, ProgressMetrics]] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize progress tracker and load cached data."""
        if self._initialized:
            return

        try:
            db_logger.info("progress_tracker_initializing")

            # Initialize stores if provided
            if self.progress_store:
                await self.progress_store.initialize()
            if self.feedback_store:
                await self.feedback_store.initialize()
            if self.quality_store:
                await self.quality_store.initialize()

            self._initialized = True
            db_logger.info("progress_tracker_initialized")

        except Exception as e:
            db_logger.error(
                "progress_tracker_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def record_session_progress(
        self,
        session_progress: SessionProgress
    ) -> SessionProgress:
        """
        Record progress for a session.

        Args:
            session_progress: Session progress data

        Returns:
            Updated session progress
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "recording_session_progress",
                session_id=session_progress.session_id
            )

            # Finalize session if end time not set
            if session_progress.end_time is None:
                session_progress.finalize()

            # Store session progress
            if self.progress_store:
                await self.progress_store.store_session_progress(session_progress)

            # Update streak
            await self._update_streak(
                session_progress.user_id,
                session_progress.start_time.date()
            )

            # Update topic mastery
            for topic in session_progress.topics:
                await self._update_topic_mastery(
                    session_progress.user_id,
                    topic,
                    session_progress.avg_quality_score
                )

            # Invalidate metrics cache
            cache_key = session_progress.user_id or "global"
            if cache_key in self._metrics_cache:
                del self._metrics_cache[cache_key]

            db_logger.info(
                "session_progress_recorded",
                session_id=session_progress.session_id,
                exchanges=session_progress.total_exchanges
            )

            return session_progress

        except Exception as e:
            db_logger.error(
                "record_session_progress_failed",
                session_id=session_progress.session_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_overall_progress(
        self,
        user_id: Optional[str] = None
    ) -> ProgressMetrics:
        """
        Get overall progress metrics.

        Args:
            user_id: Optional user ID for user-specific progress

        Returns:
            ProgressMetrics with aggregated data
        """
        if not self._initialized:
            await self.initialize()

        try:
            cache_key = user_id or "global"

            # Check cache
            if cache_key in self._metrics_cache:
                cached_time, cached_metrics = self._metrics_cache[cache_key]
                if (datetime.utcnow() - cached_time).total_seconds() < self.config.progress.cache_ttl_seconds:
                    return cached_metrics

            db_logger.info("calculating_overall_progress", user_id=user_id)

            # Get data from stores
            sessions = await self._get_sessions(user_id)
            streak = await self.get_learning_streak(user_id)
            topic_mastery = await self.get_all_topic_mastery(user_id)

            # Calculate metrics
            metrics = ProgressMetrics(
                user_id=user_id,
                sessions_count=len(sessions),
                total_exchanges=sum(s.total_exchanges for s in sessions),
                total_time_hours=sum(s.duration_minutes for s in sessions) / 60,
                current_streak=streak.current_streak if streak else 0,
                longest_streak=streak.longest_streak if streak else 0,
                topics_explored=len(topic_mastery),
                topics_mastered=sum(
                    1 for tm in topic_mastery.values()
                    if tm.level in [ProgressLevel.ADVANCED, ProgressLevel.EXPERT]
                )
            )

            # Calculate averages
            if sessions:
                metrics.avg_quality_score = statistics.mean(
                    s.avg_quality_score for s in sessions
                )
                if metrics.total_time_hours > 0:
                    metrics.learning_velocity = metrics.total_exchanges / metrics.total_time_hours

                # Time-based metrics
                metrics.first_session = min(s.start_time for s in sessions)
                metrics.last_session = max(s.start_time for s in sessions)

                # Most active hour and day
                hours = [s.start_time.hour for s in sessions]
                days = [s.start_time.strftime("%A") for s in sessions]
                if hours:
                    metrics.most_active_hour = max(set(hours), key=hours.count)
                if days:
                    metrics.most_active_day = max(set(days), key=days.count)

            # Cache result
            self._metrics_cache[cache_key] = (datetime.utcnow(), metrics)

            db_logger.info(
                "overall_progress_calculated",
                user_id=user_id,
                sessions=metrics.sessions_count,
                exchanges=metrics.total_exchanges
            )

            return metrics

        except Exception as e:
            db_logger.error(
                "get_overall_progress_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            # Return empty metrics on error
            return ProgressMetrics(user_id=user_id)

    async def get_learning_streak(
        self,
        user_id: Optional[str] = None
    ) -> LearningStreak:
        """
        Get current learning streak.

        Args:
            user_id: Optional user ID

        Returns:
            LearningStreak with current streak data
        """
        if not self._initialized:
            await self.initialize()

        try:
            cache_key = user_id or "global"

            # Check cache
            if cache_key in self._streak_cache:
                return self._streak_cache[cache_key]

            # Load from store or create new
            streak = None
            if self.progress_store:
                streak = await self.progress_store.get_streak(user_id)

            if streak is None:
                streak = LearningStreak(user_id=user_id)

            self._streak_cache[cache_key] = streak
            return streak

        except Exception as e:
            db_logger.error(
                "get_learning_streak_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return LearningStreak(user_id=user_id)

    async def get_topic_mastery(
        self,
        user_id: Optional[str],
        topic: str
    ) -> TopicMastery:
        """
        Get mastery level for a specific topic.

        Args:
            user_id: Optional user ID
            topic: Topic name

        Returns:
            TopicMastery for the topic
        """
        if not self._initialized:
            await self.initialize()

        try:
            cache_key = user_id or "global"

            # Check cache
            if cache_key in self._mastery_cache:
                if topic in self._mastery_cache[cache_key]:
                    return self._mastery_cache[cache_key][topic]

            # Load from store or create new
            mastery = None
            if self.progress_store:
                mastery = await self.progress_store.get_topic_mastery(user_id, topic)

            if mastery is None:
                mastery = TopicMastery(topic=topic, user_id=user_id)

            # Update cache
            if cache_key not in self._mastery_cache:
                self._mastery_cache[cache_key] = {}
            self._mastery_cache[cache_key][topic] = mastery

            return mastery

        except Exception as e:
            db_logger.error(
                "get_topic_mastery_failed",
                user_id=user_id,
                topic=topic,
                error=str(e),
                exc_info=True
            )
            return TopicMastery(topic=topic, user_id=user_id)

    async def get_all_topic_mastery(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, TopicMastery]:
        """
        Get mastery levels for all topics.

        Args:
            user_id: Optional user ID

        Returns:
            Dictionary of topic -> TopicMastery
        """
        if not self._initialized:
            await self.initialize()

        try:
            cache_key = user_id or "global"

            # Check if we have all mastery data cached
            if cache_key in self._mastery_cache:
                return self._mastery_cache[cache_key]

            # Load from store
            if self.progress_store:
                mastery_list = await self.progress_store.get_all_topic_mastery(user_id)
                result = {m.topic: m for m in mastery_list}
            else:
                result = {}

            self._mastery_cache[cache_key] = result
            return result

        except Exception as e:
            db_logger.error(
                "get_all_topic_mastery_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return {}

    async def get_daily_progress(
        self,
        target_date: date,
        user_id: Optional[str] = None
    ) -> DailyProgress:
        """
        Get progress for a specific day.

        Args:
            target_date: Date to get progress for
            user_id: Optional user ID

        Returns:
            DailyProgress for the day
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "getting_daily_progress",
                date=str(target_date),
                user_id=user_id
            )

            # Get sessions for the day
            sessions = await self._get_sessions_for_date(target_date, user_id)

            # Calculate daily metrics
            daily = DailyProgress(
                date=target_date,
                user_id=user_id,
                total_sessions=len(sessions),
                completed_sessions=len([s for s in sessions if s.end_time is not None]),
                total_exchanges=sum(s.total_exchanges for s in sessions),
                total_time_minutes=sum(s.duration_minutes for s in sessions)
            )

            if daily.total_sessions > 0:
                daily.avg_session_duration = daily.total_time_minutes / daily.total_sessions
                daily.avg_quality_score = statistics.mean(
                    s.avg_quality_score for s in sessions
                )

            # Collect topics
            all_topics = set()
            for session in sessions:
                all_topics.update(session.topics)
            daily.topics_covered = list(all_topics)

            # Get streak info
            streak = await self.get_learning_streak(user_id)
            daily.streak_maintained = (
                streak.last_active_date == target_date or
                (streak.last_active_date and (target_date - streak.last_active_date).days <= 1)
            )
            daily.current_streak = streak.current_streak

            db_logger.info(
                "daily_progress_calculated",
                date=str(target_date),
                sessions=daily.total_sessions
            )

            return daily

        except Exception as e:
            db_logger.error(
                "get_daily_progress_failed",
                date=str(target_date),
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return DailyProgress(date=target_date, user_id=user_id)

    async def get_weekly_progress(
        self,
        week_start: date,
        user_id: Optional[str] = None
    ) -> WeeklyProgress:
        """
        Get progress for a specific week.

        Args:
            week_start: Monday of the week
            user_id: Optional user ID

        Returns:
            WeeklyProgress for the week
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "getting_weekly_progress",
                week_start=str(week_start),
                user_id=user_id
            )

            week_end = week_start + timedelta(days=6)

            # Get daily progress for each day
            daily_progress = []
            for i in range(7):
                day = week_start + timedelta(days=i)
                daily = await self.get_daily_progress(day, user_id)
                daily_progress.append(daily)

            # Calculate weekly aggregates
            weekly = WeeklyProgress(
                week_start=week_start,
                week_end=week_end,
                user_id=user_id,
                daily_progress=daily_progress,
                active_days=sum(1 for d in daily_progress if d.total_sessions > 0),
                total_sessions=sum(d.total_sessions for d in daily_progress),
                total_exchanges=sum(d.total_exchanges for d in daily_progress),
                total_time_minutes=sum(d.total_time_minutes for d in daily_progress)
            )

            # Calculate average quality
            quality_scores = [
                d.avg_quality_score for d in daily_progress
                if d.avg_quality_score > 0
            ]
            if quality_scores:
                weekly.avg_quality_score = statistics.mean(quality_scores)

            # Find best day
            best_daily = max(
                daily_progress,
                key=lambda d: d.avg_quality_score * d.total_exchanges,
                default=None
            )
            if best_daily and best_daily.total_sessions > 0:
                weekly.best_day = best_daily.date

            # Collect all topics
            all_topics = set()
            for daily in daily_progress:
                all_topics.update(daily.topics_covered)
            weekly.topics_covered = list(all_topics)

            db_logger.info(
                "weekly_progress_calculated",
                week_start=str(week_start),
                sessions=weekly.total_sessions,
                active_days=weekly.active_days
            )

            return weekly

        except Exception as e:
            db_logger.error(
                "get_weekly_progress_failed",
                week_start=str(week_start),
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return WeeklyProgress(
                week_start=week_start,
                week_end=week_start + timedelta(days=6),
                user_id=user_id
            )

    async def get_monthly_progress(
        self,
        year: int,
        month: int,
        user_id: Optional[str] = None
    ) -> MonthlyProgress:
        """
        Get progress for a specific month.

        Args:
            year: Year
            month: Month (1-12)
            user_id: Optional user ID

        Returns:
            MonthlyProgress for the month
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "getting_monthly_progress",
                year=year,
                month=month,
                user_id=user_id
            )

            # Get all weeks in the month
            from calendar import monthrange
            _, days_in_month = monthrange(year, month)
            month_start = date(year, month, 1)

            # Find the first Monday
            first_monday = month_start
            while first_monday.weekday() != 0:
                first_monday -= timedelta(days=1)

            # Get weekly progress
            weekly_progress = []
            current_week = first_monday
            while current_week <= date(year, month, days_in_month):
                weekly = await self.get_weekly_progress(current_week, user_id)
                weekly_progress.append(weekly)
                current_week += timedelta(days=7)

            # Calculate monthly aggregates
            monthly = MonthlyProgress(
                year=year,
                month=month,
                user_id=user_id,
                weekly_progress=weekly_progress,
                active_days=sum(w.active_days for w in weekly_progress),
                total_sessions=sum(w.total_sessions for w in weekly_progress),
                total_exchanges=sum(w.total_exchanges for w in weekly_progress),
                total_time_hours=sum(w.total_time_minutes for w in weekly_progress) / 60
            )

            # Calculate average quality
            quality_scores = [
                w.avg_quality_score for w in weekly_progress
                if w.avg_quality_score > 0
            ]
            if quality_scores:
                monthly.avg_quality_score = statistics.mean(quality_scores)

            # Get streak info
            streak = await self.get_learning_streak(user_id)
            monthly.current_streak = streak.current_streak
            monthly.longest_streak = streak.longest_streak

            # Collect all topics
            all_topics = set()
            for weekly in weekly_progress:
                all_topics.update(weekly.topics_covered)

            # Get mastery for topics
            topic_mastery = await self.get_all_topic_mastery(user_id)
            monthly.topics_mastered = [
                topic for topic, mastery in topic_mastery.items()
                if mastery.level in [ProgressLevel.ADVANCED, ProgressLevel.EXPERT]
            ]

            db_logger.info(
                "monthly_progress_calculated",
                year=year,
                month=month,
                sessions=monthly.total_sessions
            )

            return monthly

        except Exception as e:
            db_logger.error(
                "get_monthly_progress_failed",
                year=year,
                month=month,
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return MonthlyProgress(year=year, month=month, user_id=user_id)

    async def create_progress_snapshot(
        self,
        user_id: Optional[str] = None
    ) -> ProgressSnapshot:
        """
        Create a point-in-time progress snapshot.

        Args:
            user_id: Optional user ID

        Returns:
            ProgressSnapshot with current state
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("creating_progress_snapshot", user_id=user_id)

            metrics = await self.get_overall_progress(user_id)
            topic_mastery = await self.get_all_topic_mastery(user_id)

            snapshot = ProgressSnapshot(
                user_id=user_id,
                metrics=metrics,
                topic_mastery=topic_mastery
            )

            # Store snapshot
            if self.progress_store:
                await self.progress_store.store_snapshot(snapshot)

            db_logger.info("progress_snapshot_created", user_id=user_id)
            return snapshot

        except Exception as e:
            db_logger.error(
                "create_progress_snapshot_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _update_streak(
        self,
        user_id: Optional[str],
        activity_date: date
    ) -> LearningStreak:
        """Update learning streak with new activity."""
        streak = await self.get_learning_streak(user_id)
        continued = streak.update(activity_date)

        # Store updated streak
        if self.progress_store:
            await self.progress_store.store_streak(streak)

        # Update cache
        cache_key = user_id or "global"
        self._streak_cache[cache_key] = streak

        if not continued:
            db_logger.info(
                "streak_reset",
                user_id=user_id,
                previous_streak=streak.streak_history[-1]["length"] if streak.streak_history else 0
            )

        return streak

    async def _update_topic_mastery(
        self,
        user_id: Optional[str],
        topic: str,
        quality_score: float
    ) -> TopicMastery:
        """Update mastery for a topic."""
        mastery = await self.get_topic_mastery(user_id, topic)
        mastery.update_mastery(quality_score)

        # Store updated mastery
        if self.progress_store:
            await self.progress_store.store_topic_mastery(mastery)

        # Update cache
        cache_key = user_id or "global"
        if cache_key not in self._mastery_cache:
            self._mastery_cache[cache_key] = {}
        self._mastery_cache[cache_key][topic] = mastery

        return mastery

    async def _get_sessions(
        self,
        user_id: Optional[str]
    ) -> List[SessionProgress]:
        """Get all sessions for a user."""
        if self.progress_store:
            return await self.progress_store.get_sessions(user_id)
        return []

    async def _get_sessions_for_date(
        self,
        target_date: date,
        user_id: Optional[str]
    ) -> List[SessionProgress]:
        """Get sessions for a specific date."""
        if self.progress_store:
            return await self.progress_store.get_sessions_for_date(target_date, user_id)
        return []

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self._streak_cache.clear()
        self._mastery_cache.clear()
        self._metrics_cache.clear()
        db_logger.info("progress_tracker_cache_cleared")


# Global tracker instance
progress_tracker = ProgressTracker()
