"""
Dashboard Data Module
=====================

Data fetching and aggregation for dashboard service.

PATTERN: Data access layer for dashboard
WHY: Separate data fetching from business logic
SPARC: Modular data fetching with error handling and batch optimization
"""

import asyncio
import statistics
from datetime import date, timedelta
from typing import Dict, List, Optional, Any
from functools import lru_cache
from app.logger import db_logger


class DashboardDataFetcher:
    """
    Handles data fetching for dashboard service.

    PATTERN: Data access layer
    WHY: Centralize data fetching logic
    """

    def __init__(
        self,
        feedback_store=None,
        quality_store=None,
        analytics=None,
        insights_gen=None
    ):
        """
        Initialize data fetcher.

        Args:
            feedback_store: FeedbackStore instance
            quality_store: QualityStore instance
            analytics: LearningAnalytics instance
            insights_gen: InsightsGenerator instance
        """
        self._feedback_store = feedback_store
        self._quality_store = quality_store
        self._analytics = analytics
        self._insights_gen = insights_gen

    async def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Get overall progress metrics.

        OPTIMIZED: Parallel fetching of sessions and quality scores

        Returns:
            Dictionary with aggregate metrics
        """
        metrics = {
            "sessions_count": 0,
            "total_exchanges": 0,
            "total_hours": 0.0,
            "avg_quality_score": 0.0,
            "learning_velocity": 0.0,
            "topics_explored": 0,
            "insights_generated": 0,
            "avg_session_duration": 0.0
        }

        if self._analytics:
            try:
                # Get data from analytics
                end_date = date.today()
                start_date = end_date - timedelta(days=30)

                # Parallel fetch sessions and quality scores
                tasks = []
                if self._feedback_store:
                    tasks.append(self._feedback_store.get_sessions_in_range(start_date, end_date))
                if self._quality_store:
                    tasks.append(self._quality_store.get_scores_in_range(start_date, end_date))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process sessions
                if self._feedback_store and len(results) > 0 and not isinstance(results[0], Exception):
                    sessions = results[0]
                    metrics["sessions_count"] = len(sessions)
                    metrics["total_exchanges"] = sum(s.exchange_count for s in sessions)
                    total_duration = sum(s.duration for s in sessions if s.duration > 0)
                    metrics["total_hours"] = total_duration / 3600
                    metrics["avg_session_duration"] = (
                        total_duration / len(sessions) / 60
                        if sessions else 0.0
                    )

                    # Count unique topics
                    all_topics = set()
                    for s in sessions:
                        all_topics.update(s.topics)
                    metrics["topics_explored"] = len(all_topics)

                # Process quality scores
                scores_idx = 1 if self._feedback_store else 0
                if self._quality_store and len(results) > scores_idx and not isinstance(results[scores_idx], Exception):
                    scores = results[scores_idx]
                    if scores:
                        metrics["avg_quality_score"] = statistics.mean(
                            [s.composite for s in scores]
                        )

            except Exception as e:
                db_logger.warning(
                    "get_overall_metrics_partial_failure",
                    error=str(e)
                )

        return metrics

    async def get_metrics_for_date(self, target_date: date) -> Dict[str, Any]:
        """
        Get metrics for a specific date.

        Args:
            target_date: Date to fetch metrics for

        Returns:
            Dictionary with date-specific metrics
        """
        metrics = {"sessions": 0, "exchanges": 0, "avg_quality": 0.0}

        if self._feedback_store:
            try:
                next_date = target_date + timedelta(days=1)
                sessions = await self._feedback_store.get_sessions_in_range(
                    target_date, next_date
                )
                metrics["sessions"] = len(sessions)
                metrics["exchanges"] = sum(s.exchange_count for s in sessions)
            except Exception:
                pass

        if self._quality_store:
            try:
                next_date = target_date + timedelta(days=1)
                scores = await self._quality_store.get_scores_in_range(
                    target_date, next_date
                )
                if scores:
                    metrics["avg_quality"] = statistics.mean(
                        [s.composite for s in scores]
                    )
            except Exception:
                pass

        return metrics

    async def get_daily_progress_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """
        Get daily progress data for date range.

        OPTIMIZED: Batch fetching to eliminate N+1 queries

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of daily progress dictionaries
        """
        # Batch fetch all data for the range (eliminating N+1 queries)
        sessions_map = {}
        quality_map = {}

        # Parallel batch fetching
        fetch_tasks = []

        if self._feedback_store:
            fetch_tasks.append(self._fetch_sessions_batch(start_date, end_date))

        if self._quality_store:
            fetch_tasks.append(self._fetch_quality_batch(start_date, end_date))

        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process results
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    db_logger.warning(f"batch_fetch_failed", error=str(result))
                    continue

                if idx == 0 and self._feedback_store:
                    sessions_map = result
                elif (idx == 1 or (idx == 0 and not self._feedback_store)) and self._quality_store:
                    quality_map = result

        # Build daily data from batched results
        daily_data = []
        current = start_date
        while current <= end_date:
            date_key = current.isoformat()
            sessions_data = sessions_map.get(date_key, {})
            quality_data = quality_map.get(date_key, {})

            day_data = {
                "date": date_key,
                "sessions": sessions_data.get("count", 0),
                "exchanges": sessions_data.get("exchanges", 0),
                "quality_score": quality_data.get("avg_quality", 0.0),
                "duration_minutes": sessions_data.get("duration", 0.0)
            }

            daily_data.append(day_data)
            current += timedelta(days=1)

        return daily_data

    async def _fetch_sessions_batch(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch fetch sessions for date range (single query).

        Returns:
            Dict mapping date -> {count, exchanges, duration}
        """
        try:
            sessions = await self._feedback_store.get_sessions_in_range(
                start_date, end_date
            )

            # Group by date
            sessions_by_date = {}
            for session in sessions:
                date_key = session.start_time.date().isoformat()

                if date_key not in sessions_by_date:
                    sessions_by_date[date_key] = {
                        "count": 0,
                        "exchanges": 0,
                        "duration": 0.0
                    }

                sessions_by_date[date_key]["count"] += 1
                sessions_by_date[date_key]["exchanges"] += session.exchange_count
                sessions_by_date[date_key]["duration"] += session.duration / 60 if session.duration > 0 else 0

            return sessions_by_date

        except Exception as e:
            db_logger.warning("fetch_sessions_batch_failed", error=str(e))
            return {}

    async def _fetch_quality_batch(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, Dict[str, float]]:
        """
        Batch fetch quality scores for date range (single query).

        Returns:
            Dict mapping date -> {avg_quality}
        """
        try:
            scores = await self._quality_store.get_scores_in_range(
                start_date, end_date
            )

            # Group by date and calculate averages
            scores_by_date = {}
            for score in scores:
                date_key = score.timestamp.date().isoformat()

                if date_key not in scores_by_date:
                    scores_by_date[date_key] = []

                scores_by_date[date_key].append(score.composite)

            # Calculate means
            quality_by_date = {}
            for date_key, score_list in scores_by_date.items():
                quality_by_date[date_key] = {
                    "avg_quality": statistics.mean(score_list)
                }

            return quality_by_date

        except Exception as e:
            db_logger.warning("fetch_quality_batch_failed", error=str(e))
            return {}

    async def get_daily_session_counts(self, year: int) -> Dict[str, int]:
        """
        Get daily session counts for a year.

        Args:
            year: Year to fetch data for

        Returns:
            Dictionary mapping date strings to session counts
        """
        counts = {}

        if self._feedback_store:
            try:
                start_date = date(year, 1, 1)
                end_date = date(year, 12, 31)
                sessions = await self._feedback_store.get_sessions_in_range(
                    start_date, end_date
                )

                for session in sessions:
                    date_key = session.start_time.date().isoformat()
                    counts[date_key] = counts.get(date_key, 0) + 1

            except Exception as e:
                db_logger.warning("get_daily_session_counts_failed", error=str(e))

        return counts

    async def get_recent_insights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent insights as dictionaries.

        Args:
            limit: Maximum number of insights to return

        Returns:
            List of insight dictionaries
        """
        insights = []

        if self._insights_gen:
            try:
                raw_insights = await self._insights_gen.generate_insights(
                    time_range_days=7
                )
                for insight in raw_insights[:limit]:
                    insights.append({
                        "id": insight.insight_id,
                        "category": insight.category.value,
                        "title": insight.title,
                        "description": insight.description,
                        "actionable": insight.actionable,
                        "created_at": insight.created_at.isoformat()
                    })
            except Exception as e:
                db_logger.warning("get_recent_insights_failed", error=str(e))

        return insights

    async def build_export_data(
        self,
        start_date: date,
        end_date: date,
        overview_data: Any,
        progress_data: Any,
        topic_data: Any,
        insights_data: Any
    ) -> Dict[str, Any]:
        """
        Build export data structure from dashboard components.

        Args:
            start_date: Export start date
            end_date: Export end date
            overview_data: Overview response data
            progress_data: Progress chart response
            topic_data: Topic breakdown response
            insights_data: Insights response

        Returns:
            Dictionary with structured export data
        """
        from dataclasses import asdict

        export_data = {
            "metadata": {
                "exported_at": date.today().isoformat(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            },
            "overview": {},
            "progress": [],
            "topics": [],
            "insights": []
        }

        # Add overview data
        if overview_data:
            export_data["overview"] = {
                "quick_stats": (
                    asdict(overview_data.quick_stats)
                    if hasattr(overview_data.quick_stats, '__dataclass_fields__')
                    else overview_data.quick_stats.dict()
                ),
                "streak": (
                    asdict(overview_data.streak_info)
                    if hasattr(overview_data.streak_info, '__dataclass_fields__')
                    else overview_data.streak_info.dict()
                )
            }

        # Add progress data
        if progress_data:
            export_data["progress"] = [
                dp.dict() for dp in progress_data.data_points
            ]

        # Add topic data
        if topic_data:
            export_data["topics"] = [
                t.dict() for t in topic_data.topics
            ]

        # Add insights
        if insights_data:
            export_data["insights"] = [
                i.dict() for i in insights_data.insights
            ]

        return export_data
