"""
Dashboard Service (Refactored)
==============================

Comprehensive service providing dashboard-ready data for visualization.
Integrates with learning analytics, insights, and progress tracking.

PATTERN: Facade pattern aggregating multiple data sources
WHY: Single entry point for all dashboard data needs
SPARC: Dashboard data aggregation with caching and optimization

Refactored from original 1,492-line file into modular components:
- dashboard_cache.py: Caching layer
- dashboard_metrics.py: KPI calculations
- dashboard_charts.py: Chart data generation
- dashboard_service.py: Main facade (this file)
"""

import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import asdict

from app.logger import db_logger
from app.analytics.analytics_config import analytics_config, DashboardConfig
from app.analytics.dashboard_cache import DashboardCache
from app.analytics.dashboard_metrics import dashboard_metrics_calculator
from app.analytics.dashboard_charts import dashboard_chart_builder
from app.analytics.dashboard_models import (
    OverviewResponse,
    QuickStats,
    LearningStreak,
    ProgressChartResponse,
    ProgressSummary,
    TrendChartResponse,
    TopicBreakdownResponse,
    ActivityHeatmapResponse,
    GoalProgressResponse,
    LearningGoal as DashboardGoal,
    InsightResponse,
    InsightItem,
    ExportFormat,
    ExportResponse,
)
from app.analytics.chart_data import ChartJSData


# Try to import learning stores (may not be fully initialized)
try:
    from app.learning.stores import (
        FeedbackStore, QualityStore, PatternStore,
        feedback_store, quality_store, pattern_store,
        SessionData, QualityScore, FeedbackData
    )
    STORES_AVAILABLE = True
except ImportError:
    STORES_AVAILABLE = False
    feedback_store = None
    quality_store = None
    pattern_store = None

try:
    from app.learning.analytics import learning_analytics, LearningAnalytics
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    learning_analytics = None

try:
    from app.learning.insights_generator import insights_generator, InsightsGenerator
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    insights_generator = None


class DashboardService:
    """
    PATTERN: Comprehensive dashboard data service (Facade)
    WHY: Single source of truth for dashboard visualization data

    Provides:
    - Overview summaries with KPI cards
    - Progress charts (line, area)
    - Trend analysis for multiple metrics
    - Topic breakdown with quality correlation
    - Activity heatmap (GitHub-style)
    - Goal progress tracking
    - Insight summaries

    USAGE:
        service = DashboardService()
        await service.initialize()

        # Get main dashboard data
        overview = await service.get_overview_data()

        # Get chart data
        progress = await service.get_progress_charts("week")
        trends = await service.get_trend_charts(["quality", "engagement"], 30)
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        cache_ttl: int = 300
    ):
        """
        Initialize dashboard service.

        Args:
            config: Dashboard configuration
            cache_ttl: Default cache TTL in seconds
        """
        self.config = config or analytics_config.dashboard
        self._cache = DashboardCache(default_ttl=cache_ttl)
        self._initialized = False

        # Store references (set during initialization)
        self._feedback_store = None
        self._quality_store = None
        self._analytics = None
        self._insights_gen = None

    async def initialize(self) -> None:
        """Initialize service and dependencies."""
        if self._initialized:
            return

        try:
            # Initialize available stores
            if STORES_AVAILABLE and feedback_store:
                await feedback_store.initialize()
                self._feedback_store = feedback_store
            if STORES_AVAILABLE and quality_store:
                await quality_store.initialize()
                self._quality_store = quality_store

            # Initialize analytics
            if ANALYTICS_AVAILABLE and learning_analytics:
                await learning_analytics.initialize()
                self._analytics = learning_analytics

            # Initialize insights generator
            if INSIGHTS_AVAILABLE and insights_generator:
                await insights_generator.initialize()
                self._insights_gen = insights_generator

            self._initialized = True
            db_logger.info("dashboard_service_initialized")

        except Exception as e:
            db_logger.error(
                "dashboard_service_initialization_failed",
                error=str(e),
                exc_info=True
            )
            # Continue in degraded mode
            self._initialized = True

    # =========================================================================
    # OVERVIEW DATA
    # =========================================================================

    async def get_overview_data(self) -> OverviewResponse:
        """
        Get main dashboard overview data.

        Returns:
            OverviewResponse with cards, stats, insights, and streak
        """
        if not self._initialized:
            await self.initialize()

        cache_key = "overview_data"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_overview_data")

            # Get metrics from stores
            overall_metrics = await self._get_overall_metrics()
            streak_info = await self._get_streak_info()
            recent_insights = await self._get_recent_insights(limit=5)

            # Get previous period for comparison
            week_ago_metrics = await self._get_metrics_for_date(
                date.today() - timedelta(days=7)
            )

            # Build overview cards using metrics calculator
            cards = dashboard_metrics_calculator.build_overview_cards(
                overall_metrics, week_ago_metrics
            )

            # Build quick stats
            quick_stats = dashboard_metrics_calculator.build_quick_stats(overall_metrics)

            response = OverviewResponse(
                cards=cards,
                quick_stats=quick_stats,
                recent_insights=recent_insights,
                streak_info=streak_info,
                generated_at=datetime.utcnow(),
                cache_ttl_seconds=self.config.cache_ttl_seconds
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "overview_data_generated",
                cards_count=len(cards),
                insights_count=len(recent_insights)
            )

            return response

        except Exception as e:
            db_logger.error(
                "get_overview_data_failed",
                error=str(e),
                exc_info=True
            )
            return OverviewResponse(
                cards=[],
                quick_stats=QuickStats(),
                recent_insights=[],
                streak_info=LearningStreak(),
                generated_at=datetime.utcnow()
            )

    # =========================================================================
    # PROGRESS CHARTS
    # =========================================================================

    async def get_progress_charts(
        self,
        period: str = "week"
    ) -> ProgressChartResponse:
        """
        Get progress visualization data.

        Args:
            period: "week", "month", or "year"

        Returns:
            ProgressChartResponse with chart-ready data
        """
        if not self._initialized:
            await self.initialize()

        period_days = {"week": 7, "month": 30, "year": 365}
        days = period_days.get(period, 7)

        cache_key = f"progress_charts_{period}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_progress_charts", period=period)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get daily progress data
            daily_data = await self._get_daily_progress_range(start_date, end_date)

            # Use chart builder
            response = dashboard_chart_builder.build_progress_chart(
                period, daily_data, start_date, end_date
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "progress_charts_generated",
                period=period,
                data_points=len(response.data_points)
            )

            return response

        except Exception as e:
            db_logger.error(
                "get_progress_charts_failed",
                period=period,
                error=str(e),
                exc_info=True
            )
            return ProgressChartResponse(
                period=period,
                start_date=date.today().isoformat(),
                end_date=date.today().isoformat(),
                data_points=[],
                summary=ProgressSummary(),
                chart_config={}
            )

    # =========================================================================
    # TREND CHARTS
    # =========================================================================

    async def get_trend_charts(
        self,
        metrics: List[str],
        days: int = 30
    ) -> TrendChartResponse:
        """
        Get trend chart data for multiple metrics.

        Args:
            metrics: List of metric names
            days: Number of days to analyze

        Returns:
            TrendChartResponse with metric trends
        """
        if not self._initialized:
            await self.initialize()

        days = min(days, 365)

        cache_key = f"trend_charts_{'_'.join(sorted(metrics))}_{days}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_trend_charts", metrics=metrics, days=days)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get daily data
            daily_data = await self._get_daily_progress_range(start_date, end_date)

            # Use chart builder
            response = dashboard_chart_builder.build_trend_chart(
                metrics, days, daily_data
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "trend_charts_generated",
                metrics_count=len(response.metrics),
                days=days
            )

            return response

        except Exception as e:
            db_logger.error(
                "get_trend_charts_failed",
                metrics=metrics,
                days=days,
                error=str(e),
                exc_info=True
            )
            return TrendChartResponse(days=days, metrics={}, labels=[])

    # =========================================================================
    # TOPIC BREAKDOWN
    # =========================================================================

    async def get_topic_breakdown(
        self,
        days: int = 30
    ) -> TopicBreakdownResponse:
        """
        Get topic analytics and distribution.

        Args:
            days: Number of days to analyze

        Returns:
            TopicBreakdownResponse with topic statistics
        """
        if not self._initialized:
            await self.initialize()

        cache_key = f"topic_breakdown_{days}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_topic_breakdown", days=days)

            # Get topic data from analytics
            topic_distribution = {}
            if self._analytics:
                topic_distribution = await self._analytics.get_topic_distribution(days)

            # Use chart builder
            response = dashboard_chart_builder.build_topic_breakdown(
                topic_distribution, days, self.config.max_topics_displayed
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "topic_breakdown_generated",
                total_topics=response.total_topics
            )

            return response

        except Exception as e:
            db_logger.error(
                "get_topic_breakdown_failed",
                days=days,
                error=str(e),
                exc_info=True
            )
            return TopicBreakdownResponse(
                total_topics=0,
                topics=[],
                top_topics=[],
                emerging_topics=[],
                chart_data={}
            )

    # =========================================================================
    # ACTIVITY HEATMAP
    # =========================================================================

    async def get_activity_heatmap(
        self,
        year: int
    ) -> ActivityHeatmapResponse:
        """
        Get activity heatmap data (GitHub-style calendar).

        Args:
            year: Year to generate heatmap for

        Returns:
            ActivityHeatmapResponse with calendar data
        """
        if not self._initialized:
            await self.initialize()

        cache_key = f"activity_heatmap_{year}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_activity_heatmap", year=year)

            # Get daily session counts for the year
            daily_counts = await self._get_daily_session_counts(year)

            # Use chart builder
            response = dashboard_chart_builder.build_activity_heatmap(
                daily_counts, year
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "activity_heatmap_generated",
                year=year,
                active_days=response.total_active_days
            )

            return response

        except Exception as e:
            db_logger.error(
                "get_activity_heatmap_failed",
                year=year,
                error=str(e),
                exc_info=True
            )
            return ActivityHeatmapResponse(
                year=year,
                total_active_days=0,
                total_sessions=0,
                max_daily_sessions=0,
                weeks=[]
            )

    # =========================================================================
    # GOAL PROGRESS
    # =========================================================================

    async def get_goal_progress(self) -> GoalProgressResponse:
        """
        Get goal tracking data.

        Returns:
            GoalProgressResponse with active and completed goals
        """
        if not self._initialized:
            await self.initialize()

        cache_key = "goal_progress"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_goal_progress")

            active_goals = await self._get_active_goals()
            completed_goals = await self._get_completed_goals()

            total_goals = len(active_goals) + len(completed_goals)
            completion_rate = (
                len(completed_goals) / total_goals * 100
                if total_goals > 0 else 0.0
            )

            # Find next milestone
            next_milestone = None
            if active_goals:
                closest = max(active_goals, key=lambda g: g.progress_percent)
                if closest.progress_percent > 50:
                    next_milestone = {
                        "goal_id": closest.goal_id,
                        "title": closest.title,
                        "progress": closest.progress_percent,
                        "remaining": closest.target_value - closest.current_value
                    }

            suggested_goals = await self._generate_goal_suggestions()

            response = GoalProgressResponse(
                active_goals=active_goals[:self.config.max_active_goals],
                completed_goals=completed_goals[:10],
                total_goals=total_goals,
                completion_rate=completion_rate,
                next_milestone=next_milestone,
                suggested_goals=suggested_goals,
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "goal_progress_generated",
                active=len(active_goals),
                completed=len(completed_goals)
            )

            return response

        except Exception as e:
            db_logger.error(
                "get_goal_progress_failed",
                error=str(e),
                exc_info=True
            )
            return GoalProgressResponse(
                active_goals=[],
                completed_goals=[],
                total_goals=0,
                completion_rate=0.0
            )

    # =========================================================================
    # INSIGHTS
    # =========================================================================

    async def get_insights(
        self,
        limit: int = 10
    ) -> InsightResponse:
        """
        Get recent insights.

        Args:
            limit: Maximum insights to return

        Returns:
            InsightResponse with insights list
        """
        if not self._initialized:
            await self.initialize()

        limit = min(limit, 100)
        cache_key = f"insights_{limit}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_insights", limit=limit)

            insights_list = []
            categories = {}
            has_critical = False
            last_generated = None

            if self._insights_gen:
                raw_insights = await self._insights_gen.generate_insights(
                    time_range_days=7
                )
                for insight in raw_insights[:limit]:
                    item = InsightItem(
                        insight_id=insight.insight_id,
                        category=insight.category.value,
                        title=insight.title,
                        description=insight.description,
                        priority=insight.priority.value if hasattr(insight, 'priority') else "medium",
                        actionable=insight.actionable,
                        recommendation=insight.recommendation,
                        confidence=insight.confidence,
                        evidence=insight.evidence,
                        created_at=insight.created_at
                    )
                    insights_list.append(item)
                    categories[item.category] = categories.get(item.category, 0) + 1
                    if item.priority == "critical":
                        has_critical = True
                    if last_generated is None or item.created_at > last_generated:
                        last_generated = item.created_at

            response = InsightResponse(
                total_insights=len(insights_list),
                insights=insights_list,
                categories=categories,
                has_critical=has_critical,
                last_generated=last_generated,
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, 60)

            db_logger.info("insights_generated", count=len(insights_list))

            return response

        except Exception as e:
            db_logger.error(
                "get_insights_failed",
                limit=limit,
                error=str(e),
                exc_info=True
            )
            return InsightResponse(
                total_insights=0,
                insights=[],
                categories={},
                has_critical=False
            )

    # =========================================================================
    # EXPORT
    # =========================================================================

    async def export_data(
        self,
        format: ExportFormat = ExportFormat.JSON,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> ExportResponse:
        """
        Export analytics data.

        Args:
            format: Export format (json, csv, pdf)
            start_date: Start of export range
            end_date: End of export range

        Returns:
            ExportResponse with data or download URL
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "exporting_data",
                format=format.value,
                start_date=str(start_date),
                end_date=str(end_date)
            )

            if end_date is None:
                end_date = date.today()
            if start_date is None:
                start_date = end_date - timedelta(days=30)

            export_data = {
                "metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "format": format.value
                },
                "overview": {},
                "progress": [],
                "topics": [],
                "insights": []
            }

            # Get overview
            overview = await self.get_overview_data()
            export_data["overview"] = {
                "quick_stats": asdict(overview.quick_stats) if hasattr(overview.quick_stats, '__dataclass_fields__') else overview.quick_stats.dict(),
                "streak": asdict(overview.streak_info) if hasattr(overview.streak_info, '__dataclass_fields__') else overview.streak_info.dict()
            }

            # Get progress data
            days = (end_date - start_date).days
            progress = await self.get_progress_charts(
                "month" if days > 7 else "week"
            )
            export_data["progress"] = [dp.dict() for dp in progress.data_points]

            # Get topic breakdown
            topics = await self.get_topic_breakdown(days)
            export_data["topics"] = [t.dict() for t in topics.topics]

            # Get insights
            insights = await self.get_insights(limit=50)
            export_data["insights"] = [i.dict() for i in insights.insights]

            record_count = (
                len(export_data["progress"]) +
                len(export_data["topics"]) +
                len(export_data["insights"])
            )

            response = ExportResponse(
                format=format,
                data=export_data if format == ExportFormat.JSON else None,
                record_count=record_count,
                date_range={
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                generated_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )

            db_logger.info(
                "data_exported",
                format=format.value,
                records=record_count
            )

            return response

        except Exception as e:
            db_logger.error(
                "export_data_failed",
                format=format.value,
                error=str(e),
                exc_info=True
            )
            return ExportResponse(
                format=format,
                record_count=0,
                date_range={},
                generated_at=datetime.utcnow()
            )

    # =========================================================================
    # CHART.JS DATA FORMATTERS
    # =========================================================================

    async def get_chartjs_progress_data(
        self,
        period: str = "week"
    ) -> ChartJSData:
        """
        Get progress data in Chart.js format.

        Args:
            period: "week", "month", or "year"

        Returns:
            ChartJSData ready for Chart.js
        """
        progress = await self.get_progress_charts(period)
        return dashboard_chart_builder.build_chartjs_progress_data(progress.data_points)

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    async def _get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall progress metrics."""
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
                end_date = date.today()
                start_date = end_date - timedelta(days=30)

                if self._feedback_store:
                    sessions = await self._feedback_store.get_sessions_in_range(
                        start_date, end_date
                    )
                    metrics["sessions_count"] = len(sessions)
                    metrics["total_exchanges"] = sum(s.exchange_count for s in sessions)
                    total_duration = sum(s.duration for s in sessions if s.duration > 0)
                    metrics["total_hours"] = total_duration / 3600
                    metrics["avg_session_duration"] = (
                        total_duration / len(sessions) / 60
                        if sessions else 0.0
                    )

                    all_topics = set()
                    for s in sessions:
                        all_topics.update(s.topics)
                    metrics["topics_explored"] = len(all_topics)

                if self._quality_store:
                    scores = await self._quality_store.get_scores_in_range(
                        start_date, end_date
                    )
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

    async def _get_metrics_for_date(self, target_date: date) -> Dict[str, Any]:
        """Get metrics for a specific date."""
        metrics = {"sessions": 0, "exchanges": 0, "avg_quality": 0.0}

        if self._feedback_store:
            try:
                next_date = target_date + timedelta(days=1)
                sessions = await self._feedback_store.get_sessions_in_range(
                    target_date, next_date
                )
                metrics["sessions"] = len(sessions)
                metrics["exchanges"] = sum(s.exchange_count for s in sessions)
            except (AttributeError, RuntimeError, TypeError):
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
            except (AttributeError, RuntimeError, TypeError):
                pass

        return metrics

    async def _get_streak_info(self) -> LearningStreak:
        """Get current streak information."""
        streak = LearningStreak(
            current_streak=0,
            longest_streak=0,
            last_active_date=None,
            streak_start_date=None,
            is_active_today=False
        )

        if self._feedback_store:
            try:
                end_date = date.today()
                start_date = end_date - timedelta(days=90)
                sessions = await self._feedback_store.get_sessions_in_range(
                    start_date, end_date
                )
                streak = dashboard_metrics_calculator.calculate_streak_from_sessions(
                    sessions, start_date, end_date
                )
            except Exception as e:
                db_logger.warning("get_streak_info_failed", error=str(e))

        return streak

    async def _get_recent_insights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent insights as dictionaries."""
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

    async def _get_daily_progress_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get daily progress data for date range."""
        daily_data = []

        current = start_date
        while current <= end_date:
            day_data = {
                "date": current.isoformat(),
                "sessions": 0,
                "exchanges": 0,
                "quality_score": 0.0,
                "duration_minutes": 0.0
            }

            if self._feedback_store:
                try:
                    next_date = current + timedelta(days=1)
                    sessions = await self._feedback_store.get_sessions_in_range(
                        current, next_date
                    )
                    day_data["sessions"] = len(sessions)
                    day_data["exchanges"] = sum(s.exchange_count for s in sessions)
                    day_data["duration_minutes"] = sum(
                        s.duration / 60 for s in sessions if s.duration > 0
                    )
                except (AttributeError, RuntimeError, TypeError):
                    pass

            if self._quality_store:
                try:
                    next_date = current + timedelta(days=1)
                    scores = await self._quality_store.get_scores_in_range(
                        current, next_date
                    )
                    if scores:
                        day_data["quality_score"] = statistics.mean(
                            [s.composite for s in scores]
                        )
                except (AttributeError, RuntimeError, TypeError):
                    pass

            daily_data.append(day_data)
            current += timedelta(days=1)

        return daily_data

    async def _get_daily_session_counts(self, year: int) -> Dict[str, int]:
        """Get daily session counts for a year."""
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

    async def _get_active_goals(self) -> List[DashboardGoal]:
        """Get active learning goals."""
        return []

    async def _get_completed_goals(self) -> List[DashboardGoal]:
        """Get completed learning goals."""
        return []

    async def _generate_goal_suggestions(self) -> List[Dict[str, Any]]:
        """Generate suggested goals based on activity."""
        suggestions = []

        metrics = await self._get_overall_metrics()

        if metrics["sessions_count"] < 5:
            suggestions.append({
                "title": "Complete 5 sessions this week",
                "target_metric": "sessions",
                "target_value": 5,
                "reason": "Build a learning habit"
            })

        if metrics["avg_quality_score"] < 0.7:
            suggestions.append({
                "title": "Achieve 70% quality score",
                "target_metric": "quality",
                "target_value": 0.7,
                "reason": "Focus on response quality"
            })

        return suggestions[:3]


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

dashboard_service = DashboardService()
