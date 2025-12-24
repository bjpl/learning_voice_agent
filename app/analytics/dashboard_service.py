"""
Dashboard Service
=================

Comprehensive service providing dashboard-ready data for visualization.
Integrates with learning analytics, insights, and progress tracking.

PATTERN: Facade pattern aggregating multiple data sources
WHY: Single entry point for all dashboard data needs
SPARC: Dashboard data aggregation with caching and optimization
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import asdict
from collections import defaultdict

from app.logger import db_logger
from app.analytics.analytics_config import analytics_config, DashboardConfig
from app.analytics.dashboard_models import (
    OverviewResponse,
    QuickStats,
    ProgressChartResponse,
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

# Import modular components
from app.analytics.dashboard_cache import DashboardCache
from app.analytics.dashboard_data import DashboardDataFetcher
from app.analytics.dashboard_metrics import DashboardMetricsCalculator
from app.analytics.dashboard_charts import DashboardChartBuilder
from app.analytics.dashboard_legacy import DashboardLegacyAPI


# Try to import learning stores (may not be fully initialized)
try:
    from app.learning.stores import (
        feedback_store, quality_store, pattern_store
    )
    STORES_AVAILABLE = True
except ImportError:
    STORES_AVAILABLE = False
    feedback_store = None
    quality_store = None
    pattern_store = None

try:
    from app.learning.analytics import learning_analytics
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    learning_analytics = None

try:
    from app.learning.insights_generator import insights_generator
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    insights_generator = None


# =============================================================================
# DASHBOARD SERVICE
# =============================================================================

class DashboardService:
    """
    PATTERN: Comprehensive dashboard data service
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

        # Modular components
        self._data_fetcher = None
        self._metrics_calc = DashboardMetricsCalculator()
        self._chart_builder = DashboardChartBuilder()
        self._legacy_api = None

        # Store references (set during initialization)
        self._feedback_store = None
        self._quality_store = None
        self._analytics = None
        self._insights_gen = None

    async def initialize(self) -> None:
        """Initialize service and dependencies"""
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

            # Initialize data fetcher with dependencies
            self._data_fetcher = DashboardDataFetcher(
                feedback_store=self._feedback_store,
                quality_store=self._quality_store,
                analytics=self._analytics,
                insights_gen=self._insights_gen
            )

            # Initialize metrics calculator with feedback store
            self._metrics_calc = DashboardMetricsCalculator()
            self._metrics_calc._feedback_store = self._feedback_store

            # Initialize legacy API adapter
            self._legacy_api = DashboardLegacyAPI(
                data_fetcher=self._data_fetcher,
                metrics_calc=self._metrics_calc
            )

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
        """Get main dashboard overview data with parallel fetching optimization."""
        if not self._initialized:
            await self.initialize()

        # Check cache
        cache_key = "overview_data"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_overview_data")

            # OPTIMIZATION: Fetch all data in parallel
            (
                overall_metrics,
                streak_info,
                recent_insights,
                week_ago_metrics
            ) = await asyncio.gather(
                self._data_fetcher.get_overall_metrics(),
                self._metrics_calc.get_streak_info(),
                self._data_fetcher.get_recent_insights(limit=5),
                self._data_fetcher.get_metrics_for_date(date.today() - timedelta(days=7)),
                return_exceptions=True
            )

            # Handle any exceptions in parallel results
            if isinstance(overall_metrics, Exception):
                db_logger.warning("overall_metrics_fetch_failed", error=str(overall_metrics))
                overall_metrics = {}
            if isinstance(streak_info, Exception):
                db_logger.warning("streak_info_fetch_failed", error=str(streak_info))
                from app.analytics.dashboard_models import LearningStreak
                streak_info = LearningStreak()
            if isinstance(recent_insights, Exception):
                db_logger.warning("recent_insights_fetch_failed", error=str(recent_insights))
                recent_insights = []
            if isinstance(week_ago_metrics, Exception):
                db_logger.warning("week_ago_metrics_fetch_failed", error=str(week_ago_metrics))
                week_ago_metrics = {}

            # Build overview cards using metrics calculator
            cards = self._metrics_calc.build_overview_cards(
                overall_metrics, week_ago_metrics
            )

            # Build quick stats using metrics calculator
            quick_stats = self._metrics_calc.build_quick_stats(overall_metrics)

            response = OverviewResponse(
                cards=cards,
                quick_stats=quick_stats,
                recent_insights=recent_insights,
                streak_info=streak_info,
                generated_at=datetime.utcnow(),
                cache_ttl_seconds=self.config.cache_ttl_seconds
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)
            db_logger.info("overview_data_generated", cards_count=len(cards), insights_count=len(recent_insights))

            return response

        except Exception as e:
            db_logger.error("get_overview_data_failed", error=str(e), exc_info=True)
            from app.analytics.dashboard_models import LearningStreak
            return OverviewResponse(
                cards=[], quick_stats=QuickStats(), recent_insights=[],
                streak_info=LearningStreak(), generated_at=datetime.utcnow()
            )

    # =========================================================================
    # PROGRESS CHARTS
    # =========================================================================

    async def get_progress_charts(self, period: str = "week") -> ProgressChartResponse:
        """Get progress visualization data for time-series charts."""
        if not self._initialized:
            await self.initialize()

        # Validate period
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
            daily_data = await self._data_fetcher.get_daily_progress_range(
                start_date, end_date
            )

            # Build chart using chart builder
            response = self._chart_builder.build_progress_chart(
                period=period,
                daily_data=daily_data,
                start_date=start_date,
                end_date=end_date
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)
            db_logger.info("progress_charts_generated", period=period, data_points=len(response.data_points))

            return response

        except Exception as e:
            db_logger.error("get_progress_charts_failed", period=period, error=str(e), exc_info=True)
            from app.analytics.dashboard_models import ProgressSummary
            return ProgressChartResponse(
                period=period, start_date=date.today().isoformat(), end_date=date.today().isoformat(),
                data_points=[], summary=ProgressSummary(), chart_config={}
            )

    # =========================================================================
    # TREND CHARTS
    # =========================================================================

    async def get_trend_charts(self, metrics: List[str], days: int = 30) -> TrendChartResponse:
        """Get trend chart data for multiple metrics comparison."""
        if not self._initialized:
            await self.initialize()

        # Limit days
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
            daily_data = await self._data_fetcher.get_daily_progress_range(
                start_date, end_date
            )

            # Build chart using chart builder
            response = self._chart_builder.build_trend_chart(
                metrics=metrics,
                days=days,
                daily_data=daily_data
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)
            db_logger.info("trend_charts_generated", metrics_count=len(response.metrics), days=days)

            return response

        except Exception as e:
            db_logger.error("get_trend_charts_failed", metrics=metrics, days=days, error=str(e), exc_info=True)
            return TrendChartResponse(days=days, metrics={}, labels=[])

    # =========================================================================
    # TOPIC BREAKDOWN
    # =========================================================================

    async def get_topic_breakdown(self, days: int = 30) -> TopicBreakdownResponse:
        """Get topic analytics and distribution with quality correlation."""
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

            # Build chart using chart builder
            response = self._chart_builder.build_topic_breakdown(
                topic_distribution=topic_distribution,
                days=days,
                max_topics=self.config.max_topics_displayed
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)
            db_logger.info("topic_breakdown_generated", total_topics=response.total_topics)

            return response

        except Exception as e:
            db_logger.error("get_topic_breakdown_failed", days=days, error=str(e), exc_info=True)
            return TopicBreakdownResponse(total_topics=0, topics=[], top_topics=[], emerging_topics=[], chart_data={})

    # =========================================================================
    # ACTIVITY HEATMAP
    # =========================================================================

    async def get_activity_heatmap(self, year: int) -> ActivityHeatmapResponse:
        """Get activity heatmap data (GitHub-style calendar)."""
        if not self._initialized:
            await self.initialize()

        cache_key = f"activity_heatmap_{year}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            db_logger.info("generating_activity_heatmap", year=year)

            # Get daily session counts for the year
            daily_counts = await self._data_fetcher.get_daily_session_counts(year)

            # Build heatmap using chart builder
            response = self._chart_builder.build_activity_heatmap(
                daily_counts=daily_counts,
                year=year
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)
            db_logger.info("activity_heatmap_generated", year=year, active_days=response.total_active_days)

            return response

        except Exception as e:
            db_logger.error("get_activity_heatmap_failed", year=year, error=str(e), exc_info=True)
            return ActivityHeatmapResponse(year=year, total_active_days=0, total_sessions=0, max_daily_sessions=0, weeks=[])

    # =========================================================================
    # GOAL PROGRESS
    # =========================================================================

    async def get_goal_progress(self) -> GoalProgressResponse:
        """
        Get goal tracking data.

        PATTERN: Goal progress tracking delegated to metrics calculator
        WHY: Track learning objectives and milestones

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

            # Delegate to metrics calculator
            active_goals = await self._metrics_calc.get_active_goals()
            completed_goals = await self._metrics_calc.get_completed_goals()
            overall_metrics = await self._data_fetcher.get_overall_metrics()
            suggested_goals = await self._metrics_calc.generate_goal_suggestions(overall_metrics)

            total_goals = len(active_goals) + len(completed_goals)
            completion_rate = len(completed_goals) / total_goals * 100 if total_goals > 0 else 0.0

            # Find next milestone (simplified)
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
            db_logger.info("goal_progress_generated", active=len(active_goals), completed=len(completed_goals))

            return response

        except Exception as e:
            db_logger.error("get_goal_progress_failed", error=str(e), exc_info=True)
            return GoalProgressResponse(
                active_goals=[], completed_goals=[], total_goals=0, completion_rate=0.0
            )

    # =========================================================================
    # INSIGHTS
    # =========================================================================

    async def get_insights(self, limit: int = 10) -> InsightResponse:
        """Get recent insights with prioritized actionable intelligence."""
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
            categories = defaultdict(int)
            has_critical = False
            last_generated = None

            # Generate insights from insights generator
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
                    categories[item.category] += 1
                    if item.priority == "critical":
                        has_critical = True
                    if last_generated is None or item.created_at > last_generated:
                        last_generated = item.created_at

            response = InsightResponse(
                total_insights=len(insights_list),
                insights=insights_list,
                categories=dict(categories),
                has_critical=has_critical,
                last_generated=last_generated,
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, 60)
            db_logger.info("insights_generated", count=len(insights_list))

            return response

        except Exception as e:
            db_logger.error("get_insights_failed", limit=limit, error=str(e), exc_info=True)
            return InsightResponse(total_insights=0, insights=[], categories={}, has_critical=False)

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
        Export analytics data (delegated to data fetcher).

        PATTERN: Data export with format options
        WHY: Enable data portability and offline analysis

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
            db_logger.info("exporting_data", format=format.value, start_date=str(start_date), end_date=str(end_date))

            # Default to last 30 days
            if end_date is None:
                end_date = date.today()
            if start_date is None:
                start_date = end_date - timedelta(days=30)

            # Gather data in parallel
            days = (end_date - start_date).days
            overview, progress, topics, insights = await asyncio.gather(
                self.get_overview_data(),
                self.get_progress_charts("month" if days > 7 else "week"),
                self.get_topic_breakdown(days),
                self.get_insights(limit=50),
                return_exceptions=True
            )

            # Build export data using data fetcher
            export_data = await self._data_fetcher.build_export_data(
                start_date, end_date, overview, progress, topics, insights
            )
            export_data["metadata"]["format"] = format.value

            record_count = (
                len(export_data.get("progress", [])) +
                len(export_data.get("topics", [])) +
                len(export_data.get("insights", []))
            )

            response = ExportResponse(
                format=format,
                data=export_data if format == ExportFormat.JSON else None,
                record_count=record_count,
                date_range={"start": start_date.isoformat(), "end": end_date.isoformat()},
                generated_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )

            db_logger.info("data_exported", format=format.value, records=record_count)
            return response

        except Exception as e:
            db_logger.error("export_data_failed", format=format.value, error=str(e), exc_info=True)
            return ExportResponse(format=format, record_count=0, date_range={}, generated_at=datetime.utcnow())

    # =========================================================================
    # CHART.JS DATA FORMATTERS
    # =========================================================================

    async def get_chartjs_progress_data(self, period: str = "week") -> ChartJSData:
        """
        Get progress data in Chart.js format (delegated to chart builder).

        Args:
            period: "week", "month", or "year"

        Returns:
            ChartJSData ready for Chart.js
        """
        progress = await self.get_progress_charts(period)
        return self._chart_builder.build_chartjs_progress_data(progress.data_points)


    # =========================================================================
    # LEGACY API COMPATIBILITY METHODS
    # =========================================================================

    async def get_dashboard_data(self, use_cache: bool = True) -> 'DashboardData':
        """Get complete dashboard data (legacy API compatibility)."""
        if not self._initialized:
            await self.initialize()

        cache_key = "dashboard_data"
        if use_cache and (cached := self._cache.get(cache_key)):
            return cached

        try:
            overview, progress, trends, topics, goals = await asyncio.gather(
                self.get_overview_data(),
                self.get_progress_charts("week"),
                self.get_trend_charts(["quality", "sessions"], 30),
                self.get_topic_breakdown(30),
                self.get_goal_progress(),
                return_exceptions=True
            )

            from app.analytics.progress_models import DashboardData
            data = DashboardData(
                overview=overview, streak=overview.streak_info, progress_summary=progress.summary,
                topic_breakdown=topics, goals=goals, generated_at=datetime.utcnow()
            )

            if use_cache:
                self._cache.set(cache_key, data, self.config.cache_ttl_seconds)
            return data

        except Exception as e:
            db_logger.error("get_dashboard_data_failed", error=str(e))
            from app.analytics.progress_models import DashboardData, ProgressMetrics, LearningStreak
            return DashboardData(overview=ProgressMetrics(), streak=LearningStreak(), generated_at=datetime.utcnow())

    async def get_quality_chart_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get quality chart data (legacy API compatibility with caching)."""
        if not self._initialized:
            await self.initialize()

        cache_key = f"quality_chart_{days}"
        if cached := self._cache.get(cache_key):
            return cached

        result = await self._legacy_api.get_quality_chart_data(days)
        self._cache.set(cache_key, result, ttl=120)
        return result

    async def get_progress_chart_data(
        self,
        metric: str = "exchanges",
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get progress chart data for specific metric (legacy API compatibility)."""
        if not self._initialized:
            await self.initialize()
        return await self._legacy_api.get_progress_chart_data(metric, days)

    async def get_trend_chart_data(
        self,
        metric: str = "quality",
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get trend chart data with rolling average (legacy API compatibility)."""
        if not self._initialized:
            await self.initialize()
        return await self._legacy_api.get_trend_chart_data(metric, days)

    async def get_activity_heatmap_by_weeks(self, weeks: int = 12) -> List[Dict[str, Any]]:
        """Get activity heatmap data for a number of weeks (legacy API compatibility)."""
        if not self._initialized:
            await self.initialize()
        return await self._legacy_api.get_activity_heatmap_by_weeks(weeks)

    async def get_topic_distribution(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get topic distribution data (legacy API compatibility)."""
        if not self._initialized:
            await self.initialize()

        try:
            breakdown = await self.get_topic_breakdown(days)
            return [
                {"topic": t.topic_name, "interactions": t.session_count, "percentage": t.percentage}
                for t in breakdown.topics
            ]
        except Exception as e:
            db_logger.error("get_topic_distribution_failed", error=str(e))
            return []

    async def get_goal_progress_data(self) -> List[Dict[str, Any]]:
        """Get goal progress data (legacy API compatibility)."""
        if not self._initialized:
            await self.initialize()

        try:
            progress = await self.get_goal_progress()
            result = [
                {"goal_id": g.goal_id, "title": g.title, "target": g.target_value,
                 "current": g.current_value, "progress": g.progress_percent, "status": "in_progress"}
                for g in progress.active_goals
            ]
            result.extend([
                {"goal_id": g.goal_id, "title": g.title, "target": g.target_value,
                 "current": g.current_value, "progress": 100.0, "status": "completed"}
                for g in progress.completed_goals
            ])
            return result
        except Exception as e:
            db_logger.error("get_goal_progress_data_failed", error=str(e))
            return []

    def clear_cache(self, user_id: Optional[str] = None) -> None:
        """Clear cache entries (legacy API compatibility)."""
        if user_id:
            for key in [k for k in self._cache._cache.keys() if user_id in k]:
                self._cache.invalidate(key)
        else:
            self._cache.clear()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

dashboard_service = DashboardService()
