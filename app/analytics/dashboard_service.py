"""
Dashboard Service
=================

Comprehensive service providing dashboard-ready data for visualization.
Integrates with learning analytics, insights, and progress tracking.

PATTERN: Facade pattern aggregating multiple data sources
WHY: Single entry point for all dashboard data needs
SPARC: Dashboard data aggregation with caching and optimization
"""

import statistics
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import uuid

from app.logger import db_logger
from app.analytics.analytics_config import analytics_config, DashboardConfig
from app.analytics.progress_models import (
    ProgressMetrics,
    LearningStreak as ProgressLearningStreak,
    DailyProgress,
    WeeklyProgress,
    TopicMastery,
    LearningGoal,
)
from app.analytics.dashboard_models import (
    OverviewCard,
    OverviewResponse,
    LearningStreak,
    QuickStats,
    ProgressChartResponse,
    ProgressDataPoint,
    ProgressSummary,
    TrendChartResponse,
    TrendMetric,
    TrendDirection,
    TopicBreakdownResponse,
    TopicStats,
    ActivityHeatmapResponse,
    ActivityCell,
    ActivityWeek,
    GoalProgressResponse,
    LearningGoal as DashboardGoal,
    InsightResponse,
    InsightItem,
    ExportFormat,
    ExportResponse,
)
from app.analytics.chart_data import (
    ChartJSData,
    ChartDataset,
    ChartDataFormatter,
    BarChartData,
    PieChartData,
    HeatmapData,
)
from app.analytics.insights_models import (
    Insight,
    InsightCategory,
    TrendData,
)


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


# =============================================================================
# CACHE IMPLEMENTATION
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL tracking"""
    data: Any
    created_at: datetime
    ttl_seconds: int

    def is_valid(self) -> bool:
        """Check if cache entry is still valid"""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed < self.ttl_seconds


class DashboardCache:
    """
    Simple in-memory cache for dashboard data.

    PATTERN: LRU-style cache with TTL
    WHY: Reduce computation for frequently accessed data
    """

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if valid"""
        entry = self._cache.get(key)
        if entry and entry.is_valid():
            return entry.data
        elif entry:
            # Expired - remove
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value with TTL"""
        self._cache[key] = CacheEntry(
            data=value,
            created_at=datetime.utcnow(),
            ttl_seconds=ttl or self._default_ttl
        )

    def invalidate(self, key: str) -> None:
        """Remove specific cache entry"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()


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

        PATTERN: Aggregated KPI dashboard
        WHY: Single call for landing page data

        Returns:
            OverviewResponse with cards, stats, insights, and streak
        """
        if not self._initialized:
            await self.initialize()

        # Check cache
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

            # Build overview cards
            cards = self._build_overview_cards(overall_metrics, week_ago_metrics)

            # Build quick stats
            quick_stats = QuickStats(
                total_exchanges=overall_metrics.get("total_exchanges", 0),
                total_hours=overall_metrics.get("total_hours", 0.0),
                learning_velocity=overall_metrics.get("learning_velocity", 0.0),
                insights_generated=overall_metrics.get("insights_generated", 0),
                avg_session_duration=overall_metrics.get("avg_session_duration", 0.0),
                topics_explored=overall_metrics.get("topics_explored", 0)
            )

            response = OverviewResponse(
                cards=cards,
                quick_stats=quick_stats,
                recent_insights=recent_insights,
                streak_info=streak_info,
                generated_at=datetime.utcnow(),
                cache_ttl_seconds=self.config.cache_ttl_seconds
            )

            # Cache result
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
            # Return empty response
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

        PATTERN: Time-series progress data
        WHY: Track learning progress over time

        Args:
            period: "week", "month", or "year"

        Returns:
            ProgressChartResponse with chart-ready data
        """
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
            daily_data = await self._get_daily_progress_range(start_date, end_date)

            # Build data points
            data_points = []
            total_sessions = 0
            total_exchanges = 0
            total_duration = 0.0
            quality_scores = []

            for day_data in daily_data:
                data_points.append(ProgressDataPoint(
                    date=day_data["date"],
                    sessions=day_data.get("sessions", 0),
                    exchanges=day_data.get("exchanges", 0),
                    quality_score=day_data.get("quality_score", 0.0),
                    duration_minutes=day_data.get("duration_minutes", 0.0)
                ))
                total_sessions += day_data.get("sessions", 0)
                total_exchanges += day_data.get("exchanges", 0)
                total_duration += day_data.get("duration_minutes", 0.0)
                if day_data.get("quality_score", 0) > 0:
                    quality_scores.append(day_data["quality_score"])

            # Calculate summary
            avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0

            # Calculate improvement rate
            improvement_rate = 0.0
            if len(quality_scores) >= 2:
                first_half = quality_scores[:len(quality_scores)//2]
                second_half = quality_scores[len(quality_scores)//2:]
                if first_half and second_half:
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    if first_avg > 0:
                        improvement_rate = ((second_avg - first_avg) / first_avg) * 100

            # Find best day
            best_day = None
            if daily_data:
                best = max(daily_data, key=lambda d: d.get("sessions", 0))
                if best.get("sessions", 0) > 0:
                    best_day = best["date"]

            summary = ProgressSummary(
                total_sessions=total_sessions,
                total_exchanges=total_exchanges,
                total_duration_hours=total_duration / 60,
                avg_quality_score=avg_quality,
                improvement_rate=improvement_rate,
                best_day=best_day
            )

            # Generate Chart.js config hints
            chart_config = ChartDataFormatter.generate_chart_config(
                chart_type="line",
                title=f"Progress - Last {period.title()}",
                y_axis_label="Count",
                show_legend=True
            )

            response = ProgressChartResponse(
                period=period,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                data_points=data_points,
                summary=summary,
                chart_config=chart_config,
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "progress_charts_generated",
                period=period,
                data_points=len(data_points)
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

        PATTERN: Multi-metric trend comparison
        WHY: Compare trends across different metrics

        Args:
            metrics: List of metric names (quality, engagement, sessions, duration)
            days: Number of days to analyze

        Returns:
            TrendChartResponse with metric trends
        """
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
            daily_data = await self._get_daily_progress_range(start_date, end_date)

            # Build labels
            labels = [d["date"] for d in daily_data]

            # Build metric trends
            metric_colors = {
                "quality": "#4F46E5",
                "engagement": "#10B981",
                "sessions": "#F59E0B",
                "duration": "#EF4444",
                "positive_rate": "#8B5CF6"
            }

            metric_display = {
                "quality": "Quality Score",
                "engagement": "Engagement",
                "sessions": "Sessions",
                "duration": "Duration",
                "positive_rate": "Positive Rate"
            }

            trend_metrics = {}
            for metric in metrics:
                # Extract values for this metric
                values = self._extract_metric_values(daily_data, metric)

                # Calculate trend direction and change
                direction, change = ChartDataFormatter.calculate_trend_direction(values)
                trend_direction = TrendDirection(direction)

                current_value = values[-1] if values else 0.0

                trend_metrics[metric] = TrendMetric(
                    metric_name=metric,
                    display_name=metric_display.get(metric, metric.title()),
                    values=values,
                    labels=labels,
                    current_value=current_value,
                    change=change,
                    trend=trend_direction,
                    color=metric_colors.get(metric, "#4F46E5")
                )

            response = TrendChartResponse(
                days=days,
                metrics=trend_metrics,
                labels=labels,
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "trend_charts_generated",
                metrics_count=len(trend_metrics),
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
            return TrendChartResponse(
                days=days,
                metrics={},
                labels=[]
            )

    # =========================================================================
    # TOPIC BREAKDOWN
    # =========================================================================

    async def get_topic_breakdown(
        self,
        days: int = 30
    ) -> TopicBreakdownResponse:
        """
        Get topic analytics and distribution.

        PATTERN: Categorical analytics with quality correlation
        WHY: Understand content distribution and performance by topic

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

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get topic data from analytics
            topic_distribution = {}
            if self._analytics:
                topic_distribution = await self._analytics.get_topic_distribution(days)

            # Build topic stats
            topics = []
            total_sessions = sum(
                t.get("session_count", t.get("count", 0))
                for t in topic_distribution.values()
            ) if topic_distribution else 0

            for topic_name, stats in topic_distribution.items():
                session_count = stats.get("session_count", stats.get("count", 0))
                topics.append(TopicStats(
                    topic_name=topic_name,
                    session_count=session_count,
                    exchange_count=stats.get("avg_exchanges", 0) * session_count,
                    avg_quality_score=stats.get("avg_quality", 0.0),
                    total_duration_minutes=stats.get("total_duration", 0.0),
                    percentage=stats.get("percentage", 0.0),
                    trend=TrendDirection.STABLE
                ))

            # Sort by session count
            topics.sort(key=lambda t: t.session_count, reverse=True)

            # Get top topics
            top_topics = [t.topic_name for t in topics[:5]]

            # Identify emerging topics (appeared in last 7 days with growth)
            emerging_topics = []  # Would need historical comparison

            # Build pie chart data
            chart_data = ChartDataFormatter.format_topic_distribution(
                {t.topic_name: t.session_count for t in topics},
                max_topics=self.config.max_topics_displayed
            )

            response = TopicBreakdownResponse(
                total_topics=len(topics),
                topics=topics[:self.config.max_topics_displayed],
                top_topics=top_topics,
                emerging_topics=emerging_topics,
                chart_data=chart_data.to_dict(),
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "topic_breakdown_generated",
                total_topics=len(topics)
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

        PATTERN: Calendar heatmap visualization
        WHY: Show activity patterns over the year

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

            # Build heatmap data
            heatmap = HeatmapData.from_daily_counts(daily_counts, year)

            # Convert to response format
            weeks = []
            cells_by_week = defaultdict(list)

            for cell in heatmap.cells:
                cells_by_week[cell.y].append(ActivityCell(
                    date=cell.date_str,
                    count=int(cell.value),
                    intensity=cell.intensity
                ))

            for week_num in sorted(cells_by_week.keys()):
                weeks.append(ActivityWeek(
                    week_number=week_num,
                    days=cells_by_week[week_num]
                ))

            # Calculate totals
            total_active_days = sum(1 for c in heatmap.cells if c.value > 0)
            total_sessions = sum(c.value for c in heatmap.cells)
            max_daily = int(heatmap.max_value)

            # Generate month labels
            month_labels = []
            for month in range(1, 13):
                month_labels.append({
                    "month": month,
                    "name": date(year, month, 1).strftime("%b"),
                    "week_start": (date(year, month, 1) - date(year, 1, 1)).days // 7
                })

            response = ActivityHeatmapResponse(
                year=year,
                total_active_days=total_active_days,
                total_sessions=int(total_sessions),
                max_daily_sessions=max_daily,
                weeks=weeks,
                month_labels=month_labels,
                generated_at=datetime.utcnow()
            )

            self._cache.set(cache_key, response, self.config.cache_ttl_seconds)

            db_logger.info(
                "activity_heatmap_generated",
                year=year,
                active_days=total_active_days
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

        PATTERN: Goal progress tracking
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

            # Get goals (would come from a goal store)
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
                # Find goal closest to completion
                closest = max(active_goals, key=lambda g: g.progress_percent)
                if closest.progress_percent > 50:
                    next_milestone = {
                        "goal_id": closest.goal_id,
                        "title": closest.title,
                        "progress": closest.progress_percent,
                        "remaining": closest.target_value - closest.current_value
                    }

            # Generate suggested goals
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

        PATTERN: Prioritized insights list
        WHY: Surface actionable intelligence

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

            self._cache.set(cache_key, response, 60)  # Shorter TTL for insights

            db_logger.info(
                "insights_generated",
                count=len(insights_list)
            )

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
            db_logger.info(
                "exporting_data",
                format=format.value,
                start_date=str(start_date),
                end_date=str(end_date)
            )

            # Default to last 30 days
            if end_date is None:
                end_date = date.today()
            if start_date is None:
                start_date = end_date - timedelta(days=30)

            # Get data for export
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
            export_data["progress"] = [
                dp.dict() for dp in progress.data_points
            ]

            # Get topic breakdown
            topics = await self.get_topic_breakdown(days)
            export_data["topics"] = [
                t.dict() for t in topics.topics
            ]

            # Get insights
            insights = await self.get_insights(limit=50)
            export_data["insights"] = [
                i.dict() for i in insights.insights
            ]

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

        data_points = [
            {"date": dp.date, "value": dp.sessions}
            for dp in progress.data_points
        ]

        chart_data = ChartDataFormatter.format_time_series(
            data_points,
            label="Sessions",
            color="#4F46E5"
        )

        # Add quality as second dataset
        quality_points = [
            {"date": dp.date, "value": dp.quality_score * 100}
            for dp in progress.data_points
        ]
        chart_data.add_dataset(
            "Quality Score",
            [p["value"] for p in quality_points],
            "#10B981"
        )

        return chart_data

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    async def _get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall progress metrics"""
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

                    # Count unique topics
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
        """Get metrics for a specific date"""
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

    async def _get_streak_info(self) -> LearningStreak:
        """Get current streak information"""
        streak = LearningStreak(
            current_streak=0,
            longest_streak=0,
            last_active_date=None,
            streak_start_date=None,
            is_active_today=False
        )

        if self._feedback_store:
            try:
                # Get sessions in last 90 days to calculate streak
                end_date = date.today()
                start_date = end_date - timedelta(days=90)
                sessions = await self._feedback_store.get_sessions_in_range(
                    start_date, end_date
                )

                # Group by date
                dates_with_sessions = set()
                for s in sessions:
                    dates_with_sessions.add(s.start_time.date())

                # Calculate current streak
                current_date = end_date
                current_streak = 0
                streak_start = None

                while current_date >= start_date:
                    if current_date in dates_with_sessions:
                        current_streak += 1
                        streak_start = current_date
                        current_date -= timedelta(days=1)
                    elif current_streak > 0:
                        break
                    else:
                        current_date -= timedelta(days=1)

                streak.current_streak = current_streak
                streak.longest_streak = current_streak  # Simplified
                streak.is_active_today = end_date in dates_with_sessions
                streak.last_active_date = max(dates_with_sessions) if dates_with_sessions else None
                streak.streak_start_date = streak_start

            except Exception as e:
                db_logger.warning("get_streak_info_failed", error=str(e))

        return streak

    async def _get_recent_insights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent insights as dictionaries"""
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
        """Get daily progress data for date range"""
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
                except Exception:
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
                except Exception:
                    pass

            daily_data.append(day_data)
            current += timedelta(days=1)

        return daily_data

    async def _get_daily_session_counts(self, year: int) -> Dict[str, int]:
        """Get daily session counts for a year"""
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

    def _build_overview_cards(
        self,
        current: Dict[str, Any],
        previous: Dict[str, Any]
    ) -> List[OverviewCard]:
        """Build overview KPI cards"""
        cards = []

        # Sessions card
        sessions = current.get("sessions_count", 0)
        prev_sessions = previous.get("sessions", 0)
        change = self._calculate_change(sessions, prev_sessions)
        cards.append(OverviewCard(
            title="Total Sessions",
            value=str(sessions),
            raw_value=float(sessions),
            change=change,
            trend=TrendDirection.UP if change and change > 0 else TrendDirection.STABLE,
            icon="chat",
            color="blue"
        ))

        # Quality card
        quality = current.get("avg_quality_score", 0.0)
        prev_quality = previous.get("avg_quality", 0.0)
        quality_change = self._calculate_change(quality, prev_quality)
        cards.append(OverviewCard(
            title="Avg Quality",
            value=f"{quality:.1%}",
            raw_value=quality,
            change=quality_change,
            trend=TrendDirection.UP if quality_change and quality_change > 0 else TrendDirection.STABLE,
            icon="star",
            color="green"
        ))

        # Topics card
        topics = current.get("topics_explored", 0)
        cards.append(OverviewCard(
            title="Topics Explored",
            value=str(topics),
            raw_value=float(topics),
            icon="book",
            color="purple"
        ))

        # Learning hours card
        hours = current.get("total_hours", 0.0)
        cards.append(OverviewCard(
            title="Learning Hours",
            value=f"{hours:.1f}h",
            raw_value=hours,
            icon="clock",
            color="orange"
        ))

        return cards

    def _calculate_change(
        self,
        current: float,
        previous: float
    ) -> Optional[float]:
        """Calculate percentage change"""
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100

    def _extract_metric_values(
        self,
        daily_data: List[Dict[str, Any]],
        metric: str
    ) -> List[float]:
        """Extract values for a specific metric from daily data"""
        metric_map = {
            "quality": "quality_score",
            "engagement": "exchanges",
            "sessions": "sessions",
            "duration": "duration_minutes",
            "positive_rate": "positive_rate"
        }

        key = metric_map.get(metric, metric)
        return [d.get(key, 0.0) for d in daily_data]

    async def _get_active_goals(self) -> List[DashboardGoal]:
        """Get active learning goals"""
        # Would integrate with a goal store
        return []

    async def _get_completed_goals(self) -> List[DashboardGoal]:
        """Get completed learning goals"""
        # Would integrate with a goal store
        return []

    async def _generate_goal_suggestions(self) -> List[Dict[str, Any]]:
        """Generate suggested goals based on activity"""
        suggestions = []

        # Simple suggestions based on current metrics
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
