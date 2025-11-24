"""
Dashboard Charts
================

Chart data generation and formatting for dashboard visualizations.

PATTERN: Builder pattern for chart construction
WHY: Isolate chart data transformation from service logic
SPARC: Clean separation of chart concerns with Chart.js compatibility
"""

import statistics
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional

from app.logger import db_logger
from app.analytics.dashboard_models import (
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
)
from app.analytics.chart_data import (
    ChartJSData,
    ChartDataFormatter,
    HeatmapData,
)


class DashboardChartBuilder:
    """
    Build chart data for dashboard visualizations.

    PATTERN: Builder for chart construction
    WHY: Centralize all chart data transformation logic

    Provides:
    - Progress chart data (line/area charts)
    - Trend chart data (multi-metric)
    - Topic breakdown (pie/doughnut)
    - Activity heatmap (GitHub-style calendar)
    - Chart.js format conversion
    """

    # Default colors for metrics
    METRIC_COLORS = {
        "quality": "#4F46E5",
        "engagement": "#10B981",
        "sessions": "#F59E0B",
        "duration": "#EF4444",
        "positive_rate": "#8B5CF6"
    }

    METRIC_DISPLAY_NAMES = {
        "quality": "Quality Score",
        "engagement": "Engagement",
        "sessions": "Sessions",
        "duration": "Duration",
        "positive_rate": "Positive Rate"
    }

    def __init__(self):
        """Initialize chart builder."""
        pass

    def build_progress_chart(
        self,
        period: str,
        daily_data: List[Dict[str, Any]],
        start_date: date,
        end_date: date
    ) -> ProgressChartResponse:
        """
        Build progress chart response from daily data.

        Args:
            period: Time period (week, month, year)
            daily_data: List of daily data dictionaries
            start_date: Chart start date
            end_date: Chart end date

        Returns:
            ProgressChartResponse with chart-ready data
        """
        # Build data points
        data_points = []
        total_sessions = 0
        total_exchanges = 0
        total_duration = 0.0
        quality_scores = []

        for day_data in daily_data:
            data_points.append(ProgressDataPoint(
                date=day_data.get("date", ""),
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
        improvement_rate = self._calculate_improvement_rate(quality_scores)

        # Find best day
        best_day = None
        if daily_data:
            best = max(daily_data, key=lambda d: d.get("sessions", 0))
            if best.get("sessions", 0) > 0:
                best_day = best.get("date")

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

        return ProgressChartResponse(
            period=period,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            data_points=data_points,
            summary=summary,
            chart_config=chart_config,
            generated_at=datetime.utcnow()
        )

    def build_trend_chart(
        self,
        metrics: List[str],
        days: int,
        daily_data: List[Dict[str, Any]]
    ) -> TrendChartResponse:
        """
        Build trend chart response for multiple metrics.

        Args:
            metrics: List of metric names to include
            days: Number of days in the trend
            daily_data: List of daily data dictionaries

        Returns:
            TrendChartResponse with multi-metric trends
        """
        # Build labels
        labels = [d.get("date", "") for d in daily_data]

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
                display_name=self.METRIC_DISPLAY_NAMES.get(metric, metric.title()),
                values=values,
                labels=labels,
                current_value=current_value,
                change=change,
                trend=trend_direction,
                color=self.METRIC_COLORS.get(metric, "#4F46E5")
            )

        return TrendChartResponse(
            days=days,
            metrics=trend_metrics,
            labels=labels,
            generated_at=datetime.utcnow()
        )

    def build_topic_breakdown(
        self,
        topic_distribution: Dict[str, Any],
        days: int,
        max_topics: int = 10
    ) -> TopicBreakdownResponse:
        """
        Build topic breakdown chart response.

        Args:
            topic_distribution: Dictionary of topic stats
            days: Analysis period in days
            max_topics: Maximum topics to display

        Returns:
            TopicBreakdownResponse with topic statistics
        """
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

        # Build pie chart data
        chart_data = ChartDataFormatter.format_topic_distribution(
            {t.topic_name: t.session_count for t in topics},
            max_topics=max_topics
        )

        return TopicBreakdownResponse(
            total_topics=len(topics),
            topics=topics[:max_topics],
            top_topics=top_topics,
            emerging_topics=[],
            chart_data=chart_data.to_dict(),
            generated_at=datetime.utcnow()
        )

    def build_activity_heatmap(
        self,
        daily_counts: Dict[str, int],
        year: int
    ) -> ActivityHeatmapResponse:
        """
        Build activity heatmap (GitHub-style calendar).

        Args:
            daily_counts: Dictionary mapping date strings to session counts
            year: Year for the heatmap

        Returns:
            ActivityHeatmapResponse with calendar data
        """
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

        return ActivityHeatmapResponse(
            year=year,
            total_active_days=total_active_days,
            total_sessions=int(total_sessions),
            max_daily_sessions=max_daily,
            weeks=weeks,
            month_labels=month_labels,
            generated_at=datetime.utcnow()
        )

    def build_chartjs_progress_data(
        self,
        data_points: List[ProgressDataPoint]
    ) -> ChartJSData:
        """
        Convert progress data to Chart.js format.

        Args:
            data_points: List of progress data points

        Returns:
            ChartJSData ready for Chart.js
        """
        formatted_data = [
            {"date": dp.date, "value": dp.sessions}
            for dp in data_points
        ]

        chart_data = ChartDataFormatter.format_time_series(
            formatted_data,
            label="Sessions",
            color="#4F46E5"
        )

        # Add quality as second dataset
        quality_points = [
            {"date": dp.date, "value": dp.quality_score * 100}
            for dp in data_points
        ]
        chart_data.add_dataset(
            "Quality Score",
            [p["value"] for p in quality_points],
            "#10B981"
        )

        return chart_data

    def _calculate_improvement_rate(self, quality_scores: List[float]) -> float:
        """Calculate improvement rate from quality scores."""
        if len(quality_scores) < 2:
            return 0.0

        first_half = quality_scores[:len(quality_scores)//2]
        second_half = quality_scores[len(quality_scores)//2:]

        if not first_half or not second_half:
            return 0.0

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if first_avg > 0:
            return ((second_avg - first_avg) / first_avg) * 100

        return 0.0

    def _extract_metric_values(
        self,
        daily_data: List[Dict[str, Any]],
        metric: str
    ) -> List[float]:
        """Extract values for a specific metric from daily data."""
        metric_map = {
            "quality": "quality_score",
            "engagement": "exchanges",
            "sessions": "sessions",
            "duration": "duration_minutes",
            "positive_rate": "positive_rate"
        }

        key = metric_map.get(metric, metric)
        return [d.get(key, 0.0) for d in daily_data]


# Module-level instance
dashboard_chart_builder = DashboardChartBuilder()
