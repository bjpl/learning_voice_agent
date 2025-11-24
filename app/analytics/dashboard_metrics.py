"""
Dashboard Metrics
=================

KPI calculations and metrics aggregation for dashboard displays.

PATTERN: Strategy pattern for metric calculations
WHY: Isolate metric logic from presentation and data fetching
SPARC: Clean separation of calculation concerns
"""

import statistics
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple

from app.logger import db_logger
from app.analytics.dashboard_models import (
    OverviewCard,
    LearningStreak,
    QuickStats,
    TrendDirection,
)


class DashboardMetricsCalculator:
    """
    Calculate dashboard metrics and KPIs.

    PATTERN: Strategy for metric calculations
    WHY: Centralize all metric calculation logic

    Provides:
    - Overview card generation with trends
    - Quick stats aggregation
    - Streak calculations
    - Trend direction analysis
    - Change percentage calculations
    """

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def build_overview_cards(
        self,
        current_metrics: Dict[str, Any],
        previous_metrics: Dict[str, Any]
    ) -> List[OverviewCard]:
        """
        Build overview KPI cards.

        Args:
            current_metrics: Current period metrics
            previous_metrics: Previous period metrics for comparison

        Returns:
            List of OverviewCard instances
        """
        cards = []

        # Sessions card
        sessions = current_metrics.get("sessions_count", 0)
        prev_sessions = previous_metrics.get("sessions", 0)
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
        quality = current_metrics.get("avg_quality_score", 0.0)
        prev_quality = previous_metrics.get("avg_quality", 0.0)
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
        topics = current_metrics.get("topics_explored", 0)
        cards.append(OverviewCard(
            title="Topics Explored",
            value=str(topics),
            raw_value=float(topics),
            icon="book",
            color="purple"
        ))

        # Learning hours card
        hours = current_metrics.get("total_hours", 0.0)
        cards.append(OverviewCard(
            title="Learning Hours",
            value=f"{hours:.1f}h",
            raw_value=hours,
            icon="clock",
            color="orange"
        ))

        return cards

    def build_quick_stats(self, metrics: Dict[str, Any]) -> QuickStats:
        """
        Build quick stats summary.

        Args:
            metrics: Overall metrics dictionary

        Returns:
            QuickStats instance
        """
        return QuickStats(
            total_exchanges=metrics.get("total_exchanges", 0),
            total_hours=metrics.get("total_hours", 0.0),
            learning_velocity=metrics.get("learning_velocity", 0.0),
            insights_generated=metrics.get("insights_generated", 0),
            avg_session_duration=metrics.get("avg_session_duration", 0.0),
            topics_explored=metrics.get("topics_explored", 0)
        )

    def calculate_streak_from_sessions(
        self,
        sessions: List[Any],
        start_date: date,
        end_date: date
    ) -> LearningStreak:
        """
        Calculate learning streak from session data.

        Args:
            sessions: List of session objects with start_time attribute
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            LearningStreak instance
        """
        streak = LearningStreak(
            current_streak=0,
            longest_streak=0,
            last_active_date=None,
            streak_start_date=None,
            is_active_today=False
        )

        if not sessions:
            return streak

        # Group by date
        dates_with_sessions = set()
        for s in sessions:
            if hasattr(s, 'start_time'):
                dates_with_sessions.add(s.start_time.date())

        if not dates_with_sessions:
            return streak

        # Calculate current streak (working backwards from end_date)
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
        streak.longest_streak = current_streak  # Simplified - would need full history
        streak.is_active_today = end_date in dates_with_sessions
        streak.last_active_date = max(dates_with_sessions) if dates_with_sessions else None
        streak.streak_start_date = streak_start

        return streak

    def calculate_improvement_rate(
        self,
        quality_scores: List[float]
    ) -> float:
        """
        Calculate improvement rate from quality scores.

        Args:
            quality_scores: List of quality scores in chronological order

        Returns:
            Improvement rate as percentage
        """
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

    def extract_metric_values(
        self,
        daily_data: List[Dict[str, Any]],
        metric: str
    ) -> List[float]:
        """
        Extract values for a specific metric from daily data.

        Args:
            daily_data: List of daily data dictionaries
            metric: Metric name to extract

        Returns:
            List of metric values
        """
        metric_map = {
            "quality": "quality_score",
            "engagement": "exchanges",
            "sessions": "sessions",
            "duration": "duration_minutes",
            "positive_rate": "positive_rate"
        }

        key = metric_map.get(metric, metric)
        return [d.get(key, 0.0) for d in daily_data]

    def calculate_trend_direction(
        self,
        values: List[float],
        threshold: float = 0.05
    ) -> Tuple[str, float]:
        """
        Calculate trend direction from value series.

        Args:
            values: List of values in chronological order
            threshold: Minimum change threshold for trend detection

        Returns:
            Tuple of (direction, change_percentage)
        """
        if len(values) < 2:
            return ("stable", 0.0)

        # Compare first and second half averages
        mid = len(values) // 2
        first_half = statistics.mean(values[:mid]) if values[:mid] else 0
        second_half = statistics.mean(values[mid:]) if values[mid:] else 0

        if first_half == 0:
            if second_half > 0:
                return ("up", 100.0)
            return ("stable", 0.0)

        change = (second_half - first_half) / first_half

        if change > threshold:
            return ("up", change * 100)
        elif change < -threshold:
            return ("down", change * 100)
        else:
            return ("stable", change * 100)

    def _calculate_change(
        self,
        current: float,
        previous: float
    ) -> Optional[float]:
        """
        Calculate percentage change.

        Args:
            current: Current value
            previous: Previous value

        Returns:
            Percentage change or None if previous is 0
        """
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100

    def aggregate_daily_metrics(
        self,
        daily_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics from daily data.

        Args:
            daily_data: List of daily data dictionaries

        Returns:
            Aggregated metrics dictionary
        """
        if not daily_data:
            return {
                "total_sessions": 0,
                "total_exchanges": 0,
                "total_duration": 0.0,
                "avg_quality": 0.0,
                "active_days": 0
            }

        total_sessions = sum(d.get("sessions", 0) for d in daily_data)
        total_exchanges = sum(d.get("exchanges", 0) for d in daily_data)
        total_duration = sum(d.get("duration_minutes", 0) for d in daily_data)

        quality_scores = [
            d.get("quality_score", 0) for d in daily_data
            if d.get("quality_score", 0) > 0
        ]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0

        active_days = sum(1 for d in daily_data if d.get("sessions", 0) > 0)

        return {
            "total_sessions": total_sessions,
            "total_exchanges": total_exchanges,
            "total_duration": total_duration,
            "avg_quality": avg_quality,
            "active_days": active_days
        }


# Module-level instance
dashboard_metrics_calculator = DashboardMetricsCalculator()
