"""
Learning Metrics Tracker - Phase 3 Learning Integration
========================================================

SPECIFICATION:
- Track search quality improvement over time
- Record which results users select
- Calculate learning effectiveness metrics
- Identify trends and patterns

ARCHITECTURE:
[User Search] -> [Results Shown] -> [User Selection] -> [Metrics Tracker]
                                                              |
                                                    [Improvement Analysis]

PATTERN: Observer pattern for metric collection
WHY: Track learning without coupling to business logic
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchMetric:
    """Individual search quality metric."""

    query: str
    timestamp: datetime
    results_count: int
    user_selected_index: Optional[int] = None
    selected_id: Optional[str] = None
    selected_similarity: Optional[float] = None
    time_to_select_ms: Optional[int] = None
    session_id: Optional[str] = None

    @property
    def mrr_score(self) -> float:
        """
        Mean Reciprocal Rank score.

        CONCEPT: How far down the list was the selected result?
        WHY: Lower rank = better search quality

        Returns 1.0 if first result, 0.5 if second, 0.33 if third, etc.
        Returns 0.0 if no selection or selection beyond results.
        """
        if self.user_selected_index is None:
            return 0.0
        if self.user_selected_index >= self.results_count:
            return 0.0
        return 1.0 / (self.user_selected_index + 1)

    @property
    def is_top_1(self) -> bool:
        """Check if user selected the top result."""
        return self.user_selected_index == 0

    @property
    def is_top_3(self) -> bool:
        """Check if user selected a top-3 result."""
        return (
            self.user_selected_index is not None and
            self.user_selected_index < 3
        )

    @property
    def is_top_5(self) -> bool:
        """Check if user selected a top-5 result."""
        return (
            self.user_selected_index is not None and
            self.user_selected_index < 5
        )


@dataclass
class ImprovementStats:
    """Statistics showing learning improvement."""

    period_start: datetime
    period_end: datetime
    total_searches: int = 0
    searches_with_selection: int = 0

    # MRR (Mean Reciprocal Rank) metrics
    avg_mrr: float = 0.0
    baseline_mrr: Optional[float] = None
    mrr_improvement: Optional[float] = None

    # Hit rate metrics
    top_1_hit_rate: float = 0.0
    top_3_hit_rate: float = 0.0
    top_5_hit_rate: float = 0.0

    # Baseline comparisons
    baseline_top_1: Optional[float] = None
    baseline_top_3: Optional[float] = None
    baseline_top_5: Optional[float] = None

    # Improvement percentages
    top_1_improvement: Optional[float] = None
    top_3_improvement: Optional[float] = None
    top_5_improvement: Optional[float] = None

    # Additional metrics
    avg_selected_similarity: float = 0.0
    avg_time_to_select_ms: float = 0.0

    def calculate_improvements(self) -> None:
        """Calculate improvement percentages vs baseline."""
        if self.baseline_mrr is not None:
            self.mrr_improvement = self.avg_mrr - self.baseline_mrr

        if self.baseline_top_1 is not None:
            self.top_1_improvement = self.top_1_hit_rate - self.baseline_top_1

        if self.baseline_top_3 is not None:
            self.top_3_improvement = self.top_3_hit_rate - self.baseline_top_3

        if self.baseline_top_5 is not None:
            self.top_5_improvement = self.top_5_hit_rate - self.baseline_top_5


class LearningMetrics:
    """
    Track vector learning improvement over time.

    CONCEPT: Measure search quality to quantify learning effectiveness
    WHY: "You can't improve what you don't measure"

    Metrics tracked:
    - MRR (Mean Reciprocal Rank): Average position of selected results
    - Hit@k: Percentage of times selected result is in top-k
    - Selection time: How quickly users find what they need
    - Similarity scores: Quality of retrieved results

    Features:
    - Rolling window analysis
    - Baseline comparison
    - Trend detection
    - Per-session tracking
    """

    def __init__(
        self,
        window_size: int = 1000,
        baseline_window: int = 100
    ):
        """
        Initialize learning metrics tracker.

        Args:
            window_size: Maximum metrics to keep in memory
            baseline_window: Number of initial searches for baseline
        """
        self.window_size = window_size
        self.baseline_window = baseline_window

        # Metric storage (circular buffer)
        self._metrics: deque[SearchMetric] = deque(maxlen=window_size)
        self._baseline_metrics: List[SearchMetric] = []
        self._baseline_calculated = False

        # Session tracking
        self._session_metrics: Dict[str, List[SearchMetric]] = defaultdict(list)

        # Statistics cache
        self._stats_cache: Optional[ImprovementStats] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Cache for 1 minute

    async def record_search_quality(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_selected_index: Optional[int] = None,
        session_id: Optional[str] = None,
        time_to_select_ms: Optional[int] = None
    ) -> None:
        """
        Record which result user selected from search results.

        CONCEPT: Implicit feedback through result selection
        WHY: User clicking result #3 means top 2 weren't good enough

        Args:
            query: Search query text
            results: List of search results shown to user
            user_selected_index: Index of result user selected (0-based)
            session_id: Optional session identifier
            time_to_select_ms: Time taken to make selection
        """
        try:
            # Extract metadata from selected result
            selected_id = None
            selected_similarity = None

            if user_selected_index is not None and user_selected_index < len(results):
                selected = results[user_selected_index]
                selected_id = selected.get('id')
                selected_similarity = selected.get('similarity') or selected.get('score')

            # Create metric
            metric = SearchMetric(
                query=query,
                timestamp=datetime.utcnow(),
                results_count=len(results),
                user_selected_index=user_selected_index,
                selected_id=selected_id,
                selected_similarity=selected_similarity,
                time_to_select_ms=time_to_select_ms,
                session_id=session_id
            )

            # Store in main buffer
            self._metrics.append(metric)

            # Store baseline if still collecting
            if not self._baseline_calculated and len(self._baseline_metrics) < self.baseline_window:
                self._baseline_metrics.append(metric)

                # Calculate baseline when we have enough data
                if len(self._baseline_metrics) >= self.baseline_window:
                    self._baseline_calculated = True
                    logger.info(
                        f"Baseline established with {len(self._baseline_metrics)} searches"
                    )

            # Track per-session
            if session_id:
                self._session_metrics[session_id].append(metric)

            # Invalidate cache
            self._stats_cache = None

            logger.debug(
                f"Recorded search metric: query='{query[:50]}...', "
                f"results={len(results)}, selected={user_selected_index}"
            )

        except Exception as e:
            logger.error(f"Failed to record search quality: {e}")

    async def get_improvement_stats(
        self,
        time_range: Optional[timedelta] = None,
        session_id: Optional[str] = None
    ) -> ImprovementStats:
        """
        Calculate search quality improvement statistics.

        ALGORITHM:
        1. Filter metrics by time range and session
        2. Calculate current performance metrics
        3. Compare to baseline metrics
        4. Compute improvement percentages

        Args:
            time_range: Optional time range to analyze (default: all data)
            session_id: Optional session to analyze (default: all sessions)

        Returns:
            ImprovementStats with detailed metrics
        """
        # Check cache
        if (
            self._stats_cache is not None and
            self._cache_timestamp is not None and
            time_range is None and
            session_id is None
        ):
            age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._stats_cache

        # Filter metrics
        metrics_to_analyze = self._filter_metrics(time_range, session_id)

        if not metrics_to_analyze:
            return ImprovementStats(
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow()
            )

        # Calculate period
        period_start = min(m.timestamp for m in metrics_to_analyze)
        period_end = max(m.timestamp for m in metrics_to_analyze)

        # Initialize stats
        stats = ImprovementStats(
            period_start=period_start,
            period_end=period_end,
            total_searches=len(metrics_to_analyze)
        )

        # Calculate metrics
        metrics_with_selection = [
            m for m in metrics_to_analyze
            if m.user_selected_index is not None
        ]
        stats.searches_with_selection = len(metrics_with_selection)

        if not metrics_with_selection:
            return stats

        # MRR calculation
        mrr_scores = [m.mrr_score for m in metrics_with_selection]
        stats.avg_mrr = sum(mrr_scores) / len(mrr_scores)

        # Hit rate calculations
        stats.top_1_hit_rate = sum(1 for m in metrics_with_selection if m.is_top_1) / len(metrics_with_selection)
        stats.top_3_hit_rate = sum(1 for m in metrics_with_selection if m.is_top_3) / len(metrics_with_selection)
        stats.top_5_hit_rate = sum(1 for m in metrics_with_selection if m.is_top_5) / len(metrics_with_selection)

        # Additional metrics
        similarities = [m.selected_similarity for m in metrics_with_selection if m.selected_similarity]
        if similarities:
            stats.avg_selected_similarity = sum(similarities) / len(similarities)

        times = [m.time_to_select_ms for m in metrics_with_selection if m.time_to_select_ms]
        if times:
            stats.avg_time_to_select_ms = sum(times) / len(times)

        # Calculate baseline if available
        if self._baseline_calculated and self._baseline_metrics:
            baseline_with_selection = [
                m for m in self._baseline_metrics
                if m.user_selected_index is not None
            ]

            if baseline_with_selection:
                # Baseline MRR
                baseline_mrr_scores = [m.mrr_score for m in baseline_with_selection]
                stats.baseline_mrr = sum(baseline_mrr_scores) / len(baseline_mrr_scores)

                # Baseline hit rates
                stats.baseline_top_1 = sum(
                    1 for m in baseline_with_selection if m.is_top_1
                ) / len(baseline_with_selection)

                stats.baseline_top_3 = sum(
                    1 for m in baseline_with_selection if m.is_top_3
                ) / len(baseline_with_selection)

                stats.baseline_top_5 = sum(
                    1 for m in baseline_with_selection if m.is_top_5
                ) / len(baseline_with_selection)

                # Calculate improvements
                stats.calculate_improvements()

        # Cache results if analyzing all data
        if time_range is None and session_id is None:
            self._stats_cache = stats
            self._cache_timestamp = datetime.utcnow()

        return stats

    def _filter_metrics(
        self,
        time_range: Optional[timedelta],
        session_id: Optional[str]
    ) -> List[SearchMetric]:
        """Filter metrics by time range and session."""
        metrics = list(self._metrics)

        # Filter by session
        if session_id and session_id in self._session_metrics:
            metrics = self._session_metrics[session_id]

        # Filter by time range
        if time_range:
            cutoff = datetime.utcnow() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff]

        return metrics

    async def get_session_stats(self, session_id: str) -> Optional[ImprovementStats]:
        """Get improvement statistics for a specific session."""
        if session_id not in self._session_metrics:
            return None

        return await self.get_improvement_stats(session_id=session_id)

    async def get_trend_analysis(
        self,
        bucket_size: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """
        Analyze trends over time using bucketed metrics.

        CONCEPT: Time-series analysis of learning improvement
        WHY: Detect if learning is actually improving over time

        Args:
            bucket_size: Size of time buckets for analysis

        Returns:
            Dictionary with trend data:
            - buckets: List of time-bucketed metrics
            - trend: "improving", "stable", or "declining"
            - trend_slope: Numerical trend slope
        """
        if len(self._metrics) < 10:
            return {
                "status": "insufficient_data",
                "total_metrics": len(self._metrics),
                "required": 10
            }

        # Create time buckets
        metrics_list = list(self._metrics)
        start_time = min(m.timestamp for m in metrics_list)
        end_time = max(m.timestamp for m in metrics_list)

        buckets = []
        current_time = start_time

        while current_time < end_time:
            bucket_end = current_time + bucket_size
            bucket_metrics = [
                m for m in metrics_list
                if current_time <= m.timestamp < bucket_end
            ]

            if bucket_metrics:
                # Calculate MRR for this bucket
                metrics_with_selection = [
                    m for m in bucket_metrics
                    if m.user_selected_index is not None
                ]

                if metrics_with_selection:
                    mrr = sum(m.mrr_score for m in metrics_with_selection) / len(metrics_with_selection)
                    buckets.append({
                        "timestamp": current_time.isoformat(),
                        "mrr": round(mrr, 3),
                        "count": len(bucket_metrics)
                    })

            current_time = bucket_end

        # Calculate trend
        if len(buckets) >= 3:
            mrr_values = [b["mrr"] for b in buckets]
            trend_slope = self._calculate_trend_slope(mrr_values)

            if trend_slope > 0.05:
                trend = "improving"
            elif trend_slope < -0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"
            trend_slope = 0.0

        return {
            "buckets": buckets,
            "trend": trend,
            "trend_slope": round(trend_slope, 4),
            "bucket_size_seconds": bucket_size.total_seconds(),
            "total_buckets": len(buckets)
        }

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate linear trend slope using least squares."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_values = list(range(n))

        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        # Calculate slope
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of learning metrics."""
        return {
            "total_searches": len(self._metrics),
            "baseline_searches": len(self._baseline_metrics),
            "baseline_established": self._baseline_calculated,
            "sessions_tracked": len(self._session_metrics),
            "window_size": self.window_size,
            "oldest_metric": min(m.timestamp for m in self._metrics).isoformat() if self._metrics else None,
            "newest_metric": max(m.timestamp for m in self._metrics).isoformat() if self._metrics else None
        }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self._metrics.clear()
        self._baseline_metrics.clear()
        self._baseline_calculated = False
        self._session_metrics.clear()
        self._stats_cache = None
        self._cache_timestamp = None


# Convenience factory
def create_learning_metrics(
    window_size: int = 1000,
    baseline_window: int = 100
) -> LearningMetrics:
    """
    Factory function for creating a LearningMetrics tracker.

    Args:
        window_size: Maximum metrics to keep in memory
        baseline_window: Number of initial searches for baseline

    Returns:
        Configured LearningMetrics instance
    """
    return LearningMetrics(window_size, baseline_window)


# Singleton instance holder
_metrics_instance: Optional[LearningMetrics] = None


def get_learning_metrics() -> LearningMetrics:
    """
    Get or create the singleton LearningMetrics instance.

    PATTERN: Singleton for global metrics collection
    WHY: All components should contribute to the same metrics pool

    Returns:
        Global LearningMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = create_learning_metrics()
    return _metrics_instance
