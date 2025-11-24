"""
Metrics Aggregator
PATTERN: Batch processing with efficient aggregation
WHY: Generate dashboard-ready metrics from raw data
SPARC: Systematic metrics collection with multiple time windows
"""
import uuid
import statistics
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum

from app.logger import db_logger
from app.learning.config import LearningConfig, learning_config
try:
    from app.learning.store import LearningStore
except ImportError:
    LearningStore = None
try:
    from app.learning.stores import (
        FeedbackStore, QualityStore,
        feedback_store, quality_store,
        SessionData, QualityScore, FeedbackData, FeedbackType
    )
except ImportError:
    # Provide fallback types if stores module has issues
    FeedbackStore = None
    QualityStore = None
    feedback_store = None
    quality_store = None
    SessionData = None
    QualityScore = None
    FeedbackData = None
    FeedbackType = None


class AggregationInterval(str, Enum):
    """Time intervals for aggregation"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a time period"""
    period_start: datetime
    period_end: datetime
    interval: AggregationInterval

    # Session metrics
    total_sessions: int = 0
    total_exchanges: int = 0
    avg_session_duration_minutes: float = 0.0
    avg_exchanges_per_session: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_by_dimension: Dict[str, float] = field(default_factory=dict)
    quality_std_dev: float = 0.0

    # Feedback metrics
    total_feedback: int = 0
    positive_feedback_rate: float = 0.0
    correction_rate: float = 0.0
    clarification_rate: float = 0.0

    # Engagement metrics
    active_users: int = 0
    returning_users: int = 0
    peak_hour: int = 0

    # Topic metrics
    active_topics: int = 0
    top_topics: List[str] = field(default_factory=list)

    # Computed at aggregation time
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Data formatted for dashboard display"""
    summary: Dict[str, Any]
    trends: List[Dict[str, Any]]
    quality_breakdown: Dict[str, Any]
    engagement_stats: Dict[str, Any]
    feedback_analysis: Dict[str, Any]
    topic_distribution: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class MetricsStore(LearningStore):
    """
    Store for aggregated metrics
    PATTERN: Time-series storage with efficient queries
    WHY: Support dashboard queries and historical analysis
    """

    async def save_aggregated_metrics(
        self,
        metrics: AggregatedMetrics
    ) -> str:
        """Save aggregated metrics"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                metrics_dict = asdict(metrics)
                metrics_dict['period_start'] = metrics.period_start.isoformat()
                metrics_dict['period_end'] = metrics.period_end.isoformat()
                metrics_dict['generated_at'] = metrics.generated_at.isoformat()
                metrics_dict['interval'] = metrics.interval.value
                metrics_dict['quality_by_dimension'] = json.dumps(metrics.quality_by_dimension)
                metrics_dict['top_topics'] = json.dumps(metrics.top_topics)
                metrics_dict['metadata'] = json.dumps(metrics.metadata)

                # Store as individual metric rows for flexibility
                metric_date = metrics.period_start.date()
                interval_type = metrics.interval.value

                metric_values = {
                    'total_sessions': metrics.total_sessions,
                    'total_exchanges': metrics.total_exchanges,
                    'avg_session_duration': metrics.avg_session_duration_minutes,
                    'avg_exchanges_per_session': metrics.avg_exchanges_per_session,
                    'avg_quality_score': metrics.avg_quality_score,
                    'quality_std_dev': metrics.quality_std_dev,
                    'total_feedback': metrics.total_feedback,
                    'positive_feedback_rate': metrics.positive_feedback_rate,
                    'correction_rate': metrics.correction_rate,
                    'clarification_rate': metrics.clarification_rate,
                    'active_users': metrics.active_users,
                    'active_topics': metrics.active_topics,
                    'peak_hour': metrics.peak_hour
                }

                for metric_name, metric_value in metric_values.items():
                    await db.execute("""
                        INSERT OR REPLACE INTO aggregated_metrics
                        (metric_date, interval_type, metric_name, metric_value, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        metric_date.isoformat(),
                        interval_type,
                        metric_name,
                        metric_value,
                        json.dumps({'generated_at': metrics.generated_at.isoformat()})
                    ))

                await db.commit()

            return f"{interval_type}_{metric_date.isoformat()}"

        except Exception as e:
            db_logger.error("save_aggregated_metrics_failed", error=str(e), exc_info=True)
            raise

    async def get_metrics_for_period(
        self,
        start_date: date,
        end_date: date,
        interval: AggregationInterval
    ) -> List[Dict[str, Any]]:
        """Get aggregated metrics for a date range"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT metric_date, metric_name, metric_value
                    FROM aggregated_metrics
                    WHERE metric_date >= ? AND metric_date < ?
                    AND interval_type = ?
                    ORDER BY metric_date ASC
                """, (start_date.isoformat(), end_date.isoformat(), interval.value))

                rows = await cursor.fetchall()

            # Group by date
            by_date = defaultdict(dict)
            for row in rows:
                by_date[row['metric_date']][row['metric_name']] = row['metric_value']

            return [
                {'date': d, **metrics}
                for d, metrics in sorted(by_date.items())
            ]

        except Exception as e:
            db_logger.error("get_metrics_for_period_failed", error=str(e))
            return []

    async def get_latest_metrics(
        self,
        interval: AggregationInterval
    ) -> Optional[Dict[str, Any]]:
        """Get most recent aggregated metrics"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT metric_date, metric_name, metric_value
                    FROM aggregated_metrics
                    WHERE interval_type = ?
                    AND metric_date = (
                        SELECT MAX(metric_date)
                        FROM aggregated_metrics
                        WHERE interval_type = ?
                    )
                """, (interval.value, interval.value))

                rows = await cursor.fetchall()

            if not rows:
                return None

            result = {'date': rows[0]['metric_date']}
            for row in rows:
                result[row['metric_name']] = row['metric_value']

            return result

        except Exception as e:
            db_logger.error("get_latest_metrics_failed", error=str(e))
            return None


class MetricsAggregator:
    """
    PATTERN: Batch processing metrics aggregator
    WHY: Efficiently compute and store dashboard metrics

    USAGE:
        aggregator = MetricsAggregator()
        await aggregator.initialize()

        # Run hourly aggregation
        await aggregator.aggregate_hourly()

        # Run daily aggregation
        await aggregator.aggregate_daily()

        # Export dashboard data
        dashboard = await aggregator.export_dashboard_data(days=30)
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback: Optional[FeedbackStore] = None,
        quality: Optional[QualityStore] = None
    ):
        """
        Initialize metrics aggregator

        Args:
            config: Learning configuration
            feedback: Feedback store instance
            quality: Quality store instance
        """
        self.config = config or learning_config
        self.feedback = feedback or feedback_store
        self.quality = quality or quality_store
        self.metrics_store = MetricsStore()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize aggregator and dependencies"""
        if self._initialized:
            return

        try:
            await self.feedback.initialize()
            await self.quality.initialize()
            await self.metrics_store.initialize()

            self._initialized = True
            db_logger.info("metrics_aggregator_initialized")

        except Exception as e:
            db_logger.error(
                "metrics_aggregator_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def aggregate_hourly(
        self,
        target_hour: Optional[datetime] = None
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for an hour

        PATTERN: Hourly batch processing
        WHY: Granular metrics for real-time dashboards

        Args:
            target_hour: Hour to aggregate (defaults to previous hour)

        Returns:
            Aggregated metrics for the hour
        """
        if not self._initialized:
            await self.initialize()

        try:
            if target_hour is None:
                now = datetime.utcnow()
                target_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

            period_start = target_hour
            period_end = target_hour + timedelta(hours=1)

            db_logger.info(
                "aggregating_hourly_metrics",
                period_start=period_start.isoformat()
            )

            metrics = await self._aggregate_period(
                period_start,
                period_end,
                AggregationInterval.HOURLY
            )

            await self.metrics_store.save_aggregated_metrics(metrics)

            db_logger.info(
                "hourly_aggregation_complete",
                sessions=metrics.total_sessions,
                quality=metrics.avg_quality_score
            )

            return metrics

        except Exception as e:
            db_logger.error(
                "aggregate_hourly_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def aggregate_daily(
        self,
        target_date: Optional[date] = None
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for a day

        PATTERN: Daily batch processing
        WHY: Summary metrics for daily reports

        Args:
            target_date: Date to aggregate (defaults to yesterday)

        Returns:
            Aggregated metrics for the day
        """
        if not self._initialized:
            await self.initialize()

        try:
            if target_date is None:
                target_date = date.today() - timedelta(days=1)

            period_start = datetime.combine(target_date, datetime.min.time())
            period_end = period_start + timedelta(days=1)

            db_logger.info(
                "aggregating_daily_metrics",
                date=target_date.isoformat()
            )

            metrics = await self._aggregate_period(
                period_start,
                period_end,
                AggregationInterval.DAILY
            )

            await self.metrics_store.save_aggregated_metrics(metrics)

            db_logger.info(
                "daily_aggregation_complete",
                date=target_date.isoformat(),
                sessions=metrics.total_sessions,
                quality=metrics.avg_quality_score
            )

            return metrics

        except Exception as e:
            db_logger.error(
                "aggregate_daily_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def aggregate_weekly(
        self,
        week_start: Optional[date] = None
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for a week

        Args:
            week_start: Start of week (defaults to last week's Monday)

        Returns:
            Aggregated metrics for the week
        """
        if not self._initialized:
            await self.initialize()

        try:
            if week_start is None:
                today = date.today()
                # Get last week's Monday
                days_since_monday = today.weekday()
                week_start = today - timedelta(days=days_since_monday + 7)

            period_start = datetime.combine(week_start, datetime.min.time())
            period_end = period_start + timedelta(weeks=1)

            db_logger.info(
                "aggregating_weekly_metrics",
                week_start=week_start.isoformat()
            )

            metrics = await self._aggregate_period(
                period_start,
                period_end,
                AggregationInterval.WEEKLY
            )

            await self.metrics_store.save_aggregated_metrics(metrics)

            return metrics

        except Exception as e:
            db_logger.error("aggregate_weekly_failed", error=str(e), exc_info=True)
            raise

    async def export_dashboard_data(
        self,
        days: int = 30
    ) -> DashboardData:
        """
        Export data formatted for dashboard

        PATTERN: Dashboard-optimized data export
        WHY: Provide ready-to-display metrics

        Args:
            days: Number of days to include

        Returns:
            DashboardData with all dashboard metrics
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("exporting_dashboard_data", days=days)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get stored metrics
            stored_metrics = await self.metrics_store.get_metrics_for_period(
                start_date, end_date, AggregationInterval.DAILY
            )

            # Get latest metrics for summary
            latest = await self.metrics_store.get_latest_metrics(AggregationInterval.DAILY)

            # Get raw data for real-time calculations
            sessions = await self.feedback.get_sessions_in_range(start_date, end_date)
            quality_scores = await self.quality.get_scores_in_range(start_date, end_date)
            feedback_list = await self.feedback.get_feedback_in_range(start_date, end_date)

            # Build dashboard data
            summary = self._build_summary(latest, sessions, quality_scores)
            trends = self._build_trends(stored_metrics)
            quality_breakdown = self._build_quality_breakdown(quality_scores)
            engagement_stats = self._build_engagement_stats(sessions)
            feedback_analysis = self._build_feedback_analysis(feedback_list)
            topic_distribution = self._build_topic_distribution(sessions)

            dashboard = DashboardData(
                summary=summary,
                trends=trends,
                quality_breakdown=quality_breakdown,
                engagement_stats=engagement_stats,
                feedback_analysis=feedback_analysis,
                topic_distribution=topic_distribution
            )

            db_logger.info("dashboard_data_exported", days=days)

            return dashboard

        except Exception as e:
            db_logger.error(
                "export_dashboard_data_failed",
                error=str(e),
                exc_info=True
            )
            # Return empty dashboard on failure
            return DashboardData(
                summary={},
                trends=[],
                quality_breakdown={},
                engagement_stats={},
                feedback_analysis={},
                topic_distribution={}
            )

    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics for current period

        PATTERN: Live metrics calculation
        WHY: Support real-time dashboard updates

        Returns:
            Dictionary of current metrics
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get data for today
            today = date.today()
            tomorrow = today + timedelta(days=1)

            sessions = await self.feedback.get_sessions_in_range(today, tomorrow)
            quality_scores = await self.quality.get_scores_in_range(today, tomorrow)
            feedback_list = await self.feedback.get_feedback_in_range(today, tomorrow)

            # Calculate real-time metrics
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'sessions_today': len(sessions),
                'exchanges_today': sum(s.exchange_count for s in sessions),
                'avg_quality_today': (
                    statistics.mean([q.composite for q in quality_scores])
                    if quality_scores else 0
                ),
                'feedback_count': len(feedback_list),
                'positive_rate': self._calculate_positive_rate(feedback_list),
                'active_now': self._count_active_sessions(sessions)
            }

            return metrics

        except Exception as e:
            db_logger.error("get_real_time_metrics_failed", error=str(e))
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }

    async def run_batch_aggregation(
        self,
        days_back: int = 7
    ) -> Dict[str, int]:
        """
        Run batch aggregation for multiple days

        PATTERN: Backfill aggregation
        WHY: Populate missing historical metrics

        Args:
            days_back: Number of days to backfill

        Returns:
            Statistics about the batch run
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("starting_batch_aggregation", days_back=days_back)

            stats = {
                'daily_aggregated': 0,
                'errors': 0
            }

            for i in range(1, days_back + 1):
                target_date = date.today() - timedelta(days=i)
                try:
                    await self.aggregate_daily(target_date)
                    stats['daily_aggregated'] += 1
                except Exception as e:
                    db_logger.error(
                        "batch_aggregation_day_failed",
                        date=target_date.isoformat(),
                        error=str(e)
                    )
                    stats['errors'] += 1

            db_logger.info(
                "batch_aggregation_complete",
                **stats
            )

            return stats

        except Exception as e:
            db_logger.error("run_batch_aggregation_failed", error=str(e), exc_info=True)
            raise

    # =========================================================================
    # Private Aggregation Methods
    # =========================================================================

    async def _aggregate_period(
        self,
        period_start: datetime,
        period_end: datetime,
        interval: AggregationInterval
    ) -> AggregatedMetrics:
        """Aggregate metrics for a time period"""
        # Get raw data
        sessions = await self.feedback.get_sessions_in_range(
            period_start.date(),
            period_end.date() + timedelta(days=1)  # Include full period
        )
        # Filter to exact period
        sessions = [
            s for s in sessions
            if period_start <= s.start_time < period_end
        ]

        quality_scores = await self.quality.get_scores_in_range(
            period_start.date(),
            period_end.date() + timedelta(days=1)
        )
        quality_scores = [
            q for q in quality_scores
            if period_start <= q.timestamp < period_end
        ]

        feedback_list = await self.feedback.get_feedback_in_range(
            period_start.date(),
            period_end.date() + timedelta(days=1)
        )
        feedback_list = [
            f for f in feedback_list
            if period_start <= f.timestamp < period_end
        ]

        # Calculate metrics
        session_metrics = self._calculate_session_metrics(sessions)
        quality_metrics = self._calculate_quality_metrics(quality_scores)
        feedback_metrics = self._calculate_feedback_metrics(feedback_list)
        engagement_metrics = self._calculate_engagement_metrics(sessions)
        topic_metrics = self._calculate_topic_metrics(sessions)

        return AggregatedMetrics(
            period_start=period_start,
            period_end=period_end,
            interval=interval,
            **session_metrics,
            **quality_metrics,
            **feedback_metrics,
            **engagement_metrics,
            **topic_metrics
        )

    def _calculate_session_metrics(
        self,
        sessions: List[SessionData]
    ) -> Dict[str, Any]:
        """Calculate session-related metrics"""
        if not sessions:
            return {
                'total_sessions': 0,
                'total_exchanges': 0,
                'avg_session_duration_minutes': 0.0,
                'avg_exchanges_per_session': 0.0
            }

        durations = [s.duration / 60 for s in sessions if s.duration > 0]
        exchanges = [s.exchange_count for s in sessions]

        return {
            'total_sessions': len(sessions),
            'total_exchanges': sum(exchanges),
            'avg_session_duration_minutes': statistics.mean(durations) if durations else 0.0,
            'avg_exchanges_per_session': statistics.mean(exchanges) if exchanges else 0.0
        }

    def _calculate_quality_metrics(
        self,
        quality_scores: List[QualityScore]
    ) -> Dict[str, Any]:
        """Calculate quality-related metrics"""
        if not quality_scores:
            return {
                'avg_quality_score': 0.0,
                'quality_by_dimension': {},
                'quality_std_dev': 0.0
            }

        composites = [q.composite for q in quality_scores]

        return {
            'avg_quality_score': statistics.mean(composites),
            'quality_by_dimension': {
                'relevance': statistics.mean([q.relevance for q in quality_scores]),
                'helpfulness': statistics.mean([q.helpfulness for q in quality_scores]),
                'accuracy': statistics.mean([q.accuracy for q in quality_scores]),
                'clarity': statistics.mean([q.clarity for q in quality_scores]),
                'completeness': statistics.mean([q.completeness for q in quality_scores])
            },
            'quality_std_dev': statistics.stdev(composites) if len(composites) > 1 else 0.0
        }

    def _calculate_feedback_metrics(
        self,
        feedback_list: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Calculate feedback-related metrics"""
        if not feedback_list:
            return {
                'total_feedback': 0,
                'positive_feedback_rate': 0.5,
                'correction_rate': 0.0,
                'clarification_rate': 0.0
            }

        total = len(feedback_list)
        positive = sum(
            1 for f in feedback_list
            if f.feedback_type in [FeedbackType.EXPLICIT_POSITIVE, FeedbackType.IMPLICIT_POSITIVE]
        )
        negative = sum(
            1 for f in feedback_list
            if f.feedback_type in [FeedbackType.EXPLICIT_NEGATIVE, FeedbackType.IMPLICIT_NEGATIVE]
        )
        corrections = sum(
            1 for f in feedback_list
            if f.feedback_type == FeedbackType.CORRECTION
        )
        clarifications = sum(
            1 for f in feedback_list
            if f.feedback_type == FeedbackType.CLARIFICATION
        )

        sentiment_total = positive + negative

        return {
            'total_feedback': total,
            'positive_feedback_rate': positive / sentiment_total if sentiment_total > 0 else 0.5,
            'correction_rate': corrections / total if total > 0 else 0.0,
            'clarification_rate': clarifications / total if total > 0 else 0.0
        }

    def _calculate_engagement_metrics(
        self,
        sessions: List[SessionData]
    ) -> Dict[str, Any]:
        """Calculate engagement-related metrics"""
        if not sessions:
            return {
                'active_users': 0,
                'returning_users': 0,
                'peak_hour': 0
            }

        # Unique sessions (proxy for users)
        unique_sessions = len(set(s.session_id for s in sessions))

        # Peak hour
        hour_counts = defaultdict(int)
        for s in sessions:
            hour_counts[s.start_time.hour] += 1

        peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h]) if hour_counts else 0

        return {
            'active_users': unique_sessions,
            'returning_users': 0,  # Would need cross-day analysis
            'peak_hour': peak_hour
        }

    def _calculate_topic_metrics(
        self,
        sessions: List[SessionData]
    ) -> Dict[str, Any]:
        """Calculate topic-related metrics"""
        all_topics = []
        for s in sessions:
            all_topics.extend(s.topics)

        if not all_topics:
            return {
                'active_topics': 0,
                'top_topics': []
            }

        # Count topics
        topic_counts = defaultdict(int)
        for topic in all_topics:
            topic_counts[topic] += 1

        # Sort by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [t[0] for t in sorted_topics[:5]]

        return {
            'active_topics': len(set(all_topics)),
            'top_topics': top_topics
        }

    # =========================================================================
    # Private Dashboard Building Methods
    # =========================================================================

    def _build_summary(
        self,
        latest: Optional[Dict[str, Any]],
        sessions: List[SessionData],
        quality_scores: List[QualityScore]
    ) -> Dict[str, Any]:
        """Build summary section for dashboard"""
        summary = {
            'period_days': 30,
            'total_sessions': len(sessions),
            'total_exchanges': sum(s.exchange_count for s in sessions),
            'avg_quality': (
                statistics.mean([q.composite for q in quality_scores])
                if quality_scores else 0
            ),
            'unique_topics': len(set(
                topic for s in sessions for topic in s.topics
            ))
        }

        if latest:
            summary['latest_date'] = latest.get('date')
            summary['latest_sessions'] = latest.get('total_sessions', 0)
            summary['latest_quality'] = latest.get('avg_quality_score', 0)

        return summary

    def _build_trends(
        self,
        stored_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build trends section for dashboard"""
        trends = []

        for day_metrics in stored_metrics:
            trends.append({
                'date': day_metrics.get('date'),
                'sessions': day_metrics.get('total_sessions', 0),
                'quality': day_metrics.get('avg_quality_score', 0),
                'positive_rate': day_metrics.get('positive_feedback_rate', 0.5)
            })

        return trends

    def _build_quality_breakdown(
        self,
        quality_scores: List[QualityScore]
    ) -> Dict[str, Any]:
        """Build quality breakdown for dashboard"""
        if not quality_scores:
            return {
                'overall': 0,
                'dimensions': {},
                'distribution': {}
            }

        composites = [q.composite for q in quality_scores]

        # Distribution
        excellent = sum(1 for c in composites if c >= 0.85)
        good = sum(1 for c in composites if 0.70 <= c < 0.85)
        fair = sum(1 for c in composites if 0.50 <= c < 0.70)
        poor = sum(1 for c in composites if c < 0.50)

        return {
            'overall': statistics.mean(composites),
            'dimensions': {
                'relevance': statistics.mean([q.relevance for q in quality_scores]),
                'helpfulness': statistics.mean([q.helpfulness for q in quality_scores]),
                'accuracy': statistics.mean([q.accuracy for q in quality_scores]),
                'clarity': statistics.mean([q.clarity for q in quality_scores]),
                'completeness': statistics.mean([q.completeness for q in quality_scores])
            },
            'distribution': {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor
            }
        }

    def _build_engagement_stats(
        self,
        sessions: List[SessionData]
    ) -> Dict[str, Any]:
        """Build engagement statistics for dashboard"""
        if not sessions:
            return {
                'avg_session_length': 0,
                'avg_exchanges': 0,
                'hour_distribution': {},
                'day_distribution': {}
            }

        # Hour distribution
        hour_dist = defaultdict(int)
        day_dist = defaultdict(int)
        durations = []
        exchanges = []

        for s in sessions:
            hour_dist[s.start_time.hour] += 1
            day_dist[s.start_time.weekday()] += 1
            if s.duration > 0:
                durations.append(s.duration / 60)
            exchanges.append(s.exchange_count)

        days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        return {
            'avg_session_length': statistics.mean(durations) if durations else 0,
            'avg_exchanges': statistics.mean(exchanges) if exchanges else 0,
            'hour_distribution': dict(hour_dist),
            'day_distribution': {
                days_of_week[k]: v for k, v in day_dist.items()
            }
        }

    def _build_feedback_analysis(
        self,
        feedback_list: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Build feedback analysis for dashboard"""
        if not feedback_list:
            return {
                'total': 0,
                'by_type': {},
                'positive_rate': 0.5,
                'trend': 'stable'
            }

        # Count by type
        by_type = defaultdict(int)
        for f in feedback_list:
            by_type[f.feedback_type.value] += 1

        positive = by_type.get('explicit_positive', 0) + by_type.get('implicit_positive', 0)
        negative = by_type.get('explicit_negative', 0) + by_type.get('implicit_negative', 0)
        total_sentiment = positive + negative

        return {
            'total': len(feedback_list),
            'by_type': dict(by_type),
            'positive_rate': positive / total_sentiment if total_sentiment > 0 else 0.5,
            'corrections': by_type.get('correction', 0),
            'clarifications': by_type.get('clarification', 0)
        }

    def _build_topic_distribution(
        self,
        sessions: List[SessionData]
    ) -> Dict[str, Any]:
        """Build topic distribution for dashboard"""
        topic_counts = defaultdict(int)

        for s in sessions:
            for topic in s.topics:
                topic_counts[topic] += 1

        if not topic_counts:
            return {
                'total_topics': 0,
                'top_topics': [],
                'distribution': {}
            }

        total = sum(topic_counts.values())
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_topics': len(topic_counts),
            'top_topics': [
                {'topic': t, 'count': c, 'percentage': c / total * 100}
                for t, c in sorted_topics[:10]
            ],
            'distribution': dict(sorted_topics)
        }

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _calculate_positive_rate(
        self,
        feedback_list: List[FeedbackData]
    ) -> float:
        """Calculate positive feedback rate"""
        if not feedback_list:
            return 0.5

        positive = sum(
            1 for f in feedback_list
            if f.feedback_type in [FeedbackType.EXPLICIT_POSITIVE, FeedbackType.IMPLICIT_POSITIVE]
        )
        negative = sum(
            1 for f in feedback_list
            if f.feedback_type in [FeedbackType.EXPLICIT_NEGATIVE, FeedbackType.IMPLICIT_NEGATIVE]
        )

        total = positive + negative
        return positive / total if total > 0 else 0.5

    def _count_active_sessions(
        self,
        sessions: List[SessionData]
    ) -> int:
        """Count sessions active in the last 30 minutes"""
        now = datetime.utcnow()
        threshold = now - timedelta(minutes=30)

        active = sum(
            1 for s in sessions
            if s.end_time is None or s.end_time > threshold
        )

        return active


# Global metrics aggregator instance
metrics_aggregator = MetricsAggregator()
