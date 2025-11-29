"""
Learning Analytics Engine
PATTERN: Comprehensive analytics with statistical rigor
WHY: Generate actionable insights from learning data
SPARC: Statistical analysis with domain-specific metrics
"""
import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

from app.logger import db_logger
from app.learning.config import LearningConfig, learning_config
from app.learning.stores import (
    FeedbackStore, QualityStore, PatternStore,
    SessionData, FeedbackData, QualityScore, FeedbackType,
    feedback_store, quality_store, pattern_store
)


@dataclass
class SessionReport:
    """Detailed session analysis report"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: float
    exchange_count: int
    quality_metrics: Dict[str, float]
    feedback_summary: Dict[str, int]
    topics: List[str]
    engagement_score: float
    highlights: List[str]
    improvement_areas: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DailyReport:
    """Daily learning summary report"""
    date: Any  # Can be date or datetime for compatibility
    sessions: Dict[str, Any]
    quality: Dict[str, Any]
    feedback: Dict[str, Any]
    patterns: List[Dict[str, Any]]
    insights: List[str]
    comparison_to_previous: Dict[str, float]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    report_version: str = "1.0.0"
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for Pydantic-like compatibility."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "sessions": self.sessions,
            "quality": self.quality,
            "feedback": self.feedback,
            "patterns": self.patterns,
            "insights": self.insights,
            "comparison_to_previous": self.comparison_to_previous,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "report_version": self.report_version,
            "dimension_scores": self.dimension_scores,
            "recommendations": self.recommendations
        }

    @property
    def total_interactions(self) -> int:
        """Total interactions count from sessions data."""
        return self.sessions.get("count", 0)

    @property
    def average_quality_score(self) -> float:
        """Average quality score from quality data."""
        return self.quality.get("average", 0.0)

    @property
    def recommendations(self) -> List[str]:
        """Generate recommendations from insights for backward compatibility."""
        recs = []
        for insight in self.insights:
            if "review" in insight.lower() or "below" in insight.lower():
                recs.append(f"Action: {insight}")
            elif "excellent" in insight.lower() or "strong" in insight.lower():
                recs.append("Continue current practices")
            else:
                recs.append(insight)
        if not recs:
            recs.append("Maintain current learning trajectory")
        return recs


@dataclass
class QualityTrend:
    """Quality trend data point"""
    date: date
    composite: float
    rolling_avg: float
    change: float
    dimensions: Dict[str, float]


@dataclass
class TrendResult:
    """Result of trend calculation for test compatibility."""
    data_points: int = 0
    start_value: float = 0.0
    end_value: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    average_value: float = 0.0
    direction: str = "stable"


class LearningAnalytics:
    """
    PATTERN: Comprehensive learning analytics engine
    WHY: Generate insights from session data, feedback, and quality scores

    USAGE:
        analytics = LearningAnalytics()
        await analytics.initialize()

        # Generate reports
        daily = await analytics.generate_daily_report()
        trends = await analytics.get_quality_trends(days=30)
        gaps = await analytics.identify_knowledge_gaps()
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback: Optional[FeedbackStore] = None,
        quality: Optional[QualityStore] = None,
        patterns: Optional[PatternStore] = None
    ):
        """
        Initialize analytics engine

        Args:
            config: Analytics configuration
            feedback: Feedback store instance
            quality: Quality store instance
            patterns: Pattern store instance
        """
        self.config = config or learning_config
        self.feedback = feedback or feedback_store
        self.quality = quality or quality_store
        self.patterns = patterns or pattern_store
        self._pattern_detector = None
        self._initialized = False
        # In-memory storage for test compatibility
        self._quality_scores: Dict[str, List[Any]] = defaultdict(list)
        self._interaction_counts: Dict[str, int] = defaultdict(int)

    async def initialize(self) -> None:
        """Initialize analytics engine and dependencies"""
        if self._initialized:
            return

        try:
            # Initialize stores
            await self.feedback.initialize()
            await self.quality.initialize()
            await self.patterns.initialize()

            # Lazy import to avoid circular dependencies
            from app.learning.pattern_detector import PatternDetector
            self._pattern_detector = PatternDetector(self.config)

            self._initialized = True
            db_logger.info("learning_analytics_initialized")

        except Exception as e:
            db_logger.error(
                "learning_analytics_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    # =========================================================================
    # Test Compatibility Methods (in-memory tracking)
    # =========================================================================

    def record_quality_score(self, score: Any) -> None:
        """
        Record a quality score in memory for tracking.

        Args:
            score: QualityScore object with session_id attribute
        """
        session_id = getattr(score, 'session_id', 'default')
        self._quality_scores[session_id].append(score)

    def increment_interaction_count(self, session_id: str) -> None:
        """
        Increment interaction count for a session.

        Args:
            session_id: Session to increment count for
        """
        self._interaction_counts[session_id] += 1

    async def get_metrics_summary(
        self,
        session_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get metrics summary for analysis.

        Args:
            session_id: Optional session to filter by
            days: Number of days to include

        Returns:
            Dictionary with metrics summary
        """
        total_scored = 0
        if session_id:
            total_scored = len(self._quality_scores.get(session_id, []))
        else:
            total_scored = sum(len(scores) for scores in self._quality_scores.values())

        return {
            "period_days": days,
            "total_scored_interactions": total_scored,
            "sessions_tracked": len(self._quality_scores),
            "total_interactions": sum(self._interaction_counts.values())
        }

    async def calculate_trend(
        self,
        dimension: Any,
        session_id: Optional[str] = None
    ) -> TrendResult:
        """
        Calculate trend for a quality dimension.

        Args:
            dimension: QualityDimension enum value
            session_id: Optional session to filter by

        Returns:
            TrendResult with trend statistics
        """
        # Get dimension name
        dim_name = dimension.value if hasattr(dimension, 'value') else str(dimension)

        # Get relevant scores
        if session_id:
            scores = self._quality_scores.get(session_id, [])
        else:
            scores = [s for session_scores in self._quality_scores.values() for s in session_scores]

        if not scores:
            return TrendResult(data_points=0)

        # Extract dimension values
        values = []
        for score in scores:
            val = getattr(score, dim_name.lower(), None)
            if val is not None:
                values.append(val)

        if not values:
            return TrendResult(data_points=0)

        # Calculate statistics
        start_val = values[0]
        end_val = values[-1]
        change = end_val - start_val
        change_pct = (change / start_val * 100) if start_val != 0 else 0.0

        # Determine direction
        if change > 0.05:
            direction = "increasing"
        elif change < -0.05:
            direction = "decreasing"
        else:
            direction = "stable"

        return TrendResult(
            data_points=len(values),
            start_value=start_val,
            end_value=end_val,
            change=change,
            change_percent=change_pct,
            min_value=min(values),
            max_value=max(values),
            average_value=statistics.mean(values),
            direction=direction
        )

    async def generate_session_report(self, session_id: str) -> Optional[SessionReport]:
        """
        Generate detailed session analysis report

        PATTERN: Comprehensive session analysis
        WHY: Understand individual session performance

        Args:
            session_id: Session to analyze

        Returns:
            SessionReport or None if session not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("generating_session_report", session_id=session_id)

            # Get session data
            session = await self.feedback.get_session(session_id)
            if not session:
                db_logger.warning("session_not_found", session_id=session_id)
                return None

            # Get quality scores
            quality_scores = await self.quality.get_scores_by_session(session_id)

            # Get feedback
            feedback_list = await self.feedback.get_feedback_by_session(session_id)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(quality_scores)

            # Summarize feedback
            feedback_summary = self._summarize_feedback(feedback_list)

            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                session, quality_scores, feedback_list
            )

            # Generate highlights and improvement areas
            highlights = self._generate_highlights(
                session, quality_metrics, feedback_summary
            )
            improvement_areas = self._identify_improvement_areas(
                quality_metrics, feedback_summary
            )

            report = SessionReport(
                session_id=session_id,
                start_time=session.start_time,
                end_time=session.end_time,
                duration_minutes=session.duration / 60 if session.duration > 0 else 0,
                exchange_count=session.exchange_count,
                quality_metrics=quality_metrics,
                feedback_summary=feedback_summary,
                topics=session.topics,
                engagement_score=engagement_score,
                highlights=highlights,
                improvement_areas=improvement_areas
            )

            db_logger.info(
                "session_report_generated",
                session_id=session_id,
                duration_minutes=report.duration_minutes,
                exchange_count=report.exchange_count
            )

            return report

        except Exception as e:
            db_logger.error(
                "generate_session_report_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return None

    async def generate_daily_report(
        self,
        report_date: Optional[date] = None,
        session_id: Optional[str] = None,
        **kwargs  # Accept 'date' for backward compatibility
    ) -> DailyReport:
        """
        Generate daily learning analytics report

        PATTERN: Aggregated daily analysis with comparisons
        WHY: Track learning progress over time

        Args:
            report_date: Date to generate report for (defaults to yesterday)

        Returns:
            DailyReport with comprehensive daily analysis
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Handle 'date' kwarg for backward compatibility
            if report_date is None and 'date' in kwargs:
                date_val = kwargs['date']
                if hasattr(date_val, 'date'):
                    report_date = date_val.date()  # datetime -> date
                else:
                    report_date = date_val  # already a date

            if report_date is None:
                report_date = date.today() - timedelta(days=1)

            db_logger.info("generating_daily_report", date=str(report_date))

            next_date = report_date + timedelta(days=1)
            prev_date = report_date - timedelta(days=1)

            # Get data for report date
            sessions = await self.feedback.get_sessions_in_range(report_date, next_date)
            quality_scores = await self.quality.get_scores_in_range(report_date, next_date)
            feedback_list = await self.feedback.get_feedback_in_range(report_date, next_date)

            # Get previous day for comparison
            prev_sessions = await self.feedback.get_sessions_in_range(prev_date, report_date)
            prev_quality = await self.quality.get_scores_in_range(prev_date, report_date)

            # Calculate session metrics
            session_metrics = self._calculate_session_metrics(sessions)

            # Calculate quality metrics
            quality_metrics = self._calculate_daily_quality_metrics(quality_scores)

            # Calculate feedback metrics
            feedback_metrics = self._calculate_feedback_metrics(feedback_list)

            # Get detected patterns
            detected_patterns = await self.patterns.get_active_patterns()

            # Calculate comparison to previous day
            comparison = self._calculate_daily_comparison(
                sessions, quality_scores,
                prev_sessions, prev_quality
            )

            # Generate daily insights
            insights = self._generate_daily_insights(
                session_metrics, quality_metrics, feedback_metrics
            )

            report = DailyReport(
                date=report_date,
                sessions=session_metrics,
                quality=quality_metrics,
                feedback=feedback_metrics,
                patterns=detected_patterns[:5],  # Top 5 patterns
                insights=insights,
                comparison_to_previous=comparison
            )

            db_logger.info(
                "daily_report_generated",
                date=str(report_date),
                sessions_count=session_metrics.get('total', 0),
                avg_quality=quality_metrics.get('avg_composite', 0)
            )

            return report

        except Exception as e:
            db_logger.error(
                "generate_daily_report_failed",
                date=str(report_date),
                error=str(e),
                exc_info=True
            )
            # Return empty report on failure
            return DailyReport(
                date=report_date or date.today(),
                sessions={},
                quality={},
                feedback={},
                patterns=[],
                insights=[],
                comparison_to_previous={}
            )

    async def get_quality_trends(
        self,
        days: int = 30,
        include_dimensions: bool = True
    ) -> List[QualityTrend]:
        """
        Get quality trends over time

        PATTERN: Rolling average trend analysis
        WHY: Identify quality improvements or regressions

        Args:
            days: Number of days to analyze
            include_dimensions: Include breakdown by dimension

        Returns:
            List of QualityTrend data points
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("calculating_quality_trends", days=days)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get daily averages
            daily_scores = await self.quality.get_daily_averages(start_date, end_date)

            if not daily_scores:
                return []

            # Calculate rolling averages and changes
            window_size = self.config.analytics.rolling_window_days
            trends = []

            for i, day_score in enumerate(daily_scores):
                # Get window for rolling average
                window_start = max(0, i - window_size + 1)
                window = daily_scores[window_start:i + 1]

                # Calculate rolling average
                rolling_avg = statistics.mean([d['composite'] for d in window])

                # Calculate change from previous period
                change = self._calculate_trend_change(daily_scores, i, window_size)

                # Build dimensions dict
                dimensions = {}
                if include_dimensions:
                    dimensions = {
                        'relevance': day_score.get('relevance', 0) or 0,
                        'helpfulness': day_score.get('helpfulness', 0) or 0,
                        'accuracy': day_score.get('accuracy', 0) or 0,
                        'clarity': day_score.get('clarity', 0) or 0,
                        'completeness': day_score.get('completeness', 0) or 0
                    }

                trends.append(QualityTrend(
                    date=date.fromisoformat(day_score['date']),
                    composite=day_score['composite'] or 0,
                    rolling_avg=rolling_avg,
                    change=change,
                    dimensions=dimensions
                ))

            db_logger.info(
                "quality_trends_calculated",
                days=days,
                data_points=len(trends)
            )

            return trends

        except Exception as e:
            db_logger.error(
                "get_quality_trends_failed",
                days=days,
                error=str(e),
                exc_info=True
            )
            return []

    async def get_topic_distribution(
        self,
        days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get distribution of topics discussed

        PATTERN: Topic frequency and quality analysis
        WHY: Understand what topics are most common and how well they're handled

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of topic -> stats
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("calculating_topic_distribution", days=days)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            sessions = await self.feedback.get_sessions_in_range(start_date, end_date)

            # Aggregate topic data
            topic_stats = defaultdict(lambda: {
                'count': 0,
                'sessions': [],
                'total_exchanges': 0
            })

            for session in sessions:
                for topic in session.topics:
                    topic_stats[topic]['count'] += 1
                    topic_stats[topic]['sessions'].append(session.session_id)
                    topic_stats[topic]['total_exchanges'] += session.exchange_count

            # Calculate percentages and averages
            total_mentions = sum(t['count'] for t in topic_stats.values())

            result = {}
            for topic, stats in topic_stats.items():
                result[topic] = {
                    'count': stats['count'],
                    'percentage': (stats['count'] / total_mentions * 100) if total_mentions > 0 else 0,
                    'avg_exchanges': stats['total_exchanges'] / stats['count'] if stats['count'] > 0 else 0,
                    'session_count': len(set(stats['sessions']))
                }

            # Sort by frequency
            result = dict(sorted(
                result.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            ))

            db_logger.info(
                "topic_distribution_calculated",
                unique_topics=len(result),
                total_mentions=total_mentions
            )

            return result

        except Exception as e:
            db_logger.error(
                "get_topic_distribution_failed",
                error=str(e),
                exc_info=True
            )
            return {}

    async def get_engagement_patterns(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get user engagement patterns

        PATTERN: Temporal and behavioral analysis
        WHY: Understand when and how users engage best

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with engagement patterns
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("calculating_engagement_patterns", days=days)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            sessions = await self.feedback.get_sessions_in_range(start_date, end_date)

            if not sessions:
                return {}

            # Time of day analysis
            hour_distribution = defaultdict(lambda: {'count': 0, 'total_duration': 0})
            day_distribution = defaultdict(lambda: {'count': 0, 'total_duration': 0})

            # Session length analysis
            session_lengths = []

            for session in sessions:
                hour = session.start_time.hour
                day = session.start_time.weekday()
                duration = session.duration

                hour_distribution[hour]['count'] += 1
                hour_distribution[hour]['total_duration'] += duration

                day_distribution[day]['count'] += 1
                day_distribution[day]['total_duration'] += duration

                if duration > 0:
                    session_lengths.append(duration / 60)  # Convert to minutes

            # Calculate averages
            for hour in hour_distribution:
                count = hour_distribution[hour]['count']
                hour_distribution[hour]['avg_duration'] = (
                    hour_distribution[hour]['total_duration'] / count / 60
                    if count > 0 else 0
                )

            for day in day_distribution:
                count = day_distribution[day]['count']
                day_distribution[day]['avg_duration'] = (
                    day_distribution[day]['total_duration'] / count / 60
                    if count > 0 else 0
                )

            # Find best times
            best_hour = max(
                hour_distribution.items(),
                key=lambda x: x[1]['count'],
                default=(0, {'count': 0})
            )
            best_day = max(
                day_distribution.items(),
                key=lambda x: x[1]['count'],
                default=(0, {'count': 0})
            )

            days_of_week = [
                'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday'
            ]

            result = {
                'time_of_day': {
                    'distribution': dict(hour_distribution),
                    'peak_hour': best_hour[0],
                    'peak_hour_sessions': best_hour[1]['count']
                },
                'day_of_week': {
                    'distribution': {
                        days_of_week[k]: v for k, v in day_distribution.items()
                    },
                    'best_day': days_of_week[best_day[0]] if best_day[0] < 7 else 'Unknown',
                    'best_day_sessions': best_day[1]['count']
                },
                'session_length': {
                    'avg_minutes': statistics.mean(session_lengths) if session_lengths else 0,
                    'median_minutes': statistics.median(session_lengths) if session_lengths else 0,
                    'min_minutes': min(session_lengths) if session_lengths else 0,
                    'max_minutes': max(session_lengths) if session_lengths else 0
                },
                'total_sessions': len(sessions)
            }

            db_logger.info(
                "engagement_patterns_calculated",
                total_sessions=len(sessions),
                peak_hour=result['time_of_day']['peak_hour']
            )

            return result

        except Exception as e:
            db_logger.error(
                "get_engagement_patterns_failed",
                error=str(e),
                exc_info=True
            )
            return {}

    async def identify_knowledge_gaps(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Identify knowledge gaps and areas of struggle

        PATTERN: Quality correlation with topics and patterns
        WHY: Find areas where user needs more support

        Args:
            days: Number of days to analyze

        Returns:
            List of knowledge gaps with context
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("identifying_knowledge_gaps", days=days)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get sessions with low quality or many corrections
            sessions = await self.feedback.get_sessions_in_range(start_date, end_date)

            # Get feedback for corrections/clarifications
            feedback_list = await self.feedback.get_feedback_in_range(start_date, end_date)

            # Group by topic and identify patterns
            topic_issues = defaultdict(lambda: {
                'corrections': 0,
                'clarifications': 0,
                'low_quality_sessions': 0,
                'examples': []
            })

            # Analyze corrections and clarifications
            session_feedback = defaultdict(list)
            for fb in feedback_list:
                session_feedback[fb.session_id].append(fb)

            for session in sessions:
                session_fb = session_feedback.get(session.session_id, [])

                # Count issues by type
                corrections = sum(
                    1 for fb in session_fb
                    if fb.feedback_type == FeedbackType.CORRECTION
                )
                clarifications = sum(
                    1 for fb in session_fb
                    if fb.feedback_type == FeedbackType.CLARIFICATION
                )

                # Check for low quality
                quality_scores = await self.quality.get_scores_by_session(session.session_id)
                avg_quality = (
                    statistics.mean([q.composite for q in quality_scores])
                    if quality_scores else 0.5
                )

                is_low_quality = avg_quality < self.config.analytics.quality_poor_threshold

                # Attribute to topics
                for topic in session.topics:
                    topic_issues[topic]['corrections'] += corrections
                    topic_issues[topic]['clarifications'] += clarifications
                    if is_low_quality:
                        topic_issues[topic]['low_quality_sessions'] += 1

                    # Store example if there were issues
                    if corrections > 0 or clarifications > 0 or is_low_quality:
                        for fb in session_fb[:2]:  # Max 2 examples per session
                            if fb.content:
                                topic_issues[topic]['examples'].append({
                                    'session_id': session.session_id,
                                    'type': fb.feedback_type.value,
                                    'content': fb.content[:100]
                                })

            # Calculate gap scores and rank
            gaps = []
            for topic, issues in topic_issues.items():
                # Calculate gap score (higher = more problematic)
                gap_score = (
                    issues['corrections'] * 3 +
                    issues['clarifications'] * 2 +
                    issues['low_quality_sessions'] * 2
                )

                if gap_score > 0:
                    gaps.append({
                        'topic': topic,
                        'gap_score': gap_score,
                        'corrections': issues['corrections'],
                        'clarifications': issues['clarifications'],
                        'low_quality_sessions': issues['low_quality_sessions'],
                        'examples': issues['examples'][:3],  # Top 3 examples
                        'recommendation': self._generate_gap_recommendation(topic, issues)
                    })

            # Sort by gap score
            gaps.sort(key=lambda x: x['gap_score'], reverse=True)

            db_logger.info(
                "knowledge_gaps_identified",
                total_gaps=len(gaps),
                top_gap=gaps[0]['topic'] if gaps else None
            )

            return gaps[:10]  # Return top 10 gaps

        except Exception as e:
            db_logger.error(
                "identify_knowledge_gaps_failed",
                error=str(e),
                exc_info=True
            )
            return []

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _calculate_quality_metrics(
        self,
        scores: List[QualityScore]
    ) -> Dict[str, float]:
        """Calculate quality metrics from scores"""
        if not scores:
            return {
                'composite': 0.0,
                'relevance': 0.0,
                'helpfulness': 0.0,
                'accuracy': 0.0,
                'clarity': 0.0,
                'completeness': 0.0,
                'count': 0
            }

        return {
            'composite': statistics.mean([s.composite for s in scores]),
            'relevance': statistics.mean([s.relevance for s in scores]),
            'helpfulness': statistics.mean([s.helpfulness for s in scores]),
            'accuracy': statistics.mean([s.accuracy for s in scores]),
            'clarity': statistics.mean([s.clarity for s in scores]),
            'completeness': statistics.mean([s.completeness for s in scores]),
            'count': len(scores)
        }

    def _summarize_feedback(
        self,
        feedback_list: List[FeedbackData]
    ) -> Dict[str, int]:
        """Summarize feedback by type"""
        summary = {ft.value: 0 for ft in FeedbackType}

        for fb in feedback_list:
            summary[fb.feedback_type.value] += 1

        # Calculate positive rate
        positive = summary.get('explicit_positive', 0) + summary.get('implicit_positive', 0)
        negative = summary.get('explicit_negative', 0) + summary.get('implicit_negative', 0)
        total = positive + negative

        summary['positive_rate'] = positive / total if total > 0 else 0.5

        return summary

    def _calculate_engagement_score(
        self,
        session: SessionData,
        quality_scores: List[QualityScore],
        feedback_list: List[FeedbackData]
    ) -> float:
        """
        Calculate engagement score for a session

        Factors:
        - Session duration (longer = more engaged, up to a point)
        - Exchange count (more exchanges = more engaged)
        - Positive feedback ratio
        - Quality score trend (improving = engaged)
        """
        score = 0.0

        # Duration factor (0-0.25)
        duration_minutes = session.duration / 60 if session.duration > 0 else 0
        duration_score = min(duration_minutes / 30, 1.0) * 0.25
        score += duration_score

        # Exchange count factor (0-0.25)
        exchange_score = min(session.exchange_count / 20, 1.0) * 0.25
        score += exchange_score

        # Feedback factor (0-0.25)
        positive = sum(
            1 for fb in feedback_list
            if fb.feedback_type in [FeedbackType.EXPLICIT_POSITIVE, FeedbackType.IMPLICIT_POSITIVE]
        )
        negative = sum(
            1 for fb in feedback_list
            if fb.feedback_type in [FeedbackType.EXPLICIT_NEGATIVE, FeedbackType.IMPLICIT_NEGATIVE]
        )
        total_fb = positive + negative
        feedback_score = (positive / total_fb if total_fb > 0 else 0.5) * 0.25
        score += feedback_score

        # Quality factor (0-0.25)
        if quality_scores:
            avg_quality = statistics.mean([q.composite for q in quality_scores])
            score += avg_quality * 0.25
        else:
            score += 0.125  # Neutral if no quality data

        return min(score, 1.0)

    def _generate_highlights(
        self,
        session: SessionData,
        quality_metrics: Dict[str, float],
        feedback_summary: Dict[str, int]
    ) -> List[str]:
        """Generate session highlights"""
        highlights = []

        # Quality highlights
        if quality_metrics.get('composite', 0) >= self.config.analytics.quality_excellent_threshold:
            highlights.append("Excellent overall quality score")

        for dimension in ['relevance', 'helpfulness', 'clarity']:
            if quality_metrics.get(dimension, 0) >= 0.9:
                highlights.append(f"Outstanding {dimension}")

        # Engagement highlights
        if session.exchange_count >= 10:
            highlights.append("Highly interactive session")

        if session.duration > 0 and session.duration / 60 >= 20:
            highlights.append("Extended engagement")

        # Feedback highlights
        if feedback_summary.get('positive_rate', 0) >= 0.8:
            highlights.append("Very positive feedback")

        return highlights[:5]  # Limit to 5 highlights

    def _identify_improvement_areas(
        self,
        quality_metrics: Dict[str, float],
        feedback_summary: Dict[str, int]
    ) -> List[str]:
        """Identify areas for improvement"""
        improvements = []

        # Quality improvements
        threshold = self.config.analytics.quality_good_threshold

        for dimension in ['relevance', 'helpfulness', 'accuracy', 'clarity', 'completeness']:
            if quality_metrics.get(dimension, 1.0) < threshold:
                improvements.append(f"Improve {dimension} of responses")

        # Feedback-based improvements
        if feedback_summary.get('correction', 0) > 2:
            improvements.append("Reduce need for corrections")

        if feedback_summary.get('clarification', 0) > 3:
            improvements.append("Provide clearer initial responses")

        return improvements[:5]  # Limit to 5 areas

    def _calculate_session_metrics(
        self,
        sessions: List[SessionData]
    ) -> Dict[str, Any]:
        """Calculate aggregated session metrics"""
        if not sessions:
            return {
                'total': 0,
                'avg_duration_minutes': 0,
                'avg_exchanges': 0,
                'total_exchanges': 0
            }

        durations = [s.duration / 60 for s in sessions if s.duration > 0]
        exchanges = [s.exchange_count for s in sessions]

        return {
            'total': len(sessions),
            'avg_duration_minutes': statistics.mean(durations) if durations else 0,
            'median_duration_minutes': statistics.median(durations) if durations else 0,
            'avg_exchanges': statistics.mean(exchanges) if exchanges else 0,
            'total_exchanges': sum(exchanges)
        }

    def _calculate_daily_quality_metrics(
        self,
        scores: List[QualityScore]
    ) -> Dict[str, Any]:
        """Calculate daily quality metrics"""
        if not scores:
            return {
                'avg_composite': 0,
                'avg_by_dimension': {},
                'count': 0,
                'trend': 'stable'
            }

        composites = [s.composite for s in scores]

        return {
            'avg_composite': statistics.mean(composites),
            'median_composite': statistics.median(composites),
            'std_composite': statistics.stdev(composites) if len(composites) > 1 else 0,
            'avg_by_dimension': {
                'relevance': statistics.mean([s.relevance for s in scores]),
                'helpfulness': statistics.mean([s.helpfulness for s in scores]),
                'accuracy': statistics.mean([s.accuracy for s in scores]),
                'clarity': statistics.mean([s.clarity for s in scores]),
                'completeness': statistics.mean([s.completeness for s in scores])
            },
            'count': len(scores),
            'trend': self._calculate_intra_day_trend(scores)
        }

    def _calculate_feedback_metrics(
        self,
        feedback_list: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Calculate feedback metrics"""
        if not feedback_list:
            return {
                'total': 0,
                'by_type': {},
                'positive_rate': 0.5
            }

        by_type = defaultdict(int)
        for fb in feedback_list:
            by_type[fb.feedback_type.value] += 1

        positive = by_type.get('explicit_positive', 0) + by_type.get('implicit_positive', 0)
        negative = by_type.get('explicit_negative', 0) + by_type.get('implicit_negative', 0)
        total_sentiment = positive + negative

        return {
            'total': len(feedback_list),
            'by_type': dict(by_type),
            'positive_rate': positive / total_sentiment if total_sentiment > 0 else 0.5,
            'corrections_count': by_type.get('correction', 0),
            'clarifications_count': by_type.get('clarification', 0)
        }

    def _calculate_daily_comparison(
        self,
        current_sessions: List[SessionData],
        current_quality: List[QualityScore],
        prev_sessions: List[SessionData],
        prev_quality: List[QualityScore]
    ) -> Dict[str, float]:
        """Calculate comparison to previous day"""
        comparison = {}

        # Session count change
        curr_count = len(current_sessions)
        prev_count = len(prev_sessions)
        if prev_count > 0:
            comparison['sessions_change'] = ((curr_count - prev_count) / prev_count) * 100
        else:
            comparison['sessions_change'] = 100.0 if curr_count > 0 else 0.0

        # Quality change
        curr_quality = (
            statistics.mean([q.composite for q in current_quality])
            if current_quality else 0
        )
        prev_quality_avg = (
            statistics.mean([q.composite for q in prev_quality])
            if prev_quality else 0
        )
        if prev_quality_avg > 0:
            comparison['quality_change'] = ((curr_quality - prev_quality_avg) / prev_quality_avg) * 100
        else:
            comparison['quality_change'] = 0.0

        return comparison

    def _generate_daily_insights(
        self,
        session_metrics: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        feedback_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate daily insights"""
        insights = []

        # Session insights
        total_sessions = session_metrics.get('total', 0)
        if total_sessions > 10:
            insights.append(f"High activity day with {total_sessions} sessions")
        elif total_sessions == 0:
            insights.append("No sessions recorded")

        # Quality insights
        avg_quality = quality_metrics.get('avg_composite', 0)
        if avg_quality >= 0.85:
            insights.append("Excellent quality day")
        elif avg_quality < 0.6:
            insights.append("Quality below target - review responses")

        # Feedback insights
        positive_rate = feedback_metrics.get('positive_rate', 0.5)
        if positive_rate >= 0.8:
            insights.append("Strong positive feedback")
        elif positive_rate < 0.5:
            insights.append("More negative feedback - investigate causes")

        corrections = feedback_metrics.get('corrections_count', 0)
        if corrections > 5:
            insights.append(f"High correction rate ({corrections} corrections)")

        return insights

    def _calculate_trend_change(
        self,
        daily_scores: List[Dict],
        index: int,
        window_size: int
    ) -> float:
        """Calculate trend change percentage"""
        if index < window_size:
            return 0.0

        current_window = daily_scores[index - window_size + 1:index + 1]
        prev_window = daily_scores[max(0, index - 2 * window_size + 1):index - window_size + 1]

        if not prev_window:
            return 0.0

        current_avg = statistics.mean([d['composite'] for d in current_window])
        prev_avg = statistics.mean([d['composite'] for d in prev_window])

        if prev_avg == 0:
            return 0.0

        return ((current_avg - prev_avg) / prev_avg) * 100

    def _calculate_intra_day_trend(
        self,
        scores: List[QualityScore]
    ) -> str:
        """Calculate trend within a day"""
        if len(scores) < 2:
            return 'stable'

        # Sort by timestamp
        sorted_scores = sorted(scores, key=lambda x: x.timestamp)

        # Compare first half to second half
        mid = len(sorted_scores) // 2
        first_half = statistics.mean([s.composite for s in sorted_scores[:mid]])
        second_half = statistics.mean([s.composite for s in sorted_scores[mid:]])

        diff = second_half - first_half
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        return 'stable'

    def _generate_gap_recommendation(
        self,
        topic: str,
        issues: Dict[str, Any]
    ) -> str:
        """Generate recommendation for a knowledge gap"""
        if issues['corrections'] > issues['clarifications']:
            return f"Review and improve accuracy for '{topic}' related responses"
        elif issues['clarifications'] > 2:
            return f"Provide more detailed initial explanations for '{topic}'"
        elif issues['low_quality_sessions'] > 0:
            return f"Focus on improving overall quality for '{topic}' discussions"
        return f"Monitor '{topic}' discussions for continued issues"


# Global analytics instance
learning_analytics = LearningAnalytics()
