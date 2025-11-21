"""
Insights Generator
PATTERN: Rule-based and statistical insight generation
WHY: Convert analytics data into actionable recommendations
SPARC: Systematic insight extraction with confidence scoring
"""
import uuid
import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from app.logger import db_logger
from app.learning.config import LearningConfig, learning_config
from app.learning.stores import (
    FeedbackStore, QualityStore, PatternStore, InsightStore,
    feedback_store, quality_store, pattern_store, insight_store,
    SessionData, QualityScore, FeedbackData, FeedbackType
)


class InsightCategory(str, Enum):
    """Categories of insights"""
    QUALITY = "quality"
    ENGAGEMENT = "engagement"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    IMPROVEMENT = "improvement"
    PERFORMANCE = "performance"


class InsightPriority(str, Enum):
    """Priority levels for insights"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """An actionable insight with evidence"""
    insight_id: str
    category: InsightCategory
    priority: InsightPriority
    title: str
    description: str
    evidence: Dict[str, Any]
    confidence: float  # 0-1
    actionable: bool
    recommendation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementRecommendation:
    """A specific improvement recommendation"""
    recommendation_id: str
    area: str
    current_state: str
    target_state: str
    actions: List[str]
    expected_impact: str
    effort_level: str  # low, medium, high
    confidence: float
    evidence: Dict[str, Any]


@dataclass
class PersonalizationSuggestion:
    """A personalization suggestion based on user behavior"""
    suggestion_id: str
    dimension: str  # response_length, formality, detail_level, etc.
    current_setting: Any
    suggested_setting: Any
    reason: str
    confidence: float
    supporting_data: Dict[str, Any]


class InsightsGenerator:
    """
    PATTERN: Multi-source insight generation
    WHY: Extract actionable insights from analytics data

    Generates insights from:
    - Quality trends and anomalies
    - Engagement patterns
    - User preferences
    - Detected patterns
    - Performance metrics

    USAGE:
        generator = InsightsGenerator()
        await generator.initialize()

        insights = await generator.generate_insights(days=7)
        recommendations = await generator.get_improvement_recommendations()
        personalizations = await generator.get_personalization_suggestions()
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback: Optional[FeedbackStore] = None,
        quality: Optional[QualityStore] = None,
        patterns: Optional[PatternStore] = None,
        insights: Optional[InsightStore] = None
    ):
        """
        Initialize insights generator

        Args:
            config: Learning configuration
            feedback: Feedback store instance
            quality: Quality store instance
            patterns: Pattern store instance
            insights: Insight store instance
        """
        self.config = config or learning_config
        self.feedback = feedback or feedback_store
        self.quality = quality or quality_store
        self.patterns = patterns or pattern_store
        self.insights_store = insights or insight_store
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize insights generator and dependencies"""
        if self._initialized:
            return

        try:
            await self.feedback.initialize()
            await self.quality.initialize()
            await self.patterns.initialize()
            await self.insights_store.initialize()

            self._initialized = True
            db_logger.info("insights_generator_initialized")

        except Exception as e:
            db_logger.error(
                "insights_generator_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def generate_insights(
        self,
        time_range_days: int = 7,
        categories: Optional[List[InsightCategory]] = None
    ) -> List[Insight]:
        """
        Generate all insights for a time range

        PATTERN: Multi-source aggregated insight generation
        WHY: Comprehensive view of actionable items

        Args:
            time_range_days: Number of days to analyze
            categories: Filter by categories (None = all)

        Returns:
            List of generated insights
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "generating_insights",
                time_range_days=time_range_days,
                categories=[c.value for c in categories] if categories else "all"
            )

            all_insights = []

            end_date = date.today()
            start_date = end_date - timedelta(days=time_range_days)

            # Generate category-specific insights
            if categories is None or InsightCategory.QUALITY in categories:
                quality_insights = await self._generate_quality_insights(
                    start_date, end_date
                )
                all_insights.extend(quality_insights)

            if categories is None or InsightCategory.ENGAGEMENT in categories:
                engagement_insights = await self._generate_engagement_insights(
                    start_date, end_date
                )
                all_insights.extend(engagement_insights)

            if categories is None or InsightCategory.PREFERENCE in categories:
                preference_insights = await self._generate_preference_insights(
                    start_date, end_date
                )
                all_insights.extend(preference_insights)

            if categories is None or InsightCategory.PATTERN in categories:
                pattern_insights = await self._generate_pattern_insights()
                all_insights.extend(pattern_insights)

            if categories is None or InsightCategory.IMPROVEMENT in categories:
                improvement_insights = await self._generate_improvement_insights(
                    start_date, end_date
                )
                all_insights.extend(improvement_insights)

            # Filter by confidence threshold
            min_confidence = self.config.insights.low_confidence_threshold
            filtered_insights = [
                i for i in all_insights
                if i.confidence >= min_confidence
            ]

            # Sort by priority and confidence
            priority_order = {
                InsightPriority.CRITICAL: 0,
                InsightPriority.HIGH: 1,
                InsightPriority.MEDIUM: 2,
                InsightPriority.LOW: 3
            }
            filtered_insights.sort(
                key=lambda i: (priority_order[i.priority], -i.confidence)
            )

            # Limit total insights
            max_insights = self.config.insights.max_total_insights
            final_insights = filtered_insights[:max_insights]

            # Save insights to store
            for insight in final_insights:
                await self._save_insight(insight)

            db_logger.info(
                "insights_generated",
                total_generated=len(all_insights),
                filtered_count=len(filtered_insights),
                final_count=len(final_insights)
            )

            return final_insights

        except Exception as e:
            db_logger.error(
                "generate_insights_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def get_improvement_recommendations(
        self,
        time_range_days: int = 30
    ) -> List[ImprovementRecommendation]:
        """
        Generate specific improvement recommendations

        PATTERN: Gap analysis with actionable steps
        WHY: Translate insights into concrete actions

        Args:
            time_range_days: Analysis period

        Returns:
            List of improvement recommendations
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("generating_improvement_recommendations")

            recommendations = []
            end_date = date.today()
            start_date = end_date - timedelta(days=time_range_days)

            # Get quality dimension stats
            dimension_stats = await self.quality.get_dimension_stats()

            # Analyze each dimension
            dimensions = ['relevance', 'helpfulness', 'accuracy', 'clarity', 'completeness']
            threshold = self.config.analytics.quality_good_threshold

            for dim in dimensions:
                dim_avg = dimension_stats.get(dim, {}).get('avg', 0)

                if dim_avg < threshold:
                    rec = self._create_dimension_recommendation(
                        dim, dim_avg, threshold, dimension_stats
                    )
                    if rec:
                        recommendations.append(rec)

            # Analyze feedback patterns
            feedback_list = await self.feedback.get_feedback_in_range(start_date, end_date)
            feedback_recs = self._analyze_feedback_for_recommendations(feedback_list)
            recommendations.extend(feedback_recs)

            # Sort by confidence and expected impact
            recommendations.sort(key=lambda r: r.confidence, reverse=True)

            db_logger.info(
                "improvement_recommendations_generated",
                count=len(recommendations)
            )

            return recommendations[:10]  # Top 10 recommendations

        except Exception as e:
            db_logger.error(
                "get_improvement_recommendations_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def get_personalization_suggestions(
        self,
        session_id: Optional[str] = None,
        time_range_days: int = 30
    ) -> List[PersonalizationSuggestion]:
        """
        Generate personalization suggestions based on user behavior

        PATTERN: Behavioral analysis for customization
        WHY: Improve user experience through personalization

        Args:
            session_id: Specific session to analyze (None = aggregate)
            time_range_days: Analysis period

        Returns:
            List of personalization suggestions
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info(
                "generating_personalization_suggestions",
                session_id=session_id
            )

            suggestions = []
            end_date = date.today()
            start_date = end_date - timedelta(days=time_range_days)

            # Get session data
            sessions = await self.feedback.get_sessions_in_range(start_date, end_date)
            if session_id:
                sessions = [s for s in sessions if s.session_id == session_id]

            if not sessions:
                return []

            # Analyze response length preferences
            length_suggestion = self._analyze_length_preferences(sessions)
            if length_suggestion:
                suggestions.append(length_suggestion)

            # Analyze engagement timing
            timing_suggestion = self._analyze_timing_preferences(sessions)
            if timing_suggestion:
                suggestions.append(timing_suggestion)

            # Analyze topic preferences
            topic_suggestions = self._analyze_topic_preferences(sessions)
            suggestions.extend(topic_suggestions)

            # Analyze feedback patterns for style preferences
            feedback_list = await self.feedback.get_feedback_in_range(start_date, end_date)
            style_suggestions = self._analyze_style_preferences(feedback_list)
            suggestions.extend(style_suggestions)

            # Sort by confidence
            suggestions.sort(key=lambda s: s.confidence, reverse=True)

            db_logger.info(
                "personalization_suggestions_generated",
                count=len(suggestions)
            )

            return suggestions

        except Exception as e:
            db_logger.error(
                "get_personalization_suggestions_failed",
                error=str(e),
                exc_info=True
            )
            return []

    # =========================================================================
    # Private Quality Insight Methods
    # =========================================================================

    async def _generate_quality_insights(
        self,
        start_date: date,
        end_date: date
    ) -> List[Insight]:
        """Generate quality-related insights"""
        insights = []

        # Get quality data
        daily_scores = await self.quality.get_daily_averages(start_date, end_date)
        dimension_stats = await self.quality.get_dimension_stats()

        if not daily_scores:
            return insights

        # Trend analysis
        if len(daily_scores) >= 3:
            trend_insight = self._analyze_quality_trend(daily_scores)
            if trend_insight:
                insights.append(trend_insight)

        # Dimension analysis
        dimension_insights = self._analyze_quality_dimensions(dimension_stats)
        insights.extend(dimension_insights)

        # Variability analysis
        variability_insight = self._analyze_quality_variability(daily_scores)
        if variability_insight:
            insights.append(variability_insight)

        return insights

    def _analyze_quality_trend(
        self,
        daily_scores: List[Dict[str, Any]]
    ) -> Optional[Insight]:
        """Analyze quality trend and generate insight"""
        if len(daily_scores) < 7:
            return None

        # Calculate trend
        composites = [d['composite'] for d in daily_scores if d.get('composite')]
        if len(composites) < 7:
            return None

        # Compare first half to second half
        mid = len(composites) // 2
        first_half_avg = statistics.mean(composites[:mid])
        second_half_avg = statistics.mean(composites[mid:])

        change = second_half_avg - first_half_avg
        change_pct = (change / first_half_avg * 100) if first_half_avg > 0 else 0

        # Determine significance
        if abs(change_pct) < 5:
            return None  # Not significant

        is_improving = change > 0
        priority = InsightPriority.HIGH if abs(change_pct) > 10 else InsightPriority.MEDIUM

        return Insight(
            insight_id=str(uuid.uuid4()),
            category=InsightCategory.QUALITY,
            priority=priority,
            title=f"Quality {'Improving' if is_improving else 'Declining'}",
            description=(
                f"Quality score has {'increased' if is_improving else 'decreased'} "
                f"by {abs(change_pct):.1f}% over the analysis period."
            ),
            evidence={
                'first_half_avg': first_half_avg,
                'second_half_avg': second_half_avg,
                'change_percent': change_pct,
                'data_points': len(composites)
            },
            confidence=min(0.6 + len(composites) * 0.02, 0.95),
            actionable=not is_improving,
            recommendation=(
                None if is_improving else
                "Review recent responses and identify factors causing quality decline"
            )
        )

    def _analyze_quality_dimensions(
        self,
        dimension_stats: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze individual quality dimensions"""
        insights = []
        threshold = self.config.analytics.quality_good_threshold

        dimensions = ['relevance', 'helpfulness', 'accuracy', 'clarity', 'completeness']

        for dim in dimensions:
            dim_data = dimension_stats.get(dim, {})
            avg = dim_data.get('avg', 0) if isinstance(dim_data, dict) else dim_data

            if avg > 0 and avg < threshold:
                gap = threshold - avg
                priority = (
                    InsightPriority.HIGH if gap > 0.2 else
                    InsightPriority.MEDIUM if gap > 0.1 else
                    InsightPriority.LOW
                )

                insights.append(Insight(
                    insight_id=str(uuid.uuid4()),
                    category=InsightCategory.QUALITY,
                    priority=priority,
                    title=f"Low {dim.title()} Score",
                    description=(
                        f"Average {dim} score ({avg:.2f}) is below target ({threshold:.2f}). "
                        f"Gap: {gap:.2f}"
                    ),
                    evidence={
                        'dimension': dim,
                        'average': avg,
                        'threshold': threshold,
                        'gap': gap
                    },
                    confidence=0.8,
                    actionable=True,
                    recommendation=self._get_dimension_recommendation(dim, gap)
                ))

        return insights

    def _analyze_quality_variability(
        self,
        daily_scores: List[Dict[str, Any]]
    ) -> Optional[Insight]:
        """Analyze quality score variability"""
        composites = [d['composite'] for d in daily_scores if d.get('composite')]

        if len(composites) < 5:
            return None

        std_dev = statistics.stdev(composites)
        mean = statistics.mean(composites)
        cv = (std_dev / mean) if mean > 0 else 0  # Coefficient of variation

        if cv > 0.2:  # High variability
            return Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.QUALITY,
                priority=InsightPriority.MEDIUM,
                title="High Quality Variability",
                description=(
                    f"Quality scores vary significantly (CV: {cv:.2f}). "
                    f"This inconsistency may affect user experience."
                ),
                evidence={
                    'std_dev': std_dev,
                    'mean': mean,
                    'coefficient_of_variation': cv
                },
                confidence=0.75,
                actionable=True,
                recommendation="Investigate causes of quality inconsistency across sessions"
            )

        return None

    # =========================================================================
    # Private Engagement Insight Methods
    # =========================================================================

    async def _generate_engagement_insights(
        self,
        start_date: date,
        end_date: date
    ) -> List[Insight]:
        """Generate engagement-related insights"""
        insights = []

        sessions = await self.feedback.get_sessions_in_range(start_date, end_date)

        if not sessions:
            return insights

        # Session frequency insight
        frequency_insight = self._analyze_session_frequency(sessions, start_date, end_date)
        if frequency_insight:
            insights.append(frequency_insight)

        # Session duration insight
        duration_insight = self._analyze_session_duration(sessions)
        if duration_insight:
            insights.append(duration_insight)

        # Peak time insight
        peak_insight = self._analyze_peak_times(sessions)
        if peak_insight:
            insights.append(peak_insight)

        return insights

    def _analyze_session_frequency(
        self,
        sessions: List[SessionData],
        start_date: date,
        end_date: date
    ) -> Optional[Insight]:
        """Analyze session frequency patterns"""
        days = (end_date - start_date).days or 1
        avg_sessions_per_day = len(sessions) / days

        if avg_sessions_per_day < 1:
            return Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.ENGAGEMENT,
                priority=InsightPriority.MEDIUM,
                title="Low Session Frequency",
                description=(
                    f"Average of {avg_sessions_per_day:.1f} sessions per day. "
                    "Consider strategies to increase engagement."
                ),
                evidence={
                    'total_sessions': len(sessions),
                    'days_analyzed': days,
                    'avg_per_day': avg_sessions_per_day
                },
                confidence=0.7,
                actionable=True,
                recommendation="Send reminders or provide incentives for regular usage"
            )
        elif avg_sessions_per_day > 5:
            return Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.ENGAGEMENT,
                priority=InsightPriority.LOW,
                title="High Engagement Detected",
                description=(
                    f"Strong engagement with {avg_sessions_per_day:.1f} sessions per day. "
                    "Users are actively using the system."
                ),
                evidence={
                    'total_sessions': len(sessions),
                    'days_analyzed': days,
                    'avg_per_day': avg_sessions_per_day
                },
                confidence=0.8,
                actionable=False
            )

        return None

    def _analyze_session_duration(
        self,
        sessions: List[SessionData]
    ) -> Optional[Insight]:
        """Analyze session duration patterns"""
        durations = [s.duration / 60 for s in sessions if s.duration > 0]

        if len(durations) < 5:
            return None

        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)

        # Short session warning
        if avg_duration < 2:
            return Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.ENGAGEMENT,
                priority=InsightPriority.MEDIUM,
                title="Short Session Duration",
                description=(
                    f"Average session is only {avg_duration:.1f} minutes. "
                    "Users may not be finding what they need."
                ),
                evidence={
                    'avg_duration_minutes': avg_duration,
                    'median_duration_minutes': median_duration,
                    'sample_size': len(durations)
                },
                confidence=0.75,
                actionable=True,
                recommendation="Improve initial response quality to encourage longer engagement"
            )

        return None

    def _analyze_peak_times(
        self,
        sessions: List[SessionData]
    ) -> Optional[Insight]:
        """Analyze peak usage times"""
        hour_counts = defaultdict(int)

        for session in sessions:
            hour_counts[session.start_time.hour] += 1

        if not hour_counts:
            return None

        peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])
        peak_count = hour_counts[peak_hour]

        if peak_count >= 5:
            return Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.ENGAGEMENT,
                priority=InsightPriority.LOW,
                title=f"Peak Usage at {peak_hour}:00",
                description=(
                    f"Highest activity ({peak_count} sessions) occurs around {peak_hour}:00. "
                    "Consider optimizing system resources for this time."
                ),
                evidence={
                    'peak_hour': peak_hour,
                    'peak_count': peak_count,
                    'distribution': dict(hour_counts)
                },
                confidence=0.7,
                actionable=True,
                recommendation=f"Ensure optimal performance during peak hours around {peak_hour}:00"
            )

        return None

    # =========================================================================
    # Private Preference Insight Methods
    # =========================================================================

    async def _generate_preference_insights(
        self,
        start_date: date,
        end_date: date
    ) -> List[Insight]:
        """Generate preference-related insights"""
        insights = []

        sessions = await self.feedback.get_sessions_in_range(start_date, end_date)
        feedback_list = await self.feedback.get_feedback_in_range(start_date, end_date)

        # Correction pattern insight
        correction_insight = self._analyze_correction_patterns(feedback_list)
        if correction_insight:
            insights.append(correction_insight)

        # Topic preference insight
        topic_insights = self._analyze_topic_preferences_for_insight(sessions)
        insights.extend(topic_insights)

        return insights

    def _analyze_correction_patterns(
        self,
        feedback_list: List[FeedbackData]
    ) -> Optional[Insight]:
        """Analyze user correction patterns"""
        corrections = [
            fb for fb in feedback_list
            if fb.feedback_type == FeedbackType.CORRECTION
        ]

        if len(corrections) < 3:
            return None

        # Look for common correction themes
        correction_rate = len(corrections) / len(feedback_list) if feedback_list else 0

        if correction_rate > 0.1:  # More than 10% are corrections
            return Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.PREFERENCE,
                priority=InsightPriority.HIGH,
                title="High Correction Rate",
                description=(
                    f"{correction_rate * 100:.1f}% of interactions require corrections. "
                    "User preferences may not be well understood."
                ),
                evidence={
                    'correction_count': len(corrections),
                    'total_feedback': len(feedback_list),
                    'correction_rate': correction_rate
                },
                confidence=0.8,
                actionable=True,
                recommendation="Review correction patterns to identify systematic response issues"
            )

        return None

    def _analyze_topic_preferences_for_insight(
        self,
        sessions: List[SessionData]
    ) -> List[Insight]:
        """Analyze topic preferences for insights"""
        insights = []

        topic_counts = defaultdict(int)
        for session in sessions:
            for topic in session.topics:
                topic_counts[topic] += 1

        if not topic_counts:
            return insights

        # Find dominant topics
        total_mentions = sum(topic_counts.values())
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        if sorted_topics and sorted_topics[0][1] / total_mentions > 0.3:
            top_topic, top_count = sorted_topics[0]
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.PREFERENCE,
                priority=InsightPriority.LOW,
                title=f"Strong Interest in '{top_topic}'",
                description=(
                    f"'{top_topic}' accounts for {top_count / total_mentions * 100:.1f}% "
                    f"of discussions. Consider enhancing content in this area."
                ),
                evidence={
                    'topic': top_topic,
                    'count': top_count,
                    'percentage': top_count / total_mentions * 100
                },
                confidence=0.75,
                actionable=True,
                recommendation=f"Expand knowledge base for '{top_topic}' related queries"
            ))

        return insights

    # =========================================================================
    # Private Pattern Insight Methods
    # =========================================================================

    async def _generate_pattern_insights(self) -> List[Insight]:
        """Generate insights from detected patterns"""
        insights = []

        # Get active patterns
        active_patterns = await self.patterns.get_active_patterns()

        for pattern in active_patterns[:5]:  # Top 5 patterns
            if pattern.get('frequency', 0) >= 3:
                insight = Insight(
                    insight_id=str(uuid.uuid4()),
                    category=InsightCategory.PATTERN,
                    priority=InsightPriority.MEDIUM,
                    title=f"Pattern: {pattern.get('pattern_type', 'Unknown')}",
                    description=(
                        f"Detected recurring pattern with frequency {pattern.get('frequency')}. "
                        f"Representative: '{pattern.get('representative_text', '')[:50]}...'"
                    ),
                    evidence={
                        'pattern_id': pattern.get('pattern_id'),
                        'pattern_type': pattern.get('pattern_type'),
                        'frequency': pattern.get('frequency'),
                        'confidence': pattern.get('confidence')
                    },
                    confidence=pattern.get('confidence', 0.5),
                    actionable=True,
                    recommendation=self._get_pattern_recommendation(pattern)
                )
                insights.append(insight)

        return insights

    # =========================================================================
    # Private Improvement Insight Methods
    # =========================================================================

    async def _generate_improvement_insights(
        self,
        start_date: date,
        end_date: date
    ) -> List[Insight]:
        """Generate improvement-focused insights"""
        insights = []

        # Get feedback for improvement opportunities
        feedback_list = await self.feedback.get_feedback_in_range(start_date, end_date)

        # Clarification analysis
        clarifications = [
            fb for fb in feedback_list
            if fb.feedback_type == FeedbackType.CLARIFICATION
        ]

        if len(clarifications) >= 3:
            clarification_rate = len(clarifications) / len(feedback_list) if feedback_list else 0
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                category=InsightCategory.IMPROVEMENT,
                priority=InsightPriority.MEDIUM,
                title="Clarification Requests",
                description=(
                    f"{len(clarifications)} clarification requests detected "
                    f"({clarification_rate * 100:.1f}% of interactions). "
                    "Responses may need more initial detail."
                ),
                evidence={
                    'clarification_count': len(clarifications),
                    'clarification_rate': clarification_rate
                },
                confidence=0.75,
                actionable=True,
                recommendation="Provide more comprehensive initial responses"
            ))

        return insights

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_dimension_recommendation(self, dimension: str, gap: float) -> str:
        """Get recommendation for improving a quality dimension"""
        recommendations = {
            'relevance': "Focus on understanding user intent better; use clarifying questions",
            'helpfulness': "Provide more actionable information and practical examples",
            'accuracy': "Verify information before responding; cite sources when possible",
            'clarity': "Use simpler language and structure responses with clear sections",
            'completeness': "Anticipate follow-up questions and address them proactively"
        }
        return recommendations.get(dimension, f"Improve {dimension} through targeted training")

    def _get_pattern_recommendation(self, pattern: Dict[str, Any]) -> str:
        """Get recommendation based on pattern type"""
        pattern_type = pattern.get('pattern_type', '')

        recommendations = {
            'recurring_question': "Consider adding this to FAQ or proactive suggestions",
            'common_correction': "Update response templates to avoid this correction",
            'engagement_time': "Optimize system availability for peak engagement times",
            'topic_cluster': "Expand knowledge base for this topic cluster",
            'quality_correlation': "Apply learnings from high-quality interactions"
        }

        return recommendations.get(pattern_type, "Review and address this pattern")

    def _create_dimension_recommendation(
        self,
        dimension: str,
        current_avg: float,
        threshold: float,
        stats: Dict[str, Any]
    ) -> Optional[ImprovementRecommendation]:
        """Create improvement recommendation for a dimension"""
        actions = {
            'relevance': [
                "Implement better query understanding",
                "Add context extraction from conversation history",
                "Use semantic matching for topic identification"
            ],
            'helpfulness': [
                "Include practical examples in responses",
                "Add step-by-step instructions where applicable",
                "Suggest related topics proactively"
            ],
            'accuracy': [
                "Cross-reference responses with knowledge base",
                "Implement fact-checking pipeline",
                "Add confidence scores to uncertain responses"
            ],
            'clarity': [
                "Use structured response formats",
                "Simplify complex explanations",
                "Add visual aids or formatting for clarity"
            ],
            'completeness': [
                "Anticipate follow-up questions",
                "Include all relevant aspects in initial response",
                "Provide links to related information"
            ]
        }

        return ImprovementRecommendation(
            recommendation_id=str(uuid.uuid4()),
            area=dimension,
            current_state=f"Average {dimension} score: {current_avg:.2f}",
            target_state=f"Target {dimension} score: {threshold:.2f}",
            actions=actions.get(dimension, [f"Improve {dimension}"]),
            expected_impact="Medium to High",
            effort_level="medium",
            confidence=0.7,
            evidence={'current_avg': current_avg, 'threshold': threshold}
        )

    def _analyze_feedback_for_recommendations(
        self,
        feedback_list: List[FeedbackData]
    ) -> List[ImprovementRecommendation]:
        """Analyze feedback to generate recommendations"""
        recommendations = []

        # Group by type
        by_type = defaultdict(int)
        for fb in feedback_list:
            by_type[fb.feedback_type.value] += 1

        total = len(feedback_list)
        if total == 0:
            return recommendations

        # High negative feedback rate
        negative = by_type.get('explicit_negative', 0) + by_type.get('implicit_negative', 0)
        if negative / total > 0.2:
            recommendations.append(ImprovementRecommendation(
                recommendation_id=str(uuid.uuid4()),
                area="user_satisfaction",
                current_state=f"{negative / total * 100:.1f}% negative feedback",
                target_state="< 10% negative feedback",
                actions=[
                    "Review negative feedback patterns",
                    "Identify common issues",
                    "Implement targeted improvements"
                ],
                expected_impact="High",
                effort_level="high",
                confidence=0.8,
                evidence={'negative_count': negative, 'total': total}
            ))

        return recommendations

    def _analyze_length_preferences(
        self,
        sessions: List[SessionData]
    ) -> Optional[PersonalizationSuggestion]:
        """Analyze response length preferences"""
        # This would analyze actual response lengths and user reactions
        # Simplified implementation
        if len(sessions) < 5:
            return None

        avg_exchanges = statistics.mean([s.exchange_count for s in sessions])

        if avg_exchanges > 5:
            return PersonalizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                dimension="response_length",
                current_setting="standard",
                suggested_setting="detailed",
                reason="User tends to have extended conversations, may prefer comprehensive responses",
                confidence=0.65,
                supporting_data={'avg_exchanges': avg_exchanges}
            )

        return None

    def _analyze_timing_preferences(
        self,
        sessions: List[SessionData]
    ) -> Optional[PersonalizationSuggestion]:
        """Analyze timing preferences"""
        if len(sessions) < 5:
            return None

        # Find most common hour
        hour_counts = defaultdict(int)
        for s in sessions:
            hour_counts[s.start_time.hour] += 1

        if not hour_counts:
            return None

        peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])

        return PersonalizationSuggestion(
            suggestion_id=str(uuid.uuid4()),
            dimension="optimal_time",
            current_setting="any_time",
            suggested_setting=f"peak_{peak_hour}",
            reason=f"User is most active around {peak_hour}:00",
            confidence=0.6,
            supporting_data={'peak_hour': peak_hour, 'distribution': dict(hour_counts)}
        )

    def _analyze_topic_preferences(
        self,
        sessions: List[SessionData]
    ) -> List[PersonalizationSuggestion]:
        """Analyze topic preferences"""
        suggestions = []

        topic_counts = defaultdict(int)
        for session in sessions:
            for topic in session.topics:
                topic_counts[topic] += 1

        if not topic_counts:
            return suggestions

        # Top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        for topic, count in sorted_topics[:3]:
            if count >= 3:
                suggestions.append(PersonalizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    dimension="topic_preference",
                    current_setting="general",
                    suggested_setting=f"emphasis_{topic}",
                    reason=f"User shows strong interest in '{topic}' ({count} sessions)",
                    confidence=min(0.5 + count * 0.05, 0.9),
                    supporting_data={'topic': topic, 'count': count}
                ))

        return suggestions

    def _analyze_style_preferences(
        self,
        feedback_list: List[FeedbackData]
    ) -> List[PersonalizationSuggestion]:
        """Analyze communication style preferences from feedback"""
        suggestions = []

        # Count positive vs negative feedback
        positive = sum(
            1 for fb in feedback_list
            if fb.feedback_type in [FeedbackType.EXPLICIT_POSITIVE, FeedbackType.IMPLICIT_POSITIVE]
        )
        negative = sum(
            1 for fb in feedback_list
            if fb.feedback_type in [FeedbackType.EXPLICIT_NEGATIVE, FeedbackType.IMPLICIT_NEGATIVE]
        )

        total = positive + negative
        if total < 5:
            return suggestions

        positive_rate = positive / total

        if positive_rate > 0.8:
            suggestions.append(PersonalizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                dimension="communication_style",
                current_setting="standard",
                suggested_setting="current_style",
                reason="Current style is well-received (high positive feedback)",
                confidence=0.7,
                supporting_data={'positive_rate': positive_rate}
            ))

        return suggestions

    async def _save_insight(self, insight: Insight) -> None:
        """Save insight to store"""
        try:
            await self.insights_store.save_insight({
                'insight_id': insight.insight_id,
                'category': insight.category.value,
                'title': insight.title,
                'description': insight.description,
                'evidence': insight.evidence,
                'confidence': insight.confidence,
                'actionable': insight.actionable,
                'recommendation': insight.recommendation,
                'created_at': insight.created_at.isoformat(),
                'valid_until': insight.valid_until.isoformat() if insight.valid_until else None,
                'status': 'active'
            })
        except Exception as e:
            db_logger.error("save_insight_failed", insight_id=insight.insight_id, error=str(e))


# Global insights generator instance
insights_generator = InsightsGenerator()
