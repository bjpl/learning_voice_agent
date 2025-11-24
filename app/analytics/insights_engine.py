"""
Insights Engine
===============

Generates actionable insights from learning data.

PATTERN: Statistical analysis with domain-specific rules
WHY: Transform raw data into actionable recommendations
SPARC: Evidence-based insights with confidence scoring

Enhanced Methods (Phase 6):
- generate_daily_insights(): Daily learning insights with streak, quality, trends
- generate_weekly_summary(): Weekly recap with comparisons and achievements
- detect_anomalies(): Z-score based unusual pattern detection
- identify_milestones(): Achievement and milestone detection
- get_recommendations(): Personalized suggestions based on patterns
- get_comparisons(): Period-over-period analysis (WoW, MoM)
"""

import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import uuid
from dataclasses import dataclass, field

from app.logger import db_logger
from app.analytics.analytics_config import AnalyticsEngineConfig, analytics_config
from app.analytics.progress_models import (
    ProgressMetrics,
    LearningStreak,
    TopicMastery,
    DailyProgress,
    TrendDirection,
    ProgressLevel,
)
from app.analytics.insights_models import (
    Insight as TypedInsight,
    InsightCategory,
    TrendData,
    TrendDirection as ModelTrendDirection,
    Anomaly,
    AnomalySeverity,
    Recommendation,
    RecommendationPriority,
    LearningStreak as TypedLearningStreak,
    PeriodComparison,
    Milestone,
    MilestoneType,
    InsightSummary,
    create_achievement_insight,
    create_attention_insight,
    create_improvement_insight,
)


class Insight:
    """Represents a generated insight."""

    def __init__(
        self,
        title: str,
        description: str,
        category: str,
        priority: str = "medium",
        confidence: float = 0.7,
        recommendations: List[str] = None,
        supporting_data: Dict[str, Any] = None
    ):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.category = category
        self.priority = priority  # "low", "medium", "high"
        self.confidence = confidence
        self.recommendations = recommendations or []
        self.supporting_data = supporting_data or {}
        self.generated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "supporting_data": self.supporting_data,
            "generated_at": self.generated_at.isoformat()
        }


class InsightsEngine:
    """
    Generate actionable insights from learning data.

    PATTERN: Rule-based insight generation with statistical validation
    WHY: Provide meaningful feedback to improve learning outcomes

    USAGE:
        engine = InsightsEngine()
        await engine.initialize()

        # Generate all insights
        insights = await engine.generate_insights(metrics, daily_progress)

        # Generate specific insight types
        anomalies = await engine.detect_anomalies(daily_progress)
        milestones = await engine.identify_milestones(metrics)
    """

    def __init__(
        self,
        config: Optional[AnalyticsEngineConfig] = None,
        progress_tracker: Optional[Any] = None
    ):
        """
        Initialize insights engine.

        Args:
            config: Analytics configuration
            progress_tracker: Progress tracker instance
        """
        self.config = config or analytics_config
        self.progress_tracker = progress_tracker
        self._initialized = False
        self._insight_history: List[Insight] = []

    async def initialize(self) -> None:
        """Initialize insights engine."""
        if self._initialized:
            return

        try:
            db_logger.info("insights_engine_initializing")
            self._initialized = True
            db_logger.info("insights_engine_initialized")

        except Exception as e:
            db_logger.error(
                "insights_engine_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def generate_insights(
        self,
        metrics: ProgressMetrics,
        daily_progress: List[DailyProgress] = None,
        topic_mastery: Dict[str, TopicMastery] = None,
        streak: LearningStreak = None
    ) -> List[Insight]:
        """
        Generate all insights from available data.

        Args:
            metrics: Overall progress metrics
            daily_progress: List of daily progress records
            topic_mastery: Dictionary of topic mastery levels
            streak: Learning streak data

        Returns:
            List of generated insights
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_logger.info("generating_insights")
            insights: List[Insight] = []

            # Progress insights
            progress_insights = await self._generate_progress_insights(metrics)
            insights.extend(progress_insights)

            # Quality insights
            if daily_progress:
                quality_insights = await self._generate_quality_insights(daily_progress)
                insights.extend(quality_insights)

                # Anomaly detection
                anomalies = await self.detect_anomalies(daily_progress)
                insights.extend(anomalies)

            # Streak insights
            if streak:
                streak_insights = await self._generate_streak_insights(streak)
                insights.extend(streak_insights)

            # Topic insights
            if topic_mastery:
                topic_insights = await self._generate_topic_insights(topic_mastery)
                insights.extend(topic_insights)

            # Milestone insights
            milestones = await self.identify_milestones(metrics)
            insights.extend(milestones)

            # Sort by priority and confidence
            priority_order = {"high": 0, "medium": 1, "low": 2}
            insights.sort(key=lambda i: (priority_order.get(i.priority, 1), -i.confidence))

            # Limit to max insights
            max_insights = self.config.insights.max_recommendations
            insights = insights[:max_insights]

            # Store in history
            self._insight_history.extend(insights)

            db_logger.info(
                "insights_generated",
                count=len(insights)
            )

            return insights

        except Exception as e:
            db_logger.error(
                "generate_insights_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def detect_anomalies(
        self,
        daily_progress: List[DailyProgress]
    ) -> List[Insight]:
        """
        Detect anomalies in learning patterns.

        Args:
            daily_progress: List of daily progress records

        Returns:
            List of anomaly insights
        """
        if not self._initialized:
            await self.initialize()

        try:
            insights: List[Insight] = []

            if len(daily_progress) < self.config.insights.anomaly_min_samples:
                return insights

            # Calculate statistics
            quality_scores = [
                d.avg_quality_score for d in daily_progress
                if d.avg_quality_score > 0
            ]
            session_counts = [d.total_sessions for d in daily_progress]
            exchange_counts = [d.total_exchanges for d in daily_progress]

            if quality_scores:
                mean_quality = statistics.mean(quality_scores)
                std_quality = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0

                # Check recent quality for anomalies
                recent = daily_progress[-7:] if len(daily_progress) >= 7 else daily_progress
                for day in recent:
                    if day.avg_quality_score > 0 and std_quality > 0:
                        z_score = (day.avg_quality_score - mean_quality) / std_quality

                        if z_score < -self.config.insights.anomaly_std_threshold:
                            insights.append(Insight(
                                title="Quality Drop Detected",
                                description=f"Quality score on {day.date} was significantly below average ({day.avg_quality_score:.2f} vs {mean_quality:.2f}).",
                                category="quality",
                                priority="high",
                                confidence=min(0.95, 0.5 + abs(z_score) * 0.1),
                                recommendations=[
                                    "Review session recordings to identify issues",
                                    "Check for difficult topics that need more attention",
                                    "Consider adjusting learning pace"
                                ],
                                supporting_data={
                                    "date": str(day.date),
                                    "score": day.avg_quality_score,
                                    "mean": mean_quality,
                                    "z_score": z_score
                                }
                            ))
                        elif z_score > self.config.insights.anomaly_std_threshold:
                            insights.append(Insight(
                                title="Exceptional Performance",
                                description=f"Quality score on {day.date} was significantly above average ({day.avg_quality_score:.2f} vs {mean_quality:.2f}).",
                                category="quality",
                                priority="medium",
                                confidence=min(0.95, 0.5 + abs(z_score) * 0.1),
                                recommendations=[
                                    "Analyze what contributed to this success",
                                    "Try to replicate the conditions in future sessions"
                                ],
                                supporting_data={
                                    "date": str(day.date),
                                    "score": day.avg_quality_score,
                                    "mean": mean_quality,
                                    "z_score": z_score
                                }
                            ))

            # Activity anomalies
            if session_counts and len(session_counts) > 1:
                mean_sessions = statistics.mean(session_counts)
                recent_sessions = sum(d.total_sessions for d in recent)

                if mean_sessions > 0 and recent_sessions == 0:
                    insights.append(Insight(
                        title="Activity Gap Detected",
                        description="No sessions recorded in the past week, compared to your usual activity level.",
                        category="engagement",
                        priority="high",
                        confidence=0.85,
                        recommendations=[
                            "Set a small goal to get back on track",
                            "Start with a short 5-minute session",
                            "Review your schedule for available learning time"
                        ],
                        supporting_data={
                            "usual_sessions_per_day": mean_sessions,
                            "recent_sessions": recent_sessions
                        }
                    ))

            return insights

        except Exception as e:
            db_logger.error(
                "detect_anomalies_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def identify_milestones(
        self,
        metrics: ProgressMetrics
    ) -> List[Insight]:
        """
        Identify reached milestones.

        Args:
            metrics: Progress metrics

        Returns:
            List of milestone insights
        """
        if not self._initialized:
            await self.initialize()

        try:
            insights: List[Insight] = []
            intervals = self.config.insights.milestone_intervals

            # Session milestones
            for milestone in intervals:
                if metrics.sessions_count >= milestone and metrics.sessions_count < milestone + 10:
                    insights.append(Insight(
                        title=f"Milestone: {milestone} Sessions",
                        description=f"Congratulations! You've completed {milestone} learning sessions.",
                        category="milestone",
                        priority="medium",
                        confidence=1.0,
                        recommendations=[
                            "Keep up the great work!",
                            f"Your next milestone is {milestone * 2} sessions"
                        ],
                        supporting_data={"sessions_count": metrics.sessions_count}
                    ))
                    break

            # Exchange milestones
            exchange_milestones = [m * 10 for m in intervals]
            for milestone in exchange_milestones:
                if metrics.total_exchanges >= milestone and metrics.total_exchanges < milestone + 50:
                    insights.append(Insight(
                        title=f"Milestone: {milestone} Exchanges",
                        description=f"You've had {milestone} learning exchanges! Great engagement.",
                        category="milestone",
                        priority="medium",
                        confidence=1.0,
                        recommendations=["Continue exploring new topics"],
                        supporting_data={"exchanges_count": metrics.total_exchanges}
                    ))
                    break

            # Time milestones (in hours)
            time_milestones = [1, 5, 10, 25, 50, 100]
            for milestone in time_milestones:
                if metrics.total_time_hours >= milestone and metrics.total_time_hours < milestone + 1:
                    insights.append(Insight(
                        title=f"Milestone: {milestone} Hours of Learning",
                        description=f"You've invested {milestone} hours in learning!",
                        category="milestone",
                        priority="medium",
                        confidence=1.0,
                        recommendations=["Consider reviewing what you've learned"],
                        supporting_data={"total_hours": metrics.total_time_hours}
                    ))
                    break

            # Streak milestones
            streak_milestones = [3, 7, 14, 30, 60, 100, 365]
            for milestone in streak_milestones:
                if metrics.current_streak >= milestone and metrics.current_streak < milestone + 3:
                    insights.append(Insight(
                        title=f"Streak Milestone: {milestone} Days",
                        description=f"Amazing! You've maintained a {milestone}-day learning streak!",
                        category="streak",
                        priority="high",
                        confidence=1.0,
                        recommendations=[
                            "Keep the momentum going",
                            "Share your achievement with others"
                        ],
                        supporting_data={"current_streak": metrics.current_streak}
                    ))
                    break

            return insights

        except Exception as e:
            db_logger.error(
                "identify_milestones_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def generate_recommendations(
        self,
        metrics: ProgressMetrics,
        topic_mastery: Dict[str, TopicMastery] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations.

        Args:
            metrics: Progress metrics
            topic_mastery: Topic mastery data

        Returns:
            List of recommendations
        """
        if not self._initialized:
            await self.initialize()

        try:
            recommendations = []

            # Activity recommendations
            if metrics.sessions_count < 5:
                recommendations.append({
                    "type": "engagement",
                    "title": "Build Your Learning Habit",
                    "description": "Try to complete at least one session per day to build consistency.",
                    "priority": "high"
                })

            # Quality recommendations
            if metrics.avg_quality_score < 0.6:
                recommendations.append({
                    "type": "quality",
                    "title": "Focus on Understanding",
                    "description": "Take more time with each response to improve comprehension.",
                    "priority": "high"
                })
            elif metrics.avg_quality_score >= 0.85:
                recommendations.append({
                    "type": "quality",
                    "title": "Challenge Yourself",
                    "description": "Your quality scores are excellent. Try more advanced topics!",
                    "priority": "medium"
                })

            # Streak recommendations
            if metrics.current_streak == 0:
                recommendations.append({
                    "type": "streak",
                    "title": "Start a Streak",
                    "description": "Begin a learning streak today! Consistency is key.",
                    "priority": "medium"
                })
            elif metrics.current_streak >= 7:
                recommendations.append({
                    "type": "streak",
                    "title": "Keep Your Streak Alive",
                    "description": f"You're on a {metrics.current_streak}-day streak! Don't break it.",
                    "priority": "high"
                })

            # Topic recommendations
            if topic_mastery:
                weak_topics = [
                    topic for topic, mastery in topic_mastery.items()
                    if mastery.level == ProgressLevel.BEGINNER and mastery.total_interactions >= 3
                ]
                if weak_topics:
                    recommendations.append({
                        "type": "topic",
                        "title": "Review Challenging Topics",
                        "description": f"Consider revisiting: {', '.join(weak_topics[:3])}",
                        "priority": "medium"
                    })

                mastered_count = sum(
                    1 for m in topic_mastery.values()
                    if m.level in [ProgressLevel.ADVANCED, ProgressLevel.EXPERT]
                )
                if mastered_count > 0:
                    recommendations.append({
                        "type": "topic",
                        "title": "Expand Your Knowledge",
                        "description": f"You've mastered {mastered_count} topics. Explore related areas!",
                        "priority": "low"
                    })

            return recommendations[:self.config.insights.max_recommendations]

        except Exception as e:
            db_logger.error(
                "generate_recommendations_failed",
                error=str(e),
                exc_info=True
            )
            return []

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _generate_progress_insights(
        self,
        metrics: ProgressMetrics
    ) -> List[Insight]:
        """Generate insights from progress metrics."""
        insights = []

        # Learning velocity insight
        if metrics.learning_velocity > 0:
            if metrics.learning_velocity > 10:  # More than 10 exchanges per hour
                insights.append(Insight(
                    title="High Learning Velocity",
                    description=f"You're averaging {metrics.learning_velocity:.1f} exchanges per hour. Great pace!",
                    category="progress",
                    priority="low",
                    confidence=0.8,
                    recommendations=["Consider taking breaks to process information"]
                ))
            elif metrics.learning_velocity < 2:
                insights.append(Insight(
                    title="Opportunity for More Engagement",
                    description="Your learning velocity is low. Try to engage more actively.",
                    category="progress",
                    priority="medium",
                    confidence=0.75,
                    recommendations=[
                        "Ask more follow-up questions",
                        "Explore topics more deeply"
                    ]
                ))

        # Topics insight
        if metrics.topics_explored > 10 and metrics.topics_mastered < metrics.topics_explored * 0.3:
            insights.append(Insight(
                title="Consider Focusing Your Learning",
                description="You've explored many topics but mastered few. Consider focusing on key areas.",
                category="progress",
                priority="medium",
                confidence=0.7,
                recommendations=[
                    "Choose 2-3 topics to focus on",
                    "Complete more sessions on each topic"
                ]
            ))

        return insights

    async def _generate_quality_insights(
        self,
        daily_progress: List[DailyProgress]
    ) -> List[Insight]:
        """Generate quality-related insights."""
        insights = []

        if len(daily_progress) < 3:
            return insights

        # Recent quality trend
        recent = daily_progress[-7:] if len(daily_progress) >= 7 else daily_progress
        recent_scores = [d.avg_quality_score for d in recent if d.avg_quality_score > 0]

        if len(recent_scores) >= 3:
            # Check for improvement or decline
            first_half = statistics.mean(recent_scores[:len(recent_scores)//2])
            second_half = statistics.mean(recent_scores[len(recent_scores)//2:])

            change = second_half - first_half
            if change > 0.1:
                insights.append(Insight(
                    title="Quality Improving",
                    description=f"Your quality scores have improved by {change*100:.1f}% recently.",
                    category="quality",
                    priority="low",
                    confidence=0.8,
                    recommendations=["Keep up the great work!"]
                ))
            elif change < -0.1:
                insights.append(Insight(
                    title="Quality Declining",
                    description=f"Your quality scores have declined by {abs(change)*100:.1f}% recently.",
                    category="quality",
                    priority="high",
                    confidence=0.8,
                    recommendations=[
                        "Review recent sessions for patterns",
                        "Take more time with responses",
                        "Ask for clarification when needed"
                    ]
                ))

        return insights

    async def _generate_streak_insights(
        self,
        streak: LearningStreak
    ) -> List[Insight]:
        """Generate streak-related insights."""
        insights = []

        # Streak at risk
        if streak.last_active_date:
            days_since = (date.today() - streak.last_active_date).days
            if days_since == 1 and streak.current_streak > 3:
                insights.append(Insight(
                    title="Streak at Risk!",
                    description=f"Your {streak.current_streak}-day streak will end if you don't learn today!",
                    category="streak",
                    priority="high",
                    confidence=1.0,
                    recommendations=["Complete a quick session to maintain your streak"]
                ))

        # Personal best
        if streak.current_streak > 0 and streak.current_streak == streak.longest_streak:
            insights.append(Insight(
                title="Personal Best Streak!",
                description=f"You're at your longest ever streak: {streak.current_streak} days!",
                category="streak",
                priority="medium",
                confidence=1.0,
                recommendations=["Keep going to set a new record!"]
            ))

        return insights

    async def _generate_topic_insights(
        self,
        topic_mastery: Dict[str, TopicMastery]
    ) -> List[Insight]:
        """Generate topic-related insights."""
        insights = []

        if not topic_mastery:
            return insights

        # Find newly mastered topics
        mastered = [
            (topic, mastery) for topic, mastery in topic_mastery.items()
            if mastery.level in [ProgressLevel.ADVANCED, ProgressLevel.EXPERT]
        ]

        if mastered:
            top_topic = max(mastered, key=lambda x: x[1].mastery_score)
            insights.append(Insight(
                title=f"Excelling in {top_topic[0]}",
                description=f"Your mastery of {top_topic[0]} is at {top_topic[1].mastery_score*100:.0f}%!",
                category="topic",
                priority="low",
                confidence=0.9,
                recommendations=["Consider exploring related advanced topics"]
            ))

        # Find topics needing attention
        struggling = [
            (topic, mastery) for topic, mastery in topic_mastery.items()
            if mastery.total_interactions >= 5 and mastery.level == ProgressLevel.BEGINNER
        ]

        if struggling:
            weak_topic = max(struggling, key=lambda x: x[1].total_interactions)
            insights.append(Insight(
                title=f"Review Recommended: {weak_topic[0]}",
                description=f"You've engaged with {weak_topic[0]} {weak_topic[1].total_interactions} times but mastery is low.",
                category="topic",
                priority="medium",
                confidence=0.75,
                recommendations=[
                    "Try a different approach to this topic",
                    "Break it down into smaller concepts",
                    "Ask more clarifying questions"
                ]
            ))

        return insights


    # =========================================================================
    # PHASE 6: ENHANCED INSIGHT GENERATION
    # =========================================================================

    async def generate_daily_insights(
        self,
        target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive daily insights.

        PATTERN: Multi-signal insight aggregation
        WHY: Provide holistic view of daily learning progress

        Args:
            target_date: Date to generate insights for (default: today)

        Returns:
            Dictionary with insights, trends, anomalies, recommendations
        """
        if not self._initialized:
            await self.initialize()

        if target_date is None:
            target_date = date.today()

        try:
            db_logger.info("generating_daily_insights", date=str(target_date))

            # Import trend analyzer
            from app.analytics.trend_analyzer import trend_analyzer

            insights: List[TypedInsight] = []
            anomalies: List[Anomaly] = []
            milestones: List[Milestone] = []

            # 1. Get streak data and generate streak insights
            streak_data = await self._get_streak_from_progress_tracker()
            streak_insights = await self._generate_typed_streak_insights(streak_data)
            insights.extend(streak_insights)

            # 2. Calculate key trends
            trends = {}
            for metric in ["quality_score", "exchange_count", "session_count"]:
                try:
                    trends[metric] = await trend_analyzer.calculate_trend(metric, days=7)
                except Exception as e:
                    db_logger.warning(f"trend_calculation_failed", metric=metric, error=str(e))

            # 3. Generate trend-based insights
            trend_insights = self._generate_trend_based_insights(trends)
            insights.extend(trend_insights)

            # 4. Check for quality insights today
            quality_insights = await self._generate_daily_quality_insights(target_date)
            insights.extend(quality_insights)

            # 5. Detect anomalies
            detected_anomalies = await self._detect_typed_anomalies()
            anomalies.extend(detected_anomalies)
            anomaly_insights = self._convert_anomalies_to_insights(anomalies)
            insights.extend(anomaly_insights)

            # 6. Identify milestones
            detected_milestones = await self._identify_typed_milestones()
            milestones.extend(detected_milestones)
            milestone_insights = self._convert_milestones_to_insights(milestones)
            insights.extend(milestone_insights)

            # 7. Generate recommendations
            recommendations = await self.get_recommendations_for_daily()

            # 8. Calculate health score
            health_score = self._calculate_health_score(
                trends, anomalies, streak_data
            )

            # Sort insights by importance
            insights.sort(key=lambda i: i.importance, reverse=True)

            result = {
                "date": target_date.isoformat(),
                "insights": [i.to_dict() for i in insights[:15]],
                "trends": {k: v.to_dict() if hasattr(v, 'to_dict') else {} for k, v in trends.items()},
                "anomalies": [a.to_dict() for a in anomalies],
                "milestones": [m.to_dict() for m in milestones],
                "recommendations": recommendations[:5],
                "streak": {
                    "current_streak": streak_data.current_streak if streak_data else 0,
                    "longest_streak": streak_data.longest_streak if streak_data else 0,
                } if streak_data else None,
                "health_score": round(health_score, 1),
                "generated_at": datetime.utcnow().isoformat()
            }

            db_logger.info(
                "daily_insights_generated",
                date=str(target_date),
                insights_count=len(insights),
                health_score=round(health_score, 1)
            )

            return result

        except Exception as e:
            db_logger.error(
                "generate_daily_insights_failed",
                date=str(target_date),
                error=str(e),
                exc_info=True
            )
            return {
                "date": target_date.isoformat() if target_date else date.today().isoformat(),
                "insights": [],
                "trends": {},
                "anomalies": [],
                "milestones": [],
                "recommendations": [],
                "streak": None,
                "health_score": 50.0,
                "generated_at": datetime.utcnow().isoformat()
            }

    async def generate_weekly_summary(
        self,
        week_end: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Generate weekly learning summary.

        PATTERN: Aggregated period analysis with comparisons
        WHY: Provide weekly progress overview and momentum check

        Args:
            week_end: End of the week (default: today)

        Returns:
            Dictionary with weekly summary data
        """
        if not self._initialized:
            await self.initialize()

        if week_end is None:
            week_end = date.today()

        week_start = week_end - timedelta(days=6)

        try:
            db_logger.info(
                "generating_weekly_summary",
                start=str(week_start),
                end=str(week_end)
            )

            from app.analytics.trend_analyzer import trend_analyzer
            from app.analytics.progress_tracker import progress_tracker

            # Get session metrics for the week
            try:
                weekly_metrics = await progress_tracker.get_progress_metrics()
            except (AttributeError, RuntimeError, ValueError):
                # AttributeError: progress_tracker not properly initialized
                # RuntimeError: async context or database issues
                # ValueError: invalid parameter values
                weekly_metrics = None

            # Calculate totals from daily progress
            total_sessions = 0
            total_exchanges = 0
            quality_scores = []

            if self.progress_tracker:
                try:
                    daily_progress_list = await self.progress_tracker.get_daily_progress(days=7)
                    for dp in daily_progress_list:
                        total_sessions += dp.total_sessions
                        total_exchanges += dp.total_exchanges
                        if dp.avg_quality_score > 0:
                            quality_scores.append(dp.avg_quality_score)
                except (AttributeError, RuntimeError, TypeError):
                    # AttributeError: progress_tracker missing method
                    # RuntimeError: async issues
                    # TypeError: unexpected data structure
                    pass

            avg_quality = statistics.mean(quality_scores) if quality_scores else 0.5

            # Calculate trends
            quality_trend = await trend_analyzer.calculate_trend("quality_score", days=7)
            engagement_trend = await trend_analyzer.calculate_trend("exchange_count", days=7)

            # Get week-over-week comparisons
            comparisons = await self.get_comparisons()

            # Generate key insights
            key_insights = []

            # Session volume insight
            if total_sessions >= 7:
                key_insights.append({
                    "type": "achievement",
                    "title": "Active Learning Week",
                    "description": f"You completed {total_sessions} learning sessions this week!",
                    "importance": 3
                })
            elif total_sessions == 0:
                key_insights.append({
                    "type": "attention",
                    "title": "No Activity This Week",
                    "description": "You haven't had any learning sessions this week.",
                    "importance": 4
                })

            # Quality insight
            if avg_quality >= 0.85:
                key_insights.append({
                    "type": "achievement",
                    "title": "Excellent Week for Quality",
                    "description": f"Average quality score of {avg_quality:.0%} - outstanding!",
                    "importance": 4
                })

            # Trend insight
            if quality_trend.direction == ModelTrendDirection.INCREASING and quality_trend.magnitude > 10:
                key_insights.append({
                    "type": "improvement",
                    "title": "Strong Quality Growth",
                    "description": f"Quality improved {quality_trend.magnitude:.0f}% week-over-week!",
                    "importance": 4
                })

            # Generate recommendations for next week
            recommendations = []
            if quality_trend.direction == ModelTrendDirection.DECREASING:
                recommendations.append({
                    "action": "Focus on quality improvement next week",
                    "rationale": f"Quality declined {quality_trend.magnitude:.0f}% this week",
                    "priority": "high"
                })

            if engagement_trend.direction == ModelTrendDirection.DECREASING:
                recommendations.append({
                    "action": "Schedule regular learning sessions",
                    "rationale": "Engagement dropped this week",
                    "priority": "medium"
                })

            result = {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "total_sessions": total_sessions,
                "total_exchanges": total_exchanges,
                "avg_quality": round(avg_quality, 4),
                "quality_trend": quality_trend.to_dict() if hasattr(quality_trend, 'to_dict') else {},
                "engagement_trend": engagement_trend.to_dict() if hasattr(engagement_trend, 'to_dict') else {},
                "key_insights": key_insights,
                "comparisons": comparisons,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }

            db_logger.info(
                "weekly_summary_generated",
                sessions=total_sessions,
                avg_quality=round(avg_quality, 2)
            )

            return result

        except Exception as e:
            db_logger.error(
                "generate_weekly_summary_failed",
                error=str(e),
                exc_info=True
            )
            return {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "total_sessions": 0,
                "total_exchanges": 0,
                "avg_quality": 0.5,
                "quality_trend": {},
                "engagement_trend": {},
                "key_insights": [],
                "comparisons": {},
                "recommendations": [],
                "generated_at": datetime.utcnow().isoformat()
            }

    async def get_comparisons(
        self,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get week-over-week and month-over-month comparisons.

        Args:
            metrics: Specific metrics to compare (default: all key metrics)

        Returns:
            Dictionary of metric comparisons
        """
        if not self._initialized:
            await self.initialize()

        if metrics is None:
            metrics = ["quality_score", "exchange_count", "session_count"]

        try:
            from app.analytics.trend_analyzer import trend_analyzer

            comparisons = {}

            for metric in metrics:
                try:
                    # Week-over-week
                    wow = await trend_analyzer.compare_periods(
                        metric, "last_week", "this_week"
                    )
                    comparisons[f"{metric}_wow"] = wow.to_dict() if hasattr(wow, 'to_dict') else {
                        "metric": metric,
                        "period1_label": "Last Week",
                        "period2_label": "This Week",
                        "percent_change": 0.0
                    }

                    # Month-over-month
                    mom = await trend_analyzer.compare_periods(
                        metric, "last_month", "this_month"
                    )
                    comparisons[f"{metric}_mom"] = mom.to_dict() if hasattr(mom, 'to_dict') else {
                        "metric": metric,
                        "period1_label": "Last Month",
                        "period2_label": "This Month",
                        "percent_change": 0.0
                    }
                except Exception as e:
                    db_logger.warning(f"comparison_failed", metric=metric, error=str(e))

            return comparisons

        except Exception as e:
            db_logger.error("get_comparisons_failed", error=str(e), exc_info=True)
            return {}

    async def get_insights_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get aggregated insights summary for dashboard.

        Args:
            days: Period to summarize

        Returns:
            InsightSummary-like dictionary
        """
        if not self._initialized:
            await self.initialize()

        try:
            daily_report = await self.generate_daily_insights()

            # Count by category
            by_category = defaultdict(int)
            for insight in daily_report.get("insights", []):
                category = insight.get("category", "unknown")
                by_category[category] += 1

            # Count by importance
            by_importance = defaultdict(int)
            for insight in daily_report.get("insights", []):
                importance = insight.get("importance", 3)
                by_importance[importance] += 1

            return {
                "generated_at": datetime.utcnow().isoformat(),
                "period_start": (date.today() - timedelta(days=days)).isoformat(),
                "period_end": date.today().isoformat(),
                "total_insights": len(daily_report.get("insights", [])),
                "by_category": dict(by_category),
                "by_importance": dict(by_importance),
                "top_insights": daily_report.get("insights", [])[:5],
                "trends_summary": daily_report.get("trends", {}),
                "anomalies_count": len(daily_report.get("anomalies", [])),
                "recommendations_count": len(daily_report.get("recommendations", [])),
                "health_score": daily_report.get("health_score", 50.0)
            }

        except Exception as e:
            db_logger.error("get_insights_summary_failed", error=str(e), exc_info=True)
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "period_start": (date.today() - timedelta(days=days)).isoformat(),
                "period_end": date.today().isoformat(),
                "total_insights": 0,
                "by_category": {},
                "by_importance": {},
                "top_insights": [],
                "trends_summary": {},
                "anomalies_count": 0,
                "recommendations_count": 0,
                "health_score": 50.0
            }

    async def get_recommendations_for_daily(
        self,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for daily insights.

        This is a wrapper that works with or without a progress tracker.

        Args:
            max_recommendations: Maximum number of recommendations

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        try:
            # Try to use existing generate_recommendations if progress tracker available
            if self.progress_tracker:
                try:
                    metrics = await self.progress_tracker.get_progress_metrics()
                    if metrics:
                        recs = await self.generate_recommendations(metrics)
                        return recs[:max_recommendations]
                except Exception:
                    pass

            # Generate basic recommendations without metrics
            streak = await self._get_streak_from_progress_tracker()

            if streak and streak.current_streak == 0:
                recommendations.append({
                    "type": "streak",
                    "title": "Start a Streak",
                    "description": "Begin a learning streak today! Consistency is key.",
                    "priority": "medium"
                })

            if not recommendations:
                recommendations.append({
                    "type": "engagement",
                    "title": "Keep Learning",
                    "description": "Continue your learning journey with regular sessions.",
                    "priority": "medium"
                })

            return recommendations[:max_recommendations]

        except Exception as e:
            db_logger.warning("get_recommendations_for_daily_failed", error=str(e))
            return []

    # =========================================================================
    # PHASE 6: HELPER METHODS
    # =========================================================================

    async def _get_streak_from_progress_tracker(self) -> Optional[LearningStreak]:
        """Get streak data from progress tracker"""
        try:
            if self.progress_tracker:
                return await self.progress_tracker.get_learning_streak()
            return None
        except Exception as e:
            db_logger.warning("get_streak_failed", error=str(e))
            return None

    async def _generate_typed_streak_insights(
        self,
        streak: Optional[LearningStreak]
    ) -> List[TypedInsight]:
        """Generate insights based on streak status using typed models"""
        insights = []

        if not streak:
            return insights

        # Streak milestone celebration
        streak_milestones = [3, 7, 14, 30, 60, 90, 100, 180, 365]
        if streak.current_streak in streak_milestones:
            insights.append(create_achievement_insight(
                title=f"Streak Milestone: {streak.current_streak} Days!",
                description=f"Congratulations! You've reached a {streak.current_streak}-day learning streak!",
                importance=5 if streak.current_streak >= 30 else 4,
                metric_name="streak",
                metric_value=float(streak.current_streak)
            ))

        # Close to next milestone
        for milestone in streak_milestones:
            if milestone > streak.current_streak:
                days_until = milestone - streak.current_streak
                if days_until <= 3:
                    insights.append(TypedInsight.create(
                        category=InsightCategory.SUGGESTION,
                        title=f"Almost at {milestone}-Day Streak!",
                        description=f"Just {days_until} more days to reach your next streak milestone!",
                        importance=3,
                        actionable=True
                    ))
                break

        # Streak at risk
        if streak.last_active_date:
            days_since = (date.today() - streak.last_active_date).days
            if days_since >= 1 and streak.current_streak > 3:
                insights.append(create_attention_insight(
                    title="Streak at Risk!",
                    description=f"Your {streak.current_streak}-day streak is at risk. Complete a session today!",
                    importance=4,
                    recommendation="Start a quick learning session to maintain your streak"
                ))

        # Personal best streak
        if streak.current_streak > 0 and streak.current_streak == streak.longest_streak:
            insights.append(create_achievement_insight(
                title="New Personal Best Streak!",
                description=f"Your {streak.current_streak}-day streak is your longest ever!",
                importance=4
            ))

        return insights

    def _generate_trend_based_insights(
        self,
        trends: Dict[str, TrendData]
    ) -> List[TypedInsight]:
        """Generate insights from trend analysis"""
        insights = []

        quality_trend = trends.get("quality_score")
        if quality_trend and hasattr(quality_trend, 'is_significant') and quality_trend.is_significant:
            if quality_trend.direction == ModelTrendDirection.INCREASING:
                insights.append(create_improvement_insight(
                    title="Quality Improving",
                    description=f"Your conversation quality has improved by {quality_trend.magnitude:.1f}% over the past week!",
                    metric_name="quality_score",
                    metric_value=quality_trend.magnitude,
                    importance=3
                ))
            elif quality_trend.direction == ModelTrendDirection.DECREASING:
                insights.append(create_attention_insight(
                    title="Quality Declining",
                    description=f"Your conversation quality has decreased by {quality_trend.magnitude:.1f}% this week.",
                    importance=4,
                    recommendation="Review recent sessions and identify areas for improvement"
                ))

        engagement_trend = trends.get("exchange_count")
        if engagement_trend and hasattr(engagement_trend, 'is_significant') and engagement_trend.is_significant:
            if engagement_trend.direction == ModelTrendDirection.INCREASING:
                insights.append(create_improvement_insight(
                    title="Engagement Growing",
                    description=f"Your engagement has increased by {engagement_trend.magnitude:.1f}%!",
                    metric_name="engagement",
                    metric_value=engagement_trend.magnitude,
                    importance=3
                ))

        return insights

    async def _generate_daily_quality_insights(
        self,
        target_date: date
    ) -> List[TypedInsight]:
        """Generate quality-based insights for a specific day"""
        insights = []

        try:
            if self.progress_tracker:
                daily_progress_list = await self.progress_tracker.get_daily_progress(days=1)
                if daily_progress_list:
                    dp = daily_progress_list[-1]
                    avg_quality = dp.avg_quality_score

                    if avg_quality >= 0.85:
                        insights.append(create_achievement_insight(
                            title="Excellent Quality Day!",
                            description=f"Today's average quality score is {avg_quality:.0%} - outstanding!",
                            importance=3,
                            metric_name="quality_score",
                            metric_value=avg_quality
                        ))
                    elif avg_quality < 0.50 and avg_quality > 0:
                        insights.append(create_attention_insight(
                            title="Quality Below Target",
                            description=f"Today's quality score ({avg_quality:.0%}) is below the target of 70%.",
                            importance=4,
                            recommendation="Focus on providing more relevant and detailed responses"
                        ))

        except Exception as e:
            db_logger.warning("generate_quality_insights_failed", error=str(e))

        return insights

    async def _detect_typed_anomalies(self) -> List[Anomaly]:
        """Detect anomalies using typed models"""
        anomalies = []

        try:
            from app.analytics.trend_analyzer import trend_analyzer

            metrics = [
                ("quality_score", True),   # (metric, higher_is_better)
                ("exchange_count", True),
            ]

            for metric, higher_is_better in metrics:
                try:
                    trend = await trend_analyzer.calculate_trend(metric, days=30)

                    if len(trend.data_points) < 5:
                        continue

                    today_value = trend.data_points[-1] if trend.data_points else None
                    if today_value is None:
                        continue

                    if trend.std_dev and trend.std_dev > 0 and trend.avg_value:
                        z_score = (today_value - trend.avg_value) / trend.std_dev

                        if abs(z_score) >= 2.0:
                            is_negative = (
                                (z_score < 0 and higher_is_better) or
                                (z_score > 0 and not higher_is_better)
                            )

                            anomalies.append(Anomaly.create(
                                metric=metric,
                                value=today_value,
                                expected_value=trend.avg_value,
                                z_score=z_score,
                                is_negative=is_negative
                            ))

                except Exception as e:
                    db_logger.warning(f"anomaly_detection_failed", metric=metric, error=str(e))

        except Exception as e:
            db_logger.error("detect_typed_anomalies_failed", error=str(e))

        return anomalies

    async def _identify_typed_milestones(self) -> List[Milestone]:
        """Identify milestones using typed models"""
        milestones = []

        try:
            # Check streak milestones
            streak = await self._get_streak_from_progress_tracker()
            if streak:
                streak_milestones = [3, 7, 14, 30, 60, 90, 100, 180, 365]
                if streak.current_streak in streak_milestones:
                    milestones.append(Milestone.create(
                        type=MilestoneType.STREAK,
                        title=f"{streak.current_streak}-Day Learning Streak!",
                        description=f"You've maintained consistent learning for {streak.current_streak} consecutive days!",
                        value=float(streak.current_streak),
                        threshold=float(streak.current_streak)
                    ))

            # Check volume milestones
            if self.progress_tracker:
                try:
                    metrics = await self.progress_tracker.get_progress_metrics()
                    if metrics:
                        volume_thresholds = [10, 25, 50, 100, 250, 500, 1000]
                        for threshold in volume_thresholds:
                            if metrics.sessions_count >= threshold and metrics.sessions_count < threshold + 5:
                                milestones.append(Milestone.create(
                                    type=MilestoneType.VOLUME,
                                    title=f"{threshold} Sessions Completed!",
                                    description=f"You've completed {metrics.sessions_count} learning sessions!",
                                    value=float(metrics.sessions_count),
                                    threshold=float(threshold)
                                ))
                                break
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    # AttributeError: progress_tracker missing method
                    # RuntimeError: async issues
                    # TypeError: unexpected data structure
                    # ValueError: invalid metric values
                    pass

        except Exception as e:
            db_logger.error("identify_typed_milestones_failed", error=str(e))

        return milestones

    def _convert_anomalies_to_insights(
        self,
        anomalies: List[Anomaly]
    ) -> List[TypedInsight]:
        """Convert anomalies to typed insights"""
        insights = []
        for anomaly in anomalies:
            category = (
                InsightCategory.ATTENTION if anomaly.is_negative
                else InsightCategory.DISCOVERY
            )
            insights.append(TypedInsight.create(
                category=category,
                title=anomaly.title,
                description=anomaly.description,
                importance=4 if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL] else 3,
                metric_name=anomaly.metric,
                metric_value=anomaly.value,
                evidence={"z_score": anomaly.z_score, "expected": anomaly.expected_value}
            ))
        return insights

    def _convert_milestones_to_insights(
        self,
        milestones: List[Milestone]
    ) -> List[TypedInsight]:
        """Convert milestones to typed insights"""
        insights = []
        for milestone in milestones:
            insights.append(TypedInsight.create(
                category=InsightCategory.ACHIEVEMENT,
                title=milestone.title,
                description=milestone.description,
                importance=min(5, milestone.celebration_level + 2),
                metric_name=milestone.type.value,
                metric_value=milestone.value
            ))
        return insights

    def _calculate_health_score(
        self,
        trends: Dict[str, TrendData],
        anomalies: List[Anomaly],
        streak: Optional[LearningStreak]
    ) -> float:
        """Calculate overall learning health score (0-100)"""
        score = 50.0  # Baseline

        # Quality trend contribution (+/- 15)
        quality_trend = trends.get("quality_score")
        if quality_trend and hasattr(quality_trend, 'direction'):
            if quality_trend.direction == ModelTrendDirection.INCREASING:
                score += min(15, quality_trend.magnitude * 0.5)
            elif quality_trend.direction == ModelTrendDirection.DECREASING:
                score -= min(15, quality_trend.magnitude * 0.5)

        # Streak contribution (+20 max)
        if streak and streak.current_streak > 0:
            streak_bonus = min(20, streak.current_streak * 2)
            score += streak_bonus

        # Anomaly penalty (-5 per negative anomaly)
        negative_anomalies = sum(1 for a in anomalies if a.is_negative)
        score -= negative_anomalies * 5

        # Engagement contribution (+15 max)
        engagement_trend = trends.get("exchange_count")
        if engagement_trend and hasattr(engagement_trend, 'direction'):
            if engagement_trend.direction == ModelTrendDirection.INCREASING:
                score += min(15, engagement_trend.magnitude * 0.3)

        return max(0, min(100, score))


# Global insights engine instance
insights_engine = InsightsEngine()
