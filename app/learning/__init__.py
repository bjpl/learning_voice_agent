"""
Learning Module - Adaptive Response Improvement System

This module provides intelligent adaptation of AI responses based on
collected feedback, quality scores, and learned user preferences.

Components:
- QualityScorer: Multi-dimensional quality scoring engine (Phase 5)
- ResponseAdapter: Customizes prompts and calibrates responses
- PreferenceLearner: Learns user preferences from feedback and engagement
- ImprovementEngine: Continuous improvement through A/B testing

Quality Scoring (Phase 5):
- QualityScore: Multi-dimensional score with composite calculation
- SessionQuality: Aggregated session-level metrics
- QualityTrend: Time-series trend analysis
- ImprovementArea: Identified areas for improvement

Analytics & Pattern Detection (Phase 5):
- LearningAnalytics: Session/daily reports, trends, knowledge gaps
- PatternDetector: Recurring patterns, quality correlations
- InsightsGenerator: Actionable insights and recommendations
- MetricsAggregator: Dashboard-ready metrics

PATTERN: Adaptive learning with exponential moving average updates
WHY: Gradual, safe improvements that can be rolled back if quality drops

API ENDPOINTS:
    GET /api/analytics/daily - Daily learning report
    GET /api/analytics/trends - Quality trends over time
    GET /api/analytics/patterns - Detected patterns
    GET /api/analytics/insights - Generated insights
    GET /api/analytics/dashboard - Dashboard data export
"""

from app.learning.config import LearningConfig, learning_config
from app.learning.store import LearningStore, learning_store
from app.learning.adapter import ResponseAdapter
from app.learning.preference_learner import PreferenceLearner
from app.learning.improvement_engine import ImprovementEngine, Improvement

# Quality Scoring (Phase 5)
from app.learning.scoring_models import (
    QualityScore,
    SessionQuality,
    QualityTrend,
    ImprovementArea,
    QualityLevel,
    ScoreDimension,
)
from app.learning.scoring_algorithms import (
    RelevanceScorer,
    EngagementScorer,
    ClarityScorer,
    HelpfulnessScorer,
    AccuracyScorer,
    CompositeScoreCalculator,
    ScoringConfig,
    SessionMetrics,
    FeedbackData,
    scoring_config,
)
from app.learning.quality_scorer import (
    QualityScorer,
    QualityScoringMiddleware,
    create_quality_scorer,
    quick_score,
)

# Analytics & Pattern Detection (Phase 5)
# Note: These imports use try/except for graceful degradation
try:
    from app.learning.stores import (
        FeedbackType,
        SessionData,
        FeedbackData as AnalyticsFeedbackData,
        QualityScore as AnalyticsQualityScore,
        FeedbackStore,
        QualityStore,
        PatternStore,
        InsightStore,
        feedback_store,
        quality_store,
        pattern_store,
        insight_store,
    )
    _STORES_AVAILABLE = True
except ImportError as e:
    _STORES_AVAILABLE = False
    FeedbackType = None
    SessionData = None
    AnalyticsFeedbackData = None
    AnalyticsQualityScore = None
    FeedbackStore = None
    QualityStore = None
    PatternStore = None
    InsightStore = None
    feedback_store = None
    quality_store = None
    pattern_store = None
    insight_store = None

try:
    from app.learning.analytics import (
        LearningAnalytics,
        SessionReport,
        DailyReport,
        QualityTrend as AnalyticsQualityTrend,
        learning_analytics,
    )
    _ANALYTICS_AVAILABLE = True
except ImportError as e:
    _ANALYTICS_AVAILABLE = False
    LearningAnalytics = None
    SessionReport = None
    DailyReport = None
    AnalyticsQualityTrend = None
    learning_analytics = None

try:
    from app.learning.pattern_detector import (
        PatternDetector,
        PatternType,
        DetectedPattern,
        Cluster,
        Correlation,
        pattern_detector,
    )
    _PATTERN_DETECTOR_AVAILABLE = True
except ImportError as e:
    _PATTERN_DETECTOR_AVAILABLE = False
    PatternDetector = None
    PatternType = None
    DetectedPattern = None
    Cluster = None
    Correlation = None
    pattern_detector = None

try:
    from app.learning.insights_generator import (
        InsightsGenerator,
        InsightCategory,
        InsightPriority,
        Insight,
        ImprovementRecommendation,
        PersonalizationSuggestion,
        insights_generator,
    )
    _INSIGHTS_AVAILABLE = True
except ImportError as e:
    _INSIGHTS_AVAILABLE = False
    InsightsGenerator = None
    InsightCategory = None
    InsightPriority = None
    Insight = None
    ImprovementRecommendation = None
    PersonalizationSuggestion = None
    insights_generator = None

try:
    from app.learning.metrics_aggregator import (
        MetricsAggregator,
        AggregationInterval,
        AggregatedMetrics,
        DashboardData,
        metrics_aggregator,
    )
    _METRICS_AVAILABLE = True
except ImportError as e:
    _METRICS_AVAILABLE = False
    MetricsAggregator = None
    AggregationInterval = None
    AggregatedMetrics = None
    DashboardData = None
    metrics_aggregator = None

__all__ = [
    # Configuration
    "LearningConfig",
    "learning_config",
    # Storage
    "LearningStore",
    "learning_store",
    # Core Classes
    "ResponseAdapter",
    "PreferenceLearner",
    "ImprovementEngine",
    "Improvement",
    # Quality Scoring - Data Models (Phase 5)
    "QualityScore",
    "SessionQuality",
    "QualityTrend",
    "ImprovementArea",
    "QualityLevel",
    "ScoreDimension",
    # Quality Scoring - Algorithms (Phase 5)
    "RelevanceScorer",
    "EngagementScorer",
    "ClarityScorer",
    "HelpfulnessScorer",
    "AccuracyScorer",
    "CompositeScoreCalculator",
    "ScoringConfig",
    "SessionMetrics",
    "FeedbackData",
    "scoring_config",
    # Quality Scoring - Main Classes (Phase 5)
    "QualityScorer",
    "QualityScoringMiddleware",
    "create_quality_scorer",
    "quick_score",
    # Analytics - Data Types (Phase 5)
    "FeedbackType",
    "SessionData",
    "AnalyticsFeedbackData",
    "AnalyticsQualityScore",
    # Analytics - Stores (Phase 5)
    "FeedbackStore",
    "QualityStore",
    "PatternStore",
    "InsightStore",
    "feedback_store",
    "quality_store",
    "pattern_store",
    "insight_store",
    # Analytics - Main Classes (Phase 5)
    "LearningAnalytics",
    "SessionReport",
    "DailyReport",
    "AnalyticsQualityTrend",
    "learning_analytics",
    # Pattern Detection (Phase 5)
    "PatternDetector",
    "PatternType",
    "DetectedPattern",
    "Cluster",
    "Correlation",
    "pattern_detector",
    # Insights Generation (Phase 5)
    "InsightsGenerator",
    "InsightCategory",
    "InsightPriority",
    "Insight",
    "ImprovementRecommendation",
    "PersonalizationSuggestion",
    "insights_generator",
    # Metrics Aggregation (Phase 5)
    "MetricsAggregator",
    "AggregationInterval",
    "AggregatedMetrics",
    "DashboardData",
    "metrics_aggregator",
]

__version__ = "2.1.0"  # Phase 5 - Analytics & Pattern Detection


async def initialize_learning_system():
    """
    Initialize all learning system components

    USAGE:
        from app.learning import initialize_learning_system
        await initialize_learning_system()
    """
    if _ANALYTICS_AVAILABLE and learning_analytics:
        await learning_analytics.initialize()
    if _INSIGHTS_AVAILABLE and insights_generator:
        await insights_generator.initialize()
    if _METRICS_AVAILABLE and metrics_aggregator:
        await metrics_aggregator.initialize()


async def run_daily_aggregation():
    """
    Run daily aggregation tasks

    Typically called by scheduler at midnight UTC
    """
    from datetime import date, timedelta

    yesterday = date.today() - timedelta(days=1)

    # Aggregate daily metrics
    if _METRICS_AVAILABLE and metrics_aggregator:
        await metrics_aggregator.aggregate_daily(yesterday)

    # Generate insights
    if _INSIGHTS_AVAILABLE and insights_generator:
        await insights_generator.generate_insights(time_range_days=7)

    # Expire old insights
    if _STORES_AVAILABLE and insight_store:
        await insight_store.expire_old_insights()
