"""
Learning System Configuration - Adaptive Response Improvement

PATTERN: Centralized configuration with sensible defaults
WHY: Easy tuning of learning parameters without code changes

Combines analytics configuration with preference learning settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


# =============================================================================
# Analytics Configuration (existing)
# =============================================================================

@dataclass
class AnalyticsConfig:
    """Configuration for learning analytics"""

    # Trend calculation
    rolling_window_days: int = 7
    trend_min_data_points: int = 3

    # Quality thresholds
    quality_excellent_threshold: float = 0.85
    quality_good_threshold: float = 0.70
    quality_poor_threshold: float = 0.50

    # Report generation
    daily_report_hour: int = 0  # UTC hour for daily reports
    weekly_report_day: int = 0  # Monday

    # Data retention
    raw_data_retention_days: int = 90
    aggregated_data_retention_days: int = 365


@dataclass
class PatternDetectionConfig:
    """Configuration for pattern detection"""

    # Clustering parameters
    similarity_threshold: float = 0.85
    min_cluster_size: int = 3
    max_clusters: int = 50

    # Frequency analysis
    min_frequency_for_pattern: int = 3
    time_window_hours: int = 168  # 1 week

    # Correlation analysis
    correlation_significance_threshold: float = 0.3
    min_samples_for_correlation: int = 10


@dataclass
class InsightsConfig:
    """Configuration for insights generation"""

    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.70
    low_confidence_threshold: float = 0.50

    # Statistical significance
    min_sample_size: int = 10
    p_value_threshold: float = 0.05

    # Insight limits
    max_insights_per_category: int = 5
    max_total_insights: int = 20

    # Categories
    insight_categories: List[str] = field(default_factory=lambda: [
        "quality", "engagement", "preference", "pattern", "improvement"
    ])


@dataclass
class MetricsConfig:
    """Configuration for metrics aggregation"""

    # Aggregation intervals
    aggregation_intervals: List[str] = field(default_factory=lambda: [
        "hourly", "daily", "weekly", "monthly"
    ])

    # Batch processing
    batch_size: int = 1000
    aggregation_interval_minutes: int = 60

    # Dashboard export
    dashboard_metrics: List[str] = field(default_factory=lambda: [
        "total_sessions",
        "total_exchanges",
        "avg_quality_score",
        "positive_feedback_rate",
        "avg_session_duration",
        "active_topics"
    ])


# =============================================================================
# Preference Learning Configuration (Phase 5)
# =============================================================================

@dataclass
class PreferenceLearningConfig:
    """Configuration for user preference learning"""

    # Learning rate (exponential moving average alpha)
    learning_rate: float = 0.1

    # Minimum interactions before preferences are reliable
    min_interactions_for_confidence: int = 5

    # Rate at which old preferences decay
    preference_decay_rate: float = 0.01

    # Response length thresholds (word count)
    short_response_threshold: int = 75
    medium_response_max: int = 200
    long_response_threshold: int = 200

    # Technical depth levels
    depth_levels: List[str] = field(default_factory=lambda: [
        "beginner", "intermediate", "expert"
    ])

    # Communication styles
    communication_styles: List[str] = field(default_factory=lambda: [
        "formal", "balanced", "casual"
    ])

    # Topic tracking
    max_tracked_topics: int = 20
    topic_relevance_threshold: float = 0.3

    # Vocabulary learning
    enable_vocabulary_learning: bool = True
    max_vocabulary_adjustments: int = 50

    # Preference storage
    preference_file_path: str = "preferences.json"

    # Preference categories
    preference_categories: List[str] = field(default_factory=lambda: [
        "response_length", "detail_level", "formality", "example_frequency"
    ])


@dataclass
class ABTestingConfig:
    """Configuration for A/B testing improvements"""

    # Traffic split for treatment group
    treatment_split: float = 0.5

    # Minimum samples before evaluating results
    min_samples_for_significance: int = 30

    # Statistical significance threshold
    p_value_threshold: float = 0.05

    # Effect size threshold for practical significance
    min_effect_size: float = 0.05

    # Maximum concurrent experiments
    max_concurrent_experiments: int = 3


@dataclass
class ImprovementConfig:
    """Configuration for improvement engine"""

    # Quality thresholds
    quality_drop_threshold: float = 0.10  # 10% drop triggers rollback
    min_quality_score: float = 0.30

    # Update frequency
    update_after_interactions: int = 10
    improvement_check_interval: int = 50

    # Rollback settings
    enable_auto_rollback: bool = True
    rollback_observation_period: int = 20  # interactions

    # Hypothesis generation
    max_hypotheses_per_dimension: int = 3

    # Weak area thresholds
    weak_area_threshold: float = 0.60


@dataclass
class AdaptationConfig:
    """Configuration for response adaptation"""

    # Maximum adjustments per response
    max_response_adjustments: int = 3

    # Confidence thresholds for applying adaptations
    min_confidence_for_adaptation: float = 0.6

    # Prompt adaptation
    enable_prompt_adaptation: bool = True
    enable_context_enhancement: bool = True
    enable_response_calibration: bool = True

    # Cache settings
    preference_cache_ttl: int = 300  # seconds


# =============================================================================
# Master Configuration
# =============================================================================

@dataclass
class QualityScoringConfig:
    """Configuration for quality scoring system (Phase 5)"""

    # Dimension weights (must sum to 1.0)
    relevance_weight: float = 0.30
    helpfulness_weight: float = 0.25
    engagement_weight: float = 0.20
    clarity_weight: float = 0.15
    accuracy_weight: float = 0.10

    # Thresholds
    high_quality_threshold: float = 0.75
    low_quality_threshold: float = 0.40
    default_score: float = 0.5

    # Engagement parameters
    expected_follow_ups: int = 3
    session_depth_target: int = 5
    optimal_duration_minutes: int = 12

    # Clarity parameters
    target_grade_level: float = 8.0
    min_text_length: int = 10

    # Feedback parameters
    feedback_recency_weight: float = 0.8
    min_feedback_count: int = 1


@dataclass
class FeedbackCollectionConfig:
    """Configuration for feedback collection in learning system."""

    implicit_feedback_enabled: bool = True
    engagement_tracking_enabled: bool = True
    correction_detection_enabled: bool = True
    correction_similarity_threshold: float = 0.7
    aggregation_window_hours: int = 24


@dataclass
class LearningConfig:
    """Master configuration for learning system"""

    # Database paths
    db_path: str = "learning_analytics.db"
    learning_db_path: str = "learning_data.db"

    # Sub-configurations - Analytics
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    patterns: PatternDetectionConfig = field(default_factory=PatternDetectionConfig)
    insights: InsightsConfig = field(default_factory=InsightsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Sub-configurations - Adaptive Learning (Phase 5)
    preferences: PreferenceLearningConfig = field(default_factory=PreferenceLearningConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    improvement: ImprovementConfig = field(default_factory=ImprovementConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

    # Feedback collection configuration
    feedback: FeedbackCollectionConfig = field(default_factory=FeedbackCollectionConfig)

    # Quality scoring configuration (Phase 5)
    quality_scoring: QualityScoringConfig = field(default_factory=QualityScoringConfig)

    # Feature flags
    enable_real_time_analytics: bool = True
    enable_pattern_detection: bool = True
    enable_insight_generation: bool = True
    enable_adaptive_learning: bool = True
    enable_ab_testing: bool = True

    # Performance tuning
    cache_ttl_seconds: int = 300
    max_concurrent_analyses: int = 5

    # Data retention
    max_feedback_history: int = 1000
    max_improvement_history: int = 100

    @property
    def preference_learning(self) -> PreferenceLearningConfig:
        """Alias for preferences - for backward compatibility"""
        return self.preferences


# Global configuration instance
learning_config = LearningConfig()


# =============================================================================
# Feedback System Configuration (for feedback_store.py compatibility)
# =============================================================================

@dataclass
class FeedbackConfig:
    """Configuration for the feedback collection system (Phase 5)."""

    # Database settings
    database_path: str = "feedback.db"

    # Rating configuration
    min_rating: int = 1
    max_rating: int = 5

    # Implicit feedback thresholds
    quick_response_threshold_ms: int = 2000
    slow_response_threshold_ms: int = 10000
    engaged_session_threshold_seconds: int = 300

    # Correction detection
    correction_edit_distance_threshold: float = 0.3
    min_correction_length: int = 5

    # Buffer settings
    implicit_buffer_size: int = 100
    implicit_buffer_flush_interval_seconds: int = 30

    # Aggregation settings
    default_time_range_hours: int = 24

    # Privacy settings
    anonymize_sessions: bool = False
    max_comment_length: int = 1000

    # Learning trigger settings
    learning_trigger_threshold: int = 10
    enable_real_time_learning: bool = True

    # Feedback collection settings
    implicit_feedback_enabled: bool = True
    engagement_tracking_enabled: bool = True
    correction_detection_enabled: bool = True
    correction_similarity_threshold: float = 0.7
    aggregation_window_hours: int = 24

    def get_rating_range(self) -> tuple:
        """Get the valid rating range as a tuple."""
        return (self.min_rating, self.max_rating)

    def is_quick_response(self, response_time_ms: int) -> bool:
        """Check if a response time is considered 'quick'."""
        return response_time_ms <= self.quick_response_threshold_ms

    def is_slow_response(self, response_time_ms: int) -> bool:
        """Check if a response time is considered 'slow'."""
        return response_time_ms >= self.slow_response_threshold_ms

    def is_engaged_session(self, duration_seconds: int) -> bool:
        """Check if a session duration indicates engagement."""
        return duration_seconds >= self.engaged_session_threshold_seconds

    def is_correction(self, edit_distance_ratio: float) -> bool:
        """Check if an edit distance ratio indicates a correction."""
        return edit_distance_ratio >= self.correction_edit_distance_threshold


# Global feedback configuration instance
feedback_config = FeedbackConfig()
