"""
Analytics Engine Configuration
==============================

Configuration classes for the Phase 6 analytics engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ProgressTrackingConfig:
    """Configuration for progress tracking."""

    # Streak settings
    streak_grace_period_hours: int = 36  # Hours before streak breaks
    max_streak_history: int = 100

    # Topic mastery
    mastery_threshold: float = 0.75
    expert_threshold: float = 0.90
    min_interactions_for_mastery: int = 10

    # Learning velocity
    velocity_window_hours: int = 24
    velocity_min_interactions: int = 5

    # Progress snapshots
    snapshot_interval_hours: int = 24
    max_snapshots: int = 365

    # Caching
    cache_ttl_seconds: int = 300


@dataclass
class InsightsConfig:
    """Configuration for insights generation."""

    # Insight generation
    min_data_points: int = 5
    confidence_threshold: float = 0.7

    # Anomaly detection
    anomaly_std_threshold: float = 2.0
    anomaly_min_samples: int = 10

    # Milestone detection
    milestone_intervals: List[int] = field(default_factory=lambda: [
        10, 25, 50, 100, 250, 500, 1000
    ])

    # Recommendation limits
    max_recommendations: int = 5
    recommendation_refresh_hours: int = 24

    # Categories
    insight_categories: List[str] = field(default_factory=lambda: [
        "progress", "quality", "engagement", "streak", "topic", "goal"
    ])


@dataclass
class TrendAnalyzerConfig:
    """Configuration for trend analysis."""

    # Rolling averages
    short_term_window: int = 7  # days
    medium_term_window: int = 30  # days
    long_term_window: int = 90  # days

    # Trend detection
    trend_significance_threshold: float = 0.05  # 5% change
    trend_min_data_points: int = 3

    # Seasonality
    detect_seasonality: bool = True
    seasonality_period_days: int = 7

    # Forecasting
    enable_forecasting: bool = True
    forecast_days: int = 7
    forecast_confidence_interval: float = 0.95


@dataclass
class DashboardConfig:
    """Configuration for dashboard service."""

    # Overview settings
    overview_period_days: int = 30

    # Chart settings
    quality_chart_days: int = 30
    activity_heatmap_weeks: int = 12

    # Goals display
    max_active_goals: int = 5

    # Achievements display
    recent_achievements_count: int = 5

    # Topic display
    max_topics_displayed: int = 10

    # Insights
    max_insights_displayed: int = 5

    # Caching
    cache_ttl_seconds: int = 300
    stale_cache_ttl_seconds: int = 600

    # Performance
    max_data_points_per_chart: int = 100


@dataclass
class GoalConfig:
    """Configuration for goal tracking."""

    # Goal limits
    max_active_goals: int = 10
    max_daily_goals: int = 3
    max_weekly_goals: int = 5

    # Default goals
    default_daily_sessions: int = 1
    default_daily_exchanges: int = 10
    default_daily_minutes: int = 15

    # Suggested goals
    enable_goal_suggestions: bool = True
    suggestion_based_on_history_days: int = 14

    # Goal types with default durations
    goal_type_durations: Dict[str, int] = field(default_factory=lambda: {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "custom": 0
    })

    # Points for completing goals
    goal_completion_points: Dict[str, int] = field(default_factory=lambda: {
        "daily": 10,
        "weekly": 50,
        "monthly": 200,
        "custom": 25
    })


@dataclass
class AchievementConfig:
    """Configuration for achievement system."""

    # Achievement tiers
    tier_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "bronze": 1.0,
        "silver": 2.0,
        "gold": 3.0,
        "platinum": 5.0
    })

    # Base points per tier
    base_points: Dict[str, int] = field(default_factory=lambda: {
        "bronze": 10,
        "silver": 25,
        "gold": 50,
        "platinum": 100
    })

    # Achievement check frequency
    check_frequency_interactions: int = 1

    # Notification settings
    notify_on_unlock: bool = True
    celebrate_milestones: bool = True

    # Categories
    categories: List[str] = field(default_factory=lambda: [
        "streak", "sessions", "quality", "topics", "goals", "time", "social"
    ])


@dataclass
class ExportConfig:
    """Configuration for data export."""

    # Export formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "json", "csv", "pdf"
    ])

    # Data limits
    max_records_per_export: int = 10000
    max_export_size_mb: int = 50

    # Report settings
    report_include_charts: bool = True
    report_include_insights: bool = True
    report_include_recommendations: bool = True

    # Date range limits
    max_date_range_days: int = 365

    # Privacy
    anonymize_by_default: bool = False
    redact_sensitive_fields: List[str] = field(default_factory=lambda: [
        "user_id", "session_id"
    ])

    # Output
    output_dir: str = "exports"
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class AnalyticsEngineConfig:
    """Master configuration for analytics engine."""

    # Database
    db_path: str = "analytics.db"

    # Sub-configurations
    progress: ProgressTrackingConfig = field(default_factory=ProgressTrackingConfig)
    insights: InsightsConfig = field(default_factory=InsightsConfig)
    trends: TrendAnalyzerConfig = field(default_factory=TrendAnalyzerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    goals: GoalConfig = field(default_factory=GoalConfig)
    achievements: AchievementConfig = field(default_factory=AchievementConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Feature flags
    enable_progress_tracking: bool = True
    enable_insights: bool = True
    enable_trends: bool = True
    enable_goals: bool = True
    enable_achievements: bool = True
    enable_exports: bool = True

    # Performance
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    batch_size: int = 100
    max_concurrent_operations: int = 5

    # Logging
    log_level: str = "INFO"
    log_performance_metrics: bool = True


# Global configuration instance
analytics_config = AnalyticsEngineConfig()
