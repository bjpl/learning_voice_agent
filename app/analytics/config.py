"""
Analytics Module Configuration
PATTERN: Centralized configuration with sensible defaults
WHY: Easy tuning of analytics parameters without code changes

Configuration for:
- Progress tracking intervals and retention
- Aggregation settings
- Caching parameters
- Dashboard performance targets
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ProgressTrackingConfig:
    """Configuration for progress tracking"""

    # Database settings
    database_path: str = "progress_analytics.db"

    # Aggregation intervals
    daily_aggregation_enabled: bool = True
    weekly_aggregation_enabled: bool = True
    monthly_aggregation_enabled: bool = True

    # Aggregation timing
    daily_aggregation_hour: int = 0  # UTC hour
    weekly_aggregation_day: int = 0  # Monday

    # Data retention (in days)
    daily_data_retention_days: int = 365  # 1 year
    weekly_data_retention_days: int = 1825  # 5 years
    monthly_data_retention_days: int = 3650  # 10 years
    raw_session_retention_days: int = 90  # 3 months

    # Quality thresholds
    high_quality_threshold: float = 0.75
    good_quality_threshold: float = 0.60
    poor_quality_threshold: float = 0.40

    # Streak settings
    streak_timezone: str = "UTC"
    streak_grace_period_hours: int = 4  # Hours past midnight to count as previous day

    # Topic mastery
    topic_confidence_decay_rate: float = 0.01  # Per day
    topic_mastery_min_exchanges: int = 5

    # Milestone settings
    enable_milestones: bool = True
    milestone_notification_enabled: bool = True


@dataclass
class AggregationConfig:
    """Configuration for data aggregation"""

    # Batch processing
    batch_size: int = 1000
    max_concurrent_aggregations: int = 3

    # Rolling window sizes
    daily_rolling_window: int = 7  # Days for trend calculation
    weekly_rolling_window: int = 4  # Weeks for trend calculation
    monthly_rolling_window: int = 3  # Months for trend calculation

    # Pre-computation settings
    precompute_daily: bool = True
    precompute_weekly: bool = True
    precompute_monthly: bool = True

    # Incremental update
    enable_incremental_updates: bool = True
    incremental_update_interval_minutes: int = 15


@dataclass
class CachingConfig:
    """Configuration for analytics caching"""

    # Cache settings
    enable_caching: bool = True
    cache_backend: str = "memory"  # Options: memory, redis

    # TTL settings (in seconds)
    dashboard_cache_ttl: int = 300  # 5 minutes
    daily_progress_cache_ttl: int = 3600  # 1 hour
    weekly_progress_cache_ttl: int = 3600  # 1 hour
    monthly_progress_cache_ttl: int = 7200  # 2 hours
    streak_cache_ttl: int = 60  # 1 minute
    topic_mastery_cache_ttl: int = 600  # 10 minutes

    # Redis settings (if redis backend)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_prefix: str = "analytics:"

    # Memory cache settings
    max_memory_cache_items: int = 1000
    memory_cache_cleanup_interval: int = 300  # 5 minutes


@dataclass
class DashboardConfig:
    """Configuration for dashboard performance"""

    # Performance targets
    dashboard_load_target_ms: int = 500
    progress_query_target_ms: int = 200

    # Data limits
    max_daily_records_per_query: int = 365
    max_weekly_records_per_query: int = 52
    max_monthly_records_per_query: int = 24
    max_topics_in_summary: int = 20

    # Chart data points
    trend_chart_points: int = 30  # Days
    quality_chart_points: int = 14  # Days

    # Summary settings
    show_streak_in_summary: bool = True
    show_milestones_in_summary: bool = True
    show_top_topics_count: int = 5
    show_recent_sessions_count: int = 5


@dataclass
class IntegrationConfig:
    """Configuration for integration with other modules"""

    # Phase 5 integration
    feedback_store_integration: bool = True
    quality_store_integration: bool = True

    # Knowledge graph integration
    knowledge_graph_integration: bool = True
    sync_topic_relationships: bool = True

    # Learning module integration
    learning_analytics_integration: bool = True

    # Database paths for integration
    feedback_db_path: str = "feedback.db"
    learning_db_path: str = "learning_analytics.db"
    knowledge_graph_enabled: bool = True


@dataclass
class AnalyticsConfig:
    """Master configuration for analytics module"""

    # Sub-configurations
    progress: ProgressTrackingConfig = field(default_factory=ProgressTrackingConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    # Feature flags
    enable_progress_tracking: bool = True
    enable_streak_tracking: bool = True
    enable_topic_mastery: bool = True
    enable_milestones: bool = True
    enable_dashboard_api: bool = True

    # Logging
    log_aggregation_events: bool = True
    log_cache_hits: bool = False
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 200

    # Development mode
    debug_mode: bool = False

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check quality thresholds ordering
        if not (self.progress.poor_quality_threshold <
                self.progress.good_quality_threshold <
                self.progress.high_quality_threshold):
            issues.append("Quality thresholds must be in ascending order")

        # Check retention periods
        if self.progress.daily_data_retention_days < 30:
            issues.append("Daily data retention should be at least 30 days")

        # Check cache TTLs
        if self.caching.streak_cache_ttl > 300:
            issues.append("Streak cache TTL should not exceed 5 minutes for accuracy")

        return issues


# Global configuration instance
analytics_config = AnalyticsConfig()


def get_config() -> AnalyticsConfig:
    """Get the global analytics configuration"""
    return analytics_config


def update_config(**kwargs) -> AnalyticsConfig:
    """Update configuration values"""
    global analytics_config

    for key, value in kwargs.items():
        if hasattr(analytics_config, key):
            setattr(analytics_config, key, value)
        elif hasattr(analytics_config.progress, key):
            setattr(analytics_config.progress, key, value)
        elif hasattr(analytics_config.aggregation, key):
            setattr(analytics_config.aggregation, key, value)
        elif hasattr(analytics_config.caching, key):
            setattr(analytics_config.caching, key, value)
        elif hasattr(analytics_config.dashboard, key):
            setattr(analytics_config.dashboard, key, value)

    return analytics_config
