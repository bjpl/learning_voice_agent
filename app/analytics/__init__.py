"""
Analytics Module - Phase 6: Insights Generation and Trend Analysis
PATTERN: Gamification with progress tracking, insights generation, and data export
WHY: Motivate learners and provide comprehensive reporting with actionable insights

Components:
- goal_models: Data models for goals, achievements, milestones
- goal_store: SQLite persistence for goals and achievements
- goal_tracker: Goal management and progress tracking
- achievement_system: Badge system with 15+ achievements
- export_service: JSON, CSV, PDF export capabilities

Phase 6 Enhanced Components:
- insights_models: Typed models for insights, trends, anomalies
- insights_engine: Daily/weekly insights generation
- trend_analyzer: Statistical trend analysis and forecasting
"""

# Goal Models
from app.analytics.goal_models import (
    GoalType,
    GoalStatus,
    AchievementRarity,
    AchievementCategory,
    Goal,
    GoalProgress,
    Milestone,
    Achievement,
    GoalSuggestion,
    CreateGoalRequest,
    UpdateGoalRequest,
    GoalResponse,
    GoalListResponse,
    AchievementListResponse,
)

# Phase 6: Typed Insights Models
from app.analytics.insights_models import (
    InsightCategory,
    TrendDirection,
    AnomalySeverity,
    RecommendationPriority,
    MetricType,
    MilestoneType,
    Insight as TypedInsight,
    TrendData,
    Anomaly,
    Recommendation,
    LearningStreak as TypedLearningStreak,
    PeriodComparison,
    Milestone as InsightMilestone,
    InsightSummary,
    ForecastResult,
    create_achievement_insight,
    create_attention_insight,
    create_improvement_insight,
)

# Progress Tracking Models (Phase 6 - Learning Progress)
from app.analytics.progress_models import (
    LearningStreak as ProgressLearningStreak,
    TopicMastery,
    SessionProgress,
    DailyProgress,
    WeeklyProgress,
    MonthlyProgress,
    ProgressSnapshot,
    ProgressLevel,
)

# Configuration
from app.analytics.config import (
    AnalyticsConfig,
    ProgressTrackingConfig,
    AggregationConfig,
    CachingConfig,
    DashboardConfig,
    IntegrationConfig,
    analytics_config as progress_analytics_config,
    get_config,
    update_config,
)

# Stores
from app.analytics.goal_store import GoalStore, goal_store
from app.analytics.progress_store import ProgressStore, progress_store

# Services
from app.analytics.goal_tracker import GoalTracker, goal_tracker, ProgressMetrics
from app.analytics.achievement_system import AchievementSystem, achievement_system
from app.analytics.export_service import ExportService, export_service, ExportFormat, ReportPeriod
from app.analytics.insights_engine import InsightsEngine, insights_engine
from app.analytics.trend_analyzer import TrendAnalyzer, trend_analyzer
from app.analytics.progress_tracker import ProgressTracker, progress_tracker
from app.analytics.dashboard_service import DashboardService, dashboard_service

# Dashboard Models (Phase 6 - REST API)
from app.analytics.dashboard_models import (
    OverviewCard,
    OverviewResponse,
    LearningStreak as DashboardLearningStreak,
    QuickStats,
    ProgressChartResponse,
    ProgressDataPoint,
    ProgressSummary,
    TrendChartResponse,
    TrendMetric,
    TrendDirection as DashboardTrendDirection,
    TopicBreakdownResponse,
    TopicStats,
    ActivityHeatmapResponse,
    ActivityCell,
    ActivityWeek,
    GoalProgressResponse,
    LearningGoal as DashboardGoal,
    InsightResponse,
    InsightItem,
    ExportFormat as DashboardExportFormat,
    ExportResponse,
    PeriodType,
)

# Chart Data Formatters (Phase 6 - Chart.js Compatible)
from app.analytics.chart_data import (
    TimeSeriesPoint,
    ChartDataset,
    ChartJSData,
    BarChartData,
    PieChartData,
    HeatmapData,
    HeatmapCell,
    ChartDataFormatter,
)

__all__ = [
    # Goal Models
    "GoalType",
    "GoalStatus",
    "AchievementRarity",
    "AchievementCategory",
    "Goal",
    "GoalProgress",
    "Milestone",
    "Achievement",
    "GoalSuggestion",
    "CreateGoalRequest",
    "UpdateGoalRequest",
    "GoalResponse",
    "GoalListResponse",
    "AchievementListResponse",
    # Phase 6: Insights Models
    "InsightCategory",
    "TrendDirection",
    "AnomalySeverity",
    "RecommendationPriority",
    "MetricType",
    "MilestoneType",
    "TypedInsight",
    "TrendData",
    "Anomaly",
    "Recommendation",
    "TypedLearningStreak",
    "PeriodComparison",
    "InsightMilestone",
    "InsightSummary",
    "ForecastResult",
    "create_achievement_insight",
    "create_attention_insight",
    "create_improvement_insight",
    # Phase 6: Progress Tracking Models
    "ProgressLearningStreak",
    "TopicMastery",
    "SessionProgress",
    "DailyProgress",
    "WeeklyProgress",
    "MonthlyProgress",
    "ProgressSnapshot",
    "ProgressLevel",
    # Phase 6: Configuration
    "AnalyticsConfig",
    "ProgressTrackingConfig",
    "AggregationConfig",
    "CachingConfig",
    "DashboardConfig",
    "IntegrationConfig",
    "progress_analytics_config",
    "get_config",
    "update_config",
    # Stores
    "GoalStore",
    "goal_store",
    "ProgressStore",
    "progress_store",
    # Services
    "GoalTracker",
    "goal_tracker",
    "ProgressMetrics",
    "AchievementSystem",
    "achievement_system",
    "ExportService",
    "export_service",
    "ExportFormat",
    "ReportPeriod",
    # Phase 6: Enhanced Services
    "InsightsEngine",
    "insights_engine",
    "TrendAnalyzer",
    "trend_analyzer",
    "ProgressTracker",
    "progress_tracker",
    "DashboardService",
    "dashboard_service",
    # Phase 6: Dashboard Models
    "OverviewCard",
    "OverviewResponse",
    "DashboardLearningStreak",
    "QuickStats",
    "ProgressChartResponse",
    "ProgressDataPoint",
    "ProgressSummary",
    "TrendChartResponse",
    "TrendMetric",
    "DashboardTrendDirection",
    "TopicBreakdownResponse",
    "TopicStats",
    "ActivityHeatmapResponse",
    "ActivityCell",
    "ActivityWeek",
    "GoalProgressResponse",
    "DashboardGoal",
    "InsightResponse",
    "InsightItem",
    "DashboardExportFormat",
    "ExportResponse",
    "PeriodType",
    # Phase 6: Chart Data Formatters
    "TimeSeriesPoint",
    "ChartDataset",
    "ChartJSData",
    "BarChartData",
    "PieChartData",
    "HeatmapData",
    "HeatmapCell",
    "ChartDataFormatter",
]

__version__ = "2.1.0"  # Phase 6 - Learning Progress Tracking


async def initialize_analytics():
    """
    Initialize all analytics system components.

    USAGE:
        from app.analytics import initialize_analytics
        await initialize_analytics()
    """
    # Initialize stores
    await progress_store.initialize()
    await goal_store.initialize()

    # Initialize services
    await progress_tracker.initialize()
    await goal_tracker.initialize()
    await insights_engine.initialize()
    await dashboard_service.initialize()
    await trend_analyzer.initialize()
