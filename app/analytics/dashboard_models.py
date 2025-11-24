"""
Dashboard Response Models
=========================

Pydantic models for dashboard API responses.
Designed for frontend consumption with Chart.js compatibility.

PATTERN: Strongly-typed API contracts
WHY: Type safety and documentation for frontend integration
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ExportFormat(str, Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"


class TrendDirection(str, Enum):
    """Trend direction indicators"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


# =============================================================================
# Overview Models
# =============================================================================

class OverviewCard(BaseModel):
    """
    Summary card for dashboard overview.

    PATTERN: KPI card with trend indicator
    WHY: Quick visual summary of key metrics
    """
    title: str = Field(..., description="Card title")
    value: str = Field(..., description="Display value (formatted)")
    raw_value: Optional[float] = Field(None, description="Raw numeric value")
    change: Optional[float] = Field(None, description="Percent change from previous period")
    trend: TrendDirection = Field(TrendDirection.STABLE, description="Trend direction")
    icon: str = Field("chart", description="Icon identifier")
    color: Optional[str] = Field(None, description="Color theme (e.g., 'green', 'blue')")

    class Config:
        use_enum_values = True


class LearningStreak(BaseModel):
    """Learning streak information"""
    current_streak: int = Field(0, description="Current consecutive days")
    longest_streak: int = Field(0, description="Longest streak achieved")
    last_active_date: Optional[date] = Field(None, description="Last active date")
    streak_start_date: Optional[date] = Field(None, description="Current streak start")
    is_active_today: bool = Field(False, description="Whether user was active today")


class QuickStats(BaseModel):
    """Quick statistics summary"""
    total_exchanges: int = Field(0, description="Total conversation exchanges")
    total_hours: float = Field(0.0, description="Total learning hours")
    learning_velocity: float = Field(0.0, description="Learning rate metric")
    insights_generated: int = Field(0, description="Number of insights generated")
    avg_session_duration: float = Field(0.0, description="Average session duration in minutes")
    topics_explored: int = Field(0, description="Number of unique topics")


class OverviewResponse(BaseModel):
    """
    Main dashboard overview response.

    PATTERN: Aggregated dashboard summary
    WHY: Single endpoint for dashboard landing page
    """
    cards: List[OverviewCard] = Field(default_factory=list, description="Summary cards")
    quick_stats: QuickStats = Field(default_factory=QuickStats, description="Quick statistics")
    recent_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Recent insights")
    streak_info: LearningStreak = Field(default_factory=LearningStreak, description="Streak information")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    cache_ttl_seconds: int = Field(300, description="Cache TTL in seconds")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }


# =============================================================================
# Progress Chart Models
# =============================================================================

class ProgressDataPoint(BaseModel):
    """Single data point for progress charts"""
    date: str = Field(..., description="Date in ISO format")
    sessions: int = Field(0, description="Number of sessions")
    exchanges: int = Field(0, description="Number of exchanges")
    quality_score: float = Field(0.0, description="Average quality score")
    duration_minutes: float = Field(0.0, description="Total duration in minutes")


class ProgressSummary(BaseModel):
    """Summary statistics for progress period"""
    total_sessions: int = 0
    total_exchanges: int = 0
    total_duration_hours: float = 0.0
    avg_quality_score: float = 0.0
    improvement_rate: float = 0.0
    best_day: Optional[str] = None


class ProgressChartResponse(BaseModel):
    """
    Progress chart data response.

    PATTERN: Time-series data for line/area charts
    WHY: Track progress over time with trend analysis
    """
    period: str = Field(..., description="Period type: week, month, year")
    start_date: str = Field(..., description="Period start date")
    end_date: str = Field(..., description="Period end date")
    data_points: List[ProgressDataPoint] = Field(default_factory=list)
    summary: ProgressSummary = Field(default_factory=ProgressSummary)
    chart_config: Dict[str, Any] = Field(default_factory=dict, description="Chart.js config hints")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Trend Chart Models
# =============================================================================

class TrendMetric(BaseModel):
    """Single metric trend data"""
    metric_name: str = Field(..., description="Metric identifier")
    display_name: str = Field(..., description="Human-readable name")
    values: List[float] = Field(default_factory=list, description="Metric values over time")
    labels: List[str] = Field(default_factory=list, description="Time labels")
    current_value: float = Field(0.0, description="Most recent value")
    change: float = Field(0.0, description="Percent change")
    trend: TrendDirection = Field(TrendDirection.STABLE)
    color: str = Field("#4F46E5", description="Chart color")


class TrendChartResponse(BaseModel):
    """
    Trend chart data response.

    PATTERN: Multi-metric trend comparison
    WHY: Compare multiple metrics over same time period
    """
    days: int = Field(..., description="Number of days in trend")
    metrics: Dict[str, TrendMetric] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list, description="Shared time labels")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# =============================================================================
# Topic Analytics Models
# =============================================================================

class TopicStats(BaseModel):
    """Statistics for a single topic"""
    topic_name: str
    session_count: int = 0
    exchange_count: int = 0
    avg_quality_score: float = 0.0
    total_duration_minutes: float = 0.0
    percentage: float = 0.0
    trend: TrendDirection = TrendDirection.STABLE
    last_discussed: Optional[datetime] = None


class TopicBreakdownResponse(BaseModel):
    """
    Topic analytics breakdown response.

    PATTERN: Categorical distribution with metrics
    WHY: Understand content distribution and quality by topic
    """
    total_topics: int = 0
    topics: List[TopicStats] = Field(default_factory=list)
    top_topics: List[str] = Field(default_factory=list, description="Top 5 topics by frequency")
    emerging_topics: List[str] = Field(default_factory=list, description="Growing topics")
    chart_data: Dict[str, Any] = Field(default_factory=dict, description="Pie chart data")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Activity Heatmap Models
# =============================================================================

class ActivityCell(BaseModel):
    """Single cell in activity heatmap"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    count: int = Field(0, description="Activity count for this day")
    intensity: int = Field(0, ge=0, le=4, description="Intensity level 0-4")
    details: Optional[Dict[str, Any]] = None


class ActivityWeek(BaseModel):
    """Week of activity data"""
    week_number: int
    days: List[ActivityCell] = Field(default_factory=list)


class ActivityHeatmapResponse(BaseModel):
    """
    Activity heatmap response (GitHub-style calendar).

    PATTERN: Calendar heatmap data
    WHY: Visualize activity patterns over time
    """
    year: int = Field(..., description="Year for the heatmap")
    total_active_days: int = 0
    total_sessions: int = 0
    max_daily_sessions: int = 0
    weeks: List[ActivityWeek] = Field(default_factory=list)
    month_labels: List[Dict[str, Any]] = Field(default_factory=list)
    legend: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"level": 0, "label": "No activity", "color": "#ebedf0"},
            {"level": 1, "label": "Low", "color": "#9be9a8"},
            {"level": 2, "label": "Medium", "color": "#40c463"},
            {"level": 3, "label": "High", "color": "#30a14e"},
            {"level": 4, "label": "Very high", "color": "#216e39"}
        ]
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Goal Progress Models
# =============================================================================

class LearningGoal(BaseModel):
    """Individual learning goal"""
    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    target_value: float
    current_value: float = 0.0
    unit: str = Field("sessions", description="Unit of measurement")
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    status: str = Field("in_progress", description="Status: not_started, in_progress, completed")
    deadline: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class GoalProgressResponse(BaseModel):
    """
    Goal progress tracking response.

    PATTERN: Goal tracking with progress metrics
    WHY: Track learning objectives and milestones
    """
    active_goals: List[LearningGoal] = Field(default_factory=list)
    completed_goals: List[LearningGoal] = Field(default_factory=list)
    total_goals: int = 0
    completion_rate: float = 0.0
    next_milestone: Optional[Dict[str, Any]] = None
    suggested_goals: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Insights Models
# =============================================================================

class InsightItem(BaseModel):
    """Single insight item"""
    insight_id: str
    category: str
    title: str
    description: str
    priority: str = Field("medium", description="Priority: low, medium, high, critical")
    actionable: bool = False
    recommendation: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None


class InsightResponse(BaseModel):
    """
    Insights list response.

    PATTERN: Prioritized insights with recommendations
    WHY: Actionable intelligence from analytics
    """
    total_insights: int = 0
    insights: List[InsightItem] = Field(default_factory=list)
    categories: Dict[str, int] = Field(default_factory=dict, description="Count by category")
    has_critical: bool = False
    last_generated: Optional[datetime] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Export Models
# =============================================================================

class ExportRequest(BaseModel):
    """Export request parameters"""
    format: ExportFormat = ExportFormat.JSON
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    include_sessions: bool = True
    include_quality: bool = True
    include_feedback: bool = True
    include_insights: bool = True


class ExportResponse(BaseModel):
    """
    Export response with download information.

    PATTERN: Data export with metadata
    WHY: Enable data portability and offline analysis
    """
    format: ExportFormat
    download_url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file_size_bytes: Optional[int] = None
    record_count: int = 0
    date_range: Dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


# =============================================================================
# API Query Parameters
# =============================================================================

class PeriodType(str, Enum):
    """Time period types"""
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class MetricType(str, Enum):
    """Available metrics for trending"""
    QUALITY = "quality"
    ENGAGEMENT = "engagement"
    SESSIONS = "sessions"
    DURATION = "duration"
    POSITIVE_RATE = "positive_rate"
