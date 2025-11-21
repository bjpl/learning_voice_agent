"""
Insights Data Models - Type Definitions for Analytics Engine
PATTERN: Contract-first development with comprehensive type safety
WHY: Ensure data integrity and enable IDE support across analytics system
SPARC: Strong typing for reliable statistical analysis

Models:
- InsightCategory: Categories of generated insights
- TrendDirection: Direction indicators for metric trends
- Insight: Individual insight with metadata
- TrendData: Trend analysis result with statistics
- Anomaly: Detected anomaly with context
- Recommendation: Actionable suggestion with impact assessment
- LearningStreak: User engagement streak tracking
- PeriodComparison: Comparison between time periods
- Milestone: Achievement milestone detection
- InsightSummary: Aggregated insights for dashboards
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


# =============================================================================
# ENUMERATIONS
# =============================================================================

class InsightCategory(str, Enum):
    """
    Categories of generated insights.

    PATTERN: Semantic categorization for filtering and prioritization
    WHY: Enable targeted display and notification routing
    """
    ACHIEVEMENT = "achievement"      # Milestones reached, streaks, records
    IMPROVEMENT = "improvement"      # Quality going up, engagement increasing
    ATTENTION = "attention"          # Declining metrics, broken streaks
    DISCOVERY = "discovery"          # New topics, learning patterns
    SUGGESTION = "suggestion"        # Recommendations for improvement
    TREND = "trend"                  # Significant trend detection
    ANOMALY = "anomaly"              # Unusual pattern detection
    MILESTONE = "milestone"          # Goal completion, achievements


class TrendDirection(str, Enum):
    """
    Direction indicators for metric trends.

    PATTERN: Simple categorical representation of trend direction
    WHY: Clear communication of metric trajectory
    """
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"  # High variance, no clear direction


class AnomalySeverity(str, Enum):
    """
    Severity levels for detected anomalies.

    PATTERN: Graduated severity for prioritization
    WHY: Enable appropriate response to different anomaly levels
    """
    LOW = "low"            # Minor deviation, informational
    MEDIUM = "medium"      # Notable deviation, worth attention
    HIGH = "high"          # Significant deviation, action recommended
    CRITICAL = "critical"  # Major deviation, immediate attention needed


class RecommendationPriority(str, Enum):
    """
    Priority levels for recommendations.

    PATTERN: Action prioritization based on impact
    WHY: Help users focus on high-impact improvements
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class MetricType(str, Enum):
    """
    Types of metrics that can be analyzed.

    PATTERN: Typed metric identification
    WHY: Consistent metric handling across analytics
    """
    QUALITY_SCORE = "quality_score"
    ENGAGEMENT_SCORE = "engagement_score"
    SESSION_DURATION = "session_duration"
    EXCHANGE_COUNT = "exchange_count"
    RESPONSE_TIME = "response_time"
    FEEDBACK_RATE = "feedback_rate"
    CORRECTION_RATE = "correction_rate"
    TOPIC_DIVERSITY = "topic_diversity"
    STREAK_LENGTH = "streak_length"
    COMPLETION_RATE = "completion_rate"


class MilestoneType(str, Enum):
    """
    Types of achievements and milestones.

    PATTERN: Achievement categorization
    WHY: Enable gamification and motivation tracking
    """
    STREAK = "streak"           # Consecutive day achievements
    VOLUME = "volume"           # Number of sessions/exchanges
    QUALITY = "quality"         # Quality score achievements
    IMPROVEMENT = "improvement" # Rate of improvement achievements
    CONSISTENCY = "consistency" # Consistent behavior achievements
    EXPLORATION = "exploration" # Topic diversity achievements


# =============================================================================
# CORE DATA MODELS
# =============================================================================

@dataclass
class Insight:
    """
    Individual insight with full metadata.

    PATTERN: Self-contained insight with evidence
    WHY: Enable audit trail and explanation of insights

    Attributes:
        id: Unique identifier
        category: InsightCategory classification
        title: Human-readable summary
        description: Detailed explanation
        importance: Priority score (1-5, higher = more important)
        metric_name: Associated metric (optional)
        metric_value: Current metric value (optional)
        evidence: Supporting data for the insight
        confidence: Statistical confidence (0-1)
        actionable: Whether user can act on this
        recommendation: Suggested action (optional)
        valid_until: Expiration datetime (optional)
        created_at: Generation timestamp
    """
    id: str
    category: InsightCategory
    title: str
    description: str
    importance: int  # 1-5 scale
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    actionable: bool = False
    recommendation: Optional[str] = None
    valid_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize insight data"""
        if not 1 <= self.importance <= 5:
            self.importance = max(1, min(5, self.importance))
        if not 0 <= self.confidence <= 1:
            self.confidence = max(0, min(1, self.confidence))

    @classmethod
    def create(
        cls,
        category: InsightCategory,
        title: str,
        description: str,
        importance: int = 3,
        **kwargs
    ) -> "Insight":
        """Factory method for creating insights with auto-generated ID"""
        return cls(
            id=f"insight-{uuid.uuid4().hex[:12]}",
            category=category,
            title=title,
            description=description,
            importance=importance,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "importance": self.importance,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "actionable": self.actionable,
            "recommendation": self.recommendation,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class TrendData:
    """
    Trend analysis result with comprehensive statistics.

    PATTERN: Statistical trend representation
    WHY: Provide complete context for trend interpretation

    Attributes:
        metric: Name of the analyzed metric
        direction: Overall trend direction
        magnitude: Percentage change magnitude
        confidence: R-squared or similar confidence measure (0-1)
        period_days: Analysis period length
        data_points: Raw data used for analysis
        slope: Linear regression slope
        intercept: Linear regression intercept
        start_value: Value at beginning of period
        end_value: Value at end of period
        min_value: Minimum value in period
        max_value: Maximum value in period
        avg_value: Average value in period
        std_dev: Standard deviation
        variance: Statistical variance
        calculated_at: Timestamp of calculation
    """
    metric: str
    direction: TrendDirection
    magnitude: float  # Percentage change
    confidence: float  # 0-1, R-squared
    period_days: int
    data_points: List[float] = field(default_factory=list)
    slope: float = 0.0
    intercept: float = 0.0
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Calculate derived statistics if data points available"""
        if self.data_points and len(self.data_points) > 0:
            if self.start_value is None:
                self.start_value = self.data_points[0]
            if self.end_value is None:
                self.end_value = self.data_points[-1]
            if self.min_value is None:
                self.min_value = min(self.data_points)
            if self.max_value is None:
                self.max_value = max(self.data_points)
            if self.avg_value is None:
                self.avg_value = sum(self.data_points) / len(self.data_points)

    @property
    def is_significant(self) -> bool:
        """Check if trend is statistically significant"""
        return self.confidence >= 0.5 and abs(self.magnitude) >= 5.0

    @property
    def is_positive(self) -> bool:
        """Check if trend direction is positive"""
        return self.direction == TrendDirection.INCREASING

    @property
    def volatility(self) -> Optional[float]:
        """Calculate coefficient of variation if possible"""
        if self.std_dev is not None and self.avg_value and self.avg_value != 0:
            return (self.std_dev / abs(self.avg_value)) * 100
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "magnitude": round(self.magnitude, 2),
            "confidence": round(self.confidence, 4),
            "period_days": self.period_days,
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 4),
            "start_value": round(self.start_value, 4) if self.start_value else None,
            "end_value": round(self.end_value, 4) if self.end_value else None,
            "min_value": round(self.min_value, 4) if self.min_value else None,
            "max_value": round(self.max_value, 4) if self.max_value else None,
            "avg_value": round(self.avg_value, 4) if self.avg_value else None,
            "std_dev": round(self.std_dev, 4) if self.std_dev else None,
            "data_points_count": len(self.data_points),
            "is_significant": self.is_significant,
            "calculated_at": self.calculated_at.isoformat()
        }


@dataclass
class Anomaly:
    """
    Detected anomaly with context and explanation.

    PATTERN: Anomaly detection result with supporting evidence
    WHY: Enable investigation and response to unusual patterns

    Attributes:
        id: Unique identifier
        metric: Name of the metric with anomaly
        title: Human-readable summary
        description: Detailed explanation
        severity: AnomalySeverity level
        value: The anomalous value
        expected_value: What was expected
        z_score: Standard deviations from mean
        is_negative: Whether anomaly indicates a problem
        detected_at: Detection timestamp
        context: Additional context information
    """
    id: str
    metric: str
    title: str
    description: str
    severity: AnomalySeverity
    value: float
    expected_value: float
    z_score: float
    is_negative: bool = False
    detected_at: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        metric: str,
        value: float,
        expected_value: float,
        z_score: float,
        is_negative: bool = False,
        **kwargs
    ) -> "Anomaly":
        """Factory method with auto-generated ID and default descriptions"""
        abs_z = abs(z_score)

        # Determine severity based on z-score
        if abs_z >= 4:
            severity = AnomalySeverity.CRITICAL
        elif abs_z >= 3:
            severity = AnomalySeverity.HIGH
        elif abs_z >= 2.5:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW

        # Generate title and description
        direction = "below" if value < expected_value else "above"
        change_pct = abs((value - expected_value) / expected_value * 100) if expected_value != 0 else 0

        title = kwargs.get(
            "title",
            f"Unusual {metric}: {change_pct:.1f}% {direction} expected"
        )
        description = kwargs.get(
            "description",
            f"{metric} is {value:.2f}, which is {abs_z:.1f} standard deviations "
            f"{direction} the expected value of {expected_value:.2f}."
        )

        return cls(
            id=f"anomaly-{uuid.uuid4().hex[:12]}",
            metric=metric,
            title=title,
            description=description,
            severity=severity,
            value=value,
            expected_value=expected_value,
            z_score=z_score,
            is_negative=is_negative,
            **{k: v for k, v in kwargs.items() if k not in ['title', 'description']}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "metric": self.metric,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "value": round(self.value, 4),
            "expected_value": round(self.expected_value, 4),
            "z_score": round(self.z_score, 2),
            "is_negative": self.is_negative,
            "detected_at": self.detected_at.isoformat(),
            "context": self.context
        }


@dataclass
class Recommendation:
    """
    Actionable recommendation with impact assessment.

    PATTERN: Structured suggestion with expected outcomes
    WHY: Guide users toward effective improvements

    Attributes:
        id: Unique identifier
        action: Specific action to take
        rationale: Why this is recommended
        expected_impact: Expected improvement description
        impact_score: Quantified impact (0-1)
        priority: RecommendationPriority level
        category: InsightCategory this relates to
        metric_target: Metric expected to improve
        difficulty: Implementation difficulty (1-5)
        time_to_impact: Expected time to see results (days)
        prerequisites: Any required prior actions
        created_at: Generation timestamp
    """
    id: str
    action: str
    rationale: str
    expected_impact: str
    impact_score: float  # 0-1 scale
    priority: RecommendationPriority
    category: InsightCategory = InsightCategory.SUGGESTION
    metric_target: Optional[str] = None
    difficulty: int = 3  # 1-5 scale
    time_to_impact: Optional[int] = None  # Days
    prerequisites: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        action: str,
        rationale: str,
        expected_impact: str,
        impact_score: float,
        **kwargs
    ) -> "Recommendation":
        """Factory method with auto-generated ID and priority calculation"""
        # Calculate priority based on impact and difficulty
        difficulty = kwargs.get("difficulty", 3)
        if impact_score >= 0.8 and difficulty <= 2:
            priority = RecommendationPriority.URGENT
        elif impact_score >= 0.6:
            priority = RecommendationPriority.HIGH
        elif impact_score >= 0.4:
            priority = RecommendationPriority.MEDIUM
        else:
            priority = RecommendationPriority.LOW

        return cls(
            id=f"rec-{uuid.uuid4().hex[:12]}",
            action=action,
            rationale=rationale,
            expected_impact=expected_impact,
            impact_score=impact_score,
            priority=kwargs.get("priority", priority),
            **{k: v for k, v in kwargs.items() if k != "priority"}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "action": self.action,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "impact_score": round(self.impact_score, 2),
            "priority": self.priority.value,
            "category": self.category.value,
            "metric_target": self.metric_target,
            "difficulty": self.difficulty,
            "time_to_impact": self.time_to_impact,
            "prerequisites": self.prerequisites,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class LearningStreak:
    """
    User engagement streak tracking.

    PATTERN: Consecutive activity tracking
    WHY: Motivate consistent engagement through gamification

    Attributes:
        current_streak: Current consecutive days
        longest_streak: All-time longest streak
        streak_start: When current streak began
        last_activity: Most recent activity date
        streak_history: Historical streak records
        milestones_achieved: List of streak milestones hit
    """
    current_streak: int
    longest_streak: int
    streak_start: Optional[date] = None
    last_activity: Optional[date] = None
    streak_history: List[Dict[str, Any]] = field(default_factory=list)
    milestones_achieved: List[int] = field(default_factory=list)

    # Standard streak milestones
    MILESTONES = [3, 7, 14, 30, 60, 90, 100, 180, 365]

    @property
    def days_until_next_milestone(self) -> Optional[int]:
        """Calculate days until next milestone"""
        for milestone in self.MILESTONES:
            if milestone > self.current_streak:
                return milestone - self.current_streak
        return None

    @property
    def next_milestone(self) -> Optional[int]:
        """Get next milestone target"""
        for milestone in self.MILESTONES:
            if milestone > self.current_streak:
                return milestone
        return None

    @property
    def is_at_milestone(self) -> bool:
        """Check if current streak is at a milestone"""
        return self.current_streak in self.MILESTONES

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "streak_start": self.streak_start.isoformat() if self.streak_start else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "days_until_next_milestone": self.days_until_next_milestone,
            "next_milestone": self.next_milestone,
            "is_at_milestone": self.is_at_milestone,
            "milestones_achieved": self.milestones_achieved
        }


@dataclass
class PeriodComparison:
    """
    Comparison between two time periods.

    PATTERN: Period-over-period analysis
    WHY: Enable week-over-week, month-over-month comparisons

    Attributes:
        metric: Metric being compared
        period1_label: Label for first period
        period2_label: Label for second period
        period1_value: Value in first period
        period2_value: Value in second period
        absolute_change: Raw difference
        percent_change: Percentage difference
        is_improvement: Whether change is positive
        significance: Statistical significance (0-1)
    """
    metric: str
    period1_label: str
    period2_label: str
    period1_value: float
    period2_value: float
    absolute_change: float = field(init=False)
    percent_change: float = field(init=False)
    is_improvement: bool = field(init=False)
    significance: float = 0.0

    def __post_init__(self):
        """Calculate derived fields"""
        self.absolute_change = self.period2_value - self.period1_value
        if self.period1_value != 0:
            self.percent_change = (self.absolute_change / abs(self.period1_value)) * 100
        else:
            self.percent_change = 100.0 if self.period2_value > 0 else 0.0
        # Positive change is improvement for most metrics
        self.is_improvement = self.absolute_change > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric": self.metric,
            "period1_label": self.period1_label,
            "period2_label": self.period2_label,
            "period1_value": round(self.period1_value, 4),
            "period2_value": round(self.period2_value, 4),
            "absolute_change": round(self.absolute_change, 4),
            "percent_change": round(self.percent_change, 2),
            "is_improvement": self.is_improvement,
            "significance": round(self.significance, 4)
        }


@dataclass
class Milestone:
    """
    Achievement milestone detection.

    PATTERN: Goal tracking with celebration triggers
    WHY: Recognize and celebrate user achievements

    Attributes:
        id: Unique identifier
        type: MilestoneType classification
        title: Achievement title
        description: What was achieved
        value: Numeric value achieved
        threshold: Threshold that was crossed
        achieved_at: When milestone was reached
        previous_best: Previous record if applicable
        celebration_level: Visual celebration intensity (1-5)
    """
    id: str
    type: MilestoneType
    title: str
    description: str
    value: float
    threshold: float
    achieved_at: datetime = field(default_factory=datetime.utcnow)
    previous_best: Optional[float] = None
    celebration_level: int = 3  # 1-5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        type: MilestoneType,
        title: str,
        description: str,
        value: float,
        threshold: float,
        **kwargs
    ) -> "Milestone":
        """Factory method with auto-generated ID"""
        # Calculate celebration level based on threshold significance
        if threshold >= 100 or type == MilestoneType.STREAK:
            celebration_level = 5
        elif threshold >= 50:
            celebration_level = 4
        elif threshold >= 10:
            celebration_level = 3
        else:
            celebration_level = 2

        return cls(
            id=f"milestone-{uuid.uuid4().hex[:12]}",
            type=type,
            title=title,
            description=description,
            value=value,
            threshold=threshold,
            celebration_level=kwargs.get("celebration_level", celebration_level),
            **{k: v for k, v in kwargs.items() if k != "celebration_level"}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "value": round(self.value, 2),
            "threshold": round(self.threshold, 2),
            "achieved_at": self.achieved_at.isoformat(),
            "previous_best": round(self.previous_best, 2) if self.previous_best else None,
            "celebration_level": self.celebration_level,
            "metadata": self.metadata
        }


@dataclass
class InsightSummary:
    """
    Aggregated insights summary for dashboards.

    PATTERN: Pre-computed dashboard data
    WHY: Efficient dashboard rendering without re-computation

    Attributes:
        generated_at: Summary generation timestamp
        period_start: Start of analysis period
        period_end: End of analysis period
        total_insights: Count of all insights
        by_category: Counts by category
        by_importance: Counts by importance level
        top_insights: Highest priority insights
        trends_summary: Key trend indicators
        anomalies_count: Number of detected anomalies
        recommendations_count: Number of recommendations
        health_score: Overall learning health (0-100)
    """
    generated_at: datetime
    period_start: date
    period_end: date
    total_insights: int
    by_category: Dict[str, int]
    by_importance: Dict[int, int]
    top_insights: List[Insight]
    trends_summary: Dict[str, TrendDirection]
    anomalies_count: int
    recommendations_count: int
    health_score: float  # 0-100
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_insights": self.total_insights,
            "by_category": self.by_category,
            "by_importance": self.by_importance,
            "top_insights": [i.to_dict() for i in self.top_insights],
            "trends_summary": {k: v.value for k, v in self.trends_summary.items()},
            "anomalies_count": self.anomalies_count,
            "recommendations_count": self.recommendations_count,
            "health_score": round(self.health_score, 1),
            "metadata": self.metadata
        }


@dataclass
class ForecastResult:
    """
    Forecast result from trend projection.

    PATTERN: Predictive analytics output
    WHY: Enable forward-looking insights

    Attributes:
        metric: Metric being forecast
        forecast_days: Number of days forecasted
        predicted_values: List of predicted values
        confidence_interval: (lower, upper) bounds
        base_value: Starting value for forecast
        predicted_direction: Expected trend direction
        confidence: Forecast confidence (0-1)
    """
    metric: str
    forecast_days: int
    predicted_values: List[float]
    confidence_interval: tuple  # (lower_bound, upper_bound)
    base_value: float
    predicted_direction: TrendDirection
    confidence: float
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def final_predicted_value(self) -> float:
        """Get the final predicted value"""
        return self.predicted_values[-1] if self.predicted_values else self.base_value

    @property
    def expected_change(self) -> float:
        """Calculate expected change percentage"""
        if self.base_value == 0:
            return 0.0
        return ((self.final_predicted_value - self.base_value) / abs(self.base_value)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric": self.metric,
            "forecast_days": self.forecast_days,
            "predicted_values": [round(v, 4) for v in self.predicted_values],
            "confidence_interval": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4)
            ),
            "base_value": round(self.base_value, 4),
            "final_predicted_value": round(self.final_predicted_value, 4),
            "predicted_direction": self.predicted_direction.value,
            "expected_change": round(self.expected_change, 2),
            "confidence": round(self.confidence, 4),
            "calculated_at": self.calculated_at.isoformat()
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_achievement_insight(
    title: str,
    description: str,
    importance: int = 4,
    **kwargs
) -> Insight:
    """Helper to create achievement insights"""
    return Insight.create(
        category=InsightCategory.ACHIEVEMENT,
        title=title,
        description=description,
        importance=importance,
        **kwargs
    )


def create_attention_insight(
    title: str,
    description: str,
    importance: int = 4,
    **kwargs
) -> Insight:
    """Helper to create attention-needed insights"""
    return Insight.create(
        category=InsightCategory.ATTENTION,
        title=title,
        description=description,
        importance=importance,
        actionable=True,
        **kwargs
    )


def create_improvement_insight(
    title: str,
    description: str,
    metric_name: str,
    metric_value: float,
    importance: int = 3,
    **kwargs
) -> Insight:
    """Helper to create improvement insights"""
    return Insight.create(
        category=InsightCategory.IMPROVEMENT,
        title=title,
        description=description,
        metric_name=metric_name,
        metric_value=metric_value,
        importance=importance,
        **kwargs
    )
