"""
Learning System Data Models
===========================

Pydantic models and data classes for the learning system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import uuid


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_ENGAGEMENT = "implicit_engagement"
    IMPLICIT_CORRECTION = "implicit_correction"
    IMPLICIT_ABANDONMENT = "implicit_abandonment"
    IMPLICIT_FOLLOW_UP = "implicit_follow_up"


class FeedbackSource(str, Enum):
    """Source of feedback."""
    USER_BUTTON = "user_button"
    USER_TEXT = "user_text"
    SYSTEM_DETECTION = "system_detection"
    API_CALL = "api_call"


class QualityDimension(str, Enum):
    """Dimensions of response quality."""
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    ENGAGEMENT = "engagement"
    CLARITY = "clarity"
    ACCURACY = "accuracy"


class Feedback(BaseModel):
    """User feedback on a response."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    query_id: str
    feedback_type: FeedbackType
    source: FeedbackSource = FeedbackSource.USER_BUTTON

    # Feedback content
    rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    text: Optional[str] = None
    correction: Optional[str] = None

    # Context
    original_query: str
    original_response: str

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QualityScore(BaseModel):
    """Quality score for a response."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    session_id: str

    # Individual dimension scores (0.0 to 1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    helpfulness: float = Field(..., ge=0.0, le=1.0)
    engagement: float = Field(..., ge=0.0, le=1.0)
    clarity: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)

    # Composite score
    composite: float = Field(..., ge=0.0, le=1.0)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scoring_method: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Raw data
    query_text: Optional[str] = None
    response_text: Optional[str] = None

    def is_high_quality(self, threshold: float = 0.75) -> bool:
        """Check if the score indicates high quality."""
        return self.composite >= threshold

    def is_low_quality(self, threshold: float = 0.40) -> bool:
        """Check if the score indicates low quality."""
        return self.composite <= threshold

    def get_weakest_dimension(self) -> QualityDimension:
        """Get the dimension with the lowest score."""
        scores = {
            QualityDimension.RELEVANCE: self.relevance,
            QualityDimension.HELPFULNESS: self.helpfulness,
            QualityDimension.ENGAGEMENT: self.engagement,
            QualityDimension.CLARITY: self.clarity,
            QualityDimension.ACCURACY: self.accuracy,
        }
        return min(scores, key=scores.get)


class UserPreference(BaseModel):
    """Learned user preference."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Preference details
    category: str
    value: Any
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Learning metadata
    learned_from_samples: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Decay settings
    decay_rate: float = 0.1

    def apply_decay(self, days_elapsed: float) -> None:
        """Apply time-based confidence decay."""
        decay_factor = (1 - self.decay_rate) ** days_elapsed
        self.confidence *= decay_factor


class LearningInsight(BaseModel):
    """Generated learning insight."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Insight content
    title: str
    description: str
    category: str

    # Metrics
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    impact: str = Field(default="medium")  # "low", "medium", "high"

    # Supporting data
    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None


class InteractionPattern(BaseModel):
    """Detected interaction pattern."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Pattern details
    pattern_type: str  # "topic", "time", "quality", "engagement"
    description: str

    # Statistics
    frequency: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    first_seen: datetime
    last_seen: datetime

    # Pattern data
    exemplars: List[str] = Field(default_factory=list)
    associated_queries: List[str] = Field(default_factory=list)
    quality_correlation: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DailyReport(BaseModel):
    """Daily learning analytics report."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: datetime

    # Summary metrics
    total_interactions: int = 0
    total_feedback_collected: int = 0
    average_quality_score: float = 0.0

    # Trend data
    quality_trend: float = 0.0  # Positive = improving, negative = declining
    engagement_trend: float = 0.0
    feedback_trend: float = 0.0

    # Breakdown by dimension
    dimension_scores: Dict[str, float] = Field(default_factory=dict)

    # Insights and patterns
    insights: List[LearningInsight] = Field(default_factory=list)
    patterns: List[InteractionPattern] = Field(default_factory=list)

    # Top performers and issues
    top_performing_queries: List[Dict[str, Any]] = Field(default_factory=list)
    low_performing_queries: List[Dict[str, Any]] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_version: str = "1.0"


class AdaptationContext(BaseModel):
    """Context for response adaptation."""

    session_id: str
    user_id: Optional[str] = None

    # Current preferences
    preferences: Dict[str, UserPreference] = Field(default_factory=dict)

    # Historical context
    recent_quality_scores: List[float] = Field(default_factory=list)
    recent_feedback: List[Feedback] = Field(default_factory=list)

    # Adaptation parameters
    adaptation_strength: float = Field(default=0.3, ge=0.0, le=1.0)

    # Prompt modifications
    prompt_additions: List[str] = Field(default_factory=list)
    system_prompt_modifiers: Dict[str, str] = Field(default_factory=dict)

    # Response calibration
    target_length: Optional[int] = None
    target_formality: Optional[str] = None
    target_detail_level: Optional[str] = None


@dataclass
class FeedbackAggregation:
    """Aggregated feedback statistics."""

    session_id: str
    start_time: datetime
    end_time: datetime

    # Counts
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    corrections: int = 0

    # Averages
    average_rating: float = 0.0
    average_engagement_time: float = 0.0

    # Patterns
    common_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityTrend:
    """Quality score trend over time."""

    dimension: QualityDimension
    period_start: datetime
    period_end: datetime

    # Trend statistics
    start_value: float
    end_value: float
    change: float
    change_percent: float

    # Additional metrics
    min_value: float
    max_value: float
    average_value: float
    data_points: int

    @property
    def is_improving(self) -> bool:
        """Check if the trend is positive."""
        return self.change > 0

    @property
    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if the change is significant."""
        return abs(self.change_percent) >= threshold
