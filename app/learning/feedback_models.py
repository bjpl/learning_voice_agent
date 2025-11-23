"""
Feedback Models - Pydantic Models for Feedback Data
PATTERN: Contract-first development with type safety
WHY: Ensure data integrity and automatic documentation

Models:
- ExplicitFeedback: User-provided ratings and comments
- ImplicitFeedback: Automatically tracked engagement metrics
- CorrectionFeedback: User corrections and rephrases
- SessionFeedback: Aggregated session-level metrics
- FeedbackStats: System-wide aggregate statistics
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class CorrectionType(str, Enum):
    """Types of user corrections."""
    REPHRASE = "rephrase"  # User rephrased their question
    CORRECTION = "correction"  # User corrected agent's response
    CLARIFICATION = "clarification"  # User provided clarification
    ELABORATION = "elaboration"  # User added more detail
    SPELLING = "spelling"  # Spelling correction
    GRAMMAR = "grammar"  # Grammar correction


class FeedbackSentiment(str, Enum):
    """Computed sentiment from feedback."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# ============================================================================
# CORE FEEDBACK MODELS
# ============================================================================

class ExplicitFeedback(BaseModel):
    """
    Explicit user feedback - ratings and comments.

    PATTERN: Direct user input for quality assessment
    WHY: Clear signal of user satisfaction
    """
    id: Optional[str] = Field(None, description="Unique feedback ID")
    session_id: str = Field(..., description="Session identifier")
    exchange_id: str = Field(..., description="Exchange/turn identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5 stars")
    helpful: bool = Field(..., description="Whether the response was helpful")
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional user comment"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When feedback was submitted"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )

    @validator('comment')
    def sanitize_comment(cls, v):
        """Sanitize comment to remove any potential PII markers."""
        if v is None:
            return v
        # Strip excessive whitespace
        return ' '.join(v.split())

    @property
    def sentiment(self) -> FeedbackSentiment:
        """Compute sentiment from rating and helpful flag."""
        if self.rating >= 4 and self.helpful:
            return FeedbackSentiment.POSITIVE
        elif self.rating <= 2 or not self.helpful:
            return FeedbackSentiment.NEGATIVE
        return FeedbackSentiment.NEUTRAL

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "exchange_id": "ex_456",
                "rating": 5,
                "helpful": True,
                "comment": "Very helpful explanation!",
                "timestamp": "2025-11-21T10:00:00Z"
            }
        }


class ImplicitFeedback(BaseModel):
    """
    Implicit engagement metrics - automatically tracked.

    PATTERN: Behavioral signals for quality inference
    WHY: Non-intrusive feedback collection
    """
    id: Optional[str] = Field(None, description="Unique feedback ID")
    session_id: str = Field(..., description="Session identifier")
    response_time_ms: int = Field(
        ...,
        ge=0,
        description="Time for agent to respond in milliseconds"
    )
    user_response_time_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Time until user's next message in milliseconds"
    )
    engagement_duration_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="Total engagement duration in seconds"
    )
    follow_up_count: int = Field(
        default=0,
        ge=0,
        description="Number of follow-up questions in sequence"
    )
    scroll_depth: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="How far user scrolled (0-1)"
    )
    copy_action: bool = Field(
        default=False,
        description="Whether user copied response text"
    )
    share_action: bool = Field(
        default=False,
        description="Whether user shared the response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When metrics were recorded"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional tracking metadata"
    )

    @property
    def engagement_score(self) -> float:
        """
        Compute engagement score from implicit signals.
        Range: 0.0 to 1.0
        """
        score = 0.0
        weights_applied = 0

        # Response time score (faster = better, but not instant)
        if self.response_time_ms is not None:
            if 500 <= self.response_time_ms <= 2000:
                score += 1.0
            elif self.response_time_ms < 500:
                score += 0.8  # Too fast might mean shallow
            elif self.response_time_ms <= 5000:
                score += 0.6
            else:
                score += 0.3
            weights_applied += 1

        # Follow-up engagement (more = interested)
        if self.follow_up_count > 0:
            follow_up_score = min(1.0, self.follow_up_count * 0.2)
            score += follow_up_score
            weights_applied += 1

        # Action signals
        if self.copy_action:
            score += 0.8
            weights_applied += 1

        if self.share_action:
            score += 1.0
            weights_applied += 1

        # Scroll depth
        if self.scroll_depth is not None:
            score += self.scroll_depth
            weights_applied += 1

        if weights_applied == 0:
            return 0.5  # Neutral default

        return score / weights_applied

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "response_time_ms": 1500,
                "user_response_time_ms": 3000,
                "engagement_duration_seconds": 120,
                "follow_up_count": 2,
                "copy_action": True,
                "share_action": False,
                "timestamp": "2025-11-21T10:00:00Z"
            }
        }


class CorrectionFeedback(BaseModel):
    """
    User correction tracking - rephrases and corrections.

    PATTERN: Learning from user corrections
    WHY: Identify misunderstandings and improve responses
    """
    id: Optional[str] = Field(None, description="Unique feedback ID")
    session_id: str = Field(..., description="Session identifier")
    original_text: str = Field(..., description="Original text before correction")
    corrected_text: str = Field(..., description="Corrected/rephrased text")
    correction_type: CorrectionType = Field(
        ...,
        description="Type of correction"
    )
    edit_distance: Optional[int] = Field(
        None,
        ge=0,
        description="Levenshtein edit distance"
    )
    edit_distance_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Edit distance as ratio of original length"
    )
    context: Optional[str] = Field(
        None,
        max_length=500,
        description="Context around the correction"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When correction was made"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional correction metadata"
    )

    @validator('corrected_text')
    def texts_must_differ(cls, v, values):
        """Ensure corrected text differs from original."""
        if 'original_text' in values and v == values['original_text']:
            raise ValueError('Corrected text must differ from original')
        return v

    @property
    def severity(self) -> str:
        """Assess correction severity based on edit distance ratio."""
        if self.edit_distance_ratio is None:
            return "unknown"
        if self.edit_distance_ratio < 0.2:
            return "minor"
        elif self.edit_distance_ratio < 0.5:
            return "moderate"
        return "major"

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "original_text": "Tell me about machien learning",
                "corrected_text": "Tell me about machine learning",
                "correction_type": "spelling",
                "edit_distance": 2,
                "edit_distance_ratio": 0.06,
                "timestamp": "2025-11-21T10:00:00Z"
            }
        }


class SessionFeedback(BaseModel):
    """
    Aggregated session-level feedback metrics.

    PATTERN: Holistic session quality assessment
    WHY: Understanding overall conversation quality
    """
    session_id: str = Field(..., description="Session identifier")

    # Explicit feedback aggregates
    explicit_feedback_count: int = Field(
        default=0,
        ge=0,
        description="Number of explicit feedback items"
    )
    average_rating: Optional[float] = Field(
        None,
        ge=1,
        le=5,
        description="Average rating for session"
    )
    helpful_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage of helpful responses"
    )

    # Implicit feedback aggregates
    implicit_feedback_count: int = Field(
        default=0,
        ge=0,
        description="Number of implicit feedback items"
    )
    average_response_time_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Average response time in milliseconds"
    )
    total_follow_ups: int = Field(
        default=0,
        ge=0,
        description="Total follow-up questions"
    )
    engagement_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Overall engagement score"
    )

    # Correction metrics
    correction_count: int = Field(
        default=0,
        ge=0,
        description="Number of corrections made"
    )
    correction_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by correction type"
    )

    # Session metadata
    session_duration_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="Total session duration in seconds"
    )
    exchange_count: int = Field(
        default=0,
        ge=0,
        description="Number of exchanges in session"
    )
    first_feedback_at: Optional[datetime] = Field(
        None,
        description="Timestamp of first feedback"
    )
    last_feedback_at: Optional[datetime] = Field(
        None,
        description="Timestamp of last feedback"
    )

    # Computed sentiment
    overall_sentiment: FeedbackSentiment = Field(
        default=FeedbackSentiment.NEUTRAL,
        description="Overall session sentiment"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "explicit_feedback_count": 3,
                "average_rating": 4.5,
                "helpful_percentage": 90.0,
                "implicit_feedback_count": 10,
                "average_response_time_ms": 1200.0,
                "total_follow_ups": 5,
                "engagement_score": 0.85,
                "correction_count": 1,
                "session_duration_seconds": 300,
                "exchange_count": 8,
                "overall_sentiment": "positive"
            }
        }


class FeedbackStats(BaseModel):
    """
    System-wide aggregate feedback statistics.

    PATTERN: Global metrics for system health monitoring
    WHY: Track overall system performance and user satisfaction
    """
    # Time range
    time_range_start: datetime = Field(..., description="Start of time range")
    time_range_end: datetime = Field(..., description="End of time range")

    # Volume metrics
    total_sessions: int = Field(default=0, ge=0)
    total_explicit_feedback: int = Field(default=0, ge=0)
    total_implicit_feedback: int = Field(default=0, ge=0)
    total_corrections: int = Field(default=0, ge=0)

    # Quality metrics
    average_rating: Optional[float] = Field(None, ge=1, le=5)
    rating_distribution: Dict[int, int] = Field(
        default_factory=dict,
        description="Count of each rating value"
    )
    helpful_percentage: Optional[float] = Field(None, ge=0, le=100)

    # Performance metrics
    average_response_time_ms: Optional[float] = Field(None, ge=0)
    p50_response_time_ms: Optional[float] = Field(None, ge=0)
    p95_response_time_ms: Optional[float] = Field(None, ge=0)
    p99_response_time_ms: Optional[float] = Field(None, ge=0)

    # Engagement metrics
    average_engagement_score: Optional[float] = Field(None, ge=0, le=1)
    average_follow_ups_per_session: Optional[float] = Field(None, ge=0)
    average_session_duration_seconds: Optional[float] = Field(None, ge=0)

    # Correction insights
    correction_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Corrections per exchange"
    )
    correction_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by correction type"
    )

    # Sentiment distribution
    sentiment_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by sentiment"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "time_range_start": "2025-11-20T00:00:00Z",
                "time_range_end": "2025-11-21T00:00:00Z",
                "total_sessions": 150,
                "total_explicit_feedback": 450,
                "total_implicit_feedback": 1200,
                "total_corrections": 30,
                "average_rating": 4.2,
                "rating_distribution": {"1": 5, "2": 10, "3": 50, "4": 200, "5": 185},
                "helpful_percentage": 88.5,
                "average_response_time_ms": 1150.0,
                "p50_response_time_ms": 1000.0,
                "p95_response_time_ms": 2500.0,
                "average_engagement_score": 0.78,
                "correction_rate": 0.02
            }
        }


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class ExplicitFeedbackRequest(BaseModel):
    """Request model for submitting explicit feedback."""
    session_id: str = Field(..., min_length=1, max_length=100)
    exchange_id: str = Field(..., min_length=1, max_length=100)
    rating: int = Field(..., ge=1, le=5)
    helpful: bool
    comment: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "exchange_id": "ex_456",
                "rating": 5,
                "helpful": True,
                "comment": "Great explanation!"
            }
        }


class ImplicitFeedbackRequest(BaseModel):
    """Request model for tracking implicit feedback."""
    session_id: str = Field(..., min_length=1, max_length=100)
    response_time_ms: int = Field(..., ge=0)
    user_response_time_ms: Optional[int] = Field(None, ge=0)
    follow_up_count: int = Field(default=0, ge=0)
    engagement_duration_seconds: Optional[int] = Field(None, ge=0)
    copy_action: bool = Field(default=False)
    share_action: bool = Field(default=False)
    scroll_depth: Optional[float] = Field(None, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "response_time_ms": 1500,
                "user_response_time_ms": 3000,
                "follow_up_count": 2,
                "copy_action": True
            }
        }


class CorrectionFeedbackRequest(BaseModel):
    """Request model for logging corrections."""
    session_id: str = Field(..., min_length=1, max_length=100)
    original_text: str = Field(..., min_length=1, max_length=2000)
    corrected_text: str = Field(..., min_length=1, max_length=2000)
    correction_type: Optional[CorrectionType] = Field(None)
    context: Optional[str] = Field(None, max_length=500)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "original_text": "What is machien learning?",
                "corrected_text": "What is machine learning?",
                "correction_type": "spelling"
            }
        }


class SessionFeedbackResponse(BaseModel):
    """Response model for session feedback query."""
    session_id: str
    feedback: SessionFeedback
    explicit_items: List[ExplicitFeedback] = Field(default_factory=list)
    correction_items: List[CorrectionFeedback] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "feedback": {
                    "session_id": "sess_abc123",
                    "explicit_feedback_count": 3,
                    "average_rating": 4.5
                },
                "explicit_items": [],
                "correction_items": []
            }
        }


class FeedbackStatsResponse(BaseModel):
    """Response model for aggregate stats query."""
    stats: FeedbackStats
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cache_ttl_seconds: int = Field(default=60)

    class Config:
        json_schema_extra = {
            "example": {
                "stats": {
                    "time_range_start": "2025-11-20T00:00:00Z",
                    "time_range_end": "2025-11-21T00:00:00Z",
                    "total_sessions": 150,
                    "average_rating": 4.2
                },
                "generated_at": "2025-11-21T12:00:00Z",
                "cache_ttl_seconds": 60
            }
        }
