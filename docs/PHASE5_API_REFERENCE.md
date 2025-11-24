# Phase 5: Real-Time Learning - API Reference

**Version:** 1.0.0
**Date:** 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration Classes](#configuration-classes)
3. [Data Models](#data-models)
4. [FeedbackCollector API](#feedbackcollector-api)
5. [FeedbackStore API](#feedbackstore-api)
6. [QualityScorer API](#qualityscorer-api)
7. [ResponseAdapter API](#responseadapter-api)
8. [PreferenceLearner API](#preferencelearner-api)
9. [LearningAnalytics API](#learninganalytics-api)
10. [PatternDetector API](#patterndetector-api)
11. [REST API Endpoints](#rest-api-endpoints)

---

## Overview

This document provides a complete API reference for all Phase 5 Real-Time Learning components.

### Import Statements

```python
from app.learning import (
    # Configuration
    LearningConfig,

    # Models
    Feedback,
    FeedbackType,
    QualityScore,
    UserPreference,
    LearningInsight,
    InteractionPattern,
    DailyReport,
    AdaptationContext,

    # Components
    FeedbackCollector,
    FeedbackStore,
    QualityScorer,
    ResponseAdapter,
    PreferenceLearner,
    LearningAnalytics,
    PatternDetector,
)
```

---

## Configuration Classes

### LearningConfig

Main configuration class for the learning system.

```python
@dataclass
class LearningConfig:
    feedback: FeedbackConfig
    quality_scoring: QualityScoringConfig
    adaptation: AdaptationConfig
    preference_learning: PreferenceLearningConfig
    analytics: AnalyticsConfig
    pattern_detection: PatternDetectionConfig

    enabled: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"
    data_dir: str = "data/learning"
    batch_size: int = 100
    async_processing: bool = True
    cache_ttl_seconds: int = 300

    @classmethod
    def from_env(cls) -> "LearningConfig"
```

### FeedbackConfig

```python
@dataclass
class FeedbackConfig:
    database_path: str = "data/learning/feedback.db"
    implicit_feedback_enabled: bool = True
    correction_detection_enabled: bool = True
    engagement_tracking_enabled: bool = True
    min_response_length_for_feedback: int = 10
    correction_similarity_threshold: float = 0.7
    engagement_timeout_seconds: int = 300
    aggregation_window_hours: int = 24
    min_feedback_for_aggregation: int = 3
```

### QualityScoringConfig

```python
@dataclass
class QualityScoringConfig:
    relevance_weight: float = 0.25
    helpfulness_weight: float = 0.25
    engagement_weight: float = 0.20
    clarity_weight: float = 0.15
    accuracy_weight: float = 0.15
    high_quality_threshold: float = 0.75
    low_quality_threshold: float = 0.40
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_response_tokens: int = 1000
    context_window_size: int = 5
```

### AdaptationConfig

```python
@dataclass
class AdaptationConfig:
    adaptation_strength: float = 0.3
    preference_decay_rate: float = 0.1
    max_context_items: int = 5
    context_relevance_threshold: float = 0.5
    target_response_length: Optional[int] = None
    formality_level: Optional[str] = None
    detail_level: Optional[str] = None
    prompt_adaptation_enabled: bool = True
    context_enhancement_enabled: bool = True
    response_calibration_enabled: bool = True
```

### PreferenceLearningConfig

```python
@dataclass
class PreferenceLearningConfig:
    learning_rate: float = 0.1
    momentum: float = 0.9
    min_samples_for_learning: int = 5
    preference_categories: List[str]
    preference_file_path: str = "data/learning/preferences.json"
    auto_save_interval_minutes: int = 30
    anonymize_preferences: bool = True
    retention_days: int = 90
```

### AnalyticsConfig

```python
@dataclass
class AnalyticsConfig:
    daily_report_time: str = "00:00"
    weekly_report_day: str = "monday"
    trend_window_days: int = 7
    trend_min_data_points: int = 3
    pattern_min_occurrences: int = 3
    pattern_confidence_threshold: float = 0.7
    max_insights_per_report: int = 10
    insight_relevance_threshold: float = 0.6
    analytics_database_path: str = "data/learning/analytics.db"
    report_output_dir: str = "data/learning/reports"
```

### PatternDetectionConfig

```python
@dataclass
class PatternDetectionConfig:
    min_pattern_frequency: int = 3
    pattern_window_hours: int = 168
    cluster_min_samples: int = 3
    cluster_eps: float = 0.5
    quality_correlation_threshold: float = 0.5
    engagement_pattern_threshold: float = 0.3
    detect_topic_patterns: bool = True
    detect_time_patterns: bool = True
    detect_quality_patterns: bool = True
    detect_engagement_patterns: bool = True
```

---

## Data Models

### FeedbackType (Enum)

```python
class FeedbackType(str, Enum):
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_ENGAGEMENT = "implicit_engagement"
    IMPLICIT_CORRECTION = "implicit_correction"
    IMPLICIT_ABANDONMENT = "implicit_abandonment"
    IMPLICIT_FOLLOW_UP = "implicit_follow_up"
```

### FeedbackSource (Enum)

```python
class FeedbackSource(str, Enum):
    USER_BUTTON = "user_button"
    USER_TEXT = "user_text"
    SYSTEM_DETECTION = "system_detection"
    API_CALL = "api_call"
```

### QualityDimension (Enum)

```python
class QualityDimension(str, Enum):
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    ENGAGEMENT = "engagement"
    CLARITY = "clarity"
    ACCURACY = "accuracy"
```

### Feedback

```python
class Feedback(BaseModel):
    id: str
    session_id: str
    query_id: str
    feedback_type: FeedbackType
    source: FeedbackSource
    rating: Optional[float]  # 0.0 to 1.0
    text: Optional[str]
    correction: Optional[str]
    original_query: str
    original_response: str
    timestamp: datetime
    user_id: Optional[str]
    metadata: Dict[str, Any]
```

### QualityScore

```python
class QualityScore(BaseModel):
    id: str
    query_id: str
    session_id: str
    relevance: float  # 0.0 to 1.0
    helpfulness: float
    engagement: float
    clarity: float
    accuracy: float
    composite: float
    timestamp: datetime
    scoring_method: str
    confidence: float
    query_text: Optional[str]
    response_text: Optional[str]

    def is_high_quality(threshold: float = 0.75) -> bool
    def is_low_quality(threshold: float = 0.40) -> bool
    def get_weakest_dimension() -> QualityDimension
```

### UserPreference

```python
class UserPreference(BaseModel):
    id: str
    user_id: Optional[str]
    session_id: Optional[str]
    category: str
    value: Any
    confidence: float  # 0.0 to 1.0
    learned_from_samples: int
    last_updated: datetime
    created_at: datetime
    decay_rate: float

    def apply_decay(days_elapsed: float) -> None
```

### LearningInsight

```python
class LearningInsight(BaseModel):
    id: str
    title: str
    description: str
    category: str
    relevance_score: float
    confidence: float
    impact: str  # "low", "medium", "high"
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime
    valid_until: Optional[datetime]
```

### InteractionPattern

```python
class InteractionPattern(BaseModel):
    id: str
    pattern_type: str  # "topic", "time", "quality", "engagement"
    description: str
    frequency: int
    confidence: float
    first_seen: datetime
    last_seen: datetime
    exemplars: List[str]
    associated_queries: List[str]
    quality_correlation: Optional[float]
    metadata: Dict[str, Any]
```

### DailyReport

```python
class DailyReport(BaseModel):
    id: str
    date: datetime
    total_interactions: int
    total_feedback_collected: int
    average_quality_score: float
    quality_trend: float
    engagement_trend: float
    feedback_trend: float
    dimension_scores: Dict[str, float]
    insights: List[LearningInsight]
    patterns: List[InteractionPattern]
    top_performing_queries: List[Dict[str, Any]]
    low_performing_queries: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime
    report_version: str
```

### AdaptationContext

```python
class AdaptationContext(BaseModel):
    session_id: str
    user_id: Optional[str]
    preferences: Dict[str, UserPreference]
    recent_quality_scores: List[float]
    recent_feedback: List[Feedback]
    adaptation_strength: float
    prompt_additions: List[str]
    system_prompt_modifiers: Dict[str, str]
    target_length: Optional[int]
    target_formality: Optional[str]
    target_detail_level: Optional[str]
```

---

## FeedbackCollector API

### Class: FeedbackCollector

```python
class FeedbackCollector:
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback_store: Optional[FeedbackStore] = None
    )
```

### Methods

#### initialize

```python
async def initialize(self) -> None
```

Initialize the feedback collector and storage.

#### close

```python
async def close(self) -> None
```

Close the feedback collector and flush pending data.

#### collect_rating

```python
async def collect_rating(
    self,
    session_id: str,
    query_id: str,
    rating: float,
    original_query: str,
    original_response: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Feedback
```

Collect an explicit rating from the user.

**Parameters:**
- `session_id`: Session identifier
- `query_id`: Query identifier
- `rating`: Rating value (0.0 to 1.0)
- `original_query`: The user's original query
- `original_response`: The system's response
- `user_id`: Optional user identifier
- `metadata`: Additional metadata

**Returns:** The stored Feedback object

#### collect_thumbs_up

```python
async def collect_thumbs_up(
    self,
    session_id: str,
    query_id: str,
    original_query: str,
    original_response: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Feedback
```

Collect positive thumbs up feedback.

#### collect_thumbs_down

```python
async def collect_thumbs_down(
    self,
    session_id: str,
    query_id: str,
    original_query: str,
    original_response: str,
    reason: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Feedback
```

Collect negative thumbs down feedback.

#### collect_text_feedback

```python
async def collect_text_feedback(
    self,
    session_id: str,
    query_id: str,
    feedback_text: str,
    original_query: str,
    original_response: str,
    is_positive: Optional[bool] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Feedback
```

Collect text-based feedback from the user.

#### track_interaction_start

```python
async def track_interaction_start(
    self,
    session_id: str,
    query_id: str,
    query: str
) -> None
```

Start tracking an interaction for implicit feedback.

#### track_response_delivered

```python
async def track_response_delivered(
    self,
    session_id: str,
    query_id: str,
    response: str
) -> None
```

Track when a response is delivered to the user.

#### track_engagement

```python
async def track_engagement(
    self,
    session_id: str,
    query_id: str,
    event_type: str,
    event_data: Optional[Dict[str, Any]] = None
) -> None
```

Track engagement events (scroll, copy, link click, etc.)

#### detect_correction

```python
async def detect_correction(
    self,
    session_id: str,
    previous_query_id: str,
    new_query: str,
    previous_query: str,
    previous_response: str
) -> Optional[Feedback]
```

Detect if a new query is a correction of a previous response.

#### track_follow_up

```python
async def track_follow_up(
    self,
    session_id: str,
    previous_query_id: str,
    follow_up_query: str,
    previous_query: str,
    previous_response: str
) -> Feedback
```

Track when a user asks a follow-up question (positive signal).

#### track_abandonment

```python
async def track_abandonment(
    self,
    session_id: str,
    query_id: str,
    original_query: str,
    original_response: str,
    idle_seconds: int
) -> Feedback
```

Track when a user abandons an interaction (negative signal).

#### finalize_interaction

```python
async def finalize_interaction(
    self,
    session_id: str,
    query_id: str
) -> Optional[Feedback]
```

Finalize tracking for an interaction and generate engagement feedback.

#### aggregate_feedback

```python
async def aggregate_feedback(
    self,
    session_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> FeedbackAggregation
```

Aggregate feedback for a session within a time window.

#### get_feedback_summary

```python
async def get_feedback_summary(
    self,
    session_id: str,
    limit: int = 10
) -> Dict[str, Any]
```

Get a summary of recent feedback for a session.

---

## FeedbackStore API

### Class: FeedbackStore

```python
class FeedbackStore:
    def __init__(self, config: Optional[LearningConfig] = None)
```

### Methods

#### initialize

```python
async def initialize(self) -> None
```

Initialize the database and create tables.

#### close

```python
async def close(self) -> None
```

Close the database connection.

#### store

```python
async def store(self, feedback: Feedback) -> str
```

Store a feedback record. Returns the feedback ID.

#### get

```python
async def get(self, feedback_id: str) -> Optional[Feedback]
```

Retrieve a feedback record by ID.

#### query

```python
async def query(
    self,
    session_id: Optional[str] = None,
    query_id: Optional[str] = None,
    user_id: Optional[str] = None,
    feedback_type: Optional[FeedbackType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Feedback]
```

Query feedback records with filters.

#### count

```python
async def count(
    self,
    session_id: Optional[str] = None,
    feedback_type: Optional[FeedbackType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> int
```

Count feedback records matching filters.

#### get_average_rating

```python
async def get_average_rating(
    self,
    session_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Optional[float]
```

Calculate average rating for the given filters.

#### get_feedback_distribution

```python
async def get_feedback_distribution(
    self,
    session_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, int]
```

Get distribution of feedback types.

#### delete

```python
async def delete(self, feedback_id: str) -> bool
```

Delete a feedback record.

#### delete_old_feedback

```python
async def delete_old_feedback(self, retention_days: int = 90) -> int
```

Delete feedback older than the retention period. Returns count of deleted records.

#### vacuum

```python
async def vacuum(self) -> None
```

Optimize the database by running VACUUM.

---

## QualityScorer API

### Class: QualityScorer

```python
class QualityScorer:
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        embedding_generator: Optional[Any] = None,
        feedback_store: Optional[FeedbackStore] = None
    )
```

### Methods

#### score_response

```python
async def score_response(
    self,
    query: str,
    response: str,
    session_id: str,
    context: Optional[List[Dict[str, str]]] = None,
    feedback_history: Optional[List[Feedback]] = None
) -> QualityScore
```

Score a response across all quality dimensions.

#### batch_score

```python
async def batch_score(
    self,
    interactions: List[Dict[str, str]],
    session_id: str
) -> List[QualityScore]
```

Score multiple interactions in batch.

#### get_dimension_weights

```python
def get_dimension_weights(self) -> Dict[QualityDimension, float]
```

Get the current dimension weights.

#### set_dimension_weights

```python
def set_dimension_weights(self, weights: Dict[QualityDimension, float]) -> None
```

Set custom dimension weights.

---

## ResponseAdapter API

### Class: ResponseAdapter

```python
class ResponseAdapter:
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        preference_learner: Optional[Any] = None
    )
```

### Methods

#### get_adaptation_context

```python
async def get_adaptation_context(
    self,
    session_id: str,
    user_id: Optional[str] = None
) -> AdaptationContext
```

Get or create an adaptation context for a session.

#### adapt_prompt

```python
async def adapt_prompt(
    self,
    base_prompt: str,
    context: AdaptationContext
) -> str
```

Adapt a prompt based on user preferences.

#### enhance_context

```python
async def enhance_context(
    self,
    query: str,
    context: AdaptationContext,
    relevant_history: Optional[List[Dict[str, str]]] = None
) -> str
```

Enhance query context with relevant historical information.

#### calibrate_response

```python
async def calibrate_response(
    self,
    response: str,
    context: AdaptationContext
) -> str
```

Calibrate a response based on preferences.

#### update_context_with_quality

```python
def update_context_with_quality(
    self,
    context: AdaptationContext,
    quality_score: QualityScore
) -> None
```

Update context with a new quality score.

#### update_context_with_feedback

```python
def update_context_with_feedback(
    self,
    context: AdaptationContext,
    feedback: Feedback
) -> None
```

Update context with new feedback.

#### add_prompt_modifier

```python
def add_prompt_modifier(
    self,
    context: AdaptationContext,
    key: str,
    modifier: str
) -> None
```

Add a custom prompt modifier to the context.

#### clear_context

```python
def clear_context(self, session_id: str, user_id: Optional[str] = None) -> None
```

Clear cached context for a session.

---

## PreferenceLearner API

### Class: PreferenceLearner

```python
class PreferenceLearner:
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback_store: Optional[FeedbackStore] = None
    )
```

### Methods

#### initialize

```python
async def initialize(self) -> None
```

Initialize the preference learner and load saved preferences.

#### close

```python
async def close(self) -> None
```

Close and save preferences.

#### get_preferences

```python
async def get_preferences(
    self,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, UserPreference]
```

Get all preferences for a session or user.

#### get_preference

```python
async def get_preference(
    self,
    category: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Optional[UserPreference]
```

Get a specific preference by category.

#### learn_from_feedback

```python
async def learn_from_feedback(
    self,
    feedback: Feedback,
    response_characteristics: Optional[Dict[str, Any]] = None
) -> List[UserPreference]
```

Learn preferences from a feedback instance.

#### learn_from_engagement

```python
async def learn_from_engagement(
    self,
    session_id: str,
    engagement_metrics: Dict[str, Any],
    response_characteristics: Dict[str, Any],
    user_id: Optional[str] = None
) -> List[UserPreference]
```

Learn preferences from engagement metrics.

#### clear_preferences

```python
async def clear_preferences(
    self,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> None
```

Clear preferences for a session or user.

#### get_learning_history

```python
async def get_learning_history(
    self,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict]
```

Get the learning history for a session or user.

#### predict_preference

```python
async def predict_preference(
    self,
    category: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Optional[tuple[Any, float]]
```

Predict the most likely preference value for a category.

#### get_preference_summary

```python
def get_preference_summary(
    self,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]
```

Get a summary of learned preferences.

---

## LearningAnalytics API

### Class: LearningAnalytics

```python
class LearningAnalytics:
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback_store: Optional[FeedbackStore] = None
    )
```

### Methods

#### initialize

```python
async def initialize(self) -> None
```

Initialize the analytics engine.

#### generate_daily_report

```python
async def generate_daily_report(
    self,
    date: Optional[datetime] = None,
    session_id: Optional[str] = None
) -> DailyReport
```

Generate a daily analytics report.

#### record_quality_score

```python
def record_quality_score(self, score: QualityScore) -> None
```

Record a quality score for analytics.

#### increment_interaction_count

```python
def increment_interaction_count(self, session_id: str) -> None
```

Increment interaction count for a session.

#### get_metrics_summary

```python
async def get_metrics_summary(
    self,
    session_id: Optional[str] = None,
    days: int = 7
) -> Dict[str, Any]
```

Get a summary of metrics for the last N days.

#### calculate_trend

```python
async def calculate_trend(
    self,
    dimension: QualityDimension,
    session_id: Optional[str] = None,
    days: int = 7
) -> QualityTrend
```

Calculate trend for a specific quality dimension.

---

## PatternDetector API

### Class: PatternDetector

```python
class PatternDetector:
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback_store: Optional[FeedbackStore] = None
    )
```

### Methods

#### initialize

```python
async def initialize(self) -> None
```

Initialize the pattern detector.

#### record_interaction

```python
def record_interaction(
    self,
    query: str,
    response: str,
    quality_score: Optional[QualityScore] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Record an interaction for pattern analysis.

#### detect_patterns

```python
async def detect_patterns(
    self,
    session_id: Optional[str] = None,
    time_window_hours: Optional[int] = None
) -> List[InteractionPattern]
```

Detect all pattern types.

#### detect_topic_patterns

```python
async def detect_topic_patterns(
    self,
    session_id: Optional[str] = None,
    time_window_hours: int = 168
) -> List[InteractionPattern]
```

Detect recurring topic patterns.

#### detect_time_patterns

```python
async def detect_time_patterns(
    self,
    session_id: Optional[str] = None,
    time_window_hours: int = 168
) -> List[InteractionPattern]
```

Detect time-based usage patterns.

#### detect_quality_correlations

```python
async def detect_quality_correlations(
    self,
    session_id: Optional[str] = None,
    time_window_hours: int = 168
) -> List[InteractionPattern]
```

Detect correlations between response characteristics and quality.

#### detect_engagement_patterns

```python
async def detect_engagement_patterns(
    self,
    session_id: Optional[str] = None,
    time_window_hours: int = 168
) -> List[InteractionPattern]
```

Detect patterns in user engagement behavior.

#### get_cached_patterns

```python
def get_cached_patterns(
    self,
    session_id: Optional[str] = None
) -> List[InteractionPattern]
```

Get cached patterns for a session.

#### clear_cache

```python
def clear_cache(self, session_id: Optional[str] = None) -> None
```

Clear pattern and interaction caches.

#### get_pattern_summary

```python
async def get_pattern_summary(
    self,
    session_id: Optional[str] = None
) -> Dict[str, Any]
```

Get a summary of detected patterns.

---

## REST API Endpoints

### Feedback Endpoints

#### Submit Feedback

```http
POST /api/feedback
Content-Type: application/json

{
    "session_id": "session-123",
    "query_id": "query-456",
    "feedback_type": "explicit_positive",
    "rating": 0.8,
    "text": "Great response!",
    "original_query": "What is AI?",
    "original_response": "AI is..."
}
```

**Response:**
```json
{
    "id": "feedback-789",
    "session_id": "session-123",
    "created_at": "2025-11-21T10:00:00Z"
}
```

#### Get Feedback Summary

```http
GET /api/feedback/summary/{session_id}?limit=10
```

**Response:**
```json
{
    "total_feedback": 15,
    "feedback_by_type": {
        "explicit_positive": 10,
        "explicit_negative": 2,
        "implicit_follow_up": 3
    },
    "average_rating": 0.75,
    "recent_feedback": [...]
}
```

### Analytics Endpoints

#### Get Daily Report

```http
GET /api/analytics/daily?date=2025-11-21&session_id=session-123
```

**Response:**
```json
{
    "date": "2025-11-21",
    "total_interactions": 50,
    "total_feedback_collected": 30,
    "average_quality_score": 0.72,
    "quality_trend": 0.05,
    "insights": [...],
    "recommendations": [...]
}
```

#### Get Metrics Summary

```http
GET /api/analytics/summary?session_id=session-123&days=7
```

**Response:**
```json
{
    "period_days": 7,
    "total_scored_interactions": 100,
    "average_quality": 0.75,
    "dimension_averages": {
        "relevance": 0.78,
        "helpfulness": 0.72,
        "engagement": 0.70,
        "clarity": 0.80,
        "accuracy": 0.75
    },
    "quality_trend": "improving"
}
```

### Preference Endpoints

#### Get Preferences

```http
GET /api/preferences/{session_id}
```

**Response:**
```json
{
    "total_preferences": 5,
    "high_confidence": [
        {"category": "response_length", "value": "detailed", "confidence": 0.85}
    ],
    "medium_confidence": [...],
    "low_confidence": [...]
}
```

#### Clear Preferences

```http
DELETE /api/preferences/{session_id}
```

**Response:**
```json
{
    "status": "success",
    "message": "Preferences cleared"
}
```

### Pattern Endpoints

#### Get Patterns

```http
GET /api/patterns?session_id=session-123&time_window_hours=168
```

**Response:**
```json
{
    "total_patterns": 5,
    "patterns_by_type": {
        "topic": [...],
        "time": [...],
        "quality": [...],
        "engagement": [...]
    },
    "high_confidence_patterns": [...],
    "actionable_insights": [...]
}
```

---

**For implementation details, see [PHASE5_IMPLEMENTATION_GUIDE.md](PHASE5_IMPLEMENTATION_GUIDE.md)**
**For testing guidance, see [PHASE5_TESTING_GUIDE.md](PHASE5_TESTING_GUIDE.md)**
