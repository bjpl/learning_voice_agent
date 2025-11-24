# Phase 5: Real-Time Learning - Implementation Guide

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Setup](#component-setup)
4. [Feedback Collection](#feedback-collection)
5. [Quality Scoring](#quality-scoring)
6. [Response Adaptation](#response-adaptation)
7. [Preference Learning](#preference-learning)
8. [Analytics and Insights](#analytics-and-insights)
9. [Pattern Detection](#pattern-detection)
10. [Privacy Considerations](#privacy-considerations)
11. [Troubleshooting](#troubleshooting)
12. [Performance Optimization](#performance-optimization)

---

## Overview

Phase 5 introduces a comprehensive real-time learning system that enables the Learning Voice Agent to continuously improve based on user feedback and interaction patterns.

### Key Capabilities

- **Feedback Collection**: Gather explicit and implicit user feedback
- **Quality Scoring**: Evaluate responses across multiple dimensions
- **Adaptive Responses**: Dynamically adjust responses based on preferences
- **Preference Learning**: Learn user preferences over time
- **Analytics**: Generate insights and reports
- **Pattern Detection**: Identify recurring interaction patterns

### Design Principles

1. **Privacy First**: User data is anonymized and retention policies enforced
2. **Non-Invasive**: Learning happens transparently without disrupting UX
3. **Explainable**: All adaptations can be traced to specific feedback
4. **Reversible**: Users can reset preferences at any time

---

## Architecture

### System Architecture

```
                    +------------------+
                    |  User Interface  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Conversation API |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v--------+ +--------v--------+ +--------v--------+
|    Feedback     | |    Quality      | |    Response     |
|    Collector    | |    Scorer       | |    Adapter      |
+--------+--------+ +--------+--------+ +--------+--------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                    +--------v---------+
                    | Preference       |
                    | Learner          |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v--------+ +--------v--------+ +--------v--------+
|   Feedback      | |   Learning      | |    Pattern      |
|   Store         | |   Analytics     | |    Detector     |
+-----------------+ +-----------------+ +-----------------+
```

### Data Flow

1. User interacts with the system
2. FeedbackCollector captures explicit/implicit feedback
3. QualityScorer evaluates response quality
4. PreferenceLearner updates user preferences
5. ResponseAdapter applies preferences to future responses
6. Analytics generates insights for review

### Module Dependencies

```python
app/learning/
├── __init__.py           # Module exports
├── config.py             # Configuration classes
├── models.py             # Data models
├── feedback_collector.py # Feedback collection
├── feedback_store.py     # Feedback persistence
├── quality_scorer.py     # Quality evaluation
├── scoring_models.py     # Scoring data models
├── scoring_algorithms.py # Scoring algorithms
├── adapter.py            # Response adaptation
├── preference_learner.py # Preference learning
├── analytics.py          # Analytics generation
├── pattern_detector.py   # Pattern detection
├── insights_generator.py # Insight generation
├── improvement_engine.py # Improvement suggestions
└── store.py              # General data store
```

---

## Component Setup

### Installation

The learning system is included with the main application. No additional dependencies are required beyond the base requirements.

### Configuration

Create or update environment variables:

```bash
# .env
LEARNING_ENABLED=true
LEARNING_DEBUG=false
LEARNING_DATA_DIR=data/learning
```

### Initialize Components

```python
from app.learning import (
    FeedbackCollector,
    FeedbackStore,
    QualityScorer,
    ResponseAdapter,
    PreferenceLearner,
    LearningAnalytics,
    PatternDetector,
    LearningConfig
)

# Load configuration
config = LearningConfig.from_env()

# Initialize components
async def initialize_learning_system():
    feedback_store = FeedbackStore(config)
    await feedback_store.initialize()

    feedback_collector = FeedbackCollector(
        config=config,
        feedback_store=feedback_store
    )
    await feedback_collector.initialize()

    quality_scorer = QualityScorer(
        config=config,
        feedback_store=feedback_store
    )

    preference_learner = PreferenceLearner(
        config=config,
        feedback_store=feedback_store
    )
    await preference_learner.initialize()

    adapter = ResponseAdapter(
        config=config,
        preference_learner=preference_learner
    )

    analytics = LearningAnalytics(
        config=config,
        feedback_store=feedback_store
    )
    await analytics.initialize()

    pattern_detector = PatternDetector(
        config=config,
        feedback_store=feedback_store
    )
    await pattern_detector.initialize()

    return {
        "feedback_collector": feedback_collector,
        "feedback_store": feedback_store,
        "quality_scorer": quality_scorer,
        "preference_learner": preference_learner,
        "adapter": adapter,
        "analytics": analytics,
        "pattern_detector": pattern_detector
    }
```

### Directory Structure

Ensure the data directory exists:

```bash
mkdir -p data/learning/reports
```

---

## Feedback Collection

### Overview

The FeedbackCollector captures both explicit and implicit feedback from user interactions.

### Explicit Feedback Types

1. **Thumbs Up/Down**: Binary satisfaction indicator
2. **Rating**: Numeric score (0.0 to 1.0)
3. **Text Feedback**: Written comments or complaints

### Implicit Feedback Types

1. **Engagement**: Time spent, scroll depth, interactions
2. **Correction**: User rephrasing or correcting the system
3. **Follow-up**: User asking follow-up questions
4. **Abandonment**: User leaving without engagement

### Collecting Explicit Feedback

```python
# Thumbs up
feedback = await feedback_collector.collect_thumbs_up(
    session_id="session-123",
    query_id="query-456",
    original_query="What is machine learning?",
    original_response="Machine learning is a subset of AI..."
)

# Thumbs down with reason
feedback = await feedback_collector.collect_thumbs_down(
    session_id="session-123",
    query_id="query-456",
    original_query="What is machine learning?",
    original_response="Machine learning is a subset of AI...",
    reason="Too technical"
)

# Numeric rating
feedback = await feedback_collector.collect_rating(
    session_id="session-123",
    query_id="query-456",
    rating=0.8,  # 80% satisfaction
    original_query="What is machine learning?",
    original_response="Machine learning is a subset of AI..."
)

# Text feedback
feedback = await feedback_collector.collect_text_feedback(
    session_id="session-123",
    query_id="query-456",
    feedback_text="Great explanation but could use more examples",
    original_query="What is machine learning?",
    original_response="Machine learning is a subset of AI...",
    is_positive=True
)
```

### Collecting Implicit Feedback

```python
# Track interaction start
await feedback_collector.track_interaction_start(
    session_id="session-123",
    query_id="query-456",
    query="What is machine learning?"
)

# Track response delivery
await feedback_collector.track_response_delivered(
    session_id="session-123",
    query_id="query-456",
    response="Machine learning is a subset of AI..."
)

# Track engagement events
await feedback_collector.track_engagement(
    session_id="session-123",
    query_id="query-456",
    event_type="scroll",
    event_data={"depth": 0.75}
)

await feedback_collector.track_engagement(
    session_id="session-123",
    query_id="query-456",
    event_type="copy",
    event_data={"text_length": 50}
)

# Detect corrections
correction = await feedback_collector.detect_correction(
    session_id="session-123",
    previous_query_id="query-456",
    new_query="No, I meant supervised machine learning",
    previous_query="What is machine learning?",
    previous_response="Machine learning is a subset of AI..."
)

# Track follow-ups (positive signal)
follow_up = await feedback_collector.track_follow_up(
    session_id="session-123",
    previous_query_id="query-456",
    follow_up_query="How does supervised learning differ from unsupervised?",
    previous_query="What is machine learning?",
    previous_response="Machine learning is a subset of AI..."
)

# Finalize interaction and generate engagement feedback
engagement_feedback = await feedback_collector.finalize_interaction(
    session_id="session-123",
    query_id="query-456"
)
```

### Feedback Aggregation

```python
# Aggregate feedback for a session
aggregation = await feedback_collector.aggregate_feedback(
    session_id="session-123",
    start_time=datetime.utcnow() - timedelta(hours=24),
    end_time=datetime.utcnow()
)

print(f"Total feedback: {aggregation.total_feedback}")
print(f"Positive: {aggregation.positive_feedback}")
print(f"Negative: {aggregation.negative_feedback}")
print(f"Corrections: {aggregation.corrections}")
print(f"Average rating: {aggregation.average_rating}")
```

---

## Quality Scoring

### Overview

The QualityScorer evaluates responses across five dimensions:

1. **Relevance**: How well the response addresses the query
2. **Helpfulness**: How useful the response is
3. **Engagement**: How engaging the response is
4. **Clarity**: How clear and understandable the response is
5. **Accuracy**: How accurate the information is

### Scoring a Response

```python
# Basic scoring
score = await quality_scorer.score_response(
    query="What is machine learning?",
    response="Machine learning is a subset of AI that enables systems to learn from data.",
    session_id="session-123"
)

print(f"Relevance: {score.relevance:.2f}")
print(f"Helpfulness: {score.helpfulness:.2f}")
print(f"Engagement: {score.engagement:.2f}")
print(f"Clarity: {score.clarity:.2f}")
print(f"Accuracy: {score.accuracy:.2f}")
print(f"Composite: {score.composite:.2f}")

# Check quality thresholds
if score.is_high_quality():
    print("High quality response!")
elif score.is_low_quality():
    print("Response needs improvement")
    print(f"Weakest dimension: {score.get_weakest_dimension()}")
```

### Enhanced Scoring (with context)

```python
# Score with conversation context
context = [
    {"user": "Tell me about AI", "agent": "AI is a broad field..."},
    {"user": "What are the types?", "agent": "There are many types..."}
]

score = await quality_scorer.score_response(
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    session_id="session-123",
    context=context
)

# Score with feedback history
feedback_history = await feedback_store.query(session_id="session-123", limit=10)

score = await quality_scorer.score_response(
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    session_id="session-123",
    feedback_history=feedback_history
)
```

### Batch Scoring

```python
# Score multiple interactions
interactions = [
    {"query": "What is AI?", "response": "AI is..."},
    {"query": "What is ML?", "response": "ML is..."},
    {"query": "What is DL?", "response": "DL is..."}
]

scores = await quality_scorer.batch_score(
    interactions=interactions,
    session_id="session-123"
)

for i, score in enumerate(scores):
    print(f"Interaction {i+1}: {score.composite:.2f}")
```

### Customizing Dimension Weights

```python
from app.learning.models import QualityDimension

# Adjust weights
quality_scorer.set_dimension_weights({
    QualityDimension.RELEVANCE: 0.30,
    QualityDimension.HELPFULNESS: 0.30,
    QualityDimension.ENGAGEMENT: 0.15,
    QualityDimension.CLARITY: 0.15,
    QualityDimension.ACCURACY: 0.10
})
```

---

## Response Adaptation

### Overview

The ResponseAdapter modifies responses based on learned user preferences.

### Getting Adaptation Context

```python
# Get or create adaptation context
context = await adapter.get_adaptation_context(
    session_id="session-123",
    user_id="user-456"  # Optional
)

print(f"Adaptation strength: {context.adaptation_strength}")
print(f"Preferences: {list(context.preferences.keys())}")
```

### Adapting Prompts

```python
# Original system prompt
base_prompt = """You are a helpful assistant. Answer questions clearly and concisely."""

# Get adaptation context
context = await adapter.get_adaptation_context(session_id="session-123")

# Adapt the prompt
adapted_prompt = await adapter.adapt_prompt(base_prompt, context)

# The adapted prompt may now include:
# - "Keep responses concise, under 3 sentences when possible."
# - "Use friendly, conversational language."
# - etc., based on learned preferences
```

### Enhancing Context

```python
# Get relevant history for context enhancement
relevant_history = [
    {"query": "What is AI?", "response": "AI is..."},
    {"query": "Types of AI?", "response": "There are..."}
]

# Enhance the query context
enhanced_context = await adapter.enhance_context(
    query="Tell me about machine learning",
    context=context,
    relevant_history=relevant_history
)

# Use enhanced context in your prompt
if enhanced_context:
    full_prompt = f"{enhanced_context}\n\nUser: {query}"
```

### Calibrating Responses

```python
# Generate a response
response = "Machine learning is a fascinating field..."

# Calibrate based on preferences
calibrated_response = await adapter.calibrate_response(
    response=response,
    context=context
)
```

### Updating Context

```python
# Update context with quality score
adapter.update_context_with_quality(context, quality_score)

# Update context with feedback
adapter.update_context_with_feedback(context, feedback)

# Add custom prompt modifier
adapter.add_prompt_modifier(
    context,
    key="technical_level",
    modifier="Explain concepts at an intermediate technical level."
)
```

---

## Preference Learning

### Overview

The PreferenceLearner learns user preferences from feedback patterns over time.

### Getting Preferences

```python
# Get all preferences for a session
preferences = await preference_learner.get_preferences(
    session_id="session-123"
)

for category, pref in preferences.items():
    print(f"{category}: {pref.value} (confidence: {pref.confidence:.2f})")

# Get a specific preference
response_length_pref = await preference_learner.get_preference(
    category="response_length",
    session_id="session-123"
)

if response_length_pref:
    print(f"Preferred response length: {response_length_pref.value}")
```

### Learning from Feedback

```python
# Extract response characteristics
response_characteristics = {
    "response_length": "detailed",
    "formality": "casual",
    "example_frequency": "frequent"
}

# Learn from feedback
updated_prefs = await preference_learner.learn_from_feedback(
    feedback=feedback,
    response_characteristics=response_characteristics
)

print(f"Updated {len(updated_prefs)} preferences")
```

### Learning from Engagement

```python
# Engagement metrics
engagement_metrics = {
    "time_spent_seconds": 45,
    "scroll_depth": 0.8,
    "interaction_count": 3
}

# Response characteristics
response_characteristics = {
    "response_length": "medium",
    "detail_level": "standard"
}

# Learn from engagement
updated_prefs = await preference_learner.learn_from_engagement(
    session_id="session-123",
    engagement_metrics=engagement_metrics,
    response_characteristics=response_characteristics
)
```

### Predicting Preferences

```python
# Predict the most likely preference value
prediction = await preference_learner.predict_preference(
    category="response_length",
    session_id="session-123"
)

if prediction:
    value, confidence = prediction
    print(f"Predicted: {value} (confidence: {confidence:.2f})")
```

### Getting Preference Summary

```python
# Get a summary of all learned preferences
summary = preference_learner.get_preference_summary(session_id="session-123")

print(f"Total preferences: {summary['total_preferences']}")
print(f"High confidence: {summary['high_confidence']}")
print(f"Medium confidence: {summary['medium_confidence']}")
print(f"Low confidence: {summary['low_confidence']}")
```

---

## Analytics and Insights

### Overview

The LearningAnalytics component generates reports and insights from learning data.

### Generating Daily Reports

```python
# Generate report for today
report = await analytics.generate_daily_report()

print(f"Date: {report.date}")
print(f"Total interactions: {report.total_interactions}")
print(f"Total feedback: {report.total_feedback_collected}")
print(f"Average quality: {report.average_quality_score:.2f}")
print(f"Quality trend: {report.quality_trend:+.2f}")

# Report for specific date
from datetime import datetime
report = await analytics.generate_daily_report(
    date=datetime(2025, 11, 20)
)
```

### Accessing Insights

```python
# Get insights from report
for insight in report.insights:
    print(f"\n{insight.title}")
    print(f"  {insight.description}")
    print(f"  Impact: {insight.impact}")
    print(f"  Recommendations:")
    for rec in insight.recommendations:
        print(f"    - {rec}")
```

### Getting Metrics Summary

```python
# Get metrics summary for last 7 days
summary = await analytics.get_metrics_summary(
    session_id="session-123",  # Optional
    days=7
)

print(f"Total scored: {summary['total_scored_interactions']}")
print(f"Average quality: {summary['average_quality']:.2f}")
print(f"Quality trend: {summary['quality_trend']}")
print(f"Dimension averages: {summary['dimension_averages']}")
```

### Calculating Dimension Trends

```python
from app.learning.models import QualityDimension

# Calculate trend for a specific dimension
trend = await analytics.calculate_trend(
    dimension=QualityDimension.RELEVANCE,
    session_id="session-123",
    days=7
)

print(f"Relevance trend:")
print(f"  Start: {trend.start_value:.2f}")
print(f"  End: {trend.end_value:.2f}")
print(f"  Change: {trend.change:+.2f} ({trend.change_percent*100:+.1f}%)")
print(f"  Improving: {trend.is_improving}")
```

### Recording Quality Scores

```python
# Record scores for analytics
analytics.record_quality_score(quality_score)

# Increment interaction count
analytics.increment_interaction_count("session-123")
```

---

## Pattern Detection

### Overview

The PatternDetector identifies recurring patterns in user interactions.

### Detecting All Patterns

```python
# Detect all pattern types
patterns = await pattern_detector.detect_patterns(
    session_id="session-123",  # Optional
    time_window_hours=168  # 1 week
)

for pattern in patterns:
    print(f"\n{pattern.pattern_type.upper()}: {pattern.description}")
    print(f"  Frequency: {pattern.frequency}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    if pattern.quality_correlation:
        print(f"  Quality correlation: {pattern.quality_correlation:+.2f}")
```

### Recording Interactions

```python
# Record interactions for pattern analysis
pattern_detector.record_interaction(
    query="What is machine learning?",
    response="Machine learning is...",
    quality_score=quality_score,
    session_id="session-123"
)
```

### Detecting Specific Pattern Types

```python
# Topic patterns
topic_patterns = await pattern_detector.detect_topic_patterns(
    session_id="session-123"
)
for p in topic_patterns:
    print(f"Topic: {p.metadata.get('topic')}")
    print(f"  Examples: {p.exemplars[:2]}")

# Time patterns
time_patterns = await pattern_detector.detect_time_patterns(
    session_id="session-123"
)
for p in time_patterns:
    if "peak_hour" in p.metadata:
        print(f"Peak hour: {p.metadata['peak_hour']}:00")
    if "peak_day" in p.metadata:
        print(f"Peak day: {p.metadata['peak_day']}")

# Quality correlations
quality_patterns = await pattern_detector.detect_quality_correlations(
    session_id="session-123"
)

# Engagement patterns
engagement_patterns = await pattern_detector.detect_engagement_patterns(
    session_id="session-123"
)
```

### Getting Pattern Summary

```python
# Get pattern summary with actionable insights
summary = await pattern_detector.get_pattern_summary(session_id="session-123")

print(f"Total patterns: {summary['total_patterns']}")
print(f"High confidence: {summary['high_confidence_patterns']}")
print(f"Actionable insights:")
for insight in summary['actionable_insights']:
    print(f"  - {insight}")
```

---

## Privacy Considerations

### Data Retention

By default, feedback data is retained for 90 days. Configure with:

```python
config.preference_learning.retention_days = 90
```

### Automatic Cleanup

```python
# Delete old feedback
deleted = await feedback_store.delete_old_feedback(retention_days=90)
print(f"Deleted {deleted} old records")
```

### Anonymization

Preferences are anonymized by default:

```python
config.preference_learning.anonymize_preferences = True
```

### User Control

Users can clear their preferences:

```python
await preference_learner.clear_preferences(
    session_id="session-123",
    user_id="user-456"
)
```

---

## Troubleshooting

### Common Issues

#### 1. Feedback Not Being Stored

```python
# Check if store is initialized
if not feedback_store._initialized:
    await feedback_store.initialize()

# Verify database path
print(f"Database path: {feedback_store.db_path}")
```

#### 2. Low Quality Scores

```python
# Debug quality scoring
score = await quality_scorer.score_response(
    query=query,
    response=response,
    session_id=session_id
)

# Check individual dimensions
print(f"Relevance: {score.relevance} (target: > 0.6)")
print(f"Helpfulness: {score.helpfulness} (target: > 0.6)")
print(f"Clarity: {score.clarity} (target: > 0.6)")
```

#### 3. Preferences Not Learning

```python
# Check learning history
history = await preference_learner.get_learning_history(session_id="session-123")
print(f"Learning events: {len(history)}")

# Check minimum samples requirement
print(f"Min samples: {config.preference_learning.min_samples_for_learning}")
```

#### 4. Patterns Not Detected

```python
# Check minimum frequency
print(f"Min pattern frequency: {config.pattern_detection.min_pattern_frequency}")

# Check recorded interactions
print(f"Cached interactions: {len(pattern_detector._query_cache)}")
```

### Debug Mode

Enable debug logging:

```python
config.debug_mode = True
config.log_level = "DEBUG"

import logging
logging.getLogger("app.learning").setLevel(logging.DEBUG)
```

---

## Performance Optimization

### Caching

Configure cache TTL:

```python
config.cache_ttl_seconds = 300  # 5 minutes
```

### Batch Processing

Use batch operations for high-volume scenarios:

```python
# Batch score multiple interactions
scores = await quality_scorer.batch_score(interactions, session_id)
```

### Async Processing

Enable async processing for non-blocking operations:

```python
config.async_processing = True
```

### Database Optimization

Periodically optimize the database:

```python
await feedback_store.vacuum()
```

### Memory Management

Clear caches when memory is constrained:

```python
# Clear pattern detector cache
pattern_detector.clear_cache()

# Clear adapter context cache
adapter.clear_context(session_id)
```

---

## Performance Benchmarks

| Operation | Target | Notes |
|-----------|--------|-------|
| Feedback collection | < 50ms | Storage + indexing |
| Quality scoring | < 200ms | All dimensions |
| Preference lookup | < 20ms | Cached |
| Analytics generation | < 1s | Daily report |
| Pattern detection | < 5s | 1000 queries |

---

## Integration Example

Complete integration with the conversation system:

```python
async def process_conversation(
    session_id: str,
    query: str,
    learning_system: dict
) -> str:
    """Process a conversation with learning integration."""

    # Extract components
    feedback_collector = learning_system["feedback_collector"]
    quality_scorer = learning_system["quality_scorer"]
    adapter = learning_system["adapter"]
    pattern_detector = learning_system["pattern_detector"]
    analytics = learning_system["analytics"]

    # Generate query ID
    query_id = f"{session_id}_{datetime.utcnow().timestamp()}"

    # Track interaction start
    await feedback_collector.track_interaction_start(
        session_id=session_id,
        query_id=query_id,
        query=query
    )

    # Get adaptation context
    context = await adapter.get_adaptation_context(session_id)

    # Adapt system prompt
    base_prompt = "You are a helpful assistant."
    adapted_prompt = await adapter.adapt_prompt(base_prompt, context)

    # Generate response (using your LLM)
    response = await generate_llm_response(adapted_prompt, query)

    # Track response delivery
    await feedback_collector.track_response_delivered(
        session_id=session_id,
        query_id=query_id,
        response=response
    )

    # Score the response
    score = await quality_scorer.score_response(
        query=query,
        response=response,
        session_id=session_id
    )

    # Record for analytics
    analytics.record_quality_score(score)
    analytics.increment_interaction_count(session_id)

    # Record for pattern detection
    pattern_detector.record_interaction(
        query=query,
        response=response,
        quality_score=score,
        session_id=session_id
    )

    return response
```

---

## Next Steps

- **Phase 6**: Integrate with mobile apps for cross-device learning
- **Phase 7**: Implement model fine-tuning based on learned preferences
- **Phase 8**: Add A/B testing framework for adaptation strategies

---

**For API details, see [PHASE5_API_REFERENCE.md](PHASE5_API_REFERENCE.md)**
**For testing guidance, see [PHASE5_TESTING_GUIDE.md](PHASE5_TESTING_GUIDE.md)**
