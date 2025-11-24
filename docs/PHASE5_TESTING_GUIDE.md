# Phase 5: Real-Time Learning - Testing Guide

**Version:** 1.0.0
**Date:** 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Test Architecture](#test-architecture)
3. [Setting Up Test Environment](#setting-up-test-environment)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [Mock Patterns](#mock-patterns)
7. [Testing Quality Scoring](#testing-quality-scoring)
8. [Testing Adaptation](#testing-adaptation)
9. [Performance Testing](#performance-testing)
10. [Coverage Requirements](#coverage-requirements)

---

## Overview

This guide covers the testing strategy for Phase 5 Real-Time Learning components. The test suite includes 150+ tests with 80%+ coverage target.

### Test Suite Structure

```
tests/learning/
├── conftest.py                    # Shared fixtures
├── test_feedback_collector.py     # 25+ tests
├── test_feedback_store.py         # 20+ tests
├── test_quality_scorer.py         # 30+ tests
├── test_adapter.py                # 25+ tests
├── test_preference_learner.py     # 25+ tests
├── test_analytics.py              # 30+ tests
├── test_pattern_detector.py       # 20+ tests
└── __init__.py

tests/integration/
├── test_phase5_integration.py     # 25+ tests
└── ...
```

### Test Categories

| Category | Count | Coverage Target |
|----------|-------|-----------------|
| Feedback Collection | 45 | 85%+ |
| Quality Scoring | 30 | 85%+ |
| Adaptation | 25 | 80%+ |
| Preference Learning | 25 | 80%+ |
| Analytics | 30 | 80%+ |
| Pattern Detection | 20 | 80%+ |
| Integration | 25 | 75%+ |

---

## Test Architecture

### Testing Philosophy

1. **Unit Tests First**: Test individual functions in isolation
2. **Mock External Dependencies**: Database, embeddings, etc.
3. **Test Edge Cases**: Boundary conditions and error handling
4. **Integration Tests**: Verify component interactions
5. **Performance Tests**: Validate timing requirements

### Fixture Hierarchy

```
Global Fixtures (conftest.py)
    ↓
Module Fixtures (test_*.py)
    ↓
Test Functions
```

---

## Setting Up Test Environment

### Install Test Dependencies

```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock aiosqlite
```

### Configure pytest

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
```

### Run Tests

```bash
# Run all learning tests
pytest tests/learning -v

# Run with coverage
pytest tests/learning --cov=app.learning --cov-report=html

# Run specific test file
pytest tests/learning/test_quality_scorer.py -v

# Run specific test
pytest tests/learning/test_quality_scorer.py::test_score_response_calculates_all_dimensions -v
```

---

## Unit Testing

### Testing Feedback Collection

```python
# tests/learning/test_feedback_collector.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from app.learning.feedback_collector import FeedbackCollector
from app.learning.models import Feedback, FeedbackType, FeedbackSource

@pytest.fixture
def mock_feedback_store():
    """Mock feedback store for testing."""
    store = AsyncMock()
    store.initialize = AsyncMock()
    store.store = AsyncMock(return_value="feedback-123")
    store.query = AsyncMock(return_value=[])
    return store

@pytest.fixture
def feedback_collector(mock_feedback_store):
    """Feedback collector with mocked dependencies."""
    collector = FeedbackCollector(feedback_store=mock_feedback_store)
    return collector

@pytest.mark.asyncio
async def test_collect_rating_stores_feedback(feedback_collector, mock_feedback_store):
    """Test that collect_rating stores feedback correctly."""
    # Arrange
    session_id = "test-session"
    query_id = "test-query"
    rating = 0.8

    # Act
    result = await feedback_collector.collect_rating(
        session_id=session_id,
        query_id=query_id,
        rating=rating,
        original_query="What is AI?",
        original_response="AI is..."
    )

    # Assert
    assert result.session_id == session_id
    assert result.query_id == query_id
    assert result.rating == rating
    assert result.feedback_type == FeedbackType.EXPLICIT_RATING
    mock_feedback_store.store.assert_called_once()

@pytest.mark.asyncio
async def test_collect_thumbs_up_sets_positive_rating(feedback_collector):
    """Test that thumbs up sets rating to 1.0."""
    result = await feedback_collector.collect_thumbs_up(
        session_id="session",
        query_id="query",
        original_query="test",
        original_response="response"
    )

    assert result.rating == 1.0
    assert result.feedback_type == FeedbackType.EXPLICIT_POSITIVE

@pytest.mark.asyncio
async def test_detect_correction_explicit_phrases(feedback_collector):
    """Test detection of explicit correction phrases."""
    result = await feedback_collector.detect_correction(
        session_id="session",
        previous_query_id="query-1",
        new_query="No, I meant supervised learning",
        previous_query="What is machine learning?",
        previous_response="Machine learning is..."
    )

    assert result is not None
    assert result.feedback_type == FeedbackType.IMPLICIT_CORRECTION
    assert result.rating < 0.5
```

### Testing Feedback Store

```python
# tests/learning/test_feedback_store.py

import pytest
from datetime import datetime, timedelta
import tempfile
import os

from app.learning.feedback_store import FeedbackStore
from app.learning.models import Feedback, FeedbackType, FeedbackSource
from app.learning.config import LearningConfig

@pytest.fixture
async def temp_db_store():
    """Feedback store with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LearningConfig()
        config.feedback.database_path = os.path.join(tmpdir, "test_feedback.db")

        store = FeedbackStore(config)
        await store.initialize()
        yield store
        await store.close()

@pytest.fixture
def sample_feedback():
    """Sample feedback for testing."""
    return Feedback(
        session_id="test-session",
        query_id="test-query",
        feedback_type=FeedbackType.EXPLICIT_POSITIVE,
        source=FeedbackSource.USER_BUTTON,
        rating=0.9,
        original_query="What is AI?",
        original_response="AI is artificial intelligence."
    )

@pytest.mark.asyncio
async def test_store_and_retrieve_feedback(temp_db_store, sample_feedback):
    """Test storing and retrieving feedback."""
    # Store
    feedback_id = await temp_db_store.store(sample_feedback)

    # Retrieve
    retrieved = await temp_db_store.get(feedback_id)

    # Assert
    assert retrieved is not None
    assert retrieved.session_id == sample_feedback.session_id
    assert retrieved.rating == sample_feedback.rating

@pytest.mark.asyncio
async def test_query_by_session(temp_db_store, sample_feedback):
    """Test querying feedback by session."""
    await temp_db_store.store(sample_feedback)

    results = await temp_db_store.query(session_id="test-session")

    assert len(results) == 1
    assert results[0].session_id == "test-session"

@pytest.mark.asyncio
async def test_query_by_time_range(temp_db_store, sample_feedback):
    """Test querying feedback by time range."""
    await temp_db_store.store(sample_feedback)

    results = await temp_db_store.query(
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow() + timedelta(hours=1)
    )

    assert len(results) == 1

@pytest.mark.asyncio
async def test_delete_old_feedback(temp_db_store, sample_feedback):
    """Test deletion of old feedback."""
    # Store feedback with old timestamp
    old_feedback = sample_feedback.model_copy()
    old_feedback.timestamp = datetime.utcnow() - timedelta(days=100)
    await temp_db_store.store(old_feedback)

    # Delete old feedback
    deleted = await temp_db_store.delete_old_feedback(retention_days=90)

    assert deleted == 1
```

---

## Mock Patterns

### Mocking Embeddings

```python
@pytest.fixture
def mock_embeddings():
    """Mock embedding generator."""
    async def generate(text):
        # Return consistent embedding based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [float(hash_val >> i & 1) for i in range(384)]

    mock = AsyncMock()
    mock.generate = generate
    return mock
```

### Mocking Quality Scores

```python
@pytest.fixture
def sample_quality_scores():
    """Generate sample quality scores."""
    def _generate(n=10, base_score=0.7, variance=0.1):
        import random
        scores = []
        for i in range(n):
            score = QualityScore(
                query_id=f"query-{i}",
                session_id="test-session",
                relevance=base_score + random.uniform(-variance, variance),
                helpfulness=base_score + random.uniform(-variance, variance),
                engagement=base_score + random.uniform(-variance, variance),
                clarity=base_score + random.uniform(-variance, variance),
                accuracy=base_score + random.uniform(-variance, variance),
                composite=base_score
            )
            scores.append(score)
        return scores
    return _generate
```

### Mocking Feedback History

```python
@pytest.fixture
def mock_feedback_history():
    """Generate mock feedback history."""
    def _generate(n=5, positive_ratio=0.7):
        feedback_list = []
        for i in range(n):
            is_positive = i / n < positive_ratio
            feedback_list.append(Feedback(
                session_id="test-session",
                query_id=f"query-{i}",
                feedback_type=FeedbackType.EXPLICIT_POSITIVE if is_positive else FeedbackType.EXPLICIT_NEGATIVE,
                source=FeedbackSource.USER_BUTTON,
                rating=0.8 if is_positive else 0.2,
                original_query=f"Question {i}",
                original_response=f"Answer {i}"
            ))
        return feedback_list
    return _generate
```

---

## Testing Quality Scoring

### Score Calculation Tests

```python
# tests/learning/test_quality_scorer.py

import pytest
from app.learning.quality_scorer import QualityScorer
from app.learning.models import QualityScore, QualityDimension

@pytest.fixture
def quality_scorer(mock_embeddings, mock_feedback_store):
    """Quality scorer with mocked dependencies."""
    return QualityScorer(
        embedding_generator=mock_embeddings,
        feedback_store=mock_feedback_store
    )

@pytest.mark.asyncio
async def test_score_response_returns_valid_score(quality_scorer):
    """Test that scoring returns valid QualityScore."""
    score = await quality_scorer.score_response(
        query="What is machine learning?",
        response="Machine learning is a subset of AI.",
        session_id="test-session"
    )

    assert isinstance(score, QualityScore)
    assert 0 <= score.relevance <= 1
    assert 0 <= score.helpfulness <= 1
    assert 0 <= score.engagement <= 1
    assert 0 <= score.clarity <= 1
    assert 0 <= score.accuracy <= 1
    assert 0 <= score.composite <= 1

@pytest.mark.asyncio
async def test_high_relevance_for_matching_content(quality_scorer):
    """Test high relevance for query-response match."""
    score = await quality_scorer.score_response(
        query="Explain neural networks",
        response="Neural networks are computing systems inspired by biological neural networks.",
        session_id="test-session"
    )

    assert score.relevance > 0.5

@pytest.mark.asyncio
async def test_low_clarity_for_complex_text(quality_scorer):
    """Test lower clarity for complex responses."""
    complex_response = (
        "The epistemological ramifications of superintelligent artificial general "
        "intelligence necessitate unprecedented reconceptualization of anthropocentric "
        "paradigms vis-a-vis technological singularity hypotheses."
    ) * 3

    score = await quality_scorer.score_response(
        query="What is AI?",
        response=complex_response,
        session_id="test-session"
    )

    assert score.clarity < 0.6

@pytest.mark.asyncio
async def test_composite_score_is_weighted_average(quality_scorer):
    """Test composite score calculation."""
    score = await quality_scorer.score_response(
        query="Test query",
        response="Test response",
        session_id="test-session"
    )

    weights = quality_scorer.get_dimension_weights()
    expected_composite = (
        weights[QualityDimension.RELEVANCE] * score.relevance +
        weights[QualityDimension.HELPFULNESS] * score.helpfulness +
        weights[QualityDimension.ENGAGEMENT] * score.engagement +
        weights[QualityDimension.CLARITY] * score.clarity +
        weights[QualityDimension.ACCURACY] * score.accuracy
    ) / sum(weights.values())

    assert abs(score.composite - expected_composite) < 0.01

@pytest.mark.asyncio
async def test_batch_scoring(quality_scorer):
    """Test batch scoring of multiple interactions."""
    interactions = [
        {"query": "What is AI?", "response": "AI is..."},
        {"query": "What is ML?", "response": "ML is..."},
        {"query": "What is DL?", "response": "DL is..."}
    ]

    scores = await quality_scorer.batch_score(interactions, "test-session")

    assert len(scores) == 3
    for score in scores:
        assert isinstance(score, QualityScore)
```

---

## Testing Adaptation

### Response Adapter Tests

```python
# tests/learning/test_adapter.py

import pytest
from app.learning.adapter import ResponseAdapter
from app.learning.models import AdaptationContext, UserPreference

@pytest.fixture
def adapter():
    """Response adapter with default config."""
    return ResponseAdapter()

@pytest.fixture
def context_with_preferences():
    """Adaptation context with preferences."""
    context = AdaptationContext(
        session_id="test-session",
        preferences={
            "response_length": UserPreference(
                category="response_length",
                value="brief",
                confidence=0.8
            ),
            "formality": UserPreference(
                category="formality",
                value="casual",
                confidence=0.7
            )
        }
    )
    return context

@pytest.mark.asyncio
async def test_adapt_prompt_adds_length_modifier(adapter, context_with_preferences):
    """Test prompt adaptation adds length modifier."""
    base_prompt = "You are a helpful assistant."

    adapted = await adapter.adapt_prompt(base_prompt, context_with_preferences)

    assert "concise" in adapted.lower() or "brief" in adapted.lower()

@pytest.mark.asyncio
async def test_adapt_prompt_adds_formality_modifier(adapter, context_with_preferences):
    """Test prompt adaptation adds formality modifier."""
    base_prompt = "You are a helpful assistant."

    adapted = await adapter.adapt_prompt(base_prompt, context_with_preferences)

    assert "casual" in adapted.lower() or "friendly" in adapted.lower()

@pytest.mark.asyncio
async def test_calibrate_response_length(adapter):
    """Test response length calibration."""
    long_response = "word " * 200  # 200 words
    context = AdaptationContext(
        session_id="test",
        target_length=50
    )

    calibrated = await adapter.calibrate_response(long_response, context)

    assert len(calibrated.split()) <= 55  # Allow small margin

@pytest.mark.asyncio
async def test_calibrate_formality_to_formal(adapter):
    """Test formality calibration to formal."""
    casual_response = "That's great! You can't go wrong with it."
    context = AdaptationContext(
        session_id="test",
        target_formality="formal"
    )

    calibrated = await adapter.calibrate_response(casual_response, context)

    assert "cannot" in calibrated
    assert "That is" in calibrated or "great" in calibrated

@pytest.mark.asyncio
async def test_get_adaptation_context_caches(adapter):
    """Test that adaptation context is cached."""
    context1 = await adapter.get_adaptation_context("session-123")
    context2 = await adapter.get_adaptation_context("session-123")

    assert context1 is context2

@pytest.mark.asyncio
async def test_clear_context_removes_cache(adapter):
    """Test context clearing."""
    await adapter.get_adaptation_context("session-123")
    adapter.clear_context("session-123")
    context2 = await adapter.get_adaptation_context("session-123")

    # Should be a new context
    assert context2.preferences == {}
```

---

## Performance Testing

### Benchmark Tests

```python
# tests/learning/test_performance.py

import pytest
import time
import asyncio

@pytest.fixture
def benchmark():
    """Simple benchmark fixture."""
    class Benchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self):
            return (self.end_time - self.start_time) * 1000

    return Benchmark

@pytest.mark.asyncio
async def test_feedback_collection_under_50ms(feedback_collector, benchmark):
    """Test feedback collection completes in under 50ms."""
    with benchmark() as b:
        await feedback_collector.collect_rating(
            session_id="test",
            query_id="test",
            rating=0.8,
            original_query="test",
            original_response="test"
        )

    assert b.elapsed_ms < 50, f"Feedback collection took {b.elapsed_ms:.2f}ms"

@pytest.mark.asyncio
async def test_quality_scoring_under_200ms(quality_scorer, benchmark):
    """Test quality scoring completes in under 200ms."""
    with benchmark() as b:
        await quality_scorer.score_response(
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            session_id="test"
        )

    assert b.elapsed_ms < 200, f"Quality scoring took {b.elapsed_ms:.2f}ms"

@pytest.mark.asyncio
async def test_preference_lookup_under_20ms(preference_learner, benchmark):
    """Test preference lookup completes in under 20ms."""
    # Pre-populate some preferences
    await preference_learner.get_preferences("test-session")

    with benchmark() as b:
        await preference_learner.get_preference("response_length", "test-session")

    assert b.elapsed_ms < 20, f"Preference lookup took {b.elapsed_ms:.2f}ms"
```

---

## Coverage Requirements

### Target Coverage by Component

| Component | Target | Critical Methods |
|-----------|--------|------------------|
| feedback_collector.py | 85% | collect_*, detect_*, track_* |
| feedback_store.py | 85% | store, query, delete |
| quality_scorer.py | 85% | score_response, _score_* |
| adapter.py | 80% | adapt_prompt, calibrate_response |
| preference_learner.py | 80% | learn_from_*, get_preference |
| analytics.py | 80% | generate_daily_report |
| pattern_detector.py | 80% | detect_*_patterns |

### Running Coverage Report

```bash
# Generate HTML coverage report
pytest tests/learning --cov=app.learning --cov-report=html --cov-fail-under=80

# View report
open htmlcov/index.html
```

### Coverage Exclusions

```ini
# .coveragerc
[run]
omit =
    */tests/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
```

---

## Integration Testing

### End-to-End Learning Loop

```python
# tests/integration/test_phase5_integration.py

import pytest

@pytest.mark.asyncio
async def test_full_learning_loop(learning_system):
    """Test complete feedback -> quality -> adaptation loop."""
    feedback_collector = learning_system["feedback_collector"]
    quality_scorer = learning_system["quality_scorer"]
    adapter = learning_system["adapter"]
    preference_learner = learning_system["preference_learner"]

    session_id = "integration-test-session"
    query = "What is machine learning?"
    response = "Machine learning is a fascinating field..."

    # 1. Score the response
    score = await quality_scorer.score_response(query, response, session_id)
    assert score.composite > 0

    # 2. Collect positive feedback
    feedback = await feedback_collector.collect_thumbs_up(
        session_id=session_id,
        query_id=score.query_id,
        original_query=query,
        original_response=response
    )
    assert feedback.rating == 1.0

    # 3. Learn from feedback
    characteristics = {"response_length": "detailed", "formality": "neutral"}
    updated = await preference_learner.learn_from_feedback(
        feedback, characteristics
    )
    assert len(updated) > 0

    # 4. Get adapted context
    context = await adapter.get_adaptation_context(session_id)
    preferences = await preference_learner.get_preferences(session_id)

    # 5. Adapt prompt
    base_prompt = "You are a helpful assistant."
    adapted = await adapter.adapt_prompt(base_prompt, context)

    # Adaptation may or may not modify prompt based on confidence
    assert len(adapted) >= len(base_prompt)

@pytest.mark.asyncio
async def test_analytics_pipeline(learning_system):
    """Test analytics generation pipeline."""
    analytics = learning_system["analytics"]
    quality_scorer = learning_system["quality_scorer"]

    # Generate some scores
    for i in range(5):
        score = await quality_scorer.score_response(
            query=f"Question {i}",
            response=f"Answer {i}",
            session_id="analytics-test"
        )
        analytics.record_quality_score(score)

    # Generate report
    report = await analytics.generate_daily_report()

    assert report.total_interactions >= 0
    assert report.average_quality_score >= 0

@pytest.mark.asyncio
async def test_performance_under_load(learning_system):
    """Test system performance under load."""
    import asyncio

    feedback_collector = learning_system["feedback_collector"]
    quality_scorer = learning_system["quality_scorer"]

    async def process_interaction(i):
        session_id = f"load-test-{i % 10}"
        query_id = f"query-{i}"

        await quality_scorer.score_response(
            query=f"Question {i}",
            response=f"Answer {i}",
            session_id=session_id
        )

        await feedback_collector.collect_rating(
            session_id=session_id,
            query_id=query_id,
            rating=0.7,
            original_query=f"Question {i}",
            original_response=f"Answer {i}"
        )

    # Process 50 interactions concurrently
    start = time.time()
    await asyncio.gather(*[process_interaction(i) for i in range(50)])
    elapsed = time.time() - start

    # Should complete in under 10 seconds
    assert elapsed < 10, f"Load test took {elapsed:.2f}s"
```

---

**For implementation details, see [PHASE5_IMPLEMENTATION_GUIDE.md](PHASE5_IMPLEMENTATION_GUIDE.md)**
**For API details, see [PHASE5_API_REFERENCE.md](PHASE5_API_REFERENCE.md)**
