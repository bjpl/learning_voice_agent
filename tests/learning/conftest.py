"""
Test Configuration and Fixtures for Phase 5 Learning System
============================================================

Provides shared fixtures for all learning component tests.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
import hashlib
import json

# Import learning components
from app.learning.config import LearningConfig, FeedbackConfig, learning_config
from app.learning.models import (
    Feedback,
    FeedbackType,
    QualityScore,
    UserPreference,
)

# Optional imports that may not exist in all configurations
try:
    from app.learning.models import (
        FeedbackSource,
        LearningInsight,
        InteractionPattern,
        DailyReport,
        AdaptationContext,
        QualityDimension
    )
except ImportError:
    FeedbackSource = None
    LearningInsight = None
    InteractionPattern = None
    DailyReport = None
    AdaptationContext = None
    QualityDimension = None

# Import scoring config if available
try:
    from app.learning.scoring_algorithms import ScoringConfig as QualityScoringConfig
except ImportError:
    QualityScoringConfig = None


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def learning_config():
    """Default learning configuration for testing."""
    config = LearningConfig()
    config.debug_mode = True
    config.enabled = True
    return config


@pytest.fixture
def test_config_with_temp_db():
    """Learning config with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LearningConfig()
        # Set database paths that exist on LearningConfig
        config.db_path = os.path.join(tmpdir, "learning.db")
        config.learning_db_path = os.path.join(tmpdir, "learning_data.db")
        # Set feedback database path
        config.feedback.database_path = os.path.join(tmpdir, "test_feedback.db")
        yield config


# ============================================================================
# Mock Embedding Generator
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Mock embedding generator that returns consistent embeddings."""
    async def generate(text: str) -> List[float]:
        """Generate deterministic embedding based on text hash."""
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Create 384-dimensional embedding from hash
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)
        return embedding

    mock = MagicMock()
    mock.generate = AsyncMock(side_effect=generate)
    return mock


# ============================================================================
# Mock Feedback Store
# ============================================================================

@pytest.fixture
def mock_feedback_store():
    """Mock feedback store for testing."""
    store = AsyncMock()
    store._feedback_data: Dict[str, Feedback] = {}

    async def mock_store(feedback: Feedback) -> str:
        store._feedback_data[feedback.id] = feedback
        return feedback.id

    async def mock_get(feedback_id: str) -> Feedback:
        return store._feedback_data.get(feedback_id)

    async def mock_query(
        session_id: str = None,
        query_id: str = None,
        feedback_type: FeedbackType = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        **kwargs
    ) -> List[Feedback]:
        results = list(store._feedback_data.values())
        if session_id:
            results = [f for f in results if f.session_id == session_id]
        if query_id:
            results = [f for f in results if f.query_id == query_id]
        if feedback_type:
            results = [f for f in results if f.feedback_type == feedback_type]
        return results[:limit]

    async def mock_count(**kwargs) -> int:
        return len(await mock_query(**kwargs))

    async def mock_get_average_rating(**kwargs) -> float:
        feedbacks = await mock_query(**kwargs)
        ratings = [f.rating for f in feedbacks if f.rating is not None]
        return sum(ratings) / len(ratings) if ratings else 0.0

    async def mock_get_feedback_distribution(**kwargs) -> Dict[str, int]:
        feedbacks = await mock_query(**kwargs)
        dist = {}
        for f in feedbacks:
            dist[f.feedback_type.value] = dist.get(f.feedback_type.value, 0) + 1
        return dist

    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.store = AsyncMock(side_effect=mock_store)
    store.get = AsyncMock(side_effect=mock_get)
    store.query = AsyncMock(side_effect=mock_query)
    store.count = AsyncMock(side_effect=mock_count)
    store.get_average_rating = AsyncMock(side_effect=mock_get_average_rating)
    store.get_feedback_distribution = AsyncMock(side_effect=mock_get_feedback_distribution)
    store.delete = AsyncMock(return_value=True)
    store.delete_old_feedback = AsyncMock(return_value=0)
    store.vacuum = AsyncMock()

    return store


# ============================================================================
# Sample Data Generators
# ============================================================================

@pytest.fixture
def sample_feedback():
    """Generate a sample feedback object."""
    def _generate(
        session_id: str = "test-session",
        query_id: str = "test-query",
        feedback_type: FeedbackType = FeedbackType.EXPLICIT_POSITIVE,
        rating: float = 0.8,
        **kwargs
    ) -> Feedback:
        return Feedback(
            session_id=session_id,
            query_id=query_id,
            feedback_type=feedback_type,
            source=kwargs.get("source", FeedbackSource.USER_BUTTON),
            rating=rating,
            text=kwargs.get("text"),
            correction=kwargs.get("correction"),
            original_query=kwargs.get("original_query", "What is AI?"),
            original_response=kwargs.get("original_response", "AI is artificial intelligence."),
            user_id=kwargs.get("user_id"),
            metadata=kwargs.get("metadata", {})
        )
    return _generate


@pytest.fixture
def sample_feedback_list(sample_feedback):
    """Generate a list of sample feedback objects."""
    def _generate(n: int = 10, positive_ratio: float = 0.7) -> List[Feedback]:
        feedbacks = []
        for i in range(n):
            is_positive = (i / n) < positive_ratio
            feedbacks.append(sample_feedback(
                query_id=f"query-{i}",
                feedback_type=FeedbackType.EXPLICIT_POSITIVE if is_positive else FeedbackType.EXPLICIT_NEGATIVE,
                rating=0.8 if is_positive else 0.3
            ))
        return feedbacks
    return _generate


@pytest.fixture
def sample_quality_score():
    """Generate a sample quality score."""
    def _generate(
        session_id: str = "test-session",
        query_id: str = "test-query",
        composite: float = 0.75,
        **kwargs
    ) -> QualityScore:
        return QualityScore(
            query_id=query_id,
            session_id=session_id,
            relevance=kwargs.get("relevance", composite + 0.05),
            helpfulness=kwargs.get("helpfulness", composite),
            engagement=kwargs.get("engagement", composite - 0.05),
            clarity=kwargs.get("clarity", composite + 0.02),
            accuracy=kwargs.get("accuracy", composite),
            composite=composite,
            query_text=kwargs.get("query_text", "What is machine learning?"),
            response_text=kwargs.get("response_text", "Machine learning is...")
        )
    return _generate


@pytest.fixture
def sample_quality_scores(sample_quality_score):
    """Generate a list of quality scores."""
    def _generate(
        n: int = 10,
        session_id: str = "test-session",
        base_score: float = 0.7,
        variance: float = 0.1
    ) -> List[QualityScore]:
        import random
        scores = []
        for i in range(n):
            score = base_score + random.uniform(-variance, variance)
            scores.append(sample_quality_score(
                session_id=session_id,
                query_id=f"query-{i}",
                composite=max(0.0, min(1.0, score))
            ))
        return scores
    return _generate


@pytest.fixture
def sample_user_preference():
    """Generate a sample user preference."""
    def _generate(
        category: str = "response_length",
        value: Any = "detailed",
        confidence: float = 0.7,
        **kwargs
    ) -> UserPreference:
        return UserPreference(
            session_id=kwargs.get("session_id", "test-session"),
            user_id=kwargs.get("user_id"),
            category=category,
            value=value,
            confidence=confidence,
            learned_from_samples=kwargs.get("learned_from_samples", 5)
        )
    return _generate


@pytest.fixture
def sample_adaptation_context(sample_user_preference):
    """Generate a sample adaptation context."""
    def _generate(
        session_id: str = "test-session",
        with_preferences: bool = True
    ) -> AdaptationContext:
        context = AdaptationContext(session_id=session_id)

        if with_preferences:
            context.preferences = {
                "response_length": sample_user_preference("response_length", "detailed", 0.8),
                "formality": sample_user_preference("formality", "casual", 0.6),
                "example_frequency": sample_user_preference("example_frequency", "frequent", 0.5)
            }

        return context
    return _generate


@pytest.fixture
def sample_interaction_pattern():
    """Generate a sample interaction pattern."""
    def _generate(
        pattern_type: str = "topic",
        description: str = "Recurring topic: machine learning",
        **kwargs
    ) -> InteractionPattern:
        return InteractionPattern(
            pattern_type=pattern_type,
            description=description,
            frequency=kwargs.get("frequency", 10),
            confidence=kwargs.get("confidence", 0.8),
            first_seen=kwargs.get("first_seen", datetime.utcnow() - timedelta(days=7)),
            last_seen=kwargs.get("last_seen", datetime.utcnow()),
            exemplars=kwargs.get("exemplars", ["What is ML?", "How does ML work?"]),
            quality_correlation=kwargs.get("quality_correlation", 0.3),
            metadata=kwargs.get("metadata", {"topic": "machine learning"})
        )
    return _generate


@pytest.fixture
def sample_learning_insight():
    """Generate a sample learning insight."""
    def _generate(
        title: str = "Low Quality Responses Detected",
        category: str = "quality",
        **kwargs
    ) -> LearningInsight:
        return LearningInsight(
            title=title,
            description=kwargs.get("description", "Average quality is below threshold."),
            category=category,
            relevance_score=kwargs.get("relevance_score", 0.9),
            confidence=kwargs.get("confidence", 0.8),
            impact=kwargs.get("impact", "high"),
            recommendations=kwargs.get("recommendations", ["Review responses", "Adjust prompts"])
        )
    return _generate


# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def feedback_collector(learning_config, mock_feedback_store):
    """FeedbackCollector with mocked dependencies."""
    from app.learning.feedback_collector import FeedbackCollector
    collector = FeedbackCollector(
        config=learning_config,
        feedback_store=mock_feedback_store
    )
    return collector


@pytest.fixture
def quality_scorer(learning_config, mock_embeddings, mock_feedback_store):
    """QualityScorer with mocked dependencies."""
    from app.learning.quality_scorer import QualityScorer
    scorer = QualityScorer(
        config=learning_config,
        embedding_generator=mock_embeddings,
        feedback_store=mock_feedback_store
    )
    return scorer


@pytest.fixture
def response_adapter(learning_config):
    """ResponseAdapter with default config."""
    from app.learning.adapter import ResponseAdapter
    return ResponseAdapter(config=learning_config)


@pytest.fixture
def preference_learner(learning_config, mock_feedback_store):
    """PreferenceLearner with mocked dependencies."""
    from app.learning.preference_learner import PreferenceLearner
    learner = PreferenceLearner(
        config=learning_config,
        feedback_store=mock_feedback_store
    )
    return learner


@pytest.fixture
def learning_analytics(learning_config, mock_feedback_store):
    """LearningAnalytics with mocked dependencies."""
    from app.learning.analytics import LearningAnalytics
    # LearningAnalytics.__init__ takes feedback= not feedback_store=
    analytics = LearningAnalytics(
        config=learning_config,
        feedback=mock_feedback_store
    )
    return analytics


@pytest.fixture
def pattern_detector(learning_config):
    """PatternDetector with default config."""
    from app.learning.pattern_detector import PatternDetector
    detector = PatternDetector(config=learning_config)
    return detector


# ============================================================================
# Integration Fixtures
# ============================================================================

@pytest.fixture
async def initialized_feedback_store(test_config_with_temp_db):
    """Real feedback store with temporary database."""
    from app.learning.feedback_store import FeedbackStore
    # FeedbackStore expects db_path string, not a config object
    store = FeedbackStore(db_path=test_config_with_temp_db.feedback.database_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def full_learning_system(test_config_with_temp_db):
    """Complete learning system for integration tests."""
    from app.learning.feedback_collector import FeedbackCollector
    from app.learning.feedback_store import FeedbackStore
    from app.learning.quality_scorer import QualityScorer
    from app.learning.adapter import ResponseAdapter
    from app.learning.preference_learner import PreferenceLearner
    from app.learning.analytics import LearningAnalytics
    from app.learning.pattern_detector import PatternDetector

    config = test_config_with_temp_db

    feedback_store = FeedbackStore(config)
    await feedback_store.initialize()

    feedback_collector = FeedbackCollector(config=config, feedback_store=feedback_store)
    await feedback_collector.initialize()

    quality_scorer = QualityScorer(config=config, feedback_store=feedback_store)

    preference_learner = PreferenceLearner(config=config, feedback_store=feedback_store)
    await preference_learner.initialize()

    adapter = ResponseAdapter(config=config, preference_learner=preference_learner)

    analytics = LearningAnalytics(config=config, feedback=feedback_store)
    await analytics.initialize()

    pattern_detector = PatternDetector(config=config)
    await pattern_detector.initialize()

    system = {
        "config": config,
        "feedback_store": feedback_store,
        "feedback_collector": feedback_collector,
        "quality_scorer": quality_scorer,
        "preference_learner": preference_learner,
        "adapter": adapter,
        "analytics": analytics,
        "pattern_detector": pattern_detector
    }

    yield system

    # Cleanup
    await feedback_collector.close()
    await preference_learner.close()
    await feedback_store.close()


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def benchmark():
    """Simple benchmark fixture for performance testing."""
    import time

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

        def assert_under(self, max_ms: float, message: str = None):
            assert self.elapsed_ms < max_ms, \
                message or f"Operation took {self.elapsed_ms:.2f}ms (max: {max_ms}ms)"

    return Benchmark


@pytest.fixture
def assert_called_once():
    """Helper to assert async mock called once."""
    def _assert(mock, *args, **kwargs):
        assert mock.call_count == 1
        if args or kwargs:
            mock.assert_called_once_with(*args, **kwargs)
    return _assert


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Add any cleanup logic here if needed
