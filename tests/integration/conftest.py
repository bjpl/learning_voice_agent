"""
Integration Test Configuration and Fixtures
Provides fixtures for integration testing
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock


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
# Learning System Configuration
# ============================================================================

@pytest.fixture
def test_config_with_temp_db():
    """Learning config with temporary database."""
    from app.learning.config import LearningConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LearningConfig()
        config.db_path = os.path.join(tmpdir, "learning.db")
        config.learning_db_path = os.path.join(tmpdir, "learning_data.db")
        config.feedback.database_path = os.path.join(tmpdir, "test_feedback.db")
        # preferences uses preference_file attribute
        if hasattr(config.preferences, 'preference_file'):
            config.preferences.preference_file = os.path.join(tmpdir, "test_prefs.json")
        config.analytics.db_path = os.path.join(tmpdir, "test_analytics.db")
        yield config


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

    # FeedbackStore expects db_path string, not config object
    feedback_store = FeedbackStore(db_path=config.feedback.database_path)
    await feedback_store.initialize()

    feedback_collector = FeedbackCollector(config=config, feedback_store=feedback_store)
    await feedback_collector.initialize()

    quality_scorer = QualityScorer(config=config, feedback_store=feedback_store)

    preference_learner = PreferenceLearner(config=config, feedback_store=feedback_store)
    await preference_learner.initialize()

    # ResponseAdapter takes store and config (AdaptationConfig), not preference_learner
    adapter = ResponseAdapter(config=config.adaptation)

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
# Research Agent Fixtures
# ============================================================================

@pytest.fixture
def research_agent():
    """Create a research agent instance for testing"""
    from app.agents.research_agent import ResearchAgent
    agent = ResearchAgent(agent_id="research-agent-1")
    agent._initialized = True
    return agent


# ============================================================================
# Hybrid Search Fixtures
# ============================================================================

@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph store for hybrid search tests"""
    store = AsyncMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.query_related_concepts = AsyncMock(return_value=[
        {'name': 'concept1', 'relevance': 0.9},
        {'name': 'concept2', 'relevance': 0.8}
    ])
    store.add_concept = AsyncMock()
    store.add_relationship = AsyncMock()
    return store


@pytest.fixture
def mock_vector_store_for_hybrid():
    """Mock vector store for hybrid search tests"""
    store = AsyncMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.search = AsyncMock(return_value=[
        {'id': 'doc1', 'score': 0.95, 'content': 'Test document 1'},
        {'id': 'doc2', 'score': 0.85, 'content': 'Test document 2'}
    ])
    store.add = AsyncMock()
    return store


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Cleanup logic if needed
