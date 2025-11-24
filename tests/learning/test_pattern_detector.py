"""
Test Suite: PatternDetector
===========================

20+ tests for pattern detection functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.learning.pattern_detector import (
    PatternDetector,
    PatternType,
    DetectedPattern,
    Cluster,
    Correlation
)
from app.learning.config import LearningConfig


class TestPatternDetectorInitialization:
    """Tests for pattern detector initialization."""

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self, learning_config):
        """Test that initialization sets the initialized flag."""
        detector = PatternDetector(config=learning_config)

        with patch('app.learning.pattern_detector.EmbeddingGenerator') as mock_emb:
            mock_instance = AsyncMock()
            mock_emb.return_value = mock_instance
            await detector.initialize()

        assert detector._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, learning_config):
        """Test that initialization only happens once."""
        detector = PatternDetector(config=learning_config)
        detector._initialized = True

        # Should return early without doing anything
        await detector.initialize()

        assert detector._embedding_generator is None

    @pytest.mark.asyncio
    async def test_initialize_handles_import_error(self, learning_config):
        """Test that initialization handles missing embedding generator."""
        detector = PatternDetector(config=learning_config)

        with patch('app.learning.pattern_detector.EmbeddingGenerator', side_effect=ImportError):
            await detector.initialize()

        # Should still be initialized, just without embeddings
        assert detector._initialized is True


class TestRecurringPatternDetection:
    """Tests for recurring pattern detection."""

    @pytest.mark.asyncio
    async def test_detect_recurring_patterns_empty_list(self, pattern_detector):
        """Test detection with empty query list."""
        patterns = await pattern_detector.detect_recurring_patterns([])
        assert patterns == []

    @pytest.mark.asyncio
    async def test_detect_recurring_patterns_returns_patterns(self, pattern_detector):
        """Test detection returns detected patterns."""
        queries = [
            "What is machine learning?",
            "What is ML?",
            "Explain machine learning",
            "How does machine learning work?",
            "Machine learning explanation",
            "Tell me about deep learning",
            "What is neural networks?"
        ]

        patterns = await pattern_detector.detect_recurring_patterns(queries)

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_detect_recurring_patterns_with_ids(self, pattern_detector):
        """Test detection with query IDs."""
        queries = ["Query A", "Query B", "Query A again"]
        query_ids = ["q1", "q2", "q3"]

        patterns = await pattern_detector.detect_recurring_patterns(
            queries, query_ids=query_ids
        )

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_detect_recurring_patterns_custom_threshold(self, pattern_detector):
        """Test detection with custom similarity threshold."""
        queries = ["Hello world", "Hello there", "Hi world"]

        patterns = await pattern_detector.detect_recurring_patterns(
            queries, similarity_threshold=0.3
        )

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_pattern_has_required_fields(self, pattern_detector):
        """Test that detected patterns have required fields."""
        queries = [
            "Python tutorial",
            "Python basics",
            "Learn Python",
            "Python programming"
        ] * 3  # Repeat to ensure minimum cluster size

        patterns = await pattern_detector.detect_recurring_patterns(queries)

        if patterns:
            pattern = patterns[0]
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'pattern_type')
            assert hasattr(pattern, 'description')
            assert hasattr(pattern, 'frequency')
            assert hasattr(pattern, 'confidence')


class TestQualityCorrelationDetection:
    """Tests for quality correlation detection."""

    @pytest.mark.asyncio
    async def test_detect_quality_correlations_empty(self, pattern_detector):
        """Test correlation detection with empty data."""
        correlations = await pattern_detector.detect_quality_correlations([])
        assert correlations == []

    @pytest.mark.asyncio
    async def test_detect_quality_correlations_insufficient_data(self, pattern_detector):
        """Test correlation detection with insufficient data."""
        data = [{"quality_score": 0.8, "response_length": 100}]

        correlations = await pattern_detector.detect_quality_correlations(data)
        assert correlations == []

    @pytest.mark.asyncio
    async def test_detect_quality_correlations_with_data(self, pattern_detector):
        """Test correlation detection with sufficient data."""
        data = [
            {"quality_score": 0.8, "response_length": 200, "timestamp": datetime.utcnow()},
            {"quality_score": 0.7, "response_length": 150, "timestamp": datetime.utcnow()},
            {"quality_score": 0.9, "response_length": 300, "timestamp": datetime.utcnow()},
            {"quality_score": 0.6, "response_length": 100, "timestamp": datetime.utcnow()},
            {"quality_score": 0.85, "response_length": 250, "timestamp": datetime.utcnow()},
            {"quality_score": 0.75, "response_length": 180, "timestamp": datetime.utcnow()},
            {"quality_score": 0.95, "response_length": 350, "timestamp": datetime.utcnow()},
            {"quality_score": 0.5, "response_length": 80, "timestamp": datetime.utcnow()},
            {"quality_score": 0.88, "response_length": 280, "timestamp": datetime.utcnow()},
            {"quality_score": 0.72, "response_length": 160, "timestamp": datetime.utcnow()},
        ]

        correlations = await pattern_detector.detect_quality_correlations(data)

        assert isinstance(correlations, list)

    @pytest.mark.asyncio
    async def test_correlation_has_required_fields(self, pattern_detector):
        """Test that correlations have required fields."""
        data = [
            {"quality_score": float(i) / 10, "response_length": i * 50}
            for i in range(1, 15)
        ]

        correlations = await pattern_detector.detect_quality_correlations(data)

        if correlations:
            corr = correlations[0]
            assert hasattr(corr, 'factor')
            assert hasattr(corr, 'target')
            assert hasattr(corr, 'correlation')
            assert hasattr(corr, 'p_value')
            assert hasattr(corr, 'sample_size')


class TestEngagementTriggerDetection:
    """Tests for engagement trigger detection."""

    @pytest.mark.asyncio
    async def test_detect_engagement_triggers_empty(self, pattern_detector):
        """Test engagement detection with empty data."""
        patterns = await pattern_detector.detect_engagement_triggers([])
        assert patterns == []

    @pytest.mark.asyncio
    async def test_detect_engagement_triggers_with_data(self, pattern_detector):
        """Test engagement detection with session data."""
        session_data = [
            {
                "start_time": datetime.utcnow().replace(hour=10),
                "duration": 300,
                "topics": ["python", "coding"]
            },
            {
                "start_time": datetime.utcnow().replace(hour=10),
                "duration": 600,
                "topics": ["python"]
            },
            {
                "start_time": datetime.utcnow().replace(hour=14),
                "duration": 200,
                "topics": ["machine learning"]
            },
            {
                "start_time": datetime.utcnow().replace(hour=10),
                "duration": 450,
                "topics": ["python", "data"]
            },
            {
                "start_time": datetime.utcnow().replace(hour=10),
                "duration": 350,
                "topics": ["coding"]
            },
            {
                "start_time": datetime.utcnow().replace(hour=10),
                "duration": 500,
                "topics": ["python"]
            },
        ]

        patterns = await pattern_detector.detect_engagement_triggers(session_data)

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_detect_time_engagement_patterns(self, pattern_detector):
        """Test time-based engagement pattern detection."""
        session_data = [
            {"start_time": datetime.utcnow().replace(hour=9), "duration": 100}
            for _ in range(6)
        ]

        patterns = await pattern_detector.detect_engagement_triggers(session_data)

        # Should detect peak hour pattern
        assert isinstance(patterns, list)


class TestDailyPatternDetection:
    """Tests for daily pattern detection."""

    @pytest.mark.asyncio
    async def test_detect_daily_patterns(self, pattern_detector):
        """Test daily pattern detection."""
        from datetime import date
        target_date = date.today()

        patterns = await pattern_detector.detect_daily_patterns(target_date)

        assert isinstance(patterns, list)


class TestPatternInsights:
    """Tests for pattern insight generation."""

    def test_get_pattern_insights_empty(self, pattern_detector):
        """Test insight generation with no patterns."""
        insights = pattern_detector.get_pattern_insights([])
        assert insights == []

    def test_get_pattern_insights_non_actionable(self, pattern_detector):
        """Test insight generation with non-actionable patterns."""
        patterns = [
            DetectedPattern(
                pattern_id="test-1",
                pattern_type=PatternType.RECURRING_QUESTION,
                description="Test pattern",
                frequency=5,
                confidence=0.8,
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                actionable=False
            )
        ]

        insights = pattern_detector.get_pattern_insights(patterns)

        # Non-actionable patterns should be filtered out
        assert len(insights) == 0

    def test_get_pattern_insights_actionable(self, pattern_detector):
        """Test insight generation with actionable patterns."""
        patterns = [
            DetectedPattern(
                pattern_id="test-1",
                pattern_type=PatternType.RECURRING_QUESTION,
                description="Test pattern",
                frequency=10,
                confidence=0.85,
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                actionable=True,
                recommendation="Create FAQ entry"
            )
        ]

        insights = pattern_detector.get_pattern_insights(patterns)

        assert len(insights) == 1
        assert 'pattern_id' in insights[0]
        assert 'title' in insights[0]
        assert 'recommendation' in insights[0]
        assert 'priority' in insights[0]

    def test_insights_sorted_by_priority(self, pattern_detector):
        """Test that insights are sorted by priority."""
        patterns = [
            DetectedPattern(
                pattern_id="low",
                pattern_type=PatternType.TOPIC_CLUSTER,
                description="Low priority",
                frequency=3,
                confidence=0.5,
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                actionable=True
            ),
            DetectedPattern(
                pattern_id="high",
                pattern_type=PatternType.COMMON_CORRECTION,
                description="High priority",
                frequency=15,
                confidence=0.9,
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                actionable=True
            )
        ]

        insights = pattern_detector.get_pattern_insights(patterns)

        if len(insights) >= 2:
            assert insights[0]['priority'] >= insights[1]['priority']


class TestClusterMethods:
    """Tests for clustering methods."""

    def test_normalize_text(self, pattern_detector):
        """Test text normalization."""
        text = "What is the best way to learn Python programming?"
        normalized = pattern_detector._normalize_text(text)

        assert isinstance(normalized, set)
        assert 'python' in normalized
        assert 'programming' in normalized
        # Stop words should be removed
        assert 'the' not in normalized
        assert 'is' not in normalized

    def test_jaccard_similarity(self, pattern_detector):
        """Test Jaccard similarity calculation."""
        set_a = {'python', 'programming', 'learn'}
        set_b = {'python', 'coding', 'learn'}

        similarity = pattern_detector._jaccard_similarity(set_a, set_b)

        # 2 common (python, learn) / 4 total (python, programming, coding, learn)
        assert 0 <= similarity <= 1
        assert similarity == 0.5

    def test_jaccard_similarity_identical(self, pattern_detector):
        """Test Jaccard similarity for identical sets."""
        set_a = {'a', 'b', 'c'}

        similarity = pattern_detector._jaccard_similarity(set_a, set_a)

        assert similarity == 1.0

    def test_jaccard_similarity_disjoint(self, pattern_detector):
        """Test Jaccard similarity for disjoint sets."""
        set_a = {'a', 'b'}
        set_b = {'c', 'd'}

        similarity = pattern_detector._jaccard_similarity(set_a, set_b)

        assert similarity == 0.0

    def test_jaccard_similarity_empty(self, pattern_detector):
        """Test Jaccard similarity with empty sets."""
        similarity = pattern_detector._jaccard_similarity(set(), {'a'})
        assert similarity == 0.0


class TestCorrelationMethods:
    """Tests for correlation calculation methods."""

    def test_pearson_correlation_perfect_positive(self, pattern_detector):
        """Test Pearson correlation for perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        r, p = pattern_detector._pearson_correlation(x, y)

        assert r is not None
        assert abs(r - 1.0) < 0.01

    def test_pearson_correlation_perfect_negative(self, pattern_detector):
        """Test Pearson correlation for perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]

        r, p = pattern_detector._pearson_correlation(x, y)

        assert r is not None
        assert abs(r - (-1.0)) < 0.01

    def test_pearson_correlation_insufficient_data(self, pattern_detector):
        """Test Pearson correlation with insufficient data."""
        x = [1.0, 2.0]
        y = [3.0, 4.0]

        r, p = pattern_detector._pearson_correlation(x, y)

        assert r is None
        assert p == 1.0


class TestPatternTypes:
    """Tests for pattern type handling."""

    def test_all_pattern_types_have_titles(self, pattern_detector):
        """Test that all pattern types have insight titles."""
        for pattern_type in PatternType:
            pattern = DetectedPattern(
                pattern_id="test",
                pattern_type=pattern_type,
                description="Test",
                frequency=5,
                confidence=0.8,
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                actionable=True
            )

            title = pattern_detector._generate_insight_title(pattern)
            assert title is not None
            assert len(title) > 0


class TestPatternDetectorFixture:
    """Tests using the pattern_detector fixture."""

    @pytest.fixture
    def pattern_detector(self, learning_config):
        """Create a pattern detector for testing."""
        detector = PatternDetector(config=learning_config)
        detector._initialized = True
        return detector

    @pytest.mark.asyncio
    async def test_cluster_with_text_similarity(self, pattern_detector):
        """Test text-based clustering."""
        texts = [
            "How to learn Python",
            "Learning Python basics",
            "Python tutorial for beginners",
            "JavaScript frameworks",
            "React vs Vue"
        ]
        text_ids = [f"t{i}" for i in range(len(texts))]

        clusters = pattern_detector._cluster_with_text_similarity(
            texts, text_ids, threshold=0.5
        )

        assert isinstance(clusters, list)
