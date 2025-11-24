"""
Tests for ImprovementEngine - Continuous Improvement through A/B Testing

PATTERN: Comprehensive tests for experimentation system
WHY: Ensure A/B testing and improvement logic works correctly
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import uuid

from app.learning.improvement_engine import (
    ImprovementEngine,
    Improvement,
    ImprovementStatus,
    ImprovementDimension,
    WeakArea,
    ABTestResult,
    QualityScorer,
    create_improvement_engine,
)
from app.learning.config import ImprovementConfig, ABTestingConfig
from app.learning.store import LearningStore, ImprovementRecord


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_store():
    """Create mock learning store"""
    store = AsyncMock(spec=LearningStore)
    store.get_quality_stats = AsyncMock(return_value={
        "total_responses": 100,
        "avg_quality": 0.65,
        "avg_relevance": 0.7,
        "avg_coherence": 0.72,
        "avg_helpfulness": 0.6,
        "min_quality": 0.3,
        "max_quality": 0.95
    })
    store.get_feedback_stats = AsyncMock(return_value={
        "total_feedback": 50,
        "helpful_count": 30,
        "not_helpful_count": 20,
        "helpful_rate": 0.6,
        "average_rating": 3.5,
        "correction_count": 5
    })
    store.save_improvement = AsyncMock(return_value="imp-123")
    store.get_improvement = AsyncMock(return_value=None)
    store.get_active_improvements = AsyncMock(return_value=[])
    store.update_improvement_status = AsyncMock(return_value=True)
    store.save_quality_metric = AsyncMock(return_value=True)
    return store


@pytest.fixture
def improvement_config():
    """Create improvement configuration"""
    return ImprovementConfig(
        quality_drop_threshold=0.10,
        min_quality_score=0.30,
        update_after_interactions=10,
        improvement_check_interval=50,
        enable_auto_rollback=True,
        rollback_observation_period=20,
        weak_area_threshold=0.60
    )


@pytest.fixture
def ab_config():
    """Create A/B testing configuration"""
    return ABTestingConfig(
        treatment_split=0.5,
        min_samples_for_significance=30,
        p_value_threshold=0.05,
        min_effect_size=0.05,
        max_concurrent_experiments=3
    )


@pytest.fixture
def engine(mock_store, improvement_config, ab_config):
    """Create ImprovementEngine with mock store"""
    return ImprovementEngine(
        store=mock_store,
        config=improvement_config,
        ab_config=ab_config
    )


@pytest.fixture
def sample_improvement():
    """Create sample improvement"""
    return Improvement(
        improvement_id=str(uuid.uuid4()),
        dimension=ImprovementDimension.RESPONSE_LENGTH,
        hypothesis="Shorter responses improve engagement",
        change_description="Reduce average response length by 20%",
        target_value="short",
        baseline_value="medium",
        status=ImprovementStatus.PENDING
    )


@pytest.fixture
def sample_weak_area():
    """Create sample weak area"""
    return WeakArea(
        dimension=ImprovementDimension.RESPONSE_LENGTH,
        current_score=0.55,
        target_score=0.70,
        sample_size=100,
        evidence=["Average quality score: 0.55"],
        suggested_change="Adjust response length",
        confidence=0.8
    )


# =============================================================================
# Improvement Data Class Tests
# =============================================================================

class TestImprovement:
    """Tests for Improvement data class"""

    def test_improvement_delta_calculation(self, sample_improvement):
        """Test improvement delta calculation"""
        sample_improvement.control_quality = 0.5
        sample_improvement.treatment_quality = 0.6

        delta = sample_improvement.improvement_delta

        assert abs(delta - 0.2) < 0.001  # (0.6 - 0.5) / 0.5 = 0.2

    def test_improvement_delta_zero_control(self, sample_improvement):
        """Test delta with zero control quality"""
        sample_improvement.control_quality = 0.0
        sample_improvement.treatment_quality = 0.5

        delta = sample_improvement.improvement_delta

        assert delta == 0.0

    def test_to_record(self, sample_improvement):
        """Test conversion to storage record"""
        record = sample_improvement.to_record()

        assert isinstance(record, ImprovementRecord)
        assert record.improvement_id == sample_improvement.improvement_id
        assert record.hypothesis == sample_improvement.hypothesis
        assert record.target_dimension == sample_improvement.dimension.value

    def test_from_record(self):
        """Test creation from storage record"""
        record = ImprovementRecord(
            improvement_id="imp-123",
            hypothesis="Test hypothesis",
            target_dimension="response_length",
            change_description="Test change",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            control_samples=50,
            treatment_samples=50,
            control_quality=0.6,
            treatment_quality=0.7,
            metadata={"target_value": "short", "baseline_value": "medium"}
        )

        improvement = Improvement.from_record(record)

        assert improvement.improvement_id == "imp-123"
        assert improvement.status == ImprovementStatus.ACTIVE
        assert improvement.control_samples == 50


# =============================================================================
# Quality Scorer Tests
# =============================================================================

class TestQualityScorer:
    """Tests for QualityScorer"""

    def test_score_basic(self):
        """Test basic scoring"""
        scorer = QualityScorer()

        score, components = scorer.score(
            response_text="This is a helpful response about Python programming.",
            query_text="Tell me about Python"
        )

        assert 0 <= score <= 1
        assert "relevance" in components
        assert "coherence" in components
        assert "helpfulness" in components
        assert "style_match" in components

    def test_score_with_feedback(self):
        """Test scoring with feedback"""
        scorer = QualityScorer()

        score, components = scorer.score(
            response_text="Test response",
            query_text="Test query",
            feedback={"helpful": True, "rating": 4.5}
        )

        # Rating 4.5/5.0 = 0.9, or helpful=True might override to 1.0
        assert components["helpfulness"] >= 0.9

    def test_score_with_preferences(self):
        """Test scoring with preferences"""
        scorer = QualityScorer()

        score, components = scorer.score(
            response_text="Short response.",
            query_text="Test",
            preferences={"response_length": "short"}
        )

        assert "style_match" in components

    def test_relevance_scoring(self):
        """Test relevance scoring"""
        scorer = QualityScorer()

        # High relevance - query terms in response
        relevance = scorer._score_relevance(
            "Python is a programming language used for data science.",
            "Tell me about Python programming"
        )
        assert relevance > 0.3  # Should have some relevance due to "Python"

        # Low relevance - no overlap
        relevance = scorer._score_relevance(
            "The weather is nice today.",
            "Tell me about Python programming"
        )
        assert relevance < 0.5  # Should be low


# =============================================================================
# Weak Area Analysis Tests
# =============================================================================

class TestAnalyzeWeakAreas:
    """Tests for weak area analysis"""

    @pytest.mark.asyncio
    async def test_analyze_finds_weak_areas(self, engine, mock_store):
        """Test that weak areas are identified"""
        weak_areas = await engine.analyze_weak_areas()

        # Should identify areas below threshold
        assert isinstance(weak_areas, list)
        # May have weak areas based on mock data (avg_quality=0.65, helpful_rate=0.6)
        for area in weak_areas:
            assert isinstance(area, WeakArea)
            # Score should be at or below threshold (inclusive boundary)
            assert area.current_score <= engine.config.weak_area_threshold

    @pytest.mark.asyncio
    async def test_analyze_no_data(self, engine, mock_store):
        """Test analysis with no data"""
        mock_store.get_quality_stats.return_value = {"total_responses": 0}
        mock_store.get_feedback_stats.return_value = {"total_feedback": 0}

        weak_areas = await engine.analyze_weak_areas()

        assert isinstance(weak_areas, list)

    @pytest.mark.asyncio
    async def test_analyze_high_quality(self, engine, mock_store):
        """Test analysis when quality is high"""
        mock_store.get_quality_stats.return_value = {
            "total_responses": 100,
            "avg_quality": 0.9,
            "avg_relevance": 0.9,
            "avg_coherence": 0.9,
            "avg_helpfulness": 0.9
        }
        mock_store.get_feedback_stats.return_value = {
            "total_feedback": 50,
            "helpful_rate": 0.95
        }

        weak_areas = await engine.analyze_weak_areas()

        # Should have fewer or no weak areas
        assert all(area.current_score < engine.config.weak_area_threshold for area in weak_areas)


# =============================================================================
# Improvement Generation Tests
# =============================================================================

class TestGenerateImprovements:
    """Tests for improvement generation"""

    @pytest.mark.asyncio
    async def test_generate_improvements(self, engine, sample_weak_area):
        """Test improvement generation"""
        improvements = await engine.generate_improvements(
            weak_areas=[sample_weak_area],
            max_improvements=3
        )

        assert isinstance(improvements, list)
        for imp in improvements:
            assert isinstance(imp, Improvement)
            assert imp.hypothesis
            assert imp.status == ImprovementStatus.PENDING

    @pytest.mark.asyncio
    async def test_generate_auto_analyzes(self, engine):
        """Test that generation auto-analyzes if no weak areas provided"""
        improvements = await engine.generate_improvements()

        assert isinstance(improvements, list)

    @pytest.mark.asyncio
    async def test_generate_respects_limit(self, engine):
        """Test that generation respects max limit"""
        weak_areas = [
            WeakArea(
                dimension=ImprovementDimension.RESPONSE_LENGTH,
                current_score=0.5,
                target_score=0.7,
                sample_size=100,
                evidence=[],
                suggested_change="Test",
                confidence=0.8
            )
            for _ in range(10)
        ]

        improvements = await engine.generate_improvements(
            weak_areas=weak_areas,
            max_improvements=2
        )

        assert len(improvements) <= 2


# =============================================================================
# Apply Improvement Tests
# =============================================================================

class TestApplyImprovement:
    """Tests for improvement activation"""

    @pytest.mark.asyncio
    async def test_apply_improvement(self, engine, mock_store, sample_improvement):
        """Test applying an improvement"""
        mock_store.get_improvement.return_value = sample_improvement.to_record()

        success = await engine.apply_improvement(sample_improvement.improvement_id)

        assert success
        assert sample_improvement.improvement_id in engine._active_improvements
        mock_store.update_improvement_status.assert_called()

    @pytest.mark.asyncio
    async def test_apply_not_found(self, engine, mock_store):
        """Test applying non-existent improvement"""
        mock_store.get_improvement.return_value = None

        success = await engine.apply_improvement("non-existent")

        assert not success

    @pytest.mark.asyncio
    async def test_apply_max_concurrent(self, engine, mock_store, sample_improvement):
        """Test max concurrent experiments limit"""
        # Fill up active improvements
        for i in range(3):
            engine._active_improvements[f"imp-{i}"] = sample_improvement

        mock_store.get_improvement.return_value = sample_improvement.to_record()

        success = await engine.apply_improvement("new-improvement")

        assert not success  # Should fail due to limit


# =============================================================================
# Impact Measurement Tests
# =============================================================================

class TestMeasureImpact:
    """Tests for impact measurement"""

    @pytest.mark.asyncio
    async def test_measure_impact(self, engine, mock_store, sample_improvement):
        """Test measuring improvement impact"""
        sample_improvement.control_samples = 50
        sample_improvement.treatment_samples = 50
        sample_improvement.control_quality = 0.6
        sample_improvement.treatment_quality = 0.72

        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        result = await engine.measure_impact(sample_improvement.improvement_id)

        assert result is not None
        assert isinstance(result, ABTestResult)
        assert result.improvement_delta == 0.2  # (0.72 - 0.6) / 0.6
        assert result.recommendation

    @pytest.mark.asyncio
    async def test_measure_insufficient_samples(self, engine, mock_store, sample_improvement):
        """Test measurement with insufficient samples"""
        sample_improvement.control_samples = 5
        sample_improvement.treatment_samples = 5
        sample_improvement.control_quality = 0.6
        sample_improvement.treatment_quality = 0.7

        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        result = await engine.measure_impact(sample_improvement.improvement_id)

        assert result is not None
        assert not result.is_significant
        assert "insufficient" in result.recommendation.lower() or "continue" in result.recommendation.lower()


# =============================================================================
# A/B Testing Assignment Tests
# =============================================================================

class TestABAssignment:
    """Tests for A/B test assignment"""

    def test_assign_treatment_consistent(self, engine, sample_improvement):
        """Test that assignment is consistent for same session"""
        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        assignment1 = engine.assign_treatment("session-1", sample_improvement.improvement_id)
        assignment2 = engine.assign_treatment("session-1", sample_improvement.improvement_id)

        assert assignment1 == assignment2

    def test_assign_treatment_varies(self, engine, sample_improvement):
        """Test that different sessions may get different assignments"""
        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        assignments = [
            engine.assign_treatment(f"session-{i}", sample_improvement.improvement_id)
            for i in range(100)
        ]

        # Should have some variation (not all True or all False)
        assert 0 < sum(assignments) < 100


# =============================================================================
# Rollback Tests
# =============================================================================

class TestRollback:
    """Tests for improvement rollback"""

    @pytest.mark.asyncio
    async def test_rollback_improvement(self, engine, mock_store, sample_improvement):
        """Test rolling back an improvement"""
        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        success = await engine.rollback_improvement(
            sample_improvement.improvement_id,
            reason="test rollback"
        )

        assert success
        assert sample_improvement.improvement_id not in engine._active_improvements
        mock_store.update_improvement_status.assert_called()

    @pytest.mark.asyncio
    async def test_auto_rollback_on_quality_drop(self, engine, mock_store, sample_improvement):
        """Test automatic rollback on quality drop"""
        sample_improvement.control_samples = 25
        sample_improvement.treatment_samples = 25
        sample_improvement.control_quality = 0.7
        sample_improvement.treatment_quality = 0.5  # Significant drop

        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        await engine._check_auto_rollback(sample_improvement)

        # Should have rolled back
        assert sample_improvement.improvement_id not in engine._active_improvements


# =============================================================================
# Observation Recording Tests
# =============================================================================

class TestRecordObservation:
    """Tests for recording A/B test observations"""

    @pytest.mark.asyncio
    async def test_record_observation(self, engine, mock_store, sample_improvement):
        """Test recording an observation"""
        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        await engine.record_observation(
            session_id="session-1",
            improvement_id=sample_improvement.improvement_id,
            quality_score=0.8,
            response_metadata={"word_count": 50}
        )

        # Should update samples
        assert sample_improvement.control_samples + sample_improvement.treatment_samples == 1
        mock_store.save_quality_metric.assert_called()


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Tests for improvement statistics"""

    @pytest.mark.asyncio
    async def test_get_improvement_stats(self, engine, sample_improvement):
        """Test getting improvement statistics"""
        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        stats = await engine.get_improvement_stats()

        assert "active_improvements" in stats
        assert stats["active_improvements"] == 1
        assert "improvements" in stats

    def test_get_active_for_session(self, engine, sample_improvement):
        """Test getting active improvements for session"""
        engine._active_improvements[sample_improvement.improvement_id] = sample_improvement

        active = engine.get_active_improvements_for_session("session-1")

        assert len(active) == 1
        imp_id, is_treatment = active[0]
        assert imp_id == sample_improvement.improvement_id
        assert isinstance(is_treatment, bool)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactory:
    """Tests for factory function"""

    def test_create_improvement_engine(self, mock_store, improvement_config, ab_config):
        """Test factory function"""
        engine = create_improvement_engine(
            store=mock_store,
            config=improvement_config,
            ab_config=ab_config
        )

        assert isinstance(engine, ImprovementEngine)
        assert engine.store == mock_store
        assert engine.config == improvement_config

    def test_create_with_defaults(self):
        """Test factory with defaults"""
        engine = create_improvement_engine()

        assert isinstance(engine, ImprovementEngine)


# =============================================================================
# Normal CDF Tests
# =============================================================================

class TestNormalCDF:
    """Tests for normal CDF approximation"""

    def test_normal_cdf_zero(self, engine):
        """Test CDF at zero"""
        result = engine._normal_cdf(0)
        assert abs(result - 0.5) < 0.01

    def test_normal_cdf_positive(self, engine):
        """Test CDF for positive values"""
        result = engine._normal_cdf(1.96)
        assert abs(result - 0.975) < 0.01

    def test_normal_cdf_negative(self, engine):
        """Test CDF for negative values"""
        result = engine._normal_cdf(-1.96)
        assert abs(result - 0.025) < 0.01
