"""
Unit Tests for Vector Feedback Bridge - Phase 3 Learning Integration
======================================================================

Test Coverage:
- Feedback processing with quality scores
- Explicit feedback handling (thumbs up/down)
- Batch training from historical data
- Statistics tracking
- Error handling and edge cases
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch

from app.learning.vector_feedback_bridge import (
    VectorFeedbackBridge,
    create_vector_feedback_bridge
)
from app.learning.config import LearningConfig


# ============================================================================
# Test Fixtures
# ============================================================================

class MockLearningVectorStore:
    """Mock vector store with learning capabilities."""

    def __init__(self):
        self.positive_training_calls = []
        self.negative_training_calls = []
        self.train_positive_return = True
        self.train_negative_return = True

    async def train_positive(self, conversation_id: str, weight: float = 1.0) -> bool:
        self.positive_training_calls.append({
            "conversation_id": conversation_id,
            "weight": weight,
            "timestamp": datetime.utcnow()
        })
        return self.train_positive_return

    async def train_negative(self, conversation_id: str, weight: float = 1.0) -> bool:
        self.negative_training_calls.append({
            "conversation_id": conversation_id,
            "weight": weight,
            "timestamp": datetime.utcnow()
        })
        return self.train_negative_return

    async def get_learning_stats(self) -> Dict[str, Any]:
        return {
            "learning_enabled": True,
            "positive_training_count": len(self.positive_training_calls),
            "negative_training_count": len(self.negative_training_calls),
            "total_training_signals": len(self.positive_training_calls) + len(self.negative_training_calls)
        }


class MockFeedbackStore:
    """Mock feedback store for batch training tests."""

    def __init__(self):
        self.conversations = []

    async def query_by_quality(
        self,
        min_score: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query conversations by quality score."""
        filtered = [
            conv for conv in self.conversations
            if conv.get("quality_score", 0) >= min_score
        ]
        return filtered[:limit]

    async def query(self, limit: int = 100) -> List[Dict[str, Any]]:
        """General query."""
        return self.conversations[:limit]

    def add_conversation(
        self,
        conversation_id: str,
        quality_score: float
    ):
        """Add a mock conversation."""
        self.conversations.append({
            "conversation_id": conversation_id,
            "id": conversation_id,
            "quality_score": quality_score,
            "rating": quality_score
        })


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    return MockLearningVectorStore()


@pytest.fixture
def mock_feedback_store():
    """Create mock feedback store."""
    return MockFeedbackStore()


@pytest.fixture
def config():
    """Create test configuration."""
    return LearningConfig()


@pytest.fixture
def bridge(mock_vector_store, config):
    """Create vector feedback bridge with mock store."""
    return VectorFeedbackBridge(mock_vector_store, config)


# ============================================================================
# Test Suite: Quality Score Feedback Processing
# ============================================================================

class TestQualityScoreFeedback:
    """Test processing of quality score feedback."""

    @pytest.mark.asyncio
    async def test_strong_positive_feedback(self, bridge, mock_vector_store):
        """Test strong positive quality score triggers positive training."""
        # Arrange
        conversation_id = "conv_001"
        quality_score = 0.9  # Strong positive (>= 0.8)

        # Act
        result = await bridge.process_feedback(conversation_id, quality_score)

        # Assert
        assert result is True
        assert len(mock_vector_store.positive_training_calls) == 1
        assert len(mock_vector_store.negative_training_calls) == 0

        call = mock_vector_store.positive_training_calls[0]
        assert call["conversation_id"] == conversation_id
        assert call["weight"] == 1.0

        stats = bridge.get_training_stats()
        assert stats["positive_trained"] == 1
        assert stats["negative_trained"] == 0
        assert stats["total_trained"] == 1

    @pytest.mark.asyncio
    async def test_weak_positive_feedback(self, bridge, mock_vector_store):
        """Test weak positive quality score triggers positive training with reduced weight."""
        # Arrange
        conversation_id = "conv_002"
        quality_score = 0.7  # Weak positive (0.6 <= score < 0.8)

        # Act
        result = await bridge.process_feedback(conversation_id, quality_score)

        # Assert
        assert result is True
        assert len(mock_vector_store.positive_training_calls) == 1

        call = mock_vector_store.positive_training_calls[0]
        assert call["conversation_id"] == conversation_id
        assert call["weight"] == 0.5  # Reduced weight for weak signal

        stats = bridge.get_training_stats()
        assert stats["positive_trained"] == 1

    @pytest.mark.asyncio
    async def test_negative_feedback(self, bridge, mock_vector_store):
        """Test negative quality score triggers negative training."""
        # Arrange
        conversation_id = "conv_003"
        quality_score = 0.2  # Negative (<= 0.3)

        # Act
        result = await bridge.process_feedback(conversation_id, quality_score)

        # Assert
        assert result is True
        assert len(mock_vector_store.negative_training_calls) == 1
        assert len(mock_vector_store.positive_training_calls) == 0

        call = mock_vector_store.negative_training_calls[0]
        assert call["conversation_id"] == conversation_id
        assert call["weight"] == 1.0

        stats = bridge.get_training_stats()
        assert stats["negative_trained"] == 1
        assert stats["positive_trained"] == 0

    @pytest.mark.asyncio
    async def test_neutral_feedback_skipped(self, bridge, mock_vector_store):
        """Test neutral quality scores are skipped."""
        # Arrange
        conversation_id = "conv_004"
        quality_score = 0.5  # Neutral (0.3 < score < 0.6)

        # Act
        result = await bridge.process_feedback(conversation_id, quality_score)

        # Assert
        assert result is False
        assert len(mock_vector_store.positive_training_calls) == 0
        assert len(mock_vector_store.negative_training_calls) == 0

        stats = bridge.get_training_stats()
        assert stats["skipped_neutral"] == 1
        assert stats["total_trained"] == 0

    @pytest.mark.asyncio
    async def test_multiple_feedback_accumulation(self, bridge, mock_vector_store):
        """Test multiple feedback signals accumulate correctly."""
        # Arrange & Act
        await bridge.process_feedback("conv_1", 0.9)  # Strong positive
        await bridge.process_feedback("conv_2", 0.7)  # Weak positive
        await bridge.process_feedback("conv_3", 0.2)  # Negative
        await bridge.process_feedback("conv_4", 0.5)  # Neutral (skipped)

        # Assert
        stats = bridge.get_training_stats()
        assert stats["positive_trained"] == 2
        assert stats["negative_trained"] == 1
        assert stats["skipped_neutral"] == 1
        assert stats["total_trained"] == 3


# ============================================================================
# Test Suite: Explicit Feedback
# ============================================================================

class TestExplicitFeedback:
    """Test explicit user feedback (thumbs up/down)."""

    @pytest.mark.asyncio
    async def test_explicit_thumbs_up(self, bridge, mock_vector_store):
        """Test explicit thumbs up triggers positive training."""
        # Arrange
        conversation_id = "conv_005"

        # Act
        result = await bridge.process_explicit_feedback(
            conversation_id,
            is_positive=True,
            weight=1.0
        )

        # Assert
        assert result is True
        assert len(mock_vector_store.positive_training_calls) == 1

        call = mock_vector_store.positive_training_calls[0]
        assert call["conversation_id"] == conversation_id
        assert call["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_explicit_thumbs_down(self, bridge, mock_vector_store):
        """Test explicit thumbs down triggers negative training."""
        # Arrange
        conversation_id = "conv_006"

        # Act
        result = await bridge.process_explicit_feedback(
            conversation_id,
            is_positive=False,
            weight=1.0
        )

        # Assert
        assert result is True
        assert len(mock_vector_store.negative_training_calls) == 1

        call = mock_vector_store.negative_training_calls[0]
        assert call["conversation_id"] == conversation_id
        assert call["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_explicit_feedback_custom_weight(self, bridge, mock_vector_store):
        """Test explicit feedback with custom weight."""
        # Arrange
        conversation_id = "conv_007"
        custom_weight = 0.75

        # Act
        await bridge.process_explicit_feedback(
            conversation_id,
            is_positive=True,
            weight=custom_weight
        )

        # Assert
        call = mock_vector_store.positive_training_calls[0]
        assert call["weight"] == custom_weight


# ============================================================================
# Test Suite: Batch Training
# ============================================================================

class TestBatchTraining:
    """Test batch training from historical data."""

    @pytest.mark.asyncio
    async def test_batch_train_from_history(
        self,
        bridge,
        mock_vector_store,
        mock_feedback_store
    ):
        """Test batch training processes multiple conversations."""
        # Arrange
        mock_feedback_store.add_conversation("conv_1", 0.9)
        mock_feedback_store.add_conversation("conv_2", 0.8)
        mock_feedback_store.add_conversation("conv_3", 0.7)
        mock_feedback_store.add_conversation("conv_4", 0.2)
        mock_feedback_store.add_conversation("conv_5", 0.5)

        # Act
        stats = await bridge.batch_train_from_history(
            min_quality=0.7,
            limit=10,
            feedback_store=mock_feedback_store
        )

        # Assert
        # Note: min_quality=0.7 filters to only conv_1, conv_2, conv_3
        assert stats["processed"] == 3
        assert stats["trained_positive"] == 3  # conv_1, conv_2, conv_3 (all >= 0.7)
        assert stats["trained_negative"] == 0  # None below 0.3
        assert stats["skipped"] == 0  # None in neutral range

    @pytest.mark.asyncio
    async def test_batch_train_with_limit(
        self,
        bridge,
        mock_vector_store,
        mock_feedback_store
    ):
        """Test batch training respects limit."""
        # Arrange
        for i in range(20):
            mock_feedback_store.add_conversation(f"conv_{i}", 0.9)

        # Act
        stats = await bridge.batch_train_from_history(
            min_quality=0.8,
            limit=5,
            feedback_store=mock_feedback_store
        )

        # Assert
        assert stats["processed"] == 5  # Limited to 5
        assert stats["trained_positive"] == 5

    @pytest.mark.asyncio
    async def test_batch_train_no_feedback_store(self, bridge):
        """Test batch training handles missing feedback store."""
        # Act
        stats = await bridge.batch_train_from_history(
            feedback_store=None
        )

        # Assert
        assert stats["processed"] == 0
        assert stats["trained_positive"] == 0

    @pytest.mark.asyncio
    async def test_batch_train_empty_store(
        self,
        bridge,
        mock_feedback_store
    ):
        """Test batch training with empty feedback store."""
        # Act
        stats = await bridge.batch_train_from_history(
            feedback_store=mock_feedback_store
        )

        # Assert
        assert stats["processed"] == 0


# ============================================================================
# Test Suite: Statistics and Monitoring
# ============================================================================

class TestStatistics:
    """Test statistics tracking and reporting."""

    @pytest.mark.asyncio
    async def test_get_training_stats(self, bridge):
        """Test getting training statistics."""
        # Arrange - simulate some training
        await bridge.process_feedback("conv_1", 0.9)
        await bridge.process_feedback("conv_2", 0.8)
        await bridge.process_feedback("conv_3", 0.2)
        await bridge.process_feedback("conv_4", 0.5)

        # Act
        stats = bridge.get_training_stats()

        # Assert
        assert stats["total_trained"] == 3
        assert stats["positive_trained"] == 2
        assert stats["negative_trained"] == 1
        assert stats["skipped_neutral"] == 1
        assert stats["failed_training"] == 0
        assert stats["success_rate"] == 1.0  # All succeeded
        assert 0.6 < stats["positive_ratio"] < 0.7  # 2/3 ≈ 0.67

    @pytest.mark.asyncio
    async def test_get_combined_stats(self, bridge, mock_vector_store):
        """Test getting combined bridge and store statistics."""
        # Arrange
        await bridge.process_feedback("conv_1", 0.9)

        # Act
        combined = await bridge.get_combined_stats()

        # Assert
        assert "bridge" in combined
        assert "vector_store" in combined
        assert "timestamp" in combined

        assert combined["bridge"]["total_trained"] == 1
        assert combined["vector_store"]["learning_enabled"] is True

    def test_reset_stats(self, bridge):
        """Test resetting statistics."""
        # Arrange - create some stats
        bridge._positive_trained = 5
        bridge._negative_trained = 3
        bridge._total_trained = 8

        # Act
        bridge.reset_stats()

        # Assert
        stats = bridge.get_training_stats()
        assert stats["total_trained"] == 0
        assert stats["positive_trained"] == 0
        assert stats["negative_trained"] == 0


# ============================================================================
# Test Suite: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_quality_score_clamped(self, bridge, mock_vector_store):
        """Test invalid quality scores are clamped to valid range."""
        # Act - score > 1.0
        result1 = await bridge.process_feedback("conv_1", 1.5)
        # Act - score < 0.0
        result2 = await bridge.process_feedback("conv_2", -0.5)

        # Assert - both should be treated as valid after clamping
        assert result1 is True  # Clamped to 1.0 (strong positive)
        assert result2 is True  # Clamped to 0.0 (negative)

        assert len(mock_vector_store.positive_training_calls) == 1
        assert len(mock_vector_store.negative_training_calls) == 1

    @pytest.mark.asyncio
    async def test_vector_store_training_failure(self, bridge, mock_vector_store):
        """Test handling of vector store training failure."""
        # Arrange
        mock_vector_store.train_positive_return = False

        # Act
        result = await bridge.process_feedback("conv_1", 0.9)

        # Assert
        assert result is False
        stats = bridge.get_training_stats()
        assert stats["failed_training"] == 1
        assert stats["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_vector_store_exception(self, bridge, mock_vector_store):
        """Test handling of vector store exceptions."""
        # Arrange
        async def failing_train(*args, **kwargs):
            raise RuntimeError("Vector store error")

        mock_vector_store.train_positive = failing_train

        # Act
        result = await bridge.process_feedback("conv_1", 0.9)

        # Assert
        assert result is False
        stats = bridge.get_training_stats()
        assert stats["failed_training"] == 1

    @pytest.mark.asyncio
    async def test_explicit_feedback_exception(self, bridge, mock_vector_store):
        """Test handling exceptions in explicit feedback."""
        # Arrange
        async def failing_train(*args, **kwargs):
            raise ValueError("Training failed")

        mock_vector_store.train_positive = failing_train

        # Act
        result = await bridge.process_explicit_feedback("conv_1", True)

        # Assert
        assert result is False
        stats = bridge.get_training_stats()
        assert stats["failed_training"] == 1


# ============================================================================
# Test Suite: Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory and convenience functions."""

    def test_create_vector_feedback_bridge(self, mock_vector_store, config):
        """Test factory function creates bridge correctly."""
        # Act
        bridge = create_vector_feedback_bridge(mock_vector_store, config)

        # Assert
        assert isinstance(bridge, VectorFeedbackBridge)
        assert bridge.vector_store == mock_vector_store
        assert bridge.config == config

    def test_create_bridge_default_config(self, mock_vector_store):
        """Test factory function with default config."""
        # Act
        bridge = create_vector_feedback_bridge(mock_vector_store)

        # Assert
        assert isinstance(bridge, VectorFeedbackBridge)
        assert bridge.config is not None


# ============================================================================
# Test Suite: Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_feedback_lifecycle(
        self,
        bridge,
        mock_vector_store,
        mock_feedback_store
    ):
        """Test complete feedback to training lifecycle."""
        # Arrange - add historical data
        for i in range(5):
            mock_feedback_store.add_conversation(f"hist_{i}", 0.8)

        # Act - batch train from history
        batch_stats = await bridge.batch_train_from_history(
            feedback_store=mock_feedback_store
        )

        # Act - process new feedback
        await bridge.process_feedback("new_1", 0.95)
        await bridge.process_explicit_feedback("new_2", True)
        await bridge.process_explicit_feedback("new_3", False)

        # Assert - check total training
        stats = bridge.get_training_stats()
        assert stats["total_trained"] == 8  # 5 historical + 3 new
        assert stats["positive_trained"] == 7
        assert stats["negative_trained"] == 1

    @pytest.mark.asyncio
    async def test_parallel_feedback_processing(self, bridge, mock_vector_store):
        """Test processing multiple feedback signals in parallel."""
        # Arrange
        feedback_items = [
            ("conv_1", 0.9),
            ("conv_2", 0.8),
            ("conv_3", 0.2),
            ("conv_4", 0.7),
            ("conv_5", 0.1)
        ]

        # Act - process in parallel
        results = await asyncio.gather(*[
            bridge.process_feedback(conv_id, score)
            for conv_id, score in feedback_items
        ])

        # Assert
        assert all(results)  # All should succeed
        stats = bridge.get_training_stats()
        assert stats["total_trained"] == 5

    @pytest.mark.asyncio
    async def test_mixed_feedback_types(self, bridge, mock_vector_store):
        """Test mixing quality scores and explicit feedback."""
        # Act
        await bridge.process_feedback("conv_1", 0.85)  # Quality score
        await bridge.process_explicit_feedback("conv_2", True)  # Explicit positive
        await bridge.process_feedback("conv_3", 0.25)  # Quality score (negative)
        await bridge.process_explicit_feedback("conv_4", False)  # Explicit negative

        # Assert
        stats = bridge.get_training_stats()
        assert stats["total_trained"] == 4
        assert stats["positive_trained"] == 2
        assert stats["negative_trained"] == 2

    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, bridge, mock_vector_store):
        """Test success rate calculation with failures."""
        # Arrange - set up some failures
        mock_vector_store.train_positive_return = False

        # Act - mix successes and failures
        await bridge.process_feedback("conv_1", 0.9)  # Will fail
        mock_vector_store.train_positive_return = True
        await bridge.process_feedback("conv_2", 0.9)  # Will succeed
        await bridge.process_feedback("conv_3", 0.9)  # Will succeed

        # Assert
        stats = bridge.get_training_stats()
        assert stats["total_trained"] == 2
        assert stats["failed_training"] == 1
        assert abs(stats["success_rate"] - 0.667) < 0.01  # 2/3


# ============================================================================
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
