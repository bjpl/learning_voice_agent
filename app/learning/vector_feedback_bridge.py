"""
Vector Feedback Bridge - Phase 3 Learning Integration
======================================================

SPECIFICATION:
- Connect quality scoring to RuVector training
- Bridge feedback signals to vector index learning
- Implement batch training from historical data
- Track learning effectiveness metrics

ARCHITECTURE:
[Quality Score] -> [Feedback Bridge] -> [RuVector train_positive/train_negative]
                          |
                    [Learning Stats]

PATTERN: Bridge pattern connecting two subsystems
WHY: Decouple quality scoring from vector store implementation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Protocol

from app.learning.config import LearningConfig, learning_config

logger = logging.getLogger(__name__)


# Protocol for type safety without circular imports
class LearningVectorStoreProtocol(Protocol):
    """Protocol for vector stores that support learning."""

    async def train_positive(self, conversation_id: str, weight: float = 1.0) -> bool:
        """Train index to boost this conversation."""
        ...

    async def train_negative(self, conversation_id: str, weight: float = 1.0) -> bool:
        """Train index to reduce weight of this conversation."""
        ...

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        ...


class VectorFeedbackBridge:
    """
    Connects quality scoring to vector index training.

    CONCEPT: Feedback-driven index optimization
    WHY: Quality scores tell us what's good - use them to improve search

    Training Strategy:
    - score >= 0.8: Strong positive signal (weight=1.0)
    - score >= 0.6: Weak positive signal (weight=0.5)
    - score <= 0.3: Negative signal (weight=1.0)
    - 0.3 < score < 0.6: No training (neutral)

    Feedback Types:
    - quality: From quality scorer (automatic)
    - explicit: From user buttons (thumbs up/down)
    - implicit: From behavior (engagement, corrections)
    """

    def __init__(
        self,
        vector_store: LearningVectorStoreProtocol,
        config: Optional[LearningConfig] = None
    ):
        """
        Initialize the vector feedback bridge.

        Args:
            vector_store: Vector store with learning capabilities
            config: Learning configuration
        """
        self.vector_store = vector_store
        self.config = config or learning_config

        # Training statistics
        self._total_trained = 0
        self._positive_trained = 0
        self._negative_trained = 0
        self._skipped_neutral = 0
        self._failed_training = 0

        # Thresholds from config or defaults
        self._strong_positive_threshold = 0.8
        self._weak_positive_threshold = 0.6
        self._negative_threshold = 0.3

    async def process_feedback(
        self,
        conversation_id: str,
        quality_score: float,
        feedback_type: str = "quality"
    ) -> bool:
        """
        Process feedback and train the vector index.

        ALGORITHM:
        1. Normalize quality score to 0-1 range
        2. Determine training signal from score
        3. Calculate training weight
        4. Apply training to vector store
        5. Update statistics

        Args:
            conversation_id: Conversation identifier
            quality_score: Quality score (0-1 from quality_scorer.py)
            feedback_type: Type of feedback (quality, explicit, implicit)

        Returns:
            True if training was applied, False if skipped or failed
        """
        try:
            # Validate score range
            if not 0 <= quality_score <= 1:
                logger.warning(
                    f"Invalid quality score {quality_score} for {conversation_id}, "
                    f"clamping to [0, 1]"
                )
                quality_score = max(0.0, min(1.0, quality_score))

            # Determine training action
            if quality_score >= self._strong_positive_threshold:
                # Strong positive signal
                weight = 1.0
                success = await self.vector_store.train_positive(
                    conversation_id, weight
                )
                if success:
                    self._positive_trained += 1
                    self._total_trained += 1
                    logger.debug(
                        f"Strong positive training: {conversation_id} "
                        f"(score={quality_score:.3f}, weight={weight})"
                    )
                else:
                    self._failed_training += 1
                return success

            elif quality_score >= self._weak_positive_threshold:
                # Weak positive signal
                weight = 0.5
                success = await self.vector_store.train_positive(
                    conversation_id, weight
                )
                if success:
                    self._positive_trained += 1
                    self._total_trained += 1
                    logger.debug(
                        f"Weak positive training: {conversation_id} "
                        f"(score={quality_score:.3f}, weight={weight})"
                    )
                else:
                    self._failed_training += 1
                return success

            elif quality_score <= self._negative_threshold:
                # Negative signal
                weight = 1.0
                success = await self.vector_store.train_negative(
                    conversation_id, weight
                )
                if success:
                    self._negative_trained += 1
                    self._total_trained += 1
                    logger.debug(
                        f"Negative training: {conversation_id} "
                        f"(score={quality_score:.3f}, weight={weight})"
                    )
                else:
                    self._failed_training += 1
                return success

            else:
                # Neutral score - skip training
                self._skipped_neutral += 1
                logger.debug(
                    f"Skipped neutral score: {conversation_id} "
                    f"(score={quality_score:.3f})"
                )
                return False

        except Exception as e:
            logger.error(
                f"Failed to process feedback for {conversation_id}: {e}"
            )
            self._failed_training += 1
            return False

    async def batch_train_from_history(
        self,
        min_quality: float = 0.7,
        limit: int = 100,
        feedback_store: Optional[Any] = None
    ) -> Dict[str, int]:
        """
        Train index from historical quality data.

        CONCEPT: Batch learning from past interactions
        WHY: Bootstrap learning when deploying to existing system

        ALGORITHM:
        1. Query feedback store for high-quality conversations
        2. Batch process feedback signals
        3. Apply training in parallel (up to max_concurrent)
        4. Return statistics

        Args:
            min_quality: Minimum quality score to include
            limit: Maximum number of conversations to train
            feedback_store: Optional feedback store to query

        Returns:
            Dictionary with training statistics:
            - processed: Total conversations processed
            - trained_positive: Successfully trained positive
            - trained_negative: Successfully trained negative
            - skipped: Skipped (neutral scores)
            - failed: Failed to train
        """
        stats = {
            "processed": 0,
            "trained_positive": 0,
            "trained_negative": 0,
            "skipped": 0,
            "failed": 0
        }

        if feedback_store is None:
            logger.warning("No feedback store provided for batch training")
            return stats

        try:
            # Query feedback store for recent high-quality conversations
            # This is a placeholder - actual implementation depends on feedback_store API
            logger.info(
                f"Starting batch training: min_quality={min_quality}, limit={limit}"
            )

            # Get conversations with quality scores
            if hasattr(feedback_store, 'query_by_quality'):
                conversations = await feedback_store.query_by_quality(
                    min_score=min_quality,
                    limit=limit
                )
            elif hasattr(feedback_store, 'query'):
                # Fallback to general query
                conversations = await feedback_store.query(limit=limit)
            else:
                logger.error("Feedback store does not support querying")
                return stats

            # Process conversations in batches
            max_concurrent = 5
            semaphore = asyncio.Semaphore(max_concurrent)

            async def train_one(conv: Dict[str, Any]) -> bool:
                async with semaphore:
                    conversation_id = conv.get('conversation_id') or conv.get('id')
                    quality_score = conv.get('quality_score') or conv.get('rating', 0.5)

                    if conversation_id:
                        return await self.process_feedback(
                            conversation_id,
                            quality_score,
                            feedback_type="historical"
                        )
                    return False

            # Process all conversations
            initial_positive = self._positive_trained
            initial_negative = self._negative_trained
            initial_skipped = self._skipped_neutral
            initial_failed = self._failed_training

            results = await asyncio.gather(
                *[train_one(conv) for conv in conversations],
                return_exceptions=True
            )

            # Calculate statistics
            stats["processed"] = len(conversations)
            stats["trained_positive"] = self._positive_trained - initial_positive
            stats["trained_negative"] = self._negative_trained - initial_negative
            stats["skipped"] = self._skipped_neutral - initial_skipped
            stats["failed"] = self._failed_training - initial_failed

            logger.info(
                f"Batch training completed: {stats['processed']} processed, "
                f"{stats['trained_positive']} positive, "
                f"{stats['trained_negative']} negative, "
                f"{stats['skipped']} skipped, "
                f"{stats['failed']} failed"
            )

            return stats

        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            return stats

    async def process_explicit_feedback(
        self,
        conversation_id: str,
        is_positive: bool,
        weight: float = 1.0
    ) -> bool:
        """
        Process explicit user feedback (thumbs up/down).

        CONCEPT: Direct user signals override quality scores
        WHY: User knows better than algorithms

        Args:
            conversation_id: Conversation identifier
            is_positive: True for thumbs up, False for thumbs down
            weight: Training weight (default 1.0 for explicit feedback)

        Returns:
            True if training was applied successfully
        """
        try:
            if is_positive:
                success = await self.vector_store.train_positive(
                    conversation_id, weight
                )
                if success:
                    self._positive_trained += 1
                    self._total_trained += 1
                    logger.info(
                        f"Explicit positive feedback: {conversation_id} "
                        f"(weight={weight})"
                    )
                else:
                    self._failed_training += 1
                return success
            else:
                success = await self.vector_store.train_negative(
                    conversation_id, weight
                )
                if success:
                    self._negative_trained += 1
                    self._total_trained += 1
                    logger.info(
                        f"Explicit negative feedback: {conversation_id} "
                        f"(weight={weight})"
                    )
                else:
                    self._failed_training += 1
                return success

        except Exception as e:
            logger.error(
                f"Failed to process explicit feedback for {conversation_id}: {e}"
            )
            self._failed_training += 1
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics for monitoring.

        Returns:
            Dictionary with training statistics:
            - total_trained: Total training operations
            - positive_trained: Positive training count
            - negative_trained: Negative training count
            - skipped_neutral: Neutral scores skipped
            - failed_training: Failed training attempts
            - success_rate: Training success rate
            - positive_ratio: Ratio of positive to total
        """
        total_attempted = (
            self._positive_trained +
            self._negative_trained +
            self._failed_training
        )

        success_rate = (
            self._total_trained / total_attempted
            if total_attempted > 0 else 0.0
        )

        positive_ratio = (
            self._positive_trained / self._total_trained
            if self._total_trained > 0 else 0.0
        )

        return {
            "total_trained": self._total_trained,
            "positive_trained": self._positive_trained,
            "negative_trained": self._negative_trained,
            "skipped_neutral": self._skipped_neutral,
            "failed_training": self._failed_training,
            "success_rate": round(success_rate, 3),
            "positive_ratio": round(positive_ratio, 3)
        }

    async def get_combined_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics from bridge and vector store.

        Returns:
            Dictionary with both bridge and store statistics
        """
        bridge_stats = self.get_training_stats()

        try:
            store_stats = await self.vector_store.get_learning_stats()
            return {
                "bridge": bridge_stats,
                "vector_store": store_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {
                "bridge": bridge_stats,
                "vector_store": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }

    def reset_stats(self) -> None:
        """Reset training statistics (for testing)."""
        self._total_trained = 0
        self._positive_trained = 0
        self._negative_trained = 0
        self._skipped_neutral = 0
        self._failed_training = 0


# Convenience factory function
def create_vector_feedback_bridge(
    vector_store: LearningVectorStoreProtocol,
    config: Optional[LearningConfig] = None
) -> VectorFeedbackBridge:
    """
    Factory function for creating a VectorFeedbackBridge.

    Args:
        vector_store: Vector store with learning capabilities
        config: Optional learning configuration

    Returns:
        Configured VectorFeedbackBridge instance
    """
    return VectorFeedbackBridge(vector_store, config)
