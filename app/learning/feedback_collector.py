"""
Feedback Collector
==================

Collects explicit and implicit feedback from user interactions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from difflib import SequenceMatcher

from app.learning.config import LearningConfig, learning_config
from app.learning.models import (
    Feedback,
    FeedbackType,
    FeedbackSource,
    FeedbackAggregation
)
from app.learning.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects and processes user feedback from multiple sources.

    Supports:
    - Explicit feedback (ratings, thumbs up/down, text comments)
    - Implicit feedback (corrections, engagement time, follow-ups)
    - Automatic aggregation and analysis
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback_store: Optional[FeedbackStore] = None
    ):
        """
        Initialize the feedback collector.

        Args:
            config: Learning configuration
            feedback_store: Storage backend for feedback
        """
        self.config = config or learning_config
        self.feedback_store = feedback_store or FeedbackStore(self.config)

        # In-memory tracking for implicit feedback
        self._interaction_cache: Dict[str, Dict[str, Any]] = {}
        self._correction_buffer: Dict[str, List[str]] = {}

    async def initialize(self) -> None:
        """Initialize the feedback collector and storage."""
        await self.feedback_store.initialize()
        logger.info("FeedbackCollector initialized")

    async def close(self) -> None:
        """Close the feedback collector and flush pending data."""
        await self.feedback_store.close()
        logger.info("FeedbackCollector closed")

    # ==================== Explicit Feedback ====================

    async def collect_rating(
        self,
        session_id: str,
        query_id: str,
        rating: float,
        original_query: str,
        original_response: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """
        Collect an explicit rating from the user.

        Args:
            session_id: Session identifier
            query_id: Query identifier
            rating: Rating value (0.0 to 1.0)
            original_query: The user's original query
            original_response: The system's response
            user_id: Optional user identifier
            metadata: Additional metadata

        Returns:
            The stored feedback object
        """
        feedback = Feedback(
            session_id=session_id,
            query_id=query_id,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            source=FeedbackSource.USER_BUTTON,
            rating=rating,
            original_query=original_query,
            original_response=original_response,
            user_id=user_id,
            metadata=metadata or {}
        )

        await self.feedback_store.store(feedback)
        logger.debug(f"Collected rating feedback: {rating} for query {query_id}")

        return feedback

    async def collect_thumbs_up(
        self,
        session_id: str,
        query_id: str,
        original_query: str,
        original_response: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Collect positive thumbs up feedback."""
        feedback = Feedback(
            session_id=session_id,
            query_id=query_id,
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            source=FeedbackSource.USER_BUTTON,
            rating=1.0,
            original_query=original_query,
            original_response=original_response,
            user_id=user_id,
            metadata=metadata or {}
        )

        await self.feedback_store.store(feedback)
        logger.debug(f"Collected thumbs up for query {query_id}")

        return feedback

    async def collect_thumbs_down(
        self,
        session_id: str,
        query_id: str,
        original_query: str,
        original_response: str,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Collect negative thumbs down feedback."""
        feedback = Feedback(
            session_id=session_id,
            query_id=query_id,
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            source=FeedbackSource.USER_BUTTON,
            rating=0.0,
            text=reason,
            original_query=original_query,
            original_response=original_response,
            user_id=user_id,
            metadata=metadata or {}
        )

        await self.feedback_store.store(feedback)
        logger.debug(f"Collected thumbs down for query {query_id}")

        return feedback

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
    ) -> Feedback:
        """Collect text-based feedback from the user."""
        # Determine feedback type from sentiment if not specified
        if is_positive is None:
            feedback_type = FeedbackType.EXPLICIT_RATING
            rating = 0.5  # Neutral
        elif is_positive:
            feedback_type = FeedbackType.EXPLICIT_POSITIVE
            rating = 0.8
        else:
            feedback_type = FeedbackType.EXPLICIT_NEGATIVE
            rating = 0.2

        feedback = Feedback(
            session_id=session_id,
            query_id=query_id,
            feedback_type=feedback_type,
            source=FeedbackSource.USER_TEXT,
            rating=rating,
            text=feedback_text,
            original_query=original_query,
            original_response=original_response,
            user_id=user_id,
            metadata=metadata or {}
        )

        await self.feedback_store.store(feedback)
        logger.debug(f"Collected text feedback for query {query_id}")

        return feedback

    # ==================== Implicit Feedback ====================

    async def track_interaction_start(
        self,
        session_id: str,
        query_id: str,
        query: str
    ) -> None:
        """
        Start tracking an interaction for implicit feedback.

        Args:
            session_id: Session identifier
            query_id: Query identifier
            query: The user's query
        """
        if not self.config.feedback.implicit_feedback_enabled:
            return

        cache_key = f"{session_id}:{query_id}"
        self._interaction_cache[cache_key] = {
            "session_id": session_id,
            "query_id": query_id,
            "query": query,
            "start_time": datetime.utcnow(),
            "response": None,
            "engagement_events": []
        }

    async def track_response_delivered(
        self,
        session_id: str,
        query_id: str,
        response: str
    ) -> None:
        """Track when a response is delivered to the user."""
        cache_key = f"{session_id}:{query_id}"
        if cache_key in self._interaction_cache:
            self._interaction_cache[cache_key]["response"] = response
            self._interaction_cache[cache_key]["response_time"] = datetime.utcnow()

    async def track_engagement(
        self,
        session_id: str,
        query_id: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track engagement events (scroll, copy, link click, etc.)

        Args:
            session_id: Session identifier
            query_id: Query identifier
            event_type: Type of engagement event
            event_data: Additional event data
        """
        if not self.config.feedback.engagement_tracking_enabled:
            return

        cache_key = f"{session_id}:{query_id}"
        if cache_key in self._interaction_cache:
            self._interaction_cache[cache_key]["engagement_events"].append({
                "type": event_type,
                "data": event_data or {},
                "timestamp": datetime.utcnow()
            })

    async def detect_correction(
        self,
        session_id: str,
        previous_query_id: str,
        new_query: str,
        previous_query: str,
        previous_response: str
    ) -> Optional[Feedback]:
        """
        Detect if a new query is a correction of a previous response.

        Args:
            session_id: Session identifier
            previous_query_id: ID of the previous query
            new_query: The new query from the user
            previous_query: The previous query
            previous_response: The previous response

        Returns:
            Feedback object if correction detected, None otherwise
        """
        if not self.config.feedback.correction_detection_enabled:
            return None

        # Check for correction patterns
        is_correction, correction_type = self._analyze_correction(
            new_query, previous_query, previous_response
        )

        if is_correction:
            feedback = Feedback(
                session_id=session_id,
                query_id=previous_query_id,
                feedback_type=FeedbackType.IMPLICIT_CORRECTION,
                source=FeedbackSource.SYSTEM_DETECTION,
                rating=0.3,  # Corrections indicate lower satisfaction
                correction=new_query,
                original_query=previous_query,
                original_response=previous_response,
                metadata={"correction_type": correction_type}
            )

            await self.feedback_store.store(feedback)
            logger.debug(f"Detected correction for query {previous_query_id}")

            return feedback

        return None

    def _analyze_correction(
        self,
        new_query: str,
        previous_query: str,
        previous_response: str
    ) -> tuple[bool, Optional[str]]:
        """
        Analyze if a query is a correction of a previous interaction.

        Returns:
            Tuple of (is_correction, correction_type)
        """
        new_lower = new_query.lower().strip()
        prev_lower = previous_query.lower().strip()

        # Check for explicit correction phrases
        correction_phrases = [
            "no, i meant", "that's not what i asked", "actually i wanted",
            "no that's wrong", "incorrect", "not quite", "let me rephrase",
            "what i really meant", "you misunderstood"
        ]

        for phrase in correction_phrases:
            if phrase in new_lower:
                return True, "explicit_correction"

        # Check for high similarity with the previous query (rephrasing)
        similarity = SequenceMatcher(None, new_lower, prev_lower).ratio()
        if similarity > self.config.feedback.correction_similarity_threshold:
            return True, "rephrasing"

        # Check if new query directly contradicts the response
        if self._contradicts_response(new_query, previous_response):
            return True, "contradiction"

        return False, None

    def _contradicts_response(self, query: str, response: str) -> bool:
        """Check if a query contradicts the response."""
        query_lower = query.lower()

        # Simple negation patterns
        negation_starters = [
            "no,", "that's wrong", "that's incorrect", "not really",
            "actually", "but i said", "i didn't ask"
        ]

        return any(query_lower.startswith(phrase) for phrase in negation_starters)

    async def track_follow_up(
        self,
        session_id: str,
        previous_query_id: str,
        follow_up_query: str,
        previous_query: str,
        previous_response: str
    ) -> Feedback:
        """
        Track when a user asks a follow-up question (positive signal).

        A follow-up indicates engagement and interest in the topic.
        """
        feedback = Feedback(
            session_id=session_id,
            query_id=previous_query_id,
            feedback_type=FeedbackType.IMPLICIT_FOLLOW_UP,
            source=FeedbackSource.SYSTEM_DETECTION,
            rating=0.7,  # Follow-ups are generally positive signals
            original_query=previous_query,
            original_response=previous_response,
            metadata={"follow_up_query": follow_up_query}
        )

        await self.feedback_store.store(feedback)
        logger.debug(f"Tracked follow-up for query {previous_query_id}")

        return feedback

    async def track_abandonment(
        self,
        session_id: str,
        query_id: str,
        original_query: str,
        original_response: str,
        idle_seconds: int
    ) -> Feedback:
        """
        Track when a user abandons an interaction (negative signal).

        Called when a user doesn't interact for an extended period.
        """
        feedback = Feedback(
            session_id=session_id,
            query_id=query_id,
            feedback_type=FeedbackType.IMPLICIT_ABANDONMENT,
            source=FeedbackSource.SYSTEM_DETECTION,
            rating=0.3,  # Abandonment is a negative signal
            original_query=original_query,
            original_response=original_response,
            metadata={"idle_seconds": idle_seconds}
        )

        await self.feedback_store.store(feedback)
        logger.debug(f"Tracked abandonment for query {query_id}")

        return feedback

    async def finalize_interaction(
        self,
        session_id: str,
        query_id: str
    ) -> Optional[Feedback]:
        """
        Finalize tracking for an interaction and generate engagement feedback.

        Args:
            session_id: Session identifier
            query_id: Query identifier

        Returns:
            Engagement feedback if generated
        """
        cache_key = f"{session_id}:{query_id}"

        if cache_key not in self._interaction_cache:
            return None

        interaction = self._interaction_cache.pop(cache_key)

        # Calculate engagement score from tracked events
        engagement_score = self._calculate_engagement_score(interaction)

        if engagement_score > 0:
            feedback = Feedback(
                session_id=session_id,
                query_id=query_id,
                feedback_type=FeedbackType.IMPLICIT_ENGAGEMENT,
                source=FeedbackSource.SYSTEM_DETECTION,
                rating=engagement_score,
                original_query=interaction["query"],
                original_response=interaction.get("response", ""),
                metadata={
                    "engagement_events": interaction["engagement_events"],
                    "interaction_duration": (
                        datetime.utcnow() - interaction["start_time"]
                    ).total_seconds()
                }
            )

            await self.feedback_store.store(feedback)
            return feedback

        return None

    def _calculate_engagement_score(self, interaction: Dict[str, Any]) -> float:
        """Calculate engagement score from tracked events."""
        if not interaction["engagement_events"]:
            return 0.0

        score = 0.0
        event_weights = {
            "scroll": 0.1,
            "copy": 0.3,
            "link_click": 0.4,
            "code_run": 0.5,
            "time_spent": 0.2,
        }

        for event in interaction["engagement_events"]:
            weight = event_weights.get(event["type"], 0.1)
            score += weight

        # Normalize to 0-1 range
        return min(1.0, score / 2.0)

    # ==================== Aggregation ====================

    async def aggregate_feedback(
        self,
        session_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> FeedbackAggregation:
        """
        Aggregate feedback for a session within a time window.

        Args:
            session_id: Session identifier
            start_time: Start of time window
            end_time: End of time window

        Returns:
            Aggregated feedback statistics
        """
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(
                hours=self.config.feedback.aggregation_window_hours
            )
        if end_time is None:
            end_time = datetime.utcnow()

        feedback_list = await self.feedback_store.query(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time
        )

        aggregation = FeedbackAggregation(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            total_feedback=len(feedback_list)
        )

        if not feedback_list:
            return aggregation

        ratings = []
        for fb in feedback_list:
            if fb.rating is not None:
                ratings.append(fb.rating)

            if fb.feedback_type in (
                FeedbackType.EXPLICIT_POSITIVE,
                FeedbackType.IMPLICIT_FOLLOW_UP
            ):
                aggregation.positive_feedback += 1
            elif fb.feedback_type in (
                FeedbackType.EXPLICIT_NEGATIVE,
                FeedbackType.IMPLICIT_ABANDONMENT
            ):
                aggregation.negative_feedback += 1
            elif fb.feedback_type == FeedbackType.IMPLICIT_CORRECTION:
                aggregation.corrections += 1

        if ratings:
            aggregation.average_rating = sum(ratings) / len(ratings)

        return aggregation

    async def get_feedback_summary(
        self,
        session_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get a summary of recent feedback for a session."""
        recent_feedback = await self.feedback_store.query(
            session_id=session_id,
            limit=limit
        )

        summary = {
            "total_feedback": len(recent_feedback),
            "feedback_by_type": {},
            "average_rating": 0.0,
            "recent_feedback": [fb.model_dump() for fb in recent_feedback[:5]]
        }

        ratings = []
        for fb in recent_feedback:
            fb_type = fb.feedback_type.value
            summary["feedback_by_type"][fb_type] = (
                summary["feedback_by_type"].get(fb_type, 0) + 1
            )
            if fb.rating is not None:
                ratings.append(fb.rating)

        if ratings:
            summary["average_rating"] = sum(ratings) / len(ratings)

        return summary
