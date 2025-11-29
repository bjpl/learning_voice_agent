"""
Preference Learner
==================

Learns user preferences from feedback and interaction patterns.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, List, Any, Union
from collections import defaultdict

from app.learning.config import LearningConfig, learning_config
from app.learning.models import (
    UserPreference,
    Feedback,
    FeedbackType,
    QualityScore
)
from app.learning.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)


class PreferenceLearner:
    """
    Learns user preferences from feedback and engagement patterns.

    Features:
    - Learns from explicit feedback (ratings, thumbs up/down)
    - Learns from implicit signals (engagement, corrections)
    - Maintains preference confidence scores
    - Applies time-based decay to preferences
    - Supports session-level and user-level preferences
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        feedback_store: Optional[FeedbackStore] = None
    ):
        """
        Initialize the preference learner.

        Args:
            config: Learning configuration
            feedback_store: Feedback store for accessing historical data
        """
        self.config = config or learning_config
        self.feedback_store = feedback_store

        # In-memory preference cache
        self._preferences: Dict[str, Dict[str, UserPreference]] = {}
        self._learning_history: Dict[str, List[Dict]] = defaultdict(list)

    async def initialize(self) -> None:
        """Initialize the preference learner and load saved preferences."""
        await self._load_preferences()
        logger.info("PreferenceLearner initialized")

    async def close(self) -> None:
        """Close and save preferences."""
        await self._save_preferences()

    async def _load_preferences(self) -> None:
        """Load preferences from persistent storage."""
        pref_path = Path(self.config.preference_learning.preference_file_path)

        if pref_path.exists():
            try:
                with open(pref_path, "r") as f:
                    data = json.load(f)

                for key, prefs in data.items():
                    self._preferences[key] = {
                        cat: UserPreference(**pref_data)
                        for cat, pref_data in prefs.items()
                    }

                logger.info(f"Loaded preferences for {len(self._preferences)} sessions/users")
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")

    async def _save_preferences(self) -> None:
        """Save preferences to persistent storage."""
        pref_path = Path(self.config.preference_learning.preference_file_path)
        pref_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {}
            for key, prefs in self._preferences.items():
                data[key] = {
                    cat: pref.model_dump()
                    for cat, pref in prefs.items()
                }

            with open(pref_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved preferences for {len(self._preferences)} sessions/users")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    def _get_preference_key(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Get the cache key for preferences."""
        if user_id:
            return f"user:{user_id}"
        elif session_id:
            return f"session:{session_id}"
        return "global"

    async def get_preferences(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, UserPreference]:
        """
        Get all preferences for a session or user.

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Dictionary of category -> UserPreference
        """
        key = self._get_preference_key(session_id, user_id)

        if key in self._preferences:
            # Apply decay to all preferences
            prefs = self._preferences[key]
            for pref in prefs.values():
                self._apply_decay(pref)
            return prefs

        return {}

    async def get_preference(
        self,
        category: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[UserPreference]:
        """
        Get a specific preference by category.

        Args:
            category: Preference category
            session_id: Session identifier
            user_id: User identifier

        Returns:
            UserPreference or None if not found
        """
        prefs = await self.get_preferences(session_id, user_id)
        return prefs.get(category)

    async def learn_from_feedback(
        self,
        feedback: Feedback,
        response_characteristics: Optional[Dict[str, Any]] = None
    ) -> List[UserPreference]:
        """
        Learn preferences from a feedback instance.

        Args:
            feedback: The feedback to learn from
            response_characteristics: Characteristics of the response

        Returns:
            List of updated preferences
        """
        if not response_characteristics:
            response_characteristics = self._extract_characteristics(
                feedback.original_response
            )

        updated_prefs = []

        # Determine signal strength from feedback type and rating
        signal = self._get_signal_strength(feedback)

        if abs(signal) < 0.1:
            return updated_prefs  # Signal too weak

        key = self._get_preference_key(
            session_id=feedback.session_id,
            user_id=feedback.user_id
        )

        if key not in self._preferences:
            self._preferences[key] = {}

        # Update preferences based on response characteristics
        for category, value in response_characteristics.items():
            if category in self.config.preference_learning.preference_categories:
                pref = await self._update_preference(
                    key=key,
                    category=category,
                    value=value,
                    signal=signal
                )
                if pref:
                    updated_prefs.append(pref)

        # Record learning history
        self._learning_history[key].append({
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_id": feedback.id,
            "signal": signal,
            "updated_categories": [p.category for p in updated_prefs]
        })

        logger.debug(
            f"Learned from feedback {feedback.id}, "
            f"updated {len(updated_prefs)} preferences"
        )

        return updated_prefs

    async def learn_from_engagement(
        self,
        session_id: str,
        engagement_metrics: Dict[str, Any],
        response_characteristics: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[UserPreference]:
        """
        Learn preferences from engagement metrics.

        Args:
            session_id: Session identifier
            engagement_metrics: Metrics like time spent, scroll depth, etc.
            response_characteristics: Characteristics of the response
            user_id: Optional user identifier

        Returns:
            List of updated preferences
        """
        updated_prefs = []

        # Calculate engagement signal
        signal = self._calculate_engagement_signal(engagement_metrics)

        if abs(signal) < 0.1:
            return updated_prefs

        key = self._get_preference_key(session_id, user_id)

        if key not in self._preferences:
            self._preferences[key] = {}

        for category, value in response_characteristics.items():
            if category in self.config.preference_learning.preference_categories:
                pref = await self._update_preference(
                    key=key,
                    category=category,
                    value=value,
                    signal=signal * 0.5  # Engagement signals are weaker
                )
                if pref:
                    updated_prefs.append(pref)

        return updated_prefs

    async def _update_preference(
        self,
        key: str,
        category: str,
        value: Any,
        signal: float
    ) -> Optional[UserPreference]:
        """
        Update a single preference with exponential moving average.

        Args:
            key: Preference cache key
            category: Preference category
            value: The observed value
            signal: Signal strength (-1 to 1)

        Returns:
            Updated preference or None
        """
        learning_rate = self.config.preference_learning.learning_rate

        if category not in self._preferences[key]:
            # Create new preference
            self._preferences[key][category] = UserPreference(
                category=category,
                value=value,
                confidence=abs(signal) * learning_rate,
                learned_from_samples=1
            )
            return self._preferences[key][category]

        pref = self._preferences[key][category]

        # Update with exponential moving average
        if signal > 0 and pref.value == value:
            # Positive signal for same value: increase confidence
            pref.confidence = min(
                1.0,
                pref.confidence + (1 - pref.confidence) * learning_rate * signal
            )
        elif signal < 0 and pref.value == value:
            # Negative signal for same value: decrease confidence
            pref.confidence = max(
                0.0,
                pref.confidence + pref.confidence * learning_rate * signal
            )
        elif signal > 0 and pref.value != value:
            # Positive signal for different value: gradually shift
            if pref.confidence < 0.5:
                pref.value = value
                pref.confidence = abs(signal) * learning_rate
        elif signal < 0 and pref.value != value:
            # Negative signal for different value: reinforce current
            pref.confidence = min(
                1.0,
                pref.confidence + (1 - pref.confidence) * learning_rate * abs(signal) * 0.5
            )

        pref.learned_from_samples += 1
        pref.last_updated = datetime.utcnow()

        return pref

    def _get_signal_strength(self, feedback: Feedback) -> float:
        """Calculate signal strength from feedback."""
        if feedback.rating is not None:
            # Convert 0-1 rating to -1 to 1 signal
            return (feedback.rating - 0.5) * 2

        # Use feedback type for signal
        type_signals = {
            FeedbackType.EXPLICIT_POSITIVE: 0.8,
            FeedbackType.EXPLICIT_NEGATIVE: -0.8,
            FeedbackType.IMPLICIT_FOLLOW_UP: 0.5,
            FeedbackType.IMPLICIT_CORRECTION: -0.6,
            FeedbackType.IMPLICIT_ABANDONMENT: -0.4,
            FeedbackType.IMPLICIT_ENGAGEMENT: 0.3,
        }

        return type_signals.get(feedback.feedback_type, 0.0)

    def _calculate_engagement_signal(self, metrics: Dict[str, Any]) -> float:
        """Calculate signal strength from engagement metrics."""
        signal = 0.0

        # Time spent (normalized)
        time_spent = metrics.get("time_spent_seconds", 0)
        if time_spent > 60:
            signal += 0.3
        elif time_spent > 30:
            signal += 0.1
        elif time_spent < 5:
            signal -= 0.2

        # Scroll depth
        scroll_depth = metrics.get("scroll_depth", 0)
        if scroll_depth > 0.8:
            signal += 0.2
        elif scroll_depth < 0.3:
            signal -= 0.1

        # Interactions (copies, clicks)
        interactions = metrics.get("interaction_count", 0)
        if interactions > 2:
            signal += 0.3
        elif interactions > 0:
            signal += 0.1

        return max(-1.0, min(1.0, signal))

    def _extract_characteristics(self, response: str) -> Dict[str, Any]:
        """Extract response characteristics for learning."""
        characteristics = {}

        # Response length category
        word_count = len(response.split())
        if word_count < 50:
            characteristics["response_length"] = "brief"
        elif word_count < 150:
            characteristics["response_length"] = "medium"
        else:
            characteristics["response_length"] = "detailed"

        # Detail level
        if "```" in response or "\n-" in response or "example" in response.lower():
            characteristics["detail_level"] = "detailed"
        elif word_count < 30:
            characteristics["detail_level"] = "brief"
        else:
            characteristics["detail_level"] = "standard"

        # Formality (heuristic)
        contractions = ["don't", "can't", "won't", "it's", "that's"]
        contraction_count = sum(1 for c in contractions if c in response.lower())
        if contraction_count > 2:
            characteristics["formality"] = "casual"
        elif contraction_count == 0:
            characteristics["formality"] = "formal"
        else:
            characteristics["formality"] = "neutral"

        # Example frequency
        example_indicators = ["example", "for instance", "such as", "like"]
        example_count = sum(1 for e in example_indicators if e in response.lower())
        if example_count >= 2:
            characteristics["example_frequency"] = "frequent"
        elif example_count == 1:
            characteristics["example_frequency"] = "moderate"
        else:
            characteristics["example_frequency"] = "minimal"

        return characteristics

    def _apply_decay(self, pref: UserPreference) -> None:
        """Apply time-based decay to a preference confidence."""
        if pref.last_updated:
            days_elapsed = (datetime.utcnow() - pref.last_updated).total_seconds() / 86400
            if days_elapsed > 0:
                pref.apply_decay(days_elapsed)

    async def clear_preferences(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """Clear preferences for a session or user."""
        key = self._get_preference_key(session_id, user_id)
        if key in self._preferences:
            del self._preferences[key]
        await self._save_preferences()

    async def get_learning_history(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get the learning history for a session or user."""
        key = self._get_preference_key(session_id, user_id)
        history = self._learning_history.get(key, [])
        return history[-limit:]

    async def predict_preference(
        self,
        category: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[tuple[Any, float]]:
        """
        Predict the most likely preference value for a category.

        Args:
            category: Preference category
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Tuple of (predicted_value, confidence) or None
        """
        pref = await self.get_preference(category, session_id, user_id)

        if pref and pref.confidence >= 0.3:
            return (pref.value, pref.confidence)

        # Fall back to defaults
        defaults = {
            "response_length": "medium",
            "formality": "neutral",
            "detail_level": "standard",
            "example_frequency": "moderate",
            "technical_depth": "intermediate"
        }

        if category in defaults:
            return (defaults[category], 0.5)

        return None

    def get_preference_summary(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get a summary of learned preferences."""
        key = self._get_preference_key(session_id, user_id)
        prefs = self._preferences.get(key, {})

        summary = {
            "total_preferences": len(prefs),
            "high_confidence": [],
            "medium_confidence": [],
            "low_confidence": []
        }

        for category, pref in prefs.items():
            self._apply_decay(pref)

            item = {
                "category": category,
                "value": pref.value,
                "confidence": round(pref.confidence, 3),
                "samples": pref.learned_from_samples
            }

            if pref.confidence >= 0.7:
                summary["high_confidence"].append(item)
            elif pref.confidence >= 0.4:
                summary["medium_confidence"].append(item)
            else:
                summary["low_confidence"].append(item)

        return summary

    async def get_adaptation_context(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get adaptation context for response generation.

        This is a wrapper for test compatibility that combines
        preferences and learning history into adaptation context.
        """
        summary = self.get_preference_summary(session_id, user_id)

        # Get preferences
        preferences = await self.get_preferences(session_id, user_id)

        # Build adaptation context as an object with attribute access
        context = SimpleNamespace(
            session_id=session_id,
            user_id=user_id,
            preferences=preferences,
            summary=summary,
            has_preferences=len(preferences) > 0,
            confidence_levels=SimpleNamespace(
                high=len(summary.get("high_confidence", [])),
                medium=len(summary.get("medium_confidence", [])),
                low=len(summary.get("low_confidence", []))
            )
        )

        return context
