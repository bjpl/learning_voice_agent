"""
Response Adapter - Adaptive Prompt and Response Customization

SPECIFICATION:
- Customizes system prompts based on learned user preferences
- Enhances query context with personalized information
- Calibrates responses to match user communication style
- Provides improvement suggestions based on session analysis

PATTERN: Strategy pattern with preference-based adaptation
WHY: Flexible, testable adaptation logic that improves over time

INTEGRATION:
- Applied in ConversationAgent before Claude API calls
- Uses PreferenceLearner for user profile data
- Coordinates with ImprovementEngine for active experiments
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import re
import json

from app.learning.config import learning_config, AdaptationConfig
from app.learning.store import LearningStore, learning_store, FeedbackRecord
from app.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UserPreferences:
    """
    User preferences learned from feedback and engagement

    PATTERN: Value object with default values
    WHY: Immutable, typed representation of preferences
    """
    # Response characteristics
    response_length: str = "medium"  # short, medium, long
    technical_depth: str = "intermediate"  # beginner, intermediate, expert
    communication_style: str = "balanced"  # formal, balanced, casual

    # Topic interests (most recent first)
    topic_interests: List[str] = field(default_factory=list)

    # Vocabulary adjustments (original -> preferred)
    vocabulary_adjustments: Dict[str, str] = field(default_factory=dict)

    # Confidence scores for each preference dimension
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    interaction_count: int = 0
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "response_length": self.response_length,
            "technical_depth": self.technical_depth,
            "communication_style": self.communication_style,
            "topic_interests": self.topic_interests,
            "vocabulary_adjustments": self.vocabulary_adjustments,
            "confidence_scores": self.confidence_scores,
            "interaction_count": self.interaction_count,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from dictionary"""
        return cls(
            response_length=data.get("response_length", "medium"),
            technical_depth=data.get("technical_depth", "intermediate"),
            communication_style=data.get("communication_style", "balanced"),
            topic_interests=data.get("topic_interests", []),
            vocabulary_adjustments=data.get("vocabulary_adjustments", {}),
            confidence_scores=data.get("confidence_scores", {}),
            interaction_count=data.get("interaction_count", 0),
            last_updated=data.get("last_updated")
        )

    def get_confidence(self, dimension: str) -> float:
        """Get confidence score for a dimension"""
        return self.confidence_scores.get(dimension, 0.5)

    def is_confident(self, dimension: str, threshold: float = 0.6) -> bool:
        """Check if preference is confident enough to use"""
        return self.get_confidence(dimension) >= threshold


@dataclass
class AdaptationResult:
    """Result of an adaptation operation"""
    success: bool
    adapted_content: str
    adaptations_applied: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementSuggestion:
    """Suggestion for improving responses"""
    dimension: str
    current_value: str
    suggested_value: str
    confidence: float
    rationale: str
    evidence: List[str] = field(default_factory=list)


# =============================================================================
# Response Adapter
# =============================================================================

class ResponseAdapter:
    """
    Adapts AI responses based on learned user preferences

    PATTERN: Adapter pattern with strategy-based customization
    WHY: Flexible adaptation that can be extended and tested independently

    Features:
    - Prompt adaptation based on user preferences
    - Context enhancement with personalized information
    - Response calibration for style matching
    - Improvement suggestions based on analysis
    """

    def __init__(
        self,
        store: Optional[LearningStore] = None,
        config: Optional[AdaptationConfig] = None
    ):
        """
        Initialize ResponseAdapter

        Args:
            store: Learning data store (uses global if not provided)
            config: Adaptation configuration
        """
        self.store = store or learning_store
        self.config = config or learning_config.adaptation

        # Cache for preferences
        self._preference_cache: Dict[str, Tuple[UserPreferences, float]] = {}
        self._cache_ttl = self.config.preference_cache_ttl

        # Active improvement experiments (session_id -> improvement_id)
        self._active_experiments: Dict[str, str] = {}

        logger.info(
            "response_adapter_initialized",
            cache_ttl=self._cache_ttl,
            prompt_adaptation=self.config.enable_prompt_adaptation,
            context_enhancement=self.config.enable_context_enhancement,
            response_calibration=self.config.enable_response_calibration
        )

    # =========================================================================
    # Main Adaptation Methods
    # =========================================================================

    async def adapt_prompt(
        self,
        base_prompt: str,
        session_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AdaptationResult:
        """
        Adapt system prompt based on learned user preferences

        PATTERN: Template method with extension points
        WHY: Consistent adaptation pipeline with customizable steps

        Args:
            base_prompt: Original system prompt
            session_id: User session identifier
            query: Current user query
            context: Additional context

        Returns:
            AdaptationResult with adapted prompt
        """
        if not self.config.enable_prompt_adaptation:
            return AdaptationResult(
                success=True,
                adapted_content=base_prompt,
                adaptations_applied=["none_disabled"],
                confidence=1.0
            )

        try:
            # Get user preferences
            prefs = await self._get_preferences(session_id)
            adaptations = []
            confidence_sum = 0.0
            confidence_count = 0

            # Build adaptation instructions
            adaptation_lines = []

            # 1. Response length adaptation
            if prefs.is_confident("response_length", self.config.min_confidence_for_adaptation):
                length_instruction = self._get_length_instruction(prefs.response_length)
                if length_instruction:
                    adaptation_lines.append(length_instruction)
                    adaptations.append(f"response_length:{prefs.response_length}")
                    confidence_sum += prefs.get_confidence("response_length")
                    confidence_count += 1

            # 2. Technical depth adaptation
            if prefs.is_confident("technical_depth", self.config.min_confidence_for_adaptation):
                depth_instruction = self._get_depth_instruction(prefs.technical_depth)
                if depth_instruction:
                    adaptation_lines.append(depth_instruction)
                    adaptations.append(f"technical_depth:{prefs.technical_depth}")
                    confidence_sum += prefs.get_confidence("technical_depth")
                    confidence_count += 1

            # 3. Communication style adaptation
            if prefs.is_confident("communication_style", self.config.min_confidence_for_adaptation):
                style_instruction = self._get_style_instruction(prefs.communication_style)
                if style_instruction:
                    adaptation_lines.append(style_instruction)
                    adaptations.append(f"communication_style:{prefs.communication_style}")
                    confidence_sum += prefs.get_confidence("communication_style")
                    confidence_count += 1

            # 4. Topic interests context
            if prefs.topic_interests:
                relevant_topics = self._get_relevant_topics(query, prefs.topic_interests)
                if relevant_topics:
                    topics_str = ", ".join(relevant_topics[:5])
                    adaptation_lines.append(f"User's areas of interest: {topics_str}")
                    adaptations.append(f"topic_context:{len(relevant_topics)}")

            # Build adapted prompt
            if adaptation_lines:
                adapted_prompt = base_prompt + "\n\nUser preferences:\n" + "\n".join(
                    f"- {line}" for line in adaptation_lines
                )
            else:
                adapted_prompt = base_prompt

            avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.5

            logger.debug(
                "prompt_adapted",
                session_id=session_id,
                adaptations_count=len(adaptations),
                confidence=avg_confidence
            )

            return AdaptationResult(
                success=True,
                adapted_content=adapted_prompt,
                adaptations_applied=adaptations,
                confidence=avg_confidence,
                metadata={
                    "preference_count": prefs.interaction_count,
                    "query_length": len(query)
                }
            )

        except Exception as e:
            logger.error(
                "prompt_adaptation_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return AdaptationResult(
                success=False,
                adapted_content=base_prompt,
                adaptations_applied=["error"],
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def enhance_context(
        self,
        query: str,
        session_id: str,
        base_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance query context with learned preferences and relevant history

        PATTERN: Context enrichment
        WHY: Provide personalized context without modifying the core query

        Args:
            query: User's current query
            session_id: User session identifier
            base_context: Existing context to enhance

        Returns:
            Enhanced context dictionary
        """
        if not self.config.enable_context_enhancement:
            return base_context or {}

        try:
            enhanced_context = dict(base_context) if base_context else {}
            prefs = await self._get_preferences(session_id)

            # Add preference-based context
            enhanced_context["user_preferences"] = {
                "response_length": prefs.response_length,
                "technical_depth": prefs.technical_depth,
                "communication_style": prefs.communication_style,
                "interaction_count": prefs.interaction_count
            }

            # Add relevant topic history
            if prefs.topic_interests:
                relevant_topics = self._get_relevant_topics(query, prefs.topic_interests)
                if relevant_topics:
                    enhanced_context["relevant_topics"] = relevant_topics[:5]

            # Add vocabulary context for specialized terms
            if prefs.vocabulary_adjustments:
                enhanced_context["vocabulary_preferences"] = len(prefs.vocabulary_adjustments)

            # Check for active experiments
            if session_id in self._active_experiments:
                enhanced_context["experiment_id"] = self._active_experiments[session_id]

            logger.debug(
                "context_enhanced",
                session_id=session_id,
                added_keys=list(set(enhanced_context.keys()) - set(base_context.keys() if base_context else []))
            )

            return enhanced_context

        except Exception as e:
            logger.error(
                "context_enhancement_failed",
                session_id=session_id,
                error=str(e)
            )
            return base_context or {}

    async def calibrate_response(
        self,
        response: str,
        session_id: str,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> AdaptationResult:
        """
        Post-process response based on user preferences

        PATTERN: Response filtering and adjustment
        WHY: Final pass to ensure response matches user expectations

        Args:
            response: Generated response text
            session_id: User session identifier
            response_metadata: Metadata about the response

        Returns:
            AdaptationResult with calibrated response
        """
        if not self.config.enable_response_calibration:
            return AdaptationResult(
                success=True,
                adapted_content=response,
                adaptations_applied=["none_disabled"],
                confidence=1.0
            )

        try:
            prefs = await self._get_preferences(session_id)
            calibrated = response
            adjustments_made = []

            # Apply vocabulary adjustments (up to max)
            if prefs.vocabulary_adjustments:
                adjustment_count = 0
                for original, preferred in prefs.vocabulary_adjustments.items():
                    if adjustment_count >= self.config.max_response_adjustments:
                        break

                    # Case-insensitive replacement with word boundaries
                    pattern = re.compile(rf'\b{re.escape(original)}\b', re.IGNORECASE)
                    if pattern.search(calibrated):
                        calibrated = pattern.sub(preferred, calibrated)
                        adjustments_made.append(f"vocab:{original}->{preferred}")
                        adjustment_count += 1

            logger.debug(
                "response_calibrated",
                session_id=session_id,
                adjustments_count=len(adjustments_made),
                original_length=len(response),
                calibrated_length=len(calibrated)
            )

            return AdaptationResult(
                success=True,
                adapted_content=calibrated,
                adaptations_applied=adjustments_made if adjustments_made else ["no_changes"],
                confidence=1.0,
                metadata={
                    "original_length": len(response),
                    "calibrated_length": len(calibrated)
                }
            )

        except Exception as e:
            logger.error(
                "response_calibration_failed",
                session_id=session_id,
                error=str(e)
            )
            return AdaptationResult(
                success=False,
                adapted_content=response,
                adaptations_applied=["error"],
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def get_improvement_suggestions(
        self,
        session_id: str,
        limit: int = 5
    ) -> List[ImprovementSuggestion]:
        """
        Get suggestions for improving responses based on session analysis

        PATTERN: Analysis-based recommendation
        WHY: Provide actionable insights for continuous improvement

        Args:
            session_id: User session identifier
            limit: Maximum number of suggestions

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        try:
            # Get feedback history
            feedback_history = await self.store.get_feedback_history(
                session_id,
                limit=100
            )

            if not feedback_history:
                return suggestions

            # Analyze feedback patterns
            prefs = await self._get_preferences(session_id)

            # 1. Check response length patterns
            length_suggestion = await self._analyze_length_feedback(
                feedback_history, prefs
            )
            if length_suggestion:
                suggestions.append(length_suggestion)

            # 2. Check technical depth patterns
            depth_suggestion = await self._analyze_depth_feedback(
                feedback_history, prefs
            )
            if depth_suggestion:
                suggestions.append(depth_suggestion)

            # 3. Check style patterns
            style_suggestion = await self._analyze_style_feedback(
                feedback_history, prefs
            )
            if style_suggestion:
                suggestions.append(style_suggestion)

            # 4. Check vocabulary corrections
            vocab_suggestions = await self._analyze_vocabulary_corrections(
                feedback_history, prefs
            )
            suggestions.extend(vocab_suggestions)

            # Sort by confidence and limit
            suggestions.sort(key=lambda s: s.confidence, reverse=True)

            logger.info(
                "improvement_suggestions_generated",
                session_id=session_id,
                total_suggestions=len(suggestions),
                returned=min(len(suggestions), limit)
            )

            return suggestions[:limit]

        except Exception as e:
            logger.error(
                "get_improvement_suggestions_failed",
                session_id=session_id,
                error=str(e)
            )
            return suggestions

    # =========================================================================
    # Preference Management
    # =========================================================================

    async def _get_preferences(self, session_id: str) -> UserPreferences:
        """
        Get user preferences with caching

        PATTERN: Cache-aside with TTL
        WHY: Reduce database queries while keeping data fresh
        """
        current_time = datetime.utcnow().timestamp()

        # Check cache
        if session_id in self._preference_cache:
            prefs, cached_at = self._preference_cache[session_id]
            if current_time - cached_at < self._cache_ttl:
                return prefs

        # Load from store
        stored_prefs = await self.store.get_preferences(session_id)

        if stored_prefs:
            prefs = UserPreferences.from_dict(stored_prefs)
        else:
            prefs = UserPreferences()

        # Update cache
        self._preference_cache[session_id] = (prefs, current_time)

        return prefs

    def invalidate_cache(self, session_id: Optional[str] = None):
        """Invalidate preference cache"""
        if session_id:
            self._preference_cache.pop(session_id, None)
        else:
            self._preference_cache.clear()

    # =========================================================================
    # Instruction Generation Helpers
    # =========================================================================

    def _get_length_instruction(self, length_pref: str) -> Optional[str]:
        """Get instruction for response length preference"""
        instructions = {
            "short": "Be concise. Keep responses under 100 words when possible.",
            "medium": "Provide balanced responses with appropriate detail.",
            "long": "Provide detailed explanations with examples when helpful."
        }
        return instructions.get(length_pref)

    def _get_depth_instruction(self, depth_pref: str) -> Optional[str]:
        """Get instruction for technical depth preference"""
        instructions = {
            "beginner": "Explain concepts simply. Avoid jargon and use analogies.",
            "intermediate": "Use standard terminology. Assume some background knowledge.",
            "expert": "Use precise technical terminology. Assume domain expertise."
        }
        return instructions.get(depth_pref)

    def _get_style_instruction(self, style_pref: str) -> Optional[str]:
        """Get instruction for communication style preference"""
        instructions = {
            "formal": "Maintain a professional and structured tone.",
            "balanced": "Be clear and approachable with appropriate formality.",
            "casual": "Be conversational and friendly in tone."
        }
        return instructions.get(style_pref)

    def _get_relevant_topics(
        self,
        query: str,
        topic_interests: List[str]
    ) -> List[str]:
        """Find topics relevant to the current query"""
        query_lower = query.lower()
        relevant = []

        for topic in topic_interests:
            topic_lower = topic.lower()
            # Check if topic appears in query or is semantically related
            if topic_lower in query_lower or any(
                word in topic_lower for word in query_lower.split()
                if len(word) > 3
            ):
                relevant.append(topic)

        return relevant

    # =========================================================================
    # Feedback Analysis Helpers
    # =========================================================================

    async def _analyze_length_feedback(
        self,
        feedback_history: List[FeedbackRecord],
        prefs: UserPreferences
    ) -> Optional[ImprovementSuggestion]:
        """Analyze feedback for response length patterns"""
        short_positive = 0
        short_negative = 0
        long_positive = 0
        long_negative = 0

        for feedback in feedback_history:
            word_count = feedback.response_metadata.get("word_count", 100)

            if feedback.helpful is True:
                if word_count < learning_config.preferences.short_response_threshold:
                    short_positive += 1
                elif word_count > learning_config.preferences.long_response_threshold:
                    long_positive += 1
            elif feedback.helpful is False:
                if word_count < learning_config.preferences.short_response_threshold:
                    short_negative += 1
                elif word_count > learning_config.preferences.long_response_threshold:
                    long_negative += 1

        # Determine if there's a clear pattern
        if short_positive > short_negative * 2 and prefs.response_length != "short":
            return ImprovementSuggestion(
                dimension="response_length",
                current_value=prefs.response_length,
                suggested_value="short",
                confidence=min(0.9, short_positive / max(1, short_positive + short_negative)),
                rationale="User consistently rates shorter responses as helpful",
                evidence=[f"{short_positive} positive ratings for short responses"]
            )

        if long_positive > long_negative * 2 and prefs.response_length != "long":
            return ImprovementSuggestion(
                dimension="response_length",
                current_value=prefs.response_length,
                suggested_value="long",
                confidence=min(0.9, long_positive / max(1, long_positive + long_negative)),
                rationale="User prefers detailed, longer responses",
                evidence=[f"{long_positive} positive ratings for detailed responses"]
            )

        return None

    async def _analyze_depth_feedback(
        self,
        feedback_history: List[FeedbackRecord],
        prefs: UserPreferences
    ) -> Optional[ImprovementSuggestion]:
        """Analyze feedback for technical depth patterns"""
        # Look for patterns in corrections mentioning complexity
        complexity_up = 0
        complexity_down = 0

        complexity_up_terms = ["more detail", "explain more", "technical", "deeper"]
        complexity_down_terms = ["simpler", "too complex", "confused", "basic"]

        for feedback in feedback_history:
            if feedback.correction:
                correction_lower = feedback.correction.lower()
                if any(term in correction_lower for term in complexity_up_terms):
                    complexity_up += 1
                if any(term in correction_lower for term in complexity_down_terms):
                    complexity_down += 1

        if complexity_up > complexity_down + 2:
            suggested = "expert" if prefs.technical_depth == "intermediate" else "intermediate"
            return ImprovementSuggestion(
                dimension="technical_depth",
                current_value=prefs.technical_depth,
                suggested_value=suggested,
                confidence=0.7,
                rationale="User has requested more technical depth",
                evidence=[f"{complexity_up} requests for more detail"]
            )

        if complexity_down > complexity_up + 2:
            suggested = "beginner" if prefs.technical_depth == "intermediate" else "intermediate"
            return ImprovementSuggestion(
                dimension="technical_depth",
                current_value=prefs.technical_depth,
                suggested_value=suggested,
                confidence=0.7,
                rationale="User has indicated responses are too complex",
                evidence=[f"{complexity_down} requests for simpler explanations"]
            )

        return None

    async def _analyze_style_feedback(
        self,
        feedback_history: List[FeedbackRecord],
        prefs: UserPreferences
    ) -> Optional[ImprovementSuggestion]:
        """Analyze feedback for communication style patterns"""
        formal_requests = 0
        casual_requests = 0

        formal_terms = ["professional", "formal", "business"]
        casual_terms = ["casual", "friendly", "informal", "relaxed"]

        for feedback in feedback_history:
            if feedback.correction:
                correction_lower = feedback.correction.lower()
                if any(term in correction_lower for term in formal_terms):
                    formal_requests += 1
                if any(term in correction_lower for term in casual_terms):
                    casual_requests += 1

        if formal_requests > casual_requests + 2 and prefs.communication_style != "formal":
            return ImprovementSuggestion(
                dimension="communication_style",
                current_value=prefs.communication_style,
                suggested_value="formal",
                confidence=0.65,
                rationale="User prefers professional communication",
                evidence=[f"{formal_requests} requests for formal tone"]
            )

        if casual_requests > formal_requests + 2 and prefs.communication_style != "casual":
            return ImprovementSuggestion(
                dimension="communication_style",
                current_value=prefs.communication_style,
                suggested_value="casual",
                confidence=0.65,
                rationale="User prefers casual communication",
                evidence=[f"{casual_requests} requests for casual tone"]
            )

        return None

    async def _analyze_vocabulary_corrections(
        self,
        feedback_history: List[FeedbackRecord],
        prefs: UserPreferences
    ) -> List[ImprovementSuggestion]:
        """Analyze feedback for vocabulary correction patterns"""
        suggestions = []

        # Track correction patterns
        correction_counts: Dict[str, int] = {}

        for feedback in feedback_history:
            if feedback.correction:
                # Look for "X instead of Y" patterns
                patterns = [
                    r"use ['\"]?(\w+)['\"]? instead of ['\"]?(\w+)['\"]?",
                    r"say ['\"]?(\w+)['\"]? not ['\"]?(\w+)['\"]?",
                    r"prefer ['\"]?(\w+)['\"]? over ['\"]?(\w+)['\"]?"
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, feedback.correction.lower())
                    for preferred, original in matches:
                        key = f"{original}->{preferred}"
                        correction_counts[key] = correction_counts.get(key, 0) + 1

        # Create suggestions for repeated corrections
        for correction, count in correction_counts.items():
            if count >= 2:
                original, preferred = correction.split("->")
                if original not in prefs.vocabulary_adjustments:
                    suggestions.append(ImprovementSuggestion(
                        dimension="vocabulary",
                        current_value=original,
                        suggested_value=preferred,
                        confidence=min(0.8, count * 0.2),
                        rationale=f"User has corrected '{original}' to '{preferred}' multiple times",
                        evidence=[f"Corrected {count} times"]
                    ))

        return suggestions[:3]  # Limit vocabulary suggestions

    # =========================================================================
    # Experiment Management
    # =========================================================================

    def register_experiment(self, session_id: str, improvement_id: str):
        """Register an active experiment for a session"""
        self._active_experiments[session_id] = improvement_id
        logger.debug(
            "experiment_registered",
            session_id=session_id,
            improvement_id=improvement_id
        )

    def unregister_experiment(self, session_id: str):
        """Unregister experiment for a session"""
        if session_id in self._active_experiments:
            del self._active_experiments[session_id]
            logger.debug("experiment_unregistered", session_id=session_id)

    def is_treatment_group(self, session_id: str) -> bool:
        """Check if session is in treatment group for any experiment"""
        return session_id in self._active_experiments

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_adaptation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get adaptation statistics for a session"""
        prefs = await self._get_preferences(session_id)

        return {
            "session_id": session_id,
            "preferences": prefs.to_dict(),
            "confidence_levels": {
                "response_length": prefs.get_confidence("response_length"),
                "technical_depth": prefs.get_confidence("technical_depth"),
                "communication_style": prefs.get_confidence("communication_style")
            },
            "vocabulary_adjustments_count": len(prefs.vocabulary_adjustments),
            "topic_interests_count": len(prefs.topic_interests),
            "interaction_count": prefs.interaction_count,
            "has_active_experiment": session_id in self._active_experiments,
            "cache_status": session_id in self._preference_cache
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_response_adapter(
    store: Optional[LearningStore] = None,
    config: Optional[AdaptationConfig] = None
) -> ResponseAdapter:
    """
    Factory function to create a ResponseAdapter

    Args:
        store: Learning store instance
        config: Adaptation configuration

    Returns:
        Configured ResponseAdapter instance
    """
    return ResponseAdapter(store=store, config=config)


# Global adapter instance
response_adapter = ResponseAdapter()
