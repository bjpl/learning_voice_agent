"""
Quality Scoring Algorithms - SPARC Implementation

SPECIFICATION:
- Relevance scoring using embedding cosine similarity
- Engagement scoring from implicit feedback metrics
- Clarity scoring using readability metrics (Flesch-Kincaid)
- Helpfulness scoring from explicit user feedback
- Accuracy scoring via self-consistency checks

ARCHITECTURE:
- Strategy pattern for interchangeable algorithms
- Async support for embedding generation
- Configurable thresholds and weights
- Extensible for new scoring methods
"""
import re
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Protocol
from enum import Enum

from app.logger import db_logger


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ScoringConfig:
    """
    CONCEPT: Centralized configuration for scoring algorithms
    WHY: Enables tuning without code changes
    """
    # Relevance thresholds
    relevance_high_threshold: float = 0.8
    relevance_low_threshold: float = 0.4

    # Engagement parameters
    expected_follow_ups: int = 3
    session_depth_target: int = 5
    max_session_duration_minutes: int = 30

    # Clarity parameters
    target_grade_level: float = 8.0  # 8th grade reading level
    min_sentence_length: int = 3
    max_sentence_length: int = 40

    # Helpfulness parameters
    feedback_recency_weight: float = 0.8  # Recent feedback weighted higher
    min_feedback_count: int = 1

    # Accuracy parameters
    consistency_threshold: float = 0.85
    fact_check_weight: float = 0.3

    # General
    min_text_length: int = 10
    default_score: float = 0.5


# Global configuration instance
scoring_config = ScoringConfig()


# ============================================================================
# SCORING PROTOCOLS
# ============================================================================

class ScoringStrategy(Protocol):
    """Protocol for scoring strategy implementations"""

    async def calculate(self, *args, **kwargs) -> float:
        """Calculate score for given inputs"""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation"""

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        ...


# ============================================================================
# RELEVANCE SCORING
# ============================================================================

class RelevanceScorer:
    """
    CONCEPT: Measure semantic similarity between query and response

    ALGORITHM:
    1. Generate embeddings for query and response
    2. Calculate cosine similarity
    3. Apply non-linear scaling for better discrimination

    WHY:
    - Semantic similarity captures meaning beyond keyword matching
    - Cosine similarity is scale-invariant
    - Non-linear scaling improves score distribution
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[ScoringConfig] = None
    ):
        self.embedding_provider = embedding_provider
        self.config = config or scoring_config
        self._cache: Dict[str, np.ndarray] = {}

    async def calculate(
        self,
        query: str,
        response: str,
        context: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate relevance score

        Args:
            query: User query text
            response: Agent response text
            context: Optional conversation context

        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not query or not response:
            return self.config.default_score

        try:
            # Get embeddings
            query_emb = await self._get_embedding(query)
            response_emb = await self._get_embedding(response)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_emb, response_emb)

            # Apply context boost if response addresses context
            if context:
                context_boost = await self._calculate_context_boost(
                    response, context
                )
                similarity = min(1.0, similarity + context_boost * 0.1)

            # Apply non-linear scaling for better discrimination
            scaled_score = self._scale_similarity(similarity)

            db_logger.debug(
                "relevance_score_calculated",
                query_length=len(query),
                response_length=len(response),
                raw_similarity=round(similarity, 4),
                scaled_score=round(scaled_score, 4)
            )

            return scaled_score

        except Exception as e:
            db_logger.error(
                "relevance_scoring_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return self.config.default_score

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        cache_key = hash(text[:500])  # Use first 500 chars for cache key

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.embedding_provider:
            embedding = await self.embedding_provider.generate_embedding(text)
        else:
            # Fallback: simple TF-IDF-like vector (for testing)
            embedding = self._simple_embedding(text)

        # Cache with size limit
        if len(self._cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = embedding
        return embedding

    def _simple_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Simple word-frequency based embedding (fallback)

        NOTE: This is a basic fallback - production should use
        sentence-transformers or similar
        """
        # Tokenize and normalize
        words = text.lower().split()
        word_set = set(words)

        # Create a simple hash-based embedding
        embedding = np.zeros(dim)
        for word in word_set:
            # Use word hash to determine embedding positions
            word_hash = hash(word)
            position = word_hash % dim
            value = (word_hash // dim) % 100 / 100.0
            embedding[position] += value * words.count(word)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors

        FORMULA: cos(a, b) = (a . b) / (||a|| * ||b||)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _scale_similarity(self, similarity: float) -> float:
        """
        Apply non-linear scaling to improve score distribution

        ALGORITHM: Sigmoid-like transformation
        - Scores near threshold get pushed away from 0.5
        - Very high/low scores saturate
        """
        # Map [-1, 1] to [0, 1]
        normalized = (similarity + 1) / 2

        # Apply sigmoid-like transformation
        # This pushes middling scores toward extremes
        k = 5  # Steepness factor
        threshold = 0.5

        scaled = 1 / (1 + math.exp(-k * (normalized - threshold)))

        return max(0.0, min(1.0, scaled))

    async def _calculate_context_boost(
        self,
        response: str,
        context: List[Dict]
    ) -> float:
        """Calculate boost for responses that address context"""
        if not context:
            return 0.0

        # Check if response references context topics
        context_text = " ".join(
            str(c.get("user_text", "") + c.get("agent_text", ""))
            for c in context[-3:]  # Last 3 exchanges
        )

        context_words = set(context_text.lower().split())
        response_words = set(response.lower().split())

        # Calculate word overlap
        common_words = context_words & response_words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "must", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "or", "and", "but"}

        meaningful_common = common_words - stop_words
        meaningful_context = context_words - stop_words

        if not meaningful_context:
            return 0.0

        overlap_ratio = len(meaningful_common) / len(meaningful_context)

        return min(1.0, overlap_ratio)


# ============================================================================
# ENGAGEMENT SCORING
# ============================================================================

@dataclass
class SessionMetrics:
    """Metrics for engagement calculation"""
    follow_up_count: int = 0
    session_turns: int = 0
    session_duration_seconds: float = 0.0
    questions_asked: int = 0
    topics_explored: int = 1
    user_elaborations: int = 0


class EngagementScorer:
    """
    CONCEPT: Measure user engagement through implicit signals

    SIGNALS:
    - Follow-up count: More follow-ups = higher engagement
    - Session depth: Longer conversations = more engaged
    - Question rate: Questions indicate active exploration
    - Topic continuity: Related topics show sustained interest

    WHY: Implicit signals often more reliable than explicit feedback
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or scoring_config

    async def calculate(
        self,
        session_metrics: SessionMetrics,
        current_turn: int = 1
    ) -> float:
        """
        Calculate engagement score from session metrics

        Args:
            session_metrics: Current session metrics
            current_turn: Current turn number in session

        Returns:
            Engagement score (0.0 to 1.0)
        """
        scores = []

        # Follow-up score (normalized by expected follow-ups)
        follow_up_score = min(
            1.0,
            session_metrics.follow_up_count / self.config.expected_follow_ups
        )
        scores.append(("follow_up", follow_up_score, 0.35))

        # Session depth score
        depth_score = min(
            1.0,
            session_metrics.session_turns / self.config.session_depth_target
        )
        scores.append(("depth", depth_score, 0.25))

        # Duration score (optimal around 10-15 minutes)
        duration_minutes = session_metrics.session_duration_seconds / 60
        optimal_duration = 12  # minutes
        duration_score = self._duration_score(duration_minutes, optimal_duration)
        scores.append(("duration", duration_score, 0.15))

        # Question engagement score
        if session_metrics.session_turns > 0:
            question_rate = session_metrics.questions_asked / session_metrics.session_turns
            question_score = min(1.0, question_rate * 2)  # Scale up
        else:
            question_score = 0.0
        scores.append(("questions", question_score, 0.15))

        # Topic exploration score
        topic_score = min(1.0, session_metrics.topics_explored / 3)
        scores.append(("topics", topic_score, 0.10))

        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in scores)

        db_logger.debug(
            "engagement_score_calculated",
            follow_up_count=session_metrics.follow_up_count,
            session_turns=session_metrics.session_turns,
            duration_minutes=round(duration_minutes, 2),
            total_score=round(total_score, 4)
        )

        return max(0.0, min(1.0, total_score))

    def _duration_score(
        self,
        actual_minutes: float,
        optimal_minutes: float
    ) -> float:
        """
        Calculate duration score using Gaussian curve

        ALGORITHM: Gaussian centered on optimal duration
        - Exact optimal = 1.0
        - Too short or too long reduces score
        """
        # Gaussian with sigma = optimal_minutes / 2
        sigma = optimal_minutes / 2
        exponent = -((actual_minutes - optimal_minutes) ** 2) / (2 * sigma ** 2)

        return math.exp(exponent)

    async def calculate_from_history(
        self,
        session_history: List[Dict],
        start_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate engagement from session history

        Args:
            session_history: List of conversation exchanges
            start_time: Session start timestamp

        Returns:
            Engagement score (0.0 to 1.0)
        """
        if not session_history:
            return self.config.default_score

        metrics = SessionMetrics()
        metrics.session_turns = len(session_history)

        # Count follow-ups (responses that reference previous content)
        for i in range(1, len(session_history)):
            current = session_history[i].get("user_text", "").lower()
            previous = session_history[i-1].get("agent_text", "").lower()

            # Check for follow-up indicators
            follow_up_indicators = [
                "more about", "tell me more", "what about", "how about",
                "can you explain", "why", "how does", "what if",
                "continue", "go on", "elaborate", "expand"
            ]

            if any(ind in current for ind in follow_up_indicators):
                metrics.follow_up_count += 1

        # Count questions
        for exchange in session_history:
            user_text = exchange.get("user_text", "")
            if "?" in user_text:
                metrics.questions_asked += 1

        # Calculate duration if start_time provided
        if start_time and session_history:
            last_timestamp = session_history[-1].get("timestamp")
            if last_timestamp:
                if isinstance(last_timestamp, str):
                    last_timestamp = datetime.fromisoformat(last_timestamp)
                metrics.session_duration_seconds = (
                    last_timestamp - start_time
                ).total_seconds()

        return await self.calculate(metrics, len(session_history))


# ============================================================================
# CLARITY SCORING
# ============================================================================

class ClarityScorer:
    """
    CONCEPT: Measure response readability and structure

    METRICS:
    - Flesch-Kincaid readability score
    - Sentence structure analysis
    - Response organization (headers, lists)
    - Word complexity analysis

    WHY: Clear responses are more useful to users
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or scoring_config

    def calculate(self, text: str) -> float:
        """
        Calculate clarity score

        Args:
            text: Response text to analyze

        Returns:
            Clarity score (0.0 to 1.0)
        """
        if not text or len(text) < self.config.min_text_length:
            return self.config.default_score

        scores = []

        # Flesch-Kincaid readability (normalized)
        fk_score = self._flesch_kincaid_grade(text)
        fk_normalized = self._normalize_grade_level(fk_score)
        scores.append(("readability", fk_normalized, 0.40))

        # Sentence structure score
        sentence_score = self._sentence_structure_score(text)
        scores.append(("sentences", sentence_score, 0.25))

        # Organization score (structure analysis)
        organization_score = self._organization_score(text)
        scores.append(("organization", organization_score, 0.20))

        # Word complexity score
        complexity_score = self._word_complexity_score(text)
        scores.append(("complexity", complexity_score, 0.15))

        # Calculate weighted total
        total_score = sum(score * weight for _, score, weight in scores)

        db_logger.debug(
            "clarity_score_calculated",
            text_length=len(text),
            fk_grade=round(fk_score, 2),
            total_score=round(total_score, 4)
        )

        return max(0.0, min(1.0, total_score))

    def _flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level

        FORMULA:
        FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

        Returns grade level (0-18+)
        """
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 12.0  # Default to 12th grade

        grade = (
            0.39 * (words / sentences) +
            11.8 * (syllables / words) -
            15.59
        )

        return max(0, min(18, grade))

    def _normalize_grade_level(self, grade: float) -> float:
        """
        Normalize grade level to 0-1 score

        ALGORITHM:
        - Target is 8th grade level (most accessible)
        - Score decreases for both simpler and more complex text
        - Uses Gaussian centered on target
        """
        target = self.config.target_grade_level
        sigma = 4  # Tolerance around target

        deviation = abs(grade - target)
        score = math.exp(-(deviation ** 2) / (2 * sigma ** 2))

        return score

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        # Match sentence-ending punctuation
        endings = re.findall(r'[.!?]+', text)
        return max(1, len(endings))

    def _count_words(self, text: str) -> int:
        """Count words in text"""
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    def _count_syllables(self, text: str) -> int:
        """
        Count syllables in text

        ALGORITHM: Vowel counting with adjustments
        - Count vowel groups
        - Subtract silent e's
        - Ensure at least 1 syllable per word
        """
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)

        total_syllables = 0
        for word in words:
            syllables = self._count_word_syllables(word)
            total_syllables += syllables

        return max(1, total_syllables)

    def _count_word_syllables(self, word: str) -> int:
        """Count syllables in a single word"""
        word = word.lower()

        # Handle special cases
        if len(word) <= 3:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1

        # Adjust for -le ending
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(1, count)

    def _sentence_structure_score(self, text: str) -> float:
        """
        Analyze sentence structure quality

        FACTORS:
        - Sentence length variation
        - Average sentence length
        - Proper punctuation
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)

        # Score based on average length (optimal: 15-20 words)
        if 15 <= avg_length <= 20:
            length_score = 1.0
        elif 10 <= avg_length <= 25:
            length_score = 0.8
        elif 5 <= avg_length <= 30:
            length_score = 0.6
        else:
            length_score = 0.4

        # Variation score (some variation is good)
        if len(lengths) > 1:
            std_dev = (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5
            variation_ratio = std_dev / avg_length if avg_length > 0 else 0

            # Optimal variation is 0.3-0.5
            if 0.3 <= variation_ratio <= 0.5:
                variation_score = 1.0
            elif 0.2 <= variation_ratio <= 0.6:
                variation_score = 0.8
            else:
                variation_score = 0.6
        else:
            variation_score = 0.5

        return (length_score * 0.6 + variation_score * 0.4)

    def _organization_score(self, text: str) -> float:
        """
        Analyze response organization

        FACTORS:
        - Presence of structure (lists, headers)
        - Logical markers (first, second, finally)
        - Paragraph breaks
        """
        score = 0.5  # Base score

        # Check for structural elements
        structural_patterns = [
            (r'^\s*[-*]\s', 0.1),      # Bullet points
            (r'^\s*\d+\.\s', 0.1),     # Numbered lists
            (r'^#+\s', 0.1),            # Headers (markdown)
            (r'\n\n', 0.1),             # Paragraph breaks
        ]

        for pattern, bonus in structural_patterns:
            if re.search(pattern, text, re.MULTILINE):
                score += bonus

        # Check for logical markers
        logical_markers = [
            "first", "second", "third", "finally", "additionally",
            "moreover", "however", "therefore", "in conclusion",
            "for example", "specifically", "in summary"
        ]

        text_lower = text.lower()
        marker_count = sum(1 for marker in logical_markers if marker in text_lower)
        score += min(0.2, marker_count * 0.05)

        return min(1.0, score)

    def _word_complexity_score(self, text: str) -> float:
        """
        Analyze word complexity

        FACTORS:
        - Average word length
        - Proportion of complex words (3+ syllables)
        """
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            return 0.5

        # Average word length (optimal: 4-6 characters)
        avg_length = sum(len(w) for w in words) / len(words)

        if 4 <= avg_length <= 6:
            length_score = 1.0
        elif 3 <= avg_length <= 7:
            length_score = 0.8
        else:
            length_score = 0.6

        # Complex word ratio (3+ syllables)
        complex_words = sum(
            1 for w in words if self._count_word_syllables(w) >= 3
        )
        complex_ratio = complex_words / len(words)

        # Optimal complex ratio: 10-20%
        if 0.1 <= complex_ratio <= 0.2:
            complexity_score = 1.0
        elif 0.05 <= complex_ratio <= 0.3:
            complexity_score = 0.8
        else:
            complexity_score = 0.6

        return (length_score * 0.5 + complexity_score * 0.5)


# ============================================================================
# HELPFULNESS SCORING
# ============================================================================

@dataclass
class FeedbackData:
    """User feedback data for helpfulness calculation"""
    rating: float  # 0-5 scale
    timestamp: datetime
    session_id: str
    response_id: Optional[str] = None
    feedback_type: str = "explicit"  # explicit, implicit
    weight: float = 1.0


class HelpfulnessScorer:
    """
    CONCEPT: Calculate helpfulness from user feedback

    ALGORITHM:
    1. Gather explicit feedback (ratings, thumbs up/down)
    2. Weight by recency (recent feedback more important)
    3. Normalize to 0-1 scale
    4. Handle missing feedback gracefully

    WHY: Direct user feedback is the most reliable quality signal
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or scoring_config

    async def calculate(
        self,
        feedback_data: List[FeedbackData],
        default_score: Optional[float] = None
    ) -> float:
        """
        Calculate helpfulness score from feedback

        Args:
            feedback_data: List of feedback entries
            default_score: Score to use when no feedback available

        Returns:
            Helpfulness score (0.0 to 1.0)
        """
        default = default_score or self.config.default_score

        if not feedback_data:
            return default

        # Apply recency weighting
        now = datetime.utcnow()
        weighted_scores = []

        for feedback in feedback_data:
            # Calculate age in days
            age_days = (now - feedback.timestamp).total_seconds() / 86400

            # Exponential decay for recency
            recency_weight = math.exp(
                -age_days * (1 - self.config.feedback_recency_weight)
            )

            # Normalize rating to 0-1
            normalized_rating = feedback.rating / 5.0

            # Combined weight
            total_weight = recency_weight * feedback.weight

            weighted_scores.append((normalized_rating, total_weight))

        # Calculate weighted average
        total_weight = sum(w for _, w in weighted_scores)

        if total_weight == 0:
            return default

        score = sum(s * w for s, w in weighted_scores) / total_weight

        db_logger.debug(
            "helpfulness_score_calculated",
            feedback_count=len(feedback_data),
            weighted_score=round(score, 4)
        )

        return max(0.0, min(1.0, score))

    async def calculate_from_ratings(
        self,
        ratings: List[Tuple[float, datetime]],
        session_id: str = ""
    ) -> float:
        """
        Convenience method to calculate from simple rating tuples

        Args:
            ratings: List of (rating, timestamp) tuples
            session_id: Session identifier

        Returns:
            Helpfulness score (0.0 to 1.0)
        """
        feedback_data = [
            FeedbackData(
                rating=rating,
                timestamp=timestamp,
                session_id=session_id
            )
            for rating, timestamp in ratings
        ]

        return await self.calculate(feedback_data)


# ============================================================================
# ACCURACY SCORING
# ============================================================================

class AccuracyScorer:
    """
    CONCEPT: Verify response accuracy through self-consistency

    ALGORITHM:
    1. Check internal consistency (no contradictions)
    2. Verify against known facts (if available)
    3. Assess confidence indicators
    4. Check source citations (if present)

    WHY: Accurate information is critical for trust

    NOTE: Full accuracy verification requires external knowledge
    base or LLM-based fact checking. This implementation provides
    heuristic-based scoring.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or scoring_config

    async def calculate(
        self,
        response: str,
        query: str,
        context: Optional[List[Dict]] = None,
        known_facts: Optional[List[str]] = None
    ) -> float:
        """
        Calculate accuracy score

        Args:
            response: Agent response text
            query: Original user query
            context: Conversation context
            known_facts: Optional list of verified facts

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not response:
            return self.config.default_score

        scores = []

        # Self-consistency check
        consistency_score = self._check_consistency(response)
        scores.append(("consistency", consistency_score, 0.35))

        # Confidence indicator check
        confidence_score = self._check_confidence_indicators(response)
        scores.append(("confidence", confidence_score, 0.25))

        # Citation/source check
        citation_score = self._check_citations(response)
        scores.append(("citations", citation_score, 0.20))

        # Hedge word analysis (uncertainty markers)
        hedging_score = self._analyze_hedging(response)
        scores.append(("hedging", hedging_score, 0.20))

        # Calculate weighted total
        total_score = sum(score * weight for _, score, weight in scores)

        db_logger.debug(
            "accuracy_score_calculated",
            response_length=len(response),
            consistency=round(consistency_score, 4),
            total_score=round(total_score, 4)
        )

        return max(0.0, min(1.0, total_score))

    def _check_consistency(self, text: str) -> float:
        """
        Check for internal contradictions

        HEURISTIC: Look for negation patterns that might indicate
        contradictory statements
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return self.config.default_score

        contradiction_patterns = [
            (r'\bis\b', r'\bis not\b'),
            (r'\bwill\b', r'\bwill not\b'),
            (r'\bcan\b', r'\bcannot\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\ball\b', r'\bnone\b'),
            (r'\btrue\b', r'\bfalse\b'),
        ]

        contradiction_count = 0

        for pattern_a, pattern_b in contradiction_patterns:
            has_a = any(re.search(pattern_a, s) for s in sentences)
            has_b = any(re.search(pattern_b, s) for s in sentences)

            if has_a and has_b:
                contradiction_count += 1

        # Score based on contradiction count
        if contradiction_count == 0:
            return 1.0
        elif contradiction_count == 1:
            return 0.7
        else:
            return max(0.3, 1.0 - contradiction_count * 0.15)

    def _check_confidence_indicators(self, text: str) -> float:
        """
        Check for appropriate confidence indicators

        GOOD: Statements with proper confidence framing
        BAD: Overconfident claims without support
        """
        text_lower = text.lower()

        # Positive indicators (show appropriate caution)
        appropriate_caution = [
            "based on", "according to", "research shows",
            "evidence suggests", "in my understanding",
            "typically", "generally", "often"
        ]

        # Negative indicators (overconfidence)
        overconfident = [
            "definitely", "absolutely", "without a doubt",
            "100%", "guaranteed", "always works",
            "never fails", "the only way"
        ]

        caution_count = sum(1 for phrase in appropriate_caution if phrase in text_lower)
        overconfident_count = sum(1 for phrase in overconfident if phrase in text_lower)

        # Score based on balance
        base_score = 0.7
        base_score += min(0.2, caution_count * 0.05)
        base_score -= min(0.3, overconfident_count * 0.1)

        return max(0.3, min(1.0, base_score))

    def _check_citations(self, text: str) -> float:
        """
        Check for source citations or references
        """
        citation_patterns = [
            r'\[\d+\]',                    # [1], [2] style
            r'\([A-Z][a-z]+,?\s*\d{4}\)',  # (Author, 2023)
            r'according to',
            r'source:',
            r'reference:',
            r'https?://',                   # URLs
        ]

        citation_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in citation_patterns
        )

        if citation_count == 0:
            return 0.5  # Neutral - not all responses need citations
        elif citation_count <= 2:
            return 0.7
        else:
            return 0.9

    def _analyze_hedging(self, text: str) -> float:
        """
        Analyze use of hedge words (uncertainty markers)

        BALANCE: Some hedging is good (honesty), too much is bad (uncertainty)
        """
        text_lower = text.lower()
        word_count = len(text.split())

        hedge_words = [
            "maybe", "perhaps", "possibly", "might",
            "could", "may", "probably", "likely",
            "seems", "appears", "suggests", "indicates"
        ]

        hedge_count = sum(
            text_lower.count(word) for word in hedge_words
        )

        if word_count == 0:
            return 0.5

        hedge_ratio = hedge_count / word_count

        # Optimal hedge ratio: 1-3%
        if 0.01 <= hedge_ratio <= 0.03:
            return 1.0
        elif hedge_ratio < 0.01:
            return 0.7  # Maybe overconfident
        elif hedge_ratio <= 0.05:
            return 0.8
        else:
            return max(0.4, 1.0 - hedge_ratio * 10)  # Too uncertain


# ============================================================================
# COMPOSITE SCORE CALCULATOR
# ============================================================================

class CompositeScoreCalculator:
    """
    CONCEPT: Combine individual dimension scores into composite

    ALGORITHM:
    - Weighted average of dimension scores
    - Optional adjustments for specific scenarios
    - Support for custom weight configurations

    WHY: Single score for easy comparison and trending
    """

    DEFAULT_WEIGHTS = {
        "relevance": 0.30,
        "helpfulness": 0.25,
        "engagement": 0.20,
        "clarity": 0.15,
        "accuracy": 0.10
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure weights sum to 1.0"""
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def calculate(
        self,
        relevance: float,
        helpfulness: float,
        engagement: float,
        clarity: float,
        accuracy: float
    ) -> float:
        """
        Calculate composite score

        Args:
            relevance: Relevance score (0-1)
            helpfulness: Helpfulness score (0-1)
            engagement: Engagement score (0-1)
            clarity: Clarity score (0-1)
            accuracy: Accuracy score (0-1)

        Returns:
            Composite score (0-1)
        """
        scores = {
            "relevance": relevance,
            "helpfulness": helpfulness,
            "engagement": engagement,
            "clarity": clarity,
            "accuracy": accuracy
        }

        composite = sum(
            scores[dim] * self.weights[dim]
            for dim in self.weights
        )

        return round(max(0.0, min(1.0, composite)), 4)

    def calculate_with_breakdown(
        self,
        relevance: float,
        helpfulness: float,
        engagement: float,
        clarity: float,
        accuracy: float
    ) -> Dict[str, Any]:
        """
        Calculate composite with detailed breakdown

        Returns dictionary with:
        - composite: Final score
        - breakdown: Per-dimension contribution
        - dominant: Highest contributing dimension
        - weakest: Lowest contributing dimension
        """
        scores = {
            "relevance": relevance,
            "helpfulness": helpfulness,
            "engagement": engagement,
            "clarity": clarity,
            "accuracy": accuracy
        }

        breakdown = {}
        for dim, score in scores.items():
            weight = self.weights[dim]
            contribution = score * weight
            breakdown[dim] = {
                "score": round(score, 4),
                "weight": weight,
                "contribution": round(contribution, 4)
            }

        composite = sum(b["contribution"] for b in breakdown.values())

        # Find dominant and weakest dimensions
        sorted_dims = sorted(
            breakdown.items(),
            key=lambda x: x[1]["contribution"],
            reverse=True
        )

        return {
            "composite": round(composite, 4),
            "breakdown": breakdown,
            "dominant": sorted_dims[0][0],
            "weakest": sorted_dims[-1][0]
        }
