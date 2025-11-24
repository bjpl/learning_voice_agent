"""
Quality Scorer Engine - SPARC Implementation
=============================================

Comprehensive multi-dimensional quality scoring for conversation responses.

SPECIFICATION:
- Score individual responses across 5 quality dimensions
- Aggregate scores at session level
- Track quality trends over time
- Identify areas needing improvement

ARCHITECTURE:
- Facade pattern combining all scoring algorithms
- Async-first design for non-blocking operations
- Integration with existing embedding and feedback systems
- Persistent score storage for trending

PATTERN: Orchestrator for quality scoring subsystem
WHY: Single entry point simplifies API and enables cross-cutting concerns
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

from app.learning.config import LearningConfig, learning_config
from app.learning.models import QualityScore as LegacyQualityScore, QualityDimension, Feedback, FeedbackType
from app.learning.feedback_store import FeedbackStore
from app.learning.scoring_models import (
    QualityScore,
    SessionQuality,
    QualityTrend,
    ImprovementArea,
    QualityLevel,
    ScoreDimension
)
from app.learning.scoring_algorithms import (
    RelevanceScorer,
    EngagementScorer,
    ClarityScorer,
    HelpfulnessScorer,
    AccuracyScorer,
    CompositeScoreCalculator,
    ScoringConfig,
    SessionMetrics,
    FeedbackData,
    scoring_config
)

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Evaluates response quality across multiple dimensions:
    - Relevance: How well the response addresses the query
    - Helpfulness: How useful the response is
    - Engagement: How engaging the response is
    - Clarity: How clear and understandable the response is
    - Accuracy: How accurate the information is

    Uses a combination of:
    - Semantic similarity (embeddings)
    - Heuristic analysis
    - Historical feedback data
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        embedding_generator: Optional[Any] = None,
        feedback_store: Optional[FeedbackStore] = None
    ):
        """
        Initialize the quality scorer.

        Args:
            config: Learning configuration
            embedding_generator: Optional embedding generator for semantic analysis
            feedback_store: Feedback store for historical data
        """
        self.config = config or learning_config
        self.embedding_generator = embedding_generator
        self.feedback_store = feedback_store

        # Dimension weights from config
        self.weights = {
            QualityDimension.RELEVANCE: self.config.quality_scoring.relevance_weight,
            QualityDimension.HELPFULNESS: self.config.quality_scoring.helpfulness_weight,
            QualityDimension.ENGAGEMENT: self.config.quality_scoring.engagement_weight,
            QualityDimension.CLARITY: self.config.quality_scoring.clarity_weight,
            QualityDimension.ACCURACY: self.config.quality_scoring.accuracy_weight,
        }

    async def score_response(
        self,
        query: str,
        response: str,
        session_id: str,
        context: Optional[List[Dict[str, str]]] = None,
        feedback_history: Optional[List[Feedback]] = None
    ) -> QualityScore:
        """
        Score a response across all quality dimensions.

        Args:
            query: The user's query
            response: The system's response
            session_id: Session identifier
            context: Optional conversation context
            feedback_history: Optional historical feedback

        Returns:
            QualityScore with all dimension scores
        """
        # Score each dimension
        relevance = await self._score_relevance(query, response, context)
        helpfulness = await self._score_helpfulness(query, response)
        engagement = await self._score_engagement(response, feedback_history)
        clarity = await self._score_clarity(response)
        accuracy = await self._score_accuracy(query, response, context)

        # Calculate composite score
        composite = self._calculate_composite(
            relevance, helpfulness, engagement, clarity, accuracy
        )

        # Generate unique query ID
        query_id = f"{session_id}_{datetime.utcnow().timestamp()}"

        score = QualityScore(
            query_id=query_id,
            session_id=session_id,
            relevance=relevance,
            helpfulness=helpfulness,
            engagement=engagement,
            clarity=clarity,
            accuracy=accuracy,
            composite=composite,
            query_text=query,
            response_text=response
        )

        logger.debug(
            f"Scored response: composite={composite:.3f}, "
            f"relevance={relevance:.3f}, helpfulness={helpfulness:.3f}"
        )

        return score

    async def _score_relevance(
        self,
        query: str,
        response: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> float:
        """
        Score response relevance to the query.

        Uses semantic similarity and keyword overlap.
        """
        # If we have an embedding generator, use semantic similarity
        if self.embedding_generator:
            try:
                query_embedding = await self.embedding_generator.generate(query)
                response_embedding = await self.embedding_generator.generate(response)

                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, response_embedding)

                # Adjust similarity to score range
                semantic_score = (similarity + 1) / 2  # Convert [-1, 1] to [0, 1]
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                semantic_score = 0.5
        else:
            semantic_score = 0.5

        # Calculate keyword overlap
        keyword_score = self._calculate_keyword_overlap(query, response)

        # Combine scores
        relevance = 0.6 * semantic_score + 0.4 * keyword_score

        return min(1.0, max(0.0, relevance))

    async def _score_helpfulness(
        self,
        query: str,
        response: str
    ) -> float:
        """
        Score how helpful the response is.

        Considers:
        - Response completeness
        - Actionable information
        - Addresses the intent
        """
        score = 0.5  # Base score

        # Check for question answering
        if query.endswith("?"):
            # Response should provide an answer
            if not self._is_just_a_question(response):
                score += 0.2

        # Check for actionable content
        actionable_indicators = [
            "you can", "try", "here's how", "steps:", "first,",
            "to do this", "solution", "answer", "result"
        ]
        response_lower = response.lower()
        if any(indicator in response_lower for indicator in actionable_indicators):
            score += 0.15

        # Check for explanations
        explanation_indicators = [
            "because", "this is because", "the reason", "this means",
            "in other words", "for example", "specifically"
        ]
        if any(indicator in response_lower for indicator in explanation_indicators):
            score += 0.1

        # Penalize very short responses
        word_count = len(response.split())
        if word_count < 5:
            score -= 0.2
        elif word_count < 15:
            score -= 0.1

        # Penalize overly long responses
        if word_count > 500:
            score -= 0.1

        return min(1.0, max(0.0, score))

    async def _score_engagement(
        self,
        response: str,
        feedback_history: Optional[List[Feedback]] = None
    ) -> float:
        """
        Score how engaging the response is.

        Uses feedback history if available.
        """
        score = 0.5  # Base score

        # Use feedback history if available
        if feedback_history:
            positive_count = sum(
                1 for fb in feedback_history
                if fb.feedback_type in (
                    FeedbackType.EXPLICIT_POSITIVE,
                    FeedbackType.IMPLICIT_FOLLOW_UP
                )
            )
            negative_count = sum(
                1 for fb in feedback_history
                if fb.feedback_type in (
                    FeedbackType.EXPLICIT_NEGATIVE,
                    FeedbackType.IMPLICIT_ABANDONMENT
                )
            )

            total = positive_count + negative_count
            if total > 0:
                score = 0.3 + 0.7 * (positive_count / total)
        else:
            # Heuristic scoring based on response characteristics
            response_lower = response.lower()

            # Engaging language
            engaging_indicators = [
                "interesting", "great question", "let me explain",
                "you might be wondering", "consider", "imagine"
            ]
            if any(indicator in response_lower for indicator in engaging_indicators):
                score += 0.1

            # Questions that invite follow-up
            if response.endswith("?") or "what do you think" in response_lower:
                score += 0.15

            # Code or examples
            if "```" in response or "example:" in response_lower:
                score += 0.1

        return min(1.0, max(0.0, score))

    async def _score_clarity(self, response: str) -> float:
        """
        Score how clear and understandable the response is.

        Considers readability metrics.
        """
        score = 0.5  # Base score

        words = response.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        sentence_count = max(1, len(sentences))

        # Average sentence length
        avg_sentence_length = word_count / sentence_count

        # Ideal sentence length is 15-20 words
        if 10 <= avg_sentence_length <= 25:
            score += 0.2
        elif avg_sentence_length > 40:
            score -= 0.2

        # Check for structure (lists, headers, etc.)
        if "\n-" in response or "\n*" in response or "\n1." in response:
            score += 0.15

        # Check for paragraph breaks
        if "\n\n" in response:
            score += 0.1

        # Penalize overly complex words (approximation)
        long_words = [w for w in words if len(w) > 12]
        if len(long_words) / max(1, word_count) > 0.2:
            score -= 0.15

        return min(1.0, max(0.0, score))

    async def _score_accuracy(
        self,
        query: str,
        response: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> float:
        """
        Score the accuracy of the response.

        This is a heuristic approximation since true accuracy
        requires ground truth data.
        """
        score = 0.6  # Default to assuming reasonable accuracy

        response_lower = response.lower()

        # Check for hedging language (uncertainty indicators)
        uncertain_indicators = [
            "i'm not sure", "i don't know", "might be", "could be",
            "possibly", "perhaps", "i think", "may or may not"
        ]
        hedging_count = sum(
            1 for indicator in uncertain_indicators
            if indicator in response_lower
        )
        if hedging_count > 2:
            score -= 0.1

        # Check for confidence indicators
        confident_indicators = [
            "definitely", "certainly", "absolutely", "is",
            "the answer is", "this is because"
        ]
        if any(indicator in response_lower for indicator in confident_indicators):
            score += 0.1

        # Check for source citations
        if "according to" in response_lower or "source:" in response_lower:
            score += 0.15

        # Check for contradictions with context
        if context:
            # Simple check: if response contradicts recent context
            for ctx in context[-3:]:
                if self._contains_contradiction(response, ctx.get("agent", "")):
                    score -= 0.2
                    break

        return min(1.0, max(0.0, score))

    def _calculate_composite(
        self,
        relevance: float,
        helpfulness: float,
        engagement: float,
        clarity: float,
        accuracy: float
    ) -> float:
        """Calculate the weighted composite score."""
        weighted_sum = (
            self.weights[QualityDimension.RELEVANCE] * relevance +
            self.weights[QualityDimension.HELPFULNESS] * helpfulness +
            self.weights[QualityDimension.ENGAGEMENT] * engagement +
            self.weights[QualityDimension.CLARITY] * clarity +
            self.weights[QualityDimension.ACCURACY] * accuracy
        )

        total_weight = sum(self.weights.values())
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _calculate_keyword_overlap(self, query: str, response: str) -> float:
        """Calculate keyword overlap between query and response."""
        # Simple word-based overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am"
        }

        query_words -= stop_words
        response_words -= stop_words

        if not query_words:
            return 0.5

        overlap = query_words & response_words
        return len(overlap) / len(query_words)

    def _is_just_a_question(self, text: str) -> bool:
        """Check if the text is just a question without an answer."""
        sentences = text.split(".")
        if len(sentences) <= 1 and text.strip().endswith("?"):
            return True
        return False

    def _contains_contradiction(self, response: str, context: str) -> bool:
        """Check if response contradicts context (simple heuristic)."""
        # Very basic contradiction detection
        response_lower = response.lower()
        context_lower = context.lower()

        # Check for explicit negation patterns
        negation_pairs = [
            ("is true", "is false"),
            ("is correct", "is incorrect"),
            ("yes", "no"),
        ]

        for pos, neg in negation_pairs:
            if (pos in context_lower and neg in response_lower) or \
               (neg in context_lower and pos in response_lower):
                return True

        return False

    async def batch_score(
        self,
        interactions: List[Dict[str, str]],
        session_id: str
    ) -> List[QualityScore]:
        """
        Score multiple interactions in batch.

        Args:
            interactions: List of {"query": str, "response": str} dicts
            session_id: Session identifier

        Returns:
            List of QualityScore objects
        """
        scores = []
        for interaction in interactions:
            score = await self.score_response(
                query=interaction["query"],
                response=interaction["response"],
                session_id=session_id
            )
            scores.append(score)

        return scores

    def get_dimension_weights(self) -> Dict[QualityDimension, float]:
        """Get the current dimension weights."""
        return self.weights.copy()

    def set_dimension_weights(self, weights: Dict[QualityDimension, float]) -> None:
        """Set custom dimension weights."""
        for dim, weight in weights.items():
            if dim in self.weights:
                self.weights[dim] = weight

        logger.info(f"Updated quality dimension weights: {self.weights}")

    # ========================================================================
    # ENHANCED SCORING METHODS (Phase 5)
    # ========================================================================

    async def score_response_enhanced(
        self,
        query: str,
        response: str,
        session_id: str,
        context: Optional[List[Dict[str, str]]] = None,
        response_id: Optional[str] = None
    ) -> QualityScore:
        """
        Enhanced scoring using dedicated scoring algorithms.

        ALGORITHM:
        1. Calculate relevance (embedding similarity)
        2. Calculate helpfulness (from feedback)
        3. Calculate engagement (session metrics)
        4. Calculate clarity (readability)
        5. Calculate accuracy (self-consistency)
        6. Compute weighted composite

        Args:
            query: User's input query
            response: Agent's response text
            session_id: Session identifier
            context: Optional conversation context
            response_id: Optional unique response identifier

        Returns:
            QualityScore with all dimensions and composite
        """
        start_time = datetime.utcnow()

        try:
            # Initialize scorers if needed
            relevance_scorer = RelevanceScorer(
                embedding_provider=self.embedding_generator,
                config=scoring_config
            )
            clarity_scorer = ClarityScorer(config=scoring_config)
            accuracy_scorer = AccuracyScorer(config=scoring_config)
            engagement_scorer = EngagementScorer(config=scoring_config)
            helpfulness_scorer = HelpfulnessScorer(config=scoring_config)

            # Score all dimensions concurrently
            relevance_task = relevance_scorer.calculate(query, response, context)
            accuracy_task = accuracy_scorer.calculate(response, query, context)

            # Clarity is sync
            clarity = clarity_scorer.calculate(response)

            # Get engagement metrics
            session_metrics = self._get_session_metrics(session_id)
            engagement_task = engagement_scorer.calculate(session_metrics)

            # Get helpfulness from feedback
            feedback_data = await self._get_feedback_data(session_id)
            helpfulness_task = helpfulness_scorer.calculate(feedback_data)

            # Await async tasks
            relevance, accuracy, engagement, helpfulness = await asyncio.gather(
                relevance_task,
                accuracy_task,
                engagement_task,
                helpfulness_task,
                return_exceptions=True
            )

            # Handle exceptions
            relevance = relevance if not isinstance(relevance, Exception) else 0.5
            accuracy = accuracy if not isinstance(accuracy, Exception) else 0.5
            engagement = engagement if not isinstance(engagement, Exception) else 0.5
            helpfulness = helpfulness if not isinstance(helpfulness, Exception) else 0.5

            # Calculate composite
            calculator = CompositeScoreCalculator()
            composite = calculator.calculate(
                relevance=relevance,
                helpfulness=helpfulness,
                engagement=engagement,
                clarity=clarity,
                accuracy=accuracy
            )

            # Create quality score
            quality_score = QualityScore(
                relevance=relevance,
                helpfulness=helpfulness,
                engagement=engagement,
                clarity=clarity,
                accuracy=accuracy,
                composite=composite,
                timestamp=datetime.utcnow(),
                response_id=response_id or f"{session_id}_{datetime.utcnow().timestamp()}",
                session_id=session_id,
                query_length=len(query),
                response_length=len(response),
                metadata={
                    "scoring_duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "algorithm_version": "enhanced_v1"
                }
            )

            # Store and update session
            await self._store_score(session_id, quality_score)

            logger.info(
                f"Enhanced response scored: composite={composite:.4f}, "
                f"relevance={relevance:.4f}, helpfulness={helpfulness:.4f}, "
                f"engagement={engagement:.4f}, clarity={clarity:.4f}, accuracy={accuracy:.4f}"
            )

            return quality_score

        except Exception as e:
            logger.error(f"Enhanced scoring failed: {e}")
            # Return default score on failure
            return QualityScore(
                relevance=0.5,
                helpfulness=0.5,
                engagement=0.5,
                clarity=0.5,
                accuracy=0.5,
                session_id=session_id,
                metadata={"error": str(e)}
            )

    async def score_session(self, session_id: str) -> SessionQuality:
        """
        Get aggregated quality metrics for an entire session.

        Args:
            session_id: Session identifier

        Returns:
            SessionQuality with aggregated metrics and trend
        """
        session = SessionQuality(session_id=session_id)

        # Get stored scores for session
        scores = await self._get_session_scores(session_id)

        for score in scores:
            session.add_response_score(score)

        logger.debug(
            f"Session {session_id} scored: "
            f"response_count={session.response_count}, "
            f"avg_composite={session.average_composite:.4f}, "
            f"trend={session.quality_trend}"
        )

        return session

    async def get_quality_trends(
        self,
        time_range: str = "7d"
    ) -> QualityTrend:
        """
        Analyze quality trends over a time period.

        Args:
            time_range: Time range string (e.g., "24h", "7d", "30d")

        Returns:
            QualityTrend with time-series analysis
        """
        # Parse time range
        duration = self._parse_time_range(time_range)
        end_time = datetime.utcnow()
        start_time = end_time - duration

        # Get scores in range
        scores = await self._get_scores_in_range(start_time, end_time)

        # Create trend object
        trend = QualityTrend(
            time_range=time_range,
            start_timestamp=start_time,
            end_timestamp=end_time
        )

        # Add data points
        for score in scores:
            trend.add_data_point(score.timestamp, score)

        # Analyze trends
        trend.analyze()

        logger.info(
            f"Quality trends for {time_range}: "
            f"data_points={len(scores)}, "
            f"overall_trend={trend.overall_trend}, "
            f"mean_score={trend.mean_score:.4f}"
        )

        return trend

    async def identify_improvement_areas(
        self,
        threshold: float = 0.75,
        min_samples: int = 5
    ) -> List[ImprovementArea]:
        """
        Identify quality dimensions needing improvement.

        ALGORITHM:
        1. Aggregate scores across recent sessions
        2. Calculate average for each dimension
        3. Identify dimensions below threshold
        4. Rank by improvement potential

        Args:
            threshold: Target quality score
            min_samples: Minimum samples for reliable analysis

        Returns:
            List of ImprovementArea sorted by priority
        """
        # Get recent scores
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        scores = await self._get_scores_in_range(start_time, end_time)

        if len(scores) < min_samples:
            logger.warning(
                f"Insufficient data for improvement analysis: "
                f"{len(scores)} samples, need {min_samples}"
            )
            return []

        # Calculate dimension averages
        dimension_totals: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))

        for score in scores:
            for dim in ScoreDimension:
                value = getattr(score, dim.value)
                current_total, current_count = dimension_totals[dim.value]
                dimension_totals[dim.value] = (current_total + value, current_count + 1)

        # Create improvement areas for dimensions below threshold
        improvement_areas = []

        for dim_name, (total, count) in dimension_totals.items():
            if count == 0:
                continue

            avg_score = total / count

            if avg_score < threshold:
                # Count affected sessions
                affected_sessions = len(set(
                    s.session_id for s in scores
                    if s.session_id and getattr(s, dim_name) < threshold
                ))

                area = ImprovementArea.from_analysis(
                    dimension=dim_name,
                    current_score=avg_score,
                    target_score=threshold,
                    affected_sessions=affected_sessions
                )
                improvement_areas.append(area)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        improvement_areas.sort(
            key=lambda a: (priority_order[a.priority], -a.improvement_potential)
        )

        logger.info(
            f"Improvement areas identified: {len(improvement_areas)}, "
            f"high_priority={[a.dimension for a in improvement_areas if a.priority == 'high']}"
        )

        return improvement_areas

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary for a session.

        Returns:
            Dictionary with session quality summary
        """
        session = await self.score_session(session_id)

        return {
            "session_id": session_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "duration_seconds": session.session_duration_seconds,
            "response_count": session.response_count,
            "average_scores": {
                "relevance": round(session.average_relevance, 4),
                "helpfulness": round(session.average_helpfulness, 4),
                "engagement": round(session.average_engagement, 4),
                "clarity": round(session.average_clarity, 4),
                "accuracy": round(session.average_accuracy, 4),
                "composite": round(session.average_composite, 4)
            },
            "quality_level": session.overall_quality_level.value,
            "quality_distribution": session.quality_distribution,
            "quality_trend": session.quality_trend,
            "trend_slope": round(session.trend_slope, 4)
        }

    async def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall quality statistics across all sessions.

        Returns:
            Dictionary with aggregate statistics
        """
        # Get all scores from last 30 days
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        scores = await self._get_scores_in_range(start_time, end_time)

        if not scores:
            return {
                "status": "no_data",
                "total_sessions": 0,
                "total_responses": 0
            }

        # Group by session
        sessions = defaultdict(list)
        for score in scores:
            if score.session_id:
                sessions[score.session_id].append(score)

        # Calculate statistics
        total_responses = len(scores)
        avg_composite = sum(s.composite for s in scores) / len(scores)

        # Quality level distribution
        level_counts = defaultdict(int)
        for score in scores:
            level_counts[score.quality_level.value] += 1

        return {
            "total_sessions": len(sessions),
            "total_responses": total_responses,
            "average_composite": round(avg_composite, 4),
            "quality_level_distribution": dict(level_counts),
            "time_range_days": 30
        }

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _get_session_metrics(self, session_id: str) -> SessionMetrics:
        """Get or create session metrics."""
        if not hasattr(self, '_session_metrics'):
            self._session_metrics = {}

        if session_id not in self._session_metrics:
            self._session_metrics[session_id] = SessionMetrics()

        return self._session_metrics[session_id]

    async def _get_feedback_data(self, session_id: str) -> List[FeedbackData]:
        """Convert stored feedback to FeedbackData format."""
        if not self.feedback_store:
            return []

        try:
            # Get feedback from store
            feedback_list = await self.feedback_store.get_feedback_by_session(session_id)

            return [
                FeedbackData(
                    rating=fb.rating if hasattr(fb, 'rating') else 3.0,
                    timestamp=fb.timestamp if hasattr(fb, 'timestamp') else datetime.utcnow(),
                    session_id=session_id,
                    feedback_type="explicit"
                )
                for fb in feedback_list
            ]
        except Exception as e:
            logger.warning(f"Failed to get feedback data: {e}")
            return []

    async def _store_score(self, session_id: str, score: QualityScore) -> None:
        """Store quality score for later retrieval."""
        if not hasattr(self, '_stored_scores'):
            self._stored_scores = defaultdict(list)

        self._stored_scores[session_id].append(score)

    async def _get_session_scores(self, session_id: str) -> List[QualityScore]:
        """Get all stored scores for a session."""
        if not hasattr(self, '_stored_scores'):
            self._stored_scores = defaultdict(list)

        return self._stored_scores.get(session_id, [])

    async def _get_scores_in_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[QualityScore]:
        """Get all scores within a time range."""
        if not hasattr(self, '_stored_scores'):
            self._stored_scores = defaultdict(list)

        all_scores = []
        for session_scores in self._stored_scores.values():
            for score in session_scores:
                if start_time <= score.timestamp <= end_time:
                    all_scores.append(score)

        return sorted(all_scores, key=lambda s: s.timestamp)

    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        unit = time_range[-1].lower()
        value = int(time_range[:-1])

        if unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "w":
            return timedelta(weeks=value)
        elif unit == "m":
            return timedelta(days=value * 30)
        else:
            raise ValueError(f"Invalid time range format: {time_range}")


# ============================================================================
# QUALITY SCORING MIDDLEWARE
# ============================================================================

class QualityScoringMiddleware:
    """
    CONCEPT: Middleware for automatic quality scoring

    PURPOSE:
    - Automatically score all responses
    - Non-blocking background processing
    - Integration with existing conversation flow

    USAGE:
        middleware = QualityScoringMiddleware(scorer)

        # In your conversation handler:
        response = await generate_response(query)
        await middleware.score_async(session_id, query, response)
    """

    def __init__(
        self,
        scorer: QualityScorer,
        async_scoring: bool = True
    ):
        self.scorer = scorer
        self.async_scoring = async_scoring
        self._pending_tasks: List[asyncio.Task] = []

    async def score_async(
        self,
        session_id: str,
        query: str,
        response: str,
        context: Optional[List[Dict]] = None
    ) -> Optional[QualityScore]:
        """
        Score response asynchronously (non-blocking).

        Returns:
            QualityScore if sync, None if async
        """
        if self.async_scoring:
            task = asyncio.create_task(
                self.scorer.score_response_enhanced(
                    query=query,
                    response=response,
                    session_id=session_id,
                    context=context
                )
            )
            self._pending_tasks.append(task)
            self._cleanup_completed_tasks()
            return None
        else:
            return await self.scorer.score_response_enhanced(
                query=query,
                response=response,
                session_id=session_id,
                context=context
            )

    def _cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from pending list."""
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

    async def wait_for_pending(self) -> None:
        """Wait for all pending scoring tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks = []


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_quality_scorer(
    config: Optional[LearningConfig] = None,
    embedding_generator: Optional[Any] = None,
    feedback_store: Optional[FeedbackStore] = None
) -> QualityScorer:
    """
    Factory function for creating a configured QualityScorer.

    Args:
        config: Learning configuration
        embedding_generator: Optional embedding provider
        feedback_store: Optional feedback storage

    Returns:
        Configured QualityScorer instance
    """
    return QualityScorer(
        config=config,
        embedding_generator=embedding_generator,
        feedback_store=feedback_store
    )


async def quick_score(
    query: str,
    response: str,
    session_id: str = "default"
) -> QualityScore:
    """
    Convenience function for quick one-off scoring.

    NOTE: Creates new scorer instance - not for production use

    Args:
        query: User query
        response: Agent response
        session_id: Optional session identifier

    Returns:
        QualityScore for the response
    """
    scorer = create_quality_scorer()
    return await scorer.score_response_enhanced(query, response, session_id)
