"""
Improvement Engine - Continuous Improvement through A/B Testing

SPECIFICATION:
- Identifies weak areas in response quality
- Generates improvement hypotheses
- Runs A/B tests to validate improvements
- Measures impact and auto-rollbacks if quality drops
- Maintains improvement history for learning

PATTERN: Hypothesis-driven experimentation
WHY: Data-driven improvements with statistical validation

INTEGRATION:
- Runs as background task
- Coordinates with ResponseAdapter for experiment application
- Uses PreferenceLearner data for weak area identification
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import asyncio
import random
import uuid
import math

from app.learning.config import learning_config, ImprovementConfig, ABTestingConfig
from app.learning.store import LearningStore, learning_store, ImprovementRecord
from app.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ImprovementStatus(str, Enum):
    """Status of an improvement experiment"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class ImprovementDimension(str, Enum):
    """Dimensions that can be improved"""
    RESPONSE_LENGTH = "response_length"
    TECHNICAL_DEPTH = "technical_depth"
    COMMUNICATION_STYLE = "communication_style"
    VOCABULARY = "vocabulary"
    CONTEXT_USAGE = "context_usage"
    RESPONSE_TIME = "response_time"


@dataclass
class Improvement:
    """An improvement hypothesis to be tested"""
    improvement_id: str
    dimension: ImprovementDimension
    hypothesis: str
    change_description: str
    target_value: Any
    baseline_value: Any
    status: ImprovementStatus = ImprovementStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    activated_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # A/B test results
    control_samples: int = 0
    treatment_samples: int = 0
    control_quality: float = 0.0
    treatment_quality: float = 0.0

    @property
    def improvement_delta(self) -> float:
        """Calculate improvement delta"""
        if self.control_quality > 0:
            return (self.treatment_quality - self.control_quality) / self.control_quality
        return 0.0

    def to_record(self) -> ImprovementRecord:
        """Convert to storage record"""
        return ImprovementRecord(
            improvement_id=self.improvement_id,
            hypothesis=self.hypothesis,
            target_dimension=self.dimension.value,
            change_description=self.change_description,
            status=self.status.value,
            created_at=self.created_at,
            activated_at=self.activated_at,
            completed_at=self.completed_at,
            control_samples=self.control_samples,
            treatment_samples=self.treatment_samples,
            control_quality=self.control_quality,
            treatment_quality=self.treatment_quality,
            improvement_delta=self.improvement_delta,
            metadata=self.metadata
        )

    @classmethod
    def from_record(cls, record: ImprovementRecord) -> "Improvement":
        """Create from storage record"""
        return cls(
            improvement_id=record.improvement_id,
            dimension=ImprovementDimension(record.target_dimension),
            hypothesis=record.hypothesis,
            change_description=record.change_description,
            target_value=record.metadata.get("target_value"),
            baseline_value=record.metadata.get("baseline_value"),
            status=ImprovementStatus(record.status),
            created_at=record.created_at,
            activated_at=record.activated_at,
            completed_at=record.completed_at,
            control_samples=record.control_samples,
            treatment_samples=record.treatment_samples,
            control_quality=record.control_quality,
            treatment_quality=record.treatment_quality,
            metadata=record.metadata
        )


@dataclass
class WeakArea:
    """Identified weak area for improvement"""
    dimension: ImprovementDimension
    current_score: float
    target_score: float
    sample_size: int
    evidence: List[str]
    suggested_change: str
    confidence: float


@dataclass
class ABTestResult:
    """Result of an A/B test"""
    improvement_id: str
    control_samples: int
    treatment_samples: int
    control_quality: float
    treatment_quality: float
    improvement_delta: float
    p_value: float
    is_significant: bool
    recommendation: str


# =============================================================================
# Quality Scorer (Helper Class)
# =============================================================================

class QualityScorer:
    """
    Scores response quality based on multiple dimensions

    PATTERN: Composite scoring
    WHY: Holistic quality assessment for improvement evaluation
    """

    def __init__(self):
        self.weights = {
            "relevance": 0.30,
            "coherence": 0.25,
            "helpfulness": 0.30,
            "style_match": 0.15
        }

    def score(
        self,
        response_text: str,
        query_text: str,
        feedback: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a response

        Returns:
            Tuple of (overall_score, component_scores)
        """
        component_scores = {}

        # Relevance score (heuristic based on keyword overlap)
        component_scores["relevance"] = self._score_relevance(response_text, query_text)

        # Coherence score (based on structure)
        component_scores["coherence"] = self._score_coherence(response_text)

        # Helpfulness score (from feedback if available)
        if feedback and feedback.get("helpful") is not None:
            component_scores["helpfulness"] = 1.0 if feedback["helpful"] else 0.0
        elif feedback and feedback.get("rating"):
            component_scores["helpfulness"] = feedback["rating"] / 5.0
        else:
            component_scores["helpfulness"] = 0.6  # Neutral default

        # Style match (if preferences available)
        if preferences:
            component_scores["style_match"] = self._score_style_match(
                response_text, preferences
            )
        else:
            component_scores["style_match"] = 0.5

        # Calculate weighted overall score
        overall = sum(
            self.weights[k] * component_scores[k]
            for k in self.weights
        )

        return (overall, component_scores)

    def _score_relevance(self, response: str, query: str) -> float:
        """Score relevance based on keyword overlap"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "can", "to", "of",
                       "in", "for", "on", "with", "at", "by", "from", "or", "and"}

        query_words -= common_words
        response_words -= common_words

        if not query_words:
            return 0.5

        overlap = len(query_words & response_words)
        return min(1.0, overlap / len(query_words))

    def _score_coherence(self, response: str) -> float:
        """Score coherence based on structure"""
        # Simple heuristics
        score = 0.5

        # Check for proper sentence structure
        sentences = response.split(". ")
        if len(sentences) > 0:
            avg_sentence_length = len(response.split()) / len(sentences)
            if 5 < avg_sentence_length < 30:
                score += 0.2

        # Check for paragraph structure
        if "\n\n" in response or len(response) < 500:
            score += 0.1

        # Penalize very short or very long responses
        word_count = len(response.split())
        if 20 < word_count < 300:
            score += 0.2

        return min(1.0, score)

    def _score_style_match(
        self,
        response: str,
        preferences: Dict[str, Any]
    ) -> float:
        """Score how well response matches user style preferences"""
        score = 0.5
        word_count = len(response.split())

        # Check response length preference
        length_pref = preferences.get("response_length", "medium")
        if length_pref == "short" and word_count < 100:
            score += 0.3
        elif length_pref == "long" and word_count > 150:
            score += 0.3
        elif length_pref == "medium" and 50 < word_count < 200:
            score += 0.3

        # Check technical depth preference
        depth_pref = preferences.get("technical_depth", "intermediate")
        technical_terms = len([w for w in response.lower().split()
                              if len(w) > 10])  # Simple proxy

        if depth_pref == "beginner" and technical_terms < 3:
            score += 0.2
        elif depth_pref == "expert" and technical_terms > 5:
            score += 0.2

        return min(1.0, score)


# =============================================================================
# Improvement Engine
# =============================================================================

class ImprovementEngine:
    """
    Continuous improvement engine with A/B testing

    PATTERN: Experiment-driven optimization
    WHY: Validates improvements with statistical significance before deployment

    Features:
    - Weak area identification
    - Hypothesis generation
    - A/B test management
    - Auto-rollback on quality drops
    - Improvement history tracking
    """

    def __init__(
        self,
        store: Optional[LearningStore] = None,
        config: Optional[ImprovementConfig] = None,
        ab_config: Optional[ABTestingConfig] = None
    ):
        """
        Initialize ImprovementEngine

        Args:
            store: Learning data store
            config: Improvement configuration
            ab_config: A/B testing configuration
        """
        self.store = store or learning_store
        self.config = config or learning_config.improvement
        self.ab_config = ab_config or learning_config.ab_testing

        # Quality scorer
        self.scorer = QualityScorer()

        # Active improvements
        self._active_improvements: Dict[str, Improvement] = {}

        # Session assignments for A/B tests
        self._session_assignments: Dict[str, Dict[str, bool]] = defaultdict(dict)

        # Quality tracking
        self._quality_history: List[Tuple[str, float]] = []

        # Background task handle
        self._background_task: Optional[asyncio.Task] = None

        logger.info(
            "improvement_engine_initialized",
            quality_drop_threshold=self.config.quality_drop_threshold,
            ab_split=self.ab_config.treatment_split,
            auto_rollback=self.config.enable_auto_rollback
        )

    # =========================================================================
    # Main Interface Methods
    # =========================================================================

    async def analyze_weak_areas(
        self,
        session_id: Optional[str] = None,
        days: int = 30
    ) -> List[WeakArea]:
        """
        Identify areas with below-threshold quality

        PATTERN: Statistical analysis of quality metrics
        WHY: Focus improvement efforts on weakest areas

        Args:
            session_id: Optional specific session to analyze
            days: Number of days to analyze

        Returns:
            List of identified weak areas
        """
        weak_areas = []

        try:
            # Get quality statistics
            quality_stats = await self.store.get_quality_stats(
                session_id=session_id,
                days=days
            )

            # Get feedback statistics
            feedback_stats = await self.store.get_feedback_stats(
                session_id=session_id,
                days=days
            )

            # Analyze each dimension
            if quality_stats.get("total_responses", 0) > 0:
                # Response quality
                avg_quality = quality_stats.get("avg_quality")
                if avg_quality and avg_quality < self.config.weak_area_threshold:
                    weak_areas.append(WeakArea(
                        dimension=ImprovementDimension.RESPONSE_LENGTH,
                        current_score=avg_quality,
                        target_score=self.config.weak_area_threshold + 0.1,
                        sample_size=quality_stats["total_responses"],
                        evidence=[f"Average quality score: {avg_quality:.2f}"],
                        suggested_change="Adjust response length based on user preferences",
                        confidence=min(0.9, quality_stats["total_responses"] / 100)
                    ))

            # Feedback-based analysis
            if feedback_stats.get("total_feedback", 0) > 10:
                helpful_rate = feedback_stats.get("helpful_rate", 0.5)
                if helpful_rate < 0.7:
                    weak_areas.append(WeakArea(
                        dimension=ImprovementDimension.TECHNICAL_DEPTH,
                        current_score=helpful_rate,
                        target_score=0.8,
                        sample_size=feedback_stats["total_feedback"],
                        evidence=[f"Helpful rate: {helpful_rate:.2%}"],
                        suggested_change="Adjust technical depth to user level",
                        confidence=min(0.9, feedback_stats["total_feedback"] / 50)
                    ))

            logger.info(
                "weak_areas_analyzed",
                session_id=session_id,
                weak_areas_found=len(weak_areas),
                days=days
            )

            return weak_areas

        except Exception as e:
            logger.error(
                "analyze_weak_areas_failed",
                session_id=session_id,
                error=str(e)
            )
            return weak_areas

    async def generate_improvements(
        self,
        weak_areas: Optional[List[WeakArea]] = None,
        max_improvements: int = 3
    ) -> List[Improvement]:
        """
        Generate improvement hypotheses for weak areas

        PATTERN: Hypothesis generation
        WHY: Structured approach to improvement

        Args:
            weak_areas: Weak areas to address (auto-analyzes if not provided)
            max_improvements: Maximum improvements to generate

        Returns:
            List of improvement hypotheses
        """
        improvements = []

        try:
            # Analyze if not provided
            if weak_areas is None:
                weak_areas = await self.analyze_weak_areas()

            # Generate improvements for each weak area
            for area in weak_areas[:max_improvements]:
                improvement = await self._generate_improvement_for_area(area)
                if improvement:
                    improvements.append(improvement)

            logger.info(
                "improvements_generated",
                weak_areas_count=len(weak_areas),
                improvements_count=len(improvements)
            )

            return improvements

        except Exception as e:
            logger.error(
                "generate_improvements_failed",
                error=str(e)
            )
            return improvements

    async def apply_improvement(
        self,
        improvement_id: str
    ) -> bool:
        """
        Activate an improvement for A/B testing

        Args:
            improvement_id: ID of improvement to activate

        Returns:
            Success status
        """
        try:
            # Get improvement
            record = await self.store.get_improvement(improvement_id)
            if not record:
                logger.warning("improvement_not_found", improvement_id=improvement_id)
                return False

            improvement = Improvement.from_record(record)

            # Check concurrent experiment limit
            active_count = len(self._active_improvements)
            if active_count >= self.ab_config.max_concurrent_experiments:
                logger.warning(
                    "max_concurrent_experiments_reached",
                    active_count=active_count,
                    max_allowed=self.ab_config.max_concurrent_experiments
                )
                return False

            # Activate improvement
            improvement.status = ImprovementStatus.ACTIVE
            improvement.activated_at = datetime.utcnow().isoformat()

            # Save to store
            await self.store.update_improvement_status(
                improvement_id,
                "active"
            )

            # Track locally
            self._active_improvements[improvement_id] = improvement

            logger.info(
                "improvement_activated",
                improvement_id=improvement_id,
                dimension=improvement.dimension.value,
                hypothesis=improvement.hypothesis
            )

            return True

        except Exception as e:
            logger.error(
                "apply_improvement_failed",
                improvement_id=improvement_id,
                error=str(e)
            )
            return False

    async def measure_impact(
        self,
        improvement_id: str
    ) -> Optional[ABTestResult]:
        """
        Measure impact of an improvement

        PATTERN: Statistical hypothesis testing
        WHY: Validate improvements with statistical significance

        Args:
            improvement_id: ID of improvement to measure

        Returns:
            ABTestResult with analysis
        """
        try:
            if improvement_id not in self._active_improvements:
                record = await self.store.get_improvement(improvement_id)
                if not record:
                    return None
                improvement = Improvement.from_record(record)
            else:
                improvement = self._active_improvements[improvement_id]

            # Get quality metrics for control and treatment groups
            control_stats = await self.store.get_quality_stats(
                improvement_id=improvement_id
            )

            # Calculate statistical significance
            control_samples = improvement.control_samples
            treatment_samples = improvement.treatment_samples
            control_quality = improvement.control_quality
            treatment_quality = improvement.treatment_quality

            # Simple z-test for proportion comparison
            if control_samples > 0 and treatment_samples > 0:
                pooled_quality = (
                    (control_quality * control_samples + treatment_quality * treatment_samples) /
                    (control_samples + treatment_samples)
                )

                # Standard error
                se = math.sqrt(
                    pooled_quality * (1 - pooled_quality) *
                    (1/control_samples + 1/treatment_samples)
                ) if pooled_quality > 0 else 0.1

                # Z-score
                z = (treatment_quality - control_quality) / se if se > 0 else 0

                # Two-tailed p-value (approximation)
                p_value = 2 * (1 - self._normal_cdf(abs(z)))
            else:
                z = 0
                p_value = 1.0

            is_significant = (
                p_value < self.ab_config.p_value_threshold and
                control_samples >= self.ab_config.min_samples_for_significance and
                treatment_samples >= self.ab_config.min_samples_for_significance
            )

            improvement_delta = improvement.improvement_delta

            # Determine recommendation
            if not is_significant:
                if control_samples + treatment_samples < self.ab_config.min_samples_for_significance * 2:
                    recommendation = "Continue testing - insufficient samples"
                else:
                    recommendation = "No significant difference detected"
            elif improvement_delta > self.ab_config.min_effect_size:
                recommendation = "Deploy improvement - significant positive impact"
            elif improvement_delta < -self.config.quality_drop_threshold:
                recommendation = "Rollback - significant negative impact"
            else:
                recommendation = "Continue testing - effect size too small"

            result = ABTestResult(
                improvement_id=improvement_id,
                control_samples=control_samples,
                treatment_samples=treatment_samples,
                control_quality=control_quality,
                treatment_quality=treatment_quality,
                improvement_delta=improvement_delta,
                p_value=p_value,
                is_significant=is_significant,
                recommendation=recommendation
            )

            logger.info(
                "impact_measured",
                improvement_id=improvement_id,
                improvement_delta=improvement_delta,
                p_value=p_value,
                is_significant=is_significant,
                recommendation=recommendation
            )

            return result

        except Exception as e:
            logger.error(
                "measure_impact_failed",
                improvement_id=improvement_id,
                error=str(e)
            )
            return None

    async def rollback_improvement(
        self,
        improvement_id: str,
        reason: str = "manual"
    ) -> bool:
        """
        Rollback an improvement

        Args:
            improvement_id: ID of improvement to rollback
            reason: Reason for rollback

        Returns:
            Success status
        """
        try:
            if improvement_id in self._active_improvements:
                del self._active_improvements[improvement_id]

            # Clear session assignments
            for session_assignments in self._session_assignments.values():
                session_assignments.pop(improvement_id, None)

            # Update store
            await self.store.update_improvement_status(
                improvement_id,
                "rolled_back"
            )

            logger.info(
                "improvement_rolled_back",
                improvement_id=improvement_id,
                reason=reason
            )

            return True

        except Exception as e:
            logger.error(
                "rollback_improvement_failed",
                improvement_id=improvement_id,
                error=str(e)
            )
            return False

    # =========================================================================
    # A/B Testing Methods
    # =========================================================================

    def assign_treatment(
        self,
        session_id: str,
        improvement_id: str
    ) -> bool:
        """
        Assign session to control or treatment group

        PATTERN: Consistent hashing for assignment
        WHY: Same session always gets same assignment

        Args:
            session_id: Session to assign
            improvement_id: Improvement experiment

        Returns:
            True if assigned to treatment, False for control
        """
        # Check existing assignment
        if improvement_id in self._session_assignments.get(session_id, {}):
            return self._session_assignments[session_id][improvement_id]

        # Deterministic assignment based on session_id hash
        hash_value = hash(f"{session_id}:{improvement_id}") % 100
        is_treatment = hash_value < (self.ab_config.treatment_split * 100)

        # Store assignment
        self._session_assignments[session_id][improvement_id] = is_treatment

        logger.debug(
            "treatment_assigned",
            session_id=session_id,
            improvement_id=improvement_id,
            is_treatment=is_treatment
        )

        return is_treatment

    async def record_observation(
        self,
        session_id: str,
        improvement_id: str,
        quality_score: float,
        response_metadata: Dict[str, Any]
    ):
        """
        Record an observation for an A/B test

        Args:
            session_id: Session that received response
            improvement_id: Active improvement being tested
            quality_score: Quality score for the response
            response_metadata: Response metadata
        """
        try:
            if improvement_id not in self._active_improvements:
                return

            improvement = self._active_improvements[improvement_id]
            is_treatment = self.assign_treatment(session_id, improvement_id)

            # Update running statistics
            if is_treatment:
                n = improvement.treatment_samples
                improvement.treatment_quality = (
                    (improvement.treatment_quality * n + quality_score) / (n + 1)
                )
                improvement.treatment_samples += 1
            else:
                n = improvement.control_samples
                improvement.control_quality = (
                    (improvement.control_quality * n + quality_score) / (n + 1)
                )
                improvement.control_samples += 1

            # Save quality metric
            await self.store.save_quality_metric(
                session_id=session_id,
                response_id=response_metadata.get("response_id", str(uuid.uuid4())),
                quality_score=quality_score,
                word_count=response_metadata.get("word_count"),
                response_time_ms=response_metadata.get("response_time_ms"),
                improvement_id=improvement_id,
                is_treatment=is_treatment
            )

            # Update store periodically
            total_samples = improvement.control_samples + improvement.treatment_samples
            if total_samples % 10 == 0:
                await self.store.update_improvement_status(
                    improvement_id,
                    "active",
                    control_samples=improvement.control_samples,
                    treatment_samples=improvement.treatment_samples,
                    control_quality=improvement.control_quality,
                    treatment_quality=improvement.treatment_quality
                )

            # Check for auto-rollback
            if self.config.enable_auto_rollback:
                await self._check_auto_rollback(improvement)

        except Exception as e:
            logger.error(
                "record_observation_failed",
                session_id=session_id,
                improvement_id=improvement_id,
                error=str(e)
            )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _generate_improvement_for_area(
        self,
        area: WeakArea
    ) -> Optional[Improvement]:
        """Generate improvement hypothesis for a weak area"""

        hypothesis_templates = {
            ImprovementDimension.RESPONSE_LENGTH: [
                ("Shorter responses improve engagement", "short", "medium"),
                ("Longer responses improve comprehension", "long", "medium"),
            ],
            ImprovementDimension.TECHNICAL_DEPTH: [
                ("Simpler explanations improve understanding", "beginner", "intermediate"),
                ("More technical detail improves satisfaction", "expert", "intermediate"),
            ],
            ImprovementDimension.COMMUNICATION_STYLE: [
                ("Formal tone improves perceived quality", "formal", "balanced"),
                ("Casual tone improves engagement", "casual", "balanced"),
            ],
        }

        templates = hypothesis_templates.get(area.dimension, [])
        if not templates:
            return None

        # Select hypothesis based on evidence
        hypothesis, target, baseline = templates[0]  # Simple selection

        improvement = Improvement(
            improvement_id=str(uuid.uuid4()),
            dimension=area.dimension,
            hypothesis=hypothesis,
            change_description=area.suggested_change,
            target_value=target,
            baseline_value=baseline,
            metadata={
                "weak_area_score": area.current_score,
                "target_score": area.target_score,
                "evidence": area.evidence,
                "target_value": target,
                "baseline_value": baseline
            }
        )

        # Save to store
        await self.store.save_improvement(improvement.to_record())

        return improvement

    async def _check_auto_rollback(self, improvement: Improvement):
        """Check if improvement should be auto-rolled back"""
        if improvement.treatment_samples < self.config.rollback_observation_period:
            return

        if improvement.improvement_delta < -self.config.quality_drop_threshold:
            logger.warning(
                "auto_rollback_triggered",
                improvement_id=improvement.improvement_id,
                improvement_delta=improvement.improvement_delta,
                threshold=-self.config.quality_drop_threshold
            )

            await self.rollback_improvement(
                improvement.improvement_id,
                reason=f"Quality drop of {improvement.improvement_delta:.2%} exceeded threshold"
            )

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF"""
        # Approximation of standard normal CDF
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    # =========================================================================
    # Background Task
    # =========================================================================

    async def start_background_task(self, interval_seconds: int = 300):
        """Start background improvement checking task"""
        async def _run():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self._run_improvement_cycle()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("background_task_error", error=str(e))

        self._background_task = asyncio.create_task(_run())
        logger.info("background_task_started", interval_seconds=interval_seconds)

    async def stop_background_task(self):
        """Stop background task"""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("background_task_stopped")

    async def _run_improvement_cycle(self):
        """Run one improvement cycle"""
        # Check active improvements
        for improvement_id, improvement in list(self._active_improvements.items()):
            result = await self.measure_impact(improvement_id)

            if result and result.is_significant:
                if result.improvement_delta > self.ab_config.min_effect_size:
                    # Improvement is successful - could deploy permanently
                    improvement.status = ImprovementStatus.COMPLETED
                    improvement.completed_at = datetime.utcnow().isoformat()
                    await self.store.update_improvement_status(
                        improvement_id,
                        "completed"
                    )
                    del self._active_improvements[improvement_id]
                    logger.info(
                        "improvement_completed_successfully",
                        improvement_id=improvement_id
                    )
                elif result.improvement_delta < -self.config.quality_drop_threshold:
                    await self.rollback_improvement(
                        improvement_id,
                        reason="Significant negative impact"
                    )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_improvement_stats(self) -> Dict[str, Any]:
        """Get overall improvement statistics"""
        active_records = await self.store.get_active_improvements()

        return {
            "active_improvements": len(self._active_improvements),
            "pending_improvements": len([
                r for r in active_records if r.status == "pending"
            ]),
            "improvements": [
                {
                    "improvement_id": imp.improvement_id,
                    "dimension": imp.dimension.value,
                    "status": imp.status.value,
                    "control_samples": imp.control_samples,
                    "treatment_samples": imp.treatment_samples,
                    "improvement_delta": imp.improvement_delta
                }
                for imp in self._active_improvements.values()
            ]
        }

    def get_active_improvements_for_session(
        self,
        session_id: str
    ) -> List[Tuple[str, bool]]:
        """Get active improvements and treatment assignments for a session"""
        return [
            (imp_id, self.assign_treatment(session_id, imp_id))
            for imp_id in self._active_improvements
        ]


# =============================================================================
# Factory Function
# =============================================================================

def create_improvement_engine(
    store: Optional[LearningStore] = None,
    config: Optional[ImprovementConfig] = None,
    ab_config: Optional[ABTestingConfig] = None
) -> ImprovementEngine:
    """Factory function to create an ImprovementEngine"""
    return ImprovementEngine(store=store, config=config, ab_config=ab_config)


# Global instance
improvement_engine = ImprovementEngine()
