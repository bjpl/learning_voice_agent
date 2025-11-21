"""
Pattern Detection Engine
PATTERN: Statistical pattern detection with clustering
WHY: Identify recurring behaviors and quality correlations
SPARC: Data-driven pattern recognition with significance testing
"""
import uuid
import statistics
import math
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from app.logger import db_logger
from app.learning.config import LearningConfig, learning_config


class PatternType(str, Enum):
    """Types of detectable patterns"""
    RECURRING_QUESTION = "recurring_question"
    COMMON_CORRECTION = "common_correction"
    ENGAGEMENT_TIME = "engagement_time"
    TOPIC_CLUSTER = "topic_cluster"
    QUALITY_CORRELATION = "quality_correlation"
    VOCABULARY_PREFERENCE = "vocabulary_preference"
    RESPONSE_LENGTH = "response_length"
    CLARIFICATION_TRIGGER = "clarification_trigger"


@dataclass
class DetectedPattern:
    """A detected pattern with metadata"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    frequency: int
    confidence: float
    first_detected: datetime
    last_updated: datetime
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    actionable: bool = False
    recommendation: Optional[str] = None


@dataclass
class Cluster:
    """A cluster of similar items"""
    cluster_id: str
    centroid_text: str
    members: List[str]
    member_ids: List[str]
    cohesion: float
    size: int


@dataclass
class Correlation:
    """A statistical correlation"""
    factor: str
    target: str
    correlation: float
    p_value: float
    sample_size: int
    significant: bool
    insight: str


class PatternDetector:
    """
    PATTERN: Multi-method pattern detection
    WHY: Find meaningful patterns in learning behavior

    Detects:
    - Recurring questions (semantic clustering)
    - Common corrections (vocabulary preferences)
    - Engagement patterns (temporal analysis)
    - Topic clusters (co-occurrence)
    - Quality correlations (statistical analysis)

    USAGE:
        detector = PatternDetector()
        patterns = await detector.detect_recurring_patterns(queries)
        correlations = await detector.detect_quality_correlations(data)
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        """
        Initialize pattern detector

        Args:
            config: Learning configuration
        """
        self.config = config or learning_config
        self._embedding_generator = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pattern detector with embedding support"""
        if self._initialized:
            return

        try:
            # Lazy import to avoid circular dependencies
            from app.vector.embeddings import EmbeddingGenerator
            self._embedding_generator = EmbeddingGenerator()
            await self._embedding_generator.initialize()

            self._initialized = True
            db_logger.info("pattern_detector_initialized")

        except Exception as e:
            db_logger.error(
                "pattern_detector_initialization_failed",
                error=str(e),
                exc_info=True
            )
            # Continue without embedding support
            self._initialized = True

    async def detect_recurring_patterns(
        self,
        queries: List[str],
        query_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[DetectedPattern]:
        """
        Detect recurring question patterns using semantic clustering

        PATTERN: Embedding-based semantic clustering
        WHY: Find questions with similar meaning even if worded differently

        Args:
            queries: List of user queries to analyze
            query_ids: Optional IDs for each query
            similarity_threshold: Minimum similarity for clustering

        Returns:
            List of detected recurring patterns
        """
        if not queries:
            return []

        await self.initialize()

        threshold = similarity_threshold or self.config.patterns.similarity_threshold
        min_size = self.config.patterns.min_cluster_size

        try:
            db_logger.info(
                "detecting_recurring_patterns",
                query_count=len(queries),
                threshold=threshold
            )

            # Assign IDs if not provided
            if query_ids is None:
                query_ids = [str(i) for i in range(len(queries))]

            # Generate embeddings if available
            if self._embedding_generator:
                clusters = await self._cluster_with_embeddings(
                    queries, query_ids, threshold
                )
            else:
                # Fall back to text-based clustering
                clusters = self._cluster_with_text_similarity(
                    queries, query_ids, threshold
                )

            # Convert clusters to patterns
            patterns = []
            for cluster in clusters:
                if cluster.size >= min_size:
                    pattern = DetectedPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.RECURRING_QUESTION,
                        description=f"Recurring question pattern: {cluster.centroid_text[:50]}...",
                        frequency=cluster.size,
                        confidence=cluster.cohesion,
                        first_detected=datetime.utcnow(),
                        last_updated=datetime.utcnow(),
                        examples=cluster.members[:5],
                        metadata={
                            'cluster_id': cluster.cluster_id,
                            'member_ids': cluster.member_ids[:20]
                        },
                        actionable=cluster.size >= min_size * 2,
                        recommendation=self._generate_recurring_recommendation(cluster)
                    )
                    patterns.append(pattern)

            db_logger.info(
                "recurring_patterns_detected",
                pattern_count=len(patterns),
                largest_cluster=max([p.frequency for p in patterns]) if patterns else 0
            )

            return sorted(patterns, key=lambda p: p.frequency, reverse=True)

        except Exception as e:
            db_logger.error(
                "detect_recurring_patterns_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def detect_quality_correlations(
        self,
        quality_data: List[Dict[str, Any]]
    ) -> List[Correlation]:
        """
        Find factors that correlate with quality scores

        PATTERN: Statistical correlation analysis
        WHY: Identify what contributes to high/low quality

        Args:
            quality_data: List of dicts with quality scores and factors

        Returns:
            List of significant correlations
        """
        if not quality_data or len(quality_data) < self.config.patterns.min_samples_for_correlation:
            return []

        try:
            db_logger.info(
                "detecting_quality_correlations",
                sample_size=len(quality_data)
            )

            correlations = []

            # Response length vs quality
            length_corr = self._correlate_factor(
                quality_data,
                'response_length',
                'quality_score'
            )
            if length_corr:
                correlations.append(length_corr)

            # Time of day vs quality
            time_corr = self._correlate_time_quality(quality_data)
            if time_corr:
                correlations.append(time_corr)

            # Topic vs quality
            topic_corr = self._correlate_topic_quality(quality_data)
            correlations.extend(topic_corr)

            # Exchange position vs quality
            position_corr = self._correlate_factor(
                quality_data,
                'exchange_position',
                'quality_score'
            )
            if position_corr:
                correlations.append(position_corr)

            # Filter to significant correlations
            significant = [
                c for c in correlations
                if abs(c.correlation) >= self.config.patterns.correlation_significance_threshold
            ]

            db_logger.info(
                "quality_correlations_detected",
                total_tested=len(correlations),
                significant_count=len(significant)
            )

            return sorted(significant, key=lambda c: abs(c.correlation), reverse=True)

        except Exception as e:
            db_logger.error(
                "detect_quality_correlations_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def detect_engagement_triggers(
        self,
        session_data: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """
        Detect patterns that trigger high engagement

        PATTERN: Behavioral analysis with statistical significance
        WHY: Understand what drives user engagement

        Args:
            session_data: Session data with engagement metrics

        Returns:
            List of engagement trigger patterns
        """
        if not session_data:
            return []

        try:
            db_logger.info(
                "detecting_engagement_triggers",
                session_count=len(session_data)
            )

            patterns = []

            # Time-based engagement patterns
            time_patterns = self._detect_time_engagement_patterns(session_data)
            patterns.extend(time_patterns)

            # Topic-based engagement patterns
            topic_patterns = self._detect_topic_engagement_patterns(session_data)
            patterns.extend(topic_patterns)

            # Session length patterns
            length_patterns = self._detect_length_engagement_patterns(session_data)
            patterns.extend(length_patterns)

            db_logger.info(
                "engagement_triggers_detected",
                pattern_count=len(patterns)
            )

            return patterns

        except Exception as e:
            db_logger.error(
                "detect_engagement_triggers_failed",
                error=str(e),
                exc_info=True
            )
            return []

    async def detect_daily_patterns(
        self,
        target_date: date
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns for a specific day

        Args:
            target_date: Date to analyze

        Returns:
            List of daily pattern summaries
        """
        try:
            # This would integrate with the stores to get actual data
            # For now, return empty list - to be populated by analytics integration
            return []

        except Exception as e:
            db_logger.error(
                "detect_daily_patterns_failed",
                date=str(target_date),
                error=str(e)
            )
            return []

    def get_pattern_insights(
        self,
        patterns: List[DetectedPattern]
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable insights from detected patterns

        PATTERN: Insight extraction from patterns
        WHY: Convert patterns into recommendations

        Args:
            patterns: List of detected patterns

        Returns:
            List of insights with recommendations
        """
        insights = []

        for pattern in patterns:
            if not pattern.actionable:
                continue

            insight = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type.value,
                'title': self._generate_insight_title(pattern),
                'description': pattern.description,
                'confidence': pattern.confidence,
                'frequency': pattern.frequency,
                'recommendation': pattern.recommendation,
                'priority': self._calculate_insight_priority(pattern)
            }
            insights.append(insight)

        # Sort by priority
        return sorted(insights, key=lambda i: i['priority'], reverse=True)

    # =========================================================================
    # Private Clustering Methods
    # =========================================================================

    async def _cluster_with_embeddings(
        self,
        texts: List[str],
        text_ids: List[str],
        threshold: float
    ) -> List[Cluster]:
        """Cluster texts using embedding similarity"""
        try:
            # Generate embeddings
            embeddings = await self._embedding_generator.generate_batch(texts)

            # Simple greedy clustering
            clusters = []
            assigned = set()

            for i, (text, text_id, emb) in enumerate(zip(texts, text_ids, embeddings)):
                if i in assigned:
                    continue

                # Start new cluster
                cluster_members = [(text, text_id, i)]
                assigned.add(i)

                # Find similar texts
                for j in range(i + 1, len(texts)):
                    if j in assigned:
                        continue

                    similarity = self._cosine_similarity(emb, embeddings[j])
                    if similarity >= threshold:
                        cluster_members.append((texts[j], text_ids[j], j))
                        assigned.add(j)

                if len(cluster_members) >= 2:
                    # Calculate cluster cohesion (average pairwise similarity)
                    cohesion = self._calculate_cluster_cohesion(
                        [embeddings[m[2]] for m in cluster_members]
                    )

                    clusters.append(Cluster(
                        cluster_id=str(uuid.uuid4()),
                        centroid_text=cluster_members[0][0],  # First member as representative
                        members=[m[0] for m in cluster_members],
                        member_ids=[m[1] for m in cluster_members],
                        cohesion=cohesion,
                        size=len(cluster_members)
                    ))

            return clusters

        except Exception as e:
            db_logger.error("embedding_clustering_failed", error=str(e))
            return self._cluster_with_text_similarity(texts, text_ids, threshold)

    def _cluster_with_text_similarity(
        self,
        texts: List[str],
        text_ids: List[str],
        threshold: float
    ) -> List[Cluster]:
        """Fall back clustering using text similarity"""
        clusters = []
        assigned = set()

        # Normalize texts for comparison
        normalized = [self._normalize_text(t) for t in texts]

        for i, (text, text_id, norm) in enumerate(zip(texts, text_ids, normalized)):
            if i in assigned:
                continue

            cluster_members = [(text, text_id)]
            assigned.add(i)

            for j in range(i + 1, len(texts)):
                if j in assigned:
                    continue

                # Simple Jaccard similarity
                similarity = self._jaccard_similarity(norm, normalized[j])
                if similarity >= threshold * 0.7:  # Adjust threshold for text
                    cluster_members.append((texts[j], text_ids[j]))
                    assigned.add(j)

            if len(cluster_members) >= 2:
                clusters.append(Cluster(
                    cluster_id=str(uuid.uuid4()),
                    centroid_text=cluster_members[0][0],
                    members=[m[0] for m in cluster_members],
                    member_ids=[m[1] for m in cluster_members],
                    cohesion=0.7,  # Estimated cohesion
                    size=len(cluster_members)
                ))

        return clusters

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _calculate_cluster_cohesion(self, embeddings: List) -> float:
        """Calculate average pairwise similarity in cluster"""
        if len(embeddings) < 2:
            return 1.0

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarities.append(self._cosine_similarity(embeddings[i], embeddings[j]))

        return statistics.mean(similarities) if similarities else 0.0

    def _normalize_text(self, text: str) -> Set[str]:
        """Normalize text to set of tokens"""
        # Simple tokenization and lowercasing
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                      'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'i',
                      'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                      'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
                      'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                      'just', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                      'while', 'about', 'me', 'my', 'your', 'this', 'that', 'these'}
        return set(w for w in words if w not in stop_words and len(w) > 2)

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # Private Correlation Methods
    # =========================================================================

    def _correlate_factor(
        self,
        data: List[Dict[str, Any]],
        factor_key: str,
        target_key: str
    ) -> Optional[Correlation]:
        """Calculate correlation between a factor and target"""
        # Extract valid pairs
        pairs = [
            (d.get(factor_key), d.get(target_key))
            for d in data
            if d.get(factor_key) is not None and d.get(target_key) is not None
        ]

        if len(pairs) < self.config.patterns.min_samples_for_correlation:
            return None

        factors = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]

        # Calculate Pearson correlation
        correlation, p_value = self._pearson_correlation(factors, targets)

        if correlation is None:
            return None

        significant = (
            abs(correlation) >= self.config.patterns.correlation_significance_threshold
            and p_value < 0.05
        )

        insight = self._generate_correlation_insight(factor_key, target_key, correlation)

        return Correlation(
            factor=factor_key,
            target=target_key,
            correlation=correlation,
            p_value=p_value,
            sample_size=len(pairs),
            significant=significant,
            insight=insight
        )

    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[Optional[float], float]:
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        if n < 3:
            return None, 1.0

        try:
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)

            # Calculate covariance and standard deviations
            cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
            std_x = statistics.stdev(x)
            std_y = statistics.stdev(y)

            if std_x == 0 or std_y == 0:
                return None, 1.0

            r = cov / (std_x * std_y)

            # Approximate p-value using t-distribution
            t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float('inf')
            # Simplified p-value estimation
            p_value = 2 * (1 - self._t_cdf(abs(t), n - 2))

            return r, p_value

        except (statistics.StatisticsError, ZeroDivisionError, ValueError):
            return None, 1.0

    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate CDF of t-distribution"""
        # Simple approximation using normal distribution for large df
        if df > 30:
            # Use normal approximation
            return 0.5 * (1 + math.erf(t / math.sqrt(2)))

        # Simple approximation for small df
        x = df / (df + t * t)
        return 1 - 0.5 * math.pow(x, df / 2)

    def _correlate_time_quality(
        self,
        data: List[Dict[str, Any]]
    ) -> Optional[Correlation]:
        """Correlate time of day with quality"""
        pairs = []
        for d in data:
            timestamp = d.get('timestamp')
            quality = d.get('quality_score')

            if timestamp and quality:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                hour = timestamp.hour
                pairs.append((hour, quality))

        if len(pairs) < self.config.patterns.min_samples_for_correlation:
            return None

        hours = [p[0] for p in pairs]
        qualities = [p[1] for p in pairs]

        correlation, p_value = self._pearson_correlation(hours, qualities)

        if correlation is None:
            return None

        # Group by morning/afternoon/evening for insight
        morning_avg = statistics.mean([q for h, q in pairs if 6 <= h < 12]) if any(6 <= h < 12 for h, _ in pairs) else 0
        afternoon_avg = statistics.mean([q for h, q in pairs if 12 <= h < 18]) if any(12 <= h < 18 for h, _ in pairs) else 0
        evening_avg = statistics.mean([q for h, q in pairs if 18 <= h < 24 or h < 6]) if any(18 <= h < 24 or h < 6 for h, _ in pairs) else 0

        best_time = 'morning' if morning_avg >= afternoon_avg and morning_avg >= evening_avg else (
            'afternoon' if afternoon_avg >= evening_avg else 'evening'
        )

        return Correlation(
            factor='time_of_day',
            target='quality_score',
            correlation=correlation,
            p_value=p_value,
            sample_size=len(pairs),
            significant=abs(correlation) > 0.2,
            insight=f"Quality tends to be highest in the {best_time}"
        )

    def _correlate_topic_quality(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Correlation]:
        """Find topics that correlate with quality"""
        # Group by topic
        topic_qualities = defaultdict(list)

        for d in data:
            topics = d.get('topics', [])
            quality = d.get('quality_score')

            if quality is not None:
                for topic in topics:
                    topic_qualities[topic].append(quality)

        correlations = []
        overall_avg = statistics.mean([
            d.get('quality_score', 0) for d in data
            if d.get('quality_score') is not None
        ]) if data else 0

        for topic, qualities in topic_qualities.items():
            if len(qualities) >= 5:  # Minimum samples per topic
                topic_avg = statistics.mean(qualities)
                diff = topic_avg - overall_avg

                # Treat as correlation-like measure
                if abs(diff) > 0.1:  # Meaningful difference
                    correlations.append(Correlation(
                        factor=f'topic_{topic}',
                        target='quality_score',
                        correlation=diff,  # Not true correlation, but directional
                        p_value=0.05,  # Simplified
                        sample_size=len(qualities),
                        significant=abs(diff) > 0.1,
                        insight=f"Topic '{topic}' has {'higher' if diff > 0 else 'lower'} than average quality"
                    ))

        return correlations

    # =========================================================================
    # Private Engagement Pattern Methods
    # =========================================================================

    def _detect_time_engagement_patterns(
        self,
        session_data: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect time-based engagement patterns"""
        patterns = []

        # Group sessions by hour
        hour_sessions = defaultdict(list)
        for session in session_data:
            timestamp = session.get('start_time')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                hour_sessions[timestamp.hour].append(session)

        # Find peak hours
        if hour_sessions:
            peak_hour = max(hour_sessions.keys(), key=lambda h: len(hour_sessions[h]))
            peak_count = len(hour_sessions[peak_hour])

            if peak_count >= 5:  # Minimum for pattern
                patterns.append(DetectedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.ENGAGEMENT_TIME,
                    description=f"Peak engagement at {peak_hour}:00",
                    frequency=peak_count,
                    confidence=min(peak_count / 10, 1.0),
                    first_detected=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    metadata={'peak_hour': peak_hour},
                    actionable=True,
                    recommendation=f"Schedule important interactions around {peak_hour}:00"
                ))

        return patterns

    def _detect_topic_engagement_patterns(
        self,
        session_data: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect topic-based engagement patterns"""
        patterns = []

        # Count topic frequency
        topic_counts = defaultdict(int)
        for session in session_data:
            for topic in session.get('topics', []):
                topic_counts[topic] += 1

        # Find top topics
        if topic_counts:
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            for topic, count in top_topics:
                if count >= 3:
                    patterns.append(DetectedPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.TOPIC_CLUSTER,
                        description=f"Frequent topic: {topic}",
                        frequency=count,
                        confidence=min(count / 10, 1.0),
                        first_detected=datetime.utcnow(),
                        last_updated=datetime.utcnow(),
                        metadata={'topic': topic},
                        actionable=True,
                        recommendation=f"Enhance knowledge base for '{topic}'"
                    ))

        return patterns

    def _detect_length_engagement_patterns(
        self,
        session_data: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect session length engagement patterns"""
        patterns = []

        durations = [
            s.get('duration', 0) / 60  # Convert to minutes
            for s in session_data
            if s.get('duration', 0) > 0
        ]

        if len(durations) >= 5:
            avg_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)

            # Check for long session pattern
            long_sessions = [d for d in durations if d > avg_duration * 1.5]
            if len(long_sessions) >= 3:
                patterns.append(DetectedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.RESPONSE_LENGTH,
                    description=f"User tends to have extended sessions (avg {avg_duration:.1f} min)",
                    frequency=len(long_sessions),
                    confidence=0.7,
                    first_detected=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    metadata={
                        'avg_duration': avg_duration,
                        'median_duration': median_duration
                    },
                    actionable=True,
                    recommendation="Consider providing more comprehensive responses"
                ))

        return patterns

    # =========================================================================
    # Private Insight Generation Methods
    # =========================================================================

    def _generate_recurring_recommendation(self, cluster: Cluster) -> str:
        """Generate recommendation for recurring question pattern"""
        if cluster.size >= 10:
            return f"Create FAQ entry for: '{cluster.centroid_text[:50]}...'"
        elif cluster.size >= 5:
            return f"Consider adding proactive information about: '{cluster.centroid_text[:30]}...'"
        return f"Monitor questions about: '{cluster.centroid_text[:30]}...'"

    def _generate_correlation_insight(
        self,
        factor: str,
        target: str,
        correlation: float
    ) -> str:
        """Generate insight text for a correlation"""
        direction = "positively" if correlation > 0 else "negatively"
        strength = (
            "strongly" if abs(correlation) > 0.7 else
            "moderately" if abs(correlation) > 0.4 else
            "weakly"
        )

        factor_name = factor.replace('_', ' ')
        target_name = target.replace('_', ' ')

        return f"{factor_name.title()} is {strength} {direction} correlated with {target_name}"

    def _generate_insight_title(self, pattern: DetectedPattern) -> str:
        """Generate title for pattern insight"""
        titles = {
            PatternType.RECURRING_QUESTION: "Recurring Question Pattern",
            PatternType.COMMON_CORRECTION: "Common Correction Needed",
            PatternType.ENGAGEMENT_TIME: "Engagement Time Pattern",
            PatternType.TOPIC_CLUSTER: "Popular Topic Cluster",
            PatternType.QUALITY_CORRELATION: "Quality Factor Identified",
            PatternType.VOCABULARY_PREFERENCE: "Vocabulary Preference",
            PatternType.RESPONSE_LENGTH: "Response Length Pattern",
            PatternType.CLARIFICATION_TRIGGER: "Clarification Trigger"
        }
        return titles.get(pattern.pattern_type, "Pattern Detected")

    def _calculate_insight_priority(self, pattern: DetectedPattern) -> float:
        """Calculate priority score for an insight"""
        # Base priority from confidence and frequency
        base = pattern.confidence * 0.4 + min(pattern.frequency / 20, 1.0) * 0.3

        # Boost for certain pattern types
        type_boost = {
            PatternType.RECURRING_QUESTION: 0.2,
            PatternType.COMMON_CORRECTION: 0.3,
            PatternType.QUALITY_CORRELATION: 0.2,
            PatternType.CLARIFICATION_TRIGGER: 0.25
        }

        return base + type_boost.get(pattern.pattern_type, 0.1)


# Global pattern detector instance
pattern_detector = PatternDetector()
