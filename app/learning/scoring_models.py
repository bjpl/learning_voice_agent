"""
Quality Scoring Data Models - SPARC Implementation

SPECIFICATION:
- Dataclasses for multi-dimensional quality scoring
- Aggregated session quality metrics
- Time-series quality trend tracking
- Pydantic models for API validation

ARCHITECTURE:
- Immutable dataclasses for scoring data
- Pydantic models for request/response validation
- Factory methods for score creation
- Serialization support for persistence
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import json


class ScoreDimension(str, Enum):
    """
    CONCEPT: Enumeration of scoring dimensions
    WHY: Type-safe dimension references across the system
    """
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    ENGAGEMENT = "engagement"
    CLARITY = "clarity"
    ACCURACY = "accuracy"


class QualityLevel(str, Enum):
    """
    CONCEPT: Categorical quality classification
    WHY: Human-readable quality interpretation
    """
    EXCELLENT = "excellent"    # >= 0.9
    GOOD = "good"             # >= 0.75
    SATISFACTORY = "satisfactory"  # >= 0.6
    NEEDS_IMPROVEMENT = "needs_improvement"  # >= 0.4
    POOR = "poor"             # < 0.4

    @classmethod
    def from_score(cls, score: float) -> 'QualityLevel':
        """Convert numeric score to quality level"""
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.75:
            return cls.GOOD
        elif score >= 0.6:
            return cls.SATISFACTORY
        elif score >= 0.4:
            return cls.NEEDS_IMPROVEMENT
        else:
            return cls.POOR


# Default weights constant (module-level for class-level access)
_DEFAULT_WEIGHTS = {
    "relevance": 0.30,
    "helpfulness": 0.25,
    "engagement": 0.20,
    "clarity": 0.15,
    "accuracy": 0.10
}


@dataclass
class QualityScore:
    """
    CONCEPT: Multi-dimensional quality score for a single response

    DIMENSIONS:
    - relevance (0-1): Semantic similarity between query and response
    - helpfulness (0-1): Weighted average of explicit user feedback
    - engagement (0-1): Follow-up rate and session depth metrics
    - clarity (0-1): Readability score (Flesch-Kincaid normalized)
    - accuracy (0-1): Self-consistency verification score

    USAGE:
        score = QualityScore(
            relevance=0.85,
            helpfulness=0.90,
            engagement=0.75,
            clarity=0.80,
            accuracy=0.88
        )
        composite = score.calculate_composite()
    """
    # Class-level constant
    DEFAULT_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: _DEFAULT_WEIGHTS.copy(),
        repr=False,
        compare=False
    )

    relevance: float = 0.0
    helpfulness: float = 0.0
    engagement: float = 0.0
    clarity: float = 0.0
    accuracy: float = 0.0
    composite: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Optional metadata for debugging and analysis
    response_id: Optional[str] = None
    session_id: Optional[str] = None
    query_length: Optional[int] = None
    response_length: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate score ranges and calculate composite if not set"""
        self._validate_scores()
        if self.composite == 0.0:
            self.composite = self.calculate_composite()

    def _validate_scores(self) -> None:
        """
        Ensure all scores are within valid range [0.0, 1.0]
        """
        for dim in ScoreDimension:
            value = getattr(self, dim.value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Score dimension '{dim.value}' must be between 0.0 and 1.0, got {value}"
                )

    def calculate_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted composite score

        ALGORITHM:
        - Default weights: relevance(30%), helpfulness(25%),
          engagement(20%), clarity(15%), accuracy(10%)
        - Custom weights must sum to 1.0

        Args:
            weights: Optional custom weight dictionary

        Returns:
            Weighted composite score (0.0 to 1.0)
        """
        weights = weights or self.DEFAULT_WEIGHTS

        # Validate weights sum to 1.0 (with small tolerance)
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        composite = sum(
            getattr(self, dim) * weight
            for dim, weight in weights.items()
        )

        return round(composite, 4)

    @property
    def quality_level(self) -> QualityLevel:
        """Get categorical quality level"""
        return QualityLevel.from_score(self.composite)

    @property
    def dimension_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed breakdown of each dimension with contribution
        """
        breakdown = {}
        for dim, weight in self.DEFAULT_WEIGHTS.items():
            value = getattr(self, dim)
            contribution = value * weight
            breakdown[dim] = {
                "score": round(value, 4),
                "weight": weight,
                "contribution": round(contribution, 4),
                "level": QualityLevel.from_score(value).value
            }
        return breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['quality_level'] = self.quality_level.value
        # Remove default weights from output
        if 'DEFAULT_WEIGHTS' in data:
            del data['DEFAULT_WEIGHTS']
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityScore':
        """Create QualityScore from dictionary"""
        # Handle timestamp conversion
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # Remove non-field keys
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class SessionQuality:
    """
    CONCEPT: Aggregated quality metrics for an entire session

    PURPOSE:
    - Track overall session effectiveness
    - Identify quality trends within session
    - Compare session quality across conversations

    USAGE:
        session = SessionQuality(session_id="abc-123")
        session.add_response_score(quality_score)
        print(f"Session average: {session.average_composite}")
    """
    session_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Aggregated scores (averages across all responses)
    average_relevance: float = 0.0
    average_helpfulness: float = 0.0
    average_engagement: float = 0.0
    average_clarity: float = 0.0
    average_accuracy: float = 0.0
    average_composite: float = 0.0

    # Session-level metrics
    response_count: int = 0
    total_user_turns: int = 0
    total_agent_turns: int = 0
    follow_up_count: int = 0

    # Quality distribution
    excellent_count: int = 0
    good_count: int = 0
    satisfactory_count: int = 0
    needs_improvement_count: int = 0
    poor_count: int = 0

    # Individual response scores (for detailed analysis)
    response_scores: List[QualityScore] = field(default_factory=list)

    # Trend indicators
    quality_trend: str = "stable"  # improving, declining, stable
    trend_slope: float = 0.0

    def add_response_score(self, score: QualityScore) -> None:
        """
        Add a response score and update aggregated metrics

        PATTERN: Running average calculation
        WHY: Memory-efficient aggregation without storing all scores
        """
        self.response_scores.append(score)
        self.response_count += 1

        # Update running averages
        n = self.response_count
        self.average_relevance = self._update_average(
            self.average_relevance, score.relevance, n
        )
        self.average_helpfulness = self._update_average(
            self.average_helpfulness, score.helpfulness, n
        )
        self.average_engagement = self._update_average(
            self.average_engagement, score.engagement, n
        )
        self.average_clarity = self._update_average(
            self.average_clarity, score.clarity, n
        )
        self.average_accuracy = self._update_average(
            self.average_accuracy, score.accuracy, n
        )
        self.average_composite = self._update_average(
            self.average_composite, score.composite, n
        )

        # Update quality distribution
        level = score.quality_level
        if level == QualityLevel.EXCELLENT:
            self.excellent_count += 1
        elif level == QualityLevel.GOOD:
            self.good_count += 1
        elif level == QualityLevel.SATISFACTORY:
            self.satisfactory_count += 1
        elif level == QualityLevel.NEEDS_IMPROVEMENT:
            self.needs_improvement_count += 1
        else:
            self.poor_count += 1

        # Update trend (simple linear regression on last 5 scores)
        self._update_trend()

    def _update_average(
        self, current_avg: float, new_value: float, n: int
    ) -> float:
        """
        Calculate running average efficiently

        FORMULA: new_avg = old_avg + (new_value - old_avg) / n
        WHY: Numerically stable incremental averaging
        """
        return current_avg + (new_value - current_avg) / n

    def _update_trend(self, window_size: int = 5) -> None:
        """
        Calculate quality trend using simple linear regression

        ALGORITHM: Least squares fit on recent scores
        OUTPUT: slope > 0.01 = improving, slope < -0.01 = declining
        """
        if len(self.response_scores) < 2:
            self.quality_trend = "stable"
            self.trend_slope = 0.0
            return

        # Use last N scores for trend calculation
        recent_scores = self.response_scores[-window_size:]
        n = len(recent_scores)

        # Simple linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(s.composite for s in recent_scores) / n

        numerator = sum(
            (i - x_mean) * (recent_scores[i].composite - y_mean)
            for i in range(n)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            self.trend_slope = numerator / denominator
        else:
            self.trend_slope = 0.0

        # Classify trend
        if self.trend_slope > 0.01:
            self.quality_trend = "improving"
        elif self.trend_slope < -0.01:
            self.quality_trend = "declining"
        else:
            self.quality_trend = "stable"

    @property
    def quality_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of quality levels"""
        if self.response_count == 0:
            return {level.value: 0.0 for level in QualityLevel}

        return {
            "excellent": self.excellent_count / self.response_count,
            "good": self.good_count / self.response_count,
            "satisfactory": self.satisfactory_count / self.response_count,
            "needs_improvement": self.needs_improvement_count / self.response_count,
            "poor": self.poor_count / self.response_count
        }

    @property
    def session_duration_seconds(self) -> float:
        """Calculate session duration in seconds"""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    @property
    def overall_quality_level(self) -> QualityLevel:
        """Get overall session quality level"""
        return QualityLevel.from_score(self.average_composite)

    def finalize(self) -> None:
        """Mark session as complete"""
        self.end_time = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "average_scores": {
                "relevance": round(self.average_relevance, 4),
                "helpfulness": round(self.average_helpfulness, 4),
                "engagement": round(self.average_engagement, 4),
                "clarity": round(self.average_clarity, 4),
                "accuracy": round(self.average_accuracy, 4),
                "composite": round(self.average_composite, 4)
            },
            "response_count": self.response_count,
            "quality_distribution": self.quality_distribution,
            "quality_trend": self.quality_trend,
            "trend_slope": round(self.trend_slope, 4),
            "overall_quality_level": self.overall_quality_level.value,
            "session_duration_seconds": round(self.session_duration_seconds, 2)
        }


@dataclass
class QualityTrend:
    """
    CONCEPT: Time-series quality data for trend analysis

    PURPOSE:
    - Track quality changes over time
    - Identify patterns (daily, weekly cycles)
    - Support quality improvement initiatives

    USAGE:
        trend = QualityTrend(time_range="7d")
        trend.add_data_point(timestamp, composite_score)
        analysis = trend.analyze()
    """
    time_range: str  # e.g., "24h", "7d", "30d"
    start_timestamp: datetime = field(default_factory=datetime.utcnow)
    end_timestamp: Optional[datetime] = None

    # Time-series data points
    data_points: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregated statistics
    min_score: float = 1.0
    max_score: float = 0.0
    mean_score: float = 0.0
    std_deviation: float = 0.0

    # Trend analysis
    overall_trend: str = "stable"  # improving, declining, stable, volatile
    trend_confidence: float = 0.0  # R-squared value

    # Dimensional trends
    dimension_trends: Dict[str, str] = field(default_factory=dict)

    # Anomalies and patterns
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)

    def add_data_point(
        self,
        timestamp: datetime,
        scores: QualityScore
    ) -> None:
        """
        Add a quality score data point to the time series
        """
        point = {
            "timestamp": timestamp.isoformat(),
            "composite": scores.composite,
            "relevance": scores.relevance,
            "helpfulness": scores.helpfulness,
            "engagement": scores.engagement,
            "clarity": scores.clarity,
            "accuracy": scores.accuracy
        }
        self.data_points.append(point)

        # Update min/max
        self.min_score = min(self.min_score, scores.composite)
        self.max_score = max(self.max_score, scores.composite)

        # Update running mean
        n = len(self.data_points)
        self.mean_score = self.mean_score + (scores.composite - self.mean_score) / n

    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis

        ALGORITHM:
        1. Calculate statistical measures
        2. Fit linear trend line
        3. Detect anomalies (Z-score > 2)
        4. Identify patterns (periodicity)

        Returns:
            Dictionary with analysis results
        """
        if len(self.data_points) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 data points for analysis"
            }

        # Calculate standard deviation
        scores = [p["composite"] for p in self.data_points]
        n = len(scores)
        variance = sum((s - self.mean_score) ** 2 for s in scores) / n
        self.std_deviation = variance ** 0.5

        # Calculate trend (linear regression)
        x_mean = (n - 1) / 2
        y_mean = self.mean_score

        numerator = sum(
            (i - x_mean) * (scores[i] - y_mean)
            for i in range(n)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator

            # Calculate R-squared
            y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
            ss_res = sum((scores[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((s - y_mean) ** 2 for s in scores)

            self.trend_confidence = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0
            self.trend_confidence = 0

        # Classify overall trend
        if self.std_deviation > 0.15:
            self.overall_trend = "volatile"
        elif slope > 0.005:
            self.overall_trend = "improving"
        elif slope < -0.005:
            self.overall_trend = "declining"
        else:
            self.overall_trend = "stable"

        # Detect anomalies (Z-score > 2)
        if self.std_deviation > 0:
            for i, score in enumerate(scores):
                z_score = abs(score - self.mean_score) / self.std_deviation
                if z_score > 2:
                    self.anomalies.append({
                        "index": i,
                        "score": score,
                        "z_score": round(z_score, 2),
                        "timestamp": self.data_points[i]["timestamp"]
                    })

        # Analyze dimension-specific trends
        for dim in ["relevance", "helpfulness", "engagement", "clarity", "accuracy"]:
            dim_scores = [p[dim] for p in self.data_points]
            dim_mean = sum(dim_scores) / n

            dim_numerator = sum(
                (i - x_mean) * (dim_scores[i] - dim_mean)
                for i in range(n)
            )

            if denominator > 0:
                dim_slope = dim_numerator / denominator
                if dim_slope > 0.005:
                    self.dimension_trends[dim] = "improving"
                elif dim_slope < -0.005:
                    self.dimension_trends[dim] = "declining"
                else:
                    self.dimension_trends[dim] = "stable"
            else:
                self.dimension_trends[dim] = "stable"

        self.end_timestamp = datetime.utcnow()

        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "time_range": self.time_range,
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "data_point_count": len(self.data_points),
            "statistics": {
                "min_score": round(self.min_score, 4),
                "max_score": round(self.max_score, 4),
                "mean_score": round(self.mean_score, 4),
                "std_deviation": round(self.std_deviation, 4)
            },
            "trend_analysis": {
                "overall_trend": self.overall_trend,
                "trend_confidence": round(self.trend_confidence, 4),
                "dimension_trends": self.dimension_trends
            },
            "anomalies": self.anomalies,
            "patterns": self.patterns
        }


@dataclass
class ImprovementArea:
    """
    CONCEPT: Identified area requiring quality improvement

    PURPOSE:
    - Highlight specific weaknesses
    - Provide actionable insights
    - Track improvement over time
    """
    dimension: str
    current_score: float
    target_score: float
    gap: float
    priority: str  # high, medium, low
    recommendations: List[str]
    affected_sessions: int
    improvement_potential: float  # Expected composite improvement if fixed

    @classmethod
    def from_analysis(
        cls,
        dimension: str,
        current_score: float,
        target_score: float = 0.75,
        affected_sessions: int = 0
    ) -> 'ImprovementArea':
        """Create improvement area from analysis"""
        gap = target_score - current_score

        # Determine priority based on gap and weight
        weights = _DEFAULT_WEIGHTS
        weighted_gap = gap * weights.get(dimension, 0.1)

        if weighted_gap > 0.1:
            priority = "high"
        elif weighted_gap > 0.05:
            priority = "medium"
        else:
            priority = "low"

        # Generate recommendations based on dimension
        recommendations = cls._generate_recommendations(dimension, current_score)

        return cls(
            dimension=dimension,
            current_score=current_score,
            target_score=target_score,
            gap=gap,
            priority=priority,
            recommendations=recommendations,
            affected_sessions=affected_sessions,
            improvement_potential=weighted_gap
        )

    @staticmethod
    def _generate_recommendations(dimension: str, score: float) -> List[str]:
        """Generate dimension-specific recommendations"""
        recommendations_map = {
            "relevance": [
                "Improve context retrieval for better response relevance",
                "Fine-tune embedding model for domain-specific queries",
                "Add query expansion for ambiguous questions"
            ],
            "helpfulness": [
                "Request explicit feedback after responses",
                "Provide more actionable and specific answers",
                "Include examples and step-by-step guidance"
            ],
            "engagement": [
                "Ask clarifying questions to encourage dialogue",
                "Offer related topics for exploration",
                "Personalize responses based on user history"
            ],
            "clarity": [
                "Use simpler language and shorter sentences",
                "Structure responses with clear headings",
                "Add summaries for complex explanations"
            ],
            "accuracy": [
                "Implement fact verification checks",
                "Cite sources when providing information",
                "Add confidence indicators to uncertain responses"
            ]
        }

        base_recommendations = recommendations_map.get(dimension, [])

        # Add score-specific recommendations
        if score < 0.4:
            base_recommendations.insert(0, f"CRITICAL: {dimension} score is very low - immediate attention needed")
        elif score < 0.6:
            base_recommendations.insert(0, f"Priority improvement area for {dimension}")

        return base_recommendations[:3]  # Return top 3 recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "dimension": self.dimension,
            "current_score": round(self.current_score, 4),
            "target_score": round(self.target_score, 4),
            "gap": round(self.gap, 4),
            "priority": self.priority,
            "recommendations": self.recommendations,
            "affected_sessions": self.affected_sessions,
            "improvement_potential": round(self.improvement_potential, 4)
        }


# Type aliases for clarity
ScoreDict = Dict[str, float]
TrendData = List[Dict[str, Any]]
