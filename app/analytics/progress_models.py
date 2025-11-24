"""
Progress Tracking Data Models
=============================

Pydantic models for progress tracking and analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import uuid


class ProgressLevel(str, Enum):
    """Learning progress level categories."""
    BEGINNER = "beginner"
    DEVELOPING = "developing"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TrendDirection(str, Enum):
    """Direction of a trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class GoalStatus(str, Enum):
    """Status of a learning goal."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    PAUSED = "paused"


class GoalType(str, Enum):
    """Types of learning goals."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class AchievementTier(str, Enum):
    """Achievement tier levels."""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class LearningStreak(BaseModel):
    """Track consecutive learning days."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Streak data
    current_streak: int = 0
    longest_streak: int = 0
    last_active_date: Optional[date] = None
    streak_start_date: Optional[date] = None

    # Streak history
    streak_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update(self, activity_date: date) -> bool:
        """
        Update streak based on activity date.

        Returns True if streak continues, False if it reset.
        """
        if self.last_active_date is None:
            # First activity
            self.current_streak = 1
            self.longest_streak = max(1, self.longest_streak)
            self.streak_start_date = activity_date
            self.last_active_date = activity_date
            return True

        days_diff = (activity_date - self.last_active_date).days

        if days_diff == 0:
            # Same day, no change
            return True
        elif days_diff == 1:
            # Consecutive day - extend streak
            self.current_streak += 1
            self.longest_streak = max(self.current_streak, self.longest_streak)
            self.last_active_date = activity_date
            return True
        else:
            # Gap in streak - record history and reset
            if self.current_streak > 0:
                self.streak_history.append({
                    "start_date": self.streak_start_date.isoformat() if self.streak_start_date else None,
                    "end_date": self.last_active_date.isoformat(),
                    "length": self.current_streak,
                    "ended_at": datetime.utcnow().isoformat()
                })

            # Reset streak
            self.current_streak = 1
            self.streak_start_date = activity_date
            self.last_active_date = activity_date
            return False

        self.updated_at = datetime.utcnow()


class TopicMastery(BaseModel):
    """Track mastery level for a specific topic."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Mastery metrics
    mastery_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    level: ProgressLevel = ProgressLevel.BEGINNER

    # Interaction counts
    total_interactions: int = 0
    successful_interactions: int = 0

    # Quality tracking
    avg_quality_score: float = 0.0
    quality_trend: TrendDirection = TrendDirection.STABLE

    # Time tracking
    total_time_minutes: float = 0.0
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update_mastery(self, quality_score: float, success: bool = True) -> None:
        """Update mastery based on new interaction."""
        self.total_interactions += 1
        if success:
            self.successful_interactions += 1

        # Update average quality (exponential moving average)
        alpha = 0.2
        self.avg_quality_score = alpha * quality_score + (1 - alpha) * self.avg_quality_score

        # Calculate mastery score
        success_rate = self.successful_interactions / self.total_interactions if self.total_interactions > 0 else 0
        self.mastery_score = 0.6 * success_rate + 0.4 * self.avg_quality_score

        # Update level
        self._update_level()

        # Update timestamps
        if self.first_interaction is None:
            self.first_interaction = datetime.utcnow()
        self.last_interaction = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def _update_level(self) -> None:
        """Update mastery level based on score and interactions."""
        min_interactions = {
            ProgressLevel.DEVELOPING: 3,
            ProgressLevel.INTERMEDIATE: 10,
            ProgressLevel.ADVANCED: 25,
            ProgressLevel.EXPERT: 50
        }

        if self.mastery_score >= 0.9 and self.total_interactions >= min_interactions[ProgressLevel.EXPERT]:
            self.level = ProgressLevel.EXPERT
        elif self.mastery_score >= 0.75 and self.total_interactions >= min_interactions[ProgressLevel.ADVANCED]:
            self.level = ProgressLevel.ADVANCED
        elif self.mastery_score >= 0.6 and self.total_interactions >= min_interactions[ProgressLevel.INTERMEDIATE]:
            self.level = ProgressLevel.INTERMEDIATE
        elif self.total_interactions >= min_interactions[ProgressLevel.DEVELOPING]:
            self.level = ProgressLevel.DEVELOPING
        else:
            self.level = ProgressLevel.BEGINNER


class SessionProgress(BaseModel):
    """Progress data for a single session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: Optional[str] = None

    # Session metrics
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0

    # Interaction counts
    total_exchanges: int = 0
    questions_asked: int = 0
    clarifications_needed: int = 0

    # Quality metrics
    avg_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0

    # Topics covered
    topics: List[str] = Field(default_factory=list)
    primary_topic: Optional[str] = None

    # Feedback summary
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0

    # Learning velocity (exchanges per minute)
    learning_velocity: float = 0.0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def finalize(self, end_time: Optional[datetime] = None) -> None:
        """Finalize session progress calculations."""
        self.end_time = end_time or datetime.utcnow()
        self.duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        if self.duration_minutes > 0:
            self.learning_velocity = self.total_exchanges / self.duration_minutes


class DailyProgress(BaseModel):
    """Aggregated progress for a single day."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: date
    user_id: Optional[str] = None

    # Session counts
    total_sessions: int = 0
    completed_sessions: int = 0

    # Interaction metrics
    total_exchanges: int = 0
    total_time_minutes: float = 0.0
    avg_session_duration: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_trend: TrendDirection = TrendDirection.STABLE

    # Topic coverage
    topics_covered: List[str] = Field(default_factory=list)
    new_topics: List[str] = Field(default_factory=list)

    # Goals
    goals_completed: int = 0
    goals_in_progress: int = 0

    # Streak
    streak_maintained: bool = False
    current_streak: int = 0

    # Achievements
    achievements_unlocked: List[str] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class WeeklyProgress(BaseModel):
    """Aggregated progress for a week."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    week_start: date
    week_end: date
    user_id: Optional[str] = None

    # Daily breakdown
    daily_progress: List[DailyProgress] = Field(default_factory=list)
    active_days: int = 0

    # Aggregated metrics
    total_sessions: int = 0
    total_exchanges: int = 0
    total_time_minutes: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_trend: TrendDirection = TrendDirection.STABLE
    best_day: Optional[date] = None

    # Topics
    topics_covered: List[str] = Field(default_factory=list)
    topic_mastery_changes: Dict[str, float] = Field(default_factory=dict)

    # Goals
    goals_completed: int = 0
    goal_completion_rate: float = 0.0

    # Comparison to previous week
    vs_previous_week: Dict[str, float] = Field(default_factory=dict)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class MonthlyProgress(BaseModel):
    """Aggregated progress for a month."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    year: int
    month: int
    user_id: Optional[str] = None

    # Weekly breakdown
    weekly_progress: List[WeeklyProgress] = Field(default_factory=list)
    active_days: int = 0

    # Aggregated metrics
    total_sessions: int = 0
    total_exchanges: int = 0
    total_time_hours: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_trend: TrendDirection = TrendDirection.STABLE

    # Streaks
    longest_streak: int = 0
    current_streak: int = 0

    # Topics and mastery
    topics_mastered: List[str] = Field(default_factory=list)
    topic_progress: Dict[str, float] = Field(default_factory=dict)

    # Goals
    goals_set: int = 0
    goals_completed: int = 0
    goal_completion_rate: float = 0.0

    # Achievements
    achievements_unlocked: List[str] = Field(default_factory=list)

    # Comparison
    vs_previous_month: Dict[str, float] = Field(default_factory=dict)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ProgressMetrics(BaseModel):
    """Overall progress metrics summary."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None

    # Session metrics
    sessions_count: int = 0
    total_exchanges: int = 0
    total_time_hours: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_percentile: float = 0.0

    # Learning velocity
    learning_velocity: float = 0.0  # Exchanges per hour
    velocity_trend: TrendDirection = TrendDirection.STABLE

    # Streaks
    current_streak: int = 0
    longest_streak: int = 0

    # Topics
    topics_explored: int = 0
    topics_mastered: int = 0

    # Goals
    goals_completed: int = 0
    goals_in_progress: int = 0
    goal_completion_rate: float = 0.0

    # Achievements
    achievements_count: int = 0
    achievement_points: int = 0

    # Time-based
    first_session: Optional[datetime] = None
    last_session: Optional[datetime] = None
    most_active_hour: Optional[int] = None
    most_active_day: Optional[str] = None

    # Metadata
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class ProgressSnapshot(BaseModel):
    """Point-in-time progress snapshot for tracking changes."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    snapshot_time: datetime = Field(default_factory=datetime.utcnow)

    # Core metrics at snapshot time
    metrics: ProgressMetrics

    # Topic mastery at snapshot time
    topic_mastery: Dict[str, TopicMastery] = Field(default_factory=dict)

    # Active goals at snapshot time
    active_goals: List[str] = Field(default_factory=list)

    # Comparison to previous snapshot
    changes_from_previous: Dict[str, Any] = Field(default_factory=dict)


class LearningGoal(BaseModel):
    """A learning goal to track progress against."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Goal details
    title: str
    description: Optional[str] = None
    goal_type: GoalType = GoalType.CUSTOM

    # Target metrics
    target_metric: str  # e.g., "sessions", "exchanges", "time_minutes", "quality_score"
    target_value: float
    current_value: float = 0.0

    # Status
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress_percentage: float = 0.0

    # Time constraints
    start_date: date = Field(default_factory=date.today)
    end_date: Optional[date] = None

    # Rewards
    achievement_id: Optional[str] = None
    points: int = 0

    # History
    progress_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def update_progress(self, value: float) -> bool:
        """
        Update progress towards goal.

        Returns True if goal is completed.
        """
        self.current_value = value
        self.progress_percentage = min(100.0, (value / self.target_value) * 100) if self.target_value > 0 else 0.0

        # Record progress
        self.progress_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": value,
            "percentage": self.progress_percentage
        })

        # Update status
        if self.status == GoalStatus.NOT_STARTED and value > 0:
            self.status = GoalStatus.IN_PROGRESS

        # Check completion
        if value >= self.target_value:
            self.status = GoalStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            self.progress_percentage = 100.0
            return True

        # Check expiration
        if self.end_date and date.today() > self.end_date:
            if self.status != GoalStatus.COMPLETED:
                self.status = GoalStatus.EXPIRED

        self.updated_at = datetime.utcnow()
        return False


class Achievement(BaseModel):
    """An achievement or badge that can be unlocked."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Achievement details
    name: str
    description: str
    icon: str = "star"
    tier: AchievementTier = AchievementTier.BRONZE

    # Requirements
    requirement_type: str  # e.g., "streak", "sessions", "topics", "quality"
    requirement_value: float
    requirement_description: str

    # Rewards
    points: int = 10

    # Rarity (0.0 = common, 1.0 = legendary)
    rarity: float = 0.5

    # Categories/tags
    category: str = "general"
    tags: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UnlockedAchievement(BaseModel):
    """Record of an achievement being unlocked."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Achievement reference
    achievement_id: str
    achievement_name: str
    achievement_tier: AchievementTier

    # Unlock details
    unlocked_at: datetime = Field(default_factory=datetime.utcnow)
    unlocked_value: float  # The value that triggered the unlock

    # Points earned
    points_earned: int = 0

    # Context
    trigger_context: Dict[str, Any] = Field(default_factory=dict)


class DashboardData(BaseModel):
    """Data structure for dashboard display."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None

    # Overview metrics
    overview: ProgressMetrics

    # Current streak
    streak: LearningStreak

    # Recent progress
    daily_progress: List[DailyProgress] = Field(default_factory=list)
    weekly_summary: Optional[WeeklyProgress] = None

    # Active goals
    active_goals: List[LearningGoal] = Field(default_factory=list)

    # Recent achievements
    recent_achievements: List[UnlockedAchievement] = Field(default_factory=list)

    # Topic mastery
    topic_mastery: List[TopicMastery] = Field(default_factory=list)

    # Insights
    insights: List[Dict[str, Any]] = Field(default_factory=list)

    # Charts data
    quality_chart_data: List[Dict[str, Any]] = Field(default_factory=list)
    activity_heatmap_data: List[Dict[str, Any]] = Field(default_factory=list)
    topic_distribution_data: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cache_ttl_seconds: int = 300
