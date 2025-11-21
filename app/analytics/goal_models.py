"""
Goal Models - Pydantic Models for Goal Tracking and Achievements
PATTERN: Contract-first development with gamification elements
WHY: Type-safe goal tracking with rich achievement system

Models:
- Goal: Learning goal with target, progress, and deadline
- GoalProgress: Progress snapshot with percentage and remaining
- Milestone: Checkpoint within a goal
- Achievement: Badge with rarity and unlock status
- GoalSuggestion: AI-generated goal recommendation
"""
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class GoalType(str, Enum):
    """Types of learning goals."""
    STREAK = "streak"          # Maintain X-day streak
    SESSIONS = "sessions"      # Complete X sessions
    TOPICS = "topics"          # Explore X topics
    QUALITY = "quality"        # Achieve X% quality score
    EXCHANGES = "exchanges"    # Complete X exchanges
    DURATION = "duration"      # Learn for X minutes
    FEEDBACK = "feedback"      # Provide X feedback items
    CUSTOM = "custom"          # User-defined goals


class GoalStatus(str, Enum):
    """Status of a goal."""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    ABANDONED = "abandoned"
    EXPIRED = "expired"


class AchievementRarity(str, Enum):
    """Rarity tier for achievements."""
    COMMON = "common"          # Easy to obtain
    UNCOMMON = "uncommon"      # Requires some effort
    RARE = "rare"              # Notable achievement
    EPIC = "epic"              # Significant milestone
    LEGENDARY = "legendary"    # Exceptional accomplishment


class AchievementCategory(str, Enum):
    """Categories of achievements."""
    BEGINNER = "beginner"      # Starting out
    STREAK = "streak"          # Consistency achievements
    QUALITY = "quality"        # Quality-related
    EXPLORATION = "exploration" # Topic diversity
    ENGAGEMENT = "engagement"  # Active participation
    MASTERY = "mastery"        # Advanced achievements
    SOCIAL = "social"          # Sharing and collaboration
    MILESTONE = "milestone"    # Progress milestones


# ============================================================================
# GOAL MODELS
# ============================================================================

class Milestone(BaseModel):
    """
    Checkpoint within a goal.

    PATTERN: Progress gamification
    WHY: Break large goals into achievable steps
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = Field(..., description="Parent goal ID")
    title: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    target_value: float = Field(..., ge=0, description="Target for this milestone")
    current_value: float = Field(default=0, ge=0)
    order: int = Field(..., ge=0, description="Order in goal progression")
    completed: bool = Field(default=False)
    completed_at: Optional[datetime] = None
    reward_message: Optional[str] = Field(None, max_length=200)

    @property
    def progress_percent(self) -> float:
        """Calculate milestone progress percentage."""
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "ms_abc123",
                "goal_id": "goal_xyz789",
                "title": "First Week",
                "target_value": 7,
                "current_value": 3,
                "order": 1
            }
        }


class Goal(BaseModel):
    """
    Learning goal with target, progress, and optional deadline.

    PATTERN: SMART goal framework
    WHY: Specific, Measurable, Achievable, Relevant, Time-bound goals
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    goal_type: GoalType = Field(..., description="Type of goal")
    target_value: float = Field(..., gt=0, description="Target value to achieve")
    current_value: float = Field(default=0, ge=0, description="Current progress")
    unit: str = Field(default="", max_length=50, description="Unit of measurement")
    status: GoalStatus = Field(default=GoalStatus.ACTIVE)
    deadline: Optional[date] = Field(None, description="Optional deadline")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    milestones: List[Milestone] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @property
    def progress_percent(self) -> float:
        """Calculate goal progress percentage."""
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)

    @property
    def is_completed(self) -> bool:
        """Check if goal is completed."""
        return self.current_value >= self.target_value

    @property
    def days_remaining(self) -> Optional[int]:
        """Days until deadline, if set."""
        if self.deadline is None:
            return None
        today = date.today()
        delta = self.deadline - today
        return max(0, delta.days)

    @property
    def is_expired(self) -> bool:
        """Check if goal deadline has passed."""
        if self.deadline is None:
            return False
        return date.today() > self.deadline

    @property
    def completion_rate(self) -> float:
        """Calculate daily completion rate needed."""
        if self.deadline is None or self.days_remaining == 0:
            return 0.0
        remaining = self.target_value - self.current_value
        return remaining / self.days_remaining if self.days_remaining > 0 else remaining

    @validator('deadline')
    def deadline_must_be_future(cls, v, values):
        """Validate deadline is in the future for new goals."""
        if v is not None and 'created_at' not in values:
            # Only validate for new goals
            if v < date.today():
                raise ValueError('Deadline must be in the future')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "goal_xyz789",
                "title": "Build a 7-Day Streak",
                "description": "Maintain a learning streak for 7 consecutive days",
                "goal_type": "streak",
                "target_value": 7,
                "current_value": 3,
                "unit": "days",
                "status": "active",
                "deadline": "2025-11-30"
            }
        }


class GoalProgress(BaseModel):
    """
    Snapshot of goal progress at a point in time.

    PATTERN: Time-series tracking
    WHY: Track progress history for analysis
    """
    goal_id: str = Field(..., description="Goal ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    value: float = Field(..., ge=0, description="Progress value at snapshot")
    progress_percent: float = Field(..., ge=0, le=100)
    delta: float = Field(default=0, description="Change since last snapshot")
    source: Optional[str] = Field(None, description="What triggered the update")

    class Config:
        json_schema_extra = {
            "example": {
                "goal_id": "goal_xyz789",
                "timestamp": "2025-11-21T10:00:00Z",
                "value": 5,
                "progress_percent": 71.4,
                "delta": 1,
                "source": "session_complete"
            }
        }


# ============================================================================
# ACHIEVEMENT MODELS
# ============================================================================

class Achievement(BaseModel):
    """
    Achievement/badge with unlock requirements.

    PATTERN: Gamification with progressive unlocks
    WHY: Motivate and reward learning behavior
    """
    id: str = Field(..., description="Unique achievement ID")
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    icon: str = Field(..., description="Emoji or icon identifier")
    category: AchievementCategory = Field(...)
    rarity: AchievementRarity = Field(default=AchievementRarity.COMMON)
    requirement: str = Field(..., description="Human-readable requirement")
    requirement_type: str = Field(..., description="Metric type for checking")
    requirement_value: float = Field(..., gt=0, description="Value needed")
    points: int = Field(default=10, ge=0, description="Points awarded")
    unlocked: bool = Field(default=False)
    unlocked_at: Optional[datetime] = None
    progress: float = Field(default=0, ge=0, description="Current progress")
    hidden: bool = Field(default=False, description="Hidden until unlocked")
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @property
    def progress_percent(self) -> float:
        """Calculate achievement progress percentage."""
        if self.requirement_value == 0:
            return 100.0 if self.unlocked else 0.0
        return min(100.0, (self.progress / self.requirement_value) * 100)

    def check_unlock(self, value: float) -> bool:
        """Check if achievement should be unlocked."""
        return value >= self.requirement_value

    class Config:
        json_schema_extra = {
            "example": {
                "id": "first-steps",
                "title": "First Steps",
                "description": "Complete your first learning session",
                "icon": "baby",
                "category": "beginner",
                "rarity": "common",
                "requirement": "Complete 1 session",
                "requirement_type": "sessions",
                "requirement_value": 1,
                "points": 10,
                "unlocked": True,
                "unlocked_at": "2025-11-21T10:00:00Z"
            }
        }


# ============================================================================
# SUGGESTION MODELS
# ============================================================================

class GoalSuggestion(BaseModel):
    """
    AI-generated goal suggestion based on user patterns.

    PATTERN: Personalized recommendations
    WHY: Help users set appropriate goals
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=500)
    goal_type: GoalType = Field(...)
    suggested_target: float = Field(..., gt=0)
    suggested_deadline_days: Optional[int] = Field(None, ge=1)
    confidence: float = Field(..., ge=0, le=1, description="Suggestion confidence")
    reason: str = Field(..., description="Why this goal is suggested")
    based_on: str = Field(..., description="What data informed this suggestion")
    difficulty: str = Field(default="moderate", description="easy/moderate/challenging")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sug_abc123",
                "title": "Build a 7-Day Streak",
                "description": "You've been consistent - try maintaining a week-long streak!",
                "goal_type": "streak",
                "suggested_target": 7,
                "suggested_deadline_days": 14,
                "confidence": 0.85,
                "reason": "Based on your current 3-day streak",
                "based_on": "streak_history",
                "difficulty": "moderate"
            }
        }


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class CreateGoalRequest(BaseModel):
    """Request model for creating a goal."""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    goal_type: GoalType = Field(...)
    target_value: float = Field(..., gt=0)
    unit: Optional[str] = Field(None, max_length=50)
    deadline: Optional[date] = None

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Complete 10 Sessions",
                "description": "Practice learning with 10 conversation sessions",
                "goal_type": "sessions",
                "target_value": 10,
                "unit": "sessions",
                "deadline": "2025-12-31"
            }
        }


class UpdateGoalRequest(BaseModel):
    """Request model for updating a goal."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    target_value: Optional[float] = Field(None, gt=0)
    deadline: Optional[date] = None
    status: Optional[GoalStatus] = None

    class Config:
        json_schema_extra = {
            "example": {
                "target_value": 15,
                "deadline": "2025-12-31"
            }
        }


class GoalResponse(BaseModel):
    """Response model for goal operations."""
    goal: Goal
    recently_completed_milestones: List[Milestone] = Field(default_factory=list)
    new_achievements: List[Achievement] = Field(default_factory=list)


class GoalListResponse(BaseModel):
    """Response model for listing goals."""
    goals: List[Goal]
    total_count: int
    active_count: int
    completed_count: int


class AchievementListResponse(BaseModel):
    """Response model for listing achievements."""
    achievements: List[Achievement]
    total_count: int
    unlocked_count: int
    total_points: int
    categories: Dict[str, int]  # Category -> unlocked count
