"""
Data Serializers - Serialization and Deserialization Helpers
============================================================

PATTERN: Data Transfer Object (DTO) serialization
WHY: Clean separation between database models and export formats

Features:
- Convert database models to portable dictionaries
- Deserialize imported data back to models
- Handle datetime conversions and enums
- Validate data integrity during import
"""

from datetime import datetime, date
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from pydantic import BaseModel, Field

from app.learning.models import (
    Feedback,
    FeedbackType,
    FeedbackSource,
    QualityScore,
    UserPreference,
)
from app.learning.feedback_models import (
    ExplicitFeedback,
    ImplicitFeedback,
    CorrectionFeedback,
    CorrectionType,
    FeedbackSentiment,
)
from app.analytics.goal_models import (
    Goal,
    GoalType,
    GoalStatus,
    Milestone,
    GoalProgress,
    Achievement,
    AchievementRarity,
    AchievementCategory,
)
from app.logger import get_logger

logger = get_logger("sync.serializers")


# ============================================================================
# DATA TRANSFER OBJECTS
# ============================================================================

@dataclass
class ConversationData:
    """
    Serializable conversation exchange data.

    Represents a single conversation exchange in export format.
    """
    id: int
    session_id: str
    timestamp: str
    user_text: str
    agent_text: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationData":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FeedbackData:
    """
    Serializable feedback data combining all feedback types.
    """
    id: str
    session_id: str
    feedback_type: str
    timestamp: str
    # Explicit feedback fields
    rating: Optional[int] = None
    helpful: Optional[bool] = None
    comment: Optional[str] = None
    exchange_id: Optional[str] = None
    # Implicit feedback fields
    response_time_ms: Optional[int] = None
    engagement_score: Optional[float] = None
    follow_up_count: Optional[int] = None
    # Correction fields
    original_text: Optional[str] = None
    corrected_text: Optional[str] = None
    correction_type: Optional[str] = None
    # Common
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackData":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SettingsData(BaseModel):
    """User settings export format."""
    version: str = "1.0"
    preferences: Dict[str, Any] = Field(default_factory=dict)
    learning_config: Dict[str, Any] = Field(default_factory=dict)
    notification_settings: Dict[str, Any] = Field(default_factory=dict)
    privacy_settings: Dict[str, Any] = Field(default_factory=dict)
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# CONVERSATION SERIALIZERS
# ============================================================================

def serialize_conversation(
    exchange: Dict[str, Any],
    include_metadata: bool = True
) -> ConversationData:
    """
    Serialize a conversation exchange to export format.

    Args:
        exchange: Database row dict with conversation data
        include_metadata: Whether to include metadata

    Returns:
        ConversationData instance
    """
    try:
        timestamp = exchange.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        elif timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        metadata = None
        if include_metadata and exchange.get("metadata"):
            metadata = exchange["metadata"]
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {"raw": metadata}

        return ConversationData(
            id=exchange.get("id", 0),
            session_id=exchange.get("session_id", ""),
            timestamp=timestamp,
            user_text=exchange.get("user_text", ""),
            agent_text=exchange.get("agent_text", ""),
            metadata=metadata
        )

    except Exception as e:
        logger.error("serialize_conversation_failed", error=str(e))
        raise


def deserialize_conversation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize conversation data for import.

    Args:
        data: Serialized conversation dict

    Returns:
        Dictionary ready for database insertion
    """
    try:
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return {
            "session_id": data["session_id"],
            "timestamp": timestamp,
            "user_text": data["user_text"],
            "agent_text": data["agent_text"],
            "metadata": data.get("metadata")
        }

    except Exception as e:
        logger.error("deserialize_conversation_failed", error=str(e))
        raise ValueError(f"Invalid conversation data: {e}")


# ============================================================================
# FEEDBACK SERIALIZERS
# ============================================================================

def serialize_feedback(
    feedback: Union[ExplicitFeedback, ImplicitFeedback, CorrectionFeedback, Feedback]
) -> FeedbackData:
    """
    Serialize feedback to export format.

    Handles all feedback types: explicit, implicit, and corrections.

    Args:
        feedback: Any feedback model instance

    Returns:
        FeedbackData instance
    """
    try:
        # Determine feedback type and extract common fields
        if isinstance(feedback, ExplicitFeedback):
            return FeedbackData(
                id=feedback.id or "",
                session_id=feedback.session_id,
                feedback_type="explicit",
                timestamp=feedback.timestamp.isoformat(),
                rating=feedback.rating,
                helpful=feedback.helpful,
                comment=feedback.comment,
                exchange_id=feedback.exchange_id,
                metadata=feedback.metadata
            )

        elif isinstance(feedback, ImplicitFeedback):
            return FeedbackData(
                id=feedback.id or "",
                session_id=feedback.session_id,
                feedback_type="implicit",
                timestamp=feedback.timestamp.isoformat(),
                response_time_ms=feedback.response_time_ms,
                engagement_score=feedback.engagement_score,
                follow_up_count=feedback.follow_up_count,
                metadata=feedback.metadata
            )

        elif isinstance(feedback, CorrectionFeedback):
            return FeedbackData(
                id=feedback.id or "",
                session_id=feedback.session_id,
                feedback_type="correction",
                timestamp=feedback.timestamp.isoformat(),
                original_text=feedback.original_text,
                corrected_text=feedback.corrected_text,
                correction_type=feedback.correction_type.value if feedback.correction_type else None,
                metadata=feedback.metadata
            )

        elif isinstance(feedback, Feedback):
            return FeedbackData(
                id=feedback.id,
                session_id=feedback.session_id,
                feedback_type=feedback.feedback_type.value if isinstance(feedback.feedback_type, Enum) else feedback.feedback_type,
                timestamp=feedback.timestamp.isoformat(),
                rating=int(feedback.rating * 5) if feedback.rating else None,  # Convert 0-1 to 1-5
                comment=feedback.text,
                metadata=feedback.metadata
            )

        else:
            raise ValueError(f"Unknown feedback type: {type(feedback)}")

    except Exception as e:
        logger.error("serialize_feedback_failed", error=str(e), feedback_type=type(feedback).__name__)
        raise


def deserialize_feedback(data: Dict[str, Any]) -> Union[ExplicitFeedback, ImplicitFeedback, CorrectionFeedback]:
    """
    Deserialize feedback data for import.

    Args:
        data: Serialized feedback dict

    Returns:
        Appropriate feedback model instance
    """
    try:
        feedback_type = data.get("feedback_type", "explicit")
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        if feedback_type == "explicit":
            return ExplicitFeedback(
                id=data.get("id"),
                session_id=data["session_id"],
                exchange_id=data.get("exchange_id", ""),
                rating=data.get("rating", 3),
                helpful=data.get("helpful", True),
                comment=data.get("comment"),
                timestamp=timestamp,
                metadata=data.get("metadata")
            )

        elif feedback_type == "implicit":
            return ImplicitFeedback(
                id=data.get("id"),
                session_id=data["session_id"],
                response_time_ms=data.get("response_time_ms", 0),
                follow_up_count=data.get("follow_up_count", 0),
                timestamp=timestamp,
                metadata=data.get("metadata")
            )

        elif feedback_type == "correction":
            correction_type = data.get("correction_type", "rephrase")
            if isinstance(correction_type, str):
                correction_type = CorrectionType(correction_type)

            return CorrectionFeedback(
                id=data.get("id"),
                session_id=data["session_id"],
                original_text=data.get("original_text", ""),
                corrected_text=data.get("corrected_text", ""),
                correction_type=correction_type,
                timestamp=timestamp,
                metadata=data.get("metadata")
            )

        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

    except Exception as e:
        logger.error("deserialize_feedback_failed", error=str(e))
        raise ValueError(f"Invalid feedback data: {e}")


# ============================================================================
# GOAL SERIALIZERS
# ============================================================================

def serialize_goal(goal: Goal) -> Dict[str, Any]:
    """
    Serialize a goal to export format.

    Args:
        goal: Goal model instance

    Returns:
        Dictionary representation for export
    """
    try:
        return {
            "id": goal.id,
            "title": goal.title,
            "description": goal.description,
            "goal_type": goal.goal_type.value,
            "target_value": goal.target_value,
            "current_value": goal.current_value,
            "unit": goal.unit,
            "status": goal.status.value,
            "deadline": goal.deadline.isoformat() if goal.deadline else None,
            "created_at": goal.created_at.isoformat(),
            "updated_at": goal.updated_at.isoformat(),
            "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
            "progress_percent": goal.progress_percent,
            "days_remaining": goal.days_remaining,
            "milestones": [serialize_milestone(m) for m in goal.milestones],
            "metadata": goal.metadata
        }

    except Exception as e:
        logger.error("serialize_goal_failed", goal_id=goal.id, error=str(e))
        raise


def serialize_milestone(milestone: Milestone) -> Dict[str, Any]:
    """Serialize a milestone to export format."""
    return {
        "id": milestone.id,
        "goal_id": milestone.goal_id,
        "title": milestone.title,
        "description": milestone.description,
        "target_value": milestone.target_value,
        "current_value": milestone.current_value,
        "order": milestone.order,
        "completed": milestone.completed,
        "completed_at": milestone.completed_at.isoformat() if milestone.completed_at else None,
        "reward_message": milestone.reward_message,
        "progress_percent": milestone.progress_percent
    }


def deserialize_goal(data: Dict[str, Any]) -> Goal:
    """
    Deserialize goal data for import.

    Args:
        data: Serialized goal dict

    Returns:
        Goal model instance
    """
    try:
        # Parse dates
        deadline = None
        if data.get("deadline"):
            deadline = date.fromisoformat(data["deadline"])

        created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(data.get("updated_at", data["created_at"]).replace("Z", "+00:00"))

        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))

        # Parse milestones
        milestones = []
        for ms_data in data.get("milestones", []):
            milestones.append(deserialize_milestone(ms_data))

        return Goal(
            id=data["id"],
            title=data["title"],
            description=data.get("description"),
            goal_type=GoalType(data["goal_type"]),
            target_value=data["target_value"],
            current_value=data.get("current_value", 0),
            unit=data.get("unit", ""),
            status=GoalStatus(data.get("status", "active")),
            deadline=deadline,
            created_at=created_at,
            updated_at=updated_at,
            completed_at=completed_at,
            milestones=milestones,
            metadata=data.get("metadata")
        )

    except Exception as e:
        logger.error("deserialize_goal_failed", error=str(e))
        raise ValueError(f"Invalid goal data: {e}")


def deserialize_milestone(data: Dict[str, Any]) -> Milestone:
    """Deserialize milestone data for import."""
    completed_at = None
    if data.get("completed_at"):
        completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))

    return Milestone(
        id=data["id"],
        goal_id=data["goal_id"],
        title=data["title"],
        description=data.get("description"),
        target_value=data["target_value"],
        current_value=data.get("current_value", 0),
        order=data.get("order", 0),
        completed=data.get("completed", False),
        completed_at=completed_at,
        reward_message=data.get("reward_message")
    )


# ============================================================================
# ACHIEVEMENT SERIALIZERS
# ============================================================================

def serialize_achievement(achievement: Achievement) -> Dict[str, Any]:
    """
    Serialize an achievement to export format.

    Args:
        achievement: Achievement model instance

    Returns:
        Dictionary representation for export
    """
    try:
        return {
            "id": achievement.id,
            "title": achievement.title,
            "description": achievement.description,
            "icon": achievement.icon,
            "category": achievement.category.value,
            "rarity": achievement.rarity.value,
            "requirement": achievement.requirement,
            "requirement_type": achievement.requirement_type,
            "requirement_value": achievement.requirement_value,
            "points": achievement.points,
            "unlocked": achievement.unlocked,
            "unlocked_at": achievement.unlocked_at.isoformat() if achievement.unlocked_at else None,
            "progress": achievement.progress,
            "progress_percent": achievement.progress_percent,
            "hidden": achievement.hidden,
            "metadata": achievement.metadata
        }

    except Exception as e:
        logger.error("serialize_achievement_failed", achievement_id=achievement.id, error=str(e))
        raise


def deserialize_achievement(data: Dict[str, Any]) -> Achievement:
    """
    Deserialize achievement data for import.

    Args:
        data: Serialized achievement dict

    Returns:
        Achievement model instance
    """
    try:
        unlocked_at = None
        if data.get("unlocked_at"):
            unlocked_at = datetime.fromisoformat(data["unlocked_at"].replace("Z", "+00:00"))

        return Achievement(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            icon=data["icon"],
            category=AchievementCategory(data["category"]),
            rarity=AchievementRarity(data.get("rarity", "common")),
            requirement=data["requirement"],
            requirement_type=data["requirement_type"],
            requirement_value=data["requirement_value"],
            points=data.get("points", 10),
            unlocked=data.get("unlocked", False),
            unlocked_at=unlocked_at,
            progress=data.get("progress", 0),
            hidden=data.get("hidden", False),
            metadata=data.get("metadata")
        )

    except Exception as e:
        logger.error("deserialize_achievement_failed", error=str(e))
        raise ValueError(f"Invalid achievement data: {e}")


# ============================================================================
# QUALITY SCORE SERIALIZERS
# ============================================================================

def serialize_quality_score(score: QualityScore) -> Dict[str, Any]:
    """Serialize a quality score to export format."""
    return {
        "id": score.id,
        "query_id": score.query_id,
        "session_id": score.session_id,
        "relevance": score.relevance,
        "helpfulness": score.helpfulness,
        "engagement": score.engagement,
        "clarity": score.clarity,
        "accuracy": score.accuracy,
        "composite": score.composite,
        "timestamp": score.timestamp.isoformat(),
        "scoring_method": score.scoring_method,
        "confidence": score.confidence,
        "query_text": score.query_text,
        "response_text": score.response_text
    }


def deserialize_quality_score(data: Dict[str, Any]) -> QualityScore:
    """Deserialize quality score data for import."""
    return QualityScore(
        id=data.get("id"),
        query_id=data["query_id"],
        session_id=data["session_id"],
        relevance=data["relevance"],
        helpfulness=data["helpfulness"],
        engagement=data["engagement"],
        clarity=data["clarity"],
        accuracy=data["accuracy"],
        composite=data["composite"],
        timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        scoring_method=data.get("scoring_method", "default"),
        confidence=data.get("confidence", 1.0),
        query_text=data.get("query_text"),
        response_text=data.get("response_text")
    )


# ============================================================================
# USER PREFERENCE SERIALIZERS
# ============================================================================

def serialize_user_preference(pref: UserPreference) -> Dict[str, Any]:
    """Serialize a user preference to export format."""
    return {
        "id": pref.id,
        "user_id": pref.user_id,
        "session_id": pref.session_id,
        "category": pref.category,
        "value": pref.value,
        "confidence": pref.confidence,
        "learned_from_samples": pref.learned_from_samples,
        "last_updated": pref.last_updated.isoformat(),
        "created_at": pref.created_at.isoformat(),
        "decay_rate": pref.decay_rate
    }


def deserialize_user_preference(data: Dict[str, Any]) -> UserPreference:
    """Deserialize user preference data for import."""
    return UserPreference(
        id=data.get("id"),
        user_id=data.get("user_id"),
        session_id=data.get("session_id"),
        category=data["category"],
        value=data["value"],
        confidence=data.get("confidence", 0.5),
        learned_from_samples=data.get("learned_from_samples", 0),
        last_updated=datetime.fromisoformat(data["last_updated"].replace("Z", "+00:00")),
        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        decay_rate=data.get("decay_rate", 0.1)
    )


# ============================================================================
# BATCH SERIALIZATION HELPERS
# ============================================================================

def serialize_conversations_batch(
    exchanges: List[Dict[str, Any]],
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Serialize a batch of conversation exchanges.

    Args:
        exchanges: List of database row dicts
        include_metadata: Whether to include metadata

    Returns:
        List of serialized conversation dicts
    """
    return [
        serialize_conversation(ex, include_metadata).to_dict()
        for ex in exchanges
    ]


def serialize_feedback_batch(
    feedback_items: List[Union[ExplicitFeedback, ImplicitFeedback, CorrectionFeedback]]
) -> List[Dict[str, Any]]:
    """
    Serialize a batch of feedback items.

    Args:
        feedback_items: List of feedback model instances

    Returns:
        List of serialized feedback dicts
    """
    return [serialize_feedback(fb).to_dict() for fb in feedback_items]


def serialize_goals_batch(goals: List[Goal]) -> List[Dict[str, Any]]:
    """Serialize a batch of goals."""
    return [serialize_goal(g) for g in goals]


def serialize_achievements_batch(achievements: List[Achievement]) -> List[Dict[str, Any]]:
    """Serialize a batch of achievements."""
    return [serialize_achievement(a) for a in achievements]
