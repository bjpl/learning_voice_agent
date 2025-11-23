"""
Achievement System - Badge and Recognition System
PATTERN: Gamification with progressive achievement unlocks
WHY: Motivate learners with recognition and rewards

Features:
- 15+ predefined achievements across 8 categories
- Automatic progress tracking and unlocking
- Rarity tiers (common to legendary)
- Points system for achievement hunting
- Integration with goal tracker
"""
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from app.analytics.goal_models import (
    Achievement, AchievementRarity, AchievementCategory
)
from app.analytics.goal_store import GoalStore, goal_store
from app.analytics.goal_tracker import ProgressMetrics
from app.logger import get_logger

# Module logger
logger = get_logger("achievement_system")


# ============================================================================
# PREDEFINED ACHIEVEMENTS
# ============================================================================

ACHIEVEMENTS: List[Achievement] = [
    # BEGINNER CATEGORY
    Achievement(
        id="first-steps",
        title="First Steps",
        description="Complete your first learning session",
        icon="baby",
        category=AchievementCategory.BEGINNER,
        rarity=AchievementRarity.COMMON,
        requirement="Complete 1 session",
        requirement_type="sessions",
        requirement_value=1,
        points=10
    ),
    Achievement(
        id="getting-started",
        title="Getting Started",
        description="Complete 5 learning sessions",
        icon="rocket",
        category=AchievementCategory.BEGINNER,
        rarity=AchievementRarity.COMMON,
        requirement="Complete 5 sessions",
        requirement_type="sessions",
        requirement_value=5,
        points=20
    ),
    Achievement(
        id="early-bird",
        title="Early Bird",
        description="Complete a session before 8 AM",
        icon="sunrise",
        category=AchievementCategory.BEGINNER,
        rarity=AchievementRarity.UNCOMMON,
        requirement="Session before 8 AM",
        requirement_type="early_session",
        requirement_value=1,
        points=25
    ),

    # STREAK CATEGORY
    Achievement(
        id="week-warrior",
        title="Week Warrior",
        description="Maintain a 7-day learning streak",
        icon="fire",
        category=AchievementCategory.STREAK,
        rarity=AchievementRarity.RARE,
        requirement="7-day streak",
        requirement_type="streak",
        requirement_value=7,
        points=50
    ),
    Achievement(
        id="fortnight-fighter",
        title="Fortnight Fighter",
        description="Maintain a 14-day learning streak",
        icon="flame",
        category=AchievementCategory.STREAK,
        rarity=AchievementRarity.RARE,
        requirement="14-day streak",
        requirement_type="streak",
        requirement_value=14,
        points=75
    ),
    Achievement(
        id="month-master",
        title="Month Master",
        description="Maintain a 30-day learning streak",
        icon="crown",
        category=AchievementCategory.STREAK,
        rarity=AchievementRarity.EPIC,
        requirement="30-day streak",
        requirement_type="streak",
        requirement_value=30,
        points=150
    ),
    Achievement(
        id="century-club",
        title="Century Club",
        description="Maintain a 100-day learning streak",
        icon="trophy",
        category=AchievementCategory.STREAK,
        rarity=AchievementRarity.LEGENDARY,
        requirement="100-day streak",
        requirement_type="streak",
        requirement_value=100,
        points=500,
        hidden=True
    ),

    # QUALITY CATEGORY
    Achievement(
        id="quality-champion",
        title="Quality Champion",
        description="Achieve 90%+ quality score for a week",
        icon="star",
        category=AchievementCategory.QUALITY,
        rarity=AchievementRarity.RARE,
        requirement="90%+ quality for 7 days",
        requirement_type="quality_streak",
        requirement_value=7,
        points=75
    ),
    Achievement(
        id="perfectionist",
        title="Perfectionist",
        description="Achieve a 95%+ quality score in a single session",
        icon="gem",
        category=AchievementCategory.QUALITY,
        rarity=AchievementRarity.RARE,
        requirement="95%+ quality in one session",
        requirement_type="session_quality",
        requirement_value=0.95,
        points=50
    ),
    Achievement(
        id="consistency-king",
        title="Consistency King",
        description="Maintain 80%+ quality for 30 days",
        icon="crown",
        category=AchievementCategory.QUALITY,
        rarity=AchievementRarity.EPIC,
        requirement="80%+ quality for 30 days",
        requirement_type="quality_streak_80",
        requirement_value=30,
        points=200
    ),

    # EXPLORATION CATEGORY
    Achievement(
        id="topic-explorer",
        title="Topic Explorer",
        description="Discuss 10 different topics",
        icon="compass",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.UNCOMMON,
        requirement="Explore 10 topics",
        requirement_type="topics",
        requirement_value=10,
        points=35
    ),
    Achievement(
        id="deep-diver",
        title="Deep Diver",
        description="Have 100+ exchanges on a single topic",
        icon="diving-mask",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.RARE,
        requirement="100+ exchanges on one topic",
        requirement_type="topic_exchanges",
        requirement_value=100,
        points=75
    ),
    Achievement(
        id="polymath",
        title="Polymath",
        description="Explore 50 different topics",
        icon="brain",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.EPIC,
        requirement="Explore 50 topics",
        requirement_type="topics",
        requirement_value=50,
        points=150
    ),

    # ENGAGEMENT CATEGORY
    Achievement(
        id="conversationalist",
        title="Conversationalist",
        description="Complete 100 total exchanges",
        icon="chat",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.UNCOMMON,
        requirement="100 total exchanges",
        requirement_type="exchanges",
        requirement_value=100,
        points=30
    ),
    Achievement(
        id="marathon-learner",
        title="Marathon Learner",
        description="Complete a session lasting 30+ minutes",
        icon="stopwatch",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.UNCOMMON,
        requirement="30+ minute session",
        requirement_type="session_duration",
        requirement_value=30,
        points=40
    ),
    Achievement(
        id="night-owl",
        title="Night Owl",
        description="Complete a session after 10 PM",
        icon="moon",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.UNCOMMON,
        requirement="Session after 10 PM",
        requirement_type="late_session",
        requirement_value=1,
        points=25
    ),

    # MASTERY CATEGORY
    Achievement(
        id="session-centurion",
        title="Session Centurion",
        description="Complete 100 learning sessions",
        icon="medal",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.EPIC,
        requirement="Complete 100 sessions",
        requirement_type="sessions",
        requirement_value=100,
        points=200
    ),
    Achievement(
        id="exchange-master",
        title="Exchange Master",
        description="Complete 1000 total exchanges",
        icon="bolt",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.EPIC,
        requirement="1000 total exchanges",
        requirement_type="exchanges",
        requirement_value=1000,
        points=250
    ),
    Achievement(
        id="learning-legend",
        title="Learning Legend",
        description="Complete 500 sessions with 85%+ average quality",
        icon="legend",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.LEGENDARY,
        requirement="500 sessions, 85%+ quality",
        requirement_type="mastery_sessions",
        requirement_value=500,
        points=1000,
        hidden=True
    ),

    # MILESTONE CATEGORY
    Achievement(
        id="ten-sessions",
        title="Double Digits",
        description="Complete 10 learning sessions",
        icon="number-10",
        category=AchievementCategory.MILESTONE,
        rarity=AchievementRarity.COMMON,
        requirement="Complete 10 sessions",
        requirement_type="sessions",
        requirement_value=10,
        points=25
    ),
    Achievement(
        id="fifty-sessions",
        title="Half Century",
        description="Complete 50 learning sessions",
        icon="number-50",
        category=AchievementCategory.MILESTONE,
        rarity=AchievementRarity.UNCOMMON,
        requirement="Complete 50 sessions",
        requirement_type="sessions",
        requirement_value=50,
        points=75
    ),
    Achievement(
        id="thousand-exchanges",
        title="Thousand Exchanges",
        description="Complete 1000 conversation exchanges",
        icon="1k",
        category=AchievementCategory.MILESTONE,
        rarity=AchievementRarity.RARE,
        requirement="Complete 1000 exchanges",
        requirement_type="exchanges",
        requirement_value=1000,
        points=100
    ),

    # SOCIAL CATEGORY
    Achievement(
        id="feedback-giver",
        title="Feedback Giver",
        description="Provide 10 feedback items",
        icon="thumbs-up",
        category=AchievementCategory.SOCIAL,
        rarity=AchievementRarity.COMMON,
        requirement="Provide 10 feedback items",
        requirement_type="feedback",
        requirement_value=10,
        points=20
    ),
    Achievement(
        id="feedback-champion",
        title="Feedback Champion",
        description="Provide 50 feedback items",
        icon="heart",
        category=AchievementCategory.SOCIAL,
        rarity=AchievementRarity.UNCOMMON,
        requirement="Provide 50 feedback items",
        requirement_type="feedback",
        requirement_value=50,
        points=50
    ),
]


@dataclass
class AchievementCheckResult:
    """Result of checking achievements."""
    newly_unlocked: List[Achievement]
    progress_updated: List[Achievement]
    total_points_earned: int


class AchievementSystem:
    """
    Achievement and badge management system.

    PATTERN: Progressive unlock gamification
    WHY: Recognize and reward learning achievements

    USAGE:
        system = AchievementSystem()
        await system.initialize()

        # Check achievements after session
        result = await system.check_achievements(metrics)

        # Get all achievements
        achievements = await system.get_all_achievements()
    """

    def __init__(self, store: Optional[GoalStore] = None):
        """
        Initialize the achievement system.

        Args:
            store: Goal store instance (uses global if not provided)
        """
        self.store = store or goal_store
        self._initialized = False
        self._achievements_seeded = False

    async def initialize(self) -> None:
        """Initialize the achievement system and seed achievements."""
        if self._initialized:
            return

        try:
            await self.store.initialize()

            # Seed predefined achievements
            await self._seed_achievements()

            self._initialized = True
            logger.info("achievement_system_initialized")

        except Exception as e:
            logger.error(
                "achievement_system_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def _seed_achievements(self) -> None:
        """Seed predefined achievements into the database."""
        if self._achievements_seeded:
            return

        try:
            existing = await self.store.get_all_achievements()
            existing_ids = {a.id for a in existing}

            seeded = 0
            for achievement in ACHIEVEMENTS:
                if achievement.id not in existing_ids:
                    await self.store.save_achievement(achievement)
                    seeded += 1

            self._achievements_seeded = True

            if seeded > 0:
                logger.info("achievements_seeded", count=seeded)

        except Exception as e:
            logger.error("seed_achievements_failed", error=str(e))

    # ========================================================================
    # ACHIEVEMENT CHECKING
    # ========================================================================

    async def check_achievements(
        self,
        metrics: ProgressMetrics,
        session_data: Optional[Dict[str, Any]] = None
    ) -> AchievementCheckResult:
        """
        Check and unlock achievements based on current metrics.

        Args:
            metrics: Current progress metrics
            session_data: Optional session-specific data (time, duration, etc.)

        Returns:
            AchievementCheckResult with newly unlocked and updated achievements
        """
        if not self._initialized:
            await self.initialize()

        try:
            achievements = await self.store.get_all_achievements()

            newly_unlocked = []
            progress_updated = []
            total_points = 0

            for achievement in achievements:
                if achievement.unlocked:
                    continue

                # Get current value for this achievement type
                current_value = self._get_value_for_requirement(
                    achievement.requirement_type,
                    metrics,
                    session_data
                )

                # Check if should unlock
                if current_value >= achievement.requirement_value:
                    unlocked = await self.store.unlock_achievement(
                        achievement.id,
                        current_value
                    )
                    if unlocked:
                        newly_unlocked.append(unlocked)
                        total_points += unlocked.points

                        logger.info(
                            "achievement_unlocked",
                            achievement_id=achievement.id,
                            title=achievement.title,
                            rarity=achievement.rarity.value,
                            points=achievement.points
                        )

                # Update progress if not unlocked
                elif current_value > achievement.progress:
                    updated = await self.store.update_achievement_progress(
                        achievement.id,
                        current_value
                    )
                    if updated:
                        progress_updated.append(updated)

            return AchievementCheckResult(
                newly_unlocked=newly_unlocked,
                progress_updated=progress_updated,
                total_points_earned=total_points
            )

        except Exception as e:
            logger.error("check_achievements_failed", error=str(e), exc_info=True)
            return AchievementCheckResult(
                newly_unlocked=[],
                progress_updated=[],
                total_points_earned=0
            )

    async def check_session_achievements(
        self,
        session_start_hour: int,
        session_duration_minutes: float,
        session_quality: float,
        metrics: ProgressMetrics
    ) -> AchievementCheckResult:
        """
        Check session-specific achievements after a session completes.

        Args:
            session_start_hour: Hour the session started (0-23)
            session_duration_minutes: Session duration in minutes
            session_quality: Quality score for the session (0-1)
            metrics: Current progress metrics

        Returns:
            AchievementCheckResult
        """
        session_data = {
            'start_hour': session_start_hour,
            'duration_minutes': session_duration_minutes,
            'quality': session_quality
        }

        return await self.check_achievements(metrics, session_data)

    def _get_value_for_requirement(
        self,
        requirement_type: str,
        metrics: ProgressMetrics,
        session_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get the current value for a requirement type."""
        # Basic metrics
        if requirement_type == "sessions":
            return metrics.total_sessions
        elif requirement_type == "exchanges":
            return metrics.total_exchanges
        elif requirement_type == "topics":
            return metrics.total_topics
        elif requirement_type == "streak":
            return metrics.current_streak
        elif requirement_type == "feedback":
            return metrics.total_feedback

        # Quality metrics
        elif requirement_type == "avg_quality":
            return metrics.avg_quality_score

        # Session-specific
        if session_data:
            if requirement_type == "early_session":
                return 1 if session_data.get('start_hour', 12) < 8 else 0
            elif requirement_type == "late_session":
                return 1 if session_data.get('start_hour', 12) >= 22 else 0
            elif requirement_type == "session_duration":
                return session_data.get('duration_minutes', 0)
            elif requirement_type == "session_quality":
                return session_data.get('quality', 0)

        # Complex requirements (need additional tracking)
        elif requirement_type == "quality_streak":
            # Would need separate tracking of consecutive quality days
            return 0
        elif requirement_type == "quality_streak_80":
            return 0
        elif requirement_type == "topic_exchanges":
            return 0
        elif requirement_type == "mastery_sessions":
            # Complex requirement: sessions with high quality
            return 0

        return 0

    # ========================================================================
    # ACHIEVEMENT RETRIEVAL
    # ========================================================================

    async def get_all_achievements(self) -> List[Achievement]:
        """Get all achievements."""
        if not self._initialized:
            await self.initialize()

        achievements = await self.store.get_all_achievements()

        # Filter hidden achievements that aren't unlocked
        return [
            a for a in achievements
            if not a.hidden or a.unlocked
        ]

    async def get_unlocked_achievements(self) -> List[Achievement]:
        """Get all unlocked achievements."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_unlocked_achievements()

    async def get_achievements_by_category(
        self,
        category: AchievementCategory
    ) -> List[Achievement]:
        """Get achievements by category."""
        if not self._initialized:
            await self.initialize()

        achievements = await self.store.get_achievements_by_category(category)

        # Filter hidden
        return [a for a in achievements if not a.hidden or a.unlocked]

    async def get_achievement(self, achievement_id: str) -> Optional[Achievement]:
        """Get a specific achievement."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_achievement(achievement_id)

    async def get_achievement_stats(self) -> Dict[str, Any]:
        """Get achievement statistics."""
        if not self._initialized:
            await self.initialize()

        stats = await self.store.get_achievement_stats()

        # Add rarity breakdown
        unlocked = await self.store.get_unlocked_achievements()
        rarity_points = {
            AchievementRarity.COMMON.value: 0,
            AchievementRarity.UNCOMMON.value: 0,
            AchievementRarity.RARE.value: 0,
            AchievementRarity.EPIC.value: 0,
            AchievementRarity.LEGENDARY.value: 0
        }

        for achievement in unlocked:
            rarity_points[achievement.rarity.value] += achievement.points

        stats['points_by_rarity'] = rarity_points

        return stats

    async def get_next_achievements(
        self,
        metrics: ProgressMetrics,
        limit: int = 5
    ) -> List[Tuple[Achievement, float]]:
        """
        Get achievements closest to being unlocked.

        Args:
            metrics: Current progress metrics
            limit: Maximum achievements to return

        Returns:
            List of (achievement, progress_percent) tuples
        """
        if not self._initialized:
            await self.initialize()

        try:
            achievements = await self.store.get_all_achievements()

            progress_list = []
            for achievement in achievements:
                if achievement.unlocked or achievement.hidden:
                    continue

                current = self._get_value_for_requirement(
                    achievement.requirement_type,
                    metrics
                )

                progress_pct = min(100.0, (current / achievement.requirement_value) * 100)

                if progress_pct > 0:
                    progress_list.append((achievement, progress_pct))

            # Sort by progress (closest to completion first)
            progress_list.sort(key=lambda x: x[1], reverse=True)

            return progress_list[:limit]

        except Exception as e:
            logger.error("get_next_achievements_failed", error=str(e))
            return []

    # ========================================================================
    # SPECIAL ACHIEVEMENT CHECKS
    # ========================================================================

    async def check_streak_achievements(
        self,
        current_streak: int,
        longest_streak: int
    ) -> List[Achievement]:
        """Check streak-related achievements."""
        if not self._initialized:
            await self.initialize()

        try:
            streak_achievements = await self.store.get_achievements_by_category(
                AchievementCategory.STREAK
            )

            unlocked = []
            for achievement in streak_achievements:
                if achievement.unlocked:
                    continue

                if achievement.requirement_type == "streak":
                    if current_streak >= achievement.requirement_value:
                        result = await self.store.unlock_achievement(
                            achievement.id,
                            current_streak
                        )
                        if result:
                            unlocked.append(result)

            return unlocked

        except Exception as e:
            logger.error("check_streak_achievements_failed", error=str(e))
            return []

    async def check_milestone_achievements(
        self,
        total_sessions: int,
        total_exchanges: int
    ) -> List[Achievement]:
        """Check milestone-related achievements."""
        if not self._initialized:
            await self.initialize()

        try:
            milestone_achievements = await self.store.get_achievements_by_category(
                AchievementCategory.MILESTONE
            )

            unlocked = []
            for achievement in milestone_achievements:
                if achievement.unlocked:
                    continue

                value = 0
                if achievement.requirement_type == "sessions":
                    value = total_sessions
                elif achievement.requirement_type == "exchanges":
                    value = total_exchanges

                if value >= achievement.requirement_value:
                    result = await self.store.unlock_achievement(
                        achievement.id,
                        value
                    )
                    if result:
                        unlocked.append(result)

            return unlocked

        except Exception as e:
            logger.error("check_milestone_achievements_failed", error=str(e))
            return []


# Singleton instance
achievement_system = AchievementSystem()
