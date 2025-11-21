"""
Test Suite for Achievement System
=================================

Comprehensive tests for achievement and badge functionality.
Target: 20+ tests covering all achievement features.
"""

import pytest
from datetime import datetime, date
from unittest.mock import AsyncMock, MagicMock, patch

from app.analytics.goal_models import (
    Achievement, AchievementRarity, AchievementCategory
)
from app.analytics.goal_tracker import ProgressMetrics
from app.analytics.achievement_system import (
    AchievementSystem, AchievementCheckResult, ACHIEVEMENTS
)


class TestAchievementSystemInitialization:
    """Tests for AchievementSystem initialization."""

    @pytest.mark.asyncio
    async def test_initializes_successfully(self, achievement_system):
        """Test that achievement system initializes without errors."""
        await achievement_system.initialize()
        assert achievement_system._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_seeds_achievements(self, achievement_system, mock_goal_store):
        """Test that predefined achievements are seeded."""
        mock_goal_store.get_all_achievements = AsyncMock(return_value=[])
        mock_goal_store.save_achievement = AsyncMock()

        await achievement_system.initialize()

        # Should have called save_achievement for each predefined achievement
        assert mock_goal_store.save_achievement.call_count == len(ACHIEVEMENTS)


class TestAchievementChecking:
    """Tests for achievement checking."""

    @pytest.mark.asyncio
    async def test_check_achievements_returns_result(self, achievement_system, mock_goal_store):
        """Test that check_achievements returns proper result."""
        mock_goal_store.get_all_achievements = AsyncMock(return_value=[])
        await achievement_system.initialize()

        metrics = ProgressMetrics(total_sessions=10)
        result = await achievement_system.check_achievements(metrics)

        assert isinstance(result, AchievementCheckResult)
        assert hasattr(result, 'newly_unlocked')
        assert hasattr(result, 'progress_updated')
        assert hasattr(result, 'total_points_earned')

    @pytest.mark.asyncio
    async def test_check_achievements_unlocks_session_achievement(
        self, achievement_system, mock_goal_store
    ):
        """Test that session-based achievement is unlocked."""
        achievement = Achievement(
            id="first-steps",
            title="First Steps",
            description="Complete first session",
            icon="star",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.COMMON,
            requirement="Complete 1 session",
            requirement_type="sessions",
            requirement_value=1,
            points=10,
            unlocked=False
        )
        mock_goal_store.get_all_achievements = AsyncMock(return_value=[achievement])
        mock_goal_store.unlock_achievement = AsyncMock(return_value=achievement)
        await achievement_system.initialize()

        metrics = ProgressMetrics(total_sessions=5)
        result = await achievement_system.check_achievements(metrics)

        mock_goal_store.unlock_achievement.assert_called()

    @pytest.mark.asyncio
    async def test_check_achievements_skips_unlocked(self, achievement_system, mock_goal_store):
        """Test that already unlocked achievements are skipped."""
        achievement = Achievement(
            id="first-steps",
            title="First Steps",
            description="Complete first session",
            icon="star",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.COMMON,
            requirement="Complete 1 session",
            requirement_type="sessions",
            requirement_value=1,
            points=10,
            unlocked=True  # Already unlocked
        )
        mock_goal_store.get_all_achievements = AsyncMock(return_value=[achievement])
        mock_goal_store.unlock_achievement = AsyncMock()
        await achievement_system.initialize()

        metrics = ProgressMetrics(total_sessions=5)
        await achievement_system.check_achievements(metrics)

        mock_goal_store.unlock_achievement.assert_not_called()


class TestSessionAchievements:
    """Tests for session-specific achievements."""

    @pytest.mark.asyncio
    async def test_check_early_bird_achievement(self, achievement_system, mock_goal_store):
        """Test early bird achievement check."""
        achievement = Achievement(
            id="early-bird",
            title="Early Bird",
            description="Session before 8 AM",
            icon="sunrise",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.UNCOMMON,
            requirement="Early session",
            requirement_type="early_session",
            requirement_value=1,
            points=25,
            unlocked=False
        )
        mock_goal_store.get_all_achievements = AsyncMock(return_value=[achievement])
        mock_goal_store.unlock_achievement = AsyncMock(return_value=achievement)
        await achievement_system.initialize()

        metrics = ProgressMetrics(total_sessions=5)
        await achievement_system.check_session_achievements(
            session_start_hour=6,  # Before 8 AM
            session_duration_minutes=15,
            session_quality=0.8,
            metrics=metrics
        )

        mock_goal_store.unlock_achievement.assert_called()

    @pytest.mark.asyncio
    async def test_check_night_owl_achievement(self, achievement_system, mock_goal_store):
        """Test night owl achievement check."""
        achievement = Achievement(
            id="night-owl",
            title="Night Owl",
            description="Session after 10 PM",
            icon="moon",
            category=AchievementCategory.ENGAGEMENT,
            rarity=AchievementRarity.UNCOMMON,
            requirement="Late session",
            requirement_type="late_session",
            requirement_value=1,
            points=25,
            unlocked=False
        )
        mock_goal_store.get_all_achievements = AsyncMock(return_value=[achievement])
        mock_goal_store.unlock_achievement = AsyncMock(return_value=achievement)
        await achievement_system.initialize()

        metrics = ProgressMetrics(total_sessions=5)
        await achievement_system.check_session_achievements(
            session_start_hour=23,  # After 10 PM
            session_duration_minutes=15,
            session_quality=0.8,
            metrics=metrics
        )

        mock_goal_store.unlock_achievement.assert_called()


class TestAchievementRetrieval:
    """Tests for achievement retrieval."""

    @pytest.mark.asyncio
    async def test_get_all_achievements(self, achievement_system, mock_goal_store):
        """Test retrieving all achievements."""
        achievements = [
            Achievement(
                id="test",
                title="Test",
                description="Test",
                icon="star",
                category=AchievementCategory.BEGINNER,
                rarity=AchievementRarity.COMMON,
                requirement="Test",
                requirement_type="sessions",
                requirement_value=1,
                points=10
            )
        ]
        mock_goal_store.get_all_achievements = AsyncMock(return_value=achievements)
        await achievement_system.initialize()

        result = await achievement_system.get_all_achievements()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_all_achievements_hides_hidden(self, achievement_system, mock_goal_store):
        """Test that hidden achievements are filtered unless unlocked."""
        achievements = [
            Achievement(
                id="hidden",
                title="Hidden",
                description="Hidden achievement",
                icon="star",
                category=AchievementCategory.MASTERY,
                rarity=AchievementRarity.LEGENDARY,
                requirement="Secret",
                requirement_type="sessions",
                requirement_value=1000,
                points=500,
                hidden=True,
                unlocked=False
            )
        ]
        mock_goal_store.get_all_achievements = AsyncMock(return_value=achievements)
        await achievement_system.initialize()

        result = await achievement_system.get_all_achievements()

        assert len(result) == 0  # Hidden and not unlocked

    @pytest.mark.asyncio
    async def test_get_unlocked_achievements(self, achievement_system, mock_goal_store):
        """Test retrieving only unlocked achievements."""
        mock_goal_store.get_unlocked_achievements = AsyncMock(return_value=[])
        await achievement_system.initialize()

        result = await achievement_system.get_unlocked_achievements()

        assert isinstance(result, list)
        mock_goal_store.get_unlocked_achievements.assert_called()

    @pytest.mark.asyncio
    async def test_get_achievements_by_category(self, achievement_system, mock_goal_store):
        """Test retrieving achievements by category."""
        mock_goal_store.get_achievements_by_category = AsyncMock(return_value=[])
        await achievement_system.initialize()

        result = await achievement_system.get_achievements_by_category(AchievementCategory.STREAK)

        mock_goal_store.get_achievements_by_category.assert_called_with(AchievementCategory.STREAK)


class TestAchievementStats:
    """Tests for achievement statistics."""

    @pytest.mark.asyncio
    async def test_get_achievement_stats(self, achievement_system, mock_goal_store):
        """Test retrieving achievement statistics."""
        mock_goal_store.get_achievement_stats = AsyncMock(return_value={
            "total_achievements": 20,
            "unlocked_count": 5,
            "total_points": 150
        })
        mock_goal_store.get_unlocked_achievements = AsyncMock(return_value=[])
        await achievement_system.initialize()

        stats = await achievement_system.get_achievement_stats()

        assert "total_achievements" in stats
        assert "points_by_rarity" in stats


class TestNextAchievements:
    """Tests for next achievement suggestions."""

    @pytest.mark.asyncio
    async def test_get_next_achievements(self, achievement_system, mock_goal_store):
        """Test getting achievements closest to unlock."""
        achievements = [
            Achievement(
                id="test1",
                title="Test 1",
                description="Test",
                icon="star",
                category=AchievementCategory.BEGINNER,
                rarity=AchievementRarity.COMMON,
                requirement="Complete 10 sessions",
                requirement_type="sessions",
                requirement_value=10,
                points=25,
                progress=8,  # 80% complete
                unlocked=False
            ),
            Achievement(
                id="test2",
                title="Test 2",
                description="Test",
                icon="star",
                category=AchievementCategory.STREAK,
                rarity=AchievementRarity.RARE,
                requirement="7-day streak",
                requirement_type="streak",
                requirement_value=7,
                points=50,
                progress=3,  # ~43% complete
                unlocked=False
            )
        ]
        mock_goal_store.get_all_achievements = AsyncMock(return_value=achievements)
        await achievement_system.initialize()

        metrics = ProgressMetrics(
            total_sessions=8,
            current_streak=3
        )

        result = await achievement_system.get_next_achievements(metrics, limit=5)

        assert isinstance(result, list)


class TestStreakAchievements:
    """Tests for streak-specific achievement checks."""

    @pytest.mark.asyncio
    async def test_check_streak_achievements(self, achievement_system, mock_goal_store):
        """Test streak achievement checking."""
        achievement = Achievement(
            id="week-warrior",
            title="Week Warrior",
            description="7-day streak",
            icon="fire",
            category=AchievementCategory.STREAK,
            rarity=AchievementRarity.RARE,
            requirement="7-day streak",
            requirement_type="streak",
            requirement_value=7,
            points=50,
            unlocked=False
        )
        mock_goal_store.get_achievements_by_category = AsyncMock(return_value=[achievement])
        mock_goal_store.unlock_achievement = AsyncMock(return_value=achievement)
        await achievement_system.initialize()

        unlocked = await achievement_system.check_streak_achievements(
            current_streak=10,
            longest_streak=10
        )

        assert len(unlocked) == 1


class TestMilestoneAchievements:
    """Tests for milestone achievement checks."""

    @pytest.mark.asyncio
    async def test_check_milestone_achievements(self, achievement_system, mock_goal_store):
        """Test milestone achievement checking."""
        achievement = Achievement(
            id="ten-sessions",
            title="Double Digits",
            description="10 sessions",
            icon="number-10",
            category=AchievementCategory.MILESTONE,
            rarity=AchievementRarity.COMMON,
            requirement="10 sessions",
            requirement_type="sessions",
            requirement_value=10,
            points=25,
            unlocked=False
        )
        mock_goal_store.get_achievements_by_category = AsyncMock(return_value=[achievement])
        mock_goal_store.unlock_achievement = AsyncMock(return_value=achievement)
        await achievement_system.initialize()

        unlocked = await achievement_system.check_milestone_achievements(
            total_sessions=15,
            total_exchanges=100
        )

        assert len(unlocked) == 1
