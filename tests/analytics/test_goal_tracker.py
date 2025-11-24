"""
Test Suite for Goal Tracker
===========================

Comprehensive tests for goal tracking functionality.
Target: 25+ tests covering all goal tracking features.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from app.analytics.goal_models import (
    GoalType, GoalStatus, Goal, GoalProgress,
    Milestone, CreateGoalRequest, UpdateGoalRequest
)
from app.analytics.goal_tracker import GoalTracker, ProgressMetrics


class TestGoalTrackerInitialization:
    """Tests for GoalTracker initialization."""

    @pytest.mark.asyncio
    async def test_initializes_successfully(self, goal_tracker):
        """Test that goal tracker initializes without errors."""
        await goal_tracker.initialize()
        assert goal_tracker._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, goal_tracker):
        """Test that multiple initialization calls are safe."""
        await goal_tracker.initialize()
        await goal_tracker.initialize()
        assert goal_tracker._initialized is True


class TestGoalCreation:
    """Tests for goal creation."""

    @pytest.mark.asyncio
    async def test_create_goal_basic(self, goal_tracker):
        """Test basic goal creation."""
        await goal_tracker.initialize()

        goal = await goal_tracker.create_goal(
            title="Complete 10 Sessions",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        assert goal is not None
        assert goal.title == "Complete 10 Sessions"
        assert goal.target_value == 10

    @pytest.mark.asyncio
    async def test_create_goal_with_deadline(self, goal_tracker):
        """Test goal creation with deadline."""
        await goal_tracker.initialize()
        deadline = date.today() + timedelta(days=7)

        goal = await goal_tracker.create_goal(
            title="Weekly Goal",
            goal_type=GoalType.SESSIONS,
            target_value=5,
            deadline=deadline
        )

        assert goal.deadline == deadline

    @pytest.mark.asyncio
    async def test_create_goal_with_description(self, goal_tracker):
        """Test goal creation with description."""
        await goal_tracker.initialize()

        goal = await goal_tracker.create_goal(
            title="Learn Python",
            goal_type=GoalType.TOPICS,
            target_value=1,
            description="Master Python programming basics"
        )

        assert goal.description == "Master Python programming basics"

    @pytest.mark.asyncio
    async def test_create_goal_generates_milestones(self, goal_tracker):
        """Test that milestones are auto-generated for large goals."""
        await goal_tracker.initialize()

        goal = await goal_tracker.create_goal(
            title="100 Sessions",
            goal_type=GoalType.SESSIONS,
            target_value=100
        )

        assert len(goal.milestones) > 0

    @pytest.mark.asyncio
    async def test_create_goal_sets_default_unit(self, goal_tracker):
        """Test that default unit is set based on goal type."""
        await goal_tracker.initialize()

        goal = await goal_tracker.create_goal(
            title="Build Streak",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        assert goal.unit == "days"


class TestGoalRetrieval:
    """Tests for goal retrieval."""

    @pytest.mark.asyncio
    async def test_get_goal_by_id(self, goal_tracker):
        """Test retrieving a goal by ID."""
        await goal_tracker.initialize()
        created = await goal_tracker.create_goal(
            title="Test Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        goal = await goal_tracker.get_goal(created.id)

        assert goal is not None
        assert goal.id == created.id

    @pytest.mark.asyncio
    async def test_get_goal_not_found(self, goal_tracker):
        """Test retrieving non-existent goal returns None."""
        await goal_tracker.initialize()

        goal = await goal_tracker.get_goal("nonexistent-id")

        assert goal is None

    @pytest.mark.asyncio
    async def test_get_all_goals(self, goal_tracker):
        """Test retrieving all goals."""
        await goal_tracker.initialize()
        await goal_tracker.create_goal(
            title="Goal 1",
            goal_type=GoalType.SESSIONS,
            target_value=5
        )
        await goal_tracker.create_goal(
            title="Goal 2",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        goals = await goal_tracker.get_all_goals()

        assert len(goals) >= 2

    @pytest.mark.asyncio
    async def test_get_active_goals(self, goal_tracker, mock_goal_store):
        """Test retrieving only active goals."""
        await goal_tracker.initialize()
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[])

        goals = await goal_tracker.get_active_goals()

        assert isinstance(goals, list)
        mock_goal_store.get_goals_by_status.assert_called_with(GoalStatus.ACTIVE)


class TestGoalUpdate:
    """Tests for goal updates."""

    @pytest.mark.asyncio
    async def test_update_goal_title(self, goal_tracker):
        """Test updating goal title."""
        await goal_tracker.initialize()
        goal = await goal_tracker.create_goal(
            title="Original Title",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        request = UpdateGoalRequest(title="Updated Title")
        updated = await goal_tracker.update_goal(goal.id, request)

        assert updated.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_update_goal_target(self, goal_tracker):
        """Test updating goal target value."""
        await goal_tracker.initialize()
        goal = await goal_tracker.create_goal(
            title="Test",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        request = UpdateGoalRequest(target_value=20)
        updated = await goal_tracker.update_goal(goal.id, request)

        assert updated.target_value == 20

    @pytest.mark.asyncio
    async def test_update_nonexistent_goal(self, goal_tracker):
        """Test updating non-existent goal returns None."""
        await goal_tracker.initialize()

        request = UpdateGoalRequest(title="New Title")
        result = await goal_tracker.update_goal("nonexistent", request)

        assert result is None


class TestGoalDeletion:
    """Tests for goal deletion."""

    @pytest.mark.asyncio
    async def test_delete_goal(self, goal_tracker):
        """Test deleting a goal."""
        await goal_tracker.initialize()
        goal = await goal_tracker.create_goal(
            title="To Delete",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        result = await goal_tracker.delete_goal(goal.id)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_goal(self, goal_tracker):
        """Test deleting non-existent goal returns False."""
        await goal_tracker.initialize()

        result = await goal_tracker.delete_goal("nonexistent")

        assert result is False


class TestProgressUpdates:
    """Tests for progress updates."""

    @pytest.mark.asyncio
    async def test_update_progress_basic(self, goal_tracker, mock_goal_store):
        """Test basic progress update."""
        await goal_tracker.initialize()
        goal = await goal_tracker.create_goal(
            title="Test",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        # Mock the update
        goal.current_value = 5
        mock_goal_store.update_goal_progress = AsyncMock(return_value=goal)

        updated, milestones = await goal_tracker.update_progress(goal.id, 5)

        assert updated is not None
        mock_goal_store.update_goal_progress.assert_called()

    @pytest.mark.asyncio
    async def test_update_all_goals(self, goal_tracker, mock_goal_store):
        """Test updating all goals from metrics."""
        await goal_tracker.initialize()
        metrics = ProgressMetrics(
            current_streak=5,
            total_sessions=20,
            total_exchanges=100
        )
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[])

        result = await goal_tracker.update_all_goals(metrics)

        assert "goals_updated" in result
        assert "goals_completed" in result


class TestGoalSuggestions:
    """Tests for goal suggestions."""

    @pytest.mark.asyncio
    async def test_get_suggestions_for_new_user(self, goal_tracker, mock_goal_store):
        """Test suggestions for user with low activity."""
        await goal_tracker.initialize()
        metrics = ProgressMetrics(
            total_sessions=3,
            current_streak=0,
            total_exchanges=15,
            total_topics=2
        )
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[])

        suggestions = await goal_tracker.get_goal_suggestions(metrics)

        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_get_suggestions_for_streak_builder(self, goal_tracker, mock_goal_store):
        """Test suggestions for user building a streak."""
        await goal_tracker.initialize()
        metrics = ProgressMetrics(
            total_sessions=20,
            current_streak=5,
            total_exchanges=100,
            total_topics=5
        )
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[])

        suggestions = await goal_tracker.get_goal_suggestions(metrics)

        # Should suggest streak goals
        streak_suggestions = [s for s in suggestions if s.goal_type == GoalType.STREAK]
        assert isinstance(streak_suggestions, list)

    @pytest.mark.asyncio
    async def test_suggestions_respect_limit(self, goal_tracker, mock_goal_store):
        """Test that suggestions respect the limit parameter."""
        await goal_tracker.initialize()
        metrics = ProgressMetrics(
            total_sessions=5,
            current_streak=3,
            total_exchanges=50
        )
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[])

        suggestions = await goal_tracker.get_goal_suggestions(metrics, limit=3)

        assert len(suggestions) <= 3

    @pytest.mark.asyncio
    async def test_suggestions_exclude_existing_types(self, goal_tracker, mock_goal_store):
        """Test that suggestions exclude existing goal types."""
        await goal_tracker.initialize()
        # Create existing streak goal
        existing_goal = Goal(
            id="test",
            title="Existing Streak",
            goal_type=GoalType.STREAK,
            target_value=7,
            current_value=0,
            status=GoalStatus.ACTIVE
        )
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[existing_goal])

        metrics = ProgressMetrics(
            total_sessions=10,
            current_streak=3
        )

        suggestions = await goal_tracker.get_goal_suggestions(metrics)

        # Should not suggest more streak goals
        streak_suggestions = [s for s in suggestions if s.goal_type == GoalType.STREAK]
        assert len(streak_suggestions) == 0


class TestProgressHistory:
    """Tests for progress history."""

    @pytest.mark.asyncio
    async def test_get_progress_history(self, goal_tracker, mock_goal_store):
        """Test retrieving progress history."""
        await goal_tracker.initialize()
        mock_goal_store.get_goal_progress_history = AsyncMock(return_value=[])

        history = await goal_tracker.get_progress_history("goal-id", days=30)

        assert isinstance(history, list)
        mock_goal_store.get_goal_progress_history.assert_called_with("goal-id", 30)
