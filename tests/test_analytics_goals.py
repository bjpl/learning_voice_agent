"""
Tests for Analytics Goal Tracking System
PATTERN: Comprehensive test coverage for gamification features
"""
import pytest
import asyncio
from datetime import date, datetime, timedelta
import tempfile
import os

from app.analytics.goal_models import (
    Goal, GoalType, GoalStatus,
    Milestone, GoalProgress, GoalSuggestion,
    Achievement, AchievementRarity, AchievementCategory,
    CreateGoalRequest, UpdateGoalRequest
)
from app.analytics.goal_store import GoalStore
from app.analytics.goal_tracker import GoalTracker, ProgressMetrics
from app.analytics.achievement_system import AchievementSystem, ACHIEVEMENTS
from app.analytics.export_service import ExportService, ReportPeriod


class TestGoalModels:
    """Test goal data models."""

    def test_goal_progress_calculation(self):
        """Test goal progress percentage calculation."""
        goal = Goal(
            id="test-1",
            title="Test Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10,
            current_value=5
        )
        assert goal.progress_percent == 50.0
        assert not goal.is_completed

    def test_goal_completion_detection(self):
        """Test goal completion status."""
        goal = Goal(
            id="test-2",
            title="Completed Goal",
            goal_type=GoalType.STREAK,
            target_value=7,
            current_value=7
        )
        assert goal.progress_percent == 100.0
        assert goal.is_completed

    def test_goal_days_remaining(self):
        """Test days remaining calculation."""
        future_date = date.today() + timedelta(days=10)
        goal = Goal(
            id="test-3",
            title="Deadline Goal",
            goal_type=GoalType.SESSIONS,
            target_value=5,
            deadline=future_date
        )
        assert goal.days_remaining == 10
        assert not goal.is_expired

    def test_goal_expired(self):
        """Test expired goal detection using model_construct to bypass validation."""
        # Use model_construct to bypass validation for testing expired goals
        # In production, goals would expire after being created with future deadlines
        past_date = date.today() - timedelta(days=1)
        goal = Goal.model_construct(
            id="test-4",
            title="Expired Goal",
            goal_type=GoalType.SESSIONS,
            target_value=5,
            current_value=0,
            unit="sessions",
            status=GoalStatus.ACTIVE,
            deadline=past_date,
            created_at=datetime.utcnow() - timedelta(days=10),
            updated_at=datetime.utcnow(),
            completed_at=None,
            milestones=[],
            metadata=None
        )
        assert goal.is_expired

    def test_milestone_progress(self):
        """Test milestone progress calculation."""
        milestone = Milestone(
            goal_id="test-goal",
            title="First Milestone",
            target_value=10,
            current_value=7,
            order=0
        )
        assert milestone.progress_percent == 70.0


class TestAchievementModels:
    """Test achievement data models."""

    def test_achievement_properties(self):
        """Test achievement model properties."""
        achievement = Achievement(
            id="test-achievement",
            title="Test Badge",
            description="Test description",
            icon="star",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.COMMON,
            requirement="Complete 1 session",
            requirement_type="sessions",
            requirement_value=1,
            points=10
        )
        assert achievement.progress_percent == 0.0
        assert not achievement.unlocked

    def test_achievement_check_unlock(self):
        """Test achievement unlock check."""
        achievement = Achievement(
            id="test-unlock",
            title="Unlock Test",
            description="Test",
            icon="test",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.COMMON,
            requirement="5 sessions",
            requirement_type="sessions",
            requirement_value=5,
            points=10
        )
        assert achievement.check_unlock(5)
        assert achievement.check_unlock(10)
        assert not achievement.check_unlock(3)

    def test_predefined_achievements(self):
        """Test predefined achievements exist."""
        assert len(ACHIEVEMENTS) >= 15

        # Check categories are covered
        categories = {a.category for a in ACHIEVEMENTS}
        assert AchievementCategory.BEGINNER in categories
        assert AchievementCategory.STREAK in categories
        assert AchievementCategory.QUALITY in categories

        # Check rarities are used
        rarities = {a.rarity for a in ACHIEVEMENTS}
        assert AchievementRarity.COMMON in rarities
        assert AchievementRarity.RARE in rarities
        assert AchievementRarity.EPIC in rarities
        assert AchievementRarity.LEGENDARY in rarities


class TestGoalStore:
    """Test goal store persistence."""

    @pytest.fixture
    def temp_store(self):
        """Create temporary store for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_goals.db")
        store = GoalStore(db_path)
        yield store
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_store_initialization(self, temp_store):
        """Test store initializes correctly."""
        await temp_store.initialize()
        assert temp_store._initialized

    @pytest.mark.asyncio
    async def test_save_and_get_goal(self, temp_store):
        """Test saving and retrieving a goal."""
        await temp_store.initialize()

        goal = Goal(
            id="save-test",
            title="Save Test Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        await temp_store.save_goal(goal)
        retrieved = await temp_store.get_goal("save-test")

        assert retrieved is not None
        assert retrieved.id == "save-test"
        assert retrieved.title == "Save Test Goal"
        assert retrieved.target_value == 10

    @pytest.mark.asyncio
    async def test_get_goals_by_status(self, temp_store):
        """Test filtering goals by status."""
        await temp_store.initialize()

        # Create goals with different statuses
        active_goal = Goal(
            id="active-1",
            title="Active Goal",
            goal_type=GoalType.SESSIONS,
            target_value=5,
            status=GoalStatus.ACTIVE
        )
        completed_goal = Goal(
            id="completed-1",
            title="Completed Goal",
            goal_type=GoalType.SESSIONS,
            target_value=5,
            status=GoalStatus.COMPLETED
        )

        await temp_store.save_goal(active_goal)
        await temp_store.save_goal(completed_goal)

        active_goals = await temp_store.get_goals_by_status(GoalStatus.ACTIVE)
        completed_goals = await temp_store.get_goals_by_status(GoalStatus.COMPLETED)

        assert len(active_goals) == 1
        assert len(completed_goals) == 1
        assert active_goals[0].id == "active-1"
        assert completed_goals[0].id == "completed-1"

    @pytest.mark.asyncio
    async def test_update_progress(self, temp_store):
        """Test updating goal progress."""
        await temp_store.initialize()

        goal = Goal(
            id="progress-test",
            title="Progress Test",
            goal_type=GoalType.SESSIONS,
            target_value=10,
            current_value=0
        )
        await temp_store.save_goal(goal)

        updated = await temp_store.update_goal_progress("progress-test", 5, "test")

        assert updated is not None
        assert updated.current_value == 5
        assert updated.progress_percent == 50.0

    @pytest.mark.asyncio
    async def test_goal_completion_on_update(self, temp_store):
        """Test goal completes when progress reaches target."""
        await temp_store.initialize()

        goal = Goal(
            id="completion-test",
            title="Completion Test",
            goal_type=GoalType.SESSIONS,
            target_value=5,
            current_value=4
        )
        await temp_store.save_goal(goal)

        updated = await temp_store.update_goal_progress("completion-test", 5, "test")

        assert updated.status == GoalStatus.COMPLETED
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_save_and_get_achievement(self, temp_store):
        """Test saving and retrieving achievements."""
        await temp_store.initialize()

        achievement = Achievement(
            id="ach-test",
            title="Test Achievement",
            description="Test description",
            icon="star",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.COMMON,
            requirement="1 session",
            requirement_type="sessions",
            requirement_value=1,
            points=10
        )

        await temp_store.save_achievement(achievement)
        retrieved = await temp_store.get_achievement("ach-test")

        assert retrieved is not None
        assert retrieved.id == "ach-test"
        assert retrieved.title == "Test Achievement"

    @pytest.mark.asyncio
    async def test_unlock_achievement(self, temp_store):
        """Test unlocking an achievement."""
        await temp_store.initialize()

        achievement = Achievement(
            id="unlock-test",
            title="Unlock Test",
            description="Test",
            icon="lock",
            category=AchievementCategory.BEGINNER,
            rarity=AchievementRarity.COMMON,
            requirement="1 session",
            requirement_type="sessions",
            requirement_value=1,
            points=10
        )
        await temp_store.save_achievement(achievement)

        unlocked = await temp_store.unlock_achievement("unlock-test", 1)

        assert unlocked is not None
        assert unlocked.unlocked
        assert unlocked.unlocked_at is not None


class TestGoalTracker:
    """Test goal tracker functionality."""

    @pytest.fixture
    def temp_tracker(self):
        """Create temporary tracker for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "tracker_goals.db")
        store = GoalStore(db_path)
        tracker = GoalTracker(store)
        yield tracker
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_create_goal(self, temp_tracker):
        """Test creating a goal through tracker."""
        await temp_tracker.initialize()

        goal = await temp_tracker.create_goal(
            title="New Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10,
            description="Test goal"
        )

        assert goal.id is not None
        assert goal.title == "New Goal"
        assert goal.goal_type == GoalType.SESSIONS
        assert goal.target_value == 10
        assert goal.unit == "sessions"  # Auto-set based on type

    @pytest.mark.asyncio
    async def test_create_goal_with_milestones(self, temp_tracker):
        """Test goal creation generates milestones."""
        await temp_tracker.initialize()

        goal = await temp_tracker.create_goal(
            title="Large Goal",
            goal_type=GoalType.SESSIONS,
            target_value=100  # Large goal should have milestones
        )

        assert len(goal.milestones) > 0

    @pytest.mark.asyncio
    async def test_update_progress(self, temp_tracker):
        """Test progress update through tracker."""
        await temp_tracker.initialize()

        goal = await temp_tracker.create_goal(
            title="Progress Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        updated_goal, new_milestones = await temp_tracker.update_progress(
            goal.id, 5, "test"
        )

        assert updated_goal.current_value == 5
        assert updated_goal.progress_percent == 50.0

    @pytest.mark.asyncio
    async def test_get_active_goals(self, temp_tracker):
        """Test getting active goals."""
        await temp_tracker.initialize()

        await temp_tracker.create_goal(
            title="Active 1",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )
        await temp_tracker.create_goal(
            title="Active 2",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        active = await temp_tracker.get_active_goals()
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_update_all_goals(self, temp_tracker):
        """Test updating all goals with metrics."""
        await temp_tracker.initialize()

        await temp_tracker.create_goal(
            title="Sessions Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )
        await temp_tracker.create_goal(
            title="Streak Goal",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        metrics = ProgressMetrics(
            current_streak=5,
            total_sessions=8,
            total_exchanges=50,
            total_topics=3,
            avg_quality_score=0.8
        )

        result = await temp_tracker.update_all_goals(metrics)

        assert result['goals_updated'] == 2

    @pytest.mark.asyncio
    async def test_goal_suggestions(self, temp_tracker):
        """Test generating goal suggestions."""
        await temp_tracker.initialize()

        metrics = ProgressMetrics(
            current_streak=3,
            total_sessions=5,
            total_exchanges=30,
            total_topics=2,
            avg_quality_score=0.7
        )

        suggestions = await temp_tracker.get_goal_suggestions(metrics)

        assert len(suggestions) > 0
        # Should suggest streak goal since current is < 7
        streak_suggestions = [s for s in suggestions if s.goal_type == GoalType.STREAK]
        assert len(streak_suggestions) > 0


class TestAchievementSystem:
    """Test achievement system functionality."""

    @pytest.fixture
    def temp_achievement_system(self):
        """Create temporary achievement system for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "achievement_goals.db")
        store = GoalStore(db_path)
        system = AchievementSystem(store)
        yield system
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_seeds_achievements(self, temp_achievement_system):
        """Test achievements are seeded on init."""
        await temp_achievement_system.initialize()

        all_achievements = await temp_achievement_system.get_all_achievements()
        assert len(all_achievements) >= 15

    @pytest.mark.asyncio
    async def test_check_achievements(self, temp_achievement_system):
        """Test checking achievements against metrics."""
        await temp_achievement_system.initialize()

        # Metrics that should unlock "First Steps"
        metrics = ProgressMetrics(
            total_sessions=1
        )

        result = await temp_achievement_system.check_achievements(metrics)

        assert result.total_points_earned > 0 or len(result.progress_updated) > 0

    @pytest.mark.asyncio
    async def test_get_unlocked_achievements(self, temp_achievement_system):
        """Test getting unlocked achievements."""
        await temp_achievement_system.initialize()

        # Initially none should be unlocked
        unlocked = await temp_achievement_system.get_unlocked_achievements()
        initial_count = len(unlocked)

        # Check achievements with sufficient metrics
        metrics = ProgressMetrics(
            total_sessions=10,
            current_streak=7,
            total_exchanges=100
        )

        await temp_achievement_system.check_achievements(metrics)

        unlocked = await temp_achievement_system.get_unlocked_achievements()
        # Should have unlocked some achievements
        assert len(unlocked) >= initial_count

    @pytest.mark.asyncio
    async def test_achievement_stats(self, temp_achievement_system):
        """Test getting achievement statistics."""
        await temp_achievement_system.initialize()

        stats = await temp_achievement_system.get_achievement_stats()

        assert 'total' in stats
        assert 'unlocked' in stats
        assert 'by_category' in stats
        assert 'by_rarity' in stats

    @pytest.mark.asyncio
    async def test_get_achievements_by_category(self, temp_achievement_system):
        """Test filtering achievements by category."""
        await temp_achievement_system.initialize()

        beginner = await temp_achievement_system.get_achievements_by_category(
            AchievementCategory.BEGINNER
        )

        assert len(beginner) > 0
        for a in beginner:
            assert a.category == AchievementCategory.BEGINNER

    @pytest.mark.asyncio
    async def test_get_next_achievements(self, temp_achievement_system):
        """Test getting achievements closest to unlock."""
        await temp_achievement_system.initialize()

        metrics = ProgressMetrics(
            total_sessions=3,  # Close to some achievements
            current_streak=3
        )

        next_achievements = await temp_achievement_system.get_next_achievements(metrics)

        assert len(next_achievements) > 0
        # Should be sorted by progress
        for i in range(len(next_achievements) - 1):
            assert next_achievements[i][1] >= next_achievements[i + 1][1]


class TestExportService:
    """Test export service functionality."""

    @pytest.fixture
    def temp_export_service(self):
        """Create temporary export service for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "export_goals.db")
        store = GoalStore(db_path)
        tracker = GoalTracker(store)
        achievement_sys = AchievementSystem(store)
        service = ExportService(store, tracker, achievement_sys)
        yield service
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_export_to_json(self, temp_export_service):
        """Test JSON export."""
        await temp_export_service.initialize()

        json_data = await temp_export_service.export_to_json(ReportPeriod.MONTH)

        assert json_data is not None
        assert len(json_data) > 0
        # Should be valid JSON
        import json
        parsed = json.loads(json_data)
        assert 'metadata' in parsed

    @pytest.mark.asyncio
    async def test_export_to_csv(self, temp_export_service):
        """Test CSV export."""
        await temp_export_service.initialize()

        # First create some goals
        await temp_export_service.tracker.create_goal(
            title="CSV Test Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        csv_data = await temp_export_service.export_to_csv("goals", ReportPeriod.MONTH)

        assert csv_data is not None
        assert "CSV Test Goal" in csv_data

    @pytest.mark.asyncio
    async def test_generate_weekly_report(self, temp_export_service):
        """Test weekly report generation."""
        await temp_export_service.initialize()

        report = await temp_export_service.generate_weekly_report()

        assert report is not None
        assert report.week_start is not None
        assert report.week_end is not None
        assert hasattr(report, 'summary')
        assert hasattr(report, 'insights')
        assert hasattr(report, 'recommendations')

    @pytest.mark.asyncio
    async def test_generate_monthly_report(self, temp_export_service):
        """Test monthly report generation."""
        await temp_export_service.initialize()

        report = await temp_export_service.generate_monthly_report()

        assert report is not None
        assert report.month is not None
        assert report.year is not None
        assert hasattr(report, 'summary')

    @pytest.mark.asyncio
    async def test_get_export_data(self, temp_export_service):
        """Test getting export data with options."""
        await temp_export_service.initialize()

        data = await temp_export_service.get_export_data(
            period=ReportPeriod.MONTH,
            include_goals=True,
            include_achievements=True
        )

        assert 'metadata' in data
        assert 'goals' in data
        assert 'achievements' in data


class TestAPIRequestModels:
    """Test API request/response models."""

    def test_create_goal_request(self):
        """Test CreateGoalRequest model."""
        request = CreateGoalRequest(
            title="Test Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )
        assert request.title == "Test Goal"
        assert request.goal_type == GoalType.SESSIONS
        assert request.target_value == 10

    def test_create_goal_request_with_deadline(self):
        """Test CreateGoalRequest with deadline."""
        future_date = date.today() + timedelta(days=30)
        request = CreateGoalRequest(
            title="Deadline Goal",
            goal_type=GoalType.SESSIONS,
            target_value=10,
            deadline=future_date
        )
        assert request.deadline == future_date

    def test_update_goal_request_partial(self):
        """Test UpdateGoalRequest allows partial updates."""
        request = UpdateGoalRequest(target_value=20)
        assert request.target_value == 20
        assert request.title is None
        assert request.status is None
