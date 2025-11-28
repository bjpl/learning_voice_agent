"""
Test Configuration and Fixtures for Phase 6 Analytics Engine
=============================================================

Provides shared fixtures for all analytics component tests.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional
import uuid
import random

# Import analytics components
from app.analytics.analytics_config import AnalyticsEngineConfig, analytics_config
from app.analytics.progress_models import (
    ProgressMetrics,
    LearningStreak,
    TopicMastery,
    SessionProgress,
    DailyProgress,
    WeeklyProgress,
    MonthlyProgress,
    ProgressSnapshot,
    TrendDirection,
    ProgressLevel,
    GoalStatus,
    GoalType,
    LearningGoal,
    Achievement,
    UnlockedAchievement,
    AchievementTier,
    DashboardData,
)


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def analytics_config_fixture():
    """Default analytics configuration for testing."""
    config = AnalyticsEngineConfig()
    config.cache_enabled = False  # Disable caching for tests
    return config


@pytest.fixture
def test_config_with_temp_db():
    """Analytics config with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AnalyticsEngineConfig()
        config.db_path = os.path.join(tmpdir, "test_analytics.db")
        config.cache_enabled = False
        yield config


# ============================================================================
# Mock Store Fixtures
# ============================================================================

@pytest.fixture
def mock_progress_store():
    """Mock progress store for testing."""
    store = AsyncMock()
    store._sessions: Dict[str, SessionProgress] = {}
    store._streaks: Dict[str, LearningStreak] = {}
    store._mastery: Dict[str, Dict[str, TopicMastery]] = {}
    store._snapshots: List[ProgressSnapshot] = []

    async def mock_store_session_progress(session: SessionProgress):
        store._sessions[session.session_id] = session
        return session

    async def mock_get_sessions(user_id: Optional[str]):
        if user_id:
            return [s for s in store._sessions.values() if s.user_id == user_id]
        return list(store._sessions.values())

    async def mock_get_sessions_for_date(target_date: date, user_id: Optional[str]):
        sessions = await mock_get_sessions(user_id)
        return [s for s in sessions if s.start_time.date() == target_date]

    async def mock_get_streak(user_id: Optional[str]):
        key = user_id or "global"
        return store._streaks.get(key)

    async def mock_store_streak(streak: LearningStreak):
        key = streak.user_id or "global"
        store._streaks[key] = streak
        return streak

    async def mock_get_topic_mastery(user_id: Optional[str], topic: str):
        key = user_id or "global"
        if key in store._mastery:
            return store._mastery[key].get(topic)
        return None

    async def mock_store_topic_mastery(mastery: TopicMastery):
        key = mastery.user_id or "global"
        if key not in store._mastery:
            store._mastery[key] = {}
        store._mastery[key][mastery.topic] = mastery
        return mastery

    async def mock_get_all_topic_mastery(user_id: Optional[str]):
        key = user_id or "global"
        if key in store._mastery:
            return list(store._mastery[key].values())
        return []

    async def mock_store_snapshot(snapshot: ProgressSnapshot):
        store._snapshots.append(snapshot)
        return snapshot

    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.store_session_progress = AsyncMock(side_effect=mock_store_session_progress)
    store.get_sessions = AsyncMock(side_effect=mock_get_sessions)
    store.get_sessions_for_date = AsyncMock(side_effect=mock_get_sessions_for_date)
    store.get_streak = AsyncMock(side_effect=mock_get_streak)
    store.store_streak = AsyncMock(side_effect=mock_store_streak)
    store.get_topic_mastery = AsyncMock(side_effect=mock_get_topic_mastery)
    store.store_topic_mastery = AsyncMock(side_effect=mock_store_topic_mastery)
    store.get_all_topic_mastery = AsyncMock(side_effect=mock_get_all_topic_mastery)
    store.store_snapshot = AsyncMock(side_effect=mock_store_snapshot)

    return store


@pytest.fixture
def mock_feedback_store():
    """Mock feedback store for testing."""
    store = AsyncMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    return store


@pytest.fixture
def mock_quality_store():
    """Mock quality store for testing."""
    store = AsyncMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    return store


@pytest.fixture
def mock_goal_store():
    """Mock goal store for testing."""
    store = AsyncMock()
    store._goals: Dict[str, Any] = {}
    store._achievements: Dict[str, Any] = {}

    async def mock_save_goal(goal):
        store._goals[goal.id] = goal
        return goal

    async def mock_get_goal(goal_id: str):
        return store._goals.get(goal_id)

    async def mock_get_all_goals():
        return list(store._goals.values())

    async def mock_delete_goal(goal_id: str):
        if goal_id in store._goals:
            del store._goals[goal_id]
            return True
        return False

    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.save_goal = AsyncMock(side_effect=mock_save_goal)
    store.get_goal = AsyncMock(side_effect=mock_get_goal)
    store.get_all_goals = AsyncMock(side_effect=mock_get_all_goals)
    store.delete_goal = AsyncMock(side_effect=mock_delete_goal)
    store.get_goals_by_status = AsyncMock(return_value=[])
    store.update_goal_progress = AsyncMock()
    store.get_goal_progress_history = AsyncMock(return_value=[])

    return store


# ============================================================================
# Sample Data Generators
# ============================================================================

@pytest.fixture
def sample_session_progress():
    """Generate a sample session progress object."""
    def _generate(
        session_id: str = None,
        user_id: str = None,
        start_time: datetime = None,
        duration_minutes: float = 15.0,
        total_exchanges: int = 10,
        avg_quality_score: float = 0.75,
        topics: List[str] = None,
        **kwargs
    ) -> SessionProgress:
        return SessionProgress(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            start_time=start_time or datetime.utcnow() - timedelta(minutes=duration_minutes),
            end_time=datetime.utcnow(),
            duration_minutes=duration_minutes,
            total_exchanges=total_exchanges,
            questions_asked=total_exchanges // 2,
            clarifications_needed=total_exchanges // 10,
            avg_quality_score=avg_quality_score,
            min_quality_score=max(0, avg_quality_score - 0.2),
            max_quality_score=min(1, avg_quality_score + 0.2),
            topics=topics or ["machine learning"],
            primary_topic=topics[0] if topics else "machine learning",
            positive_feedback_count=max(0, int(total_exchanges * avg_quality_score * 0.3)),
            negative_feedback_count=max(0, int(total_exchanges * (1 - avg_quality_score) * 0.2)),
            learning_velocity=total_exchanges / duration_minutes if duration_minutes > 0 else 0
        )
    return _generate


@pytest.fixture
def sample_session_list(sample_session_progress):
    """Generate a list of sample session progress objects."""
    def _generate(
        n: int = 10,
        user_id: str = None,
        days_span: int = 30,
        avg_quality: float = 0.75,
        variance: float = 0.1
    ) -> List[SessionProgress]:
        sessions = []
        for i in range(n):
            days_ago = random.randint(0, days_span)
            quality = max(0.1, min(1.0, avg_quality + random.uniform(-variance, variance)))
            sessions.append(sample_session_progress(
                user_id=user_id,
                start_time=datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(0, 23)),
                duration_minutes=random.uniform(5, 30),
                total_exchanges=random.randint(5, 30),
                avg_quality_score=quality,
                topics=[f"topic_{random.randint(1, 10)}"]
            ))
        return sessions
    return _generate


@pytest.fixture
def sample_learning_streak():
    """Generate a sample learning streak object."""
    def _generate(
        user_id: str = None,
        current_streak: int = 5,
        longest_streak: int = 10,
        last_active_date: date = None,
        **kwargs
    ) -> LearningStreak:
        return LearningStreak(
            user_id=user_id,
            current_streak=current_streak,
            longest_streak=max(current_streak, longest_streak),
            last_active_date=last_active_date or date.today(),
            streak_start_date=(date.today() - timedelta(days=current_streak - 1)) if current_streak > 0 else None,
            streak_history=kwargs.get("streak_history", [])
        )
    return _generate


@pytest.fixture
def sample_topic_mastery():
    """Generate a sample topic mastery object."""
    def _generate(
        topic: str = "machine learning",
        user_id: str = None,
        mastery_score: float = 0.6,
        total_interactions: int = 10,
        level: ProgressLevel = ProgressLevel.INTERMEDIATE,
        **kwargs
    ) -> TopicMastery:
        return TopicMastery(
            topic=topic,
            user_id=user_id,
            mastery_score=mastery_score,
            confidence=min(1.0, total_interactions / 20),
            level=level,
            total_interactions=total_interactions,
            successful_interactions=int(total_interactions * mastery_score),
            avg_quality_score=mastery_score,
            quality_trend=TrendDirection.STABLE,
            total_time_minutes=total_interactions * 3.0,
            first_interaction=datetime.utcnow() - timedelta(days=7),
            last_interaction=datetime.utcnow()
        )
    return _generate


@pytest.fixture
def sample_daily_progress():
    """Generate a sample daily progress object."""
    def _generate(
        target_date: date = None,
        user_id: str = None,
        total_sessions: int = 3,
        total_exchanges: int = 30,
        avg_quality_score: float = 0.75,
        **kwargs
    ) -> DailyProgress:
        return DailyProgress(
            date=target_date or date.today(),
            user_id=user_id,
            total_sessions=total_sessions,
            completed_sessions=total_sessions,
            total_exchanges=total_exchanges,
            total_time_minutes=total_exchanges * 1.5,
            avg_session_duration=(total_exchanges * 1.5) / total_sessions if total_sessions > 0 else 0,
            avg_quality_score=avg_quality_score,
            quality_trend=TrendDirection.STABLE,
            topics_covered=kwargs.get("topics_covered", ["topic_1", "topic_2"]),
            new_topics=kwargs.get("new_topics", []),
            goals_completed=kwargs.get("goals_completed", 0),
            goals_in_progress=kwargs.get("goals_in_progress", 1),
            streak_maintained=True,
            current_streak=kwargs.get("current_streak", 5),
            achievements_unlocked=kwargs.get("achievements_unlocked", [])
        )
    return _generate


@pytest.fixture
def sample_daily_progress_list(sample_daily_progress):
    """Generate a list of daily progress objects."""
    def _generate(
        days: int = 30,
        user_id: str = None,
        base_quality: float = 0.7,
        include_empty_days: bool = True
    ) -> List[DailyProgress]:
        progress_list = []
        for i in range(days):
            target_date = date.today() - timedelta(days=days - 1 - i)
            # Simulate some empty days
            if include_empty_days and random.random() < 0.2:
                progress_list.append(sample_daily_progress(
                    target_date=target_date,
                    user_id=user_id,
                    total_sessions=0,
                    total_exchanges=0,
                    avg_quality_score=0
                ))
            else:
                quality = base_quality + random.uniform(-0.1, 0.15)
                progress_list.append(sample_daily_progress(
                    target_date=target_date,
                    user_id=user_id,
                    total_sessions=random.randint(1, 5),
                    total_exchanges=random.randint(10, 50),
                    avg_quality_score=max(0.3, min(1.0, quality))
                ))
        return progress_list
    return _generate


@pytest.fixture
def sample_progress_metrics():
    """Generate a sample progress metrics object."""
    def _generate(
        user_id: str = None,
        sessions_count: int = 50,
        total_exchanges: int = 500,
        avg_quality_score: float = 0.75,
        current_streak: int = 7,
        **kwargs
    ) -> ProgressMetrics:
        return ProgressMetrics(
            user_id=user_id,
            sessions_count=sessions_count,
            total_exchanges=total_exchanges,
            total_time_hours=sessions_count * 0.25,
            avg_quality_score=avg_quality_score,
            quality_percentile=kwargs.get("quality_percentile", 0.75),
            learning_velocity=total_exchanges / (sessions_count * 0.25) if sessions_count > 0 else 0,
            velocity_trend=TrendDirection.STABLE,
            current_streak=current_streak,
            longest_streak=max(current_streak, kwargs.get("longest_streak", current_streak)),
            topics_explored=kwargs.get("topics_explored", 15),
            topics_mastered=kwargs.get("topics_mastered", 5),
            goals_completed=kwargs.get("goals_completed", 3),
            goals_in_progress=kwargs.get("goals_in_progress", 2),
            goal_completion_rate=kwargs.get("goal_completion_rate", 0.6),
            achievements_count=kwargs.get("achievements_count", 8),
            achievement_points=kwargs.get("achievement_points", 250),
            first_session=datetime.utcnow() - timedelta(days=30),
            last_session=datetime.utcnow() - timedelta(hours=2),
            most_active_hour=14,
            most_active_day="Tuesday"
        )
    return _generate


@pytest.fixture
def sample_learning_goal():
    """Generate a sample learning goal."""
    def _generate(
        title: str = "Complete 10 Sessions",
        goal_type: GoalType = GoalType.WEEKLY,
        target_value: float = 10,
        current_value: float = 3,
        status: GoalStatus = GoalStatus.IN_PROGRESS,
        **kwargs
    ) -> LearningGoal:
        return LearningGoal(
            user_id=kwargs.get("user_id"),
            title=title,
            description=kwargs.get("description", "A learning goal"),
            goal_type=goal_type,
            target_metric="sessions",
            target_value=target_value,
            current_value=current_value,
            status=status,
            progress_percentage=(current_value / target_value * 100) if target_value > 0 else 0,
            start_date=date.today() - timedelta(days=3),
            end_date=date.today() + timedelta(days=4),
            points=kwargs.get("points", 50),
            progress_history=[]
        )
    return _generate


@pytest.fixture
def sample_achievement():
    """Generate a sample achievement."""
    def _generate(
        name: str = "Week Warrior",
        tier: AchievementTier = AchievementTier.SILVER,
        requirement_type: str = "streak",
        requirement_value: float = 7,
        **kwargs
    ) -> Achievement:
        return Achievement(
            name=name,
            description=kwargs.get("description", "An achievement"),
            icon=kwargs.get("icon", "star"),
            tier=tier,
            requirement_type=requirement_type,
            requirement_value=requirement_value,
            requirement_description=f"Reach {requirement_value} {requirement_type}",
            points=kwargs.get("points", 50),
            rarity=kwargs.get("rarity", 0.5),
            category=kwargs.get("category", "streak"),
            tags=kwargs.get("tags", ["streak"])
        )
    return _generate


# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def progress_tracker(analytics_config_fixture, mock_progress_store, mock_feedback_store, mock_quality_store):
    """ProgressTracker with mocked dependencies."""
    from app.analytics.progress_tracker import ProgressTracker
    tracker = ProgressTracker(
        config=analytics_config_fixture,
        progress_store=mock_progress_store,
        feedback_store=mock_feedback_store,
        quality_store=mock_quality_store
    )
    return tracker


@pytest.fixture
def insights_engine(analytics_config_fixture):
    """InsightsEngine with default config."""
    from app.analytics.insights_engine import InsightsEngine
    return InsightsEngine(config=analytics_config_fixture)


@pytest.fixture
def trend_analyzer(analytics_config_fixture):
    """TrendAnalyzer with default config."""
    from app.analytics.trend_analyzer import TrendAnalyzer
    return TrendAnalyzer(config=analytics_config_fixture)


@pytest.fixture
def dashboard_service(analytics_config_fixture):
    """DashboardService with default config."""
    from app.analytics.dashboard_service import DashboardService
    return DashboardService(
        config=analytics_config_fixture.dashboard if hasattr(analytics_config_fixture, 'dashboard') else None,
        cache_ttl=300
    )


@pytest.fixture
def goal_tracker(mock_goal_store):
    """GoalTracker with mocked dependencies."""
    from app.analytics.goal_tracker import GoalTracker
    return GoalTracker(store=mock_goal_store)


@pytest.fixture
def achievement_system(mock_goal_store):
    """AchievementSystem with mocked dependencies."""
    from app.analytics.achievement_system import AchievementSystem
    return AchievementSystem(store=mock_goal_store)


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def benchmark():
    """Simple benchmark fixture for performance testing."""
    import time

    class Benchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self):
            return (self.end_time - self.start_time) * 1000

        def assert_under(self, max_ms: float, message: str = None):
            assert self.elapsed_ms < max_ms, \
                message or f"Operation took {self.elapsed_ms:.2f}ms (max: {max_ms}ms)"

    return Benchmark


@pytest.fixture
def freeze_time():
    """Fixture to freeze time for tests."""
    from unittest.mock import patch

    def _freeze(frozen_time: datetime):
        return patch('datetime.datetime')

    return _freeze


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Add any cleanup logic here if needed
