# Phase 6: Analytics Engine - Testing Guide

## Overview

This guide covers the testing strategy for the Phase 6 analytics engine, including test patterns, mock data generation, time-based testing, performance benchmarks, and integration testing approaches.

---

## Testing Strategy

### Test Pyramid

```
                   /\
                  /  \
                 / E2E \        <- 10% Integration tests
                /______\
               /        \
              / Integration\    <- 20% Component integration
             /____________\
            /              \
           /   Unit Tests   \   <- 70% Component unit tests
          /__________________\
```

### Test Categories

1. **Unit Tests** (70%)
   - Individual component tests
   - Model validation tests
   - Calculation accuracy tests
   - Edge case handling

2. **Integration Tests** (20%)
   - Component interaction tests
   - Data flow tests
   - Database integration tests

3. **End-to-End Tests** (10%)
   - Full workflow tests
   - Performance tests
   - Regression tests

---

## Test Structure

### Directory Layout

```
tests/analytics/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_progress_tracker.py       # 30+ tests
├── test_insights_engine.py        # 25+ tests
├── test_trend_analyzer.py         # 20+ tests
├── test_dashboard_service.py      # 30+ tests
├── test_goal_tracker.py           # 25+ tests
├── test_achievement_system.py     # 20+ tests
├── test_export_service.py         # 15+ tests
└── test_phase6_integration.py     # 25+ tests
```

### Naming Conventions

```python
# Test class names: Test{Component}{Feature}
class TestProgressTrackerInitialization:
    pass

class TestProgressTrackerStreaks:
    pass

# Test method names: test_{action}_{scenario}_{expected}
def test_record_session_updates_streak():
    pass

def test_get_streak_returns_cached_when_valid():
    pass

def test_create_goal_with_invalid_type_raises_error():
    pass
```

---

## Mock Data Generation

### Session Progress Generator

```python
@pytest.fixture
def sample_session_progress():
    """Generate sample session progress with customizable parameters."""
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
            avg_quality_score=avg_quality_score,
            topics=topics or ["default_topic"],
            primary_topic=topics[0] if topics else "default_topic",
            learning_velocity=total_exchanges / duration_minutes if duration_minutes > 0 else 0
        )
    return _generate
```

### Batch Data Generator

```python
@pytest.fixture
def sample_session_list(sample_session_progress):
    """Generate multiple sessions with realistic distribution."""
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
                start_time=datetime.utcnow() - timedelta(
                    days=days_ago,
                    hours=random.randint(8, 22)  # Realistic hours
                ),
                duration_minutes=random.uniform(5, 45),
                total_exchanges=random.randint(5, 40),
                avg_quality_score=quality,
                topics=[f"topic_{random.randint(1, 10)}"]
            ))
        return sorted(sessions, key=lambda s: s.start_time)
    return _generate
```

### Daily Progress Generator

```python
@pytest.fixture
def sample_daily_progress_list(sample_daily_progress):
    """Generate daily progress with realistic patterns."""
    def _generate(
        days: int = 30,
        user_id: str = None,
        base_quality: float = 0.7,
        include_empty_days: bool = True,
        weekend_drop: float = 0.3  # 30% less activity on weekends
    ) -> List[DailyProgress]:
        progress_list = []
        for i in range(days):
            target_date = date.today() - timedelta(days=days - 1 - i)
            is_weekend = target_date.weekday() >= 5

            # Simulate realistic patterns
            if include_empty_days and random.random() < (0.3 if is_weekend else 0.1):
                progress_list.append(sample_daily_progress(
                    target_date=target_date,
                    total_sessions=0,
                    total_exchanges=0,
                    avg_quality_score=0
                ))
            else:
                # Weekend sessions are typically lower
                session_count = random.randint(1, 3 if is_weekend else 5)
                exchange_count = random.randint(5, 20 if is_weekend else 50)
                quality = base_quality + random.uniform(-0.1, 0.15)

                progress_list.append(sample_daily_progress(
                    target_date=target_date,
                    total_sessions=session_count,
                    total_exchanges=exchange_count,
                    avg_quality_score=max(0.3, min(1.0, quality))
                ))

        return progress_list
    return _generate
```

### Streak Patterns Generator

```python
@pytest.fixture
def streak_scenarios():
    """Generate various streak scenarios for testing."""
    return {
        "new_user": LearningStreak(
            current_streak=0,
            longest_streak=0,
            last_active_date=None
        ),
        "building_streak": LearningStreak(
            current_streak=5,
            longest_streak=5,
            last_active_date=date.today() - timedelta(days=1)
        ),
        "at_risk": LearningStreak(
            current_streak=10,
            longest_streak=10,
            last_active_date=date.today() - timedelta(days=1)
        ),
        "broken_yesterday": LearningStreak(
            current_streak=1,
            longest_streak=15,
            last_active_date=date.today() - timedelta(days=2),
            streak_history=[{
                "start_date": (date.today() - timedelta(days=17)).isoformat(),
                "end_date": (date.today() - timedelta(days=3)).isoformat(),
                "length": 15
            }]
        ),
        "long_inactive": LearningStreak(
            current_streak=0,
            longest_streak=30,
            last_active_date=date.today() - timedelta(days=60)
        )
    }
```

---

## Time-Based Testing Patterns

### Freezing Time

```python
from unittest.mock import patch
from freezegun import freeze_time

class TestStreakCalculations:
    @freeze_time("2024-11-20 14:00:00")
    async def test_streak_continues_same_day(self, progress_tracker):
        """Test that activity on same day doesn't break streak."""
        streak = LearningStreak(
            current_streak=5,
            last_active_date=date(2024, 11, 20)
        )

        continued = streak.update(date(2024, 11, 20))

        assert continued is True
        assert streak.current_streak == 5

    @freeze_time("2024-11-21 08:00:00")
    async def test_streak_continues_next_day(self, progress_tracker):
        """Test that activity on next day continues streak."""
        streak = LearningStreak(
            current_streak=5,
            last_active_date=date(2024, 11, 20)
        )

        continued = streak.update(date(2024, 11, 21))

        assert continued is True
        assert streak.current_streak == 6

    @freeze_time("2024-11-23 10:00:00")
    async def test_streak_breaks_after_gap(self, progress_tracker):
        """Test that streak breaks after 2+ day gap."""
        streak = LearningStreak(
            current_streak=10,
            longest_streak=10,
            last_active_date=date(2024, 11, 20)
        )

        continued = streak.update(date(2024, 11, 23))

        assert continued is False
        assert streak.current_streak == 1
        assert streak.longest_streak == 10  # Preserved
```

### Testing Across Time Boundaries

```python
class TestMonthBoundary:
    @pytest.mark.asyncio
    async def test_weekly_progress_spans_months(self, progress_tracker, mock_progress_store):
        """Test weekly progress that spans month boundary."""
        # Week starting Oct 28, ending Nov 3
        week_start = date(2024, 10, 28)

        mock_progress_store.get_sessions_for_date = AsyncMock(return_value=[])

        weekly = await progress_tracker.get_weekly_progress(week_start)

        assert weekly.week_start == date(2024, 10, 28)
        assert weekly.week_end == date(2024, 11, 3)

    @pytest.mark.asyncio
    async def test_monthly_progress_handles_short_months(self, progress_tracker):
        """Test monthly progress for February."""
        monthly = await progress_tracker.get_monthly_progress(2024, 2)

        # February 2024 is a leap year
        assert monthly.month == 2
        # Should handle 29 days correctly


class TestDaylightSavingTime:
    @freeze_time("2024-03-10 01:30:00", tz_offset=-8)
    async def test_streak_during_dst_change(self):
        """Test that streak calculation handles DST correctly."""
        # This test ensures streaks don't break during DST transitions
        streak = LearningStreak(
            current_streak=5,
            last_active_date=date(2024, 3, 9)
        )

        continued = streak.update(date(2024, 3, 10))

        assert continued is True
```

### Testing Async Timeouts

```python
import asyncio

class TestAsyncTimeouts:
    @pytest.mark.asyncio
    async def test_progress_calculation_completes_within_timeout(self, progress_tracker):
        """Test that progress calculation completes within reasonable time."""
        await progress_tracker.initialize()

        try:
            await asyncio.wait_for(
                progress_tracker.get_overall_progress(),
                timeout=1.0  # 1 second timeout
            )
        except asyncio.TimeoutError:
            pytest.fail("Progress calculation took too long")
```

---

## Performance Benchmarks

### Benchmark Fixture

```python
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
```

### Performance Test Examples

```python
class TestDashboardPerformance:
    @pytest.mark.asyncio
    async def test_overview_api_under_200ms(self, dashboard_service, benchmark):
        """Target: Overview API < 200ms"""
        await dashboard_service.initialize()

        with benchmark() as b:
            await dashboard_service.get_overview_data()

        b.assert_under(200, "Overview API too slow")

    @pytest.mark.asyncio
    async def test_quality_chart_under_300ms(self, dashboard_service, benchmark):
        """Target: Quality chart < 300ms"""
        await dashboard_service.initialize()

        with benchmark() as b:
            await dashboard_service.get_quality_chart_data(days=30)

        b.assert_under(300, "Quality chart too slow")

    @pytest.mark.asyncio
    async def test_full_dashboard_under_500ms(self, dashboard_service, benchmark):
        """Target: Full dashboard < 500ms"""
        await dashboard_service.initialize()

        with benchmark() as b:
            await dashboard_service.get_dashboard_data(use_cache=False)

        b.assert_under(500, "Full dashboard too slow")


class TestProgressTrackerPerformance:
    @pytest.mark.asyncio
    async def test_progress_calculation_scales_linearly(
        self, progress_tracker, mock_progress_store, sample_session_list
    ):
        """Test that progress calculation scales linearly with data."""
        timings = []

        for n in [10, 50, 100, 500]:
            sessions = sample_session_list(n=n)
            mock_progress_store.get_sessions = AsyncMock(return_value=sessions)
            progress_tracker.clear_cache()
            await progress_tracker.initialize()

            start = time.perf_counter()
            await progress_tracker.get_overall_progress()
            elapsed = (time.perf_counter() - start) * 1000

            timings.append((n, elapsed))

        # Check roughly linear scaling (allow 3x for 50x data)
        ratio = timings[-1][1] / timings[0][1]
        data_ratio = timings[-1][0] / timings[0][0]
        assert ratio < data_ratio * 3, f"Non-linear scaling: {ratio}x for {data_ratio}x data"
```

---

## Integration Testing Approach

### Component Integration Tests

```python
class TestProgressToGoalIntegration:
    @pytest.mark.asyncio
    async def test_session_triggers_goal_update(
        self, progress_tracker, goal_tracker, sample_session_progress
    ):
        """Test end-to-end: session -> progress -> goal update."""
        await progress_tracker.initialize()
        await goal_tracker.initialize()

        # Create a goal
        goal = await goal_tracker.create_goal(
            title="Complete 5 Sessions",
            goal_type=GoalType.SESSIONS,
            target_value=5
        )

        # Record sessions
        for _ in range(3):
            session = sample_session_progress()
            await progress_tracker.record_session_progress(session)

        # Get metrics and update goals
        metrics = await progress_tracker.get_overall_progress()
        progress_metrics = ProgressMetrics(total_sessions=metrics.sessions_count)

        result = await goal_tracker.update_all_goals(progress_metrics)

        assert result['goals_updated'] >= 1

    @pytest.mark.asyncio
    async def test_goal_completion_unlocks_achievement(
        self, goal_tracker, achievement_system
    ):
        """Test that completing a goal triggers achievement check."""
        await goal_tracker.initialize()
        await achievement_system.initialize()

        metrics = ProgressMetrics(
            total_sessions=10,
            current_streak=7
        )

        result = await achievement_system.check_achievements(metrics)

        assert isinstance(result.newly_unlocked, list)
```

### Data Flow Integration Tests

```python
class TestDataFlowIntegration:
    @pytest.mark.asyncio
    async def test_full_data_flow(
        self,
        progress_tracker,
        insights_engine,
        trend_analyzer,
        dashboard_service,
        sample_session_list
    ):
        """Test complete data flow through all components."""
        # Initialize
        await progress_tracker.initialize()
        await insights_engine.initialize()
        await trend_analyzer.initialize()
        await dashboard_service.initialize()

        # Record sessions
        sessions = sample_session_list(n=20, days_span=14)
        for session in sessions:
            await progress_tracker.record_session_progress(session)

        # Get metrics
        metrics = await progress_tracker.get_overall_progress()
        assert metrics.sessions_count == 20

        # Get daily progress
        daily = []
        for i in range(14):
            d = await progress_tracker.get_daily_progress(
                date.today() - timedelta(days=i)
            )
            daily.append(d)

        # Generate insights
        insights = await insights_engine.generate_insights(metrics, daily_progress=daily)
        assert isinstance(insights, list)

        # Analyze trends
        trend = await trend_analyzer.analyze_quality_trend(daily)
        assert trend is not None

        # Get dashboard
        dashboard = await dashboard_service.get_dashboard_data()
        assert dashboard.overview.sessions_count == metrics.sessions_count
```

### Database Integration Tests

```python
class TestDatabaseIntegration:
    @pytest.mark.asyncio
    async def test_progress_persists_across_restarts(self, test_config_with_temp_db):
        """Test that progress data persists across tracker restarts."""
        from app.analytics.progress_store import ProgressStore

        # First instance - write data
        store1 = ProgressStore(config=test_config_with_temp_db)
        tracker1 = ProgressTracker(
            config=test_config_with_temp_db,
            progress_store=store1
        )
        await tracker1.initialize()

        session = SessionProgress(
            session_id="persist-test",
            start_time=datetime.utcnow(),
            total_exchanges=10,
            avg_quality_score=0.8
        )
        await tracker1.record_session_progress(session)

        # Second instance - read data
        store2 = ProgressStore(config=test_config_with_temp_db)
        tracker2 = ProgressTracker(
            config=test_config_with_temp_db,
            progress_store=store2
        )
        await tracker2.initialize()

        metrics = await tracker2.get_overall_progress()
        assert metrics.sessions_count == 1
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all analytics tests
pytest tests/analytics/ -v

# Run specific test file
pytest tests/analytics/test_progress_tracker.py -v

# Run specific test class
pytest tests/analytics/test_progress_tracker.py::TestProgressTrackerStreaks -v

# Run specific test
pytest tests/analytics/test_progress_tracker.py::TestProgressTrackerStreaks::test_streak_continues -v
```

### Coverage Reports

```bash
# Run with coverage
pytest tests/analytics/ --cov=app/analytics --cov-report=html

# Generate coverage report
coverage report -m

# Coverage targets:
# - Progress tracking: 85%+
# - Insights generation: 85%+
# - Dashboard service: 80%+
# - Goal tracking: 85%+
# - Export service: 80%+
# - Overall Phase 6: 80%+
```

### Performance Tests

```bash
# Run performance tests only
pytest tests/analytics/ -v -m performance

# Run with timing information
pytest tests/analytics/ -v --durations=10
```

### CI Configuration

```yaml
# .github/workflows/test-analytics.yml
name: Analytics Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt

      - name: Run tests
        run: pytest tests/analytics/ -v --cov=app/analytics --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

---

## Coverage Requirements

| Component | Target Coverage |
|-----------|----------------|
| progress_tracker.py | 85%+ |
| insights_engine.py | 85%+ |
| trend_analyzer.py | 80%+ |
| dashboard_service.py | 80%+ |
| goal_tracker.py | 85%+ |
| achievement_system.py | 80%+ |
| export_service.py | 80%+ |
| Integration tests | 75%+ |
| **Overall Phase 6** | **80%+** |
