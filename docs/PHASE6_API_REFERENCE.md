# Phase 6: Analytics Engine - API Reference

## Table of Contents

1. [ProgressTracker API](#progresstracker-api)
2. [InsightsEngine API](#insightsengine-api)
3. [TrendAnalyzer API](#trendanalyzer-api)
4. [DashboardService API](#dashboardservice-api)
5. [GoalTracker API](#goaltracker-api)
6. [AchievementSystem API](#achievementsystem-api)
7. [ExportService API](#exportservice-api)
8. [Data Models](#data-models)
9. [Response Schemas](#response-schemas)

---

## ProgressTracker API

### Class: `ProgressTracker`

Track and aggregate learning progress across sessions.

#### Constructor

```python
ProgressTracker(
    config: Optional[AnalyticsEngineConfig] = None,
    progress_store: Optional[Any] = None,
    feedback_store: Optional[Any] = None,
    quality_store: Optional[Any] = None
)
```

#### Methods

##### `initialize() -> None`

Initialize the progress tracker and underlying stores.

```python
await tracker.initialize()
```

##### `record_session_progress(session_progress: SessionProgress) -> SessionProgress`

Record progress for a session.

**Parameters:**
- `session_progress`: SessionProgress object with session data

**Returns:** Updated SessionProgress

```python
result = await tracker.record_session_progress(session)
```

##### `get_overall_progress(user_id: Optional[str] = None) -> ProgressMetrics`

Get aggregated progress metrics.

**Parameters:**
- `user_id`: Optional user ID for user-specific progress

**Returns:** ProgressMetrics with aggregated data

```python
metrics = await tracker.get_overall_progress(user_id="user-123")
```

##### `get_learning_streak(user_id: Optional[str] = None) -> LearningStreak`

Get current learning streak data.

**Parameters:**
- `user_id`: Optional user ID

**Returns:** LearningStreak object

```python
streak = await tracker.get_learning_streak()
```

##### `get_topic_mastery(user_id: Optional[str], topic: str) -> TopicMastery`

Get mastery level for a specific topic.

**Parameters:**
- `user_id`: Optional user ID
- `topic`: Topic name

**Returns:** TopicMastery object

```python
mastery = await tracker.get_topic_mastery(None, "python")
```

##### `get_all_topic_mastery(user_id: Optional[str] = None) -> Dict[str, TopicMastery]`

Get mastery for all topics.

**Returns:** Dictionary mapping topic names to TopicMastery objects

```python
all_mastery = await tracker.get_all_topic_mastery()
```

##### `get_daily_progress(target_date: date, user_id: Optional[str] = None) -> DailyProgress`

Get progress for a specific day.

**Parameters:**
- `target_date`: Date to get progress for
- `user_id`: Optional user ID

**Returns:** DailyProgress object

```python
daily = await tracker.get_daily_progress(date.today())
```

##### `get_weekly_progress(week_start: date, user_id: Optional[str] = None) -> WeeklyProgress`

Get progress for a specific week.

**Parameters:**
- `week_start`: Monday of the week
- `user_id`: Optional user ID

**Returns:** WeeklyProgress object

```python
weekly = await tracker.get_weekly_progress(monday)
```

##### `get_monthly_progress(year: int, month: int, user_id: Optional[str] = None) -> MonthlyProgress`

Get progress for a specific month.

**Parameters:**
- `year`: Year (e.g., 2024)
- `month`: Month (1-12)
- `user_id`: Optional user ID

**Returns:** MonthlyProgress object

```python
monthly = await tracker.get_monthly_progress(2024, 11)
```

##### `create_progress_snapshot(user_id: Optional[str] = None) -> ProgressSnapshot`

Create a point-in-time progress snapshot.

**Returns:** ProgressSnapshot object

```python
snapshot = await tracker.create_progress_snapshot()
```

##### `clear_cache() -> None`

Clear all internal caches.

```python
tracker.clear_cache()
```

---

## InsightsEngine API

### Class: `InsightsEngine`

Generate actionable insights from learning data.

#### Methods

##### `initialize() -> None`

Initialize the insights engine.

##### `generate_insights(...) -> List[Insight]`

Generate all insights from available data.

**Parameters:**
- `metrics`: ProgressMetrics - Overall progress metrics
- `daily_progress`: Optional[List[DailyProgress]] - Daily progress records
- `topic_mastery`: Optional[Dict[str, TopicMastery]] - Topic mastery data
- `streak`: Optional[LearningStreak] - Streak data

**Returns:** List of Insight objects, sorted by priority

```python
insights = await engine.generate_insights(
    metrics,
    daily_progress=daily,
    streak=streak
)
```

##### `detect_anomalies(daily_progress: List[DailyProgress]) -> List[Insight]`

Detect anomalies in learning patterns.

**Parameters:**
- `daily_progress`: List of daily progress records

**Returns:** List of anomaly Insight objects

```python
anomalies = await engine.detect_anomalies(daily_progress)
```

##### `identify_milestones(metrics: ProgressMetrics) -> List[Insight]`

Identify reached milestones.

**Parameters:**
- `metrics`: Progress metrics

**Returns:** List of milestone Insight objects

```python
milestones = await engine.identify_milestones(metrics)
```

##### `generate_recommendations(...) -> List[Dict[str, Any]]`

Generate personalized recommendations.

**Parameters:**
- `metrics`: ProgressMetrics
- `topic_mastery`: Optional[Dict[str, TopicMastery]]

**Returns:** List of recommendation dictionaries

```python
recommendations = await engine.generate_recommendations(metrics)
```

---

## TrendAnalyzer API

### Class: `TrendAnalyzer`

Analyze trends and patterns in learning data.

#### Methods

##### `initialize() -> None`

Initialize the trend analyzer.

##### `analyze_quality_trend(...) -> TrendData`

Analyze quality score trends.

**Parameters:**
- `daily_progress`: List[DailyProgress] - Daily progress records
- `window_days`: Optional[int] - Analysis window (default: 7)

**Returns:** TrendData object

```python
trend = await analyzer.analyze_quality_trend(daily, window_days=14)
```

##### `analyze_activity_trend(...) -> TrendData`

Analyze activity (exchanges) trends.

**Parameters:**
- `daily_progress`: List[DailyProgress]
- `window_days`: Optional[int]

**Returns:** TrendData object

##### `analyze_engagement_trend(...) -> TrendData`

Analyze engagement (time spent) trends.

**Parameters:**
- `daily_progress`: List[DailyProgress]
- `window_days`: Optional[int]

**Returns:** TrendData object

##### `calculate_rolling_average(data: List[float], window: int = 7) -> List[float]`

Calculate rolling average for a data series.

**Parameters:**
- `data`: List of data points
- `window`: Window size (default: 7)

**Returns:** List of rolling average values

```python
rolling = await analyzer.calculate_rolling_average([1,2,3,4,5], window=3)
```

##### `detect_seasonality(...) -> Dict[str, Any]`

Detect seasonal patterns in data.

**Parameters:**
- `daily_progress`: List[DailyProgress]
- `period_days`: int (default: 7)

**Returns:** Dictionary with seasonality analysis

```python
seasonality = await analyzer.detect_seasonality(daily)
```

##### `forecast(data: List[float], days: int = 7, method: str = "linear") -> List[Dict[str, Any]]`

Forecast future values.

**Parameters:**
- `data`: Historical data points
- `days`: Number of days to forecast (default: 7)
- `method`: "linear" or "ema" (default: "linear")

**Returns:** List of forecast dictionaries

```python
forecasts = await analyzer.forecast(data, days=7, method="linear")
```

##### `compare_periods(current_data: List[float], previous_data: List[float]) -> Dict[str, Any]`

Compare two time periods.

**Returns:** Dictionary with comparison statistics

```python
comparison = await analyzer.compare_periods(current, previous)
```

##### `get_trend_summary(daily_progress: List[DailyProgress]) -> Dict[str, Any]`

Get comprehensive trend summary.

**Returns:** Dictionary with all trend analyses

```python
summary = await analyzer.get_trend_summary(daily)
```

---

## DashboardService API

### Class: `DashboardService`

Provide dashboard-ready data for visualization.

#### Methods

##### `initialize() -> None`

Initialize the dashboard service.

##### `get_dashboard_data(user_id: Optional[str] = None, use_cache: bool = True) -> DashboardData`

Get complete dashboard data.

**Parameters:**
- `user_id`: Optional user ID
- `use_cache`: Whether to use cached data (default: True)

**Returns:** DashboardData object

```python
dashboard = await service.get_dashboard_data()
```

##### `get_overview_data(user_id: Optional[str] = None) -> Dict[str, Any]`

Get overview statistics for dashboard header.

**Returns:** Dictionary with overview statistics

```python
overview = await service.get_overview_data()
```

##### `get_quality_chart_data(days: int = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]`

Get quality score data for charting.

**Parameters:**
- `days`: Number of days to include (default: config value)
- `user_id`: Optional user ID

**Returns:** List of chart data points

```python
chart = await service.get_quality_chart_data(days=30)
```

##### `get_progress_chart_data(metric: str = "exchanges", days: int = 30, user_id: Optional[str] = None) -> List[Dict[str, Any]]`

Get progress data for charting.

**Parameters:**
- `metric`: "exchanges", "sessions", or "time"
- `days`: Number of days
- `user_id`: Optional user ID

**Returns:** List of chart data points with cumulative values

##### `get_trend_chart_data(metric: str = "quality", days: int = 30, user_id: Optional[str] = None) -> List[Dict[str, Any]]`

Get trend data with rolling averages.

**Returns:** List of trend data points

##### `get_activity_heatmap(weeks: int = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]`

Get activity data for heatmap visualization.

**Parameters:**
- `weeks`: Number of weeks (default: 12)
- `user_id`: Optional user ID

**Returns:** List of heatmap data points

```python
heatmap = await service.get_activity_heatmap(weeks=12)
```

##### `get_topic_distribution(days: int = 30, user_id: Optional[str] = None) -> List[Dict[str, Any]]`

Get topic distribution data.

**Returns:** List of topic distribution data

##### `get_goal_progress_data(user_id: Optional[str] = None) -> List[Dict[str, Any]]`

Get goal progress data.

**Returns:** List of goal progress data

##### `clear_cache(user_id: Optional[str] = None) -> None`

Clear dashboard cache.

---

## GoalTracker API

### Class: `GoalTracker`

Learning goal management and tracking system.

#### Methods

##### `initialize() -> None`

Initialize the goal tracker.

##### `create_goal(...) -> Goal`

Create a new learning goal.

**Parameters:**
- `title`: str - Goal title
- `goal_type`: GoalType - Type of goal
- `target_value`: float - Target value to achieve
- `description`: Optional[str] - Description
- `unit`: Optional[str] - Unit of measurement
- `deadline`: Optional[date] - Deadline date
- `milestones`: Optional[List[Dict]] - Milestone definitions
- `initial_value`: Optional[float] - Starting value

**Returns:** Goal object

```python
goal = await tracker.create_goal(
    title="7-Day Streak",
    goal_type=GoalType.STREAK,
    target_value=7
)
```

##### `update_goal(goal_id: str, request: UpdateGoalRequest) -> Optional[Goal]`

Update an existing goal.

**Returns:** Updated Goal or None if not found

##### `delete_goal(goal_id: str) -> bool`

Delete a goal.

**Returns:** True if deleted, False if not found

##### `get_goal(goal_id: str) -> Optional[Goal]`

Get a goal by ID.

##### `get_active_goals() -> List[Goal]`

Get all active goals.

##### `get_completed_goals() -> List[Goal]`

Get all completed goals.

##### `get_all_goals() -> List[Goal]`

Get all goals.

##### `update_progress(goal_id: str, new_value: float, source: Optional[str] = None) -> Tuple[Optional[Goal], List[Milestone]]`

Update progress for a specific goal.

**Returns:** Tuple of (updated goal, newly completed milestones)

```python
goal, milestones = await tracker.update_progress("goal-id", 5)
```

##### `update_all_goals(metrics: ProgressMetrics) -> Dict[str, Any]`

Update all active goals based on metrics.

**Returns:** Dictionary with update summary

```python
result = await tracker.update_all_goals(metrics)
```

##### `get_goal_suggestions(metrics: ProgressMetrics, limit: int = 5) -> List[GoalSuggestion]`

Generate AI-powered goal suggestions.

**Returns:** List of GoalSuggestion objects

```python
suggestions = await tracker.get_goal_suggestions(metrics, limit=3)
```

##### `get_progress_history(goal_id: str, days: int = 30) -> List[GoalProgress]`

Get progress history for a goal.

---

## AchievementSystem API

### Class: `AchievementSystem`

Achievement and badge management system.

#### Methods

##### `initialize() -> None`

Initialize and seed predefined achievements.

##### `check_achievements(metrics: ProgressMetrics, session_data: Optional[Dict] = None) -> AchievementCheckResult`

Check and unlock achievements based on metrics.

**Returns:** AchievementCheckResult with newly_unlocked, progress_updated, total_points_earned

```python
result = await system.check_achievements(metrics)
```

##### `check_session_achievements(...) -> AchievementCheckResult`

Check session-specific achievements.

**Parameters:**
- `session_start_hour`: int (0-23)
- `session_duration_minutes`: float
- `session_quality`: float (0-1)
- `metrics`: ProgressMetrics

##### `get_all_achievements() -> List[Achievement]`

Get all achievements (excludes hidden unless unlocked).

##### `get_unlocked_achievements() -> List[Achievement]`

Get all unlocked achievements.

##### `get_achievements_by_category(category: AchievementCategory) -> List[Achievement]`

Get achievements by category.

##### `get_achievement(achievement_id: str) -> Optional[Achievement]`

Get a specific achievement.

##### `get_achievement_stats() -> Dict[str, Any]`

Get achievement statistics.

```python
stats = await system.get_achievement_stats()
```

##### `get_next_achievements(metrics: ProgressMetrics, limit: int = 5) -> List[Tuple[Achievement, float]]`

Get achievements closest to being unlocked.

**Returns:** List of (achievement, progress_percent) tuples

##### `check_streak_achievements(current_streak: int, longest_streak: int) -> List[Achievement]`

Check streak-related achievements.

##### `check_milestone_achievements(total_sessions: int, total_exchanges: int) -> List[Achievement]`

Check milestone achievements.

---

## ExportService API

### Class: `ExportService`

Data export and reporting service.

#### Methods

##### `initialize() -> None`

Initialize the export service.

##### `export_progress_data(format: str = "json", start_date: date = None, end_date: date = None) -> Union[Dict, str, bytes]`

Export progress data.

**Parameters:**
- `format`: "json" or "csv"
- `start_date`: Optional start date filter
- `end_date`: Optional end date filter

**Returns:** Exported data in requested format

##### `export_goals(format: str = "json") -> Union[Dict, str]`

Export goals data.

##### `export_achievements(format: str = "json") -> Union[Dict, str]`

Export achievements data.

##### `generate_progress_report(period: ReportPeriod = ReportPeriod.WEEKLY) -> Dict[str, Any]`

Generate a progress report.

**Parameters:**
- `period`: ReportPeriod.DAILY, WEEKLY, MONTHLY, or CUSTOM

**Returns:** Report dictionary

---

## Data Models

### SessionProgress

```python
class SessionProgress(BaseModel):
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: float
    total_exchanges: int
    avg_quality_score: float
    topics: List[str]
    primary_topic: Optional[str]
```

### ProgressMetrics

```python
class ProgressMetrics(BaseModel):
    sessions_count: int
    total_exchanges: int
    total_time_hours: float
    avg_quality_score: float
    learning_velocity: float
    current_streak: int
    longest_streak: int
    topics_explored: int
    topics_mastered: int
```

### LearningStreak

```python
class LearningStreak(BaseModel):
    current_streak: int
    longest_streak: int
    last_active_date: Optional[date]
    streak_start_date: Optional[date]
    streak_history: List[Dict]
```

### Goal

```python
class Goal(BaseModel):
    id: str
    title: str
    description: Optional[str]
    goal_type: GoalType
    target_value: float
    current_value: float
    status: GoalStatus
    deadline: Optional[date]
    milestones: List[Milestone]
```

### Achievement

```python
class Achievement(BaseModel):
    id: str
    title: str
    description: str
    icon: str
    category: AchievementCategory
    rarity: AchievementRarity
    requirement_type: str
    requirement_value: float
    points: int
    unlocked: bool
    hidden: bool
```

---

## Response Schemas

### Dashboard Response

```json
{
  "overview": {
    "sessions_count": 50,
    "total_exchanges": 500,
    "avg_quality_score": 0.82,
    "current_streak": 7
  },
  "streak": {
    "current_streak": 7,
    "longest_streak": 14,
    "last_active_date": "2024-11-21"
  },
  "active_goals": [...],
  "recent_achievements": [...],
  "insights": [...],
  "quality_chart_data": [...],
  "activity_heatmap_data": [...]
}
```

### Goal Progress Response

```json
{
  "id": "goal-123",
  "title": "Complete 10 Sessions",
  "type": "sessions",
  "target": 10,
  "current": 7,
  "progress": 70.0,
  "status": "in_progress",
  "end_date": "2024-11-28"
}
```

### Achievement Unlock Response

```json
{
  "newly_unlocked": [
    {
      "id": "week-warrior",
      "title": "Week Warrior",
      "rarity": "rare",
      "points": 50
    }
  ],
  "progress_updated": [...],
  "total_points_earned": 50
}
```
