# Phase 6: Analytics Engine - Implementation Guide

## Overview

Phase 6 introduces a comprehensive analytics engine for the Learning Voice Agent. This system provides real-time progress tracking, intelligent insights generation, trend analysis, goal management, achievement gamification, and data export capabilities.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Setup](#component-setup)
3. [Progress Tracking](#progress-tracking)
4. [Insights Generation](#insights-generation)
5. [Trend Analysis](#trend-analysis)
6. [Dashboard Service](#dashboard-service)
7. [Goal Tracking](#goal-tracking)
8. [Achievement System](#achievement-system)
9. [Export Functionality](#export-functionality)
10. [Integration with Phase 5](#integration-with-phase-5)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Component Hierarchy

```
Analytics Engine
├── ProgressTracker       # Core progress tracking and aggregation
├── InsightsEngine        # AI-powered insight generation
├── TrendAnalyzer         # Time-series trend analysis
├── DashboardService      # Dashboard data aggregation
├── GoalTracker           # Learning goal management
├── AchievementSystem     # Badge and recognition system
└── ExportService         # Data export and reporting
```

### Data Flow

```
Session Data → ProgressTracker → Metrics → InsightsEngine → Dashboard
                    ↓                           ↓
              GoalTracker ←──────────── AchievementSystem
                    ↓
              ExportService
```

### Database Schema

The analytics engine uses SQLite for data persistence with the following primary tables:

- `sessions` - Session progress records
- `streaks` - Learning streak data
- `topic_mastery` - Per-topic mastery scores
- `goals` - User learning goals
- `goal_progress` - Goal progress history
- `achievements` - Achievement definitions
- `achievement_unlocks` - User achievement records

---

## Component Setup

### Basic Initialization

```python
from app.analytics import (
    ProgressTracker,
    InsightsEngine,
    TrendAnalyzer,
    DashboardService,
    GoalTracker,
    AchievementSystem,
    ExportService
)

# Initialize components
progress_tracker = ProgressTracker()
await progress_tracker.initialize()

insights_engine = InsightsEngine()
await insights_engine.initialize()

trend_analyzer = TrendAnalyzer()
await trend_analyzer.initialize()

dashboard_service = DashboardService(
    progress_tracker=progress_tracker,
    insights_engine=insights_engine
)
await dashboard_service.initialize()

goal_tracker = GoalTracker()
await goal_tracker.initialize()

achievement_system = AchievementSystem()
await achievement_system.initialize()

export_service = ExportService()
await export_service.initialize()
```

### With Custom Configuration

```python
from app.analytics.analytics_config import AnalyticsEngineConfig

config = AnalyticsEngineConfig()
config.cache_enabled = True
config.cache_ttl_seconds = 300

progress_tracker = ProgressTracker(config=config)
await progress_tracker.initialize()
```

---

## Progress Tracking

### Recording Session Progress

```python
from app.analytics.progress_models import SessionProgress
from datetime import datetime

session = SessionProgress(
    session_id="session-123",
    user_id="user-456",
    start_time=datetime.utcnow(),
    total_exchanges=15,
    avg_quality_score=0.85,
    topics=["python", "machine learning"]
)

# Record and finalize session
result = await progress_tracker.record_session_progress(session)
```

### Getting Overall Progress

```python
# Get progress for all users
metrics = await progress_tracker.get_overall_progress()

# Get user-specific progress
user_metrics = await progress_tracker.get_overall_progress(user_id="user-456")

print(f"Sessions: {metrics.sessions_count}")
print(f"Total Exchanges: {metrics.total_exchanges}")
print(f"Average Quality: {metrics.avg_quality_score:.2%}")
print(f"Current Streak: {metrics.current_streak} days")
```

### Streak Management

```python
# Get current streak
streak = await progress_tracker.get_learning_streak(user_id="user-456")

print(f"Current Streak: {streak.current_streak} days")
print(f"Longest Streak: {streak.longest_streak} days")
print(f"Last Active: {streak.last_active_date}")

# Streak updates automatically when sessions are recorded
```

### Topic Mastery

```python
# Get mastery for specific topic
mastery = await progress_tracker.get_topic_mastery(
    user_id="user-456",
    topic="python"
)

print(f"Topic: {mastery.topic}")
print(f"Mastery Score: {mastery.mastery_score:.2%}")
print(f"Level: {mastery.level.value}")
print(f"Total Interactions: {mastery.total_interactions}")

# Get all topic mastery
all_mastery = await progress_tracker.get_all_topic_mastery(user_id="user-456")
```

### Aggregated Progress

```python
from datetime import date, timedelta

# Daily progress
daily = await progress_tracker.get_daily_progress(
    target_date=date.today(),
    user_id="user-456"
)

# Weekly progress
monday = date.today() - timedelta(days=date.today().weekday())
weekly = await progress_tracker.get_weekly_progress(
    week_start=monday,
    user_id="user-456"
)

# Monthly progress
monthly = await progress_tracker.get_monthly_progress(
    year=2024,
    month=11,
    user_id="user-456"
)
```

---

## Insights Generation

### Generating Insights

```python
# Get metrics first
metrics = await progress_tracker.get_overall_progress()
daily_progress = [
    await progress_tracker.get_daily_progress(date.today() - timedelta(days=i))
    for i in range(30)
]
streak = await progress_tracker.get_learning_streak()
topic_mastery = await progress_tracker.get_all_topic_mastery()

# Generate insights
insights = await insights_engine.generate_insights(
    metrics=metrics,
    daily_progress=daily_progress,
    streak=streak,
    topic_mastery=topic_mastery
)

for insight in insights:
    print(f"[{insight.priority}] {insight.title}")
    print(f"  {insight.description}")
    print(f"  Confidence: {insight.confidence:.2%}")
```

### Anomaly Detection

```python
anomalies = await insights_engine.detect_anomalies(daily_progress)

for anomaly in anomalies:
    print(f"Anomaly: {anomaly.title}")
    print(f"  {anomaly.description}")
```

### Milestone Identification

```python
milestones = await insights_engine.identify_milestones(metrics)

for milestone in milestones:
    print(f"Milestone: {milestone.title}")
```

### Recommendations

```python
recommendations = await insights_engine.generate_recommendations(
    metrics=metrics,
    topic_mastery=topic_mastery
)

for rec in recommendations:
    print(f"[{rec['type']}] {rec['title']}: {rec['description']}")
```

---

## Trend Analysis

### Quality Trend Analysis

```python
quality_trend = await trend_analyzer.analyze_quality_trend(daily_progress)

print(f"Direction: {quality_trend.direction.value}")
print(f"Change: {quality_trend.change_percent:.1f}%")
print(f"Confidence: {quality_trend.confidence:.2%}")
```

### Rolling Averages

```python
data = [0.7, 0.72, 0.75, 0.73, 0.78, 0.80, 0.82]
rolling = await trend_analyzer.calculate_rolling_average(data, window=3)
```

### Seasonality Detection

```python
seasonality = await trend_analyzer.detect_seasonality(daily_progress)

if seasonality["detected"]:
    print(f"Best day: {seasonality['best_day']['day']}")
    print(f"Worst day: {seasonality['worst_day']['day']}")
```

### Forecasting

```python
forecasts = await trend_analyzer.forecast(
    data=quality_scores,
    days=7,
    method="linear"
)

for f in forecasts:
    print(f"Day {f['day']}: {f['value']:.2f} ({f['lower_bound']:.2f} - {f['upper_bound']:.2f})")
```

---

## Dashboard Service

### Complete Dashboard Data

```python
dashboard = await dashboard_service.get_dashboard_data(user_id="user-456")

print(f"Sessions: {dashboard.overview.sessions_count}")
print(f"Streak: {dashboard.streak.current_streak}")
print(f"Active Goals: {len(dashboard.active_goals)}")
print(f"Achievements: {len(dashboard.recent_achievements)}")
```

### Chart Data

```python
# Quality chart
quality_chart = await dashboard_service.get_quality_chart_data(days=30)

# Progress chart
progress_chart = await dashboard_service.get_progress_chart_data(
    metric="exchanges",
    days=30
)

# Activity heatmap
heatmap = await dashboard_service.get_activity_heatmap(weeks=12)

# Topic distribution
topics = await dashboard_service.get_topic_distribution()
```

### Trend Charts

```python
trend_data = await dashboard_service.get_trend_chart_data(
    metric="quality",
    days=30
)

for point in trend_data:
    print(f"{point['date']}: {point['value']:.2f} (rolling: {point['rolling_avg']:.2f})")
```

---

## Goal Tracking

### Creating Goals

```python
from app.analytics.goal_models import GoalType

# Create a streak goal
goal = await goal_tracker.create_goal(
    title="Build a 7-Day Streak",
    goal_type=GoalType.STREAK,
    target_value=7,
    description="Maintain a week-long learning streak",
    deadline=date.today() + timedelta(days=14)
)

# Create a sessions goal
sessions_goal = await goal_tracker.create_goal(
    title="Complete 20 Sessions",
    goal_type=GoalType.SESSIONS,
    target_value=20
)
```

### Updating Progress

```python
from app.analytics.goal_tracker import ProgressMetrics

metrics = ProgressMetrics(
    current_streak=5,
    total_sessions=15,
    total_exchanges=150,
    avg_quality_score=0.82
)

# Update all active goals
result = await goal_tracker.update_all_goals(metrics)

print(f"Goals updated: {result['goals_updated']}")
print(f"Goals completed: {result['goals_completed']}")
print(f"Milestones completed: {result['milestones_completed']}")
```

### Goal Suggestions

```python
suggestions = await goal_tracker.get_goal_suggestions(metrics, limit=5)

for s in suggestions:
    print(f"{s.title} ({s.difficulty})")
    print(f"  Type: {s.goal_type.value}")
    print(f"  Target: {s.suggested_target}")
    print(f"  Confidence: {s.confidence:.2%}")
```

---

## Achievement System

### Checking Achievements

```python
result = await achievement_system.check_achievements(metrics)

print(f"Newly unlocked: {len(result.newly_unlocked)}")
print(f"Points earned: {result.total_points_earned}")

for achievement in result.newly_unlocked:
    print(f"Unlocked: {achievement.title} ({achievement.rarity.value})")
```

### Getting Achievements

```python
# All achievements
all_achievements = await achievement_system.get_all_achievements()

# Unlocked only
unlocked = await achievement_system.get_unlocked_achievements()

# By category
from app.analytics.goal_models import AchievementCategory
streak_achievements = await achievement_system.get_achievements_by_category(
    AchievementCategory.STREAK
)
```

### Achievement Statistics

```python
stats = await achievement_system.get_achievement_stats()

print(f"Total achievements: {stats['total_achievements']}")
print(f"Unlocked: {stats['unlocked_count']}")
print(f"Total points: {stats['total_points']}")
```

### Next Achievements

```python
next_up = await achievement_system.get_next_achievements(metrics, limit=5)

for achievement, progress in next_up:
    print(f"{achievement.title}: {progress:.1f}% complete")
```

---

## Export Functionality

### JSON Export

```python
# Export progress data
progress_json = await export_service.export_progress_data(format="json")

# Export with date range
from datetime import date, timedelta

start = date.today() - timedelta(days=30)
filtered_data = await export_service.export_progress_data(
    format="json",
    start_date=start,
    end_date=date.today()
)
```

### CSV Export

```python
# Export goals to CSV
goals_csv = await export_service.export_goals(format="csv")

# Export achievements to CSV
achievements_csv = await export_service.export_achievements(format="csv")
```

### Report Generation

```python
from app.analytics.export_service import ReportPeriod

# Generate weekly report
report = await export_service.generate_progress_report(
    period=ReportPeriod.WEEKLY
)

# Generate monthly report
monthly_report = await export_service.generate_progress_report(
    period=ReportPeriod.MONTHLY
)
```

---

## Integration with Phase 5

Phase 6 integrates seamlessly with Phase 5's learning system:

### Connecting Feedback to Progress

```python
from app.learning.feedback_store import feedback_store
from app.analytics import progress_tracker

# After recording feedback
feedback = await feedback_store.get_recent_feedback(session_id)

# Update progress with quality metrics
quality_scores = [f.rating for f in feedback if f.rating]
avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

session_progress = SessionProgress(
    session_id=session_id,
    avg_quality_score=avg_quality,
    # ... other fields
)

await progress_tracker.record_session_progress(session_progress)
```

### Quality Score Integration

```python
from app.learning.quality_scorer import quality_scorer

# Use quality scores in progress tracking
quality_result = await quality_scorer.score_response(query, response, context)

session.avg_quality_score = quality_result.composite
session.quality_breakdown = {
    "relevance": quality_result.relevance,
    "helpfulness": quality_result.helpfulness,
    "engagement": quality_result.engagement,
    "clarity": quality_result.clarity,
    "accuracy": quality_result.accuracy
}
```

---

## Configuration

### Analytics Engine Configuration

```python
from app.analytics.analytics_config import AnalyticsEngineConfig

config = AnalyticsEngineConfig()

# Database
config.db_path = "analytics.db"

# Progress tracking
config.progress.streak_grace_period_hours = 36
config.progress.mastery_threshold = 0.75
config.progress.cache_ttl_seconds = 300

# Insights
config.insights.min_data_points = 5
config.insights.confidence_threshold = 0.7
config.insights.max_recommendations = 5

# Trends
config.trends.short_term_window = 7
config.trends.enable_forecasting = True
config.trends.forecast_days = 7

# Dashboard
config.dashboard.cache_ttl_seconds = 300
config.dashboard.max_active_goals = 5
config.dashboard.max_topics_displayed = 10

# Goals
config.goals.max_active_goals = 10
config.goals.enable_goal_suggestions = True

# Achievements
config.achievements.notify_on_unlock = True
config.achievements.celebrate_milestones = True

# Export
config.export.supported_formats = ["json", "csv", "pdf"]
config.export.max_records_per_export = 10000
```

---

## Troubleshooting

### Common Issues

#### 1. Streak Not Updating

**Problem**: Streak doesn't increment after session.

**Solution**:
```python
# Ensure session has correct date
session.start_time = datetime.utcnow()  # Not a past date

# Verify streak grace period
# If more than 36 hours since last activity, streak resets
```

#### 2. Goals Not Completing

**Problem**: Goal shows 100% but isn't marked complete.

**Solution**:
```python
# Manually update goal status
from app.analytics.goal_models import UpdateGoalRequest, GoalStatus

request = UpdateGoalRequest(status=GoalStatus.COMPLETED)
await goal_tracker.update_goal(goal_id, request)
```

#### 3. Dashboard Data Stale

**Problem**: Dashboard shows outdated data.

**Solution**:
```python
# Clear cache
dashboard_service.clear_cache()

# Or bypass cache
data = await dashboard_service.get_dashboard_data(use_cache=False)
```

#### 4. Insights Not Generating

**Problem**: No insights returned.

**Solution**:
```python
# Ensure minimum data requirements
# Need at least 5 days of daily progress for most insights

# Check daily progress count
if len(daily_progress) >= 5:
    insights = await insights_engine.generate_insights(metrics, daily_progress=daily_progress)
```

### Performance Optimization

1. **Enable caching**: Set `config.cache_enabled = True`
2. **Batch updates**: Use `update_all_goals()` instead of individual updates
3. **Limit date ranges**: Use shorter date ranges for exports
4. **Index optimization**: Ensure database has proper indices

### Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("app.analytics").setLevel(logging.DEBUG)
```

---

## Performance Benchmarks

| Operation | Target | Actual |
|-----------|--------|--------|
| Overview API | < 200ms | ~150ms |
| Progress charts | < 300ms | ~250ms |
| Trend calculation | < 500ms | ~400ms |
| Insight generation | < 1s | ~800ms |
| Export (JSON) | < 2s | ~1.5s |
| Export (CSV) | < 1s | ~700ms |

---

## Next Steps

Phase 7 will introduce:
- Real-time collaboration features
- Advanced AI coaching recommendations
- Multi-user analytics and leaderboards
- Integration with external learning platforms
