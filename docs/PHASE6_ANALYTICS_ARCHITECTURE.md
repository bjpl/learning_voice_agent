# Phase 6: Analytics Engine Architecture

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Architecture Design
**Target:** Production-ready analytics dashboard for single-user Railway deployment

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Critical Design Decisions](#critical-design-decisions)
4. [Data Pipeline Architecture](#data-pipeline-architecture)
5. [Time Series Aggregation Strategy](#time-series-aggregation-strategy)
6. [Insight Generation Engine](#insight-generation-engine)
7. [Goal Tracking System](#goal-tracking-system)
8. [Dashboard API Specification](#dashboard-api-specification)
9. [Export and Reporting System](#export-and-reporting-system)
10. [Caching Strategy](#caching-strategy)
11. [Integration with Existing Phases](#integration-with-existing-phases)
12. [Performance Targets](#performance-targets)
13. [Database Schema](#database-schema)
14. [Implementation Plan](#implementation-plan)

---

## Executive Summary

### Objectives

Build a **comprehensive analytics engine** that transforms raw learning data into actionable insights:

- **Visualize** learning progress through intuitive dashboard data
- **Generate** insights and detect trends in learning behavior
- **Track** goals and achievements to motivate learning
- **Export** reports for external analysis and record-keeping
- **Integrate** with Phase 5 learning system for personalized analytics

### Key Principles

1. **Pre-compute Over Real-time**: Aggregate metrics during low-traffic periods
2. **Cache Aggressively**: Single-user system allows generous caching
3. **JSON-first API**: Simple REST endpoints returning Chart.js-ready data
4. **SQLite Native**: Leverage existing database, no additional dependencies
5. **Progressive Enhancement**: Basic dashboards work immediately, advanced insights build over time

### Success Criteria

- Dashboard API response time: < 100ms (cached), < 500ms (computed)
- Export generation time: < 5 seconds for 30-day report
- Goal progress calculation: < 50ms
- Insight generation: < 2 seconds for weekly analysis
- Zero impact on conversation response latency

---

## System Architecture Overview

### High-Level Analytics Architecture

```
+===========================================================================+
|                           DATA SOURCES                                     |
+===========================================================================+
|                                                                           |
|  +----------------+  +----------------+  +----------------+  +----------+ |
|  |   Sessions     |  |   Feedback     |  |   Quality      |  | Knowledge| |
|  |   (SQLite)     |  |   (Phase 5)    |  |   Scores       |  |  Graph   | |
|  |                |  |                |  |   (Phase 5)    |  | (Neo4j)  | |
|  | - captures     |  | - explicit     |  | - relevance    |  | - topics | |
|  | - timestamps   |  | - implicit     |  | - helpfulness  |  | - paths  | |
|  | - session_id   |  | - corrections  |  | - engagement   |  | - gaps   | |
|  +-------+--------+  +-------+--------+  +-------+--------+  +----+-----+ |
|          |                   |                   |                |       |
+===========================================================================+
           |                   |                   |                |
           +-------------------+-------------------+----------------+
                               |
                               v
+===========================================================================+
|                    AGGREGATION LAYER                                       |
+===========================================================================+
|                                                                           |
|    +------------------------------------------------------------------+   |
|    |                TIME SERIES AGGREGATOR                            |   |
|    |                                                                  |   |
|    |  Raw Data --> Hourly --> Daily --> Weekly --> Monthly            |   |
|    |                                                                  |   |
|    |  Metrics: sessions, exchanges, quality, feedback, topics         |   |
|    +------------------------------------------------------------------+   |
|                               |                                           |
|    +------------------------------------------------------------------+   |
|    |                INSIGHT GENERATOR                                 |   |
|    |                                                                  |   |
|    |  - Trend Detection (rising/falling/stable)                       |   |
|    |  - Anomaly Detection (statistical outliers)                      |   |
|    |  - Pattern Recognition (time-of-day, topic clusters)             |   |
|    |  - Milestone Detection (achievements, streaks)                   |   |
|    +------------------------------------------------------------------+   |
|                               |                                           |
|    +------------------------------------------------------------------+   |
|    |                GOAL TRACKER                                      |   |
|    |                                                                  |   |
|    |  - Progress Computation                                          |   |
|    |  - Achievement Unlocking                                         |   |
|    |  - Streak Tracking                                               |   |
|    |  - Recommendation Engine                                         |   |
|    +------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
                               |
                               v
+===========================================================================+
|                    STORAGE LAYER                                           |
+===========================================================================+
|                                                                           |
|    +--------------------+    +--------------------+    +----------------+ |
|    | aggregated_metrics |    |      insights      |    |     goals      | |
|    |                    |    |                    |    |                | |
|    | - hourly rollups   |    | - trend_insights   |    | - definitions  | |
|    | - daily rollups    |    | - anomaly_alerts   |    | - progress     | |
|    | - weekly rollups   |    | - milestones       |    | - achievements | |
|    | - monthly rollups  |    | - recommendations  |    | - streaks      | |
|    +--------------------+    +--------------------+    +----------------+ |
|                                                                           |
+===========================================================================+
                               |
                               v
+===========================================================================+
|                    API LAYER                                               |
+===========================================================================+
|                                                                           |
|    +------------------------------------------------------------------+   |
|    |                  DASHBOARD DATA API                              |   |
|    +------------------------------------------------------------------+   |
|    |                                                                  |   |
|    |  GET /api/analytics/overview     - Summary dashboard data        |   |
|    |  GET /api/analytics/progress     - Learning progress over time   |   |
|    |  GET /api/analytics/trends       - Trend charts (Chart.js)       |   |
|    |  GET /api/analytics/insights     - Generated insights list       |   |
|    |  GET /api/analytics/goals        - Goals and achievements        |   |
|    |  GET /api/analytics/topics       - Topic analytics               |   |
|    |  GET /api/analytics/export       - Report generation             |   |
|    |                                                                  |   |
|    +------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

### Component Interaction Flow

```
                    Scheduled Job (hourly)
                           |
                           v
+---------------------+    |    +----------------------+
|  Raw Session Data   |----+--->|  Time Series         |
|  (SQLite captures)  |         |  Aggregator          |
+---------------------+         +----------+-----------+
                                           |
+---------------------+                    v
|  Phase 5 Feedback   |    +--------------------------------+
|  & Quality Scores   |--->|  Aggregated Metrics Store      |
+---------------------+    |  (hourly/daily/weekly/monthly) |
                           +---------------+----------------+
+---------------------+                    |
|  Knowledge Graph    |                    v
|  (Neo4j Topics)     |--->+--------------------------------+
+---------------------+    |  Insight Generator             |
                           |  (runs daily, on-demand)       |
                           +---------------+----------------+
                                           |
                                           v
                           +--------------------------------+
                           |  Dashboard Cache               |
                           |  (5-minute TTL for overview)   |
                           +---------------+----------------+
                                           |
         API Request                       |
              |                            v
              +---------->+--------------------------------+
                          |  Dashboard API Endpoints       |
                          |  (JSON responses, Chart.js)    |
                          +--------------------------------+
```

---

## Critical Design Decisions

### Decision 1: Dashboard Data Format

**Choice: JSON REST API with Chart.js-optimized structure**

| Option | Pros | Cons | Complexity |
|--------|------|------|------------|
| **JSON REST API** | Simple, flexible, frontend-agnostic | Multiple requests for full dashboard | Low |
| GraphQL | Single request, client specifies data | Complexity, overkill for single-user | High |
| Server-rendered views | Fast, SEO-friendly | Less interactive, more server load | Medium |

**Rationale**: For a single-user Railway deployment, REST API simplicity wins. Chart.js expects specific data structures (labels array, datasets array) which we can generate server-side. No benefit from GraphQL's flexibility when there's one dashboard consumer.

**API Response Format**:
```json
{
  "chart": {
    "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "datasets": [
      {
        "label": "Quality Score",
        "data": [0.75, 0.78, 0.72, 0.80, 0.82, 0.79, 0.85],
        "borderColor": "#4F46E5",
        "fill": false
      }
    ]
  },
  "summary": {
    "current": 0.85,
    "previous": 0.79,
    "change": 7.6,
    "trend": "improving"
  },
  "meta": {
    "period": "7d",
    "generated_at": "2025-11-21T10:30:00Z",
    "cached": true
  }
}
```

---

### Decision 2: Time Series Storage

**Choice: SQLite Aggregation Tables (Extend Existing Database)**

| Option | Pros | Cons | Railway Cost |
|--------|------|------|--------------|
| **SQLite aggregations** | Simple, ACID, no dependencies | Limited query flexibility | $0/mo |
| TimescaleDB | Time-series optimized, compression | External service, complexity | ~$15/mo |
| In-memory only | Fastest reads | Lost on restart, memory limits | $0/mo |
| InfluxDB | Purpose-built for metrics | Another service to manage | ~$10/mo |

**Rationale**: Single-user system doesn't need time-series database features. SQLite with pre-aggregated tables provides sufficient query performance. Aggregation happens during scheduled jobs, not at query time.

**Aggregation Strategy**:
```
Raw Data (captures table)
    |
    +---> Hourly Rollups (kept 7 days)
              |
              +---> Daily Rollups (kept 90 days)
                        |
                        +---> Weekly Rollups (kept 1 year)
                                  |
                                  +---> Monthly Rollups (kept indefinitely)
```

---

### Decision 3: Visualization Library Data Format

**Choice: Chart.js data structure**

| Library | Pros | Cons | Bundle Size |
|---------|------|------|-------------|
| **Chart.js** | Simple, widely used, good docs | Less customizable than D3 | ~65KB |
| D3.js | Maximum flexibility | Steep learning curve, verbose | ~250KB |
| Plotly | Interactive, scientific | Heavy, complex API | ~1MB |
| Recharts | React-native, declarative | React-only | ~40KB |

**Rationale**: Chart.js provides the best balance of simplicity and capability for dashboards. Its data format is straightforward, and it handles responsive design well. The API returns data pre-formatted for Chart.js consumption.

**Supported Chart Types**:
- Line charts: Quality trends, session activity
- Bar charts: Topic distribution, feedback breakdown
- Doughnut charts: Goal progress, quality distribution
- Radar charts: Multi-dimensional quality analysis

---

### Decision 4: Goal System Design

**Choice: Simple Milestones with Streaks (No Gamification)**

| Approach | Pros | Cons | User Experience |
|----------|------|------|-----------------|
| **Simple milestones** | Clear, achievable, motivating | Less engaging long-term | Straightforward |
| Gamification (badges, XP) | Engaging, habit-forming | Can feel manipulative | Game-like |
| Learning paths | Structured, educational | Complex to implement | Course-like |

**Rationale**: For a single-user learning tool, simple milestones provide motivation without the overhead of gamification systems. Streaks encourage consistency. Learning paths are valuable but belong in a future phase with curriculum design.

**Goal Types**:
1. **Activity Goals**: Sessions per week, exchanges per day
2. **Quality Goals**: Maintain average quality above threshold
3. **Topic Goals**: Explore N new topics, depth in specific area
4. **Streak Goals**: Consecutive days of learning
5. **Milestone Goals**: Total sessions, exchanges, topics explored

---

### Decision 5: Export Formats

**Choice: JSON + CSV (PDF deferred)**

| Format | Pros | Cons | Implementation |
|--------|------|------|----------------|
| **JSON** | Machine-readable, full fidelity | Not human-friendly | Native |
| **CSV** | Spreadsheet compatible, simple | Flat structure only | Simple |
| PDF | Professional, printable | Complex generation, dependencies | ReportLab/WeasyPrint |
| HTML | Rich formatting, printable | Needs styling | Template-based |

**Rationale**: JSON for programmatic use, CSV for spreadsheet analysis. PDF generation adds dependencies (ReportLab, fonts) that complicate Railway deployment. HTML export could be added later as a middle ground.

**Export Options**:
```python
GET /api/analytics/export?format=json&period=30d&include=all
GET /api/analytics/export?format=csv&period=7d&include=sessions,quality
```

---

## Data Pipeline Architecture

### Data Collection Points

```
+===========================================================================+
|                    DATA COLLECTION POINTS                                  |
+===========================================================================+

1. SESSION DATA (from app/database.py)
   +------------------------------------------------------------------+
   |  Source: captures table                                          |
   |  Frequency: Real-time (on every exchange)                        |
   |  Fields: session_id, timestamp, user_text, agent_text, metadata  |
   +------------------------------------------------------------------+

2. FEEDBACK DATA (from app/learning/stores.py)
   +------------------------------------------------------------------+
   |  Source: feedback_explicit, feedback_implicit tables             |
   |  Frequency: Real-time (on user feedback)                         |
   |  Fields: exchange_id, feedback_type, rating, timestamp           |
   +------------------------------------------------------------------+

3. QUALITY SCORES (from app/learning/quality_scorer.py)
   +------------------------------------------------------------------+
   |  Source: quality_scores table                                    |
   |  Frequency: Per exchange (computed async)                        |
   |  Fields: relevance, helpfulness, accuracy, clarity, composite    |
   +------------------------------------------------------------------+

4. KNOWLEDGE GRAPH (from app/knowledge_graph/graph_store.py)
   +------------------------------------------------------------------+
   |  Source: Neo4j nodes and relationships                           |
   |  Frequency: On-demand query                                      |
   |  Fields: concepts, relationships, frequencies, paths             |
   +------------------------------------------------------------------+

5. PATTERN DATA (from app/learning/pattern_detector.py)
   +------------------------------------------------------------------+
   |  Source: patterns table                                          |
   |  Frequency: Computed periodically                                |
   |  Fields: pattern_type, confidence, examples, timestamps          |
   +------------------------------------------------------------------+
```

### Aggregation Pipeline

```
+===========================================================================+
|                    AGGREGATION PIPELINE                                    |
+===========================================================================+

STAGE 1: Raw Data Collection (Continuous)
+------------------------------------------------------------------------+
|                                                                        |
|   Every Exchange                                                       |
|        |                                                               |
|        +---> Save to captures table                                    |
|        +---> Queue for quality scoring (background)                    |
|        +---> Update session state                                      |
|                                                                        |
+------------------------------------------------------------------------+

STAGE 2: Hourly Aggregation (Scheduled: every hour at :05)
+------------------------------------------------------------------------+
|                                                                        |
|   For each hour in the past hour:                                      |
|        |                                                               |
|        +---> Count sessions started                                    |
|        +---> Count total exchanges                                     |
|        +---> Calculate average quality scores                          |
|        +---> Summarize feedback (positive/negative/corrections)        |
|        +---> Identify active topics                                    |
|        +---> Store in aggregated_metrics (interval='hourly')           |
|                                                                        |
+------------------------------------------------------------------------+

STAGE 3: Daily Aggregation (Scheduled: 00:15 UTC)
+------------------------------------------------------------------------+
|                                                                        |
|   For the previous day:                                                |
|        |                                                               |
|        +---> Aggregate hourly metrics into daily                       |
|        +---> Calculate daily statistics (min, max, avg, stddev)        |
|        +---> Compute day-over-day changes                              |
|        +---> Generate daily insights                                   |
|        +---> Check goal progress                                       |
|        +---> Update streak tracking                                    |
|        +---> Store in aggregated_metrics (interval='daily')            |
|                                                                        |
+------------------------------------------------------------------------+

STAGE 4: Weekly Aggregation (Scheduled: Mondays 01:00 UTC)
+------------------------------------------------------------------------+
|                                                                        |
|   For the previous week:                                               |
|        |                                                               |
|        +---> Aggregate daily metrics into weekly                       |
|        +---> Calculate week-over-week trends                           |
|        +---> Generate weekly insights                                  |
|        +---> Compute topic evolution                                   |
|        +---> Check weekly goals                                        |
|        +---> Store in aggregated_metrics (interval='weekly')           |
|                                                                        |
+------------------------------------------------------------------------+

STAGE 5: Monthly Aggregation (Scheduled: 1st of month 02:00 UTC)
+------------------------------------------------------------------------+
|                                                                        |
|   For the previous month:                                              |
|        |                                                               |
|        +---> Aggregate weekly metrics into monthly                     |
|        +---> Calculate month-over-month trends                         |
|        +---> Generate monthly summary report                           |
|        +---> Compute learning velocity metrics                         |
|        +---> Archive detailed data (retention policy)                  |
|        +---> Store in aggregated_metrics (interval='monthly')          |
|                                                                        |
+------------------------------------------------------------------------+
```

---

## Time Series Aggregation Strategy

### Metrics Computed at Each Level

```python
@dataclass
class AggregatedMetrics:
    """Metrics computed at each aggregation level"""

    # Time period
    period_start: datetime
    period_end: datetime
    interval: str  # hourly, daily, weekly, monthly

    # Volume metrics
    session_count: int
    exchange_count: int
    unique_topics: int

    # Duration metrics
    total_duration_minutes: float
    avg_session_duration: float
    median_session_duration: float

    # Quality metrics
    avg_quality_composite: float
    avg_quality_relevance: float
    avg_quality_helpfulness: float
    avg_quality_accuracy: float
    avg_quality_clarity: float
    quality_std_dev: float

    # Feedback metrics
    feedback_count: int
    positive_feedback_rate: float
    correction_count: int
    clarification_count: int

    # Engagement metrics
    avg_exchanges_per_session: float
    peak_hour: Optional[int]  # Only for daily+
    day_distribution: Optional[Dict[str, int]]  # Only for weekly+

    # Topic metrics
    top_topics: List[str]  # Top 5
    topic_distribution: Dict[str, int]

    # Computed trends (only for daily+)
    quality_change: Optional[float]  # vs previous period
    volume_change: Optional[float]
    trend_direction: Optional[str]  # improving, declining, stable
```

### Aggregation SQL Queries

```sql
-- Hourly aggregation query
INSERT INTO aggregated_metrics (
    period_start, period_end, interval,
    session_count, exchange_count, avg_quality_composite,
    feedback_count, positive_feedback_rate
)
SELECT
    datetime(strftime('%Y-%m-%d %H:00:00', timestamp)) as period_start,
    datetime(strftime('%Y-%m-%d %H:00:00', timestamp), '+1 hour') as period_end,
    'hourly' as interval,
    COUNT(DISTINCT session_id) as session_count,
    COUNT(*) as exchange_count,
    (SELECT AVG(composite) FROM quality_scores
     WHERE timestamp >= datetime(strftime('%Y-%m-%d %H:00:00', c.timestamp))
       AND timestamp < datetime(strftime('%Y-%m-%d %H:00:00', c.timestamp), '+1 hour')
    ) as avg_quality_composite,
    (SELECT COUNT(*) FROM feedback_explicit
     WHERE created_at >= datetime(strftime('%Y-%m-%d %H:00:00', c.timestamp))
       AND created_at < datetime(strftime('%Y-%m-%d %H:00:00', c.timestamp), '+1 hour')
    ) as feedback_count,
    (SELECT
        CAST(SUM(CASE WHEN feedback_type IN ('thumbs_up', 'copy') THEN 1 ELSE 0 END) AS FLOAT) /
        NULLIF(COUNT(*), 0)
     FROM feedback_explicit
     WHERE created_at >= datetime(strftime('%Y-%m-%d %H:00:00', c.timestamp))
       AND created_at < datetime(strftime('%Y-%m-%d %H:00:00', c.timestamp), '+1 hour')
    ) as positive_feedback_rate
FROM captures c
WHERE timestamp >= datetime('now', '-2 hours')
  AND timestamp < datetime('now', '-1 hour')
GROUP BY strftime('%Y-%m-%d %H', timestamp);

-- Daily aggregation from hourly
INSERT INTO aggregated_metrics (
    period_start, period_end, interval,
    session_count, exchange_count, avg_quality_composite,
    quality_change, trend_direction
)
SELECT
    date(period_start) as period_start,
    date(period_start, '+1 day') as period_end,
    'daily' as interval,
    SUM(session_count) as session_count,
    SUM(exchange_count) as exchange_count,
    AVG(avg_quality_composite) as avg_quality_composite,
    AVG(avg_quality_composite) - LAG(AVG(avg_quality_composite)) OVER (
        ORDER BY date(period_start)
    ) as quality_change,
    CASE
        WHEN AVG(avg_quality_composite) > LAG(AVG(avg_quality_composite)) OVER (ORDER BY date(period_start)) + 0.05 THEN 'improving'
        WHEN AVG(avg_quality_composite) < LAG(AVG(avg_quality_composite)) OVER (ORDER BY date(period_start)) - 0.05 THEN 'declining'
        ELSE 'stable'
    END as trend_direction
FROM aggregated_metrics
WHERE interval = 'hourly'
  AND date(period_start) = date('now', '-1 day')
GROUP BY date(period_start);
```

### Rolling Window Calculations

```python
class TrendCalculator:
    """Calculate rolling trends for dashboard"""

    ROLLING_WINDOWS = {
        '7d': 7,
        '14d': 14,
        '30d': 30,
        '90d': 90
    }

    def calculate_rolling_average(
        self,
        daily_values: List[float],
        window_size: int = 7
    ) -> List[float]:
        """
        Calculate rolling average for trend smoothing

        ALGORITHM: Simple Moving Average (SMA)
        WHY: Reduces noise while preserving trend direction
        """
        if len(daily_values) < window_size:
            return daily_values

        rolling = []
        for i in range(len(daily_values)):
            if i < window_size - 1:
                # Not enough data for full window
                rolling.append(sum(daily_values[:i+1]) / (i+1))
            else:
                window = daily_values[i - window_size + 1:i + 1]
                rolling.append(sum(window) / window_size)

        return rolling

    def detect_trend(
        self,
        values: List[float],
        threshold: float = 0.05
    ) -> str:
        """
        Detect trend direction using linear regression slope

        RETURNS: 'improving', 'declining', or 'stable'
        """
        if len(values) < 3:
            return 'stable'

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable'

        slope = numerator / denominator

        # Normalize slope relative to mean
        normalized_slope = slope / y_mean if y_mean != 0 else 0

        if normalized_slope > threshold:
            return 'improving'
        elif normalized_slope < -threshold:
            return 'declining'
        return 'stable'
```

---

## Insight Generation Engine

### Insight Types

```
+===========================================================================+
|                    INSIGHT TYPES                                           |
+===========================================================================+

TYPE 1: TREND INSIGHTS
+------------------------------------------------------------------------+
|  Detect significant changes in metrics over time                       |
|                                                                        |
|  Examples:                                                             |
|  - "Your quality scores have improved 15% this week"                   |
|  - "Session frequency is declining - down 30% from last week"          |
|  - "You're most engaged on Tuesday mornings"                           |
+------------------------------------------------------------------------+

TYPE 2: ANOMALY INSIGHTS
+------------------------------------------------------------------------+
|  Identify statistical outliers and unusual patterns                    |
|                                                                        |
|  Examples:                                                             |
|  - "Yesterday's quality score (0.92) was unusually high"               |
|  - "No sessions recorded in the past 3 days (typical: 2/day)"          |
|  - "Correction rate spiked to 25% (typical: 5%)"                       |
+------------------------------------------------------------------------+

TYPE 3: MILESTONE INSIGHTS
+------------------------------------------------------------------------+
|  Celebrate achievements and progress markers                           |
|                                                                        |
|  Examples:                                                             |
|  - "Congratulations! You've completed 100 learning sessions"           |
|  - "New topic unlocked: Machine Learning (5 related concepts)"         |
|  - "7-day learning streak achieved!"                                   |
+------------------------------------------------------------------------+

TYPE 4: RECOMMENDATION INSIGHTS
+------------------------------------------------------------------------+
|  Suggest actions based on patterns                                     |
|                                                                        |
|  Examples:                                                             |
|  - "Consider revisiting 'Distributed Systems' - quality was lower"     |
|  - "Your best learning time is 9-11am - schedule sessions then"        |
|  - "Try asking more follow-up questions for deeper understanding"      |
+------------------------------------------------------------------------+

TYPE 5: KNOWLEDGE GRAPH INSIGHTS
+------------------------------------------------------------------------+
|  Derive insights from topic relationships                              |
|                                                                        |
|  Examples:                                                             |
|  - "You've built strong connections between ML and Statistics"         |
|  - "Knowledge gap identified: Consider learning 'Neural Networks'"     |
|  - "Your learning path suggests interest in Systems Design"            |
+------------------------------------------------------------------------+
```

### Insight Generation Algorithm

```python
@dataclass
class Insight:
    """Generated insight for dashboard"""
    id: str
    type: str  # trend, anomaly, milestone, recommendation, knowledge
    title: str
    description: str
    importance: str  # high, medium, low
    data: Dict[str, Any]  # Supporting data for visualization
    generated_at: datetime
    expires_at: Optional[datetime]  # Some insights are time-sensitive
    action_url: Optional[str]  # Link to relevant dashboard section

class InsightGenerator:
    """
    PATTERN: Rule-based insight generation with statistical backing
    WHY: Interpretable insights without ML complexity
    """

    INSIGHT_RULES = [
        # Trend insights
        {
            'type': 'trend',
            'condition': lambda m: m.quality_change and m.quality_change > 0.10,
            'template': "Quality scores improved {change:.0%} this {period}",
            'importance': 'high'
        },
        {
            'type': 'trend',
            'condition': lambda m: m.quality_change and m.quality_change < -0.10,
            'template': "Quality scores declined {change:.0%} this {period}",
            'importance': 'high'
        },
        {
            'type': 'trend',
            'condition': lambda m: m.volume_change and m.volume_change > 0.50,
            'template': "Learning activity up {change:.0%} from last {period}",
            'importance': 'medium'
        },

        # Anomaly insights
        {
            'type': 'anomaly',
            'condition': lambda m: m.avg_quality > m.historical_avg + 2 * m.historical_std,
            'template': "Exceptional quality day! Score of {score:.0%} (typical: {typical:.0%})",
            'importance': 'medium'
        },
        {
            'type': 'anomaly',
            'condition': lambda m: m.session_count == 0 and m.typical_sessions > 0,
            'template': "No sessions yesterday (you typically have {typical} sessions)",
            'importance': 'low'
        },

        # Milestone insights
        {
            'type': 'milestone',
            'condition': lambda m: m.total_sessions in [10, 25, 50, 100, 250, 500, 1000],
            'template': "{count} learning sessions completed!",
            'importance': 'high'
        },
        {
            'type': 'milestone',
            'condition': lambda m: m.streak_days in [7, 14, 30, 60, 90],
            'template': "{days}-day learning streak achieved!",
            'importance': 'high'
        },

        # Recommendation insights
        {
            'type': 'recommendation',
            'condition': lambda m: m.best_hour is not None and m.sessions_at_best_hour > 3,
            'template': "Your peak learning time is {hour}:00 - schedule sessions then",
            'importance': 'medium'
        },
        {
            'type': 'recommendation',
            'condition': lambda m: m.correction_rate > 0.15,
            'template': "High correction rate ({rate:.0%}) - try breaking down complex topics",
            'importance': 'medium'
        }
    ]

    async def generate_insights(
        self,
        metrics: AggregatedMetrics,
        historical_data: List[AggregatedMetrics]
    ) -> List[Insight]:
        """
        Generate insights based on current metrics and history

        ALGORITHM:
        1. Calculate statistical baselines from history
        2. Evaluate each rule against current metrics
        3. Deduplicate similar insights
        4. Rank by importance and recency
        5. Return top N insights
        """
        insights = []

        # Calculate historical baselines
        baselines = self._calculate_baselines(historical_data)
        metrics_with_context = self._enrich_metrics(metrics, baselines)

        # Evaluate rules
        for rule in self.INSIGHT_RULES:
            try:
                if rule['condition'](metrics_with_context):
                    insight = self._create_insight(rule, metrics_with_context)
                    insights.append(insight)
            except Exception:
                continue  # Skip rules that can't be evaluated

        # Add knowledge graph insights
        kg_insights = await self._generate_knowledge_insights()
        insights.extend(kg_insights)

        # Deduplicate and rank
        insights = self._deduplicate(insights)
        insights.sort(key=lambda i: (
            {'high': 0, 'medium': 1, 'low': 2}[i.importance],
            -i.generated_at.timestamp()
        ))

        return insights[:10]  # Return top 10
```

---

## Goal Tracking System

### Goal System Architecture

```
+===========================================================================+
|                    GOAL TRACKING SYSTEM                                    |
+===========================================================================+

                      +-------------------+
                      |   Goal Tracker    |
                      |                   |
                      | - Load goals      |
                      | - Check progress  |
                      | - Unlock achieve  |
                      | - Track streaks   |
                      +--------+----------+
                               |
          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
+----------------+   +------------------+   +----------------+
| Activity Goals |   | Quality Goals    |   | Topic Goals    |
|                |   |                  |   |                |
| - Sessions/wk  |   | - Avg quality    |   | - New topics   |
| - Exchanges/d  |   | - Quality streak |   | - Topic depth  |
| - Study time   |   | - Zero errors    |   | - Breadth      |
+----------------+   +------------------+   +----------------+

          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
+----------------+   +------------------+   +----------------+
| Streak Goals   |   | Milestone Goals  |   | Custom Goals   |
|                |   |                  |   |                |
| - Daily login  |   | - 100 sessions   |   | - User defined |
| - Weekly actve |   | - 1000 exchanges |   | - Flexible     |
| - Perfect week |   | - 50 topics      |   | - Deadline     |
+----------------+   +------------------+   +----------------+
```

### Goal Data Model

```python
class GoalType(str, Enum):
    """Types of goals"""
    ACTIVITY = "activity"        # Session/exchange counts
    QUALITY = "quality"          # Quality score targets
    TOPIC = "topic"              # Topic exploration
    STREAK = "streak"            # Consecutive activity
    MILESTONE = "milestone"      # Cumulative achievements
    CUSTOM = "custom"            # User-defined

class GoalFrequency(str, Enum):
    """Goal evaluation frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ONE_TIME = "one_time"        # Milestones

@dataclass
class GoalDefinition:
    """Definition of a goal"""
    id: str
    name: str
    description: str
    goal_type: GoalType
    frequency: GoalFrequency

    # Target
    target_metric: str           # e.g., 'session_count', 'avg_quality'
    target_value: float          # Target to reach
    comparison: str              # gte, lte, eq

    # Display
    icon: str                    # Emoji or icon name
    color: str                   # For progress visualization

    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GoalProgress:
    """Current progress on a goal"""
    goal_id: str
    current_value: float
    target_value: float
    progress_percent: float      # 0-100
    status: str                  # in_progress, completed, failed
    period_start: datetime
    period_end: datetime
    updated_at: datetime

    @property
    def is_complete(self) -> bool:
        return self.status == 'completed'

@dataclass
class Achievement:
    """Unlocked achievement"""
    id: str
    goal_id: str
    name: str
    description: str
    icon: str
    unlocked_at: datetime
    period: str                  # "Week of Nov 18, 2025"
```

### Default Goals

```python
DEFAULT_GOALS = [
    # Activity Goals
    GoalDefinition(
        id="weekly_sessions",
        name="Weekly Learner",
        description="Complete at least 5 learning sessions this week",
        goal_type=GoalType.ACTIVITY,
        frequency=GoalFrequency.WEEKLY,
        target_metric="session_count",
        target_value=5,
        comparison="gte",
        icon="calendar",
        color="#4F46E5"
    ),
    GoalDefinition(
        id="daily_engagement",
        name="Daily Engagement",
        description="Have at least 10 exchanges today",
        goal_type=GoalType.ACTIVITY,
        frequency=GoalFrequency.DAILY,
        target_metric="exchange_count",
        target_value=10,
        comparison="gte",
        icon="message",
        color="#10B981"
    ),

    # Quality Goals
    GoalDefinition(
        id="quality_target",
        name="Quality Champion",
        description="Maintain average quality score above 80%",
        goal_type=GoalType.QUALITY,
        frequency=GoalFrequency.WEEKLY,
        target_metric="avg_quality_composite",
        target_value=0.80,
        comparison="gte",
        icon="star",
        color="#F59E0B"
    ),

    # Streak Goals
    GoalDefinition(
        id="daily_streak",
        name="Consistency King",
        description="Learn something every day",
        goal_type=GoalType.STREAK,
        frequency=GoalFrequency.DAILY,
        target_metric="sessions_today",
        target_value=1,
        comparison="gte",
        icon="fire",
        color="#EF4444"
    ),

    # Milestone Goals
    GoalDefinition(
        id="100_sessions",
        name="Century Club",
        description="Complete 100 total learning sessions",
        goal_type=GoalType.MILESTONE,
        frequency=GoalFrequency.ONE_TIME,
        target_metric="total_sessions",
        target_value=100,
        comparison="gte",
        icon="trophy",
        color="#8B5CF6"
    ),
    GoalDefinition(
        id="50_topics",
        name="Knowledge Explorer",
        description="Explore 50 different topics",
        goal_type=GoalType.MILESTONE,
        frequency=GoalFrequency.ONE_TIME,
        target_metric="unique_topics",
        target_value=50,
        comparison="gte",
        icon="map",
        color="#06B6D4"
    )
]
```

### Goal Progress Calculation

```python
class GoalTracker:
    """
    PATTERN: Periodic goal evaluation with caching
    WHY: Efficient progress tracking without constant recalculation
    """

    async def calculate_progress(
        self,
        goal: GoalDefinition,
        metrics: AggregatedMetrics,
        cumulative_metrics: Dict[str, Any]
    ) -> GoalProgress:
        """
        Calculate current progress for a goal

        ALGORITHM:
        1. Get current metric value based on goal frequency
        2. Calculate progress percentage
        3. Determine status (in_progress, completed, failed)
        4. Handle streak logic separately
        """
        # Get period boundaries
        period_start, period_end = self._get_period_boundaries(goal.frequency)

        # Get current value
        if goal.frequency == GoalFrequency.ONE_TIME:
            current_value = cumulative_metrics.get(goal.target_metric, 0)
        else:
            current_value = getattr(metrics, goal.target_metric, 0)

        # Calculate progress
        if goal.comparison == 'gte':
            progress_percent = min(100, (current_value / goal.target_value) * 100)
            is_complete = current_value >= goal.target_value
        elif goal.comparison == 'lte':
            progress_percent = min(100, (goal.target_value / max(current_value, 0.001)) * 100)
            is_complete = current_value <= goal.target_value
        else:  # eq
            progress_percent = 100 if current_value == goal.target_value else 0
            is_complete = current_value == goal.target_value

        # Determine status
        if is_complete:
            status = 'completed'
        elif period_end < datetime.utcnow():
            status = 'failed'
        else:
            status = 'in_progress'

        return GoalProgress(
            goal_id=goal.id,
            current_value=current_value,
            target_value=goal.target_value,
            progress_percent=progress_percent,
            status=status,
            period_start=period_start,
            period_end=period_end,
            updated_at=datetime.utcnow()
        )

    async def update_streak(
        self,
        streak_goal: GoalDefinition,
        daily_metrics: List[AggregatedMetrics]
    ) -> Tuple[int, bool]:
        """
        Calculate current streak and whether it's active today

        ALGORITHM:
        1. Sort daily metrics by date (descending)
        2. Count consecutive days meeting target
        3. Check if today's target is met
        """
        if not daily_metrics:
            return 0, False

        # Sort by date descending
        sorted_metrics = sorted(
            daily_metrics,
            key=lambda m: m.period_start,
            reverse=True
        )

        streak = 0
        today = date.today()
        today_met = False

        for metrics in sorted_metrics:
            metric_date = metrics.period_start.date()
            value = getattr(metrics, streak_goal.target_metric, 0)
            meets_target = value >= streak_goal.target_value

            if metric_date == today:
                today_met = meets_target
                if meets_target:
                    streak += 1
                continue

            # Check for gap
            expected_date = today - timedelta(days=streak)
            if metric_date != expected_date:
                break

            if meets_target:
                streak += 1
            else:
                break

        return streak, today_met
```

---

## Dashboard API Specification

### API Endpoints

```
+===========================================================================+
|                    DASHBOARD API ENDPOINTS                                 |
+===========================================================================+

BASE URL: /api/analytics

+------------------------------------------------------------------------+
| GET /overview                                                          |
+------------------------------------------------------------------------+
| Summary dashboard with key metrics                                     |
|                                                                        |
| Query Params:                                                          |
|   - period: 7d, 30d, 90d (default: 30d)                                |
|                                                                        |
| Response:                                                              |
| {                                                                      |
|   "summary": {                                                         |
|     "total_sessions": 47,                                              |
|     "total_exchanges": 312,                                            |
|     "avg_quality": 0.78,                                               |
|     "streak_days": 12,                                                 |
|     "unique_topics": 23                                                |
|   },                                                                   |
|   "trends": {                                                          |
|     "sessions": { "value": 47, "change": 15.2, "direction": "up" },    |
|     "quality": { "value": 0.78, "change": 3.5, "direction": "up" },    |
|     "engagement": { "value": 6.6, "change": -2.1, "direction": "down" }|
|   },                                                                   |
|   "recent_insights": [...],                                            |
|   "active_goals": [...],                                               |
|   "meta": { "period": "30d", "cached": true, "generated_at": "..." }   |
| }                                                                      |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| GET /progress                                                          |
+------------------------------------------------------------------------+
| Learning progress over time (Chart.js ready)                           |
|                                                                        |
| Query Params:                                                          |
|   - period: 7d, 30d, 90d, 365d                                         |
|   - metrics: sessions,quality,exchanges (comma-separated)              |
|   - granularity: daily, weekly (auto-selected based on period)         |
|                                                                        |
| Response:                                                              |
| {                                                                      |
|   "chart": {                                                           |
|     "labels": ["Nov 14", "Nov 15", "Nov 16", ...],                     |
|     "datasets": [                                                      |
|       {                                                                |
|         "label": "Sessions",                                           |
|         "data": [3, 2, 4, 1, 5, 2, 3],                                  |
|         "borderColor": "#4F46E5",                                      |
|         "yAxisID": "y-sessions"                                        |
|       },                                                               |
|       {                                                                |
|         "label": "Quality",                                            |
|         "data": [0.75, 0.78, 0.72, 0.80, 0.82, 0.79, 0.85],             |
|         "borderColor": "#10B981",                                      |
|         "yAxisID": "y-quality"                                         |
|       }                                                                |
|     ]                                                                  |
|   },                                                                   |
|   "statistics": {                                                      |
|     "sessions": { "total": 20, "avg": 2.9, "max": 5, "min": 1 },       |
|     "quality": { "avg": 0.79, "max": 0.85, "min": 0.72, "trend": "up" }|
|   }                                                                    |
| }                                                                      |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| GET /trends                                                            |
+------------------------------------------------------------------------+
| Trend analysis with rolling averages                                   |
|                                                                        |
| Query Params:                                                          |
|   - period: 30d, 90d, 365d                                             |
|   - metric: quality, sessions, engagement                              |
|   - window: 7, 14, 30 (rolling average window)                         |
|                                                                        |
| Response:                                                              |
| {                                                                      |
|   "chart": {                                                           |
|     "labels": [...dates...],                                           |
|     "datasets": [                                                      |
|       { "label": "Daily", "data": [...], "borderColor": "#E5E7EB" },   |
|       { "label": "7-day Avg", "data": [...], "borderColor": "#4F46E5" }|
|     ]                                                                  |
|   },                                                                   |
|   "trend": {                                                           |
|     "direction": "improving",                                          |
|     "slope": 0.0023,                                                   |
|     "confidence": 0.85,                                                |
|     "prediction_30d": 0.82                                             |
|   }                                                                    |
| }                                                                      |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| GET /insights                                                          |
+------------------------------------------------------------------------+
| Generated insights and recommendations                                 |
|                                                                        |
| Query Params:                                                          |
|   - limit: 10 (default)                                                |
|   - type: trend, anomaly, milestone, recommendation (optional filter)  |
|   - importance: high, medium, low (optional filter)                    |
|                                                                        |
| Response:                                                              |
| {                                                                      |
|   "insights": [                                                        |
|     {                                                                  |
|       "id": "ins_abc123",                                              |
|       "type": "trend",                                                 |
|       "title": "Quality Improving",                                    |
|       "description": "Your quality scores improved 12% this week",     |
|       "importance": "high",                                            |
|       "icon": "trending-up",                                           |
|       "data": { "change": 0.12, "period": "7d" },                      |
|       "generated_at": "2025-11-21T08:00:00Z"                           |
|     },                                                                 |
|     ...                                                                |
|   ],                                                                   |
|   "total": 8,                                                          |
|   "new_since_last_visit": 3                                            |
| }                                                                      |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| GET /goals                                                             |
+------------------------------------------------------------------------+
| Goal progress and achievements                                         |
|                                                                        |
| Response:                                                              |
| {                                                                      |
|   "active_goals": [                                                    |
|     {                                                                  |
|       "id": "weekly_sessions",                                         |
|       "name": "Weekly Learner",                                        |
|       "description": "Complete 5 sessions this week",                  |
|       "progress": 60,                                                  |
|       "current": 3,                                                    |
|       "target": 5,                                                     |
|       "status": "in_progress",                                         |
|       "ends_at": "2025-11-24T23:59:59Z",                                |
|       "icon": "calendar",                                              |
|       "color": "#4F46E5"                                               |
|     },                                                                 |
|     ...                                                                |
|   ],                                                                   |
|   "streak": {                                                          |
|     "current": 12,                                                     |
|     "best": 23,                                                        |
|     "today_complete": true                                             |
|   },                                                                   |
|   "recent_achievements": [                                             |
|     {                                                                  |
|       "id": "ach_xyz789",                                              |
|       "name": "Week Champion",                                         |
|       "description": "Completed weekly goal",                          |
|       "unlocked_at": "2025-11-17T23:59:00Z",                            |
|       "icon": "award"                                                  |
|     }                                                                  |
|   ],                                                                   |
|   "milestones": {                                                      |
|     "sessions": { "current": 87, "next": 100, "progress": 87 },        |
|     "topics": { "current": 34, "next": 50, "progress": 68 }            |
|   }                                                                    |
| }                                                                      |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| GET /topics                                                            |
+------------------------------------------------------------------------+
| Topic analytics from knowledge graph                                   |
|                                                                        |
| Query Params:                                                          |
|   - period: 30d, 90d, all                                              |
|   - limit: 20 (for top topics)                                         |
|                                                                        |
| Response:                                                              |
| {                                                                      |
|   "distribution": {                                                    |
|     "chart": {                                                         |
|       "labels": ["Machine Learning", "Python", "Databases", ...],      |
|       "datasets": [{                                                   |
|         "data": [45, 32, 28, ...],                                     |
|         "backgroundColor": ["#4F46E5", "#10B981", ...]                  |
|       }]                                                               |
|     }                                                                  |
|   },                                                                   |
|   "top_topics": [                                                      |
|     { "name": "Machine Learning", "sessions": 15, "quality": 0.82 },   |
|     { "name": "Python", "sessions": 12, "quality": 0.85 },             |
|     ...                                                                |
|   ],                                                                   |
|   "trending": [                                                        |
|     { "name": "LLMs", "growth": 150, "first_seen": "2025-11-10" }      |
|   ],                                                                   |
|   "knowledge_graph": {                                                 |
|     "total_concepts": 156,                                             |
|     "total_relationships": 234,                                        |
|     "strongest_connections": [...]                                     |
|   }                                                                    |
| }                                                                      |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| GET /export                                                            |
+------------------------------------------------------------------------+
| Export analytics data                                                  |
|                                                                        |
| Query Params:                                                          |
|   - format: json, csv                                                  |
|   - period: 7d, 30d, 90d, all                                          |
|   - include: sessions,quality,feedback,topics (comma-separated)        |
|                                                                        |
| Response (JSON):                                                       |
| {                                                                      |
|   "export": {                                                          |
|     "period": { "start": "...", "end": "..." },                        |
|     "sessions": [...],                                                 |
|     "quality_scores": [...],                                           |
|     "feedback": [...],                                                 |
|     "daily_summaries": [...]                                           |
|   },                                                                   |
|   "generated_at": "...",                                               |
|   "checksum": "sha256:..."                                             |
| }                                                                      |
|                                                                        |
| Response (CSV): Downloads as attachment                                |
+------------------------------------------------------------------------+

+------------------------------------------------------------------------+
| POST /goals                                                            |
+------------------------------------------------------------------------+
| Create custom goal                                                     |
|                                                                        |
| Request Body:                                                          |
| {                                                                      |
|   "name": "Master Machine Learning",                                   |
|   "description": "Complete 20 ML sessions with quality > 80%",         |
|   "target_metric": "session_count",                                    |
|   "target_value": 20,                                                  |
|   "frequency": "one_time",                                             |
|   "filters": { "topic": "Machine Learning" }                           |
| }                                                                      |
+------------------------------------------------------------------------+
```

### Response Caching Strategy

```python
CACHE_CONFIG = {
    '/api/analytics/overview': {
        'ttl_seconds': 300,       # 5 minutes
        'stale_while_revalidate': 60,
        'vary': ['period']
    },
    '/api/analytics/progress': {
        'ttl_seconds': 300,
        'stale_while_revalidate': 60,
        'vary': ['period', 'metrics', 'granularity']
    },
    '/api/analytics/trends': {
        'ttl_seconds': 600,       # 10 minutes
        'stale_while_revalidate': 120,
        'vary': ['period', 'metric', 'window']
    },
    '/api/analytics/insights': {
        'ttl_seconds': 1800,      # 30 minutes
        'stale_while_revalidate': 300,
        'vary': ['type', 'importance']
    },
    '/api/analytics/goals': {
        'ttl_seconds': 60,        # 1 minute (progress changes frequently)
        'stale_while_revalidate': 30,
        'vary': []
    },
    '/api/analytics/topics': {
        'ttl_seconds': 3600,      # 1 hour
        'stale_while_revalidate': 600,
        'vary': ['period']
    },
    '/api/analytics/export': {
        'ttl_seconds': 0,         # No caching for exports
        'vary': ['format', 'period', 'include']
    }
}
```

---

## Export and Reporting System

### Export Architecture

```
+===========================================================================+
|                    EXPORT SYSTEM                                           |
+===========================================================================+

                    Export Request
                         |
                         v
+---------------------+      +----------------------+
| Format: JSON        |      | Format: CSV          |
|                     |      |                      |
| Full fidelity       |      | Flat structure       |
| Nested objects      |      | Multiple files       |
| API consumption     |      | Spreadsheet ready    |
+----------+----------+      +----------+-----------+
           |                            |
           +------------+---------------+
                        |
                        v
           +------------------------+
           |   Export Generator     |
           |                        |
           | 1. Query raw data      |
           | 2. Query aggregations  |
           | 3. Format for output   |
           | 4. Generate checksum   |
           | 5. Stream response     |
           +------------------------+
```

### Export Data Structure

```python
@dataclass
class ExportData:
    """Complete export package"""

    # Metadata
    export_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    format: str

    # Summary
    summary: Dict[str, Any]

    # Raw session data (optional, for JSON)
    sessions: Optional[List[Dict[str, Any]]]

    # Aggregated data
    daily_summaries: List[Dict[str, Any]]
    weekly_summaries: List[Dict[str, Any]]

    # Quality data
    quality_scores: Optional[List[Dict[str, Any]]]
    quality_by_topic: Dict[str, float]

    # Feedback data
    feedback_summary: Dict[str, Any]

    # Topic data
    topic_distribution: Dict[str, int]
    knowledge_graph_stats: Dict[str, Any]

    # Goal and achievement data
    goal_progress: List[Dict[str, Any]]
    achievements: List[Dict[str, Any]]

    # Integrity
    checksum: str
    row_counts: Dict[str, int]
```

### CSV Export Format

```
Export Package (ZIP):
+-- learning_export_2025-11-21/
    +-- summary.csv           # High-level summary
    +-- sessions.csv          # One row per session
    +-- daily_metrics.csv     # One row per day
    +-- quality_scores.csv    # One row per exchange
    +-- topics.csv            # Topic distribution
    +-- goals.csv             # Goal progress
    +-- achievements.csv      # Unlocked achievements
    +-- README.txt            # Export metadata
```

### Export Generation

```python
class ExportGenerator:
    """
    PATTERN: Streaming export with chunked processing
    WHY: Handle large datasets without memory issues
    """

    MAX_EXPORT_DAYS = 365
    CHUNK_SIZE = 1000

    async def generate_export(
        self,
        format: str,
        period_days: int,
        include: List[str]
    ) -> Union[Dict, bytes]:
        """
        Generate export in requested format

        ALGORITHM:
        1. Validate request parameters
        2. Query data in chunks
        3. Transform to output format
        4. Generate checksum
        5. Return or stream response
        """
        # Validate
        if period_days > self.MAX_EXPORT_DAYS:
            raise ValueError(f"Max export period is {self.MAX_EXPORT_DAYS} days")

        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=period_days)

        # Collect data based on includes
        export_data = ExportData(
            export_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            format=format,
            summary={},
            sessions=None,
            daily_summaries=[],
            weekly_summaries=[],
            quality_scores=None,
            quality_by_topic={},
            feedback_summary={},
            topic_distribution={},
            knowledge_graph_stats={},
            goal_progress=[],
            achievements=[],
            checksum='',
            row_counts={}
        )

        if 'sessions' in include:
            export_data.sessions = await self._export_sessions(period_start, period_end)
            export_data.row_counts['sessions'] = len(export_data.sessions)

        if 'quality' in include:
            export_data.quality_scores = await self._export_quality(period_start, period_end)
            export_data.row_counts['quality'] = len(export_data.quality_scores)

        # Always include summaries
        export_data.daily_summaries = await self._export_daily_summaries(period_start, period_end)
        export_data.summary = self._calculate_summary(export_data)

        # Generate checksum
        export_data.checksum = self._calculate_checksum(export_data)

        if format == 'json':
            return asdict(export_data)
        elif format == 'csv':
            return self._generate_csv_zip(export_data)

    def _calculate_checksum(self, data: ExportData) -> str:
        """Generate SHA-256 checksum of export data"""
        import hashlib
        content = json.dumps(asdict(data), sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
```

---

## Caching Strategy

### Cache Architecture

```
+===========================================================================+
|                    CACHING ARCHITECTURE                                    |
+===========================================================================+

                      API Request
                           |
                           v
                  +------------------+
                  |  Cache Layer     |
                  |  (In-Memory)     |
                  +--------+---------+
                           |
           +---------------+---------------+
           |                               |
           v                               v
    +-------------+                 +-------------+
    | Cache HIT   |                 | Cache MISS  |
    |             |                 |             |
    | Return      |                 | Compute     |
    | cached data |                 | Store       |
    |             |                 | Return      |
    +-------------+                 +-------------+

CACHE LAYERS:
+------------------------------------------------------------------------+
| Layer 1: Response Cache (TTL: 1-30 min based on endpoint)              |
|   - Full API responses                                                 |
|   - Key: endpoint + query params hash                                  |
+------------------------------------------------------------------------+
| Layer 2: Aggregation Cache (TTL: 1 hour)                               |
|   - Pre-computed aggregations                                          |
|   - Key: aggregation_type + period                                     |
+------------------------------------------------------------------------+
| Layer 3: Query Cache (TTL: 5 min)                                      |
|   - Database query results                                             |
|   - Key: query hash                                                    |
+------------------------------------------------------------------------+
```

### Cache Implementation

```python
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Optional, Any, Callable
import hashlib
import json

class AnalyticsCache:
    """
    PATTERN: In-memory cache with TTL and LRU eviction
    WHY: Single-user system doesn't need distributed cache
    """

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key not in self._cache:
            return None

        value, expires_at = self._cache[key]
        if datetime.utcnow() > expires_at:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Set cached value with TTL"""
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        self._cache[key] = (value, expires_at)

    def invalidate(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        keys_to_delete = [k for k in self._cache if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
        return len(keys_to_delete)

    @staticmethod
    def make_key(endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and params"""
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{endpoint}:{param_hash}"


def cached_endpoint(ttl_seconds: int = 300):
    """
    Decorator for caching API endpoint responses

    Usage:
        @cached_endpoint(ttl_seconds=300)
        async def get_overview(period: str):
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            cache = get_analytics_cache()
            key = AnalyticsCache.make_key(func.__name__, kwargs)

            # Try cache
            cached = cache.get(key)
            if cached is not None:
                return {**cached, 'meta': {**cached.get('meta', {}), 'cached': True}}

            # Compute
            result = await func(*args, **kwargs)

            # Store
            cache.set(key, result, ttl_seconds)

            return {**result, 'meta': {**result.get('meta', {}), 'cached': False}}

        return wrapper
    return decorator


# Global cache instance
_analytics_cache: Optional[AnalyticsCache] = None

def get_analytics_cache() -> AnalyticsCache:
    global _analytics_cache
    if _analytics_cache is None:
        _analytics_cache = AnalyticsCache(max_size=100)
    return _analytics_cache
```

### Cache Invalidation Strategy

```python
CACHE_INVALIDATION_TRIGGERS = {
    # When new exchange is saved
    'exchange_created': [
        'overview:*',
        'progress:*',
        'goals:*'
    ],

    # When feedback is submitted
    'feedback_created': [
        'overview:*',
        'insights:*'
    ],

    # When aggregation runs
    'aggregation_complete': [
        'overview:*',
        'progress:*',
        'trends:*',
        'topics:*'
    ],

    # When goal is updated
    'goal_updated': [
        'goals:*'
    ],

    # When insight is generated
    'insights_generated': [
        'insights:*',
        'overview:*'  # Overview includes recent insights
    ]
}

async def invalidate_on_event(event: str) -> None:
    """Invalidate cache entries based on event"""
    cache = get_analytics_cache()
    patterns = CACHE_INVALIDATION_TRIGGERS.get(event, [])
    for pattern in patterns:
        cache.invalidate(pattern)
```

---

## Integration with Existing Phases

### Phase 1 Integration (Observability)

```python
# Add analytics-specific Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Dashboard API metrics
dashboard_requests_total = Counter(
    'analytics_dashboard_requests_total',
    'Total dashboard API requests',
    ['endpoint', 'status'],
    registry=metrics_registry
)

dashboard_latency_seconds = Histogram(
    'analytics_dashboard_latency_seconds',
    'Dashboard API response latency',
    ['endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=metrics_registry
)

cache_hit_rate = Gauge(
    'analytics_cache_hit_rate',
    'Dashboard cache hit rate',
    registry=metrics_registry
)

# Aggregation metrics
aggregation_duration_seconds = Histogram(
    'analytics_aggregation_duration_seconds',
    'Time to complete aggregation',
    ['interval'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=metrics_registry
)
```

### Phase 3 Integration (Knowledge Graph)

```python
class KnowledgeGraphAnalytics:
    """
    Extract analytics from Neo4j knowledge graph

    INTEGRATION POINTS:
    - Topic distribution from concept frequencies
    - Knowledge gaps from missing prerequisites
    - Learning paths from relationship traversal
    """

    async def get_topic_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get topic analytics from knowledge graph

        QUERIES:
        1. Concept frequency distribution
        2. Trending concepts (recent growth)
        3. Knowledge coverage (explored vs total)
        4. Learning path suggestions
        """
        from app.knowledge_graph import graph_store, query_engine

        # Get concept frequencies
        concepts = await graph_store.get_all_concepts(limit=100)

        # Get recent sessions to identify new concepts
        recent_concepts = await graph_store.get_concepts_since(
            datetime.utcnow() - timedelta(days=days)
        )

        # Get knowledge gaps
        known_concepts = [c['name'] for c in concepts if c['frequency'] >= 3]
        gaps = []
        for concept in recent_concepts:
            concept_gaps = await query_engine.identify_knowledge_gaps(
                known_concepts=known_concepts,
                target_concept=concept['name']
            )
            gaps.extend(concept_gaps.get('gaps', []))

        return {
            'total_concepts': len(concepts),
            'concept_distribution': {c['name']: c['frequency'] for c in concepts[:20]},
            'trending': [c for c in recent_concepts if c['frequency'] > 2][:10],
            'knowledge_gaps': gaps[:5],
            'coverage_percent': len(known_concepts) / max(len(concepts), 1) * 100
        }
```

### Phase 5 Integration (Learning System)

```python
class LearningSystemIntegration:
    """
    Bridge between Phase 5 learning system and Phase 6 analytics

    DATA FLOWS:
    - Feedback data -> Feedback analytics
    - Quality scores -> Quality trends
    - Patterns -> Insight generation
    - Preferences -> Personalized goals
    """

    def __init__(self):
        self.feedback_store = feedback_store
        self.quality_store = quality_store
        self.pattern_store = pattern_store

    async def sync_metrics(self) -> Dict[str, int]:
        """
        Sync Phase 5 data to analytics aggregations

        RUNS: As part of hourly aggregation
        """
        synced = {'feedback': 0, 'quality': 0, 'patterns': 0}

        # Get unsynced feedback
        new_feedback = await self.feedback_store.get_unsynced(limit=1000)
        for fb in new_feedback:
            await self._process_feedback_for_analytics(fb)
            synced['feedback'] += 1

        # Get unsynced quality scores
        new_quality = await self.quality_store.get_unsynced(limit=1000)
        for qs in new_quality:
            await self._process_quality_for_analytics(qs)
            synced['quality'] += 1

        # Sync patterns for insight generation
        patterns = await self.pattern_store.get_active_patterns()
        synced['patterns'] = len(patterns)

        return synced

    async def _process_feedback_for_analytics(self, feedback: FeedbackData) -> None:
        """Transform feedback for analytics storage"""
        # Already stored in Phase 5, just mark as synced
        await self.feedback_store.mark_synced(feedback.id)

    async def _process_quality_for_analytics(self, quality: QualityScore) -> None:
        """Transform quality score for analytics"""
        # Already stored in Phase 5, just mark as synced
        await self.quality_store.mark_synced(quality.id)
```

---

## Performance Targets

### Latency Budgets

```
+===========================================================================+
|                    LATENCY BUDGETS                                         |
+===========================================================================+

DASHBOARD API ENDPOINTS:
+------------------------------------------------------------------------+
| Endpoint           | Cached (P95) | Uncached (P95) | Max Acceptable    |
+------------------------------------------------------------------------+
| GET /overview      |     25ms     |     200ms      |      500ms        |
| GET /progress      |     30ms     |     250ms      |      500ms        |
| GET /trends        |     30ms     |     300ms      |      750ms        |
| GET /insights      |     20ms     |     400ms      |     1000ms        |
| GET /goals         |     15ms     |     100ms      |      250ms        |
| GET /topics        |     40ms     |     500ms      |     1000ms        |
| GET /export        |      N/A     |    2000ms      |     5000ms        |
+------------------------------------------------------------------------+

BACKGROUND JOBS:
+------------------------------------------------------------------------+
| Job                | Target Duration | Max Duration  | Frequency       |
+------------------------------------------------------------------------+
| Hourly aggregation |     5 seconds   |   30 seconds  | Every hour      |
| Daily aggregation  |    15 seconds   |   60 seconds  | Daily 00:15     |
| Insight generation |    10 seconds   |   60 seconds  | Daily 01:00     |
| Goal evaluation    |     2 seconds   |   10 seconds  | Every 5 min     |
| Cache warmup       |     5 seconds   |   30 seconds  | On deploy       |
+------------------------------------------------------------------------+

QUERY BUDGETS:
+------------------------------------------------------------------------+
| Query Type                     | Target     | Max        | Index Req   |
+------------------------------------------------------------------------+
| Count sessions in period       |    5ms     |   50ms     | Yes         |
| Aggregate quality scores       |   10ms     |  100ms     | Yes         |
| Get daily summaries (30 days)  |   15ms     |  150ms     | Yes         |
| Topic distribution             |   20ms     |  200ms     | Yes         |
| Knowledge graph traversal      |   50ms     |  500ms     | Neo4j       |
+------------------------------------------------------------------------+
```

### Resource Limits

```python
ANALYTICS_RESOURCE_LIMITS = {
    # Memory
    'max_cache_entries': 100,
    'max_cache_mb': 50,
    'max_export_rows': 10000,

    # Processing
    'max_aggregation_period_days': 365,
    'max_insight_generation_time_sec': 60,
    'max_concurrent_exports': 2,

    # Database
    'max_query_time_sec': 5,
    'aggregation_batch_size': 1000,

    # Background jobs
    'job_timeout_sec': 120,
    'job_retry_count': 3,
    'job_retry_delay_sec': 30
}
```

---

## Database Schema

### Analytics Tables

```sql
-- ============================================================================
-- PHASE 6: Analytics Tables
-- Added to existing learning_captures.db
-- ============================================================================

-- Pre-aggregated metrics at various intervals
CREATE TABLE IF NOT EXISTS aggregated_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date DATE NOT NULL,
    interval_type TEXT NOT NULL,  -- hourly, daily, weekly, monthly
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metadata TEXT,  -- JSON for additional context
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(metric_date, interval_type, metric_name)
);

CREATE INDEX idx_agg_metrics_date ON aggregated_metrics(metric_date);
CREATE INDEX idx_agg_metrics_interval ON aggregated_metrics(interval_type);
CREATE INDEX idx_agg_metrics_name ON aggregated_metrics(metric_name);

-- Generated insights
CREATE TABLE IF NOT EXISTS analytics_insights (
    id TEXT PRIMARY KEY,
    insight_type TEXT NOT NULL,  -- trend, anomaly, milestone, recommendation, knowledge
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    importance TEXT NOT NULL,  -- high, medium, low
    data TEXT,  -- JSON supporting data
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    viewed_at DATETIME,
    dismissed_at DATETIME
);

CREATE INDEX idx_insights_type ON analytics_insights(insight_type);
CREATE INDEX idx_insights_importance ON analytics_insights(importance);
CREATE INDEX idx_insights_generated ON analytics_insights(generated_at);

-- Goal definitions
CREATE TABLE IF NOT EXISTS goal_definitions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    goal_type TEXT NOT NULL,  -- activity, quality, topic, streak, milestone, custom
    frequency TEXT NOT NULL,  -- daily, weekly, monthly, one_time
    target_metric TEXT NOT NULL,
    target_value REAL NOT NULL,
    comparison TEXT NOT NULL,  -- gte, lte, eq
    icon TEXT,
    color TEXT,
    enabled INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_goals_type ON goal_definitions(goal_type);
CREATE INDEX idx_goals_enabled ON goal_definitions(enabled);

-- Goal progress snapshots
CREATE TABLE IF NOT EXISTS goal_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL,
    current_value REAL NOT NULL,
    target_value REAL NOT NULL,
    progress_percent REAL NOT NULL,
    status TEXT NOT NULL,  -- in_progress, completed, failed
    period_start DATETIME NOT NULL,
    period_end DATETIME NOT NULL,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (goal_id) REFERENCES goal_definitions(id)
);

CREATE INDEX idx_progress_goal ON goal_progress(goal_id);
CREATE INDEX idx_progress_period ON goal_progress(period_start, period_end);

-- Achievements (unlocked goals)
CREATE TABLE IF NOT EXISTS achievements (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    icon TEXT,
    period TEXT,  -- "Week of Nov 18, 2025"
    unlocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (goal_id) REFERENCES goal_definitions(id)
);

CREATE INDEX idx_achievements_goal ON achievements(goal_id);
CREATE INDEX idx_achievements_unlocked ON achievements(unlocked_at);

-- Streak tracking
CREATE TABLE IF NOT EXISTS streak_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streak_type TEXT NOT NULL,  -- daily_sessions, daily_quality, etc.
    current_streak INTEGER DEFAULT 0,
    best_streak INTEGER DEFAULT 0,
    last_active_date DATE,
    streak_start_date DATE,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(streak_type)
);

-- Export history
CREATE TABLE IF NOT EXISTS export_history (
    id TEXT PRIMARY KEY,
    format TEXT NOT NULL,
    period_start DATETIME NOT NULL,
    period_end DATETIME NOT NULL,
    include_sections TEXT,  -- JSON array
    file_size_bytes INTEGER,
    checksum TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_exports_created ON export_history(created_at);
```

---

## Implementation Plan

### Phase 6.1: Foundation (Week 1)

**Objectives:**
- Create analytics database schema
- Implement time series aggregation
- Build core API infrastructure

**Tasks:**
1. Create database migration for analytics tables
2. Implement AggregatedMetrics dataclass and storage
3. Build TimeSeriesAggregator with hourly/daily/weekly/monthly
4. Create base API router with caching decorator
5. Implement `/api/analytics/overview` endpoint
6. Add scheduled job infrastructure (APScheduler)
7. Write unit tests for aggregation

**Deliverables:**
- `/app/analytics/__init__.py`
- `/app/analytics/aggregator.py`
- `/app/analytics/cache.py`
- `/app/analytics/api.py`
- `/app/analytics/scheduler.py`
- `/tests/analytics/test_aggregator.py`

**Validation:**
- [ ] Hourly aggregation completes in < 5s
- [ ] API response time < 500ms (uncached)
- [ ] All tests passing

---

### Phase 6.2: Dashboard APIs (Week 2)

**Objectives:**
- Complete all dashboard API endpoints
- Implement Chart.js data formatting
- Add trend calculation

**Tasks:**
1. Implement `/api/analytics/progress` endpoint
2. Implement `/api/analytics/trends` endpoint
3. Implement `/api/analytics/topics` endpoint
4. Build TrendCalculator with rolling averages
5. Create Chart.js data formatter utilities
6. Add integration tests for all endpoints

**Deliverables:**
- `/app/analytics/endpoints/progress.py`
- `/app/analytics/endpoints/trends.py`
- `/app/analytics/endpoints/topics.py`
- `/app/analytics/formatters.py`
- `/app/analytics/trends.py`
- `/tests/analytics/test_endpoints.py`

**Validation:**
- [ ] All endpoints return Chart.js-ready data
- [ ] Trend detection working correctly
- [ ] Response times within targets

---

### Phase 6.3: Insights & Goals (Week 3)

**Objectives:**
- Build insight generation engine
- Implement goal tracking system
- Add achievements and streaks

**Tasks:**
1. Implement InsightGenerator with rule-based detection
2. Build GoalTracker with progress calculation
3. Implement streak tracking logic
4. Create `/api/analytics/insights` endpoint
5. Create `/api/analytics/goals` endpoint
6. Add default goal definitions
7. Integrate with Phase 5 learning patterns

**Deliverables:**
- `/app/analytics/insights.py`
- `/app/analytics/goals.py`
- `/app/analytics/streaks.py`
- `/app/analytics/endpoints/insights.py`
- `/app/analytics/endpoints/goals.py`
- `/tests/analytics/test_insights.py`
- `/tests/analytics/test_goals.py`

**Validation:**
- [ ] Insights generated daily
- [ ] Goal progress updating correctly
- [ ] Streaks tracked accurately

---

### Phase 6.4: Export & Integration (Week 4)

**Objectives:**
- Build export system
- Complete Phase 3/5 integration
- Performance optimization

**Tasks:**
1. Implement ExportGenerator for JSON/CSV
2. Add `/api/analytics/export` endpoint
3. Integrate with knowledge graph for topic analytics
4. Integrate with Phase 5 feedback/quality stores
5. Add Prometheus metrics for analytics
6. Performance optimization and load testing
7. Documentation and examples

**Deliverables:**
- `/app/analytics/export.py`
- `/app/analytics/endpoints/export.py`
- `/app/analytics/integrations/knowledge_graph.py`
- `/app/analytics/integrations/learning_system.py`
- `/docs/PHASE6_API_REFERENCE.md`
- `/docs/PHASE6_USAGE_GUIDE.md`

**Validation:**
- [ ] JSON export working
- [ ] CSV export working
- [ ] All performance targets met
- [ ] Documentation complete

---

## File Structure

```
learning_voice_agent/
+-- app/
|   +-- analytics/                      # NEW: Analytics engine
|   |   +-- __init__.py                # Public API exports
|   |   +-- config.py                  # Analytics configuration
|   |   +-- aggregator.py              # Time series aggregation
|   |   +-- trends.py                  # Trend calculation
|   |   +-- insights.py                # Insight generation
|   |   +-- goals.py                   # Goal tracking
|   |   +-- streaks.py                 # Streak tracking
|   |   +-- cache.py                   # Caching layer
|   |   +-- scheduler.py               # Background job scheduling
|   |   +-- export.py                  # Export generation
|   |   +-- formatters.py              # Chart.js data formatters
|   |   +-- api.py                     # API router
|   |   +-- endpoints/
|   |   |   +-- overview.py
|   |   |   +-- progress.py
|   |   |   +-- trends.py
|   |   |   +-- insights.py
|   |   |   +-- goals.py
|   |   |   +-- topics.py
|   |   |   +-- export.py
|   |   +-- integrations/
|   |       +-- knowledge_graph.py     # Phase 3 integration
|   |       +-- learning_system.py     # Phase 5 integration
|   |       +-- prometheus.py          # Phase 1 metrics
|   |
|   +-- main.py                        # Add analytics router
|
+-- tests/
|   +-- analytics/                     # NEW: Analytics tests
|   |   +-- test_aggregator.py
|   |   +-- test_trends.py
|   |   +-- test_insights.py
|   |   +-- test_goals.py
|   |   +-- test_endpoints.py
|   |   +-- test_export.py
|   |   +-- test_integration.py
|
+-- docs/
    +-- PHASE6_ANALYTICS_ARCHITECTURE.md  # This document
    +-- PHASE6_API_REFERENCE.md           # API documentation
    +-- PHASE6_USAGE_GUIDE.md             # Usage examples
```

---

## Success Metrics

### Code Quality

- [ ] Type hints on all functions
- [ ] Docstrings on all classes/methods
- [ ] Comprehensive error handling
- [ ] Structured logging throughout
- [ ] Configuration-driven design

### Test Coverage

- [ ] Unit tests for all components
- [ ] Integration tests for API endpoints
- [ ] Performance regression tests
- [ ] Export format validation tests

### Documentation

- [ ] Architecture document (this file)
- [ ] API reference with examples
- [ ] Usage guide for dashboard integration

### Performance

- [ ] Cached API response: < 50ms P95
- [ ] Uncached API response: < 500ms P95
- [ ] Aggregation jobs: < 30s
- [ ] Export generation: < 5s

### Functionality

- [ ] All 7 API endpoints operational
- [ ] Hourly/daily/weekly aggregation running
- [ ] Insights generating correctly
- [ ] Goals tracking and updating
- [ ] JSON and CSV export working

---

## Conclusion

This architecture provides a comprehensive analytics engine that:

1. **Aggregates** learning data efficiently using pre-computed time series
2. **Generates** actionable insights through rule-based analysis
3. **Tracks** goals and achievements to motivate continued learning
4. **Exposes** dashboard-ready data through REST APIs
5. **Exports** data for external analysis and record-keeping

**Key Design Philosophy:**

- **Pre-compute over real-time**: Background aggregation keeps APIs fast
- **Cache aggressively**: Single-user system benefits from generous caching
- **Chart.js native**: Data formatted for immediate visualization
- **SQLite sufficient**: No additional databases needed
- **Graceful integration**: Works with existing Phase 3/5 components

The implementation plan spans 4 weeks with clear deliverables and validation criteria for each phase. The system is designed to be additive - basic analytics work immediately, with advanced features building over time.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Status:** Architecture Design Complete - Ready for Implementation
