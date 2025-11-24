# Phase 6: Dashboard Implementation Guide

## Overview

This guide covers the dashboard data structures, Chart.js integration patterns, frontend consumption, caching strategies, and mobile considerations for the Phase 6 analytics engine.

---

## Dashboard Data Structures

### DashboardData Model

The primary dashboard response contains all data needed for frontend rendering:

```python
class DashboardData(BaseModel):
    # Identification
    id: str
    user_id: Optional[str]

    # Overview metrics
    overview: ProgressMetrics

    # Streak information
    streak: LearningStreak

    # Recent progress (last 30 days by default)
    daily_progress: List[DailyProgress]
    weekly_summary: Optional[WeeklyProgress]

    # Goals and achievements
    active_goals: List[LearningGoal]
    recent_achievements: List[UnlockedAchievement]

    # Topic mastery
    topic_mastery: List[TopicMastery]

    # AI-generated insights
    insights: List[Dict[str, Any]]

    # Pre-computed chart data
    quality_chart_data: List[Dict[str, Any]]
    activity_heatmap_data: List[Dict[str, Any]]
    topic_distribution_data: List[Dict[str, Any]]

    # Metadata
    generated_at: datetime
    cache_ttl_seconds: int
```

### Overview Metrics Structure

```json
{
  "total_sessions": 50,
  "total_exchanges": 500,
  "total_time_hours": 12.5,
  "avg_quality": 82.3,
  "current_streak": 7,
  "longest_streak": 14,
  "topics_explored": 15,
  "topics_mastered": 5,
  "goals_completed": 3,
  "achievements": 8
}
```

### Quality Chart Data Structure

```json
[
  {
    "date": "2024-11-01",
    "quality": 78.5,
    "sessions": 2,
    "exchanges": 25,
    "time_minutes": 35.5
  },
  {
    "date": "2024-11-02",
    "quality": 82.1,
    "sessions": 3,
    "exchanges": 32,
    "time_minutes": 45.0
  }
]
```

### Activity Heatmap Data Structure

```json
[
  {
    "date": "2024-10-28",
    "weekday": 0,
    "week": 0,
    "activity": 15,
    "intensity": 2
  },
  {
    "date": "2024-10-29",
    "weekday": 1,
    "week": 0,
    "activity": 28,
    "intensity": 3
  }
]
```

Intensity levels:
- 0: No activity
- 1: Low (1-5 exchanges)
- 2: Medium (6-15 exchanges)
- 3: High (16-30 exchanges)
- 4: Very High (31+ exchanges)

### Topic Distribution Data Structure

```json
[
  {
    "topic": "Python",
    "interactions": 150,
    "percentage": 30.0,
    "mastery_score": 85.0,
    "level": "advanced"
  },
  {
    "topic": "Machine Learning",
    "interactions": 100,
    "percentage": 20.0,
    "mastery_score": 72.0,
    "level": "intermediate"
  }
]
```

---

## Chart.js Integration Guide

### Quality Score Line Chart

```javascript
// Quality trend chart configuration
const qualityChartConfig = {
  type: 'line',
  data: {
    labels: chartData.map(d => d.date),
    datasets: [
      {
        label: 'Quality Score',
        data: chartData.map(d => d.quality),
        borderColor: '#4F46E5',
        backgroundColor: 'rgba(79, 70, 229, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: '7-Day Average',
        data: chartData.map(d => d.rolling_avg),
        borderColor: '#10B981',
        borderDash: [5, 5],
        fill: false,
        tension: 0.4
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        min: 0,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%';
          }
        }
      },
      x: {
        grid: {
          display: false
        }
      }
    }
  }
};

const qualityChart = new Chart(
  document.getElementById('qualityChart'),
  qualityChartConfig
);
```

### Progress Bar Chart

```javascript
// Sessions/Exchanges progress chart
const progressChartConfig = {
  type: 'bar',
  data: {
    labels: progressData.map(d => d.date),
    datasets: [
      {
        label: 'Sessions',
        data: progressData.map(d => d.sessions),
        backgroundColor: '#6366F1',
        borderRadius: 4,
        yAxisID: 'y'
      },
      {
        label: 'Exchanges',
        data: progressData.map(d => d.exchanges),
        backgroundColor: '#10B981',
        borderRadius: 4,
        yAxisID: 'y1'
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top'
      }
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Sessions'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Exchanges'
        },
        grid: {
          drawOnChartArea: false
        }
      }
    }
  }
};
```

### Topic Distribution Pie Chart

```javascript
// Topic distribution doughnut chart
const topicChartConfig = {
  type: 'doughnut',
  data: {
    labels: topicData.map(d => d.topic),
    datasets: [{
      data: topicData.map(d => d.percentage),
      backgroundColor: [
        '#6366F1', '#10B981', '#F59E0B', '#EF4444',
        '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'
      ],
      borderWidth: 0
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          usePointStyle: true,
          padding: 15
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const data = topicData[context.dataIndex];
            return `${data.topic}: ${data.percentage}% (${data.level})`;
          }
        }
      }
    }
  }
};
```

### Activity Heatmap

```javascript
// Custom heatmap using HTML/CSS grid
function renderHeatmap(heatmapData, containerId) {
  const container = document.getElementById(containerId);
  const weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  // Create grid
  const grid = document.createElement('div');
  grid.className = 'heatmap-grid';
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = `auto repeat(${getWeekCount(heatmapData)}, 1fr)`;
  grid.style.gap = '2px';

  // Add weekday labels
  weekdays.forEach(day => {
    const label = document.createElement('div');
    label.className = 'heatmap-label';
    label.textContent = day;
    grid.appendChild(label);
  });

  // Add cells
  heatmapData.forEach(d => {
    const cell = document.createElement('div');
    cell.className = `heatmap-cell intensity-${d.intensity}`;
    cell.title = `${d.date}: ${d.activity} exchanges`;
    cell.style.gridRow = d.weekday + 2;
    cell.style.gridColumn = d.week + 2;
    grid.appendChild(cell);
  });

  container.appendChild(grid);
}

// CSS for heatmap
const heatmapStyles = `
  .heatmap-cell {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }
  .intensity-0 { background-color: #ebedf0; }
  .intensity-1 { background-color: #9be9a8; }
  .intensity-2 { background-color: #40c463; }
  .intensity-3 { background-color: #30a14e; }
  .intensity-4 { background-color: #216e39; }
`;
```

### Goal Progress Radial Chart

```javascript
// Radial progress for goals
function renderGoalProgress(goal, containerId) {
  const config = {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [goal.progress, 100 - goal.progress],
        backgroundColor: ['#6366F1', '#E5E7EB'],
        borderWidth: 0
      }]
    },
    options: {
      cutout: '75%',
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      }
    },
    plugins: [{
      id: 'centerText',
      afterDraw: function(chart) {
        const ctx = chart.ctx;
        const centerX = (chart.chartArea.left + chart.chartArea.right) / 2;
        const centerY = (chart.chartArea.top + chart.chartArea.bottom) / 2;

        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = 'bold 24px Inter';
        ctx.fillStyle = '#1F2937';
        ctx.fillText(`${goal.progress.toFixed(0)}%`, centerX, centerY);
      }
    }]
  };

  return new Chart(document.getElementById(containerId), config);
}
```

---

## Frontend Consumption Patterns

### React Integration

```typescript
// hooks/useDashboard.ts
import { useState, useEffect, useCallback } from 'react';

interface DashboardData {
  overview: OverviewData;
  streak: StreakData;
  dailyProgress: DailyProgress[];
  activeGoals: Goal[];
  recentAchievements: Achievement[];
  insights: Insight[];
  qualityChartData: ChartDataPoint[];
  activityHeatmapData: HeatmapDataPoint[];
  topicDistributionData: TopicData[];
}

export function useDashboard(userId?: string) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchDashboard = useCallback(async (useCache = true) => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (userId) params.append('user_id', userId);
      if (!useCache) params.append('use_cache', 'false');

      const response = await fetch(`/api/analytics/dashboard?${params}`);
      if (!response.ok) throw new Error('Failed to fetch dashboard');

      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchDashboard();
  }, [fetchDashboard]);

  const refresh = useCallback(() => fetchDashboard(false), [fetchDashboard]);

  return { data, loading, error, refresh };
}
```

```tsx
// components/Dashboard.tsx
import React from 'react';
import { useDashboard } from '../hooks/useDashboard';
import { QualityChart } from './charts/QualityChart';
import { ActivityHeatmap } from './charts/ActivityHeatmap';
import { GoalProgress } from './GoalProgress';
import { AchievementBadge } from './AchievementBadge';
import { InsightCard } from './InsightCard';

export function Dashboard({ userId }: { userId?: string }) {
  const { data, loading, error, refresh } = useDashboard(userId);

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  return (
    <div className="dashboard-container">
      {/* Overview Stats */}
      <section className="stats-grid">
        <StatCard
          title="Sessions"
          value={data.overview.total_sessions}
          icon="calendar"
        />
        <StatCard
          title="Exchanges"
          value={data.overview.total_exchanges}
          icon="chat"
        />
        <StatCard
          title="Quality"
          value={`${data.overview.avg_quality.toFixed(1)}%`}
          icon="star"
        />
        <StatCard
          title="Streak"
          value={`${data.streak.current_streak} days`}
          icon="fire"
          highlight={data.streak.current_streak >= 7}
        />
      </section>

      {/* Charts */}
      <section className="charts-section">
        <div className="chart-container">
          <h3>Quality Trend</h3>
          <QualityChart data={data.qualityChartData} />
        </div>
        <div className="chart-container">
          <h3>Activity</h3>
          <ActivityHeatmap data={data.activityHeatmapData} />
        </div>
      </section>

      {/* Goals and Achievements */}
      <section className="goals-achievements">
        <div className="goals-section">
          <h3>Active Goals</h3>
          {data.activeGoals.map(goal => (
            <GoalProgress key={goal.id} goal={goal} />
          ))}
        </div>
        <div className="achievements-section">
          <h3>Recent Achievements</h3>
          {data.recentAchievements.map(achievement => (
            <AchievementBadge key={achievement.id} achievement={achievement} />
          ))}
        </div>
      </section>

      {/* Insights */}
      <section className="insights-section">
        <h3>Insights</h3>
        {data.insights.map(insight => (
          <InsightCard key={insight.id} insight={insight} />
        ))}
      </section>
    </div>
  );
}
```

### Vue.js Integration

```vue
<!-- Dashboard.vue -->
<template>
  <div class="dashboard" v-if="!loading">
    <!-- Stats Overview -->
    <div class="stats-grid">
      <StatCard
        v-for="stat in statsCards"
        :key="stat.key"
        :title="stat.title"
        :value="stat.value"
        :icon="stat.icon"
      />
    </div>

    <!-- Quality Chart -->
    <div class="chart-section">
      <LineChart :data="qualityChartData" :options="chartOptions" />
    </div>

    <!-- Goals -->
    <GoalList :goals="dashboard.activeGoals" @refresh="refresh" />
  </div>
  <LoadingState v-else />
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useDashboardApi } from '@/composables/useDashboardApi';

const { dashboard, loading, fetchDashboard, refresh } = useDashboardApi();

const statsCards = computed(() => [
  { key: 'sessions', title: 'Sessions', value: dashboard.value?.overview.total_sessions, icon: 'calendar' },
  { key: 'exchanges', title: 'Exchanges', value: dashboard.value?.overview.total_exchanges, icon: 'chat' },
  { key: 'quality', title: 'Quality', value: `${dashboard.value?.overview.avg_quality.toFixed(1)}%`, icon: 'star' },
  { key: 'streak', title: 'Streak', value: `${dashboard.value?.streak.current_streak} days`, icon: 'fire' },
]);

onMounted(() => fetchDashboard());
</script>
```

---

## Caching Strategies

### Server-Side Caching

```python
# Cache configuration
class DashboardCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self.ttl_seconds = 300  # 5 minutes
        self.stale_ttl_seconds = 600  # 10 minutes for stale-while-revalidate

    def get(self, key: str, allow_stale: bool = False) -> Optional[Any]:
        if key not in self._cache:
            return None

        cached_time, data = self._cache[key]
        age = (datetime.utcnow() - cached_time).total_seconds()

        if age < self.ttl_seconds:
            return data
        elif allow_stale and age < self.stale_ttl_seconds:
            return data  # Return stale data, trigger background refresh

        return None

    def set(self, key: str, data: Any) -> None:
        self._cache[key] = (datetime.utcnow(), data)

    def invalidate(self, key: str = None) -> None:
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()
```

### Client-Side Caching

```typescript
// React Query / SWR pattern
import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export function useDashboard(userId?: string) {
  const { data, error, mutate } = useSWR(
    userId ? `/api/analytics/dashboard?user_id=${userId}` : '/api/analytics/dashboard',
    fetcher,
    {
      refreshInterval: 60000,  // Refresh every minute
      revalidateOnFocus: true,
      dedupingInterval: 5000,  // Dedupe requests within 5 seconds
      keepPreviousData: true,  // Keep showing previous data while revalidating
    }
  );

  return {
    data,
    loading: !error && !data,
    error,
    refresh: () => mutate()
  };
}
```

---

## Real-Time Updates

### WebSocket Integration

```typescript
// WebSocket for real-time dashboard updates
class DashboardSocket {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<Function>> = new Map();

  connect(userId: string) {
    this.ws = new WebSocket(`wss://api.example.com/ws/dashboard?user_id=${userId}`);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.emit(data.type, data.payload);
    };
  }

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  private emit(event: string, data: any) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(cb => cb(data));
    }
  }
}

// Usage
const socket = new DashboardSocket();
socket.connect('user-123');

socket.on('streak_update', (streak) => {
  // Update streak display
});

socket.on('achievement_unlocked', (achievement) => {
  // Show celebration animation
});

socket.on('goal_completed', (goal) => {
  // Update goal display
});
```

---

## Mobile Considerations

### Responsive Layout

```css
/* Mobile-first responsive dashboard */
.dashboard-container {
  padding: 1rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
}

.chart-container {
  width: 100%;
  height: 200px;
  margin-bottom: 1rem;
}

.goals-achievements {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* Tablet and above */
@media (min-width: 768px) {
  .stats-grid {
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
  }

  .chart-container {
    height: 300px;
  }

  .goals-achievements {
    flex-direction: row;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .dashboard-container {
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }

  .charts-section {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 1.5rem;
  }
}
```

### Touch-Friendly Charts

```javascript
// Chart.js mobile optimizations
const mobileChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'nearest',
    axis: 'x',
    intersect: false
  },
  plugins: {
    tooltip: {
      enabled: true,
      position: 'nearest',
      // Larger touch targets
      padding: 12,
      bodyFont: { size: 14 }
    },
    legend: {
      labels: {
        // Larger legend for touch
        boxWidth: 16,
        padding: 16,
        font: { size: 12 }
      }
    }
  },
  // Larger touch targets for data points
  elements: {
    point: {
      radius: 6,
      hitRadius: 20,
      hoverRadius: 8
    }
  }
};
```

### Performance Optimization

```typescript
// Lazy loading for mobile
import { lazy, Suspense } from 'react';

const QualityChart = lazy(() => import('./charts/QualityChart'));
const ActivityHeatmap = lazy(() => import('./charts/ActivityHeatmap'));

function MobileDashboard() {
  return (
    <div>
      <StatsOverview />

      <Suspense fallback={<ChartSkeleton />}>
        <QualityChart />
      </Suspense>

      {/* Only load heatmap when scrolled into view */}
      <LazyLoad>
        <Suspense fallback={<ChartSkeleton />}>
          <ActivityHeatmap />
        </Suspense>
      </LazyLoad>
    </div>
  );
}
```

### Offline Support

```typescript
// Service Worker caching for offline dashboard
const CACHE_NAME = 'dashboard-v1';
const DASHBOARD_URL = '/api/analytics/dashboard';

self.addEventListener('fetch', (event) => {
  if (event.request.url.includes(DASHBOARD_URL)) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        try {
          // Try network first
          const response = await fetch(event.request);
          cache.put(event.request, response.clone());
          return response;
        } catch (error) {
          // Fall back to cache
          const cached = await cache.match(event.request);
          if (cached) {
            return cached;
          }
          throw error;
        }
      })
    );
  }
});
```
