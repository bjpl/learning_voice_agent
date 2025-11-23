/**
 * Analytics store with caching using Pinia Composition API
 * Integrates with API services for backend communication
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import analyticsService from '@/services/analytics';
import type {
  OverviewResponse,
  ProgressChartResponse,
  TrendResponse,
  TrendCategory,
  InsightResponse,
  SkillBreakdown,
} from '@/types';

// Legacy interfaces for backward compatibility
export interface AnalyticsData {
  totalSessions: number;
  totalDuration: number;
  averageSessionLength: number;
  topicsDiscussed: TopicStat[];
  dailyActivity: DailyActivity[];
  weeklyProgress: WeeklyProgress[];
  skillLevels: SkillLevel[];
}

export interface TopicStat {
  topic: string;
  count: number;
  percentage: number;
}

export interface DailyActivity {
  date: string;
  sessions: number;
  duration: number;
  messagesCount: number;
}

export interface WeeklyProgress {
  week: string;
  score: number;
  improvement: number;
}

export interface SkillLevel {
  skill: string;
  level: number;
  maxLevel: number;
  progress: number;
}

export interface InsightData {
  id: string;
  type: 'tip' | 'warning' | 'achievement' | 'suggestion';
  title: string;
  description: string;
  createdAt: Date;
  dismissed: boolean;
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const useAnalyticsStore = defineStore('analytics', () => {
  // Legacy state
  const data = ref<AnalyticsData>({
    totalSessions: 0,
    totalDuration: 0,
    averageSessionLength: 0,
    topicsDiscussed: [],
    dailyActivity: [],
    weeklyProgress: [],
    skillLevels: []
  });
  const insights = ref<InsightData[]>([]);
  const selectedPeriod = ref<'week' | 'month' | 'year'>('week');

  // API-integrated state
  const overview = ref<OverviewResponse | null>(null);
  const progress = ref<ProgressChartResponse | null>(null);
  const trends = ref<Map<TrendCategory, TrendResponse>>(new Map());
  const apiInsights = ref<InsightResponse | null>(null);
  const skills = ref<SkillBreakdown[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  const lastUpdated = ref<Date | null>(null);

  // Cache storage
  const cache = ref<Map<string, CacheEntry<unknown>>>(new Map());

  // Legacy computed
  const activeInsights = computed(() =>
    insights.value.filter(i => !i.dismissed)
  );

  const formattedDuration = computed(() => {
    const hours = Math.floor(data.value.totalDuration / 3600);
    const minutes = Math.floor((data.value.totalDuration % 3600) / 60);
    return `${hours}h ${minutes}m`;
  });

  const topSkill = computed(() => {
    if (data.value.skillLevels.length === 0) return null;
    return data.value.skillLevels.reduce((a, b) =>
      a.progress > b.progress ? a : b
    );
  });

  // API computed
  const currentStreak = computed(() => overview.value?.current_streak ?? 0);

  const totalPracticeMinutes = computed(
    () => overview.value?.total_practice_minutes ?? 0
  );

  const userLevel = computed(() => overview.value?.level ?? null);

  const experiencePoints = computed(() => overview.value?.xp ?? null);

  const improvementRate = computed(
    () => overview.value?.improvement_rate ?? 0
  );

  const progressTrend = computed(
    () => progress.value?.summary?.trend ?? 'stable'
  );

  const topInsights = computed(() => {
    if (!apiInsights.value?.insights) return [];
    return apiInsights.value.insights
      .filter((i) => i.type !== 'personalized_tip')
      .slice(0, 3);
  });

  const strengthAreas = computed(() => {
    return skills.value
      .filter((s) => s.trend === 'up')
      .sort((a, b) => b.change - a.change);
  });

  const weaknessAreas = computed(() => {
    return skills.value
      .filter((s) => s.trend === 'down')
      .sort((a, b) => a.change - b.change);
  });

  // Legacy actions
  async function fetchAnalytics(): Promise<void> {
    isLoading.value = true;
    error.value = null;
    try {
      // Try API first, fall back to mock data
      try {
        await fetchOverview(true);
        await fetchProgress(selectedPeriod.value as 'day' | 'week' | 'month' | 'year', true);

        // Map API data to legacy format
        if (overview.value) {
          data.value.totalSessions = overview.value.session_count;
          data.value.totalDuration = overview.value.total_practice_minutes * 60;
          data.value.averageSessionLength = overview.value.average_session_duration;
        }
      } catch {
        // Use mock data as fallback
        data.value = {
          totalSessions: 47,
          totalDuration: 28800,
          averageSessionLength: 612,
          topicsDiscussed: [
            { topic: 'Grammar', count: 23, percentage: 35 },
            { topic: 'Vocabulary', count: 18, percentage: 27 },
            { topic: 'Pronunciation', count: 15, percentage: 23 },
            { topic: 'Conversation', count: 10, percentage: 15 }
          ],
          dailyActivity: generateDailyActivity(),
          weeklyProgress: generateWeeklyProgress(),
          skillLevels: [
            { skill: 'Listening', level: 4, maxLevel: 10, progress: 72 },
            { skill: 'Speaking', level: 3, maxLevel: 10, progress: 58 },
            { skill: 'Vocabulary', level: 5, maxLevel: 10, progress: 85 },
            { skill: 'Grammar', level: 4, maxLevel: 10, progress: 67 }
          ]
        };
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch analytics';
    } finally {
      isLoading.value = false;
    }
  }

  function generateDailyActivity(): DailyActivity[] {
    const days: DailyActivity[] = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      days.push({
        date: date.toISOString().split('T')[0],
        sessions: Math.floor(Math.random() * 5) + 1,
        duration: Math.floor(Math.random() * 3600) + 600,
        messagesCount: Math.floor(Math.random() * 30) + 5
      });
    }
    return days;
  }

  function generateWeeklyProgress(): WeeklyProgress[] {
    return [
      { week: 'Week 1', score: 65, improvement: 0 },
      { week: 'Week 2', score: 72, improvement: 7 },
      { week: 'Week 3', score: 78, improvement: 6 },
      { week: 'Week 4', score: 85, improvement: 7 }
    ];
  }

  function addInsight(insight: Omit<InsightData, 'id' | 'createdAt' | 'dismissed'>): void {
    insights.value.push({
      ...insight,
      id: crypto.randomUUID(),
      createdAt: new Date(),
      dismissed: false
    });
  }

  function dismissInsight(id: string): void {
    const insight = insights.value.find(i => i.id === id);
    if (insight) {
      insight.dismissed = true;
    }
  }

  function setPeriod(period: 'week' | 'month' | 'year'): void {
    selectedPeriod.value = period;
  }

  // API actions
  const fetchOverview = async (forceRefresh = false): Promise<OverviewResponse> => {
    const cacheKey = 'overview';

    if (!forceRefresh) {
      const cached = getFromCache<OverviewResponse>(cacheKey);
      if (cached) {
        overview.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await analyticsService.getOverview();
      overview.value = apiData;
      setCache(cacheKey, apiData);
      lastUpdated.value = new Date();
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch overview';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchProgress = async (
    period: 'day' | 'week' | 'month' | 'year' = 'week',
    forceRefresh = false
  ): Promise<ProgressChartResponse> => {
    const cacheKey = `progress_${period}`;

    if (!forceRefresh) {
      const cached = getFromCache<ProgressChartResponse>(cacheKey);
      if (cached) {
        progress.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await analyticsService.getProgress(period);
      progress.value = apiData;
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch progress';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchTrends = async (
    category: TrendCategory,
    forceRefresh = false
  ): Promise<TrendResponse> => {
    const cacheKey = `trends_${category}`;

    if (!forceRefresh) {
      const cached = getFromCache<TrendResponse>(cacheKey);
      if (cached) {
        trends.value.set(category, cached);
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await analyticsService.getTrends(category);
      trends.value.set(category, apiData);
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch trends';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchInsights = async (forceRefresh = false): Promise<InsightResponse> => {
    const cacheKey = 'insights';

    if (!forceRefresh) {
      const cached = getFromCache<InsightResponse>(cacheKey);
      if (cached) {
        apiInsights.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await analyticsService.getInsights();
      apiInsights.value = apiData;
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch insights';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchSkillBreakdown = async (
    forceRefresh = false
  ): Promise<SkillBreakdown[]> => {
    const cacheKey = 'skills';

    if (!forceRefresh) {
      const cached = getFromCache<SkillBreakdown[]>(cacheKey);
      if (cached) {
        skills.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await analyticsService.getSkillBreakdown();
      skills.value = apiData;
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch skills';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchAllDashboardData = async (forceRefresh = false): Promise<void> => {
    await Promise.all([
      fetchOverview(forceRefresh),
      fetchProgress('week', forceRefresh),
      fetchInsights(forceRefresh),
      fetchSkillBreakdown(forceRefresh),
    ]);
  };

  const exportData = async (format: 'json' | 'csv' = 'json'): Promise<Blob> => {
    return analyticsService.exportAnalytics(format);
  };

  const clearCache = (): void => {
    cache.value.clear();
    overview.value = null;
    progress.value = null;
    trends.value.clear();
    apiInsights.value = null;
    skills.value = [];
  };

  const clearError = (): void => {
    error.value = null;
  };

  // Cache helpers
  function getFromCache<T>(key: string): T | null {
    const entry = cache.value.get(key) as CacheEntry<T> | undefined;

    if (!entry) return null;

    if (Date.now() > entry.expiresAt) {
      cache.value.delete(key);
      return null;
    }

    return entry.data;
  }

  function setCache<T>(key: string, cacheData: T): void {
    const entry: CacheEntry<T> = {
      data: cacheData,
      timestamp: Date.now(),
      expiresAt: Date.now() + CACHE_DURATION,
    };
    cache.value.set(key, entry as CacheEntry<unknown>);
  }

  return {
    // Legacy state
    data,
    insights,
    selectedPeriod,

    // API state
    overview,
    progress,
    trends,
    apiInsights,
    skills,
    isLoading,
    error,
    lastUpdated,

    // Legacy computed
    activeInsights,
    formattedDuration,
    topSkill,

    // API computed
    currentStreak,
    totalPracticeMinutes,
    userLevel,
    experiencePoints,
    improvementRate,
    progressTrend,
    topInsights,
    strengthAreas,
    weaknessAreas,

    // Legacy actions
    fetchAnalytics,
    addInsight,
    dismissInsight,
    setPeriod,

    // API actions
    fetchOverview,
    fetchProgress,
    fetchTrends,
    fetchInsights,
    fetchSkillBreakdown,
    fetchAllDashboardData,
    exportData,
    clearCache,
    clearError,
  };
});

export type AnalyticsStore = ReturnType<typeof useAnalyticsStore>;
