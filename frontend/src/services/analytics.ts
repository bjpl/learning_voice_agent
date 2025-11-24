/**
 * Analytics and dashboard API service
 */

import { apiClient } from './api';
import type {
  ApiResponse,
  OverviewResponse,
  ProgressChartResponse,
  TrendResponse,
  TrendCategory,
  InsightResponse,
  SkillBreakdown,
  AnalyticsPeriod,
} from '@/types';

/**
 * Get analytics overview
 */
export const getOverview = async (): Promise<OverviewResponse> => {
  const response = await apiClient.get<ApiResponse<OverviewResponse>>(
    '/api/analytics/overview'
  );

  return response.data.data;
};

/**
 * Get progress chart data
 */
export const getProgress = async (
  period: 'day' | 'week' | 'month' | 'year' = 'week'
): Promise<ProgressChartResponse> => {
  const response = await apiClient.get<ApiResponse<ProgressChartResponse>>(
    '/api/analytics/progress',
    { params: { period } }
  );

  return response.data.data;
};

/**
 * Get trend data for a specific category
 */
export const getTrends = async (
  category: TrendCategory = 'fluency',
  period?: AnalyticsPeriod
): Promise<TrendResponse> => {
  const params: Record<string, string> = { category };

  if (period) {
    params.start = period.start;
    params.end = period.end;
  }

  const response = await apiClient.get<ApiResponse<TrendResponse>>(
    '/api/analytics/trends',
    { params }
  );

  return response.data.data;
};

/**
 * Get personalized insights
 */
export const getInsights = async (): Promise<InsightResponse> => {
  const response = await apiClient.get<ApiResponse<InsightResponse>>(
    '/api/analytics/insights'
  );

  return response.data.data;
};

/**
 * Get skill breakdown analysis
 */
export const getSkillBreakdown = async (): Promise<SkillBreakdown[]> => {
  const response = await apiClient.get<ApiResponse<SkillBreakdown[]>>(
    '/api/analytics/skills'
  );

  return response.data.data;
};

/**
 * Get practice statistics for a date range
 */
export const getPracticeStats = async (
  startDate: string,
  endDate: string
): Promise<{
  totalSessions: number;
  totalMinutes: number;
  avgSessionLength: number;
  mostActiveDay: string;
  practiceByDay: Record<string, number>;
}> => {
  const response = await apiClient.get<ApiResponse<{
    totalSessions: number;
    totalMinutes: number;
    avgSessionLength: number;
    mostActiveDay: string;
    practiceByDay: Record<string, number>;
  }>>('/api/analytics/practice-stats', {
    params: { startDate, endDate },
  });

  return response.data.data;
};

/**
 * Export analytics data
 */
export const exportAnalytics = async (
  format: 'json' | 'csv' = 'json',
  period?: AnalyticsPeriod
): Promise<Blob> => {
  const params: Record<string, string> = { format };

  if (period) {
    params.start = period.start;
    params.end = period.end;
  }

  const response = await apiClient.get('/api/analytics/export', {
    params,
    responseType: 'blob',
  });

  return response.data;
};

export default {
  getOverview,
  getProgress,
  getTrends,
  getInsights,
  getSkillBreakdown,
  getPracticeStats,
  exportAnalytics,
};
