/**
 * Goals and achievements API service
 */

import { apiClient } from './api';
import type {
  ApiResponse,
  Goal,
  GoalCreateRequest,
  GoalUpdateRequest,
  GoalProgress,
  GoalSuggestion,
  Achievement,
} from '@/types';

/**
 * Get all goals for current user
 */
export const getGoals = async (): Promise<Goal[]> => {
  const response = await apiClient.get<ApiResponse<Goal[]>>('/api/goals');
  return response.data.data;
};

/**
 * Get a specific goal by ID
 */
export const getGoal = async (goalId: string): Promise<Goal> => {
  const response = await apiClient.get<ApiResponse<Goal>>(
    `/api/goals/${goalId}`
  );
  return response.data.data;
};

/**
 * Create a new goal
 */
export const createGoal = async (goal: GoalCreateRequest): Promise<Goal> => {
  const response = await apiClient.post<ApiResponse<Goal>>('/api/goals', goal);
  return response.data.data;
};

/**
 * Update an existing goal
 */
export const updateGoal = async (
  goalId: string,
  updates: GoalUpdateRequest
): Promise<Goal> => {
  const response = await apiClient.patch<ApiResponse<Goal>>(
    `/api/goals/${goalId}`,
    updates
  );
  return response.data.data;
};

/**
 * Delete a goal
 */
export const deleteGoal = async (goalId: string): Promise<void> => {
  await apiClient.delete(`/api/goals/${goalId}`);
};

/**
 * Get progress for all active goals
 */
export const getGoalsProgress = async (): Promise<GoalProgress[]> => {
  const response = await apiClient.get<ApiResponse<GoalProgress[]>>(
    '/api/goals/progress'
  );
  return response.data.data;
};

/**
 * Get suggested goals based on user activity
 */
export const getSuggestedGoals = async (): Promise<GoalSuggestion[]> => {
  const response = await apiClient.get<ApiResponse<GoalSuggestion[]>>(
    '/api/goals/suggestions'
  );
  return response.data.data;
};

/**
 * Get all achievements
 */
export const getAchievements = async (): Promise<Achievement[]> => {
  const response = await apiClient.get<ApiResponse<Achievement[]>>(
    '/api/achievements'
  );
  return response.data.data;
};

/**
 * Get unlocked achievements only
 */
export const getUnlockedAchievements = async (): Promise<Achievement[]> => {
  const response = await apiClient.get<ApiResponse<Achievement[]>>(
    '/api/achievements',
    { params: { unlocked: true } }
  );
  return response.data.data;
};

/**
 * Get achievement details by ID
 */
export const getAchievement = async (
  achievementId: string
): Promise<Achievement> => {
  const response = await apiClient.get<ApiResponse<Achievement>>(
    `/api/achievements/${achievementId}`
  );
  return response.data.data;
};

/**
 * Check for newly unlocked achievements
 */
export const checkNewAchievements = async (): Promise<Achievement[]> => {
  const response = await apiClient.post<ApiResponse<Achievement[]>>(
    '/api/achievements/check'
  );
  return response.data.data;
};

/**
 * Mark achievement celebration as shown
 */
export const markAchievementSeen = async (
  achievementId: string
): Promise<void> => {
  await apiClient.post(`/api/achievements/${achievementId}/seen`);
};

export default {
  getGoals,
  getGoal,
  createGoal,
  updateGoal,
  deleteGoal,
  getGoalsProgress,
  getSuggestedGoals,
  getAchievements,
  getUnlockedAchievements,
  getAchievement,
  checkNewAchievements,
  markAchievementSeen,
};
