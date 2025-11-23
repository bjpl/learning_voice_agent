/**
 * Goals and achievements store using Pinia Composition API
 * Integrates with API services for backend communication
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import goalsService from '@/services/goals';
import type {
  Goal as ApiGoal,
  GoalCreateRequest,
  GoalUpdateRequest,
  GoalProgress,
  GoalSuggestion,
  Achievement,
  AchievementUnlockedEvent,
} from '@/types';

// Legacy interface for backward compatibility
export interface Goal {
  id: string;
  title: string;
  description: string;
  type: 'daily' | 'weekly' | 'monthly' | 'custom';
  targetValue: number;
  currentValue: number;
  unit: string;
  deadline?: Date;
  createdAt: Date;
  completedAt?: Date;
  category: string;
  isActive: boolean;
}

export interface GoalCategory {
  id: string;
  name: string;
  icon: string;
  color: string;
}

export const useGoalsStore = defineStore('goals', () => {
  // Legacy state
  const goals = ref<Goal[]>([]);
  const categories = ref<GoalCategory[]>([
    { id: '1', name: 'Speaking', icon: 'mic', color: 'blue' },
    { id: '2', name: 'Listening', icon: 'headphones', color: 'green' },
    { id: '3', name: 'Vocabulary', icon: 'book', color: 'purple' },
    { id: '4', name: 'Grammar', icon: 'pencil', color: 'orange' },
    { id: '5', name: 'Practice', icon: 'clock', color: 'pink' }
  ]);

  // API-integrated state
  const apiGoals = ref<ApiGoal[]>([]);
  const achievements = ref<Achievement[]>([]);
  const goalsProgress = ref<Map<string, GoalProgress>>(new Map());
  const suggestions = ref<GoalSuggestion[]>([]);
  const recentlyUnlocked = ref<AchievementUnlockedEvent[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  // Legacy computed
  const activeGoals = computed(() => goals.value.filter(g => g.isActive));
  const completedGoals = computed(() => goals.value.filter(g => !g.isActive && g.completedAt));

  const dailyGoals = computed(() =>
    activeGoals.value.filter(g => g.type === 'daily')
  );

  const weeklyGoals = computed(() =>
    activeGoals.value.filter(g => g.type === 'weekly')
  );

  const overallProgress = computed(() => {
    if (activeGoals.value.length === 0) return 0;
    const total = activeGoals.value.reduce((sum, g) => {
      return sum + Math.min((g.currentValue / g.targetValue) * 100, 100);
    }, 0);
    return Math.round(total / activeGoals.value.length);
  });

  const goalsByCategory = computed(() => {
    const grouped: Record<string, Goal[]> = {};
    activeGoals.value.forEach(goal => {
      if (!grouped[goal.category]) {
        grouped[goal.category] = [];
      }
      grouped[goal.category].push(goal);
    });
    return grouped;
  });

  // API computed - Goals
  const apiActiveGoals = computed(() => {
    return apiGoals.value.filter((g) => g.status === 'active');
  });

  const apiCompletedGoals = computed(() => {
    return apiGoals.value.filter((g) => g.status === 'completed');
  });

  const overdueGoals = computed(() => {
    const now = new Date();
    return apiActiveGoals.value.filter((g) => {
      if (!g.deadline) return false;
      return new Date(g.deadline) < now;
    });
  });

  const goalsByPriority = computed(() => {
    return {
      high: apiActiveGoals.value.filter((g) => g.priority === 'high'),
      medium: apiActiveGoals.value.filter((g) => g.priority === 'medium'),
      low: apiActiveGoals.value.filter((g) => g.priority === 'low'),
    };
  });

  const totalGoalsProgress = computed(() => {
    if (apiActiveGoals.value.length === 0) return 0;
    const total = apiActiveGoals.value.reduce((acc, g) => {
      return acc + (g.current_value / g.target_value) * 100;
    }, 0);
    return Math.round(total / apiActiveGoals.value.length);
  });

  // API computed - Achievements
  const unlockedAchievements = computed(() => {
    return achievements.value.filter((a) => a.unlocked);
  });

  const lockedAchievements = computed(() => {
    return achievements.value.filter((a) => !a.unlocked);
  });

  const achievementsByRarity = computed(() => {
    const grouped: Record<string, Achievement[]> = {
      legendary: [],
      epic: [],
      rare: [],
      uncommon: [],
      common: [],
    };
    achievements.value.forEach((a) => {
      grouped[a.rarity].push(a);
    });
    return grouped;
  });

  const achievementsByCategory = computed(() => {
    const grouped: Record<string, Achievement[]> = {};
    achievements.value.forEach((a) => {
      if (!grouped[a.category]) {
        grouped[a.category] = [];
      }
      grouped[a.category].push(a);
    });
    return grouped;
  });

  const totalXPEarned = computed(() => {
    return unlockedAchievements.value.reduce((acc, a) => acc + a.xp_reward, 0);
  });

  const achievementProgress = computed(() => {
    const total = achievements.value.length;
    const unlocked = unlockedAchievements.value.length;
    return {
      total,
      unlocked,
      percentage: total > 0 ? Math.round((unlocked / total) * 100) : 0,
    };
  });

  const nearbyAchievements = computed(() => {
    return lockedAchievements.value
      .filter((a) => a.progress && a.progress.percentage >= 75)
      .sort((a, b) => (b.progress?.percentage ?? 0) - (a.progress?.percentage ?? 0));
  });

  // Legacy actions
  async function fetchGoals(): Promise<void> {
    isLoading.value = true;
    error.value = null;
    try {
      // Try API first
      try {
        await fetchApiGoals();
        // Map API goals to legacy format
        goals.value = apiGoals.value.map(mapApiGoalToLegacy);
      } catch {
        // Use mock data as fallback
        goals.value = [
          {
            id: '1',
            title: 'Practice Speaking',
            description: 'Complete 3 voice conversations today',
            type: 'daily',
            targetValue: 3,
            currentValue: 1,
            unit: 'conversations',
            category: 'Speaking',
            createdAt: new Date(),
            isActive: true
          },
          {
            id: '2',
            title: 'Learn Vocabulary',
            description: 'Learn 20 new words this week',
            type: 'weekly',
            targetValue: 20,
            currentValue: 12,
            unit: 'words',
            category: 'Vocabulary',
            createdAt: new Date(),
            isActive: true
          },
          {
            id: '3',
            title: 'Study Time',
            description: 'Practice for 30 minutes daily',
            type: 'daily',
            targetValue: 30,
            currentValue: 15,
            unit: 'minutes',
            category: 'Practice',
            createdAt: new Date(),
            isActive: true
          }
        ];
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch goals';
    } finally {
      isLoading.value = false;
    }
  }

  function mapApiGoalToLegacy(apiGoal: ApiGoal): Goal {
    return {
      id: apiGoal.id,
      title: apiGoal.title,
      description: apiGoal.description,
      type: apiGoal.type === 'practice_time' ? 'daily' :
            apiGoal.type === 'streak_days' ? 'weekly' : 'custom',
      targetValue: apiGoal.target_value,
      currentValue: apiGoal.current_value,
      unit: apiGoal.unit,
      deadline: apiGoal.deadline ? new Date(apiGoal.deadline) : undefined,
      createdAt: new Date(apiGoal.created_at),
      completedAt: apiGoal.completed_at ? new Date(apiGoal.completed_at) : undefined,
      category: apiGoal.metadata?.category ?? 'Practice',
      isActive: apiGoal.status === 'active',
    };
  }

  function createGoal(goal: Omit<Goal, 'id' | 'createdAt' | 'currentValue' | 'isActive'>): Goal {
    const newGoal: Goal = {
      ...goal,
      id: crypto.randomUUID(),
      createdAt: new Date(),
      currentValue: 0,
      isActive: true
    };
    goals.value.push(newGoal);
    return newGoal;
  }

  function updateGoalProgress(id: string, value: number): void {
    const goal = goals.value.find(g => g.id === id);
    if (goal) {
      goal.currentValue = Math.min(value, goal.targetValue);
      if (goal.currentValue >= goal.targetValue) {
        goal.isActive = false;
        goal.completedAt = new Date();
      }
    }
  }

  function incrementGoal(id: string, amount: number = 1): void {
    const goal = goals.value.find(g => g.id === id);
    if (goal) {
      updateGoalProgress(id, goal.currentValue + amount);
    }
  }

  function deleteGoal(id: string): void {
    const index = goals.value.findIndex(g => g.id === id);
    if (index !== -1) {
      goals.value.splice(index, 1);
    }
  }

  function resetDailyGoals(): void {
    goals.value
      .filter(g => g.type === 'daily' && g.isActive)
      .forEach(g => {
        g.currentValue = 0;
      });
  }

  // API actions - Goals
  const fetchApiGoals = async (): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      apiGoals.value = await goalsService.getGoals();
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch goals';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const createApiGoal = async (goalData: GoalCreateRequest): Promise<ApiGoal> => {
    isLoading.value = true;
    error.value = null;

    try {
      const newGoal = await goalsService.createGoal(goalData);
      apiGoals.value.push(newGoal);
      // Also add to legacy format
      goals.value.push(mapApiGoalToLegacy(newGoal));
      return newGoal;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to create goal';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const updateApiGoal = async (
    goalId: string,
    updates: GoalUpdateRequest
  ): Promise<ApiGoal> => {
    isLoading.value = true;
    error.value = null;

    try {
      const updatedGoal = await goalsService.updateGoal(goalId, updates);
      const index = apiGoals.value.findIndex((g) => g.id === goalId);
      if (index > -1) {
        apiGoals.value[index] = updatedGoal;
        // Update legacy version too
        const legacyIndex = goals.value.findIndex((g) => g.id === goalId);
        if (legacyIndex > -1) {
          goals.value[legacyIndex] = mapApiGoalToLegacy(updatedGoal);
        }
      }
      return updatedGoal;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to update goal';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const deleteApiGoal = async (goalId: string): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      await goalsService.deleteGoal(goalId);
      apiGoals.value = apiGoals.value.filter((g) => g.id !== goalId);
      goalsProgress.value.delete(goalId);
      // Remove from legacy too
      deleteGoal(goalId);
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to delete goal';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchGoalsProgress = async (): Promise<void> => {
    try {
      const progressList = await goalsService.getGoalsProgress();
      progressList.forEach((p) => {
        goalsProgress.value.set(p.goal_id, p);
      });
    } catch (e) {
      console.error('Failed to fetch goals progress:', e);
    }
  };

  const fetchSuggestions = async (): Promise<void> => {
    try {
      suggestions.value = await goalsService.getSuggestedGoals();
    } catch (e) {
      console.error('Failed to fetch suggestions:', e);
    }
  };

  const acceptSuggestion = async (suggestion: GoalSuggestion): Promise<ApiGoal> => {
    const goalData: GoalCreateRequest = {
      type: suggestion.type,
      title: suggestion.title,
      description: suggestion.description,
      target_value: suggestion.target_value,
      unit: suggestion.unit,
    };
    return createApiGoal(goalData);
  };

  // API actions - Achievements
  const fetchAchievements = async (): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      achievements.value = await goalsService.getAchievements();
    } catch (e) {
      error.value =
        e instanceof Error ? e.message : 'Failed to fetch achievements';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const checkForNewAchievements = async (): Promise<Achievement[]> => {
    try {
      const newAchievements = await goalsService.checkNewAchievements();

      newAchievements.forEach((achievement) => {
        recentlyUnlocked.value.push({
          achievement,
          unlocked_at: new Date().toISOString(),
          celebration_shown: false,
        });

        const index = achievements.value.findIndex(
          (a) => a.id === achievement.id
        );
        if (index > -1) {
          achievements.value[index] = achievement;
        }
      });

      return newAchievements;
    } catch (e) {
      console.error('Failed to check achievements:', e);
      return [];
    }
  };

  const markAchievementCelebrated = async (
    achievementId: string
  ): Promise<void> => {
    await goalsService.markAchievementSeen(achievementId);

    const event = recentlyUnlocked.value.find(
      (e) => e.achievement.id === achievementId
    );
    if (event) {
      event.celebration_shown = true;
    }
  };

  const dismissUnlockedAchievement = (achievementId: string): void => {
    recentlyUnlocked.value = recentlyUnlocked.value.filter(
      (e) => e.achievement.id !== achievementId
    );
  };

  // Combined actions
  const fetchAllGoalsData = async (): Promise<void> => {
    await Promise.all([
      fetchGoals(),
      fetchGoalsProgress(),
      fetchSuggestions(),
    ]);
  };

  const fetchAll = async (): Promise<void> => {
    await Promise.all([fetchAllGoalsData(), fetchAchievements()]);
  };

  const clearError = (): void => {
    error.value = null;
  };

  return {
    // Legacy state
    goals,
    categories,

    // API state
    apiGoals,
    achievements,
    goalsProgress,
    suggestions,
    recentlyUnlocked,
    isLoading,
    error,

    // Legacy computed
    activeGoals,
    completedGoals,
    dailyGoals,
    weeklyGoals,
    overallProgress,
    goalsByCategory,

    // API computed - Goals
    apiActiveGoals,
    apiCompletedGoals,
    overdueGoals,
    goalsByPriority,
    totalGoalsProgress,

    // API computed - Achievements
    unlockedAchievements,
    lockedAchievements,
    achievementsByRarity,
    achievementsByCategory,
    totalXPEarned,
    achievementProgress,
    nearbyAchievements,

    // Legacy actions
    fetchGoals,
    createGoal,
    updateGoalProgress,
    incrementGoal,
    deleteGoal,
    resetDailyGoals,

    // API actions - Goals
    fetchApiGoals,
    createApiGoal,
    updateApiGoal,
    deleteApiGoal,
    fetchGoalsProgress,
    fetchSuggestions,
    acceptSuggestion,

    // API actions - Achievements
    fetchAchievements,
    checkForNewAchievements,
    markAchievementCelebrated,
    dismissUnlockedAchievement,

    // Combined actions
    fetchAllGoalsData,
    fetchAll,
    clearError,
  };
});

export type GoalsStore = ReturnType<typeof useGoalsStore>;
