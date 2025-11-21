/**
 * Central store exports for the application
 */

// Store exports
export { useUserStore } from './user';
export { useConversationStore, type ConversationStore, type ConversationState } from './conversation';
export { useAnalyticsStore, type AnalyticsStore } from './analytics';
export { useGoalsStore, type GoalsStore } from './goals';
export { useAchievementsStore } from './achievements';
export { useUIStore } from './ui';
export { useSessionStore, type SessionStore, type UserPreferences } from './session';
export { useFeedbackStore, type FeedbackStore } from './feedback';

// Type exports
export type { User, UserPreferences as LegacyUserPreferences } from './user';
export type { Message, Conversation } from './conversation';
export type { AnalyticsData, TopicStat, DailyActivity, WeeklyProgress, SkillLevel, InsightData } from './analytics';
export type { Goal, GoalCategory } from './goals';
export type { Achievement, AchievementCategory } from './achievements';
export type { Toast, ConfirmDialogOptions } from './ui';

/**
 * Initialize all stores - call this in main.ts after creating Pinia
 */
export const initializeStores = async (): Promise<void> => {
  const sessionStore = useSessionStore();
  const feedbackStore = useFeedbackStore();

  // Initialize session
  await sessionStore.initializeSession();

  // Process any pending feedback
  if (feedbackStore.hasPendingFeedback && navigator.onLine) {
    feedbackStore.processQueue();
  }
};

/**
 * Reset all stores - useful for logout or app reset
 */
export const resetAllStores = (): void => {
  const sessionStore = useSessionStore();
  const conversationStore = useConversationStore();
  const analyticsStore = useAnalyticsStore();
  const goalsStore = useGoalsStore();
  const feedbackStore = useFeedbackStore();

  sessionStore.endCurrentSession();
  conversationStore.clearMessages();
  analyticsStore.clearCache();
  feedbackStore.clearQueue();

  // Reset goals store state
  goalsStore.$reset();
};
