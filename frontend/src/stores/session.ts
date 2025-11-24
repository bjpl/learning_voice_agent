/**
 * Session and preferences store using Pinia Composition API
 */

import { ref, computed, watch } from 'vue';
import { defineStore } from 'pinia';
import authService, { type SessionInfo } from '@/services/auth';

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  voiceEnabled: boolean;
  autoPlayAudio: boolean;
  showCorrections: boolean;
  difficultyLevel: 'beginner' | 'intermediate' | 'advanced';
  notificationsEnabled: boolean;
  soundEffects: boolean;
  fontSize: 'small' | 'medium' | 'large';
}

const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'system',
  language: 'en',
  voiceEnabled: true,
  autoPlayAudio: true,
  showCorrections: true,
  difficultyLevel: 'intermediate',
  notificationsEnabled: true,
  soundEffects: true,
  fontSize: 'medium',
};

const PREFERENCES_KEY = 'user_preferences';

export const useSessionStore = defineStore('session', () => {
  // State
  const sessionInfo = ref<SessionInfo | null>(null);
  const preferences = ref<UserPreferences>(loadPreferences());
  const isOnline = ref(navigator.onLine);
  const isApiAvailable = ref(false);
  const lastPingTime = ref<Date | null>(null);

  // Computed
  const isSessionActive = computed(() => {
    return sessionInfo.value?.isActive ?? false;
  });

  const sessionId = computed(() => {
    return sessionInfo.value?.sessionId ?? null;
  });

  const sessionDuration = computed(() => {
    if (!sessionInfo.value?.createdAt) return 0;
    const start = new Date(sessionInfo.value.createdAt);
    const now = new Date();
    return Math.floor((now.getTime() - start.getTime()) / 1000);
  });

  const effectiveTheme = computed(() => {
    if (preferences.value.theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
    }
    return preferences.value.theme;
  });

  // Actions
  const initializeSession = async (): Promise<SessionInfo> => {
    sessionInfo.value = authService.initSession();
    await checkApiAvailability();
    startOnlineMonitoring();
    return sessionInfo.value;
  };

  const endCurrentSession = (): void => {
    authService.endSession();
    sessionInfo.value = null;
  };

  const createNewSession = (): SessionInfo => {
    sessionInfo.value = authService.createNewSession();
    return sessionInfo.value;
  };

  const touchSession = (): void => {
    if (sessionInfo.value) {
      authService.touchSession();
      sessionInfo.value.lastActive = new Date().toISOString();
    }
  };

  const checkApiAvailability = async (): Promise<boolean> => {
    try {
      isApiAvailable.value = await authService.ping();
      lastPingTime.value = new Date();
      return isApiAvailable.value;
    } catch {
      isApiAvailable.value = false;
      return false;
    }
  };

  const updatePreference = <K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ): void => {
    preferences.value[key] = value;
    savePreferences();
  };

  const updatePreferences = (updates: Partial<UserPreferences>): void => {
    preferences.value = { ...preferences.value, ...updates };
    savePreferences();
  };

  const resetPreferences = (): void => {
    preferences.value = { ...DEFAULT_PREFERENCES };
    savePreferences();
  };

  // Private functions
  function loadPreferences(): UserPreferences {
    const stored = localStorage.getItem(PREFERENCES_KEY);
    if (stored) {
      try {
        return { ...DEFAULT_PREFERENCES, ...JSON.parse(stored) };
      } catch {
        return { ...DEFAULT_PREFERENCES };
      }
    }
    return { ...DEFAULT_PREFERENCES };
  }

  function savePreferences(): void {
    localStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences.value));
  }

  function startOnlineMonitoring(): void {
    window.addEventListener('online', () => {
      isOnline.value = true;
      checkApiAvailability();
    });

    window.addEventListener('offline', () => {
      isOnline.value = false;
      isApiAvailable.value = false;
    });

    // Periodic API check every 30 seconds
    setInterval(() => {
      if (isOnline.value) {
        checkApiAvailability();
      }
    }, 30000);
  }

  // Watch for theme changes
  watch(
    () => preferences.value.theme,
    () => {
      document.documentElement.setAttribute('data-theme', effectiveTheme.value);
    },
    { immediate: true }
  );

  return {
    // State
    sessionInfo,
    preferences,
    isOnline,
    isApiAvailable,
    lastPingTime,

    // Computed
    isSessionActive,
    sessionId,
    sessionDuration,
    effectiveTheme,

    // Actions
    initializeSession,
    endCurrentSession,
    createNewSession,
    touchSession,
    checkApiAvailability,
    updatePreference,
    updatePreferences,
    resetPreferences,
  };
});

export type SessionStore = ReturnType<typeof useSessionStore>;
