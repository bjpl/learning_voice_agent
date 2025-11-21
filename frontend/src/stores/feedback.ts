/**
 * Feedback store with offline queue support using Pinia Composition API
 */

import { ref, computed, watch } from 'vue';
import { defineStore } from 'pinia';
import feedbackService from '@/services/feedback';
import type {
  ExplicitFeedback,
  ImplicitFeedback,
  CorrectionFeedback,
  FeedbackQueueItem,
  FeedbackSummary,
} from '@/types';

const QUEUE_STORAGE_KEY = 'feedback_queue';
const MAX_RETRIES = 3;
const RETRY_DELAY = 5000;

export const useFeedbackStore = defineStore('feedback', () => {
  // State
  const queue = ref<FeedbackQueueItem[]>(loadQueueFromStorage());
  const isProcessing = ref(false);
  const summary = ref<FeedbackSummary | null>(null);
  const error = ref<string | null>(null);
  const lastSubmitted = ref<Date | null>(null);

  // Computed
  const queueSize = computed(() => queue.value.length);

  const hasPendingFeedback = computed(() => queue.value.length > 0);

  const pendingExplicit = computed(() => {
    return queue.value.filter((item) => item.type === 'explicit').length;
  });

  const pendingImplicit = computed(() => {
    return queue.value.filter((item) => item.type === 'implicit').length;
  });

  const pendingCorrections = computed(() => {
    return queue.value.filter((item) => item.type === 'correction').length;
  });

  const failedItems = computed(() => {
    return queue.value.filter((item) => item.retries >= MAX_RETRIES);
  });

  // Actions
  const submitExplicitFeedback = async (
    feedback: Omit<ExplicitFeedback, 'id' | 'session_id' | 'timestamp'>
  ): Promise<void> => {
    try {
      await feedbackService.submitExplicitFeedback(feedback);
      lastSubmitted.value = new Date();
    } catch {
      // Queue for later if submission fails
      addToQueue('explicit', feedback as ExplicitFeedback);
    }
  };

  const submitImplicitFeedback = async (
    feedback: Omit<ImplicitFeedback, 'id' | 'session_id' | 'timestamp'>
  ): Promise<void> => {
    try {
      await feedbackService.submitImplicitFeedback(feedback);
      lastSubmitted.value = new Date();
    } catch {
      // Queue for later if submission fails
      addToQueue('implicit', feedback as ImplicitFeedback);
    }
  };

  const submitCorrectionFeedback = async (
    feedback: Omit<CorrectionFeedback, 'id' | 'session_id' | 'timestamp'>
  ): Promise<void> => {
    try {
      await feedbackService.submitCorrectionFeedback(feedback);
      lastSubmitted.value = new Date();
    } catch {
      // Queue for later if submission fails
      addToQueue('correction', feedback as CorrectionFeedback);
    }
  };

  const addToQueue = (
    type: 'explicit' | 'implicit' | 'correction',
    payload: ExplicitFeedback | ImplicitFeedback | CorrectionFeedback
  ): void => {
    queue.value.push({
      type,
      payload,
      retries: 0,
      created_at: new Date().toISOString(),
    });
  };

  const processQueue = async (): Promise<void> => {
    if (isProcessing.value || queue.value.length === 0) {
      return;
    }

    isProcessing.value = true;
    error.value = null;

    const itemsToProcess = [...queue.value];
    const processedIds: number[] = [];
    const failedIds: number[] = [];

    for (let i = 0; i < itemsToProcess.length; i++) {
      const item = itemsToProcess[i];

      if (item.retries >= MAX_RETRIES) {
        continue; // Skip permanently failed items
      }

      try {
        switch (item.type) {
          case 'explicit':
            await feedbackService.submitExplicitFeedback(
              item.payload as Omit<ExplicitFeedback, 'id' | 'session_id' | 'timestamp'>
            );
            break;
          case 'implicit':
            await feedbackService.submitImplicitFeedback(
              item.payload as Omit<ImplicitFeedback, 'id' | 'session_id' | 'timestamp'>
            );
            break;
          case 'correction':
            await feedbackService.submitCorrectionFeedback(
              item.payload as Omit<CorrectionFeedback, 'id' | 'session_id' | 'timestamp'>
            );
            break;
        }
        processedIds.push(i);
      } catch {
        item.retries++;
        failedIds.push(i);
      }

      // Small delay between submissions
      await sleep(100);
    }

    // Remove successfully processed items
    queue.value = queue.value.filter((_, index) => !processedIds.includes(index));

    if (processedIds.length > 0) {
      lastSubmitted.value = new Date();
    }

    isProcessing.value = false;

    // Schedule retry if there are failed items
    if (failedIds.length > 0 && hasPendingFeedback.value) {
      setTimeout(() => processQueue(), RETRY_DELAY);
    }
  };

  const processBatch = async (): Promise<{ success: number; failed: number }> => {
    if (queue.value.length === 0) {
      return { success: 0, failed: 0 };
    }

    isProcessing.value = true;
    error.value = null;

    try {
      const items = queue.value.map((item) => ({
        type: item.type,
        payload: item.payload,
      }));

      const result = await feedbackService.submitFeedbackBatch(items);

      // Clear successfully submitted items
      if (result.success > 0) {
        queue.value = queue.value.slice(result.success);
        lastSubmitted.value = new Date();
      }

      return result;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Batch submission failed';
      throw e;
    } finally {
      isProcessing.value = false;
    }
  };

  const fetchSummary = async (): Promise<void> => {
    try {
      summary.value = await feedbackService.getFeedbackSummary();
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch summary';
    }
  };

  const clearQueue = (): void => {
    queue.value = [];
  };

  const clearFailedItems = (): void => {
    queue.value = queue.value.filter((item) => item.retries < MAX_RETRIES);
  };

  const retryFailedItems = (): void => {
    queue.value.forEach((item) => {
      if (item.retries >= MAX_RETRIES) {
        item.retries = 0;
      }
    });
    processQueue();
  };

  const clearError = (): void => {
    error.value = null;
  };

  // Storage helpers
  function loadQueueFromStorage(): FeedbackQueueItem[] {
    const stored = localStorage.getItem(QUEUE_STORAGE_KEY);
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch {
        return [];
      }
    }
    return [];
  }

  function saveQueueToStorage(): void {
    localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(queue.value));
  }

  // Utility
  function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Watch queue changes and persist
  watch(
    queue,
    () => {
      saveQueueToStorage();
    },
    { deep: true }
  );

  // Auto-process queue when online
  if (typeof window !== 'undefined') {
    window.addEventListener('online', () => {
      processQueue();
    });

    // Process queue on startup if online
    if (navigator.onLine && queue.value.length > 0) {
      setTimeout(() => processQueue(), 1000);
    }
  }

  return {
    // State
    queue,
    isProcessing,
    summary,
    error,
    lastSubmitted,

    // Computed
    queueSize,
    hasPendingFeedback,
    pendingExplicit,
    pendingImplicit,
    pendingCorrections,
    failedItems,

    // Actions
    submitExplicitFeedback,
    submitImplicitFeedback,
    submitCorrectionFeedback,
    addToQueue,
    processQueue,
    processBatch,
    fetchSummary,
    clearQueue,
    clearFailedItems,
    retryFailedItems,
    clearError,
  };
});

export type FeedbackStore = ReturnType<typeof useFeedbackStore>;
