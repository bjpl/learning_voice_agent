<script setup lang="ts">
/**
 * ExportDialog - Export data with selectable options
 *
 * PATTERN: Modal dialog with form controls
 * WHY: Allow users to customize what data they export
 */
import { ref, computed, watch } from 'vue';
import { useSync } from '@/composables/useSync';
import type { ExportOptions, ExportFormat } from '@/types/sync';

interface Props {
  isOpen: boolean;
}

defineProps<Props>();

const emit = defineEmits<{
  (e: 'close'): void;
  (e: 'exported', file: Blob): void;
}>();

const { exportData, exportProgress, isExporting } = useSync();

// Form state
const options = ref<Partial<ExportOptions>>({
  format: 'json' as ExportFormat,
  includeConversations: true,
  includeGoals: true,
  includeAchievements: true,
  includeFeedback: true,
  includeAnalytics: true,
  includeSettings: true,
  encrypted: false,
  compression: 'none',
});

const useDateRange = ref(false);
const startDate = ref('');
const endDate = ref('');

// Watch date inputs
watch([startDate, endDate], () => {
  if (startDate.value || endDate.value) {
    options.value.dateRange = {
      start: startDate.value || '',
      end: endDate.value || '',
    };
  } else {
    options.value.dateRange = undefined;
  }
});

watch(useDateRange, (val) => {
  if (!val) {
    options.value.dateRange = undefined;
    startDate.value = '';
    endDate.value = '';
  }
});

const hasSelection = computed(() => {
  return (
    options.value.includeConversations ||
    options.value.includeGoals ||
    options.value.includeAchievements ||
    options.value.includeFeedback ||
    options.value.includeAnalytics ||
    options.value.includeSettings
  );
});

const handleExport = async () => {
  try {
    const blob = await exportData(options.value);

    // Generate filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `learning-voice-agent-backup-${timestamp}.json`;

    // Trigger download
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    emit('exported', blob);
    emit('close');
  } catch (error) {
    console.error('Export failed:', error);
  }
};

const handleBackdropClick = (event: MouseEvent) => {
  if (event.target === event.currentTarget && !isExporting.value) {
    emit('close');
  }
};

const selectAll = () => {
  options.value.includeConversations = true;
  options.value.includeGoals = true;
  options.value.includeAchievements = true;
  options.value.includeAnalytics = true;
  options.value.includeSettings = true;
};

const selectNone = () => {
  options.value.includeConversations = false;
  options.value.includeGoals = false;
  options.value.includeAchievements = false;
  options.value.includeAnalytics = false;
  options.value.includeSettings = false;
};
</script>

<template>
  <Teleport to="body">
    <Transition
      enter-active-class="transition-opacity duration-200 ease-out"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition-opacity duration-150 ease-in"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div
        v-if="isOpen"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
        @click="handleBackdropClick"
      >
        <Transition
          enter-active-class="transition-all duration-200 ease-out"
          enter-from-class="opacity-0 scale-95"
          enter-to-class="opacity-100 scale-100"
          leave-active-class="transition-all duration-150 ease-in"
          leave-from-class="opacity-100 scale-100"
          leave-to-class="opacity-0 scale-95"
        >
          <div
            v-if="isOpen"
            class="w-full max-w-lg rounded-xl bg-white dark:bg-gray-800 shadow-xl overflow-hidden"
            role="dialog"
            aria-modal="true"
            aria-labelledby="export-dialog-title"
          >
            <!-- Header -->
            <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div class="flex items-center justify-between">
                <h2
                  id="export-dialog-title"
                  class="text-xl font-semibold text-gray-900 dark:text-white"
                >
                  Export Data
                </h2>
                <button
                  @click="emit('close')"
                  :disabled="isExporting"
                  class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <!-- Content -->
            <div class="px-6 py-4 max-h-96 overflow-y-auto">
              <!-- Quick Actions -->
              <div class="flex gap-2 mb-4">
                <button
                  @click="selectAll"
                  class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Select All
                </button>
                <span class="text-gray-300 dark:text-gray-600">|</span>
                <button
                  @click="selectNone"
                  class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Select None
                </button>
              </div>

              <!-- Data Selection -->
              <div class="space-y-3 mb-6">
                <label class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <input
                    v-model="options.includeConversations"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 rounded border-gray-300 dark:border-gray-600 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                  <div class="flex-1">
                    <span class="font-medium text-gray-900 dark:text-white">Conversations</span>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Chat history and messages</p>
                  </div>
                  <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </label>

                <label class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <input
                    v-model="options.includeGoals"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 rounded border-gray-300 dark:border-gray-600 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                  <div class="flex-1">
                    <span class="font-medium text-gray-900 dark:text-white">Goals</span>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Learning goals and progress</p>
                  </div>
                  <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </label>

                <label class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <input
                    v-model="options.includeAchievements"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 rounded border-gray-300 dark:border-gray-600 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                  <div class="flex-1">
                    <span class="font-medium text-gray-900 dark:text-white">Achievements</span>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Badges and unlocked achievements</p>
                  </div>
                  <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                  </svg>
                </label>

                <label class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <input
                    v-model="options.includeAnalytics"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 rounded border-gray-300 dark:border-gray-600 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                  <div class="flex-1">
                    <span class="font-medium text-gray-900 dark:text-white">Analytics</span>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Learning statistics and trends</p>
                  </div>
                  <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </label>

                <label class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <input
                    v-model="options.includeSettings"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 rounded border-gray-300 dark:border-gray-600 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                  <div class="flex-1">
                    <span class="font-medium text-gray-900 dark:text-white">Settings</span>
                    <p class="text-sm text-gray-500 dark:text-gray-400">App preferences and configuration</p>
                  </div>
                  <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </label>
              </div>

              <!-- Date Range Filter -->
              <div class="border-t border-gray-200 dark:border-gray-700 pt-4">
                <label class="flex items-center gap-3 mb-3 cursor-pointer">
                  <input
                    v-model="useDateRange"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 rounded border-gray-300 dark:border-gray-600 focus:ring-blue-500"
                  />
                  <span class="font-medium text-gray-900 dark:text-white">Filter by date range</span>
                </label>

                <Transition
                  enter-active-class="transition-all duration-200"
                  enter-from-class="opacity-0 max-h-0"
                  enter-to-class="opacity-100 max-h-24"
                  leave-active-class="transition-all duration-150"
                  leave-from-class="opacity-100 max-h-24"
                  leave-to-class="opacity-0 max-h-0"
                >
                  <div v-if="useDateRange" class="grid grid-cols-2 gap-4 overflow-hidden">
                    <div>
                      <label class="block text-sm text-gray-600 dark:text-gray-400 mb-1">From</label>
                      <input
                        v-model="startDate"
                        type="date"
                        class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label class="block text-sm text-gray-600 dark:text-gray-400 mb-1">To</label>
                      <input
                        v-model="endDate"
                        type="date"
                        class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>
                </Transition>
              </div>
            </div>

            <!-- Progress Bar -->
            <Transition
              enter-active-class="transition-all duration-200"
              enter-from-class="opacity-0"
              enter-to-class="opacity-100"
              leave-active-class="transition-all duration-150"
              leave-from-class="opacity-100"
              leave-to-class="opacity-0"
            >
              <div v-if="isExporting" class="px-6 py-2">
                <div class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                  <span>Exporting...</span>
                  <span>{{ exportProgress }}%</span>
                </div>
                <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    class="h-full bg-blue-600 rounded-full transition-all duration-300"
                    :style="{ width: `${exportProgress}%` }"
                  />
                </div>
              </div>
            </Transition>

            <!-- Footer -->
            <div class="flex justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
              <button
                @click="emit('close')"
                :disabled="isExporting"
                class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                @click="handleExport"
                :disabled="!hasSelection || isExporting"
                class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:bg-blue-400 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <svg v-if="!isExporting" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                <svg v-else class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                {{ isExporting ? 'Exporting...' : 'Export' }}
              </button>
            </div>
          </div>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>
