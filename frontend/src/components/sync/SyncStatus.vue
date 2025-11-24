<script setup lang="ts">
/**
 * SyncStatus - Display sync state and trigger manual sync
 *
 * PATTERN: Status card with action button
 * WHY: Show users their sync state at a glance
 */
import { computed } from 'vue';
import { useSync } from '@/composables/useSync';

const {
  syncStatus,
  lastSync,
  nextAutoBackup,
  dataSize,
  syncNow,
  isExporting,
  isImporting,
} = useSync();

const isBusy = computed(() => {
  return syncStatus.value === 'syncing' || isExporting.value || isImporting.value;
});

const statusConfig = computed(() => {
  switch (syncStatus.value) {
    case 'syncing':
      return {
        icon: 'sync',
        color: 'text-blue-500',
        bgColor: 'bg-blue-100 dark:bg-blue-900/30',
        label: 'Syncing...',
      };
    case 'synced':
      return {
        icon: 'check',
        color: 'text-green-500',
        bgColor: 'bg-green-100 dark:bg-green-900/30',
        label: 'Synced',
      };
    case 'error':
      return {
        icon: 'error',
        color: 'text-red-500',
        bgColor: 'bg-red-100 dark:bg-red-900/30',
        label: 'Sync Error',
      };
    case 'conflict':
      return {
        icon: 'warning',
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-100 dark:bg-yellow-900/30',
        label: 'Conflicts',
      };
    case 'offline':
      return {
        icon: 'offline',
        color: 'text-gray-500',
        bgColor: 'bg-gray-100 dark:bg-gray-700',
        label: 'Offline',
      };
    default:
      return {
        icon: 'cloud',
        color: 'text-gray-500',
        bgColor: 'bg-gray-100 dark:bg-gray-700',
        label: 'Ready',
      };
  }
});

const formatDate = (date: Date | null): string => {
  if (!date) return 'Never';
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
  return date.toLocaleDateString();
};

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatNextBackup = (date: Date | null): string => {
  if (!date) return 'Not scheduled';
  const now = new Date();
  const diff = date.getTime() - now.getTime();

  if (diff <= 0) return 'Due now';
  if (diff < 3600000) return `In ${Math.ceil(diff / 60000)} min`;
  if (diff < 86400000) return `In ${Math.ceil(diff / 3600000)} hours`;
  return date.toLocaleDateString();
};

const handleSync = async () => {
  await syncNow();
};
</script>

<template>
  <div class="rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div class="flex items-start justify-between mb-6">
      <div class="flex items-center gap-3">
        <div :class="[statusConfig.bgColor, 'p-3 rounded-lg']">
          <!-- Sync Icon -->
          <svg
            v-if="statusConfig.icon === 'sync'"
            :class="[statusConfig.color, 'w-6 h-6 animate-spin']"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          <!-- Check Icon -->
          <svg
            v-else-if="statusConfig.icon === 'check'"
            :class="[statusConfig.color, 'w-6 h-6']"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <!-- Error Icon -->
          <svg
            v-else-if="statusConfig.icon === 'error'"
            :class="[statusConfig.color, 'w-6 h-6']"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <!-- Cloud Icon -->
          <svg
            v-else
            :class="[statusConfig.color, 'w-6 h-6']"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"
            />
          </svg>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            Sync Status
          </h3>
          <p :class="[statusConfig.color, 'text-sm font-medium']">
            {{ statusConfig.label }}
          </p>
        </div>
      </div>
      <button
        @click="handleSync"
        :disabled="isBusy"
        class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
      >
        <span v-if="isBusy" class="flex items-center gap-2">
          <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Syncing...
        </span>
        <span v-else>Sync Now</span>
      </button>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <!-- Last Sync -->
      <div class="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
        <div class="flex items-center gap-2 mb-1">
          <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Last Sync
          </span>
        </div>
        <p class="text-sm font-semibold text-gray-900 dark:text-white">
          {{ formatDate(lastSync) }}
        </p>
      </div>

      <!-- Next Auto-Backup -->
      <div class="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
        <div class="flex items-center gap-2 mb-1">
          <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Next Backup
          </span>
        </div>
        <p class="text-sm font-semibold text-gray-900 dark:text-white">
          {{ formatNextBackup(nextAutoBackup) }}
        </p>
      </div>

      <!-- Data Size -->
      <div class="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
        <div class="flex items-center gap-2 mb-1">
          <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
          </svg>
          <span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Data Size
          </span>
        </div>
        <p class="text-sm font-semibold text-gray-900 dark:text-white">
          {{ formatSize(dataSize) }}
        </p>
      </div>
    </div>
  </div>
</template>
