<script setup lang="ts">
/**
 * BackupHistory - Display and manage backup history
 *
 * PATTERN: List with restore and delete actions
 * WHY: Allow users to see and restore from previous backups
 */
import { ref, computed, onMounted } from 'vue';
import { useSync } from '@/composables/useSync';
import type { BackupHistoryItem, BackupType } from '@/types/sync';

const { backupHistory, getBackupHistory, restoreBackup, deleteBackup } = useSync();

const isRestoring = ref(false);
const restoringId = ref<string | null>(null);
const showConfirmRestore = ref(false);
const showConfirmDelete = ref(false);
const selectedBackup = ref<BackupHistoryItem | null>(null);

onMounted(() => {
  getBackupHistory();
});

const sortedBackups = computed(() => {
  return [...backupHistory.value].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
});

const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr);
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const getRelativeTime = (dateStr: string): string => {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)} days ago`;
  return formatDate(dateStr);
};

const isAutoBackup = (backupType: BackupType): boolean => {
  return backupType === 'automatic' || backupType === 'scheduled';
};

const confirmRestore = (backup: BackupHistoryItem) => {
  selectedBackup.value = backup;
  showConfirmRestore.value = true;
};

const confirmDelete = (backup: BackupHistoryItem) => {
  selectedBackup.value = backup;
  showConfirmDelete.value = true;
};

const handleRestore = async () => {
  if (!selectedBackup.value) return;

  isRestoring.value = true;
  restoringId.value = selectedBackup.value.id;

  try {
    await restoreBackup(selectedBackup.value.id);
    showConfirmRestore.value = false;
    selectedBackup.value = null;
  } catch (error) {
    console.error('Restore failed:', error);
  } finally {
    isRestoring.value = false;
    restoringId.value = null;
  }
};

const handleDelete = () => {
  if (selectedBackup.value) {
    deleteBackup(selectedBackup.value.id);
    showConfirmDelete.value = false;
    selectedBackup.value = null;
  }
};

const cancelAction = () => {
  showConfirmRestore.value = false;
  showConfirmDelete.value = false;
  selectedBackup.value = null;
};
</script>

<template>
  <div class="rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            Backup History
          </h3>
          <p class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            Previous backups of your data
          </p>
        </div>
        <span class="px-2.5 py-1 text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full">
          {{ backupHistory.length }} backup{{ backupHistory.length !== 1 ? 's' : '' }}
        </span>
      </div>
    </div>

    <!-- Backup List -->
    <div class="divide-y divide-gray-200 dark:divide-gray-700 max-h-96 overflow-y-auto">
      <div
        v-for="backup in sortedBackups"
        :key="backup.id"
        class="px-6 py-4 flex items-center gap-4"
      >
        <!-- Icon -->
        <div
          :class="[
            'p-3 rounded-lg',
            isAutoBackup(backup.type)
              ? 'bg-green-100 dark:bg-green-900/30'
              : 'bg-blue-100 dark:bg-blue-900/30'
          ]"
        >
          <svg
            v-if="isAutoBackup(backup.type)"
            class="w-5 h-5 text-green-600 dark:text-green-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <svg
            v-else
            class="w-5 h-5 text-blue-600 dark:text-blue-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
        </div>

        <!-- Backup Info -->
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            <span class="font-medium text-gray-900 dark:text-white">
              {{ getRelativeTime(backup.createdAt) }}
            </span>
            <span
              :class="[
                'px-2 py-0.5 text-xs font-medium rounded-full capitalize',
                isAutoBackup(backup.type)
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                  : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
              ]"
            >
              {{ backup.type }}
            </span>
          </div>
          <div class="flex items-center gap-3 mt-1">
            <span class="text-sm text-gray-500 dark:text-gray-400">
              {{ formatSize(backup.size) }}
            </span>
            <span class="text-gray-300 dark:text-gray-600">|</span>
            <span class="text-sm text-gray-500 dark:text-gray-400">
              {{ backup.deviceName }}
            </span>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex items-center gap-2">
          <button
            @click="confirmRestore(backup)"
            :disabled="isRestoring"
            class="px-3 py-1.5 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors disabled:opacity-50"
          >
            <span v-if="restoringId === backup.id" class="flex items-center gap-1">
              <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Restoring...
            </span>
            <span v-else>Restore</span>
          </button>
          <button
            @click="confirmDelete(backup)"
            :disabled="isRestoring"
            class="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
            title="Delete backup"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      <!-- Empty State -->
      <div
        v-if="backupHistory.length === 0"
        class="px-6 py-12 text-center"
      >
        <svg class="w-12 h-12 mx-auto text-gray-300 dark:text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
        </svg>
        <p class="text-gray-500 dark:text-gray-400 mb-1">No backups yet</p>
        <p class="text-sm text-gray-400 dark:text-gray-500">
          Backups will appear here when you export your data
        </p>
      </div>
    </div>

    <!-- Confirm Restore Dialog -->
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
          v-if="showConfirmRestore"
          class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          @click.self="cancelAction"
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
              v-if="showConfirmRestore"
              class="w-full max-w-md rounded-xl bg-white dark:bg-gray-800 shadow-xl"
            >
              <div class="p-6">
                <div class="flex items-start gap-4">
                  <div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                    <svg class="h-6 w-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  </div>
                  <div class="flex-1">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
                      Restore Backup
                    </h3>
                    <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
                      Are you sure you want to restore from this backup?
                      This will replace your current data with the backup from
                      <strong>{{ selectedBackup ? formatDate(selectedBackup.createdAt) : '' }}</strong>.
                    </p>
                  </div>
                </div>
              </div>
              <div class="flex justify-end gap-3 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
                <button
                  @click="cancelAction"
                  :disabled="isRestoring"
                  class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  @click="handleRestore"
                  :disabled="isRestoring"
                  class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  <svg v-if="isRestoring" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  {{ isRestoring ? 'Restoring...' : 'Restore' }}
                </button>
              </div>
            </div>
          </Transition>
        </div>
      </Transition>
    </Teleport>

    <!-- Confirm Delete Dialog -->
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
          v-if="showConfirmDelete"
          class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          @click.self="cancelAction"
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
              v-if="showConfirmDelete"
              class="w-full max-w-md rounded-xl bg-white dark:bg-gray-800 shadow-xl"
            >
              <div class="p-6">
                <div class="flex items-start gap-4">
                  <div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
                    <svg class="h-6 w-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </div>
                  <div class="flex-1">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
                      Delete Backup
                    </h3>
                    <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
                      Are you sure you want to delete this backup from
                      <strong>{{ selectedBackup ? formatDate(selectedBackup.createdAt) : '' }}</strong>?
                      This action cannot be undone.
                    </p>
                  </div>
                </div>
              </div>
              <div class="flex justify-end gap-3 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
                <button
                  @click="cancelAction"
                  class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  @click="handleDelete"
                  class="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  Delete
                </button>
              </div>
            </div>
          </Transition>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>
