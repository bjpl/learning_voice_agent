<script setup lang="ts">
/**
 * ImportDialog - Import data from backup file
 *
 * PATTERN: Modal dialog with file upload and preview
 * WHY: Allow users to restore their data safely
 */
import { ref, computed } from 'vue';
import { useSync } from '@/composables/useSync';
import type { ValidationResult, MergeStrategy } from '@/types/sync';

interface ImportPreview {
  conversations: number;
  goals: number;
  achievements: number;
  settings: boolean;
  totalSize: number;
  backupDate: string;
  version: string;
  isCompatible: boolean;
  warnings: string[];
}

interface Props {
  isOpen: boolean;
}

defineProps<Props>();

const emit = defineEmits<{
  (e: 'close'): void;
  (e: 'imported'): void;
}>();

const { importData, validateBackup, importProgress, isImporting } = useSync();

// State
const selectedFile = ref<File | null>(null);
const preview = ref<ImportPreview | null>(null);
const previewError = ref<string | null>(null);
const isDragging = ref(false);
const mergeStrategy = ref<MergeStrategy>('merge_newest');

const canImport = computed(() => {
  return selectedFile.value && preview.value && preview.value.isCompatible;
});

const handleFileSelect = async (event: Event) => {
  const input = event.target as HTMLInputElement;
  if (input.files && input.files[0]) {
    await processFile(input.files[0]);
  }
};

const handleDrop = async (event: DragEvent) => {
  event.preventDefault();
  isDragging.value = false;

  if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
    await processFile(event.dataTransfer.files[0]);
  }
};

const handleDragOver = (event: DragEvent) => {
  event.preventDefault();
  isDragging.value = true;
};

const handleDragLeave = () => {
  isDragging.value = false;
};

const processFile = async (file: File) => {
  selectedFile.value = file;
  preview.value = null;
  previewError.value = null;

  try {
    const result: ValidationResult = await validateBackup(file);
    preview.value = {
      conversations: result.itemCounts.conversations,
      goals: result.itemCounts.goals,
      achievements: result.itemCounts.achievements,
      settings: result.itemCounts.settings > 0,
      totalSize: file.size,
      backupDate: new Date().toISOString(), // Will be updated from metadata
      version: result.version,
      isCompatible: result.compatible,
      warnings: result.warnings,
    };
  } catch (error) {
    previewError.value = error instanceof Error ? error.message : 'Failed to read file';
  }
};

const clearFile = () => {
  selectedFile.value = null;
  preview.value = null;
  previewError.value = null;
};

const handleImport = async () => {
  if (!selectedFile.value) return;

  try {
    await importData(selectedFile.value, { strategy: mergeStrategy.value });
    emit('imported');
    emit('close');
  } catch (error) {
    console.error('Import failed:', error);
  }
};

const handleBackdropClick = (event: MouseEvent) => {
  if (event.target === event.currentTarget && !isImporting.value) {
    emit('close');
  }
};

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

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
            aria-labelledby="import-dialog-title"
          >
            <!-- Header -->
            <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div class="flex items-center justify-between">
                <h2
                  id="import-dialog-title"
                  class="text-xl font-semibold text-gray-900 dark:text-white"
                >
                  Import Data
                </h2>
                <button
                  @click="emit('close')"
                  :disabled="isImporting"
                  class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <!-- Content -->
            <div class="px-6 py-4">
              <!-- File Upload -->
              <div
                v-if="!selectedFile"
                :class="[
                  'border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer',
                  isDragging
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
                ]"
                @drop="handleDrop"
                @dragover="handleDragOver"
                @dragleave="handleDragLeave"
                @click="($refs.fileInput as HTMLInputElement).click()"
              >
                <input
                  ref="fileInput"
                  type="file"
                  accept=".json"
                  class="hidden"
                  @change="handleFileSelect"
                />
                <svg class="w-12 h-12 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p class="text-gray-600 dark:text-gray-400 mb-2">
                  <span class="text-blue-600 dark:text-blue-400 font-medium">Click to upload</span>
                  or drag and drop
                </p>
                <p class="text-sm text-gray-500 dark:text-gray-500">JSON backup file</p>
              </div>

              <!-- File Selected -->
              <div v-else>
                <!-- File Info -->
                <div class="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg mb-4">
                  <div class="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                    <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div class="flex-1 min-w-0">
                    <p class="font-medium text-gray-900 dark:text-white truncate">
                      {{ selectedFile.name }}
                    </p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">
                      {{ formatSize(selectedFile.size) }}
                    </p>
                  </div>
                  <button
                    @click="clearFile"
                    :disabled="isImporting"
                    class="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
                  >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>

                <!-- Preview Error -->
                <div v-if="previewError" class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg mb-4">
                  <div class="flex items-center gap-2 text-red-700 dark:text-red-400">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span class="font-medium">{{ previewError }}</span>
                  </div>
                </div>

                <!-- Preview -->
                <div v-else-if="preview" class="space-y-4">
                  <!-- Backup Info -->
                  <div class="grid grid-cols-2 gap-3 text-sm">
                    <div class="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                      <span class="text-gray-500 dark:text-gray-400">Backup Date</span>
                      <p class="font-medium text-gray-900 dark:text-white">
                        {{ formatDate(preview.backupDate) }}
                      </p>
                    </div>
                    <div class="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                      <span class="text-gray-500 dark:text-gray-400">Version</span>
                      <p class="font-medium text-gray-900 dark:text-white">
                        {{ preview.version }}
                      </p>
                    </div>
                  </div>

                  <!-- Warnings -->
                  <div v-if="preview.warnings.length > 0" class="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                    <div class="flex items-start gap-2">
                      <svg class="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      <div class="flex-1">
                        <p class="font-medium text-yellow-800 dark:text-yellow-300 mb-1">Warnings</p>
                        <ul class="text-sm text-yellow-700 dark:text-yellow-400 space-y-1">
                          <li v-for="(warning, index) in preview.warnings" :key="index">
                            {{ warning }}
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <!-- Data Preview -->
                  <div>
                    <h3 class="font-medium text-gray-900 dark:text-white mb-2">Data to Import</h3>
                    <div class="grid grid-cols-2 gap-2">
                      <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700/50 rounded">
                        <svg class="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        <span class="text-sm text-gray-700 dark:text-gray-300">
                          {{ preview.conversations }} conversations
                        </span>
                      </div>
                      <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700/50 rounded">
                        <svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        <span class="text-sm text-gray-700 dark:text-gray-300">
                          {{ preview.goals }} goals
                        </span>
                      </div>
                      <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700/50 rounded">
                        <svg class="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                        </svg>
                        <span class="text-sm text-gray-700 dark:text-gray-300">
                          {{ preview.achievements }} achievements
                        </span>
                      </div>
                      <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700/50 rounded">
                        <svg class="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span class="text-sm text-gray-700 dark:text-gray-300">
                          {{ preview.settings ? 'Settings included' : 'No settings' }}
                        </span>
                      </div>
                    </div>
                  </div>

                  <!-- Merge Strategy -->
                  <div>
                    <h3 class="font-medium text-gray-900 dark:text-white mb-2">Merge Strategy</h3>
                    <div class="space-y-2">
                      <label class="flex items-start gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-600 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                        <input
                          v-model="mergeStrategy"
                          type="radio"
                          value="merge"
                          class="mt-0.5 w-4 h-4 text-blue-600 border-gray-300 dark:border-gray-600 focus:ring-blue-500"
                        />
                        <div>
                          <span class="font-medium text-gray-900 dark:text-white">Merge</span>
                          <p class="text-sm text-gray-500 dark:text-gray-400">
                            Keep existing data and add new items. Conflicts will be reported.
                          </p>
                        </div>
                      </label>
                      <label class="flex items-start gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-600 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                        <input
                          v-model="mergeStrategy"
                          type="radio"
                          value="skip_existing"
                          class="mt-0.5 w-4 h-4 text-blue-600 border-gray-300 dark:border-gray-600 focus:ring-blue-500"
                        />
                        <div>
                          <span class="font-medium text-gray-900 dark:text-white">Skip Existing</span>
                          <p class="text-sm text-gray-500 dark:text-gray-400">
                            Only import items that don't already exist locally.
                          </p>
                        </div>
                      </label>
                      <label class="flex items-start gap-3 p-3 rounded-lg border border-red-200 dark:border-red-800 cursor-pointer hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors">
                        <input
                          v-model="mergeStrategy"
                          type="radio"
                          value="replace"
                          class="mt-0.5 w-4 h-4 text-red-600 border-gray-300 dark:border-gray-600 focus:ring-red-500"
                        />
                        <div>
                          <span class="font-medium text-red-700 dark:text-red-400">Replace All</span>
                          <p class="text-sm text-red-600 dark:text-red-400">
                            Delete all existing data and replace with backup. This cannot be undone.
                          </p>
                        </div>
                      </label>
                    </div>
                  </div>
                </div>
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
              <div v-if="isImporting" class="px-6 py-2">
                <div class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                  <span>Importing...</span>
                  <span>{{ importProgress }}%</span>
                </div>
                <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    class="h-full bg-blue-600 rounded-full transition-all duration-300"
                    :style="{ width: `${importProgress}%` }"
                  />
                </div>
              </div>
            </Transition>

            <!-- Footer -->
            <div class="flex justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
              <button
                @click="emit('close')"
                :disabled="isImporting"
                class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                @click="handleImport"
                :disabled="!canImport || isImporting"
                class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:bg-blue-400 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <svg v-if="!isImporting" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                <svg v-else class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                {{ isImporting ? 'Importing...' : 'Import' }}
              </button>
            </div>
          </div>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>
