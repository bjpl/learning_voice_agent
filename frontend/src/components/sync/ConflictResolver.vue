<script setup lang="ts">
/**
 * ConflictResolver - Resolve sync conflicts between local and remote data
 *
 * PATTERN: Side-by-side comparison with resolution actions
 * WHY: Allow users to make informed decisions about data conflicts
 */
import { ref, computed } from 'vue';
import { useSync } from '@/composables/useSync';
import type { SyncConflict, ConflictResolution } from '@/types/sync';

interface Props {
  isOpen: boolean;
}

defineProps<Props>();

const emit = defineEmits<{
  (e: 'close'): void;
  (e: 'resolved'): void;
}>();

const { conflicts, resolveConflict } = useSync();

const selectedConflict = ref<SyncConflict | null>(null);

const conflictList = computed(() => conflicts.value);

const selectConflict = (conflict: SyncConflict) => {
  selectedConflict.value = conflict;
};

const handleResolve = (resolution: 'keep_local' | 'keep_remote' | 'merge') => {
  if (!selectedConflict.value) return;

  const conflictResolution: ConflictResolution = {
    conflictId: selectedConflict.value.id,
    resolution,
  };

  resolveConflict(conflictResolution);
  selectedConflict.value = null;

  if (conflicts.value.length === 0) {
    emit('resolved');
  }
};

const handleResolveAll = (resolution: 'keep_local' | 'keep_remote') => {
  conflicts.value.forEach((conflict) => {
    resolveConflict({
      conflictId: conflict.id,
      resolution,
    });
  });
  emit('resolved');
};

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
};


const getTypeIcon = (itemType: SyncConflict['itemType']): string => {
  switch (itemType) {
    case 'conversation':
    case 'message':
      return 'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z';
    case 'goal':
      return 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z';
    case 'achievement':
      return 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z';
    case 'setting':
      return 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z';
    case 'feedback':
      return 'M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z';
    default:
      return 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z';
  }
};

const formatDateString = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const handleBackdropClick = (event: MouseEvent) => {
  if (event.target === event.currentTarget) {
    emit('close');
  }
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
            class="w-full max-w-4xl max-h-[90vh] rounded-xl bg-white dark:bg-gray-800 shadow-xl overflow-hidden flex flex-col"
            role="dialog"
            aria-modal="true"
            aria-labelledby="conflict-dialog-title"
          >
            <!-- Header -->
            <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
              <div class="flex items-center justify-between">
                <div>
                  <h2
                    id="conflict-dialog-title"
                    class="text-xl font-semibold text-gray-900 dark:text-white"
                  >
                    Resolve Conflicts
                  </h2>
                  <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {{ conflictList.length }} conflict{{ conflictList.length !== 1 ? 's' : '' }} found
                  </p>
                </div>
                <button
                  @click="emit('close')"
                  class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <!-- Content -->
            <div class="flex-1 overflow-hidden flex min-h-0">
              <!-- Conflict List -->
              <div class="w-80 border-r border-gray-200 dark:border-gray-700 overflow-y-auto flex-shrink-0">
                <div class="p-2 space-y-1">
                  <button
                    v-for="conflict in conflictList"
                    :key="conflict.id"
                    @click="selectConflict(conflict)"
                    :class="[
                      'w-full text-left p-3 rounded-lg transition-colors',
                      selectedConflict?.id === conflict.id
                        ? 'bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-700/50 border border-transparent'
                    ]"
                  >
                    <div class="flex items-center gap-3">
                      <div class="p-2 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
                        <svg class="w-4 h-4 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d="getTypeIcon(conflict.itemType)" />
                        </svg>
                      </div>
                      <div class="flex-1 min-w-0">
                        <p class="font-medium text-gray-900 dark:text-white text-sm truncate capitalize">
                          {{ conflict.itemType }}
                        </p>
                        <p class="text-xs text-gray-500 dark:text-gray-400 truncate">
                          ID: {{ conflict.itemId }}
                        </p>
                      </div>
                    </div>
                  </button>
                </div>

                <!-- Empty State -->
                <div v-if="conflictList.length === 0" class="p-6 text-center">
                  <svg class="w-12 h-12 mx-auto text-green-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p class="text-gray-600 dark:text-gray-400">All conflicts resolved</p>
                </div>
              </div>

              <!-- Conflict Detail -->
              <div class="flex-1 overflow-y-auto">
                <div v-if="selectedConflict" class="p-6">
                  <!-- Comparison -->
                  <div class="grid grid-cols-2 gap-4 mb-6">
                    <!-- Local Value -->
                    <div class="rounded-lg border border-blue-200 dark:border-blue-800 overflow-hidden">
                      <div class="px-4 py-2 bg-blue-50 dark:bg-blue-900/30 border-b border-blue-200 dark:border-blue-800">
                        <div class="flex items-center justify-between">
                          <span class="font-medium text-blue-700 dark:text-blue-300">Local</span>
                          <span class="text-xs text-blue-600 dark:text-blue-400">
                            {{ formatDateString(selectedConflict.localVersion.modifiedAt) }}
                          </span>
                        </div>
                      </div>
                      <div class="p-4 bg-white dark:bg-gray-800">
                        <pre class="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap overflow-x-auto">{{ formatValue(selectedConflict.localVersion.data) }}</pre>
                      </div>
                    </div>

                    <!-- Remote Value -->
                    <div class="rounded-lg border border-purple-200 dark:border-purple-800 overflow-hidden">
                      <div class="px-4 py-2 bg-purple-50 dark:bg-purple-900/30 border-b border-purple-200 dark:border-purple-800">
                        <div class="flex items-center justify-between">
                          <span class="font-medium text-purple-700 dark:text-purple-300">Remote</span>
                          <span class="text-xs text-purple-600 dark:text-purple-400">
                            {{ formatDateString(selectedConflict.remoteVersion.modifiedAt) }}
                          </span>
                        </div>
                      </div>
                      <div class="p-4 bg-white dark:bg-gray-800">
                        <pre class="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap overflow-x-auto">{{ formatValue(selectedConflict.remoteVersion.data) }}</pre>
                      </div>
                    </div>
                  </div>

                  <!-- Resolution Actions -->
                  <div class="flex gap-3">
                    <button
                      @click="handleResolve('keep_local')"
                      class="flex-1 px-4 py-3 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors border border-blue-200 dark:border-blue-800"
                    >
                      <svg class="w-5 h-5 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Keep Local
                    </button>
                    <button
                      @click="handleResolve('keep_remote')"
                      class="flex-1 px-4 py-3 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 font-medium rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors border border-purple-200 dark:border-purple-800"
                    >
                      <svg class="w-5 h-5 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      Keep Remote
                    </button>
                  </div>
                </div>

                <!-- No Selection -->
                <div v-else class="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                  <div class="text-center">
                    <svg class="w-12 h-12 mx-auto text-gray-300 dark:text-gray-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                    </svg>
                    <p>Select a conflict to view details</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- Footer -->
            <div class="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 flex-shrink-0">
              <div class="flex gap-2">
                <button
                  @click="handleResolveAll('keep_local')"
                  :disabled="conflictList.length === 0"
                  class="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
                >
                  Keep All Local
                </button>
                <button
                  @click="handleResolveAll('keep_remote')"
                  :disabled="conflictList.length === 0"
                  class="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
                >
                  Keep All Remote
                </button>
              </div>
              <button
                @click="emit('close')"
                class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>
