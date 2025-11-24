<script setup lang="ts">
import { useUIStore } from '@/stores/ui'

const uiStore = useUIStore()

const version = '1.0.0'
</script>

<template>
  <footer class="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 px-4 py-3 lg:px-6">
    <div class="flex flex-col items-center justify-between gap-2 sm:flex-row">
      <div class="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
        <span>VoiceLearn v{{ version }}</span>
        <span class="hidden sm:inline">|</span>
        <a
          href="#"
          class="hidden hover:text-gray-700 dark:hover:text-gray-300 sm:inline"
        >
          Privacy Policy
        </a>
        <span class="hidden sm:inline">|</span>
        <a
          href="#"
          class="hidden hover:text-gray-700 dark:hover:text-gray-300 sm:inline"
        >
          Terms of Service
        </a>
      </div>

      <div class="flex items-center gap-3">
        <!-- Connection status -->
        <div class="flex items-center gap-1.5">
          <span
            :class="[
              'h-2 w-2 rounded-full',
              uiStore.isOnline ? 'bg-green-500' : 'bg-red-500'
            ]"
          />
          <span class="text-xs text-gray-500 dark:text-gray-400">
            {{ uiStore.isOnline ? 'Connected' : 'Offline' }}
          </span>
        </div>

        <!-- Help link -->
        <a
          href="#"
          class="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
        >
          <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="hidden sm:inline">Help</span>
        </a>
      </div>
    </div>
  </footer>
</template>
