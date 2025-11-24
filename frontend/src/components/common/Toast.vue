<script setup lang="ts">
import { computed } from 'vue'
import type { Toast } from '@/stores/ui'

interface Props {
  toast: Toast
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'dismiss', id: string): void
}>()

const styles = computed(() => {
  switch (props.toast.type) {
    case 'success':
      return {
        bg: 'bg-green-50 dark:bg-green-900/30',
        border: 'border-green-200 dark:border-green-800',
        icon: 'text-green-500',
        title: 'text-green-800 dark:text-green-200',
        message: 'text-green-700 dark:text-green-300'
      }
    case 'error':
      return {
        bg: 'bg-red-50 dark:bg-red-900/30',
        border: 'border-red-200 dark:border-red-800',
        icon: 'text-red-500',
        title: 'text-red-800 dark:text-red-200',
        message: 'text-red-700 dark:text-red-300'
      }
    case 'warning':
      return {
        bg: 'bg-yellow-50 dark:bg-yellow-900/30',
        border: 'border-yellow-200 dark:border-yellow-800',
        icon: 'text-yellow-500',
        title: 'text-yellow-800 dark:text-yellow-200',
        message: 'text-yellow-700 dark:text-yellow-300'
      }
    default:
      return {
        bg: 'bg-blue-50 dark:bg-blue-900/30',
        border: 'border-blue-200 dark:border-blue-800',
        icon: 'text-blue-500',
        title: 'text-blue-800 dark:text-blue-200',
        message: 'text-blue-700 dark:text-blue-300'
      }
  }
})

const iconPath = computed(() => {
  switch (props.toast.type) {
    case 'success':
      return 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'
    case 'error':
      return 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z'
    case 'warning':
      return 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'
    default:
      return 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
  }
})
</script>

<template>
  <div
    :class="[styles.bg, styles.border, 'pointer-events-auto w-full max-w-sm overflow-hidden rounded-lg border shadow-lg']"
    role="alert"
  >
    <div class="p-4">
      <div class="flex items-start gap-3">
        <div :class="[styles.icon, 'flex-shrink-0']">
          <svg
            class="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              :d="iconPath"
            />
          </svg>
        </div>
        <div class="flex-1 min-w-0">
          <p :class="[styles.title, 'text-sm font-medium']">
            {{ toast.title }}
          </p>
          <p
            v-if="toast.message"
            :class="[styles.message, 'mt-1 text-sm']"
          >
            {{ toast.message }}
          </p>
        </div>
        <button
          v-if="toast.dismissible"
          type="button"
          class="flex-shrink-0 rounded-md p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-300 transition-colors"
          @click="emit('dismiss', toast.id)"
        >
          <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fill-rule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clip-rule="evenodd"
            />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>
