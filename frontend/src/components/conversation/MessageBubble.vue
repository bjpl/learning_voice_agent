<script setup lang="ts">
import { computed } from 'vue'
import type { Message } from '@/stores/conversation'

interface Props {
  message: Message
  showTimestamp?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showTimestamp: true
})

const isUser = computed(() => props.message.role === 'user')
const isAssistant = computed(() => props.message.role === 'assistant')

const formattedTime = computed(() => {
  const date = new Date(props.message.timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
})

const bubbleClasses = computed(() => {
  if (isUser.value) {
    return 'bg-blue-600 text-white ml-auto rounded-br-none'
  }
  if (isAssistant.value) {
    return 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white mr-auto rounded-bl-none'
  }
  return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200 mx-auto text-center text-sm'
})
</script>

<template>
  <div
    :class="[
      'flex w-full',
      isUser ? 'justify-end' : isAssistant ? 'justify-start' : 'justify-center'
    ]"
  >
    <div class="flex max-w-[80%] flex-col gap-1">
      <!-- Avatar for assistant -->
      <div
        v-if="isAssistant"
        class="flex items-start gap-2"
      >
        <div class="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-blue-500">
          <svg class="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        </div>
        <div class="flex flex-col gap-1">
          <div :class="[bubbleClasses, 'rounded-2xl px-4 py-2.5']">
            <p class="whitespace-pre-wrap text-sm leading-relaxed">
              {{ message.content }}
            </p>
          </div>
          <span
            v-if="showTimestamp"
            class="text-xs text-gray-400 dark:text-gray-500"
          >
            {{ formattedTime }}
          </span>
        </div>
      </div>

      <!-- User message -->
      <div
        v-else-if="isUser"
        class="flex flex-col items-end gap-1"
      >
        <div :class="[bubbleClasses, 'rounded-2xl px-4 py-2.5']">
          <p class="whitespace-pre-wrap text-sm leading-relaxed">
            {{ message.content }}
          </p>
        </div>
        <span
          v-if="showTimestamp"
          class="text-xs text-gray-400 dark:text-gray-500"
        >
          {{ formattedTime }}
        </span>
      </div>

      <!-- System message -->
      <div
        v-else
        class="py-2"
      >
        <div :class="[bubbleClasses, 'rounded-lg px-4 py-2']">
          <p class="text-sm">{{ message.content }}</p>
        </div>
      </div>
    </div>
  </div>
</template>
