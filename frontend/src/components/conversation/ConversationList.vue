<script setup lang="ts">
import { ref, watch, nextTick, onMounted } from 'vue'
import { useConversationStore } from '@/stores/conversation'
import MessageBubble from './MessageBubble.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'

const conversationStore = useConversationStore()

const containerRef = ref<HTMLDivElement | null>(null)
const autoScroll = ref(true)

function scrollToBottom(smooth = true): void {
  if (!containerRef.value || !autoScroll.value) return
  nextTick(() => {
    containerRef.value?.scrollTo({
      top: containerRef.value.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto'
    })
  })
}

function handleScroll(): void {
  if (!containerRef.value) return
  const { scrollTop, scrollHeight, clientHeight } = containerRef.value
  const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
  autoScroll.value = isNearBottom
}

watch(
  () => conversationStore.currentMessages.length,
  () => {
    scrollToBottom()
  }
)

onMounted(() => {
  scrollToBottom(false)
})

defineExpose({ scrollToBottom })
</script>

<template>
  <div
    ref="containerRef"
    class="flex-1 overflow-y-auto px-4 py-6"
    @scroll="handleScroll"
  >
    <!-- Empty state -->
    <EmptyState
      v-if="conversationStore.currentMessages.length === 0 && !conversationStore.isLoading"
      icon="chat"
      title="Start a conversation"
      description="Click the microphone button below to begin speaking, or type a message to get started."
    />

    <!-- Messages -->
    <div
      v-else
      class="mx-auto max-w-3xl space-y-4"
    >
      <TransitionGroup
        name="message"
        tag="div"
        class="space-y-4"
      >
        <MessageBubble
          v-for="message in conversationStore.currentMessages"
          :key="message.id"
          :message="message"
        />
      </TransitionGroup>

      <!-- Processing indicator -->
      <div
        v-if="conversationStore.isProcessing"
        class="flex items-start gap-2"
      >
        <div class="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-blue-500">
          <svg class="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        </div>
        <div class="rounded-2xl rounded-bl-none bg-gray-100 dark:bg-gray-800 px-4 py-3">
          <div class="flex items-center gap-1">
            <span class="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]" />
            <span class="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]" />
            <span class="h-2 w-2 animate-bounce rounded-full bg-gray-400" />
          </div>
        </div>
      </div>
    </div>

    <!-- Loading state -->
    <div
      v-if="conversationStore.isLoading"
      class="flex items-center justify-center py-8"
    >
      <LoadingSpinner label="Loading conversation..." />
    </div>

    <!-- Scroll to bottom button -->
    <Transition
      enter-active-class="transition-opacity duration-200"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition-opacity duration-200"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <button
        v-if="!autoScroll && conversationStore.currentMessages.length > 0"
        type="button"
        class="fixed bottom-24 right-8 rounded-full bg-white dark:bg-gray-800 p-2 shadow-lg ring-1 ring-gray-200 dark:ring-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        @click="autoScroll = true; scrollToBottom()"
      >
        <svg class="h-5 w-5 text-gray-600 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </button>
    </Transition>
  </div>
</template>

<style scoped>
.message-enter-active {
  transition: all 0.3s ease-out;
}

.message-leave-active {
  transition: all 0.2s ease-in;
}

.message-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.message-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}
</style>
