<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useConversationStore, type Conversation } from '@/stores/conversation'
import { useUIStore } from '@/stores/ui'
import SearchInput from '@/components/common/SearchInput.vue'
import EmptyState from '@/components/common/EmptyState.vue'

const router = useRouter()
const conversationStore = useConversationStore()
const uiStore = useUIStore()

const searchQuery = ref('')
const sortBy = ref<'date' | 'messages'>('date')

const filteredConversations = computed(() => {
  let result = conversationStore.sortedConversations

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(c =>
      c.title.toLowerCase().includes(query) ||
      c.messages.some(m => m.content.toLowerCase().includes(query))
    )
  }

  if (sortBy.value === 'messages') {
    result = [...result].sort((a, b) => b.messages.length - a.messages.length)
  }

  return result
})

function openConversation(conversation: Conversation): void {
  conversationStore.setCurrentConversation(conversation.id)
  router.push('/conversation')
}

async function deleteConversation(conversation: Conversation): Promise<void> {
  const confirmed = await uiStore.confirm({
    title: 'Delete Conversation',
    message: `Are you sure you want to delete "${conversation.title}"? This action cannot be undone.`,
    confirmText: 'Delete',
    type: 'danger'
  })

  if (confirmed) {
    conversationStore.deleteConversation(conversation.id)
    uiStore.showToast({
      type: 'success',
      title: 'Conversation deleted',
      message: 'The conversation has been permanently removed.'
    })
  }
}

function formatDate(date: Date): string {
  const now = new Date()
  const d = new Date(date)
  const diffMs = now.getTime() - d.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) {
    return 'Today'
  } else if (diffDays === 1) {
    return 'Yesterday'
  } else if (diffDays < 7) {
    return `${diffDays} days ago`
  } else {
    return d.toLocaleDateString()
  }
}

function formatDuration(conversation: Conversation): string {
  if (conversation.messages.length < 2) return 'N/A'
  const first = new Date(conversation.messages[0].timestamp)
  const last = new Date(conversation.messages[conversation.messages.length - 1].timestamp)
  const diffMs = last.getTime() - first.getTime()
  const minutes = Math.round(diffMs / (1000 * 60))
  return minutes < 1 ? '<1 min' : `${minutes} min`
}
</script>

<template>
  <div class="p-4 lg:p-6">
    <div class="mx-auto max-w-5xl">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">Conversation History</h1>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Review and continue your past conversations
        </p>
      </div>

      <!-- Search and filters -->
      <div class="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div class="w-full sm:max-w-sm">
          <SearchInput
            v-model="searchQuery"
            placeholder="Search conversations..."
          />
        </div>
        <div class="flex items-center gap-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Sort by:</span>
          <select
            v-model="sortBy"
            class="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="date">Date</option>
            <option value="messages">Messages</option>
          </select>
        </div>
      </div>

      <!-- Stats -->
      <div class="mb-6 flex gap-4">
        <div class="rounded-lg bg-white dark:bg-gray-800 px-4 py-3 shadow-sm ring-1 ring-gray-200 dark:ring-gray-700">
          <p class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ conversationStore.conversationCount }}
          </p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Conversations</p>
        </div>
        <div class="rounded-lg bg-white dark:bg-gray-800 px-4 py-3 shadow-sm ring-1 ring-gray-200 dark:ring-gray-700">
          <p class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ conversationStore.conversations.reduce((sum, c) => sum + c.messages.length, 0) }}
          </p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Messages</p>
        </div>
      </div>

      <!-- Conversations list -->
      <EmptyState
        v-if="filteredConversations.length === 0"
        icon="chat"
        :title="searchQuery ? 'No results found' : 'No conversations yet'"
        :description="searchQuery ? 'Try adjusting your search query' : 'Start a new conversation to see it here'"
        :action-label="searchQuery ? undefined : 'Start Conversation'"
        @action="router.push('/conversation')"
      />

      <div v-else class="space-y-3">
        <div
          v-for="conversation in filteredConversations"
          :key="conversation.id"
          class="group rounded-xl bg-white dark:bg-gray-800 p-4 shadow-sm ring-1 ring-gray-200 dark:ring-gray-700 hover:ring-blue-300 dark:hover:ring-blue-600 transition-all cursor-pointer"
          @click="openConversation(conversation)"
        >
          <div class="flex items-start justify-between">
            <div class="flex items-start gap-4 min-w-0 flex-1">
              <div class="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-blue-100 dark:bg-blue-900/30">
                <svg class="h-6 w-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <div class="min-w-0 flex-1">
                <h3 class="font-semibold text-gray-900 dark:text-white truncate">
                  {{ conversation.title }}
                </h3>
                <p
                  v-if="conversation.messages.length > 0"
                  class="mt-1 text-sm text-gray-500 dark:text-gray-400 line-clamp-2"
                >
                  {{ conversation.messages[conversation.messages.length - 1].content }}
                </p>
                <div class="mt-2 flex items-center gap-4 text-xs text-gray-400 dark:text-gray-500">
                  <span>{{ formatDate(conversation.updatedAt) }}</span>
                  <span>{{ conversation.messages.length }} messages</span>
                  <span>{{ formatDuration(conversation) }}</span>
                </div>
              </div>
            </div>
            <div class="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                type="button"
                class="rounded-lg p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors"
                title="Continue conversation"
                @click.stop="openConversation(conversation)"
              >
                <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
              <button
                type="button"
                class="rounded-lg p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                title="Delete conversation"
                @click.stop="deleteConversation(conversation)"
              >
                <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          </div>

          <!-- Tags if available -->
          <div v-if="conversation.tags && conversation.tags.length > 0" class="mt-3 flex flex-wrap gap-2">
            <span
              v-for="tag in conversation.tags"
              :key="tag"
              class="rounded-full bg-gray-100 dark:bg-gray-700 px-2 py-0.5 text-xs text-gray-600 dark:text-gray-400"
            >
              {{ tag }}
            </span>
          </div>
        </div>
      </div>

      <!-- Load more (placeholder for pagination) -->
      <div v-if="filteredConversations.length >= 10" class="mt-6 flex justify-center">
        <button
          type="button"
          class="rounded-lg px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          Load more conversations
        </button>
      </div>
    </div>
  </div>
</template>
