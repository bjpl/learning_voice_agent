<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useUIStore } from '@/stores/ui'
import { useGoalsStore } from '@/stores/goals'

const route = useRoute()
const uiStore = useUIStore()
const goalsStore = useGoalsStore()

interface NavItem {
  name: string
  to: string
  icon: string
  badge?: number | string
}

const mainNavItems: NavItem[] = [
  { name: 'Dashboard', to: '/', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6' },
  { name: 'Conversation', to: '/conversation', icon: 'M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z' },
  { name: 'History', to: '/history', icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' }
]

const progressNavItems = computed<NavItem[]>(() => [
  { name: 'Analytics', to: '/analytics', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { name: 'Goals', to: '/goals', icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z', badge: goalsStore.activeGoals.length || undefined },
  { name: 'Achievements', to: '/achievements', icon: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z' }
])

const isActive = (path: string): boolean => {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}

function closeMobileSidebar(): void {
  if (uiStore.isMobile) {
    uiStore.setSidebarOpen(false)
  }
}
</script>

<template>
  <aside
    :class="[
      'fixed inset-y-0 left-0 z-50 flex w-64 flex-col bg-gray-900 transition-transform duration-300 lg:static lg:translate-x-0',
      uiStore.sidebarOpen ? 'translate-x-0' : '-translate-x-full',
      uiStore.sidebarCollapsed && !uiStore.isMobile && 'lg:w-20'
    ]"
  >
    <!-- Logo -->
    <div class="flex h-16 items-center justify-between border-b border-gray-800 px-4">
      <router-link to="/" class="flex items-center gap-2" @click="closeMobileSidebar">
        <svg class="h-8 w-8 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
        <span
          v-if="!uiStore.sidebarCollapsed"
          class="text-lg font-bold text-white"
        >
          VoiceLearn
        </span>
      </router-link>
      <button
        type="button"
        class="rounded-lg p-1.5 text-gray-400 hover:bg-gray-800 lg:hidden"
        @click="uiStore.setSidebarOpen(false)"
      >
        <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 overflow-y-auto px-3 py-4">
      <div class="space-y-1">
        <p
          v-if="!uiStore.sidebarCollapsed"
          class="px-3 text-xs font-semibold uppercase tracking-wider text-gray-500"
        >
          Main
        </p>
        <router-link
          v-for="item in mainNavItems"
          :key="item.name"
          :to="item.to"
          :class="[
            'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
            isActive(item.to)
              ? 'bg-blue-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          ]"
          @click="closeMobileSidebar"
        >
          <svg class="h-5 w-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" :d="item.icon" />
          </svg>
          <span v-if="!uiStore.sidebarCollapsed">{{ item.name }}</span>
        </router-link>
      </div>

      <div class="mt-8 space-y-1">
        <p
          v-if="!uiStore.sidebarCollapsed"
          class="px-3 text-xs font-semibold uppercase tracking-wider text-gray-500"
        >
          Progress
        </p>
        <router-link
          v-for="item in progressNavItems"
          :key="item.name"
          :to="item.to"
          :class="[
            'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
            isActive(item.to)
              ? 'bg-blue-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          ]"
          @click="closeMobileSidebar"
        >
          <svg class="h-5 w-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" :d="item.icon" />
          </svg>
          <span v-if="!uiStore.sidebarCollapsed" class="flex-1">{{ item.name }}</span>
          <span
            v-if="item.badge && !uiStore.sidebarCollapsed"
            class="rounded-full bg-blue-500/20 px-2 py-0.5 text-xs font-medium text-blue-400"
          >
            {{ item.badge }}
          </span>
        </router-link>
      </div>
    </nav>

    <!-- Collapse toggle (desktop only) -->
    <div class="hidden border-t border-gray-800 p-3 lg:block">
      <button
        type="button"
        class="flex w-full items-center justify-center rounded-lg py-2 text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
        @click="uiStore.toggleSidebar()"
      >
        <svg
          :class="['h-5 w-5 transition-transform', uiStore.sidebarCollapsed && 'rotate-180']"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
        </svg>
      </button>
    </div>
  </aside>

  <!-- Mobile overlay -->
  <Transition
    enter-active-class="transition-opacity duration-300"
    enter-from-class="opacity-0"
    enter-to-class="opacity-100"
    leave-active-class="transition-opacity duration-300"
    leave-from-class="opacity-100"
    leave-to-class="opacity-0"
  >
    <div
      v-if="uiStore.isMobile && uiStore.sidebarOpen"
      class="fixed inset-0 z-40 bg-black/50 lg:hidden"
      @click="uiStore.setSidebarOpen(false)"
    />
  </Transition>
</template>
