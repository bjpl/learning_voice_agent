<script setup lang="ts">
import { ref, computed } from 'vue'
import { useUserStore } from '@/stores/user'
import { useUIStore } from '@/stores/ui'

const userStore = useUserStore()
const uiStore = useUIStore()

const showUserMenu = ref(false)
const showNotifications = ref(false)

const notifications = ref([
  { id: '1', title: 'Goal completed!', message: 'You reached your daily speaking goal', time: '5m ago', read: false },
  { id: '2', title: 'New achievement', message: 'You unlocked "Chatterbox"', time: '1h ago', read: false },
  { id: '3', title: 'Weekly report ready', message: 'Check your progress for this week', time: '2h ago', read: true }
])

const unreadCount = computed(() => notifications.value.filter(n => !n.read).length)

function toggleUserMenu(): void {
  showUserMenu.value = !showUserMenu.value
  if (showUserMenu.value) showNotifications.value = false
}

function toggleNotifications(): void {
  showNotifications.value = !showNotifications.value
  if (showNotifications.value) showUserMenu.value = false
}

function markAllRead(): void {
  notifications.value.forEach(n => n.read = true)
}

function handleLogout(): void {
  userStore.logout()
  showUserMenu.value = false
}

function closeMenus(): void {
  showUserMenu.value = false
  showNotifications.value = false
}
</script>

<template>
  <header class="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 px-4 lg:px-6">
    <div class="flex items-center gap-4">
      <button
        type="button"
        class="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 lg:hidden"
        @click="uiStore.toggleSidebar()"
      >
        <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <div class="hidden lg:flex items-center gap-2">
        <svg class="h-8 w-8 text-blue-600" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
        <span class="text-xl font-bold text-gray-900 dark:text-white">VoiceLearn</span>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <!-- Dark mode toggle -->
      <button
        type="button"
        class="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        @click="uiStore.setDarkMode(!uiStore.isDarkMode)"
      >
        <svg v-if="uiStore.isDarkMode" class="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd" />
        </svg>
        <svg v-else class="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
        </svg>
      </button>

      <!-- Notifications -->
      <div class="relative">
        <button
          type="button"
          class="relative rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          @click="toggleNotifications"
        >
          <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
          </svg>
          <span v-if="unreadCount > 0" class="absolute top-1 right-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white">
            {{ unreadCount }}
          </span>
        </button>
        <Transition
          enter-active-class="transition ease-out duration-100"
          enter-from-class="transform opacity-0 scale-95"
          enter-to-class="transform opacity-100 scale-100"
          leave-active-class="transition ease-in duration-75"
          leave-from-class="transform opacity-100 scale-100"
          leave-to-class="transform opacity-0 scale-95"
        >
          <div
            v-if="showNotifications"
            class="absolute right-0 mt-2 w-80 origin-top-right rounded-xl bg-white dark:bg-gray-800 shadow-lg ring-1 ring-black ring-opacity-5"
          >
            <div class="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 px-4 py-3">
              <h3 class="text-sm font-semibold text-gray-900 dark:text-white">Notifications</h3>
              <button
                type="button"
                class="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400"
                @click="markAllRead"
              >
                Mark all read
              </button>
            </div>
            <div class="max-h-72 overflow-y-auto">
              <div
                v-for="notification in notifications"
                :key="notification.id"
                :class="[
                  'px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer',
                  !notification.read && 'bg-blue-50 dark:bg-blue-900/20'
                ]"
              >
                <p class="text-sm font-medium text-gray-900 dark:text-white">{{ notification.title }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">{{ notification.message }}</p>
                <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">{{ notification.time }}</p>
              </div>
            </div>
          </div>
        </Transition>
      </div>

      <!-- User menu -->
      <div class="relative">
        <button
          type="button"
          class="flex items-center gap-2 rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          @click="toggleUserMenu"
        >
          <div class="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-sm font-medium text-white">
            {{ userStore.initials }}
          </div>
          <span class="hidden text-sm font-medium text-gray-700 dark:text-gray-300 md:block">
            {{ userStore.displayName }}
          </span>
          <svg class="hidden h-4 w-4 text-gray-400 md:block" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
          </svg>
        </button>
        <Transition
          enter-active-class="transition ease-out duration-100"
          enter-from-class="transform opacity-0 scale-95"
          enter-to-class="transform opacity-100 scale-100"
          leave-active-class="transition ease-in duration-75"
          leave-from-class="transform opacity-100 scale-100"
          leave-to-class="transform opacity-0 scale-95"
        >
          <div
            v-if="showUserMenu"
            class="absolute right-0 mt-2 w-56 origin-top-right rounded-xl bg-white dark:bg-gray-800 shadow-lg ring-1 ring-black ring-opacity-5"
          >
            <div class="border-b border-gray-200 dark:border-gray-700 px-4 py-3">
              <p class="text-sm font-medium text-gray-900 dark:text-white">{{ userStore.displayName }}</p>
              <p class="text-xs text-gray-500 dark:text-gray-400 truncate">{{ userStore.user?.email }}</p>
            </div>
            <div class="py-1">
              <router-link
                to="/settings"
                class="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                @click="closeMenus"
              >
                <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Settings
              </router-link>
              <button
                type="button"
                class="flex w-full items-center gap-2 px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                @click="handleLogout"
              >
                <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign out
              </button>
            </div>
          </div>
        </Transition>
      </div>
    </div>
  </header>
</template>
