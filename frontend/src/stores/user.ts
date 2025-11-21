import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface User {
  id: string
  name: string
  email: string
  avatar?: string
  preferences: UserPreferences
  createdAt: Date
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system'
  language: string
  voiceEnabled: boolean
  notificationsEnabled: boolean
  autoSaveEnabled: boolean
  speakingRate: number
  voicePitch: number
}

export const useUserStore = defineStore('user', () => {
  const user = ref<User | null>(null)
  const isAuthenticated = ref(false)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const displayName = computed(() => user.value?.name || 'Guest')
  const initials = computed(() => {
    if (!user.value?.name) return 'G'
    return user.value.name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  })

  const defaultPreferences: UserPreferences = {
    theme: 'system',
    language: 'en',
    voiceEnabled: true,
    notificationsEnabled: true,
    autoSaveEnabled: true,
    speakingRate: 1.0,
    voicePitch: 1.0
  }

  async function login(email: string, password: string): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      // API call would go here
      user.value = {
        id: '1',
        name: 'Demo User',
        email,
        preferences: { ...defaultPreferences },
        createdAt: new Date()
      }
      isAuthenticated.value = true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Login failed'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  async function logout(): Promise<void> {
    user.value = null
    isAuthenticated.value = false
  }

  async function updatePreferences(prefs: Partial<UserPreferences>): Promise<void> {
    if (!user.value) return
    user.value.preferences = { ...user.value.preferences, ...prefs }
  }

  function setUser(newUser: User): void {
    user.value = newUser
    isAuthenticated.value = true
  }

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    displayName,
    initials,
    login,
    logout,
    updatePreferences,
    setUser
  }
})
