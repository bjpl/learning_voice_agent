import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

// Mock uuid
vi.mock('uuid', () => ({
  v4: () => 'test-uuid-1234',
}))

describe('Session Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('should initialize with default state', async () => {
    const { useUserStore } = await import('@/stores/user')
    const store = useUserStore()

    expect(store.isAuthenticated).toBe(false)
    expect(store.sessionId).toBe('')
  })

  it('should create a new session', async () => {
    const { useUserStore } = await import('@/stores/user')
    const store = useUserStore()

    store.initSession()

    expect(store.sessionId).toBeTruthy()
    expect(store.isAuthenticated).toBe(true)
  })

  it('should update preferences', async () => {
    const { useUserStore } = await import('@/stores/user')
    const store = useUserStore()

    store.updatePreferences({ theme: 'dark' })

    expect(store.preferences.theme).toBe('dark')
  })

  it('should toggle dark mode', async () => {
    const { useUserStore } = await import('@/stores/user')
    const store = useUserStore()

    const initialDark = store.preferences.theme === 'dark'
    store.toggleDarkMode()

    expect(store.preferences.theme === 'dark').toBe(!initialDark)
  })

  it('should clear session on logout', async () => {
    const { useUserStore } = await import('@/stores/user')
    const store = useUserStore()

    store.initSession()
    expect(store.isAuthenticated).toBe(true)

    store.logout()
    expect(store.isAuthenticated).toBe(false)
    expect(store.sessionId).toBe('')
  })
})
