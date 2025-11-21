import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

describe('Conversation Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('should initialize with empty messages', async () => {
    const { useConversationStore } = await import('@/stores/conversation')
    const store = useConversationStore()

    expect(store.messages).toEqual([])
    expect(store.isLoading).toBe(false)
  })

  it('should add a user message', async () => {
    const { useConversationStore } = await import('@/stores/conversation')
    const store = useConversationStore()

    store.addMessage({
      id: '1',
      role: 'user',
      content: 'Hello',
      timestamp: new Date().toISOString(),
    })

    expect(store.messages).toHaveLength(1)
    expect(store.messages[0].content).toBe('Hello')
    expect(store.messages[0].role).toBe('user')
  })

  it('should clear messages', async () => {
    const { useConversationStore } = await import('@/stores/conversation')
    const store = useConversationStore()

    store.addMessage({
      id: '1',
      role: 'user',
      content: 'Hello',
      timestamp: new Date().toISOString(),
    })

    expect(store.messages).toHaveLength(1)

    store.clearMessages()

    expect(store.messages).toHaveLength(0)
  })

  it('should track loading state', async () => {
    const { useConversationStore } = await import('@/stores/conversation')
    const store = useConversationStore()

    expect(store.isLoading).toBe(false)

    store.setLoading(true)
    expect(store.isLoading).toBe(true)

    store.setLoading(false)
    expect(store.isLoading).toBe(false)
  })

  it('should handle errors', async () => {
    const { useConversationStore } = await import('@/stores/conversation')
    const store = useConversationStore()

    store.setError('Test error')

    expect(store.error).toBe('Test error')
  })
})
