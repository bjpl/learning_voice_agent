import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  url: string
  readyState: number = MockWebSocket.CONNECTING
  onopen: ((e: Event) => void) | null = null
  onclose: ((e: CloseEvent) => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: ((e: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      if (this.onopen) {
        this.onopen(new Event('open'))
      }
    }, 10)
  }

  send = vi.fn()
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED
    if (this.onclose) {
      this.onclose(new CloseEvent('close'))
    }
  })

  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }))
    }
  }

  simulateError() {
    if (this.onerror) {
      this.onerror(new Event('error'))
    }
  }
}

global.WebSocket = MockWebSocket as any

describe('useWebSocket Composable', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('should connect to WebSocket', async () => {
    const { useWebSocket } = await import('@/composables/useWebSocket')
    const ws = useWebSocket('ws://localhost:8000/ws')

    ws.connect()

    expect(ws.status.value).toBe('connecting')

    await vi.advanceTimersByTimeAsync(20)

    expect(ws.status.value).toBe('connected')
    expect(ws.isConnected.value).toBe(true)
  })

  it('should send messages', async () => {
    const { useWebSocket } = await import('@/composables/useWebSocket')
    const ws = useWebSocket('ws://localhost:8000/ws')

    ws.connect()
    await vi.advanceTimersByTimeAsync(20)

    ws.send({ type: 'test', data: 'hello' })

    expect(ws.socket.value?.send).toHaveBeenCalled()
  })

  it('should receive messages', async () => {
    const { useWebSocket } = await import('@/composables/useWebSocket')
    const ws = useWebSocket('ws://localhost:8000/ws')

    ws.connect()
    await vi.advanceTimersByTimeAsync(20)

    const mockWs = ws.socket.value as unknown as MockWebSocket
    mockWs.simulateMessage({ type: 'response', data: 'world' })

    expect(ws.lastMessage.value).toEqual({ type: 'response', data: 'world' })
  })

  it('should disconnect', async () => {
    const { useWebSocket } = await import('@/composables/useWebSocket')
    const ws = useWebSocket('ws://localhost:8000/ws')

    ws.connect()
    await vi.advanceTimersByTimeAsync(20)

    ws.disconnect()

    expect(ws.socket.value?.close).toHaveBeenCalled()
  })

  it('should reconnect on error', async () => {
    const { useWebSocket } = await import('@/composables/useWebSocket')
    const ws = useWebSocket('ws://localhost:8000/ws', {
      autoReconnect: true,
      reconnectInterval: 1000,
    })

    ws.connect()
    await vi.advanceTimersByTimeAsync(20)

    const mockWs = ws.socket.value as unknown as MockWebSocket
    mockWs.simulateError()
    mockWs.close()

    // Wait for reconnect attempt
    await vi.advanceTimersByTimeAsync(1100)

    expect(ws.reconnectAttempts.value).toBeGreaterThanOrEqual(1)
  })

  it('should respect max reconnect attempts', async () => {
    const { useWebSocket } = await import('@/composables/useWebSocket')
    const ws = useWebSocket('ws://localhost:8000/ws', {
      autoReconnect: true,
      reconnectInterval: 100,
      maxReconnectAttempts: 3,
    })

    ws.connect()
    await vi.advanceTimersByTimeAsync(20)

    // Force close multiple times
    for (let i = 0; i < 5; i++) {
      const mockWs = ws.socket.value as unknown as MockWebSocket
      if (mockWs) {
        mockWs.close()
      }
      await vi.advanceTimersByTimeAsync(200)
    }

    expect(ws.reconnectAttempts.value).toBeLessThanOrEqual(3)
  })
})
