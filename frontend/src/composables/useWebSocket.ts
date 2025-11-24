/**
 * WebSocket Connection Composable
 * Provides WebSocket connection management with reconnect logic,
 * heartbeat mechanism, and event handling
 */

import { ref, computed, onUnmounted, type Ref } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting' | 'error';

export interface WebSocketMessage<T = unknown> {
  type: string;
  payload: T;
  timestamp: number;
  id?: string;
}

export interface WebSocketOptions {
  url: string;
  protocols?: string | string[];
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  reconnectDelayMax?: number;
  heartbeatInterval?: number;
  heartbeatMessage?: string | object;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (data: unknown) => void;
  onReconnect?: (attempt: number) => void;
}

export interface UseWebSocketReturn {
  // State
  status: Ref<WebSocketStatus>;
  isConnected: Ref<boolean>;
  lastMessage: Ref<unknown>;
  error: Ref<string | null>;
  reconnectAttempt: Ref<number>;

  // Methods
  connect: () => void;
  disconnect: (code?: number, reason?: string) => void;
  send: <T>(data: T) => boolean;
  sendMessage: <T>(type: string, payload: T) => boolean;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_RECONNECT_ATTEMPTS = 5;
const DEFAULT_RECONNECT_DELAY = 1000;
const DEFAULT_RECONNECT_DELAY_MAX = 30000;
const DEFAULT_HEARTBEAT_INTERVAL = 30000;
const DEFAULT_HEARTBEAT_MESSAGE = { type: 'ping' };

// ============================================================================
// Composable Implementation
// ============================================================================

export function useWebSocket(options: WebSocketOptions): UseWebSocketReturn {
  const {
    url,
    protocols,
    reconnect = true,
    reconnectAttempts = DEFAULT_RECONNECT_ATTEMPTS,
    reconnectDelay = DEFAULT_RECONNECT_DELAY,
    reconnectDelayMax = DEFAULT_RECONNECT_DELAY_MAX,
    heartbeatInterval = DEFAULT_HEARTBEAT_INTERVAL,
    heartbeatMessage = DEFAULT_HEARTBEAT_MESSAGE,
    onOpen,
    onClose,
    onError,
    onMessage,
    onReconnect
  } = options;

  // Reactive state
  const status = ref<WebSocketStatus>('disconnected');
  const lastMessage = ref<unknown>(null);
  const error = ref<string | null>(null);
  const reconnectAttempt = ref(0);

  // Computed
  const isConnected = computed(() => status.value === 'connected');

  // Internal state
  let socket: WebSocket | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  let heartbeatTimeout: ReturnType<typeof setTimeout> | null = null;
  let heartbeatResponseTimeout: ReturnType<typeof setTimeout> | null = null;
  let isManualClose = false;

  // ============================================================================
  // Heartbeat Logic
  // ============================================================================

  function startHeartbeat(): void {
    stopHeartbeat();

    if (heartbeatInterval <= 0) return;

    heartbeatTimeout = setInterval(() => {
      if (socket?.readyState === WebSocket.OPEN) {
        const message = typeof heartbeatMessage === 'string'
          ? heartbeatMessage
          : JSON.stringify(heartbeatMessage);

        socket.send(message);

        // Set timeout for heartbeat response
        heartbeatResponseTimeout = setTimeout(() => {
          console.warn('WebSocket heartbeat timeout - reconnecting');
          socket?.close();
        }, 10000);
      }
    }, heartbeatInterval);
  }

  function stopHeartbeat(): void {
    if (heartbeatTimeout) {
      clearInterval(heartbeatTimeout);
      heartbeatTimeout = null;
    }
    if (heartbeatResponseTimeout) {
      clearTimeout(heartbeatResponseTimeout);
      heartbeatResponseTimeout = null;
    }
  }

  function resetHeartbeatResponse(): void {
    if (heartbeatResponseTimeout) {
      clearTimeout(heartbeatResponseTimeout);
      heartbeatResponseTimeout = null;
    }
  }

  // ============================================================================
  // Reconnection Logic
  // ============================================================================

  function getReconnectDelay(): number {
    // Exponential backoff with jitter
    const baseDelay = Math.min(
      reconnectDelay * Math.pow(2, reconnectAttempt.value),
      reconnectDelayMax
    );
    const jitter = baseDelay * 0.2 * Math.random();
    return baseDelay + jitter;
  }

  function scheduleReconnect(): void {
    if (!reconnect || reconnectAttempt.value >= reconnectAttempts) {
      status.value = 'error';
      error.value = `Failed to reconnect after ${reconnectAttempts} attempts`;
      return;
    }

    status.value = 'reconnecting';
    reconnectAttempt.value++;

    const delay = getReconnectDelay();
    console.log(`WebSocket reconnecting in ${delay}ms (attempt ${reconnectAttempt.value}/${reconnectAttempts})`);

    onReconnect?.(reconnectAttempt.value);

    reconnectTimeout = setTimeout(() => {
      connect();
    }, delay);
  }

  function cancelReconnect(): void {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  }

  // ============================================================================
  // Connection Methods
  // ============================================================================

  function connect(): void {
    // Cleanup existing connection
    if (socket) {
      socket.onopen = null;
      socket.onclose = null;
      socket.onerror = null;
      socket.onmessage = null;

      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    }

    isManualClose = false;
    error.value = null;
    status.value = 'connecting';

    try {
      socket = protocols ? new WebSocket(url, protocols) : new WebSocket(url);

      socket.onopen = (event: Event) => {
        status.value = 'connected';
        reconnectAttempt.value = 0;
        error.value = null;

        startHeartbeat();
        onOpen?.(event);
      };

      socket.onclose = (event: CloseEvent) => {
        stopHeartbeat();
        status.value = 'disconnected';

        onClose?.(event);

        // Auto-reconnect if not manually closed
        if (!isManualClose && reconnect) {
          scheduleReconnect();
        }
      };

      socket.onerror = (event: Event) => {
        error.value = 'WebSocket connection error';
        onError?.(event);
      };

      socket.onmessage = (event: MessageEvent) => {
        resetHeartbeatResponse();

        let data: unknown;
        try {
          data = JSON.parse(event.data);
        } catch {
          data = event.data;
        }

        // Handle pong messages
        if (typeof data === 'object' && data !== null && 'type' in data) {
          const messageData = data as { type: string };
          if (messageData.type === 'pong') {
            return;
          }
        }

        lastMessage.value = data;
        onMessage?.(data);
      };

    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to create WebSocket connection';
      status.value = 'error';
    }
  }

  function disconnect(code = 1000, reason = 'Client disconnect'): void {
    isManualClose = true;
    cancelReconnect();
    stopHeartbeat();

    if (socket) {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close(code, reason);
      }
      socket = null;
    }

    status.value = 'disconnected';
    reconnectAttempt.value = 0;
  }

  // ============================================================================
  // Messaging Methods
  // ============================================================================

  function send<T>(data: T): boolean {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket is not connected');
      return false;
    }

    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      socket.send(message);
      return true;
    } catch (err) {
      console.error('Failed to send WebSocket message:', err);
      return false;
    }
  }

  function sendMessage<T>(type: string, payload: T): boolean {
    const message: WebSocketMessage<T> = {
      type,
      payload,
      timestamp: Date.now(),
      id: generateMessageId()
    };
    return send(message);
  }

  function generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  // ============================================================================
  // Cleanup
  // ============================================================================

  onUnmounted(() => {
    disconnect();
  });

  // ============================================================================
  // Return
  // ============================================================================

  return {
    // State
    status,
    isConnected,
    lastMessage,
    error,
    reconnectAttempt,

    // Methods
    connect,
    disconnect,
    send,
    sendMessage
  };
}

// ============================================================================
// Factory Function for Common Use Cases
// ============================================================================

export function createVoiceWebSocket(baseUrl: string) {
  return useWebSocket({
    url: `${baseUrl}/ws/voice`,
    heartbeatInterval: 25000,
    reconnectAttempts: 10,
    heartbeatMessage: { type: 'ping', service: 'voice' }
  });
}

export function createTranscriptWebSocket(baseUrl: string) {
  return useWebSocket({
    url: `${baseUrl}/ws/transcript`,
    heartbeatInterval: 20000,
    reconnectAttempts: 5,
    heartbeatMessage: { type: 'ping', service: 'transcript' }
  });
}

export default useWebSocket;
