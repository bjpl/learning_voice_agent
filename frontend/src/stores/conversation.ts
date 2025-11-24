/**
 * Conversation store using Pinia Composition API
 * Integrates with API services for backend communication
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import conversationService from '@/services/conversation';
import type {
  Message,
  ConversationResponse,
  ConversationContext,
  ConversationSession,
  CorrectionItem,
} from '@/types';

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  summary?: string;
  tags?: string[];
}

export interface ConversationState {
  isLoading: boolean;
  isSending: boolean;
  error: string | null;
}

export const useConversationStore = defineStore('conversation', () => {
  // State
  const conversations = ref<Conversation[]>([]);
  const currentConversation = ref<Conversation | null>(null);
  const messages = ref<Message[]>([]);
  const sessions = ref<ConversationSession[]>([]);
  const activeConversationId = ref<string | null>(null);
  const isLoading = ref(false);
  const isRecording = ref(false);
  const isProcessing = ref(false);
  const isSending = ref(false);
  const error = ref<string | null>(null);
  const currentContext = ref<ConversationContext | null>(null);
  const lastResponse = ref<ConversationResponse | null>(null);
  const corrections = ref<CorrectionItem[]>([]);

  // Computed
  const currentMessages = computed(() => currentConversation.value?.messages || messages.value);
  const conversationCount = computed(() => conversations.value.length);
  const messageCount = computed(() => messages.value.length);
  const hasActiveConversation = computed(() => currentConversation.value !== null || messages.value.length > 0);

  const lastMessage = computed(() => {
    const msgs = currentMessages.value;
    return msgs.length > 0 ? msgs[msgs.length - 1] : null;
  });

  const userMessages = computed(() => {
    return currentMessages.value.filter((m) => m.role === 'user');
  });

  const assistantMessages = computed(() => {
    return currentMessages.value.filter((m) => m.role === 'assistant');
  });

  const sortedConversations = computed(() =>
    [...conversations.value].sort((a, b) =>
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    )
  );

  const totalCorrections = computed(() => corrections.value.length);

  const correctionsByType = computed(() => {
    const grouped: Record<string, CorrectionItem[]> = {};
    corrections.value.forEach((c) => {
      if (!grouped[c.type]) {
        grouped[c.type] = [];
      }
      grouped[c.type].push(c);
    });
    return grouped;
  });

  // Actions - Local conversation management
  function createConversation(): Conversation {
    const conversation: Conversation = {
      id: crypto.randomUUID(),
      title: `Conversation ${conversations.value.length + 1}`,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    conversations.value.push(conversation);
    currentConversation.value = conversation;
    return conversation;
  }

  function addMessage(message: Omit<Message, 'id' | 'timestamp'>): Message {
    if (!currentConversation.value) {
      createConversation();
    }
    const newMessage: Message = {
      ...message,
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString()
    };
    currentConversation.value!.messages.push(newMessage);
    currentConversation.value!.updatedAt = new Date();
    messages.value.push(newMessage);
    return newMessage;
  }

  function setCurrentConversation(id: string): void {
    const conversation = conversations.value.find(c => c.id === id);
    if (conversation) {
      currentConversation.value = conversation;
      messages.value = [...conversation.messages];
    }
  }

  function deleteConversation(id: string): void {
    const index = conversations.value.findIndex(c => c.id === id);
    if (index !== -1) {
      conversations.value.splice(index, 1);
      if (currentConversation.value?.id === id) {
        currentConversation.value = conversations.value[0] || null;
      }
    }
  }

  // Actions - API integration
  const sendTextMessage = async (text: string): Promise<ConversationResponse> => {
    isSending.value = true;
    error.value = null;

    try {
      const response = await conversationService.sendTextMessage(
        text,
        currentContext.value ?? undefined
      );

      // Add messages to state
      addMessage(response.user_message);
      addMessage(response.assistant_message);

      // Update conversation ID
      activeConversationId.value = response.id;

      // Store corrections
      if (response.corrections) {
        corrections.value.push(...response.corrections);
      }

      lastResponse.value = response;
      return response;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to send message';
      throw e;
    } finally {
      isSending.value = false;
    }
  };

  const sendAudioMessage = async (audioBase64: string): Promise<ConversationResponse> => {
    isSending.value = true;
    error.value = null;

    try {
      const response = await conversationService.sendAudioMessage(
        audioBase64,
        currentContext.value ?? undefined
      );

      // Add messages to state
      addMessage(response.user_message);
      addMessage(response.assistant_message);

      // Update conversation ID
      activeConversationId.value = response.id;

      // Store corrections
      if (response.corrections) {
        corrections.value.push(...response.corrections);
      }

      lastResponse.value = response;
      return response;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to send audio';
      throw e;
    } finally {
      isSending.value = false;
    }
  };

  const loadHistory = async (): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      messages.value = await conversationService.getSessionHistory();
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load history';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const loadSessions = async (): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      sessions.value = await conversationService.getSessions();
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load sessions';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  async function searchConversations(query: string): Promise<Conversation[]> {
    const lowerQuery = query.toLowerCase();
    return conversations.value.filter(c =>
      c.title.toLowerCase().includes(lowerQuery) ||
      c.messages.some(m => m.content.toLowerCase().includes(lowerQuery))
    );
  }

  // Context management
  const setContext = (context: ConversationContext): void => {
    currentContext.value = context;
  };

  const clearContext = (): void => {
    currentContext.value = null;
  };

  // State management
  function clearCurrentConversation(): void {
    currentConversation.value = null;
  }

  const clearMessages = (): void => {
    messages.value = [];
    corrections.value = [];
    activeConversationId.value = null;
    lastResponse.value = null;
  };

  const clearHistory = async (): Promise<void> => {
    await conversationService.clearHistory();
    clearMessages();
    sessions.value = [];
    conversations.value = [];
    currentConversation.value = null;
  };

  const deleteSession = async (sessionId: string): Promise<void> => {
    await conversationService.deleteSession(sessionId);
    sessions.value = sessions.value.filter((s) => s.id !== sessionId);
  };

  function setRecording(value: boolean): void {
    isRecording.value = value;
  }

  function setProcessing(value: boolean): void {
    isProcessing.value = value;
  }

  const addOptimisticMessage = (content: string, role: 'user'): string => {
    const id = `temp_${Date.now()}`;
    const message: Message = {
      id,
      role,
      content,
      timestamp: new Date().toISOString(),
    };
    messages.value.push(message);
    return id;
  };

  const removeOptimisticMessage = (id: string): void => {
    const index = messages.value.findIndex((m) => m.id === id);
    if (index > -1) {
      messages.value.splice(index, 1);
    }
  };

  const clearError = (): void => {
    error.value = null;
  };

  return {
    // State
    conversations,
    currentConversation,
    messages,
    sessions,
    activeConversationId,
    isLoading,
    isRecording,
    isProcessing,
    isSending,
    error,
    currentContext,
    lastResponse,
    corrections,

    // Computed
    currentMessages,
    conversationCount,
    messageCount,
    hasActiveConversation,
    lastMessage,
    userMessages,
    assistantMessages,
    sortedConversations,
    totalCorrections,
    correctionsByType,

    // Actions - Local
    createConversation,
    addMessage,
    setCurrentConversation,
    deleteConversation,
    clearCurrentConversation,
    searchConversations,
    setRecording,
    setProcessing,

    // Actions - API
    sendTextMessage,
    sendAudioMessage,
    loadHistory,
    loadSessions,
    setContext,
    clearContext,
    clearMessages,
    clearHistory,
    deleteSession,
    addOptimisticMessage,
    removeOptimisticMessage,
    clearError,
  };
});

export type ConversationStore = ReturnType<typeof useConversationStore>;
export type { Message, ConversationResponse };
