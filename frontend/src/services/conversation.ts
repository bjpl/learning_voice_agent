/**
 * Conversation API service
 */

import { apiClient, getSessionId } from './api';
import type {
  ApiResponse,
  ConversationRequest,
  ConversationResponse,
  ConversationHistory,
  ConversationSession,
  Message,
} from '@/types';

/**
 * Send a conversation message (text or audio)
 */
export const sendMessage = async (
  text?: string,
  audioBase64?: string,
  context?: ConversationRequest['context']
): Promise<ConversationResponse> => {
  const sessionId = getSessionId();

  if (!sessionId) {
    throw new Error('No active session. Please initialize a session first.');
  }

  const request: ConversationRequest = {
    session_id: sessionId,
    context,
  };

  if (text) {
    request.text = text;
  }

  if (audioBase64) {
    request.audio_base64 = audioBase64;
  }

  const response = await apiClient.post<ApiResponse<ConversationResponse>>(
    '/conversation',
    request
  );

  return response.data.data;
};

/**
 * Send a text message
 */
export const sendTextMessage = async (
  text: string,
  context?: ConversationRequest['context']
): Promise<ConversationResponse> => {
  return sendMessage(text, undefined, context);
};

/**
 * Send an audio message
 */
export const sendAudioMessage = async (
  audioBase64: string,
  context?: ConversationRequest['context']
): Promise<ConversationResponse> => {
  return sendMessage(undefined, audioBase64, context);
};

/**
 * Get conversation history for current session
 */
export const getSessionHistory = async (): Promise<Message[]> => {
  const sessionId = getSessionId();

  if (!sessionId) {
    throw new Error('No active session.');
  }

  const response = await apiClient.get<ApiResponse<Message[]>>(
    `/api/conversation/history/${sessionId}`
  );

  return response.data.data;
};

/**
 * Get all conversation sessions
 */
export const getSessions = async (): Promise<ConversationSession[]> => {
  const response = await apiClient.get<ApiResponse<ConversationSession[]>>(
    '/api/conversation/sessions'
  );

  return response.data.data;
};

/**
 * Get conversation history summary
 */
export const getHistorySummary = async (): Promise<ConversationHistory> => {
  const response = await apiClient.get<ApiResponse<ConversationHistory>>(
    '/api/conversation/history/summary'
  );

  return response.data.data;
};

/**
 * Delete a conversation session
 */
export const deleteSession = async (sessionId: string): Promise<void> => {
  await apiClient.delete(`/api/conversation/sessions/${sessionId}`);
};

/**
 * Clear all conversation history
 */
export const clearHistory = async (): Promise<void> => {
  await apiClient.delete('/api/conversation/history');
};

/**
 * Export conversation session
 */
export const exportSession = async (
  sessionId: string,
  format: 'json' | 'text' = 'json'
): Promise<string | Record<string, unknown>> => {
  const response = await apiClient.get<ApiResponse<string | Record<string, unknown>>>(
    `/api/conversation/sessions/${sessionId}/export`,
    { params: { format } }
  );

  return response.data.data;
};

export default {
  sendMessage,
  sendTextMessage,
  sendAudioMessage,
  getSessionHistory,
  getSessions,
  getHistorySummary,
  deleteSession,
  clearHistory,
  exportSession,
};
