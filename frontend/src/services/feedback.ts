/**
 * Feedback submission API service
 */

import { apiClient, getSessionId } from './api';
import type {
  ApiResponse,
  ExplicitFeedback,
  ImplicitFeedback,
  CorrectionFeedback,
  FeedbackSummary,
} from '@/types';

/**
 * Submit explicit feedback (ratings, comments)
 */
export const submitExplicitFeedback = async (
  feedback: Omit<ExplicitFeedback, 'id' | 'session_id' | 'timestamp'>
): Promise<ExplicitFeedback> => {
  const sessionId = getSessionId();

  if (!sessionId) {
    throw new Error('No active session.');
  }

  const request: Omit<ExplicitFeedback, 'id' | 'timestamp'> = {
    ...feedback,
    session_id: sessionId,
  };

  const response = await apiClient.post<ApiResponse<ExplicitFeedback>>(
    '/api/feedback/explicit',
    request
  );

  return response.data.data;
};

/**
 * Submit implicit feedback (behavioral signals)
 */
export const submitImplicitFeedback = async (
  feedback: Omit<ImplicitFeedback, 'id' | 'session_id' | 'timestamp'>
): Promise<ImplicitFeedback> => {
  const sessionId = getSessionId();

  if (!sessionId) {
    throw new Error('No active session.');
  }

  const request: Omit<ImplicitFeedback, 'id' | 'timestamp'> = {
    ...feedback,
    session_id: sessionId,
  };

  const response = await apiClient.post<ApiResponse<ImplicitFeedback>>(
    '/api/feedback/implicit',
    request
  );

  return response.data.data;
};

/**
 * Submit correction feedback (acceptance/rejection)
 */
export const submitCorrectionFeedback = async (
  feedback: Omit<CorrectionFeedback, 'id' | 'session_id' | 'timestamp'>
): Promise<CorrectionFeedback> => {
  const sessionId = getSessionId();

  if (!sessionId) {
    throw new Error('No active session.');
  }

  const request: Omit<CorrectionFeedback, 'id' | 'timestamp'> = {
    ...feedback,
    session_id: sessionId,
  };

  const response = await apiClient.post<ApiResponse<CorrectionFeedback>>(
    '/api/feedback/correction',
    request
  );

  return response.data.data;
};

/**
 * Get feedback summary for current session
 */
export const getFeedbackSummary = async (): Promise<FeedbackSummary> => {
  const sessionId = getSessionId();

  if (!sessionId) {
    throw new Error('No active session.');
  }

  const response = await apiClient.get<ApiResponse<FeedbackSummary>>(
    `/api/feedback/summary/${sessionId}`
  );

  return response.data.data;
};

/**
 * Batch submit multiple feedback items (for offline queue)
 */
export const submitFeedbackBatch = async (
  items: Array<{
    type: 'explicit' | 'implicit' | 'correction';
    payload: ExplicitFeedback | ImplicitFeedback | CorrectionFeedback;
  }>
): Promise<{ success: number; failed: number }> => {
  const response = await apiClient.post<
    ApiResponse<{ success: number; failed: number }>
  >('/api/feedback/batch', { items });

  return response.data.data;
};

export default {
  submitExplicitFeedback,
  submitImplicitFeedback,
  submitCorrectionFeedback,
  getFeedbackSummary,
  submitFeedbackBatch,
};
