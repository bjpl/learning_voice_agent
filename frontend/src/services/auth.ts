/**
 * Session management service
 */

import {
  getSessionId,
  setSessionId,
  clearSessionId,
  generateSessionId,
  apiClient,
} from './api';
import type { ApiResponse, HealthCheck } from '@/types';

export interface SessionInfo {
  sessionId: string;
  createdAt: string;
  lastActive: string;
  isActive: boolean;
}

const SESSION_INFO_KEY = 'session_info';

/**
 * Initialize or restore session
 */
export const initSession = (): SessionInfo => {
  let sessionId = getSessionId();
  let sessionInfo = getStoredSessionInfo();

  if (!sessionId || !sessionInfo) {
    sessionId = generateSessionId();
    setSessionId(sessionId);

    sessionInfo = {
      sessionId,
      createdAt: new Date().toISOString(),
      lastActive: new Date().toISOString(),
      isActive: true,
    };

    storeSessionInfo(sessionInfo);
  } else {
    // Update last active
    sessionInfo.lastActive = new Date().toISOString();
    sessionInfo.isActive = true;
    storeSessionInfo(sessionInfo);
  }

  return sessionInfo;
};

/**
 * End current session
 */
export const endSession = (): void => {
  const sessionInfo = getStoredSessionInfo();

  if (sessionInfo) {
    sessionInfo.isActive = false;
    sessionInfo.lastActive = new Date().toISOString();
    storeSessionInfo(sessionInfo);
  }

  clearSessionId();
};

/**
 * Create a new session (clearing old one)
 */
export const createNewSession = (): SessionInfo => {
  endSession();
  localStorage.removeItem(SESSION_INFO_KEY);
  return initSession();
};

/**
 * Get current session info
 */
export const getCurrentSession = (): SessionInfo | null => {
  const sessionId = getSessionId();
  const sessionInfo = getStoredSessionInfo();

  if (!sessionId || !sessionInfo) {
    return null;
  }

  return sessionInfo;
};

/**
 * Check if session is valid
 */
export const isSessionValid = (): boolean => {
  const sessionInfo = getStoredSessionInfo();

  if (!sessionInfo) {
    return false;
  }

  // Session expires after 24 hours of inactivity
  const lastActive = new Date(sessionInfo.lastActive);
  const now = new Date();
  const hoursDiff = (now.getTime() - lastActive.getTime()) / (1000 * 60 * 60);

  return hoursDiff < 24;
};

/**
 * Update session last active time
 */
export const touchSession = (): void => {
  const sessionInfo = getStoredSessionInfo();

  if (sessionInfo) {
    sessionInfo.lastActive = new Date().toISOString();
    storeSessionInfo(sessionInfo);
  }
};

/**
 * Store session info in localStorage
 */
const storeSessionInfo = (info: SessionInfo): void => {
  localStorage.setItem(SESSION_INFO_KEY, JSON.stringify(info));
};

/**
 * Get stored session info from localStorage
 */
const getStoredSessionInfo = (): SessionInfo | null => {
  const stored = localStorage.getItem(SESSION_INFO_KEY);

  if (!stored) {
    return null;
  }

  try {
    return JSON.parse(stored) as SessionInfo;
  } catch {
    return null;
  }
};

/**
 * Check API health status
 */
export const checkHealth = async (): Promise<HealthCheck> => {
  const response = await apiClient.get<ApiResponse<HealthCheck>>('/health');
  return response.data.data;
};

/**
 * Ping API to check connectivity
 */
export const ping = async (): Promise<boolean> => {
  try {
    await apiClient.get('/health', { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
};

export default {
  initSession,
  endSession,
  createNewSession,
  getCurrentSession,
  isSessionValid,
  touchSession,
  checkHealth,
  ping,
};
