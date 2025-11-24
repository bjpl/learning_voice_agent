/**
 * Base axios client with interceptors, retry logic, and authentication
 */

import axios, {
  type AxiosInstance,
  type AxiosError,
  type InternalAxiosRequestConfig,
  type AxiosResponse,
} from 'axios';
import type { ApiError, RateLimitInfo } from '@/types';

const BASE_URL = 'http://localhost:8000';
const DEFAULT_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Session storage key
const SESSION_KEY = 'session_id';

/**
 * Get session ID from localStorage
 */
export const getSessionId = (): string | null => {
  return localStorage.getItem(SESSION_KEY);
};

/**
 * Set session ID in localStorage
 */
export const setSessionId = (sessionId: string): void => {
  localStorage.setItem(SESSION_KEY, sessionId);
};

/**
 * Remove session ID from localStorage
 */
export const clearSessionId = (): void => {
  localStorage.removeItem(SESSION_KEY);
};

/**
 * Generate a new session ID
 */
export const generateSessionId = (): string => {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
};

/**
 * Create axios instance with default configuration
 */
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: BASE_URL,
    timeout: DEFAULT_TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor
  client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      const sessionId = getSessionId();
      if (sessionId) {
        config.headers['X-Session-ID'] = sessionId;
      }

      // Add timestamp for debugging
      config.headers['X-Request-Time'] = new Date().toISOString();

      return config;
    },
    (error: AxiosError) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor
  client.interceptors.response.use(
    (response: AxiosResponse) => {
      // Extract rate limit info if present
      const rateLimitInfo = extractRateLimitInfo(response);
      if (rateLimitInfo) {
        response.data._rateLimitInfo = rateLimitInfo;
      }

      return response;
    },
    async (error: AxiosError<ApiError>) => {
      const originalRequest = error.config as InternalAxiosRequestConfig & {
        _retryCount?: number;
      };

      // Handle rate limiting
      if (error.response?.status === 429) {
        const retryAfter = error.response.headers['retry-after'];
        const delay = retryAfter ? parseInt(retryAfter) * 1000 : RETRY_DELAY;

        if (!originalRequest._retryCount || originalRequest._retryCount < MAX_RETRIES) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
          await sleep(delay);
          return client(originalRequest);
        }
      }

      // Retry on network errors or 5xx errors
      if (shouldRetry(error, originalRequest)) {
        originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
        const delay = RETRY_DELAY * Math.pow(2, originalRequest._retryCount - 1);
        await sleep(delay);
        return client(originalRequest);
      }

      // Transform error to standard format
      const apiError = transformError(error);
      return Promise.reject(apiError);
    }
  );

  return client;
};

/**
 * Check if request should be retried
 */
const shouldRetry = (
  error: AxiosError,
  config: InternalAxiosRequestConfig & { _retryCount?: number }
): boolean => {
  const retryCount = config._retryCount || 0;

  if (retryCount >= MAX_RETRIES) {
    return false;
  }

  // Retry on network errors
  if (!error.response) {
    return true;
  }

  // Retry on 5xx errors (server errors)
  if (error.response.status >= 500) {
    return true;
  }

  return false;
};

/**
 * Extract rate limit information from response headers
 */
const extractRateLimitInfo = (response: AxiosResponse): RateLimitInfo | null => {
  const headers = response.headers;

  const limit = headers['x-ratelimit-limit'];
  const remaining = headers['x-ratelimit-remaining'];
  const reset = headers['x-ratelimit-reset'];

  if (limit && remaining && reset) {
    return {
      limit: parseInt(limit),
      remaining: parseInt(remaining),
      reset: parseInt(reset),
    };
  }

  return null;
};

/**
 * Transform axios error to standard API error format
 */
const transformError = (error: AxiosError<ApiError>): ApiError => {
  if (error.response?.data) {
    return {
      code: error.response.data.code || 'UNKNOWN_ERROR',
      message: error.response.data.message || 'An unexpected error occurred',
      details: error.response.data.details,
      statusCode: error.response.status,
      timestamp: new Date().toISOString(),
    };
  }

  if (error.code === 'ECONNABORTED') {
    return {
      code: 'TIMEOUT',
      message: 'Request timed out',
      statusCode: 408,
      timestamp: new Date().toISOString(),
    };
  }

  if (!error.response) {
    return {
      code: 'NETWORK_ERROR',
      message: 'Network error. Please check your connection.',
      statusCode: 0,
      timestamp: new Date().toISOString(),
    };
  }

  return {
    code: 'UNKNOWN_ERROR',
    message: error.message || 'An unexpected error occurred',
    statusCode: error.response?.status || 500,
    timestamp: new Date().toISOString(),
  };
};

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

// Export singleton client
export const apiClient = createApiClient();

// Export utility functions
export { sleep };

export default apiClient;
