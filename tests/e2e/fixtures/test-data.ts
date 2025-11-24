/**
 * Test Data and Constants for E2E Tests
 *
 * Centralized test data to ensure consistency across test suites.
 */

import { v4 as uuidv4 } from 'uuid';

// Generate unique identifiers for test isolation
const testRunId = uuidv4().slice(0, 8);

/**
 * Test user credentials and data
 */
export const TEST_USERS = {
  primary: {
    email: `e2e-test-user-${testRunId}@example.com`,
    password: 'TestPassword123!',
    fullName: 'E2E Test User',
  },
  secondary: {
    email: `e2e-test-user2-${testRunId}@example.com`,
    password: 'SecondPassword456!',
    fullName: 'Secondary Test User',
  },
  admin: {
    email: process.env.TEST_ADMIN_EMAIL || `admin-${testRunId}@example.com`,
    password: process.env.TEST_ADMIN_PASSWORD || 'AdminPassword789!',
    fullName: 'Admin Test User',
  },
};

/**
 * Test conversation data
 */
export const TEST_CONVERSATIONS = {
  simple: {
    messages: [
      'Hello, I want to learn about machine learning',
      'What is a neural network?',
      'Can you give me an example?',
    ],
  },
  complex: {
    messages: [
      'I need help understanding quantum computing fundamentals',
      'What makes quantum bits different from classical bits?',
      'How does superposition work in practice?',
      'What are some real-world applications?',
    ],
  },
};

/**
 * API endpoints for testing
 */
export const API_ENDPOINTS = {
  // Health
  health: '/health',
  healthDetailed: '/health/detailed',

  // Auth
  register: '/api/auth/register',
  login: '/api/auth/login',
  loginJson: '/api/auth/login/json',
  refresh: '/api/auth/refresh',
  logout: '/api/auth/logout',

  // User
  userProfile: '/api/user/me',
  userPassword: '/api/user/me/password',

  // GDPR
  gdprExport: '/api/gdpr/export',
  gdprDelete: '/api/gdpr/delete',
  gdprDeleteCancel: '/api/gdpr/delete/cancel',

  // Conversation
  conversation: '/api/conversation',
  search: '/api/search',
  semanticSearch: '/api/semantic-search',
  sessionHistory: (sessionId: string) => `/api/session/${sessionId}/history`,

  // WebSocket
  websocket: (sessionId: string) => `/ws/${sessionId}`,

  // Stats
  stats: '/api/stats',
};

/**
 * Timeouts for various operations
 */
export const TIMEOUTS = {
  short: 5000,
  medium: 15000,
  long: 30000,
  websocket: 60000,
};

/**
 * Regular expressions for validation
 */
export const PATTERNS = {
  uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
  jwt: /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$/,
  isoDate: /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/,
  email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
};

/**
 * Expected response structures for validation
 */
export const EXPECTED_RESPONSES = {
  token: {
    requiredFields: ['access_token', 'refresh_token', 'token_type', 'expires_in'],
  },
  user: {
    requiredFields: ['id', 'email', 'role', 'status', 'created_at'],
  },
  conversation: {
    requiredFields: ['session_id', 'user_text', 'agent_text'],
  },
  gdprExport: {
    requiredFields: ['export_id', 'status', 'created_at'],
  },
  gdprDelete: {
    requiredFields: ['status', 'scheduled_at', 'completion_date', 'items_to_delete'],
  },
};

/**
 * Generate unique test data
 */
export function generateUniqueEmail(): string {
  return `e2e-${uuidv4().slice(0, 8)}@example.com`;
}

export function generateUniquePassword(): string {
  return `TestPass${Math.random().toString(36).slice(2)}123!`;
}

export function generateSessionId(): string {
  return uuidv4();
}
