/**
 * Authentication Helpers for E2E Tests
 *
 * Reusable authentication utilities for test setup and teardown.
 */

import { APIRequestContext, expect } from '@playwright/test';
import { API_ENDPOINTS, PATTERNS, TEST_USERS } from './test-data';

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

export interface UserData {
  id: string;
  email: string;
  fullName?: string;
  role: string;
  status: string;
}

/**
 * Register a new user via API
 */
export async function registerUser(
  request: APIRequestContext,
  userData: {
    email: string;
    password: string;
    fullName?: string;
  }
): Promise<UserData> {
  const response = await request.post(API_ENDPOINTS.register, {
    data: {
      email: userData.email,
      password: userData.password,
      full_name: userData.fullName,
    },
  });

  expect(response.status()).toBe(201);

  const data = await response.json();
  return {
    id: data.id,
    email: data.email,
    fullName: data.full_name,
    role: data.role,
    status: data.status,
  };
}

/**
 * Login user and get tokens via API
 */
export async function loginUser(
  request: APIRequestContext,
  credentials: {
    email: string;
    password: string;
  }
): Promise<AuthTokens> {
  const response = await request.post(API_ENDPOINTS.loginJson, {
    data: credentials,
  });

  expect(response.status()).toBe(200);

  const data = await response.json();
  expect(data.access_token).toMatch(PATTERNS.jwt);
  expect(data.refresh_token).toMatch(PATTERNS.jwt);

  return {
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresIn: data.expires_in,
  };
}

/**
 * Refresh access token
 */
export async function refreshAccessToken(
  request: APIRequestContext,
  refreshToken: string
): Promise<AuthTokens> {
  const response = await request.post(API_ENDPOINTS.refresh, {
    data: { refresh_token: refreshToken },
  });

  expect(response.status()).toBe(200);

  const data = await response.json();
  return {
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresIn: data.expires_in,
  };
}

/**
 * Logout user
 */
export async function logoutUser(
  request: APIRequestContext,
  tokens: AuthTokens
): Promise<void> {
  const response = await request.post(API_ENDPOINTS.logout, {
    headers: {
      Authorization: `Bearer ${tokens.accessToken}`,
    },
    data: {
      refresh_token: tokens.refreshToken,
    },
  });

  expect(response.status()).toBe(204);
}

/**
 * Get current user profile
 */
export async function getCurrentUser(
  request: APIRequestContext,
  accessToken: string
): Promise<UserData> {
  const response = await request.get(API_ENDPOINTS.userProfile, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  expect(response.status()).toBe(200);

  const data = await response.json();
  return {
    id: data.id,
    email: data.email,
    fullName: data.full_name,
    role: data.role,
    status: data.status,
  };
}

/**
 * Delete user account (GDPR)
 */
export async function deleteUserAccount(
  request: APIRequestContext,
  accessToken: string,
  reason?: string
): Promise<void> {
  const response = await request.post(API_ENDPOINTS.gdprDelete, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
    data: {
      confirm: true,
      reason: reason || 'E2E test cleanup',
    },
  });

  expect(response.status()).toBe(200);
}

/**
 * Create authenticated request context
 */
export function createAuthHeaders(accessToken: string): Record<string, string> {
  return {
    Authorization: `Bearer ${accessToken}`,
    'Content-Type': 'application/json',
  };
}

/**
 * Setup test user - register and login
 */
export async function setupTestUser(
  request: APIRequestContext,
  userData = TEST_USERS.primary
): Promise<{ user: UserData; tokens: AuthTokens }> {
  // Try to register (may fail if user exists)
  try {
    await registerUser(request, userData);
  } catch (error) {
    // User might already exist, continue to login
  }

  // Login to get tokens
  const tokens = await loginUser(request, {
    email: userData.email,
    password: userData.password,
  });

  // Get user profile
  const user = await getCurrentUser(request, tokens.accessToken);

  return { user, tokens };
}

/**
 * Cleanup test user - delete account
 */
export async function cleanupTestUser(
  request: APIRequestContext,
  accessToken: string
): Promise<void> {
  try {
    await deleteUserAccount(request, accessToken, 'E2E test cleanup');
  } catch (error) {
    // Ignore errors during cleanup
    console.warn('Failed to cleanup test user:', error);
  }
}
