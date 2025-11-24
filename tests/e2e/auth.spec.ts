/**
 * Authentication E2E Tests
 *
 * Tests the complete authentication flow:
 * - User registration
 * - Login with credentials
 * - Token refresh flow
 * - Logout
 */

import { test, expect } from '@playwright/test';
import { ApiPage } from './pages/api.page';
import {
  TEST_USERS,
  API_ENDPOINTS,
  PATTERNS,
  EXPECTED_RESPONSES,
  generateUniqueEmail,
  generateUniquePassword,
} from './fixtures/test-data';

test.describe('Authentication Flow', () => {
  let apiPage: ApiPage;

  test.beforeEach(async ({ request }) => {
    apiPage = new ApiPage(request);
  });

  test.describe('User Registration', () => {
    test('should register a new user with valid credentials', async () => {
      const email = generateUniqueEmail();
      const password = 'ValidPassword123!';

      const { response, data } = await apiPage.register(email, password, 'Test User');

      expect(response.status()).toBe(201);
      expect(data).toBeDefined();
      expect(data.email).toBe(email);
      expect(data.full_name).toBe('Test User');
      expect(data.id).toMatch(PATTERNS.uuid);
      expect(data.role).toBe('user');
      expect(data.status).toBe('active');
      expect(data.created_at).toMatch(PATTERNS.isoDate);
    });

    test('should reject registration with weak password', async () => {
      const email = generateUniqueEmail();

      // Test various weak passwords
      const weakPasswords = [
        { password: 'short', error: 'at least 8 characters' },
        { password: 'nouppercase123', error: 'uppercase letter' },
        { password: 'NOLOWERCASE123', error: 'lowercase letter' },
        { password: 'NoDigitsHere!', error: 'digit' },
      ];

      for (const { password, error } of weakPasswords) {
        const { response, data } = await apiPage.register(email, password);
        expect(response.status()).toBe(422);
        // Validation error should mention the specific requirement
        const errorMessage = JSON.stringify(data.detail).toLowerCase();
        expect(errorMessage).toContain(error.toLowerCase());
      }
    });

    test('should reject registration with invalid email', async () => {
      const invalidEmails = ['notanemail', 'missing@domain', '@nodomain.com', 'spaces in@email.com'];

      for (const email of invalidEmails) {
        const { response } = await apiPage.register(email, 'ValidPassword123!');
        expect(response.status()).toBe(422);
      }
    });

    test('should reject duplicate email registration', async () => {
      const email = generateUniqueEmail();
      const password = 'ValidPassword123!';

      // First registration should succeed
      const { response: firstResponse } = await apiPage.register(email, password);
      expect(firstResponse.status()).toBe(201);

      // Second registration with same email should fail
      const { response: secondResponse } = await apiPage.register(email, password);
      expect(secondResponse.status()).toBe(400);
    });
  });

  test.describe('User Login', () => {
    let testEmail: string;
    let testPassword: string;

    test.beforeAll(async ({ request }) => {
      // Create a test user for login tests
      testEmail = generateUniqueEmail();
      testPassword = 'TestPassword123!';
      const api = new ApiPage(request);
      await api.register(testEmail, testPassword, 'Login Test User');
    });

    test('should login with valid credentials (JSON)', async () => {
      const { response, data } = await apiPage.login(testEmail, testPassword);

      expect(response.status()).toBe(200);
      expect(data).toBeDefined();

      // Verify token structure
      for (const field of EXPECTED_RESPONSES.token.requiredFields) {
        expect(data).toHaveProperty(field);
      }

      expect(data.access_token).toMatch(PATTERNS.jwt);
      expect(data.refresh_token).toMatch(PATTERNS.jwt);
      expect(data.token_type).toBe('bearer');
      expect(data.expires_in).toBeGreaterThan(0);
    });

    test('should login with valid credentials (form data)', async () => {
      const { response, data } = await apiPage.loginWithForm(testEmail, testPassword);

      expect(response.status()).toBe(200);
      expect(data.access_token).toMatch(PATTERNS.jwt);
      expect(data.refresh_token).toMatch(PATTERNS.jwt);
    });

    test('should reject login with wrong password', async () => {
      const { response, data } = await apiPage.login(testEmail, 'WrongPassword123!');

      expect(response.status()).toBe(401);
      expect(data.detail).toBeDefined();
    });

    test('should reject login with non-existent email', async () => {
      const { response, data } = await apiPage.login('nonexistent@example.com', 'AnyPassword123!');

      expect(response.status()).toBe(401);
      expect(data.detail).toBeDefined();
    });

    test('should reject login with empty credentials', async () => {
      // Empty email
      const { response: emptyEmailResponse } = await apiPage.login('', 'SomePassword123!');
      expect(emptyEmailResponse.status()).toBe(422);

      // Empty password
      const { response: emptyPasswordResponse } = await apiPage.login(testEmail, '');
      expect(emptyPasswordResponse.status()).toBe(422);
    });
  });

  test.describe('Token Refresh', () => {
    let accessToken: string;
    let refreshToken: string;

    test.beforeEach(async ({ request }) => {
      // Create and login a fresh user
      const email = generateUniqueEmail();
      const password = 'RefreshTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data } = await api.login(email, password);
      accessToken = data.access_token;
      refreshToken = data.refresh_token;
    });

    test('should refresh access token with valid refresh token', async () => {
      const { response, data } = await apiPage.refresh(refreshToken);

      expect(response.status()).toBe(200);
      expect(data.access_token).toMatch(PATTERNS.jwt);
      expect(data.refresh_token).toMatch(PATTERNS.jwt);
      expect(data.token_type).toBe('bearer');

      // New access token should be different
      expect(data.access_token).not.toBe(accessToken);
    });

    test('should reject refresh with invalid token', async () => {
      const { response } = await apiPage.refresh('invalid.refresh.token');

      expect(response.status()).toBe(401);
    });

    test('should reject refresh with malformed token', async () => {
      const { response } = await apiPage.refresh('not-a-jwt');

      expect(response.status()).toBe(401);
    });

    test('should reject refresh with empty token', async () => {
      const { response } = await apiPage.refresh('');

      expect(response.status()).toBe(422);
    });
  });

  test.describe('User Logout', () => {
    let accessToken: string;
    let refreshToken: string;
    let api: ApiPage;

    test.beforeEach(async ({ request }) => {
      // Create and login a fresh user
      const email = generateUniqueEmail();
      const password = 'LogoutTestPassword123!';
      api = new ApiPage(request);

      await api.register(email, password);
      const { data } = await api.login(email, password);
      accessToken = data.access_token;
      refreshToken = data.refresh_token;
      api.setAuthToken(accessToken);
    });

    test('should logout successfully', async () => {
      const { response } = await api.logout(refreshToken);

      expect(response.status()).toBe(204);
    });

    test('should invalidate access token after logout', async () => {
      // Logout
      await api.logout(refreshToken);

      // Try to use the old token - should fail
      const { response } = await api.getProfile();

      // Token should be blacklisted (401 Unauthorized)
      expect(response.status()).toBe(401);
    });

    test('should reject logout without authentication', async () => {
      api.clearAuthToken();

      const { response } = await api.logout();

      expect(response.status()).toBe(401);
    });
  });

  test.describe('Protected Endpoints', () => {
    test('should access user profile with valid token', async ({ request }) => {
      // Setup: create and login user
      const email = generateUniqueEmail();
      const password = 'ProfileTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password, 'Profile Test User');
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Get profile
      const { response, data } = await api.getProfile();

      expect(response.status()).toBe(200);
      expect(data.email).toBe(email);
      expect(data.full_name).toBe('Profile Test User');

      // Verify all required fields
      for (const field of EXPECTED_RESPONSES.user.requiredFields) {
        expect(data).toHaveProperty(field);
      }
    });

    test('should reject access without token', async () => {
      const { response } = await apiPage.getProfile();

      expect(response.status()).toBe(401);
    });

    test('should reject access with invalid token', async () => {
      apiPage.setAuthToken('invalid.access.token');

      const { response } = await apiPage.getProfile();

      expect(response.status()).toBe(401);
    });

    test('should reject access with expired token', async () => {
      // Create an expired token (this would typically be a token with exp in the past)
      // For now, we test with a malformed token
      apiPage.setAuthToken('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjB9.invalid');

      const { response } = await apiPage.getProfile();

      expect(response.status()).toBe(401);
    });
  });

  test.describe('Password Change', () => {
    test('should change password successfully', async ({ request }) => {
      const email = generateUniqueEmail();
      const originalPassword = 'OriginalPassword123!';
      const newPassword = 'NewPassword456!';
      const api = new ApiPage(request);

      // Register and login
      await api.register(email, originalPassword);
      const { data: loginData } = await api.login(email, originalPassword);
      api.setAuthToken(loginData.access_token);

      // Change password
      const { response } = await api.changePassword(originalPassword, newPassword);
      expect(response.status()).toBe(204);

      // Login with new password should work
      const { response: newLoginResponse } = await api.login(email, newPassword);
      expect(newLoginResponse.status()).toBe(200);

      // Login with old password should fail
      const { response: oldLoginResponse } = await api.login(email, originalPassword);
      expect(oldLoginResponse.status()).toBe(401);
    });

    test('should reject password change with wrong current password', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'CurrentPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      const { response } = await api.changePassword('WrongCurrent123!', 'NewPassword456!');
      expect(response.status()).toBe(400);
    });

    test('should reject weak new password', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'CurrentPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      const { response } = await api.changePassword(password, 'weak');
      expect(response.status()).toBe(422);
    });
  });

  test.describe('Profile Update', () => {
    test('should update user profile', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'UpdateTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password, 'Original Name');
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Update profile
      const { response, data } = await api.updateProfile({ full_name: 'Updated Name' });

      expect(response.status()).toBe(200);
      expect(data.full_name).toBe('Updated Name');
      expect(data.email).toBe(email);
    });
  });
});

test.describe('Health Check Endpoints', () => {
  let apiPage: ApiPage;

  test.beforeEach(async ({ request }) => {
    apiPage = new ApiPage(request);
  });

  test('should return healthy status', async () => {
    const health = await apiPage.checkHealth();

    expect(health.status).toBe('healthy');
  });

  test('should return detailed health status', async () => {
    const health = await apiPage.checkHealthDetailed();

    expect(health).toHaveProperty('status');
    expect(health).toHaveProperty('timestamp');
    expect(health).toHaveProperty('components');
  });
});
