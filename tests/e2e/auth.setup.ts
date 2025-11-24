/**
 * Authentication Setup for E2E Tests
 *
 * This file runs before other tests to set up authenticated state.
 * It creates a test user and saves the authentication state for reuse.
 */

import { test as setup, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { TEST_USERS, API_ENDPOINTS, PATTERNS } from './fixtures/test-data';

const authFile = path.join(__dirname, 'fixtures', '.auth', 'user.json');

setup('authenticate', async ({ request }) => {
  const testUser = TEST_USERS.primary;

  // Step 1: Try to register (may fail if user exists)
  try {
    const registerResponse = await request.post(API_ENDPOINTS.register, {
      data: {
        email: testUser.email,
        password: testUser.password,
        full_name: testUser.fullName,
      },
    });

    if (registerResponse.status() === 201) {
      console.log(`[Auth Setup] Created test user: ${testUser.email}`);
    }
  } catch (error) {
    // User might already exist, continue to login
    console.log(`[Auth Setup] User registration skipped (may already exist)`);
  }

  // Step 2: Login to get tokens
  const loginResponse = await request.post(API_ENDPOINTS.loginJson, {
    data: {
      email: testUser.email,
      password: testUser.password,
    },
  });

  if (loginResponse.status() !== 200) {
    throw new Error(`Login failed with status ${loginResponse.status()}`);
  }

  const loginData = await loginResponse.json();

  expect(loginData.access_token).toMatch(PATTERNS.jwt);
  expect(loginData.refresh_token).toMatch(PATTERNS.jwt);

  console.log(`[Auth Setup] Successfully authenticated user: ${testUser.email}`);

  // Step 3: Save authentication state
  const authState = {
    cookies: [],
    origins: [
      {
        origin: process.env.BASE_URL || 'http://localhost:8000',
        localStorage: [
          {
            name: 'accessToken',
            value: loginData.access_token,
          },
          {
            name: 'refreshToken',
            value: loginData.refresh_token,
          },
          {
            name: 'userEmail',
            value: testUser.email,
          },
        ],
      },
    ],
  };

  // Ensure directory exists
  const authDir = path.dirname(authFile);
  if (!fs.existsSync(authDir)) {
    fs.mkdirSync(authDir, { recursive: true });
  }

  // Write auth state
  fs.writeFileSync(authFile, JSON.stringify(authState, null, 2));

  console.log(`[Auth Setup] Authentication state saved to: ${authFile}`);
});
