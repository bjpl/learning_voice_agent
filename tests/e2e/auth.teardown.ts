/**
 * Authentication Teardown for E2E Tests
 *
 * This file runs after all tests to clean up test users and state.
 */

import { test as teardown } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { TEST_USERS, API_ENDPOINTS } from './fixtures/test-data';

const authFile = path.join(__dirname, 'fixtures', '.auth', 'user.json');

teardown('cleanup', async ({ request }) => {
  console.log(`[Auth Teardown] Starting cleanup...`);

  // Only cleanup in CI environment to preserve local test data
  if (!process.env.CI) {
    console.log(`[Auth Teardown] Skipping cleanup (not in CI)`);
    return;
  }

  try {
    // Read stored auth state
    if (fs.existsSync(authFile)) {
      const authState = JSON.parse(fs.readFileSync(authFile, 'utf-8'));
      const origin = authState.origins?.[0];
      const accessToken = origin?.localStorage?.find(
        (item: { name: string }) => item.name === 'accessToken'
      )?.value;

      if (accessToken) {
        // Request account deletion for test user
        const deleteResponse = await request.post(API_ENDPOINTS.gdprDelete, {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
          data: {
            confirm: true,
            reason: 'E2E test cleanup',
          },
        });

        if (deleteResponse.status() === 200) {
          console.log(`[Auth Teardown] Scheduled deletion for test user`);
        }
      }

      // Remove auth file
      fs.unlinkSync(authFile);
      console.log(`[Auth Teardown] Removed auth state file`);
    }
  } catch (error) {
    console.warn(`[Auth Teardown] Cleanup warning: ${error}`);
  }

  console.log(`[Auth Teardown] Cleanup complete`);
});
