/**
 * Global Setup for Playwright E2E Tests
 *
 * Runs once before all tests to:
 * - Verify API is accessible
 * - Create test fixtures directory
 * - Initialize test data if needed
 */

import { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0]?.use?.baseURL || 'http://localhost:8000';

  console.log(`[Global Setup] Starting E2E test setup...`);
  console.log(`[Global Setup] Base URL: ${baseURL}`);

  // Create auth storage directory
  const authDir = path.join(__dirname, '.auth');
  if (!fs.existsSync(authDir)) {
    fs.mkdirSync(authDir, { recursive: true });
    console.log(`[Global Setup] Created auth directory: ${authDir}`);
  }

  // Create empty storage state file if it doesn't exist
  const userAuthPath = path.join(authDir, 'user.json');
  if (!fs.existsSync(userAuthPath)) {
    fs.writeFileSync(userAuthPath, JSON.stringify({
      cookies: [],
      origins: []
    }));
    console.log(`[Global Setup] Created initial auth state`);
  }

  // Verify API is accessible (with retries)
  const maxRetries = 5;
  const retryDelay = 2000;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(`${baseURL}/health`);
      if (response.ok) {
        console.log(`[Global Setup] API health check passed`);
        break;
      }
    } catch (error) {
      if (attempt === maxRetries) {
        console.warn(`[Global Setup] API not accessible after ${maxRetries} attempts. Tests may fail.`);
      } else {
        console.log(`[Global Setup] Waiting for API... (attempt ${attempt}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }

  console.log(`[Global Setup] Setup complete`);
}

export default globalSetup;
