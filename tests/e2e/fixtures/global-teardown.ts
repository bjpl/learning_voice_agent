/**
 * Global Teardown for Playwright E2E Tests
 *
 * Runs once after all tests to:
 * - Clean up test data
 * - Remove temporary files
 * - Generate summary reports
 */

import { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

async function globalTeardown(config: FullConfig) {
  console.log(`[Global Teardown] Starting cleanup...`);

  // Clean up test user data if in CI environment
  if (process.env.CI) {
    const authDir = path.join(__dirname, '.auth');
    if (fs.existsSync(authDir)) {
      // Remove auth files but keep directory
      const files = fs.readdirSync(authDir);
      for (const file of files) {
        fs.unlinkSync(path.join(authDir, file));
      }
      console.log(`[Global Teardown] Cleaned auth directory`);
    }
  }

  console.log(`[Global Teardown] Cleanup complete`);
}

export default globalTeardown;
