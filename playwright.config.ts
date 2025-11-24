import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for Learning Voice Agent E2E Tests
 *
 * Environment-aware configuration supporting dev, staging, and production.
 * Supports parallel execution with multiple browsers.
 */

// Read environment variables
const ENV = process.env.TEST_ENV || 'dev';

// Base URLs by environment
const BASE_URLS: Record<string, string> = {
  dev: 'http://localhost:8000',
  staging: process.env.STAGING_URL || 'https://staging.learning-voice-agent.example.com',
  prod: process.env.PROD_URL || 'https://learning-voice-agent.example.com',
};

export default defineConfig({
  // Test directory
  testDir: './tests/e2e',

  // Test file patterns
  testMatch: '**/*.spec.ts',

  // Parallel execution
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 0,

  // Limit workers on CI for stability
  workers: process.env.CI ? 2 : undefined,

  // Reporter configuration
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
    ['json', { outputFile: 'playwright-results.json' }],
    ...(process.env.CI ? [['github'] as ['github']] : []),
  ],

  // Global setup and teardown
  globalSetup: './tests/e2e/fixtures/global-setup.ts',
  globalTeardown: './tests/e2e/fixtures/global-teardown.ts',

  // Shared settings for all projects
  use: {
    // Base URL for navigation
    baseURL: BASE_URLS[ENV],

    // Collect trace when retrying failed tests
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Video recording
    video: process.env.CI ? 'on-first-retry' : 'off',

    // Extra HTTP headers
    extraHTTPHeaders: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },

    // Timeout for actions
    actionTimeout: 15000,

    // Timeout for navigation
    navigationTimeout: 30000,
  },

  // Configure projects for major browsers
  projects: [
    // Authentication setup project - runs first
    {
      name: 'setup',
      testMatch: /.*\.setup\.ts/,
      teardown: 'cleanup',
    },

    // Cleanup project - runs last
    {
      name: 'cleanup',
      testMatch: /.*\.teardown\.ts/,
    },

    // Chromium tests
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        storageState: './tests/e2e/fixtures/.auth/user.json',
      },
      dependencies: ['setup'],
    },

    // Firefox tests
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        storageState: './tests/e2e/fixtures/.auth/user.json',
      },
      dependencies: ['setup'],
    },

    // WebKit tests (Safari)
    {
      name: 'webkit',
      use: {
        ...devices['Desktop Safari'],
        storageState: './tests/e2e/fixtures/.auth/user.json',
      },
      dependencies: ['setup'],
    },

    // Mobile Chrome tests
    {
      name: 'mobile-chrome',
      use: {
        ...devices['Pixel 5'],
        storageState: './tests/e2e/fixtures/.auth/user.json',
      },
      dependencies: ['setup'],
    },

    // Mobile Safari tests
    {
      name: 'mobile-safari',
      use: {
        ...devices['iPhone 12'],
        storageState: './tests/e2e/fixtures/.auth/user.json',
      },
      dependencies: ['setup'],
    },

    // API tests (no browser needed)
    {
      name: 'api',
      testMatch: '**/*.api.spec.ts',
      use: {
        baseURL: BASE_URLS[ENV],
      },
    },
  ],

  // Global timeout
  timeout: 60000,

  // Expect timeout
  expect: {
    timeout: 10000,
  },

  // Web server configuration for local development
  webServer: ENV === 'dev' ? {
    command: 'cd .. && uvicorn app.main:app --host 0.0.0.0 --port 8000',
    url: 'http://localhost:8000/health',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
    stdout: 'pipe',
    stderr: 'pipe',
  } : undefined,

  // Output directory for test artifacts
  outputDir: 'test-results',
});
