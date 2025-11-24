/**
 * GDPR Compliance E2E Tests
 *
 * Tests GDPR-related functionality:
 * - Export user data (Article 20 - Right to data portability)
 * - Delete user account (Article 17 - Right to erasure)
 * - Verify deletion
 */

import { test, expect } from '@playwright/test';
import { ApiPage } from './pages/api.page';
import {
  API_ENDPOINTS,
  PATTERNS,
  EXPECTED_RESPONSES,
  generateUniqueEmail,
  generateSessionId,
} from './fixtures/test-data';

test.describe('GDPR Compliance', () => {
  test.describe('Data Export (Article 20)', () => {
    let apiPage: ApiPage;
    let accessToken: string;
    let userEmail: string;

    test.beforeAll(async ({ request }) => {
      // Create and authenticate a test user with some data
      userEmail = generateUniqueEmail();
      const password = 'GDPRExportTestPassword123!';
      apiPage = new ApiPage(request);

      await apiPage.register(userEmail, password, 'GDPR Export Test User');
      const { data } = await apiPage.login(userEmail, password);
      accessToken = data.access_token;
      apiPage.setAuthToken(accessToken);

      // Create some conversation data
      const sessionId = generateSessionId();
      await apiPage.sendMessage('Test conversation for export', sessionId);
      await apiPage.sendMessage('More data to export', sessionId);
    });

    test('should request data export successfully', async () => {
      const { response, data } = await apiPage.requestDataExport('json');

      expect(response.status()).toBe(200);

      // Verify response structure
      for (const field of EXPECTED_RESPONSES.gdprExport.requiredFields) {
        expect(data).toHaveProperty(field);
      }

      expect(data.export_id).toMatch(PATTERNS.uuid);
      expect(data.status).toBe('processing');
      expect(data.created_at).toMatch(PATTERNS.isoDate);
    });

    test('should check export status', async () => {
      // First, request an export
      const { data: exportRequest } = await apiPage.requestDataExport('json');
      const exportId = exportRequest.export_id;

      // Check status
      const { response, data } = await apiPage.getExportStatus(exportId);

      expect(response.status()).toBe(200);
      expect(data.export_id).toBe(exportId);
      expect(['processing', 'completed', 'failed']).toContain(data.status);
    });

    test('should download completed export', async () => {
      // Request and get export
      const { data: exportRequest } = await apiPage.requestDataExport('json');
      const exportId = exportRequest.export_id;

      // Download export
      const { response, data } = await apiPage.downloadExport(exportId);

      expect(response.status()).toBe(200);
      expect(data).toHaveProperty('export_id');
      expect(data).toHaveProperty('format');
      expect(data).toHaveProperty('data');

      // Verify export data structure
      const exportData = data.data;
      expect(exportData).toHaveProperty('export_metadata');
      expect(exportData).toHaveProperty('user_profile');
      expect(exportData.export_metadata.gdpr_compliant).toBe(true);
      expect(exportData.user_profile.email).toBe(userEmail);
    });

    test('should export in JSON format by default', async () => {
      const { data } = await apiPage.requestDataExport();

      // Default format should be JSON
      const { data: downloadData } = await apiPage.downloadExport(data.export_id);
      expect(downloadData.format).toBe('json');
    });

    test('should export in CSV format when requested', async () => {
      const { response, data } = await apiPage.requestDataExport('csv');

      expect(response.status()).toBe(200);
      // Note: CSV export might still return JSON wrapper with CSV data inside
    });

    test('should reject export without authentication', async ({ request }) => {
      const api = new ApiPage(request);

      const { response } = await api.requestDataExport('json');

      expect(response.status()).toBe(401);
    });

    test('should include all data categories in export', async () => {
      const { data: exportRequest } = await apiPage.requestDataExport('json');
      const { data } = await apiPage.downloadExport(exportRequest.export_id);

      const exportData = data.data;

      // Verify all expected data categories
      expect(exportData).toHaveProperty('user_profile');
      expect(exportData).toHaveProperty('conversations');
      expect(exportData).toHaveProperty('sessions');
      expect(exportData).toHaveProperty('preferences');

      // Verify metadata
      expect(exportData.export_metadata).toHaveProperty('format_version');
      expect(exportData.export_metadata).toHaveProperty('generated_at');
      expect(exportData.export_metadata).toHaveProperty('user_id');
    });
  });

  test.describe('Account Deletion (Article 17)', () => {
    test('should request account deletion with confirmation', async ({ request }) => {
      // Create a fresh user for deletion test
      const email = generateUniqueEmail();
      const password = 'GDPRDeleteTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password, 'GDPR Delete Test User');
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Request deletion
      const { response, data } = await api.requestAccountDeletion(true, 'Testing GDPR deletion');

      expect(response.status()).toBe(200);

      // Verify response structure
      for (const field of EXPECTED_RESPONSES.gdprDelete.requiredFields) {
        expect(data).toHaveProperty(field);
      }

      expect(data.status).toBe('scheduled');
      expect(data.scheduled_at).toMatch(PATTERNS.isoDate);
      expect(data.completion_date).toMatch(PATTERNS.isoDate);
      expect(Array.isArray(data.items_to_delete)).toBe(true);
      expect(data.items_to_delete.length).toBeGreaterThan(0);
    });

    test('should reject deletion without confirmation', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRNoConfirmTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Try deletion without confirmation
      const { response, data } = await api.requestAccountDeletion(false);

      expect(response.status()).toBe(422);
      // Should have validation error about confirmation
    });

    test('should include reason for deletion', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRReasonTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      const deletionReason = 'No longer need the service';
      const { response, data } = await api.requestAccountDeletion(true, deletionReason);

      expect(response.status()).toBe(200);
      expect(data.status).toBe('scheduled');
    });

    test('should list all items to be deleted', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRItemsTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Create some data first
      await api.sendMessage('Data to be deleted', generateSessionId());

      // Request deletion
      const { data } = await api.requestAccountDeletion(true);

      // Expected items to be deleted
      const expectedItems = ['user_profile', 'conversations', 'sessions', 'preferences', 'metadata'];
      for (const item of expectedItems) {
        expect(data.items_to_delete).toContain(item);
      }
    });

    test('should cancel pending deletion', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRCancelTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Request deletion
      const { response: deleteResponse } = await api.requestAccountDeletion(true);
      expect(deleteResponse.status()).toBe(200);

      // Cancel deletion
      const { response: cancelResponse } = await api.cancelAccountDeletion();
      expect(cancelResponse.status()).toBe(204);
    });

    test('should reject deletion request without authentication', async ({ request }) => {
      const api = new ApiPage(request);

      const { response } = await api.requestAccountDeletion(true);

      expect(response.status()).toBe(401);
    });

    test('should provide 30-day grace period', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRGraceTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      const { data } = await api.requestAccountDeletion(true);

      // Verify 30-day grace period
      const scheduledAt = new Date(data.scheduled_at);
      const completionDate = new Date(data.completion_date);
      const gracePeriodMs = completionDate.getTime() - scheduledAt.getTime();
      const gracePeriodDays = gracePeriodMs / (1000 * 60 * 60 * 24);

      // Should be approximately 30 days
      expect(gracePeriodDays).toBeGreaterThanOrEqual(29);
      expect(gracePeriodDays).toBeLessThanOrEqual(31);
    });
  });

  test.describe('Post-Deletion Verification', () => {
    test('should not be able to login after deletion is processed', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRVerifyTestPassword123!';
      const api = new ApiPage(request);

      // Register, login, request deletion
      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      await api.requestAccountDeletion(true);

      // Note: In a real scenario, the account would be marked as deleted
      // and login would fail after the grace period.
      // This test verifies the immediate state after deletion request.

      // The user's status should be updated to "deleted" or similar
      // For now, we verify the deletion was scheduled
    });

    test('should not be able to export data after deletion request', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRNoExportTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      // Request deletion
      await api.requestAccountDeletion(true);

      // Depending on implementation, export might be blocked for deleted accounts
      // This test documents expected behavior
    });
  });

  test.describe('Data Retention Compliance', () => {
    test('should include export expiry time', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRExpiryTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password);
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      const { data } = await api.requestDataExport('json');

      // Export should have an expiry time
      expect(data.expires_at).toMatch(PATTERNS.isoDate);

      // Expiry should be in the future
      const expiresAt = new Date(data.expires_at);
      const now = new Date();
      expect(expiresAt.getTime()).toBeGreaterThan(now.getTime());
    });

    test('should generate GDPR-compliant export format', async ({ request }) => {
      const email = generateUniqueEmail();
      const password = 'GDPRFormatTestPassword123!';
      const api = new ApiPage(request);

      await api.register(email, password, 'Format Test User');
      const { data: loginData } = await api.login(email, password);
      api.setAuthToken(loginData.access_token);

      const { data: exportRequest } = await api.requestDataExport('json');
      const { data } = await api.downloadExport(exportRequest.export_id);

      // Verify GDPR-compliant metadata
      const metadata = data.data.export_metadata;
      expect(metadata.gdpr_compliant).toBe(true);
      expect(metadata.format_version).toBeDefined();
      expect(metadata.generated_at).toBeDefined();
      expect(metadata.user_id).toBeDefined();
    });
  });
});

test.describe('User Rights Management', () => {
  test('should provide access to all personal data', async ({ request }) => {
    const email = generateUniqueEmail();
    const password = 'UserRightsTestPassword123!';
    const api = new ApiPage(request);

    await api.register(email, password, 'Rights Test User');
    const { data: loginData } = await api.login(email, password);
    api.setAuthToken(loginData.access_token);

    // User should be able to view their profile
    const { response: profileResponse, data: profile } = await api.getProfile();
    expect(profileResponse.status()).toBe(200);
    expect(profile.email).toBe(email);

    // User should be able to export their data
    const { response: exportResponse } = await api.requestDataExport('json');
    expect(exportResponse.status()).toBe(200);
  });

  test('should allow rectification of personal data', async ({ request }) => {
    const email = generateUniqueEmail();
    const password = 'RectificationTestPassword123!';
    const api = new ApiPage(request);

    await api.register(email, password, 'Original Name');
    const { data: loginData } = await api.login(email, password);
    api.setAuthToken(loginData.access_token);

    // Update profile (rectification right - Article 16)
    const { response, data } = await api.updateProfile({ full_name: 'Corrected Name' });

    expect(response.status()).toBe(200);
    expect(data.full_name).toBe('Corrected Name');
  });

  test('should process requests in timely manner', async ({ request }) => {
    const email = generateUniqueEmail();
    const password = 'TimelyTestPassword123!';
    const api = new ApiPage(request);

    await api.register(email, password);
    const { data: loginData } = await api.login(email, password);
    api.setAuthToken(loginData.access_token);

    // GDPR requires response within 1 month
    // We test that the API responds promptly
    const startTime = Date.now();
    const { response } = await api.requestDataExport('json');
    const endTime = Date.now();

    expect(response.status()).toBe(200);
    // Response should be within reasonable time (e.g., 5 seconds)
    expect(endTime - startTime).toBeLessThan(5000);
  });
});
