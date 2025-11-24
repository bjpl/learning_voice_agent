/**
 * API Page Object for Direct API Testing
 *
 * Provides methods for testing API endpoints directly.
 */

import { APIRequestContext, expect } from '@playwright/test';
import { API_ENDPOINTS, PATTERNS, TIMEOUTS } from '../fixtures/test-data';

export class ApiPage {
  private readonly request: APIRequestContext;
  private accessToken: string | null = null;

  constructor(request: APIRequestContext) {
    this.request = request;
  }

  /**
   * Set authentication token for subsequent requests
   */
  setAuthToken(token: string): void {
    this.accessToken = token;
  }

  /**
   * Clear authentication token
   */
  clearAuthToken(): void {
    this.accessToken = null;
  }

  /**
   * Get authorization headers
   */
  private getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.accessToken) {
      headers['Authorization'] = `Bearer ${this.accessToken}`;
    }
    return headers;
  }

  // ==========================================================================
  // Health Endpoints
  // ==========================================================================

  async checkHealth(): Promise<{ status: string }> {
    const response = await this.request.get(API_ENDPOINTS.health);
    expect(response.status()).toBe(200);
    return response.json();
  }

  async checkHealthDetailed(): Promise<Record<string, any>> {
    const response = await this.request.get(API_ENDPOINTS.healthDetailed);
    expect(response.status()).toBe(200);
    return response.json();
  }

  // ==========================================================================
  // Authentication Endpoints
  // ==========================================================================

  async register(email: string, password: string, fullName?: string) {
    const response = await this.request.post(API_ENDPOINTS.register, {
      data: {
        email,
        password,
        full_name: fullName,
      },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async login(email: string, password: string) {
    const response = await this.request.post(API_ENDPOINTS.loginJson, {
      data: { email, password },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async loginWithForm(username: string, password: string) {
    const response = await this.request.post(API_ENDPOINTS.login, {
      form: { username, password },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async refresh(refreshToken: string) {
    const response = await this.request.post(API_ENDPOINTS.refresh, {
      data: { refresh_token: refreshToken },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async logout(refreshToken?: string) {
    const response = await this.request.post(API_ENDPOINTS.logout, {
      headers: this.getAuthHeaders(),
      data: refreshToken ? { refresh_token: refreshToken } : undefined,
    });
    return { response };
  }

  // ==========================================================================
  // User Endpoints
  // ==========================================================================

  async getProfile() {
    const response = await this.request.get(API_ENDPOINTS.userProfile, {
      headers: this.getAuthHeaders(),
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async updateProfile(updates: { full_name?: string; email?: string }) {
    const response = await this.request.patch(API_ENDPOINTS.userProfile, {
      headers: this.getAuthHeaders(),
      data: updates,
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async changePassword(currentPassword: string, newPassword: string) {
    const response = await this.request.post(API_ENDPOINTS.userPassword, {
      headers: this.getAuthHeaders(),
      data: {
        current_password: currentPassword,
        new_password: newPassword,
      },
    });
    return { response };
  }

  // ==========================================================================
  // GDPR Endpoints
  // ==========================================================================

  async requestDataExport(format: 'json' | 'csv' = 'json') {
    const response = await this.request.post(API_ENDPOINTS.gdprExport, {
      headers: this.getAuthHeaders(),
      data: {
        format,
        include_conversations: true,
        include_sessions: true,
        include_preferences: true,
      },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async getExportStatus(exportId: string) {
    const response = await this.request.get(`${API_ENDPOINTS.gdprExport}/${exportId}`, {
      headers: this.getAuthHeaders(),
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async downloadExport(exportId: string) {
    const response = await this.request.get(`${API_ENDPOINTS.gdprExport}/${exportId}/download`, {
      headers: this.getAuthHeaders(),
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async requestAccountDeletion(confirm: boolean, reason?: string) {
    const response = await this.request.post(API_ENDPOINTS.gdprDelete, {
      headers: this.getAuthHeaders(),
      data: { confirm, reason },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async cancelAccountDeletion() {
    const response = await this.request.post(API_ENDPOINTS.gdprDeleteCancel, {
      headers: this.getAuthHeaders(),
    });
    return { response };
  }

  // ==========================================================================
  // Conversation Endpoints
  // ==========================================================================

  async sendMessage(text: string, sessionId?: string) {
    const response = await this.request.post(API_ENDPOINTS.conversation, {
      headers: this.getAuthHeaders(),
      data: {
        text,
        session_id: sessionId,
      },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async search(query: string, limit: number = 10) {
    const response = await this.request.post(API_ENDPOINTS.search, {
      headers: this.getAuthHeaders(),
      data: { query, limit },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async semanticSearch(query: string, limit: number = 10, threshold: number = 0.5) {
    const response = await this.request.post(API_ENDPOINTS.semanticSearch, {
      headers: this.getAuthHeaders(),
      params: { query, limit, threshold },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async getSessionHistory(sessionId: string, limit: number = 20) {
    const response = await this.request.get(API_ENDPOINTS.sessionHistory(sessionId), {
      headers: this.getAuthHeaders(),
      params: { limit },
    });
    return { response, data: await response.json().catch(() => null) };
  }

  async getStats() {
    const response = await this.request.get(API_ENDPOINTS.stats, {
      headers: this.getAuthHeaders(),
    });
    return { response, data: await response.json().catch(() => null) };
  }
}
