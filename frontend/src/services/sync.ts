/**
 * Sync, backup, and data export/import API service
 */

import { apiClient } from './api';
import type {
  ApiResponse,
} from '@/types';
import type {
  SyncStatus,
  ExportOptions,
  ImportOptions,
  ImportResult,
  ValidationResult,
  DeviceInfo,
  DeviceRegistration,
  SyncConflict,
  ConflictResolution,
  BackupHistoryItem,
  SyncProgress,
} from '@/types/sync';

/**
 * Get current sync status
 */
export const getSyncStatus = async (): Promise<SyncStatus> => {
  const response = await apiClient.get<ApiResponse<SyncStatus>>(
    '/api/sync/status'
  );

  return response.data.data;
};

/**
 * Trigger a manual sync
 */
export const triggerSync = async (): Promise<SyncStatus> => {
  const response = await apiClient.post<ApiResponse<SyncStatus>>(
    '/api/sync/trigger'
  );

  return response.data.data;
};

/**
 * Export user data as a downloadable file
 */
export const exportData = async (
  options: Partial<ExportOptions> = {}
): Promise<Blob> => {
  const defaultOptions: ExportOptions = {
    format: 'json',
    includeConversations: true,
    includeGoals: true,
    includeAchievements: true,
    includeFeedback: true,
    includeSettings: true,
    includeAnalytics: true,
    encrypted: false,
    compression: 'none',
    ...options,
  };

  const response = await apiClient.post('/api/sync/export', defaultOptions, {
    responseType: 'blob',
    timeout: 120000, // 2 minutes for large exports
  });

  return response.data;
};

/**
 * Export data with progress tracking
 */
export const exportDataWithProgress = async (
  options: Partial<ExportOptions> = {},
  onProgress?: (progress: SyncProgress) => void
): Promise<Blob> => {
  const defaultOptions: ExportOptions = {
    format: 'json',
    includeConversations: true,
    includeGoals: true,
    includeAchievements: true,
    includeFeedback: true,
    includeSettings: true,
    includeAnalytics: true,
    encrypted: false,
    compression: 'none',
    ...options,
  };

  const response = await apiClient.post('/api/sync/export', defaultOptions, {
    responseType: 'blob',
    timeout: 120000,
    onDownloadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        onProgress({
          operation: 'export',
          status: 'in_progress',
          currentStep: 1,
          totalSteps: 1,
          currentItem: 'Downloading export',
          itemsProcessed: 0,
          totalItems: 0,
          bytesProcessed: progressEvent.loaded,
          totalBytes: progressEvent.total,
          estimatedTimeRemaining: null,
          startedAt: new Date().toISOString(),
          errors: [],
        });
      }
    },
  });

  return response.data;
};

/**
 * Import data from a backup file
 */
export const importData = async (
  file: File,
  options: Partial<Omit<ImportOptions, 'file'>> = {}
): Promise<ImportResult> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('strategy', options.strategy || 'merge_newest');
  formData.append('validateOnly', String(options.validateOnly || false));
  formData.append('skipConflicts', String(options.skipConflicts || false));
  formData.append('preserveLocalChanges', String(options.preserveLocalChanges || true));

  if (options.password) {
    formData.append('password', options.password);
  }

  const response = await apiClient.post<ApiResponse<ImportResult>>(
    '/api/sync/import',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes for large imports
    }
  );

  return response.data.data;
};

/**
 * Import data with progress tracking for large files
 */
export const importDataWithProgress = async (
  file: File,
  options: Partial<Omit<ImportOptions, 'file'>> = {},
  onProgress?: (progress: SyncProgress) => void
): Promise<ImportResult> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('strategy', options.strategy || 'merge_newest');
  formData.append('validateOnly', String(options.validateOnly || false));
  formData.append('skipConflicts', String(options.skipConflicts || false));
  formData.append('preserveLocalChanges', String(options.preserveLocalChanges || true));

  if (options.password) {
    formData.append('password', options.password);
  }

  const response = await apiClient.post<ApiResponse<ImportResult>>(
    '/api/sync/import',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000,
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          onProgress({
            operation: 'import',
            status: 'in_progress',
            currentStep: 1,
            totalSteps: 2,
            currentItem: 'Uploading file',
            itemsProcessed: 0,
            totalItems: 0,
            bytesProcessed: progressEvent.loaded,
            totalBytes: progressEvent.total,
            estimatedTimeRemaining: null,
            startedAt: new Date().toISOString(),
            errors: [],
          });
        }
      },
    }
  );

  return response.data.data;
};

/**
 * Validate a backup file before import
 */
export const validateBackup = async (file: File): Promise<ValidationResult> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<ApiResponse<ValidationResult>>(
    '/api/sync/validate',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 1 minute for validation
    }
  );

  return response.data.data;
};

/**
 * Get list of registered devices
 */
export const getDevices = async (): Promise<DeviceInfo[]> => {
  const response = await apiClient.get<ApiResponse<DeviceInfo[]>>(
    '/api/sync/devices'
  );

  return response.data.data;
};

/**
 * Register current device for sync
 */
export const registerDevice = async (
  registration: DeviceRegistration
): Promise<DeviceInfo> => {
  const response = await apiClient.post<ApiResponse<DeviceInfo>>(
    '/api/sync/devices',
    registration
  );

  return response.data.data;
};

/**
 * Update device information
 */
export const updateDevice = async (
  deviceId: string,
  updates: Partial<DeviceRegistration>
): Promise<DeviceInfo> => {
  const response = await apiClient.patch<ApiResponse<DeviceInfo>>(
    `/api/sync/devices/${deviceId}`,
    updates
  );

  return response.data.data;
};

/**
 * Remove a device from sync
 */
export const removeDevice = async (deviceId: string): Promise<void> => {
  await apiClient.delete(`/api/sync/devices/${deviceId}`);
};

/**
 * Trust or untrust a device
 */
export const setDeviceTrust = async (
  deviceId: string,
  trusted: boolean
): Promise<DeviceInfo> => {
  const response = await apiClient.patch<ApiResponse<DeviceInfo>>(
    `/api/sync/devices/${deviceId}/trust`,
    { trusted }
  );

  return response.data.data;
};

/**
 * Get list of unresolved conflicts
 */
export const getConflicts = async (): Promise<SyncConflict[]> => {
  const response = await apiClient.get<ApiResponse<SyncConflict[]>>(
    '/api/sync/conflicts'
  );

  return response.data.data;
};

/**
 * Resolve multiple conflicts
 */
export const resolveConflicts = async (
  resolutions: ConflictResolution[]
): Promise<void> => {
  await apiClient.post('/api/sync/conflicts/resolve', { resolutions });
};

/**
 * Resolve a single conflict
 */
export const resolveConflict = async (
  resolution: ConflictResolution
): Promise<void> => {
  await apiClient.post(
    `/api/sync/conflicts/${resolution.conflictId}/resolve`,
    resolution
  );
};

/**
 * Auto-resolve all resolvable conflicts
 */
export const autoResolveConflicts = async (): Promise<number> => {
  const response = await apiClient.post<ApiResponse<{ resolved: number }>>(
    '/api/sync/conflicts/auto-resolve'
  );

  return response.data.data.resolved;
};

/**
 * Get backup history
 */
export const getBackupHistory = async (
  limit = 10
): Promise<BackupHistoryItem[]> => {
  const response = await apiClient.get<ApiResponse<BackupHistoryItem[]>>(
    '/api/sync/backups',
    { params: { limit } }
  );

  return response.data.data;
};

/**
 * Create a manual backup
 */
export const createBackup = async (): Promise<BackupHistoryItem> => {
  const response = await apiClient.post<ApiResponse<BackupHistoryItem>>(
    '/api/sync/backups'
  );

  return response.data.data;
};

/**
 * Download a backup by ID
 */
export const downloadBackup = async (backupId: string): Promise<Blob> => {
  const response = await apiClient.get(`/api/sync/backups/${backupId}/download`, {
    responseType: 'blob',
    timeout: 120000,
  });

  return response.data;
};

/**
 * Delete a backup
 */
export const deleteBackup = async (backupId: string): Promise<void> => {
  await apiClient.delete(`/api/sync/backups/${backupId}`);
};

/**
 * Restore from a backup
 */
export const restoreBackup = async (backupId: string): Promise<ImportResult> => {
  const response = await apiClient.post<ApiResponse<ImportResult>>(
    `/api/sync/backups/${backupId}/restore`
  );

  return response.data.data;
};

/**
 * Helper function to download a blob as a file
 */
export const downloadBlob = (
  blob: Blob,
  filename: string
): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

/**
 * Generate a filename for exports
 */
export const generateExportFilename = (
  format: 'json' | 'csv' | 'zip' = 'json'
): string => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `voice-agent-backup-${timestamp}.${format}`;
};

/**
 * Get current sync progress (for long-running operations)
 */
export const getSyncProgress = async (
  operationId: string
): Promise<SyncProgress> => {
  const response = await apiClient.get<ApiResponse<SyncProgress>>(
    `/api/sync/progress/${operationId}`
  );

  return response.data.data;
};

export default {
  getSyncStatus,
  triggerSync,
  exportData,
  exportDataWithProgress,
  importData,
  importDataWithProgress,
  validateBackup,
  getDevices,
  registerDevice,
  updateDevice,
  removeDevice,
  setDeviceTrust,
  getConflicts,
  resolveConflicts,
  resolveConflict,
  autoResolveConflicts,
  getBackupHistory,
  createBackup,
  downloadBackup,
  deleteBackup,
  restoreBackup,
  downloadBlob,
  generateExportFilename,
  getSyncProgress,
};
