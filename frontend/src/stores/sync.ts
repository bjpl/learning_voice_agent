/**
 * Sync store with Pinia Composition API
 * Manages sync status, data export/import, and device management
 */

import { defineStore } from 'pinia';
import { ref, computed, onMounted, onUnmounted } from 'vue';
import syncService from '@/services/sync';
import type {
  SyncStatus,
  SyncStatusType,
  ExportOptions,
  ImportResult,
  ValidationResult,
  DeviceInfo,
  DeviceRegistration,
  SyncConflict,
  ConflictResolution,
  BackupHistoryItem,
  SyncProgress,
  MergeStrategy,
} from '@/types/sync';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

const CACHE_DURATION = 30 * 1000; // 30 seconds for sync data
const STATUS_REFRESH_INTERVAL = 60 * 1000; // 1 minute auto-refresh

export const useSyncStore = defineStore('sync', () => {
  // Core state
  const status = ref<SyncStatus | null>(null);
  const devices = ref<DeviceInfo[]>([]);
  const conflicts = ref<SyncConflict[]>([]);
  const backupHistory = ref<BackupHistoryItem[]>([]);

  // Loading states
  const isLoading = ref(false);
  const isExporting = ref(false);
  const isImporting = ref(false);
  const isSyncing = ref(false);
  const isValidating = ref(false);

  // Progress tracking
  const exportProgress = ref<SyncProgress | null>(null);
  const importProgress = ref<SyncProgress | null>(null);

  // Error state
  const error = ref<string | null>(null);
  const lastUpdated = ref<Date | null>(null);

  // Validation state
  const lastValidationResult = ref<ValidationResult | null>(null);

  // Cache storage
  const cache = ref<Map<string, CacheEntry<unknown>>>(new Map());

  // Auto-refresh interval
  let statusRefreshInterval: ReturnType<typeof setInterval> | null = null;

  // Computed properties
  const syncStatusType = computed((): SyncStatusType => {
    return status.value?.status ?? 'idle';
  });

  const lastSyncTime = computed((): Date | null => {
    if (!status.value?.lastSyncAt) return null;
    return new Date(status.value.lastSyncAt);
  });

  const lastSyncFormatted = computed((): string => {
    if (!lastSyncTime.value) return 'Never';
    return formatRelativeTime(lastSyncTime.value);
  });

  const hasPendingChanges = computed((): boolean => {
    return (status.value?.pendingChanges ?? 0) > 0;
  });

  const pendingChangesCount = computed((): number => {
    return status.value?.pendingChanges ?? 0;
  });

  const hasConflicts = computed((): boolean => {
    return conflicts.value.length > 0;
  });

  const conflictCount = computed((): number => {
    return conflicts.value.length;
  });

  const autoResolvableConflicts = computed((): SyncConflict[] => {
    return conflicts.value.filter((c) => c.autoResolvable);
  });

  const manualConflicts = computed((): SyncConflict[] => {
    return conflicts.value.filter((c) => !c.autoResolvable);
  });

  const currentDevice = computed((): DeviceInfo | null => {
    return devices.value.find((d) => d.isCurrentDevice) ?? null;
  });

  const otherDevices = computed((): DeviceInfo[] => {
    return devices.value.filter((d) => !d.isCurrentDevice);
  });

  const trustedDevices = computed((): DeviceInfo[] => {
    return devices.value.filter((d) => d.trusted);
  });

  const storageUsagePercent = computed((): number => {
    if (!status.value?.storageLimit) return 0;
    return Math.round((status.value.storageUsed / status.value.storageLimit) * 100);
  });

  const storageUsageFormatted = computed((): string => {
    if (!status.value) return '0 B / 0 B';
    return `${formatBytes(status.value.storageUsed)} / ${formatBytes(status.value.storageLimit)}`;
  });

  const isOnline = computed((): boolean => {
    return status.value?.status !== 'offline';
  });

  const canSync = computed((): boolean => {
    return isOnline.value && !isSyncing.value && !isExporting.value && !isImporting.value;
  });

  // Actions
  const fetchStatus = async (forceRefresh = false): Promise<SyncStatus> => {
    const cacheKey = 'status';

    if (!forceRefresh) {
      const cached = getFromCache<SyncStatus>(cacheKey);
      if (cached) {
        status.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await syncService.getSyncStatus();
      status.value = apiData;
      setCache(cacheKey, apiData);
      lastUpdated.value = new Date();
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch sync status';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const triggerSync = async (): Promise<SyncStatus> => {
    if (!canSync.value) {
      throw new Error('Sync is not available at this time');
    }

    isSyncing.value = true;
    error.value = null;

    try {
      const result = await syncService.triggerSync();
      status.value = result;
      setCache('status', result);
      lastUpdated.value = new Date();

      // Refresh conflicts after sync
      await fetchConflicts(true);

      return result;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Sync failed';
      throw e;
    } finally {
      isSyncing.value = false;
    }
  };

  const exportData = async (
    options: Partial<ExportOptions> = {},
    withProgress = false
  ): Promise<void> => {
    isExporting.value = true;
    exportProgress.value = null;
    error.value = null;

    try {
      let blob: Blob;

      if (withProgress) {
        blob = await syncService.exportDataWithProgress(options, (progress) => {
          exportProgress.value = progress;
        });
      } else {
        blob = await syncService.exportData(options);
      }

      // Generate filename and trigger download
      const filename = syncService.generateExportFilename(options.format ?? 'json');
      syncService.downloadBlob(blob, filename);

      exportProgress.value = {
        operation: 'export',
        status: 'completed',
        currentStep: 1,
        totalSteps: 1,
        currentItem: 'Export complete',
        itemsProcessed: 0,
        totalItems: 0,
        bytesProcessed: blob.size,
        totalBytes: blob.size,
        estimatedTimeRemaining: null,
        startedAt: new Date().toISOString(),
        errors: [],
      };
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Export failed';
      if (exportProgress.value) {
        exportProgress.value.status = 'failed';
        exportProgress.value.errors.push(error.value);
      }
      throw e;
    } finally {
      isExporting.value = false;
    }
  };

  const validateBackup = async (file: File): Promise<ValidationResult> => {
    isValidating.value = true;
    error.value = null;

    try {
      const result = await syncService.validateBackup(file);
      lastValidationResult.value = result;
      return result;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Validation failed';
      throw e;
    } finally {
      isValidating.value = false;
    }
  };

  const importData = async (
    file: File,
    strategy: MergeStrategy = 'merge_newest',
    withProgress = false
  ): Promise<ImportResult> => {
    isImporting.value = true;
    importProgress.value = null;
    error.value = null;

    try {
      let result: ImportResult;

      if (withProgress) {
        result = await syncService.importDataWithProgress(
          file,
          { strategy },
          (progress) => {
            importProgress.value = progress;
          }
        );
      } else {
        result = await syncService.importData(file, { strategy });
      }

      // Update progress to completed
      importProgress.value = {
        operation: 'import',
        status: result.success ? 'completed' : 'failed',
        currentStep: 2,
        totalSteps: 2,
        currentItem: result.success ? 'Import complete' : 'Import failed',
        itemsProcessed: result.importedItems,
        totalItems: result.totalItems,
        bytesProcessed: file.size,
        totalBytes: file.size,
        estimatedTimeRemaining: null,
        startedAt: new Date().toISOString(),
        errors: result.errors.map((e) => e.error),
      };

      // Handle conflicts from import
      if (result.conflicts.length > 0) {
        conflicts.value = [...conflicts.value, ...result.conflicts];
      }

      // Refresh sync status after import
      await fetchStatus(true);

      return result;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Import failed';
      if (importProgress.value) {
        importProgress.value.status = 'failed';
        importProgress.value.errors.push(error.value);
      }
      throw e;
    } finally {
      isImporting.value = false;
    }
  };

  const fetchDevices = async (forceRefresh = false): Promise<DeviceInfo[]> => {
    const cacheKey = 'devices';

    if (!forceRefresh) {
      const cached = getFromCache<DeviceInfo[]>(cacheKey);
      if (cached) {
        devices.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await syncService.getDevices();
      devices.value = apiData;
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch devices';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const registerDevice = async (
    registration: DeviceRegistration
  ): Promise<DeviceInfo> => {
    isLoading.value = true;
    error.value = null;

    try {
      const device = await syncService.registerDevice(registration);
      devices.value.push(device);
      clearCache('devices');
      return device;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to register device';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const removeDevice = async (deviceId: string): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      await syncService.removeDevice(deviceId);
      devices.value = devices.value.filter((d) => d.id !== deviceId);
      clearCache('devices');
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to remove device';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const setDeviceTrust = async (
    deviceId: string,
    trusted: boolean
  ): Promise<void> => {
    error.value = null;

    try {
      const updatedDevice = await syncService.setDeviceTrust(deviceId, trusted);
      const index = devices.value.findIndex((d) => d.id === deviceId);
      if (index !== -1) {
        devices.value[index] = updatedDevice;
      }
      clearCache('devices');
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to update device trust';
      throw e;
    }
  };

  const fetchConflicts = async (forceRefresh = false): Promise<SyncConflict[]> => {
    const cacheKey = 'conflicts';

    if (!forceRefresh) {
      const cached = getFromCache<SyncConflict[]>(cacheKey);
      if (cached) {
        conflicts.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await syncService.getConflicts();
      conflicts.value = apiData;
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch conflicts';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const resolveConflict = async (resolution: ConflictResolution): Promise<void> => {
    error.value = null;

    try {
      await syncService.resolveConflict(resolution);
      conflicts.value = conflicts.value.filter(
        (c) => c.id !== resolution.conflictId
      );
      clearCache('conflicts');
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to resolve conflict';
      throw e;
    }
  };

  const resolveAllConflicts = async (
    resolutions: ConflictResolution[]
  ): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      await syncService.resolveConflicts(resolutions);
      const resolvedIds = new Set(resolutions.map((r) => r.conflictId));
      conflicts.value = conflicts.value.filter((c) => !resolvedIds.has(c.id));
      clearCache('conflicts');
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to resolve conflicts';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const autoResolveConflicts = async (): Promise<number> => {
    isLoading.value = true;
    error.value = null;

    try {
      const resolvedCount = await syncService.autoResolveConflicts();
      await fetchConflicts(true);
      return resolvedCount;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to auto-resolve conflicts';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchBackupHistory = async (
    limit = 10,
    forceRefresh = false
  ): Promise<BackupHistoryItem[]> => {
    const cacheKey = `backups_${limit}`;

    if (!forceRefresh) {
      const cached = getFromCache<BackupHistoryItem[]>(cacheKey);
      if (cached) {
        backupHistory.value = cached;
        return cached;
      }
    }

    isLoading.value = true;
    error.value = null;

    try {
      const apiData = await syncService.getBackupHistory(limit);
      backupHistory.value = apiData;
      setCache(cacheKey, apiData);
      return apiData;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch backup history';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const createBackup = async (): Promise<BackupHistoryItem> => {
    isLoading.value = true;
    error.value = null;

    try {
      const backup = await syncService.createBackup();
      backupHistory.value.unshift(backup);
      clearCache('backups_10');
      return backup;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to create backup';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const downloadBackup = async (backupId: string, filename?: string): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      const blob = await syncService.downloadBackup(backupId);
      const finalFilename = filename ?? `backup-${backupId}.json`;
      syncService.downloadBlob(blob, finalFilename);
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to download backup';
      throw e;
    } finally {
      isLoading.value = false;
    }
  };

  const deleteBackup = async (backupId: string): Promise<void> => {
    error.value = null;

    try {
      await syncService.deleteBackup(backupId);
      backupHistory.value = backupHistory.value.filter((b) => b.id !== backupId);
      clearCache('backups_10');
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to delete backup';
      throw e;
    }
  };

  const restoreBackup = async (backupId: string): Promise<ImportResult> => {
    isImporting.value = true;
    error.value = null;

    try {
      const result = await syncService.restoreBackup(backupId);
      await fetchStatus(true);
      return result;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to restore backup';
      throw e;
    } finally {
      isImporting.value = false;
    }
  };

  const clearError = (): void => {
    error.value = null;
  };

  const clearCache = (key?: string): void => {
    if (key) {
      cache.value.delete(key);
    } else {
      cache.value.clear();
    }
  };

  const resetProgress = (): void => {
    exportProgress.value = null;
    importProgress.value = null;
  };

  // Start auto-refresh of status
  const startAutoRefresh = (): void => {
    if (statusRefreshInterval) return;

    statusRefreshInterval = setInterval(() => {
      if (!isExporting.value && !isImporting.value && !isSyncing.value) {
        fetchStatus().catch(() => {
          // Silently handle refresh errors
        });
      }
    }, STATUS_REFRESH_INTERVAL);
  };

  // Stop auto-refresh
  const stopAutoRefresh = (): void => {
    if (statusRefreshInterval) {
      clearInterval(statusRefreshInterval);
      statusRefreshInterval = null;
    }
  };

  // Initialize on mount
  const initialize = async (): Promise<void> => {
    await Promise.all([
      fetchStatus(),
      fetchDevices(),
      fetchConflicts(),
    ]);
    startAutoRefresh();
  };

  // Cache helpers
  function getFromCache<T>(key: string): T | null {
    const entry = cache.value.get(key) as CacheEntry<T> | undefined;

    if (!entry) return null;

    if (Date.now() > entry.expiresAt) {
      cache.value.delete(key);
      return null;
    }

    return entry.data;
  }

  function setCache<T>(key: string, cacheData: T): void {
    const entry: CacheEntry<T> = {
      data: cacheData,
      timestamp: Date.now(),
      expiresAt: Date.now() + CACHE_DURATION,
    };
    cache.value.set(key, entry as CacheEntry<unknown>);
  }

  // Utility functions
  function formatRelativeTime(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffSec < 60) return 'Just now';
    if (diffMin < 60) return `${diffMin} minute${diffMin > 1 ? 's' : ''} ago`;
    if (diffHour < 24) return `${diffHour} hour${diffHour > 1 ? 's' : ''} ago`;
    if (diffDay < 7) return `${diffDay} day${diffDay > 1 ? 's' : ''} ago`;

    return date.toLocaleDateString();
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  }

  // Lifecycle hooks for components using this store
  onMounted(() => {
    initialize();
  });

  onUnmounted(() => {
    stopAutoRefresh();
  });

  return {
    // State
    status,
    devices,
    conflicts,
    backupHistory,
    isLoading,
    isExporting,
    isImporting,
    isSyncing,
    isValidating,
    exportProgress,
    importProgress,
    error,
    lastUpdated,
    lastValidationResult,

    // Computed
    syncStatusType,
    lastSyncTime,
    lastSyncFormatted,
    hasPendingChanges,
    pendingChangesCount,
    hasConflicts,
    conflictCount,
    autoResolvableConflicts,
    manualConflicts,
    currentDevice,
    otherDevices,
    trustedDevices,
    storageUsagePercent,
    storageUsageFormatted,
    isOnline,
    canSync,

    // Actions
    fetchStatus,
    triggerSync,
    exportData,
    validateBackup,
    importData,
    fetchDevices,
    registerDevice,
    removeDevice,
    setDeviceTrust,
    fetchConflicts,
    resolveConflict,
    resolveAllConflicts,
    autoResolveConflicts,
    fetchBackupHistory,
    createBackup,
    downloadBackup,
    deleteBackup,
    restoreBackup,
    clearError,
    clearCache,
    resetProgress,
    initialize,
    startAutoRefresh,
    stopAutoRefresh,
  };
});

export type SyncStore = ReturnType<typeof useSyncStore>;
