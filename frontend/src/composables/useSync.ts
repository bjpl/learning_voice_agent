/**
 * Sync Composable - Data synchronization and backup management
 *
 * PATTERN: Reactive sync state management
 * WHY: Centralized sync functionality for exports, imports, and device management
 */
import { ref, computed, onMounted } from 'vue';
import type {
  SyncStatusType,
  BackupData,
  BackupMetadata,
  BackupHistoryItem,
  DeviceInfo,
  DeviceType,
  ExportOptions,
  ImportOptions,
  ImportResult,
  ValidationResult,
  SyncConflict,
  ConflictResolution,
  SyncItemCounts,
  MergeStrategy,
} from '@/types/sync';

// Sync state
const syncStatus = ref<SyncStatusType>('idle');
const lastSyncAt = ref<string | null>(null);
const nextAutoBackup = ref<Date | null>(null);
const dataSize = ref<number>(0);
const syncError = ref<string | null>(null);
const isExporting = ref(false);
const isImporting = ref(false);
const exportProgress = ref(0);
const importProgress = ref(0);
const devices = ref<DeviceInfo[]>([]);
const backupHistory = ref<BackupHistoryItem[]>([]);
const conflicts = ref<SyncConflict[]>([]);

// Constants
const APP_VERSION = '1.0.0';
const AUTO_BACKUP_INTERVAL = 24 * 60 * 60 * 1000; // 24 hours
const STORAGE_KEY_PREFIX = 'lva_sync_';

export function useSync() {
  /**
   * Calculate total data size from localStorage
   */
  const calculateDataSize = (): number => {
    let total = 0;
    for (const key in localStorage) {
      if (localStorage.hasOwnProperty(key)) {
        total += localStorage.getItem(key)?.length || 0;
      }
    }
    return total * 2; // UTF-16 encoding
  };

  /**
   * Generate a simple checksum for data integrity
   */
  const generateChecksum = (data: string): string => {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  };

  /**
   * Get current device ID or generate one
   */
  const getDeviceId = (): string => {
    let deviceId = localStorage.getItem(`${STORAGE_KEY_PREFIX}device_id`);
    if (!deviceId) {
      deviceId = `device_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem(`${STORAGE_KEY_PREFIX}device_id`, deviceId);
    }
    return deviceId;
  };

  /**
   * Detect current device type
   */
  const detectDeviceType = (): DeviceType => {
    const userAgent = navigator.userAgent.toLowerCase();
    if (/ipad|tablet|playbook|silk/.test(userAgent)) return 'tablet';
    if (/mobile|iphone|ipod|android/.test(userAgent)) return 'mobile';
    if (/electron/.test(userAgent)) return 'desktop';
    return 'web';
  };

  /**
   * Get a friendly device name
   */
  const getDeviceName = (): string => {
    const ua = navigator.userAgent;
    if (/iPhone/.test(ua)) return 'iPhone';
    if (/iPad/.test(ua)) return 'iPad';
    if (/Android/.test(ua)) return 'Android Device';
    if (/Mac/.test(ua)) return 'Mac';
    if (/Windows/.test(ua)) return 'Windows PC';
    if (/Linux/.test(ua)) return 'Linux PC';
    return 'Unknown Device';
  };

  /**
   * Get platform string
   */
  const getPlatform = (): string => {
    const ua = navigator.userAgent;
    if (/iPhone|iPad|iPod/.test(ua)) return 'iOS';
    if (/Android/.test(ua)) return 'Android';
    if (/Mac/.test(ua)) return 'macOS';
    if (/Windows/.test(ua)) return 'Windows';
    if (/Linux/.test(ua)) return 'Linux';
    return navigator.platform || 'Unknown';
  };

  /**
   * Count items in localStorage collections
   */
  const countItems = (): SyncItemCounts => {
    const getCount = (key: string): number => {
      try {
        const data = localStorage.getItem(key);
        if (data) {
          const parsed = JSON.parse(data);
          return Array.isArray(parsed) ? parsed.length : 1;
        }
      } catch {
        // ignore
      }
      return 0;
    };

    return {
      conversations: getCount('lva_conversations'),
      messages: 0, // Messages are embedded in conversations
      goals: getCount('lva_goals'),
      achievements: getCount('lva_achievements'),
      feedback: getCount('lva_feedback'),
      settings: localStorage.getItem('lva_settings') ? 1 : 0,
      analytics: getCount('lva_analytics'),
    };
  };

  /**
   * Export user data with selected options
   */
  const exportData = async (options: Partial<ExportOptions>): Promise<Blob> => {
    isExporting.value = true;
    exportProgress.value = 0;
    syncStatus.value = 'syncing';

    try {
      const itemCounts = countItems();
      const deviceId = getDeviceId();

      const metadata: BackupMetadata = {
        version: '1.0',
        createdAt: new Date().toISOString(),
        createdBy: 'user',
        deviceId,
        deviceName: getDeviceName(),
        appVersion: APP_VERSION,
        checksum: '',
        itemCounts,
        encrypted: options.encrypted || false,
        compressionType: options.compression || 'none',
      };

      const exportedData: BackupData = {
        metadata,
        conversations: [],
        goals: [],
        achievements: [],
        feedback: [],
        settings: {
          language: 'en',
          theme: 'system',
          notifications: {
            enabled: true,
            sound: true,
            dailyReminder: false,
            reminderTime: null,
            achievementAlerts: true,
            streakAlerts: true,
          },
          privacy: {
            shareAnalytics: false,
            shareProgress: false,
            publicProfile: false,
          },
          accessibility: {
            fontSize: 'medium',
            highContrast: false,
            reduceMotion: false,
            screenReaderOptimized: false,
          },
        },
        analytics: [],
      };

      // Gather data based on options
      if (options.includeConversations !== false) {
        exportProgress.value = 20;
        const conversationsData = localStorage.getItem('lva_conversations');
        if (conversationsData) {
          let conversations = JSON.parse(conversationsData);
          // Filter by date range if specified
          if (options.dateRange) {
            const startDate = new Date(options.dateRange.start);
            const endDate = new Date(options.dateRange.end);
            conversations = conversations.filter((c: { createdAt: string }) => {
              const date = new Date(c.createdAt);
              return date >= startDate && date <= endDate;
            });
          }
          exportedData.conversations = conversations;
        }
      }

      if (options.includeGoals !== false) {
        exportProgress.value = 40;
        const goalsData = localStorage.getItem('lva_goals');
        if (goalsData) {
          exportedData.goals = JSON.parse(goalsData);
        }
      }

      if (options.includeAchievements !== false) {
        exportProgress.value = 60;
        const achievementsData = localStorage.getItem('lva_achievements');
        if (achievementsData) {
          exportedData.achievements = JSON.parse(achievementsData);
        }
      }

      if (options.includeFeedback !== false) {
        exportProgress.value = 70;
        const feedbackData = localStorage.getItem('lva_feedback');
        if (feedbackData) {
          exportedData.feedback = JSON.parse(feedbackData);
        }
      }

      if (options.includeAnalytics !== false) {
        exportProgress.value = 80;
        const analyticsData = localStorage.getItem('lva_analytics');
        if (analyticsData) {
          exportedData.analytics = JSON.parse(analyticsData);
        }
      }

      if (options.includeSettings !== false) {
        exportProgress.value = 90;
        const settingsData = localStorage.getItem('lva_settings');
        if (settingsData) {
          exportedData.settings = JSON.parse(settingsData);
        }
      }

      // Generate checksum
      const dataString = JSON.stringify(exportedData);
      exportedData.metadata.checksum = generateChecksum(dataString);

      exportProgress.value = 100;
      syncStatus.value = 'synced';
      lastSyncAt.value = new Date().toISOString();

      // Add to backup history
      const backupItem: BackupHistoryItem = {
        id: `backup_${Date.now()}`,
        createdAt: new Date().toISOString(),
        deviceId,
        deviceName: getDeviceName(),
        size: dataString.length * 2,
        itemCounts: exportedData.metadata.itemCounts,
        type: 'manual',
        status: 'completed',
      };
      backupHistory.value.unshift(backupItem);
      saveBackupHistory();

      return new Blob([JSON.stringify(exportedData, null, 2)], {
        type: 'application/json',
      });
    } catch (error) {
      syncStatus.value = 'error';
      syncError.value = error instanceof Error ? error.message : 'Export failed';
      throw error;
    } finally {
      isExporting.value = false;
    }
  };

  /**
   * Validate backup file before import
   */
  const validateBackup = async (file: File): Promise<ValidationResult> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          const backup: BackupData = JSON.parse(content);

          // Validate structure
          if (!backup.metadata || !backup.metadata.version) {
            throw new Error('Invalid backup file format');
          }

          // Verify checksum
          const storedChecksum = backup.metadata.checksum;
          backup.metadata.checksum = '';
          const dataString = JSON.stringify(backup);
          const calculatedChecksum = generateChecksum(dataString);
          backup.metadata.checksum = storedChecksum;
          const checksumValid = storedChecksum === calculatedChecksum;

          // Check version compatibility
          const isCompatible = compareVersions(backup.metadata.version, '1.0') >= 0;

          const warnings: string[] = [];
          if (!checksumValid) {
            warnings.push('Checksum mismatch - file may be corrupted');
          }
          if (!isCompatible) {
            warnings.push(`Backup version (${backup.metadata.version}) may not be fully compatible`);
          }

          resolve({
            valid: true,
            version: backup.metadata.version,
            compatible: isCompatible,
            itemCounts: backup.metadata.itemCounts,
            encrypted: backup.metadata.encrypted,
            checksum: storedChecksum,
            checksumValid,
            errors: [],
            warnings,
            estimatedImportTime: Math.ceil(content.length / 100000), // ~1 second per 100KB
          });
        } catch (error) {
          reject(new Error('Failed to parse backup file'));
        }
      };

      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

  /**
   * Import data from backup file
   */
  const importData = async (
    file: File,
    options: Partial<ImportOptions>
  ): Promise<ImportResult> => {
    isImporting.value = true;
    importProgress.value = 0;
    syncStatus.value = 'syncing';
    conflicts.value = [];

    const result: ImportResult = {
      success: false,
      totalItems: 0,
      importedItems: 0,
      skippedItems: 0,
      conflictCount: 0,
      conflicts: [],
      errors: [],
      duration: 0,
      warnings: [],
    };

    const startTime = Date.now();

    try {
      const content = await file.text();
      const backup: BackupData = JSON.parse(content);
      const strategy = options.strategy || 'merge_newest';

      importProgress.value = 20;

      // Count total items
      result.totalItems =
        (backup.conversations?.length || 0) +
        (backup.goals?.length || 0) +
        (backup.achievements?.length || 0) +
        (backup.feedback?.length || 0) +
        (backup.analytics?.length || 0) +
        (backup.settings ? 1 : 0);

      // Import conversations
      if (backup.conversations?.length) {
        importProgress.value = 40;
        const imported = await importCollection(
          'lva_conversations',
          backup.conversations,
          strategy,
          'conversation'
        );
        result.importedItems += imported.imported;
        result.skippedItems += imported.skipped;
      }

      // Import goals
      if (backup.goals?.length) {
        importProgress.value = 55;
        const imported = await importCollection(
          'lva_goals',
          backup.goals,
          strategy,
          'goal'
        );
        result.importedItems += imported.imported;
        result.skippedItems += imported.skipped;
      }

      // Import achievements
      if (backup.achievements?.length) {
        importProgress.value = 70;
        const imported = await importCollection(
          'lva_achievements',
          backup.achievements,
          strategy,
          'achievement'
        );
        result.importedItems += imported.imported;
        result.skippedItems += imported.skipped;
      }

      // Import feedback
      if (backup.feedback?.length) {
        importProgress.value = 80;
        const imported = await importCollection(
          'lva_feedback',
          backup.feedback,
          strategy,
          'feedback'
        );
        result.importedItems += imported.imported;
        result.skippedItems += imported.skipped;
      }

      // Import analytics
      if (backup.analytics?.length) {
        importProgress.value = 90;
        if (strategy === 'replace_all') {
          localStorage.setItem('lva_analytics', JSON.stringify(backup.analytics));
          result.importedItems += backup.analytics.length;
        }
      }

      // Import settings
      if (backup.settings) {
        importProgress.value = 95;
        if (strategy === 'replace_all') {
          localStorage.setItem('lva_settings', JSON.stringify(backup.settings));
          result.importedItems += 1;
        } else if (strategy === 'merge_newest') {
          const existingSettings = JSON.parse(localStorage.getItem('lva_settings') || '{}');
          const merged = { ...existingSettings, ...backup.settings };
          localStorage.setItem('lva_settings', JSON.stringify(merged));
          result.importedItems += 1;
        }
      }

      importProgress.value = 100;
      result.success = true;
      result.conflictCount = conflicts.value.length;
      result.conflicts = [...conflicts.value];
      syncStatus.value = conflicts.value.length > 0 ? 'conflict' : 'synced';
      lastSyncAt.value = new Date().toISOString();
      dataSize.value = calculateDataSize();
    } catch (error) {
      syncStatus.value = 'error';
      syncError.value = error instanceof Error ? error.message : 'Import failed';
      result.errors.push({
        itemType: 'backup',
        itemId: 'root',
        error: syncError.value,
        recoverable: false,
      });
    } finally {
      isImporting.value = false;
      result.duration = Date.now() - startTime;
    }

    return result;
  };

  /**
   * Import a collection with merge strategy
   */
  const importCollection = async (
    key: string,
    newItems: unknown[],
    strategy: MergeStrategy,
    itemType: SyncConflict['itemType']
  ): Promise<{ imported: number; skipped: number }> => {
    const existingData = localStorage.getItem(key);
    const existingItems: Array<{ id: string; updatedAt?: string; createdAt?: string }> =
      existingData ? JSON.parse(existingData) : [];

    let imported = 0;
    let skipped = 0;
    let finalItems: unknown[];

    switch (strategy) {
      case 'replace_all':
        finalItems = newItems;
        imported = newItems.length;
        break;

      case 'keep_local':
        finalItems = existingItems;
        skipped = newItems.length;
        break;

      case 'keep_remote':
        finalItems = newItems;
        imported = newItems.length;
        break;

      case 'merge_newest':
      case 'merge_manual': {
        const existingMap = new Map(existingItems.map((item) => [item.id, item]));
        const mergedItems = [...existingItems];

        for (const newItem of newItems as Array<{ id: string; updatedAt?: string; createdAt?: string }>) {
          const existing = existingMap.get(newItem.id);
          if (existing) {
            const existingDate = new Date(existing.updatedAt || existing.createdAt || 0);
            const newDate = new Date(newItem.updatedAt || newItem.createdAt || 0);

            if (strategy === 'merge_newest') {
              if (newDate > existingDate) {
                const idx = mergedItems.findIndex((i) => (i as { id: string }).id === newItem.id);
                if (idx >= 0) {
                  mergedItems[idx] = newItem;
                  imported++;
                }
              } else {
                skipped++;
              }
            } else {
              // merge_manual - create conflict
              conflicts.value.push({
                id: `conflict_${Date.now()}_${newItem.id}`,
                itemType,
                itemId: newItem.id,
                localVersion: {
                  data: existing,
                  modifiedAt: existingDate.toISOString(),
                  modifiedBy: 'local',
                  deviceId: getDeviceId(),
                  version: 1,
                },
                remoteVersion: {
                  data: newItem,
                  modifiedAt: newDate.toISOString(),
                  modifiedBy: 'import',
                  deviceId: 'import',
                  version: 1,
                },
                detectedAt: new Date().toISOString(),
                autoResolvable: false,
                suggestedResolution: newDate > existingDate ? 'keep_remote' : 'keep_local',
              });
              skipped++;
            }
          } else {
            mergedItems.push(newItem);
            imported++;
          }
        }
        finalItems = mergedItems;
        break;
      }

      case 'append_only': {
        const existingIds = new Set(existingItems.map((item) => item.id));
        const newOnlyItems = (newItems as Array<{ id: string }>).filter(
          (item) => !existingIds.has(item.id)
        );
        finalItems = [...existingItems, ...newOnlyItems];
        imported = newOnlyItems.length;
        skipped = newItems.length - newOnlyItems.length;
        break;
      }

      default:
        finalItems = existingItems;
        skipped = newItems.length;
    }

    localStorage.setItem(key, JSON.stringify(finalItems));
    return { imported, skipped };
  };

  /**
   * Resolve a sync conflict
   */
  const resolveConflict = (resolution: ConflictResolution): void => {
    const conflict = conflicts.value.find((c) => c.id === resolution.conflictId);
    if (!conflict) return;

    const storageKey = `lva_${conflict.itemType}s`;
    const existingData = localStorage.getItem(storageKey);
    if (!existingData) return;

    const items: Array<{ id: string }> = JSON.parse(existingData);
    const itemIndex = items.findIndex((item) => item.id === conflict.itemId);

    if (itemIndex === -1) return;

    switch (resolution.resolution) {
      case 'keep_local':
        // No change needed
        break;
      case 'keep_remote':
        items[itemIndex] = conflict.remoteVersion.data as { id: string };
        break;
      case 'merge':
        if (resolution.mergedData) {
          items[itemIndex] = resolution.mergedData as { id: string };
        }
        break;
      case 'skip':
        // No change needed
        break;
    }

    localStorage.setItem(storageKey, JSON.stringify(items));
    conflicts.value = conflicts.value.filter((c) => c.id !== resolution.conflictId);

    if (conflicts.value.length === 0) {
      syncStatus.value = 'synced';
    }
  };

  /**
   * Register current device
   */
  const registerDevice = (): DeviceInfo => {
    const device: DeviceInfo = {
      id: getDeviceId(),
      name: getDeviceName(),
      type: detectDeviceType(),
      platform: getPlatform(),
      appVersion: APP_VERSION,
      lastSeen: new Date().toISOString(),
      lastSyncAt: lastSyncAt.value,
      isCurrentDevice: true,
      trusted: true,
      registeredAt: new Date().toISOString(),
    };

    // Update device in list
    const existingIndex = devices.value.findIndex((d) => d.id === device.id);
    if (existingIndex >= 0) {
      devices.value[existingIndex] = device;
    } else {
      devices.value.push(device);
    }

    saveDevices();
    return device;
  };

  /**
   * List all registered devices
   */
  const listDevices = (): DeviceInfo[] => {
    loadDevices();
    return devices.value;
  };

  /**
   * Remove a device
   */
  const removeDevice = (deviceId: string): void => {
    devices.value = devices.value.filter((d) => d.id !== deviceId);
    saveDevices();
  };

  /**
   * Get backup history
   */
  const getBackupHistory = (): BackupHistoryItem[] => {
    loadBackupHistory();
    return backupHistory.value;
  };

  /**
   * Restore from a backup
   */
  const restoreBackup = async (backupId: string): Promise<void> => {
    const backup = backupHistory.value.find((b) => b.id === backupId);
    if (!backup) {
      throw new Error('Backup not found');
    }
    syncStatus.value = 'syncing';
    // Simulated restore - in a real app this would fetch from storage
    await new Promise((resolve) => setTimeout(resolve, 1000));
    syncStatus.value = 'synced';
    lastSyncAt.value = new Date().toISOString();
  };

  /**
   * Delete a backup
   */
  const deleteBackup = (backupId: string): void => {
    backupHistory.value = backupHistory.value.filter((b) => b.id !== backupId);
    saveBackupHistory();
  };

  /**
   * Trigger manual sync
   */
  const syncNow = async (): Promise<void> => {
    syncStatus.value = 'syncing';
    try {
      // In a real implementation, this would sync with a backend
      await new Promise((resolve) => setTimeout(resolve, 1500));
      lastSyncAt.value = new Date().toISOString();
      syncStatus.value = 'synced';
      dataSize.value = calculateDataSize();
      scheduleNextAutoBackup();
    } catch (error) {
      syncStatus.value = 'error';
      syncError.value = error instanceof Error ? error.message : 'Sync failed';
    }
  };

  /**
   * Schedule next auto backup
   */
  const scheduleNextAutoBackup = (): void => {
    nextAutoBackup.value = new Date(Date.now() + AUTO_BACKUP_INTERVAL);
    localStorage.setItem(
      `${STORAGE_KEY_PREFIX}next_auto_backup`,
      nextAutoBackup.value.toISOString()
    );
  };

  /**
   * Compare semantic versions
   */
  const compareVersions = (a: string, b: string): number => {
    const aParts = a.split('.').map(Number);
    const bParts = b.split('.').map(Number);

    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      const aVal = aParts[i] || 0;
      const bVal = bParts[i] || 0;
      if (aVal > bVal) return 1;
      if (aVal < bVal) return -1;
    }
    return 0;
  };

  // Persistence helpers
  const saveDevices = (): void => {
    localStorage.setItem(`${STORAGE_KEY_PREFIX}devices`, JSON.stringify(devices.value));
  };

  const loadDevices = (): void => {
    const data = localStorage.getItem(`${STORAGE_KEY_PREFIX}devices`);
    if (data) {
      devices.value = JSON.parse(data);
    }
  };

  const saveBackupHistory = (): void => {
    localStorage.setItem(`${STORAGE_KEY_PREFIX}backup_history`, JSON.stringify(backupHistory.value));
  };

  const loadBackupHistory = (): void => {
    const data = localStorage.getItem(`${STORAGE_KEY_PREFIX}backup_history`);
    if (data) {
      backupHistory.value = JSON.parse(data);
    }
  };

  // Initialize
  onMounted(() => {
    // Load persisted state
    const lastSyncStr = localStorage.getItem(`${STORAGE_KEY_PREFIX}last_sync`);
    if (lastSyncStr) {
      lastSyncAt.value = lastSyncStr;
    }

    const nextBackupStr = localStorage.getItem(`${STORAGE_KEY_PREFIX}next_auto_backup`);
    if (nextBackupStr) {
      nextAutoBackup.value = new Date(nextBackupStr);
    }

    loadDevices();
    loadBackupHistory();
    dataSize.value = calculateDataSize();

    // Register current device
    registerDevice();
  });

  return {
    // State
    syncStatus: computed(() => syncStatus.value),
    lastSync: computed(() => lastSyncAt.value ? new Date(lastSyncAt.value) : null),
    nextAutoBackup: computed(() => nextAutoBackup.value),
    dataSize: computed(() => dataSize.value),
    syncError: computed(() => syncError.value),
    isExporting: computed(() => isExporting.value),
    isImporting: computed(() => isImporting.value),
    exportProgress: computed(() => exportProgress.value),
    importProgress: computed(() => importProgress.value),
    devices: computed(() => devices.value),
    backupHistory: computed(() => backupHistory.value),
    conflicts: computed(() => conflicts.value),
    hasConflicts: computed(() => conflicts.value.length > 0),

    // Actions
    exportData,
    importData,
    validateBackup,
    resolveConflict,
    registerDevice,
    listDevices,
    removeDevice,
    getBackupHistory,
    restoreBackup,
    deleteBackup,
    syncNow,
  };
}
