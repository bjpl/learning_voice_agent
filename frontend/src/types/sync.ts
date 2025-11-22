/**
 * Sync, backup, and data export/import types
 */

/**
 * Current sync status of the application
 */
export type SyncStatusType = 'idle' | 'syncing' | 'synced' | 'error' | 'offline' | 'conflict';

/**
 * Sync status response from API
 */
export interface SyncStatus {
  status: SyncStatusType;
  lastSyncAt: string | null;
  lastSyncDuration: number | null;
  pendingChanges: number;
  syncedDevices: number;
  storageUsed: number;
  storageLimit: number;
  version: string;
  serverTime: string;
}

/**
 * Metadata about sync operations
 */
export interface SyncMetadata {
  id: string;
  deviceId: string;
  deviceName: string;
  syncedAt: string;
  dataVersion: string;
  checksum: string;
  itemCounts: SyncItemCounts;
}

/**
 * Counts of synced items by type
 */
export interface SyncItemCounts {
  conversations: number;
  messages: number;
  goals: number;
  achievements: number;
  feedback: number;
  settings: number;
  analytics: number;
}

/**
 * Complete backup data structure
 */
export interface BackupData {
  metadata: BackupMetadata;
  conversations: unknown[];
  goals: unknown[];
  achievements: unknown[];
  feedback: unknown[];
  settings: UserSettings;
  analytics: unknown[];
}

/**
 * Backup file metadata
 */
export interface BackupMetadata {
  version: string;
  createdAt: string;
  createdBy: string;
  deviceId: string;
  deviceName: string;
  appVersion: string;
  checksum: string;
  itemCounts: SyncItemCounts;
  encrypted: boolean;
  compressionType: 'none' | 'gzip' | 'brotli';
}

/**
 * User settings included in backup
 */
export interface UserSettings {
  language: string;
  theme: 'light' | 'dark' | 'system';
  notifications: NotificationSettings;
  privacy: PrivacySettings;
  accessibility: AccessibilitySettings;
}

/**
 * Notification preferences
 */
export interface NotificationSettings {
  enabled: boolean;
  sound: boolean;
  dailyReminder: boolean;
  reminderTime: string | null;
  achievementAlerts: boolean;
  streakAlerts: boolean;
}

/**
 * Privacy settings
 */
export interface PrivacySettings {
  shareAnalytics: boolean;
  shareProgress: boolean;
  publicProfile: boolean;
}

/**
 * Accessibility settings
 */
export interface AccessibilitySettings {
  fontSize: 'small' | 'medium' | 'large';
  highContrast: boolean;
  reduceMotion: boolean;
  screenReaderOptimized: boolean;
}

/**
 * Options for data export
 */
export interface ExportOptions {
  format: ExportFormat;
  includeConversations: boolean;
  includeGoals: boolean;
  includeAchievements: boolean;
  includeFeedback: boolean;
  includeSettings: boolean;
  includeAnalytics: boolean;
  dateRange?: DateRange;
  encrypted: boolean;
  password?: string;
  compression: 'none' | 'gzip' | 'brotli';
}

/**
 * Supported export formats
 */
export type ExportFormat = 'json' | 'csv' | 'zip';

/**
 * Date range for filtered exports
 */
export interface DateRange {
  start: string;
  end: string;
}

/**
 * Options for data import
 */
export interface ImportOptions {
  file: File;
  strategy: MergeStrategy;
  password?: string;
  validateOnly: boolean;
  skipConflicts: boolean;
  preserveLocalChanges: boolean;
}

/**
 * Strategy for handling data conflicts during import
 */
export type MergeStrategy =
  | 'replace_all'       // Replace all local data with imported
  | 'keep_local'        // Keep local data, ignore conflicts
  | 'keep_remote'       // Keep imported data on conflicts
  | 'merge_newest'      // Keep newest version of each item
  | 'merge_manual'      // Prompt for each conflict
  | 'append_only';      // Only add new items, never update

/**
 * Result of an import operation
 */
export interface ImportResult {
  success: boolean;
  totalItems: number;
  importedItems: number;
  skippedItems: number;
  conflictCount: number;
  conflicts: SyncConflict[];
  errors: ImportError[];
  duration: number;
  warnings: string[];
}

/**
 * Error during import
 */
export interface ImportError {
  itemType: string;
  itemId: string;
  error: string;
  recoverable: boolean;
}

/**
 * Validation result for backup files
 */
export interface ValidationResult {
  valid: boolean;
  version: string;
  compatible: boolean;
  itemCounts: SyncItemCounts;
  encrypted: boolean;
  checksum: string;
  checksumValid: boolean;
  errors: ValidationError[];
  warnings: string[];
  estimatedImportTime: number;
}

/**
 * Validation error details
 */
export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}

/**
 * Data conflict between local and remote versions
 */
export interface SyncConflict {
  id: string;
  itemType: ConflictItemType;
  itemId: string;
  localVersion: ConflictVersion;
  remoteVersion: ConflictVersion;
  detectedAt: string;
  autoResolvable: boolean;
  suggestedResolution: ConflictResolutionType;
}

/**
 * Types of items that can have conflicts
 */
export type ConflictItemType =
  | 'conversation'
  | 'message'
  | 'goal'
  | 'achievement'
  | 'feedback'
  | 'setting';

/**
 * Version info for conflict comparison
 */
export interface ConflictVersion {
  data: unknown;
  modifiedAt: string;
  modifiedBy: string;
  deviceId: string;
  version: number;
}

/**
 * Resolution for a sync conflict
 */
export interface ConflictResolution {
  conflictId: string;
  resolution: ConflictResolutionType;
  mergedData?: unknown;
}

/**
 * How to resolve a conflict
 */
export type ConflictResolutionType =
  | 'keep_local'
  | 'keep_remote'
  | 'merge'
  | 'skip';

/**
 * Device information for multi-device sync
 */
export interface DeviceInfo {
  id: string;
  name: string;
  type: DeviceType;
  platform: string;
  appVersion: string;
  lastSeen: string;
  lastSyncAt: string | null;
  isCurrentDevice: boolean;
  trusted: boolean;
  registeredAt: string;
}

/**
 * Type of device
 */
export type DeviceType = 'desktop' | 'mobile' | 'tablet' | 'web' | 'unknown';

/**
 * Request to register a new device
 */
export interface DeviceRegistration {
  name: string;
  type: DeviceType;
  platform: string;
  appVersion: string;
}

/**
 * Historical backup entry
 */
export interface BackupHistoryItem {
  id: string;
  createdAt: string;
  deviceId: string;
  deviceName: string;
  size: number;
  itemCounts: SyncItemCounts;
  type: BackupType;
  status: BackupStatus;
  downloadUrl?: string;
  expiresAt?: string;
}

/**
 * Type of backup
 */
export type BackupType = 'manual' | 'automatic' | 'scheduled' | 'pre_update';

/**
 * Status of a backup
 */
export type BackupStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'expired';

/**
 * Progress information for long-running operations
 */
export interface SyncProgress {
  operation: SyncOperation;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  currentStep: number;
  totalSteps: number;
  currentItem: string;
  itemsProcessed: number;
  totalItems: number;
  bytesProcessed: number;
  totalBytes: number;
  estimatedTimeRemaining: number | null;
  startedAt: string;
  errors: string[];
}

/**
 * Types of sync operations
 */
export type SyncOperation =
  | 'export'
  | 'import'
  | 'sync'
  | 'backup'
  | 'restore'
  | 'validate';

/**
 * Sync event for real-time updates
 */
export interface SyncEvent {
  type: SyncEventType;
  timestamp: string;
  deviceId: string;
  data?: unknown;
}

/**
 * Types of sync events
 */
export type SyncEventType =
  | 'sync_started'
  | 'sync_completed'
  | 'sync_failed'
  | 'conflict_detected'
  | 'conflict_resolved'
  | 'device_connected'
  | 'device_disconnected'
  | 'data_changed';

/**
 * Preview of data to be imported
 */
export interface ImportPreview {
  conversations: number;
  goals: number;
  achievements: number;
  settings: boolean;
  totalSize: number;
  backupDate: string;
  version: string;
  isCompatible: boolean;
  warnings: string[];
}

// Backward compatibility aliases
export type BackupItem = BackupHistoryItem;
export type Device = DeviceInfo;
