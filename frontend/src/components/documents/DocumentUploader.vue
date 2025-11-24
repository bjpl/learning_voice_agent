<script setup lang="ts">
/**
 * DocumentUploader Component
 * PDF/document upload with validation and progress
 */

import { ref, computed } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface UploadedDocument {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  extension: string;
  uploadProgress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  errorMessage?: string;
  url?: string;
  pageCount?: number;
}

export interface DocumentUploaderProps {
  multiple?: boolean;
  maxSize?: number; // in MB
  maxFiles?: number;
  acceptedTypes?: string[];
  autoUpload?: boolean;
  uploadUrl?: string;
  disabled?: boolean;
}

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<DocumentUploaderProps>(), {
  multiple: true,
  maxSize: 50,
  maxFiles: 10,
  acceptedTypes: () => [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/plain',
    'text/csv'
  ],
  autoUpload: false,
  uploadUrl: '/api/documents/upload',
  disabled: false
});

const emit = defineEmits<{
  (e: 'select', documents: UploadedDocument[]): void;
  (e: 'upload', document: UploadedDocument): void;
  (e: 'uploadComplete', documents: UploadedDocument[]): void;
  (e: 'remove', document: UploadedDocument): void;
  (e: 'error', error: string): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const inputRef = ref<HTMLInputElement | null>(null);
const isDragging = ref(false);
const documents = ref<UploadedDocument[]>([]);
const errors = ref<string[]>([]);

// ============================================================================
// Computed
// ============================================================================

const acceptString = computed(() => props.acceptedTypes.join(','));

const canUploadMore = computed(() => {
  return !props.maxFiles || documents.value.length < props.maxFiles;
});

const remainingSlots = computed(() => {
  if (!props.maxFiles) return Infinity;
  return props.maxFiles - documents.value.length;
});

const pendingDocuments = computed(() => {
  return documents.value.filter(d => d.status === 'pending');
});

const hasUploading = computed(() => {
  return documents.value.some(d => d.status === 'uploading');
});

// ============================================================================
// Document Type Mappings
// ============================================================================

const documentIcons: Record<string, string> = {
  'application/pdf': 'pdf',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'doc',
  'application/vnd.ms-excel': 'xls',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xls',
  'application/vnd.ms-powerpoint': 'ppt',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'ppt',
  'text/plain': 'txt',
  'text/csv': 'csv'
};

const documentColors: Record<string, string> = {
  pdf: '#ef4444',
  doc: '#3b82f6',
  xls: '#10b981',
  ppt: '#f59e0b',
  txt: '#6b7280',
  csv: '#10b981'
};

// ============================================================================
// Methods
// ============================================================================

function generateId(): string {
  return `doc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getExtension(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() || '';
}

function getDocumentType(mimeType: string): string {
  return documentIcons[mimeType] || 'file';
}

function getDocumentColor(type: string): string {
  return documentColors[type] || '#6b7280';
}

function validateFile(file: File): string | null {
  // Check type
  if (!props.acceptedTypes.includes(file.type)) {
    const ext = getExtension(file.name);
    return `File type .${ext} is not supported`;
  }

  // Check size
  const maxBytes = props.maxSize * 1024 * 1024;
  if (file.size > maxBytes) {
    return `File size exceeds ${props.maxSize}MB limit`;
  }

  return null;
}

async function processFiles(files: FileList | File[]): Promise<void> {
  errors.value = [];
  const fileArray = Array.from(files);

  // Check max files
  if (props.maxFiles && documents.value.length + fileArray.length > props.maxFiles) {
    errors.value.push(`Maximum ${props.maxFiles} files allowed`);
    emit('error', `Maximum ${props.maxFiles} files allowed`);
    return;
  }

  const newDocuments: UploadedDocument[] = [];

  for (const file of fileArray) {
    const validationError = validateFile(file);
    if (validationError) {
      errors.value.push(`${file.name}: ${validationError}`);
      emit('error', validationError);
      continue;
    }

    const document: UploadedDocument = {
      id: generateId(),
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      extension: getExtension(file.name),
      uploadProgress: 0,
      status: 'pending'
    };

    newDocuments.push(document);
  }

  if (newDocuments.length > 0) {
    documents.value = [...documents.value, ...newDocuments];
    emit('select', newDocuments);

    if (props.autoUpload) {
      await uploadDocuments(newDocuments);
    }
  }
}

async function uploadDocument(document: UploadedDocument): Promise<void> {
  document.status = 'uploading';
  document.uploadProgress = 0;

  try {
    const formData = new FormData();
    formData.append('file', document.file);

    const xhr = new XMLHttpRequest();

    await new Promise<void>((resolve, reject) => {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          document.uploadProgress = Math.round((event.loaded / event.total) * 100);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            document.url = response.url;
            document.pageCount = response.pageCount;
          } catch {
            // Response might not be JSON
          }
          document.status = 'success';
          document.uploadProgress = 100;
          emit('upload', document);
          resolve();
        } else {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.open('POST', props.uploadUrl);
      xhr.send(formData);
    });

  } catch (err) {
    document.status = 'error';
    document.errorMessage = err instanceof Error ? err.message : 'Upload failed';
    emit('error', document.errorMessage);
  }
}

async function uploadDocuments(docs?: UploadedDocument[]): Promise<void> {
  const toUpload = docs || pendingDocuments.value;

  await Promise.all(toUpload.map(doc => uploadDocument(doc)));

  emit('uploadComplete', documents.value.filter(d => d.status === 'success'));
}

function handleFileInput(event: Event): void {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    processFiles(target.files);
    target.value = '';
  }
}

function handleDragEnter(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = true;
}

function handleDragLeave(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = false;
}

function handleDragOver(event: DragEvent): void {
  event.preventDefault();
}

function handleDrop(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = false;

  if (props.disabled || !canUploadMore.value) return;

  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    processFiles(files);
  }
}

function openFileDialog(): void {
  if (props.disabled || !canUploadMore.value) return;
  inputRef.value?.click();
}

function removeDocument(document: UploadedDocument): void {
  documents.value = documents.value.filter(d => d.id !== document.id);
  emit('remove', document);
}

function retryUpload(document: UploadedDocument): void {
  document.status = 'pending';
  document.errorMessage = undefined;
  uploadDocument(document);
}

function clearAll(): void {
  documents.value = [];
}

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  documents,
  uploadDocuments,
  clearAll,
  openFileDialog
});
</script>

<template>
  <div class="document-uploader" :class="{ 'document-uploader--disabled': disabled }">
    <!-- Drop Zone -->
    <div
      class="document-uploader__dropzone"
      :class="{
        'document-uploader__dropzone--dragging': isDragging,
        'document-uploader__dropzone--disabled': disabled || !canUploadMore
      }"
      @dragenter="handleDragEnter"
      @dragleave="handleDragLeave"
      @dragover="handleDragOver"
      @drop="handleDrop"
      @click="openFileDialog"
    >
      <input
        ref="inputRef"
        type="file"
        :accept="acceptString"
        :multiple="multiple"
        :disabled="disabled"
        class="document-uploader__input"
        @change="handleFileInput"
      />

      <div class="document-uploader__content">
        <!-- Upload Icon -->
        <svg class="document-uploader__icon" width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
          <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11zm-3-7v4h-2v-4H9l4-4 4 4h-4z"/>
        </svg>

        <!-- Text -->
        <p class="document-uploader__text">
          <span class="document-uploader__text--primary">
            Drop documents here or click to upload
          </span>
          <span class="document-uploader__text--secondary">
            PDF, Word, Excel, PowerPoint, TXT, CSV up to {{ maxSize }}MB
          </span>
        </p>

        <!-- Remaining slots -->
        <p v-if="maxFiles" class="document-uploader__slots">
          {{ remainingSlots }} of {{ maxFiles }} slots remaining
        </p>
      </div>
    </div>

    <!-- Errors -->
    <div v-if="errors.length > 0" class="document-uploader__errors">
      <div v-for="error in errors" :key="error" class="document-uploader__error">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        {{ error }}
      </div>
    </div>

    <!-- Document List -->
    <div v-if="documents.length > 0" class="document-uploader__list">
      <div
        v-for="doc in documents"
        :key="doc.id"
        class="document-uploader__item"
        :class="[`document-uploader__item--${doc.status}`]"
      >
        <!-- Icon -->
        <div
          class="document-uploader__item-icon"
          :style="{ backgroundColor: getDocumentColor(getDocumentType(doc.type)) }"
        >
          <span>{{ doc.extension.toUpperCase() }}</span>
        </div>

        <!-- Info -->
        <div class="document-uploader__item-info">
          <span class="document-uploader__item-name">{{ doc.name }}</span>
          <span class="document-uploader__item-meta">
            {{ formatSize(doc.size) }}
            <template v-if="doc.pageCount"> &middot; {{ doc.pageCount }} pages</template>
          </span>

          <!-- Progress Bar -->
          <div v-if="doc.status === 'uploading'" class="document-uploader__progress">
            <div
              class="document-uploader__progress-bar"
              :style="{ width: `${doc.uploadProgress}%` }"
            />
          </div>

          <!-- Error Message -->
          <span v-if="doc.status === 'error'" class="document-uploader__item-error">
            {{ doc.errorMessage }}
          </span>
        </div>

        <!-- Status Icon -->
        <div class="document-uploader__item-status">
          <!-- Uploading -->
          <div v-if="doc.status === 'uploading'" class="document-uploader__spinner" />

          <!-- Success -->
          <svg v-else-if="doc.status === 'success'" width="20" height="20" viewBox="0 0 24 24" fill="#10b981">
            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
          </svg>

          <!-- Error -->
          <button v-else-if="doc.status === 'error'" class="document-uploader__retry" @click="retryUpload(doc)">
            Retry
          </button>
        </div>

        <!-- Remove Button -->
        <button
          class="document-uploader__remove"
          @click.stop="removeDocument(doc)"
          title="Remove"
          :disabled="doc.status === 'uploading'"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- Actions -->
    <div v-if="documents.length > 0" class="document-uploader__actions">
      <button
        class="document-uploader__btn document-uploader__btn--secondary"
        @click="clearAll"
        :disabled="hasUploading"
      >
        Clear All
      </button>

      <button
        v-if="!autoUpload && pendingDocuments.length > 0"
        class="document-uploader__btn document-uploader__btn--primary"
        @click="uploadDocuments()"
        :disabled="hasUploading"
      >
        Upload {{ pendingDocuments.length }} Document{{ pendingDocuments.length !== 1 ? 's' : '' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.document-uploader {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.document-uploader--disabled {
  opacity: 0.6;
  pointer-events: none;
}

/* Drop Zone */
.document-uploader__dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 180px;
  padding: 32px;
  border: 2px dashed #d1d5db;
  border-radius: 12px;
  background-color: #f9fafb;
  cursor: pointer;
  transition: all 0.2s ease;
}

.document-uploader__dropzone:hover {
  border-color: #3b82f6;
  background-color: #eff6ff;
}

.document-uploader__dropzone--dragging {
  border-color: #3b82f6;
  background-color: #eff6ff;
  transform: scale(1.02);
}

.document-uploader__dropzone--disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.document-uploader__input {
  display: none;
}

.document-uploader__content {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.document-uploader__icon {
  color: #9ca3af;
  margin-bottom: 16px;
}

.document-uploader__dropzone:hover .document-uploader__icon,
.document-uploader__dropzone--dragging .document-uploader__icon {
  color: #3b82f6;
}

.document-uploader__text {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
}

.document-uploader__text--primary {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.document-uploader__text--secondary {
  font-size: 0.75rem;
  color: #6b7280;
}

.document-uploader__slots {
  margin: 8px 0 0;
  font-size: 0.75rem;
  color: #9ca3af;
}

/* Errors */
.document-uploader__errors {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.document-uploader__error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background-color: #fee2e2;
  color: #dc2626;
  border-radius: 6px;
  font-size: 0.75rem;
}

/* Document List */
.document-uploader__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.document-uploader__item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background-color: #f9fafb;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.document-uploader__item--success {
  background-color: #f0fdf4;
  border-color: #bbf7d0;
}

.document-uploader__item--error {
  background-color: #fef2f2;
  border-color: #fecaca;
}

.document-uploader__item-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 8px;
  color: white;
  font-size: 0.625rem;
  font-weight: 700;
}

.document-uploader__item-info {
  flex: 1;
  min-width: 0;
}

.document-uploader__item-name {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-uploader__item-meta {
  font-size: 0.75rem;
  color: #6b7280;
}

.document-uploader__item-error {
  display: block;
  font-size: 0.75rem;
  color: #dc2626;
}

/* Progress */
.document-uploader__progress {
  height: 4px;
  margin-top: 8px;
  background-color: #e5e7eb;
  border-radius: 2px;
  overflow: hidden;
}

.document-uploader__progress-bar {
  height: 100%;
  background-color: #3b82f6;
  border-radius: 2px;
  transition: width 0.2s ease;
}

/* Status */
.document-uploader__item-status {
  display: flex;
  align-items: center;
}

.document-uploader__spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.document-uploader__retry {
  padding: 4px 8px;
  border: none;
  border-radius: 4px;
  background-color: #3b82f6;
  color: white;
  font-size: 0.75rem;
  cursor: pointer;
}

.document-uploader__retry:hover {
  background-color: #2563eb;
}

.document-uploader__remove {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background-color: transparent;
  color: #9ca3af;
  cursor: pointer;
  transition: all 0.2s ease;
}

.document-uploader__remove:hover:not(:disabled) {
  background-color: #fee2e2;
  color: #ef4444;
}

.document-uploader__remove:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Actions */
.document-uploader__actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.document-uploader__btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.document-uploader__btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.document-uploader__btn--secondary {
  background-color: #f3f4f6;
  color: #374151;
}

.document-uploader__btn--secondary:hover:not(:disabled) {
  background-color: #e5e7eb;
}

.document-uploader__btn--primary {
  background-color: #3b82f6;
  color: white;
}

.document-uploader__btn--primary:hover:not(:disabled) {
  background-color: #2563eb;
}
</style>
