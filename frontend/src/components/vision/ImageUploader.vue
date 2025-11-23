<script setup lang="ts">
/**
 * ImageUploader Component
 * Drag-drop image upload with preview
 */

import { ref, computed } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface UploadedImage {
  id: string;
  file: File;
  url: string;
  name: string;
  size: number;
  type: string;
  width?: number;
  height?: number;
}

export interface ImageUploaderProps {
  multiple?: boolean;
  maxSize?: number; // in MB
  maxFiles?: number;
  acceptedTypes?: string[];
  showPreview?: boolean;
  disabled?: boolean;
}

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<ImageUploaderProps>(), {
  multiple: true,
  maxSize: 10,
  maxFiles: 10,
  acceptedTypes: () => ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  showPreview: true,
  disabled: false
});

const emit = defineEmits<{
  (e: 'upload', images: UploadedImage[]): void;
  (e: 'remove', image: UploadedImage): void;
  (e: 'error', error: string): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const inputRef = ref<HTMLInputElement | null>(null);
const isDragging = ref(false);
const images = ref<UploadedImage[]>([]);
const errors = ref<string[]>([]);

// ============================================================================
// Computed
// ============================================================================

const acceptString = computed(() => props.acceptedTypes.join(','));

const canUploadMore = computed(() => {
  return !props.maxFiles || images.value.length < props.maxFiles;
});

const remainingSlots = computed(() => {
  if (!props.maxFiles) return Infinity;
  return props.maxFiles - images.value.length;
});

// ============================================================================
// Methods
// ============================================================================

function generateId(): string {
  return `img-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function validateFile(file: File): string | null {
  // Check type
  if (!props.acceptedTypes.includes(file.type)) {
    return `File type ${file.type} is not supported`;
  }

  // Check size
  const maxBytes = props.maxSize * 1024 * 1024;
  if (file.size > maxBytes) {
    return `File size exceeds ${props.maxSize}MB limit`;
  }

  return null;
}

async function getImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.width, height: img.height });
    img.onerror = reject;
    img.src = url;
  });
}

async function processFiles(files: FileList | File[]): Promise<void> {
  errors.value = [];
  const fileArray = Array.from(files);

  // Check max files
  if (props.maxFiles && images.value.length + fileArray.length > props.maxFiles) {
    errors.value.push(`Maximum ${props.maxFiles} files allowed`);
    emit('error', `Maximum ${props.maxFiles} files allowed`);
    return;
  }

  const newImages: UploadedImage[] = [];

  for (const file of fileArray) {
    const validationError = validateFile(file);
    if (validationError) {
      errors.value.push(`${file.name}: ${validationError}`);
      emit('error', validationError);
      continue;
    }

    const url = URL.createObjectURL(file);

    try {
      const dimensions = await getImageDimensions(url);

      const uploadedImage: UploadedImage = {
        id: generateId(),
        file,
        url,
        name: file.name,
        size: file.size,
        type: file.type,
        width: dimensions.width,
        height: dimensions.height
      };

      newImages.push(uploadedImage);
    } catch {
      errors.value.push(`${file.name}: Failed to load image`);
      URL.revokeObjectURL(url);
    }
  }

  if (newImages.length > 0) {
    images.value = [...images.value, ...newImages];
    emit('upload', newImages);
  }
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

function removeImage(image: UploadedImage): void {
  URL.revokeObjectURL(image.url);
  images.value = images.value.filter(img => img.id !== image.id);
  emit('remove', image);
}

function clearAll(): void {
  images.value.forEach(img => URL.revokeObjectURL(img.url));
  images.value = [];
}

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  images,
  clearAll,
  openFileDialog
});
</script>

<template>
  <div class="image-uploader" :class="{ 'image-uploader--disabled': disabled }">
    <!-- Drop Zone -->
    <div
      class="image-uploader__dropzone"
      :class="{
        'image-uploader__dropzone--dragging': isDragging,
        'image-uploader__dropzone--disabled': disabled || !canUploadMore
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
        class="image-uploader__input"
        @change="handleFileInput"
      />

      <div class="image-uploader__content">
        <!-- Upload Icon -->
        <svg class="image-uploader__icon" width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 7v2.99s-1.99.01-2 0V7h-3s.01-1.99 0-2h3V2h2v3h3v2h-3zm-3 4V8h-3V5H5c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-8h-3zM5 19l3-4 2 3 3-4 4 5H5z"/>
        </svg>

        <!-- Text -->
        <p class="image-uploader__text">
          <span class="image-uploader__text--primary">
            Drop images here or click to upload
          </span>
          <span class="image-uploader__text--secondary">
            {{ acceptedTypes.map(t => t.split('/')[1].toUpperCase()).join(', ') }}
            up to {{ maxSize }}MB
          </span>
        </p>

        <!-- Remaining slots -->
        <p v-if="maxFiles" class="image-uploader__slots">
          {{ remainingSlots }} of {{ maxFiles }} slots remaining
        </p>
      </div>
    </div>

    <!-- Errors -->
    <div v-if="errors.length > 0" class="image-uploader__errors">
      <div v-for="error in errors" :key="error" class="image-uploader__error">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        {{ error }}
      </div>
    </div>

    <!-- Preview Grid -->
    <div v-if="showPreview && images.length > 0" class="image-uploader__preview">
      <div
        v-for="image in images"
        :key="image.id"
        class="image-uploader__preview-item"
      >
        <img
          :src="image.url"
          :alt="image.name"
          class="image-uploader__preview-image"
        />

        <div class="image-uploader__preview-overlay">
          <div class="image-uploader__preview-info">
            <span class="image-uploader__preview-name">{{ image.name }}</span>
            <span class="image-uploader__preview-size">{{ formatSize(image.size) }}</span>
            <span v-if="image.width && image.height" class="image-uploader__preview-dimensions">
              {{ image.width }} x {{ image.height }}
            </span>
          </div>

          <button
            class="image-uploader__preview-remove"
            @click.stop="removeImage(image)"
            title="Remove image"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div v-if="images.length > 0" class="image-uploader__actions">
      <button
        class="image-uploader__btn image-uploader__btn--secondary"
        @click="clearAll"
      >
        Clear All
      </button>
      <span class="image-uploader__count">
        {{ images.length }} image{{ images.length !== 1 ? 's' : '' }} selected
      </span>
    </div>
  </div>
</template>

<style scoped>
.image-uploader {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.image-uploader--disabled {
  opacity: 0.6;
  pointer-events: none;
}

/* Drop Zone */
.image-uploader__dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  padding: 32px;
  border: 2px dashed #d1d5db;
  border-radius: 12px;
  background-color: #f9fafb;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-uploader__dropzone:hover {
  border-color: #3b82f6;
  background-color: #eff6ff;
}

.image-uploader__dropzone--dragging {
  border-color: #3b82f6;
  background-color: #eff6ff;
  transform: scale(1.02);
}

.image-uploader__dropzone--disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.image-uploader__input {
  display: none;
}

.image-uploader__content {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.image-uploader__icon {
  color: #9ca3af;
  margin-bottom: 16px;
}

.image-uploader__dropzone:hover .image-uploader__icon,
.image-uploader__dropzone--dragging .image-uploader__icon {
  color: #3b82f6;
}

.image-uploader__text {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
}

.image-uploader__text--primary {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.image-uploader__text--secondary {
  font-size: 0.75rem;
  color: #6b7280;
}

.image-uploader__slots {
  margin: 8px 0 0;
  font-size: 0.75rem;
  color: #9ca3af;
}

/* Errors */
.image-uploader__errors {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.image-uploader__error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background-color: #fee2e2;
  color: #dc2626;
  border-radius: 6px;
  font-size: 0.75rem;
}

/* Preview Grid */
.image-uploader__preview {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}

.image-uploader__preview-item {
  position: relative;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  background-color: #f3f4f6;
}

.image-uploader__preview-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-uploader__preview-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 8px;
  background: linear-gradient(180deg, rgba(0,0,0,0.4) 0%, transparent 40%, transparent 60%, rgba(0,0,0,0.6) 100%);
  opacity: 0;
  transition: opacity 0.2s ease;
}

.image-uploader__preview-item:hover .image-uploader__preview-overlay {
  opacity: 1;
}

.image-uploader__preview-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.image-uploader__preview-name {
  font-size: 0.625rem;
  font-weight: 500;
  color: white;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.image-uploader__preview-size,
.image-uploader__preview-dimensions {
  font-size: 0.625rem;
  color: rgba(255, 255, 255, 0.8);
}

.image-uploader__preview-remove {
  position: absolute;
  top: 8px;
  right: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background-color: rgba(0, 0, 0, 0.5);
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-uploader__preview-remove:hover {
  background-color: #ef4444;
}

/* Actions */
.image-uploader__actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.image-uploader__btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-uploader__btn--secondary {
  background-color: #f3f4f6;
  color: #374151;
}

.image-uploader__btn--secondary:hover {
  background-color: #e5e7eb;
}

.image-uploader__count {
  font-size: 0.75rem;
  color: #6b7280;
}
</style>
