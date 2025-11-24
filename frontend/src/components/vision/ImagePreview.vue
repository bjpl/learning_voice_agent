<script setup lang="ts">
/**
 * ImagePreview Component
 * Zoomable image preview with pan and zoom controls
 */

import { ref, computed, watch, onMounted, onUnmounted } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface ImagePreviewProps {
  src: string;
  alt?: string;
  initialZoom?: number;
  minZoom?: number;
  maxZoom?: number;
  zoomStep?: number;
  showControls?: boolean;
  showInfo?: boolean;
  enableWheel?: boolean;
  enableDrag?: boolean;
}

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<ImagePreviewProps>(), {
  alt: 'Preview image',
  initialZoom: 1,
  minZoom: 0.5,
  maxZoom: 5,
  zoomStep: 0.25,
  showControls: true,
  showInfo: true,
  enableWheel: true,
  enableDrag: true
});

const emit = defineEmits<{
  (e: 'close'): void;
  (e: 'zoomChange', zoom: number): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const containerRef = ref<HTMLDivElement | null>(null);
const imageRef = ref<HTMLImageElement | null>(null);
const zoom = ref(props.initialZoom);
const position = ref({ x: 0, y: 0 });
const isDragging = ref(false);
const dragStart = ref({ x: 0, y: 0 });
const isLoading = ref(true);
const imageSize = ref({ width: 0, height: 0 });
const naturalSize = ref({ width: 0, height: 0 });

// ============================================================================
// Computed
// ============================================================================

const zoomPercent = computed(() => Math.round(zoom.value * 100));

const imageStyle = computed(() => ({
  transform: `translate(${position.value.x}px, ${position.value.y}px) scale(${zoom.value})`,
  cursor: isDragging.value ? 'grabbing' : (zoom.value > 1 ? 'grab' : 'default')
}));

const canZoomIn = computed(() => zoom.value < props.maxZoom);
const canZoomOut = computed(() => zoom.value > props.minZoom);

// ============================================================================
// Methods
// ============================================================================

function setZoom(newZoom: number): void {
  const clampedZoom = Math.max(props.minZoom, Math.min(props.maxZoom, newZoom));
  zoom.value = clampedZoom;
  emit('zoomChange', clampedZoom);

  // Reset position if zooming out to fit
  if (clampedZoom <= 1) {
    position.value = { x: 0, y: 0 };
  }
}

function zoomIn(): void {
  setZoom(zoom.value + props.zoomStep);
}

function zoomOut(): void {
  setZoom(zoom.value - props.zoomStep);
}

function resetZoom(): void {
  setZoom(1);
  position.value = { x: 0, y: 0 };
}

function fitToScreen(): void {
  if (!containerRef.value || !naturalSize.value.width) return;

  const containerRect = containerRef.value.getBoundingClientRect();
  const widthRatio = containerRect.width / naturalSize.value.width;
  const heightRatio = containerRect.height / naturalSize.value.height;
  const fitZoom = Math.min(widthRatio, heightRatio, 1);

  setZoom(fitZoom);
  position.value = { x: 0, y: 0 };
}

function handleWheel(event: WheelEvent): void {
  if (!props.enableWheel) return;

  event.preventDefault();
  const delta = event.deltaY > 0 ? -props.zoomStep : props.zoomStep;
  setZoom(zoom.value + delta);
}

function handleMouseDown(event: MouseEvent): void {
  if (!props.enableDrag || zoom.value <= 1) return;

  isDragging.value = true;
  dragStart.value = {
    x: event.clientX - position.value.x,
    y: event.clientY - position.value.y
  };
}

function handleMouseMove(event: MouseEvent): void {
  if (!isDragging.value) return;

  position.value = {
    x: event.clientX - dragStart.value.x,
    y: event.clientY - dragStart.value.y
  };
}

function handleMouseUp(): void {
  isDragging.value = false;
}

function handleImageLoad(event: Event): void {
  const img = event.target as HTMLImageElement;
  naturalSize.value = {
    width: img.naturalWidth,
    height: img.naturalHeight
  };
  isLoading.value = false;
}

function handleDoubleClick(): void {
  if (zoom.value === 1) {
    setZoom(2);
  } else {
    resetZoom();
  }
}

function handleKeydown(event: KeyboardEvent): void {
  switch (event.key) {
    case '+':
    case '=':
      zoomIn();
      break;
    case '-':
      zoomOut();
      break;
    case '0':
      resetZoom();
      break;
    case 'Escape':
      emit('close');
      break;
  }
}

// ============================================================================
// Lifecycle
// ============================================================================

onMounted(() => {
  document.addEventListener('keydown', handleKeydown);
  document.addEventListener('mouseup', handleMouseUp);
  document.addEventListener('mousemove', handleMouseMove);
});

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown);
  document.removeEventListener('mouseup', handleMouseUp);
  document.removeEventListener('mousemove', handleMouseMove);
});

// ============================================================================
// Watchers
// ============================================================================

watch(() => props.src, () => {
  isLoading.value = true;
  resetZoom();
});

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  zoom,
  position,
  zoomIn,
  zoomOut,
  resetZoom,
  fitToScreen
});
</script>

<template>
  <div
    ref="containerRef"
    class="image-preview"
    tabindex="0"
    @wheel="handleWheel"
    @mousedown="handleMouseDown"
    @dblclick="handleDoubleClick"
  >
    <!-- Loading Indicator -->
    <div v-if="isLoading" class="image-preview__loading">
      <div class="image-preview__spinner"></div>
      <span>Loading image...</span>
    </div>

    <!-- Image Container -->
    <div class="image-preview__image-container">
      <img
        ref="imageRef"
        :src="src"
        :alt="alt"
        class="image-preview__image"
        :style="imageStyle"
        @load="handleImageLoad"
        draggable="false"
      />
    </div>

    <!-- Controls -->
    <div v-if="showControls" class="image-preview__controls">
      <button
        class="image-preview__btn"
        :disabled="!canZoomOut"
        @click="zoomOut"
        title="Zoom out (-)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 13H5v-2h14v2z"/>
        </svg>
      </button>

      <div class="image-preview__zoom-display">
        {{ zoomPercent }}%
      </div>

      <button
        class="image-preview__btn"
        :disabled="!canZoomIn"
        @click="zoomIn"
        title="Zoom in (+)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
        </svg>
      </button>

      <div class="image-preview__divider"></div>

      <button
        class="image-preview__btn"
        @click="resetZoom"
        title="Reset zoom (0)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/>
        </svg>
      </button>

      <button
        class="image-preview__btn"
        @click="fitToScreen"
        title="Fit to screen"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M3 5v14h18V5H3zm16 12H5V7h14v10z"/>
        </svg>
      </button>
    </div>

    <!-- Image Info -->
    <div v-if="showInfo && naturalSize.width" class="image-preview__info">
      <span>{{ naturalSize.width }} x {{ naturalSize.height }}</span>
    </div>

    <!-- Close Button -->
    <button
      class="image-preview__close"
      @click="$emit('close')"
      title="Close (Esc)"
    >
      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
      </svg>
    </button>

    <!-- Help Hint -->
    <div class="image-preview__hint">
      <span>Scroll to zoom</span>
      <span>Double-click to toggle zoom</span>
      <span>Drag to pan</span>
    </div>
  </div>
</template>

<style scoped>
.image-preview {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  background-color: #000;
  overflow: hidden;
  outline: none;
}

/* Loading */
.image-preview__loading {
  position: absolute;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: white;
  z-index: 10;
}

.image-preview__spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Image */
.image-preview__image-container {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.image-preview__image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  transition: transform 0.1s ease-out;
  user-select: none;
}

/* Controls */
.image-preview__controls {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background-color: rgba(0, 0, 0, 0.7);
  border-radius: 8px;
  backdrop-filter: blur(8px);
}

.image-preview__btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background-color: transparent;
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-preview__btn:hover:not(:disabled) {
  background-color: rgba(255, 255, 255, 0.2);
}

.image-preview__btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.image-preview__zoom-display {
  min-width: 50px;
  text-align: center;
  font-size: 0.875rem;
  font-weight: 500;
  color: white;
}

.image-preview__divider {
  width: 1px;
  height: 24px;
  background-color: rgba(255, 255, 255, 0.3);
  margin: 0 4px;
}

/* Info */
.image-preview__info {
  position: absolute;
  top: 20px;
  left: 20px;
  padding: 6px 12px;
  background-color: rgba(0, 0, 0, 0.7);
  border-radius: 6px;
  font-size: 0.75rem;
  color: white;
  backdrop-filter: blur(8px);
}

/* Close */
.image-preview__close {
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
  backdrop-filter: blur(8px);
}

.image-preview__close:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

/* Hint */
.image-preview__hint {
  position: absolute;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 16px;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.image-preview:hover .image-preview__hint {
  opacity: 1;
}

.image-preview__hint span {
  white-space: nowrap;
}
</style>
