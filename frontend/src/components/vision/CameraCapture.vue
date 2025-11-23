<script setup lang="ts">
/**
 * CameraCapture Component
 * Webcam capture with facing mode toggle and photo capture
 */

import { ref, computed, onMounted, onUnmounted, watch } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface CapturedPhoto {
  id: string;
  dataUrl: string;
  blob: Blob;
  timestamp: number;
  width: number;
  height: number;
  facingMode: FacingMode;
}

export type FacingMode = 'user' | 'environment';

export interface CameraCaptureProps {
  autoStart?: boolean;
  facingMode?: FacingMode;
  width?: number;
  height?: number;
  quality?: number;
  format?: 'image/jpeg' | 'image/png' | 'image/webp';
  showControls?: boolean;
  showPreview?: boolean;
  mirror?: boolean;
}

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<CameraCaptureProps>(), {
  autoStart: false,
  facingMode: 'user',
  width: 1280,
  height: 720,
  quality: 0.92,
  format: 'image/jpeg',
  showControls: true,
  showPreview: true,
  mirror: true
});

const emit = defineEmits<{
  (e: 'capture', photo: CapturedPhoto): void;
  (e: 'error', error: string): void;
  (e: 'streamStart'): void;
  (e: 'streamStop'): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const stream = ref<MediaStream | null>(null);
const isStreaming = ref(false);
const isLoading = ref(false);
const error = ref<string | null>(null);
const currentFacingMode = ref<FacingMode>(props.facingMode);
const capturedPhotos = ref<CapturedPhoto[]>([]);
const countdown = ref<number | null>(null);
const hasMultipleCameras = ref(false);

// ============================================================================
// Computed
// ============================================================================

const videoStyle = computed(() => ({
  transform: props.mirror && currentFacingMode.value === 'user' ? 'scaleX(-1)' : 'none'
}));

const isSupported = computed(() => {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
});

// ============================================================================
// Methods
// ============================================================================

function generateId(): string {
  return `photo-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

async function checkCameras(): Promise<void> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    hasMultipleCameras.value = videoDevices.length > 1;
  } catch {
    hasMultipleCameras.value = false;
  }
}

async function startCamera(): Promise<void> {
  if (!isSupported.value) {
    error.value = 'Camera is not supported in this browser';
    emit('error', error.value);
    return;
  }

  isLoading.value = true;
  error.value = null;

  try {
    // Stop existing stream
    stopCamera();

    const constraints: MediaStreamConstraints = {
      video: {
        facingMode: currentFacingMode.value,
        width: { ideal: props.width },
        height: { ideal: props.height }
      },
      audio: false
    };

    stream.value = await navigator.mediaDevices.getUserMedia(constraints);

    if (videoRef.value) {
      videoRef.value.srcObject = stream.value;
      await videoRef.value.play();
    }

    isStreaming.value = true;
    emit('streamStart');
    await checkCameras();

  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Failed to access camera';
    error.value = errorMessage;
    emit('error', errorMessage);
  } finally {
    isLoading.value = false;
  }
}

function stopCamera(): void {
  if (stream.value) {
    stream.value.getTracks().forEach(track => track.stop());
    stream.value = null;
  }

  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }

  isStreaming.value = false;
  emit('streamStop');
}

async function switchCamera(): Promise<void> {
  currentFacingMode.value = currentFacingMode.value === 'user' ? 'environment' : 'user';
  await startCamera();
}

async function capturePhoto(): Promise<CapturedPhoto | null> {
  if (!videoRef.value || !canvasRef.value || !isStreaming.value) {
    return null;
  }

  const video = videoRef.value;
  const canvas = canvasRef.value;
  const ctx = canvas.getContext('2d');

  if (!ctx) return null;

  // Set canvas size to video dimensions
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Apply mirror if needed
  if (props.mirror && currentFacingMode.value === 'user') {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }

  // Draw video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Reset transform
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        resolve(null);
        return;
      }

      const dataUrl = canvas.toDataURL(props.format, props.quality);

      const photo: CapturedPhoto = {
        id: generateId(),
        dataUrl,
        blob,
        timestamp: Date.now(),
        width: canvas.width,
        height: canvas.height,
        facingMode: currentFacingMode.value
      };

      capturedPhotos.value.unshift(photo);
      emit('capture', photo);
      resolve(photo);

    }, props.format, props.quality);
  });
}

async function captureWithCountdown(seconds: number = 3): Promise<void> {
  countdown.value = seconds;

  const interval = setInterval(() => {
    countdown.value!--;
    if (countdown.value === 0) {
      clearInterval(interval);
      countdown.value = null;
      capturePhoto();
    }
  }, 1000);
}

function deletePhoto(photoId: string): void {
  capturedPhotos.value = capturedPhotos.value.filter(p => p.id !== photoId);
}

function clearPhotos(): void {
  capturedPhotos.value = [];
}

async function downloadPhoto(photo: CapturedPhoto): Promise<void> {
  const link = document.createElement('a');
  link.href = photo.dataUrl;
  link.download = `capture-${photo.timestamp}.${props.format.split('/')[1]}`;
  link.click();
}

// ============================================================================
// Lifecycle
// ============================================================================

onMounted(async () => {
  if (props.autoStart) {
    await startCamera();
  }
});

onUnmounted(() => {
  stopCamera();
});

// ============================================================================
// Watchers
// ============================================================================

watch(() => props.facingMode, (newMode) => {
  if (newMode !== currentFacingMode.value) {
    currentFacingMode.value = newMode;
    if (isStreaming.value) {
      startCamera();
    }
  }
});

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  startCamera,
  stopCamera,
  switchCamera,
  capturePhoto,
  captureWithCountdown,
  capturedPhotos,
  isStreaming,
  currentFacingMode
});
</script>

<template>
  <div class="camera-capture">
    <!-- Camera View -->
    <div class="camera-capture__viewport">
      <!-- Video Element -->
      <video
        ref="videoRef"
        class="camera-capture__video"
        :style="videoStyle"
        playsinline
        muted
      />

      <!-- Canvas (hidden, for capture) -->
      <canvas ref="canvasRef" class="camera-capture__canvas" />

      <!-- Loading Overlay -->
      <div v-if="isLoading" class="camera-capture__overlay">
        <div class="camera-capture__spinner"></div>
        <span>Starting camera...</span>
      </div>

      <!-- Error Overlay -->
      <div v-else-if="error" class="camera-capture__overlay camera-capture__overlay--error">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        <p>{{ error }}</p>
        <button class="camera-capture__btn" @click="startCamera">
          Try Again
        </button>
      </div>

      <!-- Inactive State -->
      <div v-else-if="!isStreaming" class="camera-capture__overlay">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
          <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
        </svg>
        <p>Camera is off</p>
        <button class="camera-capture__btn" @click="startCamera">
          Start Camera
        </button>
      </div>

      <!-- Countdown Overlay -->
      <div v-if="countdown !== null" class="camera-capture__countdown">
        {{ countdown }}
      </div>

      <!-- Capture Flash -->
      <div class="camera-capture__flash" />
    </div>

    <!-- Controls -->
    <div v-if="showControls" class="camera-capture__controls">
      <!-- Start/Stop -->
      <button
        class="camera-capture__control-btn"
        @click="isStreaming ? stopCamera() : startCamera()"
        :title="isStreaming ? 'Stop camera' : 'Start camera'"
      >
        <svg v-if="isStreaming" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M21 6.5l-4 4V7c0-.55-.45-1-1-1H9.82L21 17.18V6.5zM3.27 2L2 3.27 4.73 6H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.21 0 .39-.08.54-.18L19.73 21 21 19.73 3.27 2z"/>
        </svg>
        <svg v-else width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
        </svg>
      </button>

      <!-- Switch Camera -->
      <button
        v-if="hasMultipleCameras"
        class="camera-capture__control-btn"
        :disabled="!isStreaming"
        @click="switchCamera"
        title="Switch camera"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M9 12c0 1.66 1.34 3 3 3s3-1.34 3-3-1.34-3-3-3-3 1.34-3 3zm9-3v2h-2V9h2zm2-4H4v14h16V5zm-5 3.5V7H9v1.5l-3 3 3 3V13h5v1.5l3-3-3-3z"/>
        </svg>
      </button>

      <!-- Capture Button -->
      <button
        class="camera-capture__capture-btn"
        :disabled="!isStreaming || countdown !== null"
        @click="capturePhoto"
        title="Take photo"
      >
        <div class="camera-capture__capture-inner" />
      </button>

      <!-- Timer Capture -->
      <button
        class="camera-capture__control-btn"
        :disabled="!isStreaming || countdown !== null"
        @click="captureWithCountdown(3)"
        title="3 second timer"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M15 1H9v2h6V1zm-4 13h2V8h-2v6zm8.03-6.61l1.42-1.42c-.43-.51-.9-.99-1.41-1.41l-1.42 1.42C16.07 4.74 14.12 4 12 4c-4.97 0-9 4.03-9 9s4.02 9 9 9 9-4.03 9-9c0-2.12-.74-4.07-1.97-5.61zM12 20c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z"/>
        </svg>
      </button>
    </div>

    <!-- Photo Preview Gallery -->
    <div v-if="showPreview && capturedPhotos.length > 0" class="camera-capture__gallery">
      <div class="camera-capture__gallery-header">
        <span>Captured Photos ({{ capturedPhotos.length }})</span>
        <button
          class="camera-capture__gallery-clear"
          @click="clearPhotos"
        >
          Clear All
        </button>
      </div>

      <div class="camera-capture__gallery-grid">
        <div
          v-for="photo in capturedPhotos"
          :key="photo.id"
          class="camera-capture__photo"
        >
          <img :src="photo.dataUrl" :alt="`Captured at ${photo.timestamp}`" />
          <div class="camera-capture__photo-overlay">
            <button
              class="camera-capture__photo-btn"
              @click="downloadPhoto(photo)"
              title="Download"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
              </svg>
            </button>
            <button
              class="camera-capture__photo-btn camera-capture__photo-btn--delete"
              @click="deletePhoto(photo.id)"
              title="Delete"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.camera-capture {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Viewport */
.camera-capture__viewport {
  position: relative;
  aspect-ratio: 16/9;
  background-color: #000;
  border-radius: 12px;
  overflow: hidden;
}

.camera-capture__video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.camera-capture__canvas {
  display: none;
}

/* Overlays */
.camera-capture__overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
}

.camera-capture__overlay--error {
  color: #fca5a5;
}

.camera-capture__overlay p {
  margin: 0;
  font-size: 0.875rem;
}

.camera-capture__spinner {
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

.camera-capture__btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background-color: #3b82f6;
  color: white;
  font-size: 0.875rem;
  cursor: pointer;
}

.camera-capture__btn:hover {
  background-color: #2563eb;
}

/* Countdown */
.camera-capture__countdown {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 6rem;
  font-weight: bold;
  color: white;
  background-color: rgba(0, 0, 0, 0.5);
  animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

/* Flash */
.camera-capture__flash {
  position: absolute;
  inset: 0;
  background-color: white;
  opacity: 0;
  pointer-events: none;
  animation: flash 0.2s ease-out;
}

@keyframes flash {
  0% { opacity: 0.8; }
  100% { opacity: 0; }
}

/* Controls */
.camera-capture__controls {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 12px;
  background-color: #f9fafb;
  border-radius: 12px;
}

.camera-capture__control-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background-color: #e5e7eb;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
}

.camera-capture__control-btn:hover:not(:disabled) {
  background-color: #d1d5db;
}

.camera-capture__control-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.camera-capture__capture-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 64px;
  height: 64px;
  border: 4px solid #374151;
  border-radius: 50%;
  background-color: white;
  cursor: pointer;
  transition: all 0.2s ease;
}

.camera-capture__capture-btn:hover:not(:disabled) {
  border-color: #ef4444;
}

.camera-capture__capture-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.camera-capture__capture-inner {
  width: 48px;
  height: 48px;
  background-color: #ef4444;
  border-radius: 50%;
  transition: all 0.2s ease;
}

.camera-capture__capture-btn:hover:not(:disabled) .camera-capture__capture-inner {
  transform: scale(0.9);
}

/* Gallery */
.camera-capture__gallery {
  background-color: #f9fafb;
  border-radius: 12px;
  padding: 12px;
}

.camera-capture__gallery-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  font-size: 0.875rem;
  color: #374151;
}

.camera-capture__gallery-clear {
  padding: 4px 8px;
  border: none;
  border-radius: 4px;
  background-color: transparent;
  color: #6b7280;
  font-size: 0.75rem;
  cursor: pointer;
}

.camera-capture__gallery-clear:hover {
  background-color: #e5e7eb;
  color: #ef4444;
}

.camera-capture__gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 8px;
}

.camera-capture__photo {
  position: relative;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
}

.camera-capture__photo img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.camera-capture__photo-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  background-color: rgba(0, 0, 0, 0.5);
  opacity: 0;
  transition: opacity 0.2s ease;
}

.camera-capture__photo:hover .camera-capture__photo-overlay {
  opacity: 1;
}

.camera-capture__photo-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background-color: rgba(255, 255, 255, 0.9);
  color: #374151;
  cursor: pointer;
}

.camera-capture__photo-btn:hover {
  background-color: white;
}

.camera-capture__photo-btn--delete:hover {
  background-color: #fee2e2;
  color: #ef4444;
}
</style>
