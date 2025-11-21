<script setup lang="ts">
/**
 * VoiceControls Component
 * Playback controls for audio (play, pause, speed)
 */

import { ref, computed, watch, onMounted, onUnmounted } from 'vue';

// ============================================================================
// Props & Emits
// ============================================================================

interface Props {
  audioUrl?: string | null;
  autoPlay?: boolean;
  loop?: boolean;
  showSpeedControl?: boolean;
  showVolumeControl?: boolean;
  showProgressBar?: boolean;
  showTimestamps?: boolean;
  speedOptions?: number[];
}

const props = withDefaults(defineProps<Props>(), {
  audioUrl: null,
  autoPlay: false,
  loop: false,
  showSpeedControl: true,
  showVolumeControl: true,
  showProgressBar: true,
  showTimestamps: true,
  speedOptions: () => [0.5, 0.75, 1, 1.25, 1.5, 2]
});

const emit = defineEmits<{
  (e: 'play'): void;
  (e: 'pause'): void;
  (e: 'ended'): void;
  (e: 'timeupdate', time: number): void;
  (e: 'seek', time: number): void;
  (e: 'speedchange', speed: number): void;
  (e: 'volumechange', volume: number): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const audioRef = ref<HTMLAudioElement | null>(null);
const isPlaying = ref(false);
const currentTime = ref(0);
const duration = ref(0);
const volume = ref(1);
const playbackSpeed = ref(1);
const isMuted = ref(false);
const isLoading = ref(false);
const showSpeedMenu = ref(false);

// ============================================================================
// Computed
// ============================================================================

const progress = computed(() => {
  if (!duration.value) return 0;
  return (currentTime.value / duration.value) * 100;
});

const formattedCurrentTime = computed(() => formatTime(currentTime.value));
const formattedDuration = computed(() => formatTime(duration.value));

const volumeIcon = computed(() => {
  if (isMuted.value || volume.value === 0) return 'muted';
  if (volume.value < 0.5) return 'low';
  return 'high';
});

// ============================================================================
// Methods
// ============================================================================

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function togglePlay(): void {
  if (!audioRef.value) return;

  if (isPlaying.value) {
    audioRef.value.pause();
  } else {
    audioRef.value.play();
  }
}

function handlePlay(): void {
  isPlaying.value = true;
  emit('play');
}

function handlePause(): void {
  isPlaying.value = false;
  emit('pause');
}

function handleEnded(): void {
  isPlaying.value = false;
  emit('ended');
}

function handleTimeUpdate(): void {
  if (!audioRef.value) return;
  currentTime.value = audioRef.value.currentTime;
  emit('timeupdate', currentTime.value);
}

function handleLoadedMetadata(): void {
  if (!audioRef.value) return;
  duration.value = audioRef.value.duration;
  isLoading.value = false;
}

function handleLoadStart(): void {
  isLoading.value = true;
}

function handleCanPlay(): void {
  isLoading.value = false;
}

function seek(time: number): void {
  if (!audioRef.value) return;
  audioRef.value.currentTime = Math.max(0, Math.min(time, duration.value));
  emit('seek', audioRef.value.currentTime);
}

function handleProgressClick(event: MouseEvent): void {
  const target = event.currentTarget as HTMLElement;
  const rect = target.getBoundingClientRect();
  const percent = (event.clientX - rect.left) / rect.width;
  const time = percent * duration.value;
  seek(time);
}

function skip(seconds: number): void {
  seek(currentTime.value + seconds);
}

function setSpeed(speed: number): void {
  if (!audioRef.value) return;
  playbackSpeed.value = speed;
  audioRef.value.playbackRate = speed;
  showSpeedMenu.value = false;
  emit('speedchange', speed);
}

function setVolume(newVolume: number): void {
  if (!audioRef.value) return;
  volume.value = Math.max(0, Math.min(1, newVolume));
  audioRef.value.volume = volume.value;
  isMuted.value = volume.value === 0;
  emit('volumechange', volume.value);
}

function toggleMute(): void {
  if (!audioRef.value) return;

  if (isMuted.value) {
    audioRef.value.volume = volume.value || 0.5;
    isMuted.value = false;
  } else {
    audioRef.value.volume = 0;
    isMuted.value = true;
  }
}

function handleVolumeInput(event: Event): void {
  const target = event.target as HTMLInputElement;
  setVolume(parseFloat(target.value));
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

function handleKeydown(event: KeyboardEvent): void {
  switch (event.key) {
    case ' ':
      event.preventDefault();
      togglePlay();
      break;
    case 'ArrowLeft':
      skip(-5);
      break;
    case 'ArrowRight':
      skip(5);
      break;
    case 'ArrowUp':
      setVolume(volume.value + 0.1);
      break;
    case 'ArrowDown':
      setVolume(volume.value - 0.1);
      break;
    case 'm':
      toggleMute();
      break;
  }
}

// ============================================================================
// Watchers
// ============================================================================

watch(() => props.audioUrl, (newUrl) => {
  if (newUrl && audioRef.value) {
    audioRef.value.load();
    if (props.autoPlay) {
      audioRef.value.play();
    }
  }
});

// ============================================================================
// Lifecycle
// ============================================================================

onMounted(() => {
  document.addEventListener('keydown', handleKeydown);
});

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown);
});

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  play: () => audioRef.value?.play(),
  pause: () => audioRef.value?.pause(),
  seek,
  setSpeed,
  setVolume,
  currentTime,
  duration,
  isPlaying
});
</script>

<template>
  <div class="voice-controls" tabindex="0">
    <!-- Hidden Audio Element -->
    <audio
      ref="audioRef"
      :src="audioUrl || undefined"
      :loop="loop"
      preload="metadata"
      @play="handlePlay"
      @pause="handlePause"
      @ended="handleEnded"
      @timeupdate="handleTimeUpdate"
      @loadedmetadata="handleLoadedMetadata"
      @loadstart="handleLoadStart"
      @canplay="handleCanPlay"
    />

    <!-- Progress Bar -->
    <div
      v-if="showProgressBar"
      class="voice-controls__progress"
      @click="handleProgressClick"
    >
      <div class="voice-controls__progress-track">
        <div
          class="voice-controls__progress-fill"
          :style="{ width: `${progress}%` }"
        />
        <div
          class="voice-controls__progress-handle"
          :style="{ left: `${progress}%` }"
        />
      </div>
    </div>

    <!-- Main Controls -->
    <div class="voice-controls__main">
      <!-- Timestamps (Left) -->
      <div v-if="showTimestamps" class="voice-controls__time">
        {{ formattedCurrentTime }} / {{ formattedDuration }}
      </div>

      <!-- Playback Controls (Center) -->
      <div class="voice-controls__playback">
        <!-- Skip Back -->
        <button
          class="voice-controls__btn voice-controls__btn--skip"
          @click="skip(-10)"
          title="Skip back 10 seconds"
          :disabled="!audioUrl"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
            <text x="12" y="15" text-anchor="middle" font-size="7" font-weight="bold">10</text>
          </svg>
        </button>

        <!-- Play/Pause -->
        <button
          class="voice-controls__btn voice-controls__btn--play"
          @click="togglePlay"
          :disabled="!audioUrl || isLoading"
          :title="isPlaying ? 'Pause' : 'Play'"
        >
          <!-- Loading -->
          <svg v-if="isLoading" width="24" height="24" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" opacity="0.3"/>
            <path
              d="M12 2a10 10 0 0 1 10 10"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
            >
              <animateTransform
                attributeName="transform"
                type="rotate"
                from="0 12 12"
                to="360 12 12"
                dur="1s"
                repeatCount="indefinite"
              />
            </path>
          </svg>
          <!-- Play -->
          <svg v-else-if="!isPlaying" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
          <!-- Pause -->
          <svg v-else width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="4" width="4" height="16"/>
            <rect x="14" y="4" width="4" height="16"/>
          </svg>
        </button>

        <!-- Skip Forward -->
        <button
          class="voice-controls__btn voice-controls__btn--skip"
          @click="skip(10)"
          title="Skip forward 10 seconds"
          :disabled="!audioUrl"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z"/>
            <text x="12" y="15" text-anchor="middle" font-size="7" font-weight="bold">10</text>
          </svg>
        </button>
      </div>

      <!-- Right Controls -->
      <div class="voice-controls__right">
        <!-- Speed Control -->
        <div v-if="showSpeedControl" class="voice-controls__speed">
          <button
            class="voice-controls__btn voice-controls__btn--speed"
            @click="showSpeedMenu = !showSpeedMenu"
            title="Playback speed"
          >
            {{ playbackSpeed }}x
          </button>

          <div v-if="showSpeedMenu" class="voice-controls__speed-menu">
            <button
              v-for="speed in speedOptions"
              :key="speed"
              class="voice-controls__speed-option"
              :class="{ 'voice-controls__speed-option--active': playbackSpeed === speed }"
              @click="setSpeed(speed)"
            >
              {{ speed }}x
            </button>
          </div>
        </div>

        <!-- Volume Control -->
        <div v-if="showVolumeControl" class="voice-controls__volume">
          <button
            class="voice-controls__btn"
            @click="toggleMute"
            :title="isMuted ? 'Unmute' : 'Mute'"
          >
            <!-- Muted -->
            <svg v-if="volumeIcon === 'muted'" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
            </svg>
            <!-- Low Volume -->
            <svg v-else-if="volumeIcon === 'low'" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z"/>
            </svg>
            <!-- High Volume -->
            <svg v-else width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
            </svg>
          </button>

          <input
            type="range"
            class="voice-controls__volume-slider"
            min="0"
            max="1"
            step="0.05"
            :value="isMuted ? 0 : volume"
            @input="handleVolumeInput"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.voice-controls {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 12px 16px;
  background-color: #f9fafb;
  border-radius: 12px;
  outline: none;
}

.voice-controls:focus {
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
}

/* Progress Bar */
.voice-controls__progress {
  cursor: pointer;
  padding: 8px 0;
}

.voice-controls__progress-track {
  position: relative;
  height: 4px;
  background-color: #e5e7eb;
  border-radius: 2px;
}

.voice-controls__progress-fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background-color: #3b82f6;
  border-radius: 2px;
  transition: width 0.1s linear;
}

.voice-controls__progress-handle {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  background-color: #3b82f6;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  opacity: 0;
  transition: opacity 0.2s ease;
}

.voice-controls__progress:hover .voice-controls__progress-handle {
  opacity: 1;
}

/* Main Controls */
.voice-controls__main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.voice-controls__time {
  font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
  font-size: 0.75rem;
  color: #6b7280;
  min-width: 80px;
}

.voice-controls__playback {
  display: flex;
  align-items: center;
  gap: 8px;
}

.voice-controls__right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Buttons */
.voice-controls__btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
  border: none;
  border-radius: 8px;
  background-color: transparent;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
}

.voice-controls__btn:hover:not(:disabled) {
  background-color: #e5e7eb;
}

.voice-controls__btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.voice-controls__btn--play {
  width: 48px;
  height: 48px;
  background-color: #3b82f6;
  color: white;
  border-radius: 50%;
}

.voice-controls__btn--play:hover:not(:disabled) {
  background-color: #2563eb;
}

.voice-controls__btn--skip {
  width: 36px;
  height: 36px;
}

.voice-controls__btn--speed {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 4px 8px;
}

/* Speed Menu */
.voice-controls__speed {
  position: relative;
}

.voice-controls__speed-menu {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  padding: 4px;
  margin-bottom: 8px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 10;
}

.voice-controls__speed-option {
  padding: 6px 12px;
  border: none;
  background-color: transparent;
  font-size: 0.875rem;
  color: #374151;
  cursor: pointer;
  border-radius: 4px;
  white-space: nowrap;
}

.voice-controls__speed-option:hover {
  background-color: #f3f4f6;
}

.voice-controls__speed-option--active {
  background-color: #eff6ff;
  color: #3b82f6;
  font-weight: 600;
}

/* Volume */
.voice-controls__volume {
  display: flex;
  align-items: center;
  gap: 8px;
}

.voice-controls__volume-slider {
  width: 80px;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: #e5e7eb;
  border-radius: 2px;
  outline: none;
}

.voice-controls__volume-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 12px;
  height: 12px;
  background: #3b82f6;
  border-radius: 50%;
  cursor: pointer;
}

.voice-controls__volume-slider::-moz-range-thumb {
  width: 12px;
  height: 12px;
  background: #3b82f6;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}
</style>
