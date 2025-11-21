<script setup lang="ts">
/**
 * VoiceRecorder Component
 * Record button with visual feedback and duration display
 */

import { computed, ref, watch } from 'vue';
import { useVoice, type VoiceRecordingOptions } from '@/composables/useVoice';

// ============================================================================
// Props & Emits
// ============================================================================

interface Props {
  maxDuration?: number;
  showDuration?: boolean;
  showWaveform?: boolean;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  color?: string;
}

const props = withDefaults(defineProps<Props>(), {
  maxDuration: 300,
  showDuration: true,
  showWaveform: true,
  disabled: false,
  size: 'md',
  color: '#ef4444'
});

const emit = defineEmits<{
  (e: 'start'): void;
  (e: 'stop', blob: Blob): void;
  (e: 'pause'): void;
  (e: 'resume'): void;
  (e: 'error', error: string): void;
  (e: 'data', data: Blob): void;
}>();

// ============================================================================
// Composable
// ============================================================================

const voiceOptions: VoiceRecordingOptions = {
  maxDuration: props.maxDuration,
  onDataAvailable: (data) => emit('data', data),
  onStop: (blob) => emit('stop', blob),
  onError: (err) => emit('error', err.message)
};

const {
  isRecording,
  isPaused,
  duration,
  formattedDuration,
  error,
  isSupported,
  analyzerData,
  startRecording,
  stopRecording,
  pauseRecording,
  resumeRecording,
  resetRecording
} = useVoice(voiceOptions);

// ============================================================================
// Local State
// ============================================================================

const isHovering = ref(false);

// ============================================================================
// Computed
// ============================================================================

const buttonSize = computed(() => {
  const sizes = {
    sm: { button: '48px', icon: '20px' },
    md: { button: '64px', icon: '28px' },
    lg: { button: '80px', icon: '36px' }
  };
  return sizes[props.size];
});

const buttonClasses = computed(() => [
  'voice-recorder__button',
  {
    'voice-recorder__button--recording': isRecording.value,
    'voice-recorder__button--paused': isPaused.value,
    'voice-recorder__button--disabled': props.disabled || !isSupported.value,
    'voice-recorder__button--hovering': isHovering.value
  }
]);

const progressPercent = computed(() => {
  if (!props.maxDuration) return 0;
  return Math.min((duration.value / props.maxDuration) * 100, 100);
});

const volumeScale = computed(() => {
  if (!analyzerData.value) return 1;
  return 1 + analyzerData.value.volume * 0.3;
});

// ============================================================================
// Methods
// ============================================================================

async function handleClick() {
  if (props.disabled || !isSupported.value) return;

  if (isRecording.value) {
    await stopRecording();
  } else {
    emit('start');
    await startRecording();
  }
}

function handlePauseToggle() {
  if (!isRecording.value) return;

  if (isPaused.value) {
    resumeRecording();
    emit('resume');
  } else {
    pauseRecording();
    emit('pause');
  }
}

function handleReset() {
  resetRecording();
}

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  isRecording,
  isPaused,
  duration,
  startRecording,
  stopRecording,
  pauseRecording,
  resumeRecording,
  resetRecording
});
</script>

<template>
  <div class="voice-recorder">
    <!-- Error Message -->
    <div v-if="error" class="voice-recorder__error">
      {{ error }}
    </div>

    <!-- Not Supported Warning -->
    <div v-if="!isSupported" class="voice-recorder__warning">
      Voice recording is not supported in this browser
    </div>

    <!-- Main Recording Area -->
    <div class="voice-recorder__main">
      <!-- Progress Ring -->
      <svg
        v-if="isRecording && maxDuration"
        class="voice-recorder__progress"
        :width="buttonSize.button"
        :height="buttonSize.button"
      >
        <circle
          class="voice-recorder__progress-bg"
          :cx="parseInt(buttonSize.button) / 2"
          :cy="parseInt(buttonSize.button) / 2"
          :r="parseInt(buttonSize.button) / 2 - 4"
          fill="none"
          stroke-width="3"
        />
        <circle
          class="voice-recorder__progress-bar"
          :cx="parseInt(buttonSize.button) / 2"
          :cy="parseInt(buttonSize.button) / 2"
          :r="parseInt(buttonSize.button) / 2 - 4"
          fill="none"
          stroke-width="3"
          :stroke="color"
          :stroke-dasharray="`${2 * Math.PI * (parseInt(buttonSize.button) / 2 - 4)}`"
          :stroke-dashoffset="`${2 * Math.PI * (parseInt(buttonSize.button) / 2 - 4) * (1 - progressPercent / 100)}`"
        />
      </svg>

      <!-- Record Button -->
      <button
        :class="buttonClasses"
        :style="{
          width: buttonSize.button,
          height: buttonSize.button,
          '--record-color': color,
          transform: isRecording ? `scale(${volumeScale})` : 'scale(1)'
        }"
        :disabled="disabled || !isSupported"
        @click="handleClick"
        @mouseenter="isHovering = true"
        @mouseleave="isHovering = false"
        :aria-label="isRecording ? 'Stop recording' : 'Start recording'"
      >
        <!-- Microphone Icon (not recording) -->
        <svg
          v-if="!isRecording"
          :width="buttonSize.icon"
          :height="buttonSize.icon"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
        </svg>

        <!-- Stop Icon (recording) -->
        <svg
          v-else
          :width="buttonSize.icon"
          :height="buttonSize.icon"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <rect x="6" y="6" width="12" height="12" rx="2"/>
        </svg>
      </button>
    </div>

    <!-- Duration Display -->
    <div
      v-if="showDuration && (isRecording || duration > 0)"
      class="voice-recorder__duration"
    >
      <span class="voice-recorder__time">{{ formattedDuration }}</span>
      <span v-if="isPaused" class="voice-recorder__paused-label">PAUSED</span>
      <span v-else-if="isRecording" class="voice-recorder__recording-label">
        <span class="voice-recorder__pulse"></span>
        REC
      </span>
    </div>

    <!-- Controls -->
    <div v-if="isRecording" class="voice-recorder__controls">
      <button
        class="voice-recorder__control-btn"
        @click="handlePauseToggle"
        :aria-label="isPaused ? 'Resume recording' : 'Pause recording'"
      >
        <!-- Pause Icon -->
        <svg v-if="!isPaused" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="4" width="4" height="16"/>
          <rect x="14" y="4" width="4" height="16"/>
        </svg>
        <!-- Play Icon -->
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5v14l11-7z"/>
        </svg>
      </button>

      <button
        class="voice-recorder__control-btn voice-recorder__control-btn--danger"
        @click="handleReset"
        aria-label="Cancel recording"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.voice-recorder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.voice-recorder__main {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.voice-recorder__progress {
  position: absolute;
  transform: rotate(-90deg);
  pointer-events: none;
}

.voice-recorder__progress-bg {
  stroke: rgba(0, 0, 0, 0.1);
}

.voice-recorder__progress-bar {
  transition: stroke-dashoffset 0.1s linear;
}

.voice-recorder__button {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 50%;
  background-color: #f3f4f6;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.voice-recorder__button:hover:not(:disabled) {
  background-color: #e5e7eb;
  transform: scale(1.05);
}

.voice-recorder__button--recording {
  background-color: var(--record-color, #ef4444);
  color: white;
  animation: pulse 1.5s infinite;
}

.voice-recorder__button--paused {
  animation: none;
  opacity: 0.8;
}

.voice-recorder__button--disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
  }
  50% {
    box-shadow: 0 0 0 12px rgba(239, 68, 68, 0);
  }
}

.voice-recorder__duration {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
}

.voice-recorder__time {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
}

.voice-recorder__recording-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #ef4444;
  text-transform: uppercase;
}

.voice-recorder__pulse {
  width: 8px;
  height: 8px;
  background-color: #ef4444;
  border-radius: 50%;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.voice-recorder__paused-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #f59e0b;
  text-transform: uppercase;
}

.voice-recorder__controls {
  display: flex;
  gap: 8px;
}

.voice-recorder__control-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 50%;
  background-color: #f3f4f6;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
}

.voice-recorder__control-btn:hover {
  background-color: #e5e7eb;
}

.voice-recorder__control-btn--danger:hover {
  background-color: #fee2e2;
  color: #ef4444;
}

.voice-recorder__error {
  padding: 8px 16px;
  background-color: #fee2e2;
  color: #dc2626;
  border-radius: 6px;
  font-size: 0.875rem;
}

.voice-recorder__warning {
  padding: 8px 16px;
  background-color: #fef3c7;
  color: #d97706;
  border-radius: 6px;
  font-size: 0.875rem;
}
</style>
