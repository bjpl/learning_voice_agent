<script setup lang="ts">
/**
 * VoiceWaveform Component
 * Canvas-based audio waveform visualization
 */

import { ref, watch, onMounted, onUnmounted, computed } from 'vue';

// ============================================================================
// Props & Emits
// ============================================================================

interface Props {
  audioUrl?: string | null;
  analyzerData?: {
    frequencyData: Uint8Array;
    timeDomainData: Uint8Array;
    volume: number;
  } | null;
  width?: number;
  height?: number;
  barWidth?: number;
  barGap?: number;
  barColor?: string;
  barActiveColor?: string;
  backgroundColor?: string;
  mode?: 'frequency' | 'waveform' | 'bars';
  animate?: boolean;
  currentTime?: number;
  duration?: number;
}

const props = withDefaults(defineProps<Props>(), {
  audioUrl: null,
  analyzerData: null,
  width: 300,
  height: 100,
  barWidth: 3,
  barGap: 2,
  barColor: '#cbd5e1',
  barActiveColor: '#3b82f6',
  backgroundColor: 'transparent',
  mode: 'bars',
  animate: false,
  currentTime: 0,
  duration: 0
});

const emit = defineEmits<{
  (e: 'seek', time: number): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const canvasRef = ref<HTMLCanvasElement | null>(null);
const containerRef = ref<HTMLDivElement | null>(null);
const waveformData = ref<number[]>([]);
const isLoading = ref(false);
const animationFrameId = ref<number | null>(null);

// ============================================================================
// Computed
// ============================================================================

const progressPercent = computed(() => {
  if (!props.duration) return 0;
  return (props.currentTime / props.duration) * 100;
});

// ============================================================================
// Audio Analysis
// ============================================================================

async function analyzeAudio(url: string): Promise<number[]> {
  isLoading.value = true;

  try {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();

    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0);
    const samples = Math.floor(props.width / (props.barWidth + props.barGap));
    const blockSize = Math.floor(channelData.length / samples);
    const data: number[] = [];

    for (let i = 0; i < samples; i++) {
      const start = blockSize * i;
      let sum = 0;

      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(channelData[start + j] || 0);
      }

      data.push(sum / blockSize);
    }

    // Normalize
    const max = Math.max(...data);
    const normalized = data.map(v => v / max);

    audioContext.close();
    return normalized;

  } catch (err) {
    console.error('Failed to analyze audio:', err);
    return [];
  } finally {
    isLoading.value = false;
  }
}

// ============================================================================
// Drawing Functions
// ============================================================================

function drawStaticWaveform(): void {
  const canvas = canvasRef.value;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { width, height, barWidth, barGap, barColor, barActiveColor, backgroundColor } = props;

  // Clear canvas
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, width, height);

  if (waveformData.value.length === 0) {
    // Draw placeholder
    drawPlaceholder(ctx);
    return;
  }

  const centerY = height / 2;
  const progressIndex = Math.floor((progressPercent.value / 100) * waveformData.value.length);

  waveformData.value.forEach((value, index) => {
    const x = index * (barWidth + barGap);
    const barHeight = Math.max(value * (height - 4), 2);
    const y = centerY - barHeight / 2;

    ctx.fillStyle = index < progressIndex ? barActiveColor : barColor;
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
    ctx.fill();
  });
}

function drawLiveWaveform(): void {
  const canvas = canvasRef.value;
  if (!canvas || !props.analyzerData) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { width, height, barColor, backgroundColor } = props;

  // Clear canvas
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, width, height);

  const { frequencyData, timeDomainData } = props.analyzerData;

  if (props.mode === 'frequency') {
    drawFrequencyBars(ctx, frequencyData);
  } else if (props.mode === 'waveform') {
    drawWaveformLine(ctx, timeDomainData);
  } else {
    drawFrequencyBars(ctx, frequencyData);
  }
}

function drawFrequencyBars(ctx: CanvasRenderingContext2D, data: Uint8Array): void {
  const { width, height, barWidth, barGap, barActiveColor } = props;
  const centerY = height / 2;
  const barCount = Math.floor(width / (barWidth + barGap));
  const step = Math.floor(data.length / barCount);

  for (let i = 0; i < barCount; i++) {
    const value = data[i * step] / 255;
    const barHeight = Math.max(value * (height - 4), 2);
    const x = i * (barWidth + barGap);
    const y = centerY - barHeight / 2;

    // Gradient based on intensity
    const hue = 200 + value * 60;
    ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;

    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
    ctx.fill();
  }
}

function drawWaveformLine(ctx: CanvasRenderingContext2D, data: Uint8Array): void {
  const { width, height, barActiveColor } = props;
  const centerY = height / 2;
  const sliceWidth = width / data.length;

  ctx.lineWidth = 2;
  ctx.strokeStyle = barActiveColor;
  ctx.beginPath();

  let x = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] / 128.0;
    const y = (v * height) / 2;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  ctx.lineTo(width, centerY);
  ctx.stroke();
}

function drawPlaceholder(ctx: CanvasRenderingContext2D): void {
  const { width, height, barWidth, barGap, barColor } = props;
  const centerY = height / 2;
  const barCount = Math.floor(width / (barWidth + barGap));

  for (let i = 0; i < barCount; i++) {
    const x = i * (barWidth + barGap);
    const barHeight = 4 + Math.sin(i * 0.3) * 2;
    const y = centerY - barHeight / 2;

    ctx.fillStyle = barColor;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// ============================================================================
// Animation Loop
// ============================================================================

function animate(): void {
  if (props.analyzerData && props.animate) {
    drawLiveWaveform();
    animationFrameId.value = requestAnimationFrame(animate);
  }
}

function startAnimation(): void {
  if (animationFrameId.value) {
    cancelAnimationFrame(animationFrameId.value);
  }
  animate();
}

function stopAnimation(): void {
  if (animationFrameId.value) {
    cancelAnimationFrame(animationFrameId.value);
    animationFrameId.value = null;
  }
}

// ============================================================================
// Event Handlers
// ============================================================================

function handleCanvasClick(event: MouseEvent): void {
  if (!props.duration || !canvasRef.value) return;

  const rect = canvasRef.value.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const percent = x / rect.width;
  const time = percent * props.duration;

  emit('seek', time);
}

// ============================================================================
// Watchers
// ============================================================================

watch(() => props.audioUrl, async (newUrl) => {
  if (newUrl) {
    waveformData.value = await analyzeAudio(newUrl);
    drawStaticWaveform();
  } else {
    waveformData.value = [];
    drawStaticWaveform();
  }
});

watch(() => props.analyzerData, () => {
  if (props.analyzerData && props.animate) {
    startAnimation();
  }
}, { deep: true });

watch(() => props.animate, (animate) => {
  if (animate && props.analyzerData) {
    startAnimation();
  } else {
    stopAnimation();
    drawStaticWaveform();
  }
});

watch(() => props.currentTime, () => {
  if (!props.animate) {
    drawStaticWaveform();
  }
});

// ============================================================================
// Lifecycle
// ============================================================================

onMounted(async () => {
  if (props.audioUrl) {
    waveformData.value = await analyzeAudio(props.audioUrl);
  }
  drawStaticWaveform();

  if (props.animate && props.analyzerData) {
    startAnimation();
  }
});

onUnmounted(() => {
  stopAnimation();
});

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  redraw: drawStaticWaveform,
  waveformData
});
</script>

<template>
  <div
    ref="containerRef"
    class="voice-waveform"
    :class="{ 'voice-waveform--loading': isLoading, 'voice-waveform--clickable': duration > 0 }"
  >
    <canvas
      ref="canvasRef"
      :width="width"
      :height="height"
      class="voice-waveform__canvas"
      @click="handleCanvasClick"
    />

    <!-- Loading Indicator -->
    <div v-if="isLoading" class="voice-waveform__loading">
      <div class="voice-waveform__spinner"></div>
    </div>

    <!-- Progress Indicator -->
    <div
      v-if="duration > 0 && !animate"
      class="voice-waveform__progress"
      :style="{ left: `${progressPercent}%` }"
    />
  </div>
</template>

<style scoped>
.voice-waveform {
  position: relative;
  display: inline-block;
  border-radius: 8px;
  overflow: hidden;
}

.voice-waveform--loading {
  opacity: 0.6;
}

.voice-waveform--clickable {
  cursor: pointer;
}

.voice-waveform__canvas {
  display: block;
}

.voice-waveform__loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.voice-waveform__spinner {
  width: 24px;
  height: 24px;
  border: 2px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.voice-waveform__progress {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background-color: #3b82f6;
  pointer-events: none;
  transition: left 0.1s linear;
}

.voice-waveform__progress::before {
  content: '';
  position: absolute;
  top: -4px;
  left: -4px;
  width: 10px;
  height: 10px;
  background-color: #3b82f6;
  border-radius: 50%;
  box-shadow: 0 0 4px rgba(59, 130, 246, 0.5);
}
</style>
