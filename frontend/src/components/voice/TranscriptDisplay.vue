<script setup lang="ts">
/**
 * TranscriptDisplay Component
 * Real-time transcript with timestamps and speaker identification
 */

import { ref, computed, watch, nextTick } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface TranscriptSegment {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  speaker?: string;
  confidence?: number;
  isFinal?: boolean;
}

export interface TranscriptDisplayProps {
  segments: TranscriptSegment[];
  currentTime?: number;
  autoScroll?: boolean;
  showTimestamps?: boolean;
  showSpeakers?: boolean;
  showConfidence?: boolean;
  highlightActive?: boolean;
  interimText?: string;
  editable?: boolean;
}

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<TranscriptDisplayProps>(), {
  currentTime: 0,
  autoScroll: true,
  showTimestamps: true,
  showSpeakers: true,
  showConfidence: false,
  highlightActive: true,
  interimText: '',
  editable: false
});

const emit = defineEmits<{
  (e: 'seek', time: number): void;
  (e: 'edit', segment: TranscriptSegment): void;
  (e: 'copy', text: string): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const containerRef = ref<HTMLDivElement | null>(null);
const editingId = ref<string | null>(null);
const editText = ref('');

// ============================================================================
// Computed
// ============================================================================

const activeSegmentId = computed(() => {
  if (!props.highlightActive) return null;

  const active = props.segments.find(
    segment =>
      props.currentTime >= segment.startTime &&
      props.currentTime <= segment.endTime
  );

  return active?.id || null;
});

const fullText = computed(() => {
  return props.segments.map(s => s.text).join(' ');
});

const speakerColors: Record<string, string> = {
  'Speaker 1': '#3b82f6',
  'Speaker 2': '#10b981',
  'Speaker 3': '#f59e0b',
  'Speaker 4': '#ef4444',
  'default': '#6b7280'
};

// ============================================================================
// Methods
// ============================================================================

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
}

function getSpeakerColor(speaker?: string): string {
  if (!speaker) return speakerColors.default;
  return speakerColors[speaker] || speakerColors.default;
}

function getConfidenceClass(confidence?: number): string {
  if (!confidence) return '';
  if (confidence >= 0.9) return 'transcript-display__confidence--high';
  if (confidence >= 0.7) return 'transcript-display__confidence--medium';
  return 'transcript-display__confidence--low';
}

function handleSegmentClick(segment: TranscriptSegment): void {
  emit('seek', segment.startTime);
}

function startEdit(segment: TranscriptSegment): void {
  if (!props.editable) return;
  editingId.value = segment.id;
  editText.value = segment.text;
}

function saveEdit(segment: TranscriptSegment): void {
  if (editText.value !== segment.text) {
    emit('edit', { ...segment, text: editText.value });
  }
  editingId.value = null;
  editText.value = '';
}

function cancelEdit(): void {
  editingId.value = null;
  editText.value = '';
}

async function copyFullText(): Promise<void> {
  try {
    await navigator.clipboard.writeText(fullText.value);
    emit('copy', fullText.value);
  } catch (err) {
    console.error('Failed to copy text:', err);
  }
}

function scrollToActive(): void {
  if (!containerRef.value || !activeSegmentId.value) return;

  const activeElement = containerRef.value.querySelector(
    `[data-segment-id="${activeSegmentId.value}"]`
  );

  if (activeElement) {
    activeElement.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  }
}

// ============================================================================
// Watchers
// ============================================================================

watch(activeSegmentId, () => {
  if (props.autoScroll) {
    nextTick(scrollToActive);
  }
});

watch(() => props.segments.length, () => {
  if (props.autoScroll && containerRef.value) {
    nextTick(() => {
      containerRef.value!.scrollTop = containerRef.value!.scrollHeight;
    });
  }
});

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  copyFullText,
  scrollToActive,
  fullText
});
</script>

<template>
  <div class="transcript-display">
    <!-- Header -->
    <div class="transcript-display__header">
      <h3 class="transcript-display__title">Transcript</h3>

      <div class="transcript-display__actions">
        <button
          v-if="segments.length > 0"
          class="transcript-display__btn"
          @click="copyFullText"
          title="Copy transcript"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
          </svg>
          Copy
        </button>
      </div>
    </div>

    <!-- Transcript Content -->
    <div ref="containerRef" class="transcript-display__content">
      <!-- Empty State -->
      <div v-if="segments.length === 0 && !interimText" class="transcript-display__empty">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" opacity="0.3">
          <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"/>
        </svg>
        <p>No transcript available</p>
        <p class="transcript-display__empty-hint">Start recording to see the transcript</p>
      </div>

      <!-- Segments -->
      <div
        v-for="segment in segments"
        :key="segment.id"
        :data-segment-id="segment.id"
        class="transcript-display__segment"
        :class="{
          'transcript-display__segment--active': activeSegmentId === segment.id,
          'transcript-display__segment--interim': !segment.isFinal
        }"
        @click="handleSegmentClick(segment)"
      >
        <!-- Timestamp -->
        <div v-if="showTimestamps" class="transcript-display__timestamp">
          {{ formatTime(segment.startTime) }}
        </div>

        <!-- Speaker -->
        <div
          v-if="showSpeakers && segment.speaker"
          class="transcript-display__speaker"
          :style="{ color: getSpeakerColor(segment.speaker) }"
        >
          {{ segment.speaker }}
        </div>

        <!-- Text -->
        <div class="transcript-display__text-wrapper">
          <!-- Edit Mode -->
          <textarea
            v-if="editingId === segment.id"
            v-model="editText"
            class="transcript-display__edit-input"
            @blur="saveEdit(segment)"
            @keydown.enter.prevent="saveEdit(segment)"
            @keydown.escape="cancelEdit"
            autofocus
          />

          <!-- Display Mode -->
          <p
            v-else
            class="transcript-display__text"
            :class="getConfidenceClass(segment.confidence)"
            @dblclick="startEdit(segment)"
          >
            {{ segment.text }}
          </p>

          <!-- Confidence -->
          <span
            v-if="showConfidence && segment.confidence"
            class="transcript-display__confidence-badge"
            :class="getConfidenceClass(segment.confidence)"
          >
            {{ Math.round(segment.confidence * 100) }}%
          </span>
        </div>
      </div>

      <!-- Interim Text (currently being transcribed) -->
      <div v-if="interimText" class="transcript-display__segment transcript-display__segment--interim">
        <div v-if="showTimestamps" class="transcript-display__timestamp">
          --:--.-
        </div>
        <p class="transcript-display__text transcript-display__text--interim">
          {{ interimText }}
          <span class="transcript-display__cursor"></span>
        </p>
      </div>
    </div>

    <!-- Footer -->
    <div v-if="segments.length > 0" class="transcript-display__footer">
      <span class="transcript-display__count">
        {{ segments.length }} segment{{ segments.length !== 1 ? 's' : '' }}
      </span>
      <span class="transcript-display__duration">
        Duration: {{ formatTime(segments[segments.length - 1]?.endTime || 0) }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.transcript-display {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: white;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  overflow: hidden;
}

/* Header */
.transcript-display__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #e5e7eb;
  background-color: #f9fafb;
}

.transcript-display__title {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
}

.transcript-display__actions {
  display: flex;
  gap: 8px;
}

.transcript-display__btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  border: none;
  border-radius: 6px;
  background-color: transparent;
  color: #6b7280;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.transcript-display__btn:hover {
  background-color: #e5e7eb;
  color: #374151;
}

/* Content */
.transcript-display__content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

/* Empty State */
.transcript-display__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #9ca3af;
  text-align: center;
}

.transcript-display__empty p {
  margin: 8px 0 0;
}

.transcript-display__empty-hint {
  font-size: 0.75rem;
}

/* Segments */
.transcript-display__segment {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 12px;
  margin-bottom: 8px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.transcript-display__segment:hover {
  background-color: #f3f4f6;
}

.transcript-display__segment--active {
  background-color: #eff6ff;
  border-left: 3px solid #3b82f6;
}

.transcript-display__segment--interim {
  opacity: 0.7;
}

/* Timestamp */
.transcript-display__timestamp {
  font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
  font-size: 0.625rem;
  color: #9ca3af;
  min-width: 48px;
  padding-top: 2px;
}

/* Speaker */
.transcript-display__speaker {
  font-size: 0.75rem;
  font-weight: 600;
  min-width: 80px;
}

/* Text */
.transcript-display__text-wrapper {
  flex: 1;
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.transcript-display__text {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
  color: #374151;
  flex: 1;
}

.transcript-display__text--interim {
  color: #6b7280;
  font-style: italic;
}

.transcript-display__edit-input {
  flex: 1;
  padding: 4px 8px;
  border: 1px solid #3b82f6;
  border-radius: 4px;
  font-size: 0.875rem;
  line-height: 1.5;
  resize: none;
  outline: none;
  min-height: 60px;
}

/* Confidence */
.transcript-display__confidence-badge {
  font-size: 0.625rem;
  padding: 2px 6px;
  border-radius: 10px;
  background-color: #f3f4f6;
}

.transcript-display__confidence--high {
  background-color: #d1fae5;
  color: #059669;
}

.transcript-display__confidence--medium {
  background-color: #fef3c7;
  color: #d97706;
}

.transcript-display__confidence--low {
  background-color: #fee2e2;
  color: #dc2626;
}

/* Cursor animation for interim text */
.transcript-display__cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background-color: #3b82f6;
  margin-left: 2px;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* Footer */
.transcript-display__footer {
  display: flex;
  justify-content: space-between;
  padding: 8px 16px;
  border-top: 1px solid #e5e7eb;
  background-color: #f9fafb;
  font-size: 0.75rem;
  color: #6b7280;
}

/* Scrollbar */
.transcript-display__content::-webkit-scrollbar {
  width: 6px;
}

.transcript-display__content::-webkit-scrollbar-track {
  background: transparent;
}

.transcript-display__content::-webkit-scrollbar-thumb {
  background-color: #d1d5db;
  border-radius: 3px;
}

.transcript-display__content::-webkit-scrollbar-thumb:hover {
  background-color: #9ca3af;
}
</style>
