<script setup lang="ts">
/**
 * ImageAnalysis Component
 * Display AI analysis results (labels, description, objects)
 */

import { ref, computed } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface AnalysisLabel {
  name: string;
  confidence: number;
  category?: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  id: string;
  label: string;
  confidence: number;
  boundingBox: BoundingBox;
  color?: string;
}

export interface AnalysisResult {
  id: string;
  imageUrl?: string;
  description?: string;
  labels: AnalysisLabel[];
  objects: DetectedObject[];
  colors?: string[];
  text?: string[];
  faces?: number;
  safeSearch?: {
    adult: string;
    violence: string;
    medical: string;
  };
  timestamp: number;
  processingTime?: number;
}

export interface ImageAnalysisProps {
  result: AnalysisResult | null;
  loading?: boolean;
  error?: string | null;
  showImage?: boolean;
  showLabels?: boolean;
  showObjects?: boolean;
  showDescription?: boolean;
  showColors?: boolean;
  showText?: boolean;
  confidenceThreshold?: number;
}

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<ImageAnalysisProps>(), {
  loading: false,
  error: null,
  showImage: true,
  showLabels: true,
  showObjects: true,
  showDescription: true,
  showColors: true,
  showText: true,
  confidenceThreshold: 0.5
});

const emit = defineEmits<{
  (e: 'labelClick', label: AnalysisLabel): void;
  (e: 'objectClick', object: DetectedObject): void;
  (e: 'retry'): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const activeTab = ref<'labels' | 'objects' | 'text'>('labels');
const hoveredObject = ref<string | null>(null);

// ============================================================================
// Computed
// ============================================================================

const filteredLabels = computed(() => {
  if (!props.result?.labels) return [];
  return props.result.labels.filter(l => l.confidence >= props.confidenceThreshold);
});

const filteredObjects = computed(() => {
  if (!props.result?.objects) return [];
  return props.result.objects.filter(o => o.confidence >= props.confidenceThreshold);
});

const labelsByCategory = computed(() => {
  const categories: Record<string, AnalysisLabel[]> = {};

  filteredLabels.value.forEach(label => {
    const category = label.category || 'General';
    if (!categories[category]) {
      categories[category] = [];
    }
    categories[category].push(label);
  });

  return categories;
});

// ============================================================================
// Methods
// ============================================================================

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.9) return '#10b981';
  if (confidence >= 0.7) return '#f59e0b';
  return '#ef4444';
}

function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

function formatProcessingTime(ms?: number): string {
  if (!ms) return '';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function handleLabelClick(label: AnalysisLabel): void {
  emit('labelClick', label);
}

function handleObjectClick(object: DetectedObject): void {
  emit('objectClick', object);
}

function handleObjectHover(objectId: string | null): void {
  hoveredObject.value = objectId;
}

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  activeTab,
  filteredLabels,
  filteredObjects
});
</script>

<template>
  <div class="image-analysis">
    <!-- Loading State -->
    <div v-if="loading" class="image-analysis__loading">
      <div class="image-analysis__spinner"></div>
      <span>Analyzing image...</span>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="image-analysis__error">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
      </svg>
      <p>{{ error }}</p>
      <button class="image-analysis__btn" @click="$emit('retry')">
        Try Again
      </button>
    </div>

    <!-- Empty State -->
    <div v-else-if="!result" class="image-analysis__empty">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" opacity="0.3">
        <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
      </svg>
      <p>No analysis results</p>
      <p class="image-analysis__empty-hint">Upload an image to analyze</p>
    </div>

    <!-- Results -->
    <div v-else class="image-analysis__results">
      <!-- Header -->
      <div class="image-analysis__header">
        <h3 class="image-analysis__title">Analysis Results</h3>
        <span v-if="result.processingTime" class="image-analysis__time">
          {{ formatProcessingTime(result.processingTime) }}
        </span>
      </div>

      <!-- Image with Bounding Boxes -->
      <div v-if="showImage && result.imageUrl" class="image-analysis__image-container">
        <img :src="result.imageUrl" alt="Analyzed image" class="image-analysis__image" />

        <!-- Bounding Boxes -->
        <div
          v-for="obj in filteredObjects"
          :key="obj.id"
          class="image-analysis__bounding-box"
          :class="{ 'image-analysis__bounding-box--active': hoveredObject === obj.id }"
          :style="{
            left: `${obj.boundingBox.x * 100}%`,
            top: `${obj.boundingBox.y * 100}%`,
            width: `${obj.boundingBox.width * 100}%`,
            height: `${obj.boundingBox.height * 100}%`,
            borderColor: obj.color || '#3b82f6'
          }"
          @mouseenter="handleObjectHover(obj.id)"
          @mouseleave="handleObjectHover(null)"
        >
          <span
            class="image-analysis__bounding-label"
            :style="{ backgroundColor: obj.color || '#3b82f6' }"
          >
            {{ obj.label }} ({{ formatConfidence(obj.confidence) }})
          </span>
        </div>
      </div>

      <!-- Description -->
      <div v-if="showDescription && result.description" class="image-analysis__description">
        <h4>Description</h4>
        <p>{{ result.description }}</p>
      </div>

      <!-- Tabs -->
      <div class="image-analysis__tabs">
        <button
          v-if="showLabels"
          class="image-analysis__tab"
          :class="{ 'image-analysis__tab--active': activeTab === 'labels' }"
          @click="activeTab = 'labels'"
        >
          Labels ({{ filteredLabels.length }})
        </button>
        <button
          v-if="showObjects"
          class="image-analysis__tab"
          :class="{ 'image-analysis__tab--active': activeTab === 'objects' }"
          @click="activeTab = 'objects'"
        >
          Objects ({{ filteredObjects.length }})
        </button>
        <button
          v-if="showText && result.text?.length"
          class="image-analysis__tab"
          :class="{ 'image-analysis__tab--active': activeTab === 'text' }"
          @click="activeTab = 'text'"
        >
          Text ({{ result.text.length }})
        </button>
      </div>

      <!-- Tab Content -->
      <div class="image-analysis__tab-content">
        <!-- Labels -->
        <div v-if="activeTab === 'labels' && showLabels" class="image-analysis__labels">
          <div
            v-for="(labels, category) in labelsByCategory"
            :key="category"
            class="image-analysis__category"
          >
            <h5 class="image-analysis__category-title">{{ category }}</h5>
            <div class="image-analysis__label-list">
              <button
                v-for="label in labels"
                :key="label.name"
                class="image-analysis__label"
                @click="handleLabelClick(label)"
              >
                <span class="image-analysis__label-name">{{ label.name }}</span>
                <span
                  class="image-analysis__label-confidence"
                  :style="{ color: getConfidenceColor(label.confidence) }"
                >
                  {{ formatConfidence(label.confidence) }}
                </span>
              </button>
            </div>
          </div>
        </div>

        <!-- Objects -->
        <div v-if="activeTab === 'objects' && showObjects" class="image-analysis__objects">
          <div
            v-for="obj in filteredObjects"
            :key="obj.id"
            class="image-analysis__object"
            :class="{ 'image-analysis__object--active': hoveredObject === obj.id }"
            @mouseenter="handleObjectHover(obj.id)"
            @mouseleave="handleObjectHover(null)"
            @click="handleObjectClick(obj)"
          >
            <div
              class="image-analysis__object-color"
              :style="{ backgroundColor: obj.color || '#3b82f6' }"
            />
            <span class="image-analysis__object-label">{{ obj.label }}</span>
            <span
              class="image-analysis__object-confidence"
              :style="{ color: getConfidenceColor(obj.confidence) }"
            >
              {{ formatConfidence(obj.confidence) }}
            </span>
          </div>
        </div>

        <!-- Text -->
        <div v-if="activeTab === 'text' && showText && result.text" class="image-analysis__text">
          <div
            v-for="(text, index) in result.text"
            :key="index"
            class="image-analysis__text-item"
          >
            {{ text }}
          </div>
        </div>
      </div>

      <!-- Colors -->
      <div v-if="showColors && result.colors?.length" class="image-analysis__colors">
        <h4>Dominant Colors</h4>
        <div class="image-analysis__color-list">
          <div
            v-for="color in result.colors"
            :key="color"
            class="image-analysis__color"
            :style="{ backgroundColor: color }"
            :title="color"
          />
        </div>
      </div>

      <!-- Additional Info -->
      <div class="image-analysis__footer">
        <span v-if="result.faces">{{ result.faces }} face{{ result.faces !== 1 ? 's' : '' }} detected</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-analysis {
  display: flex;
  flex-direction: column;
  background-color: white;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  overflow: hidden;
}

/* Loading */
.image-analysis__loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px;
  gap: 16px;
  color: #6b7280;
}

.image-analysis__spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Error & Empty States */
.image-analysis__error,
.image-analysis__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px;
  text-align: center;
  color: #6b7280;
}

.image-analysis__error svg {
  color: #ef4444;
}

.image-analysis__error p,
.image-analysis__empty p {
  margin: 12px 0 0;
}

.image-analysis__empty-hint {
  font-size: 0.75rem;
  color: #9ca3af;
}

.image-analysis__btn {
  margin-top: 16px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background-color: #3b82f6;
  color: white;
  font-size: 0.875rem;
  cursor: pointer;
}

.image-analysis__btn:hover {
  background-color: #2563eb;
}

/* Results */
.image-analysis__results {
  display: flex;
  flex-direction: column;
}

/* Header */
.image-analysis__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #e5e7eb;
  background-color: #f9fafb;
}

.image-analysis__title {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
}

.image-analysis__time {
  font-size: 0.75rem;
  color: #6b7280;
}

/* Image Container */
.image-analysis__image-container {
  position: relative;
  background-color: #f3f4f6;
}

.image-analysis__image {
  display: block;
  width: 100%;
  height: auto;
}

.image-analysis__bounding-box {
  position: absolute;
  border: 2px solid;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.image-analysis__bounding-box--active {
  border-width: 3px;
  z-index: 10;
}

.image-analysis__bounding-label {
  position: absolute;
  top: -20px;
  left: -2px;
  padding: 2px 6px;
  font-size: 0.625rem;
  color: white;
  border-radius: 2px;
  white-space: nowrap;
}

/* Description */
.image-analysis__description {
  padding: 16px;
  border-bottom: 1px solid #e5e7eb;
}

.image-analysis__description h4 {
  margin: 0 0 8px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
}

.image-analysis__description p {
  margin: 0;
  font-size: 0.875rem;
  color: #374151;
  line-height: 1.5;
}

/* Tabs */
.image-analysis__tabs {
  display: flex;
  border-bottom: 1px solid #e5e7eb;
}

.image-analysis__tab {
  flex: 1;
  padding: 12px;
  border: none;
  background-color: transparent;
  font-size: 0.875rem;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-analysis__tab:hover {
  background-color: #f3f4f6;
}

.image-analysis__tab--active {
  color: #3b82f6;
  border-bottom: 2px solid #3b82f6;
  margin-bottom: -1px;
}

/* Tab Content */
.image-analysis__tab-content {
  padding: 16px;
  max-height: 300px;
  overflow-y: auto;
}

/* Labels */
.image-analysis__category {
  margin-bottom: 16px;
}

.image-analysis__category:last-child {
  margin-bottom: 0;
}

.image-analysis__category-title {
  margin: 0 0 8px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
}

.image-analysis__label-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.image-analysis__label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  background-color: #f9fafb;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-analysis__label:hover {
  background-color: #eff6ff;
  border-color: #3b82f6;
}

.image-analysis__label-name {
  font-size: 0.75rem;
  color: #374151;
}

.image-analysis__label-confidence {
  font-size: 0.625rem;
  font-weight: 600;
}

/* Objects */
.image-analysis__objects {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.image-analysis__object {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.image-analysis__object:hover,
.image-analysis__object--active {
  background-color: #f3f4f6;
}

.image-analysis__object-color {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.image-analysis__object-label {
  flex: 1;
  font-size: 0.875rem;
  color: #374151;
}

.image-analysis__object-confidence {
  font-size: 0.75rem;
  font-weight: 600;
}

/* Text */
.image-analysis__text {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.image-analysis__text-item {
  padding: 8px 12px;
  background-color: #f9fafb;
  border-radius: 6px;
  font-size: 0.875rem;
  color: #374151;
}

/* Colors */
.image-analysis__colors {
  padding: 16px;
  border-top: 1px solid #e5e7eb;
}

.image-analysis__colors h4 {
  margin: 0 0 12px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
}

.image-analysis__color-list {
  display: flex;
  gap: 8px;
}

.image-analysis__color {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  cursor: pointer;
}

/* Footer */
.image-analysis__footer {
  padding: 8px 16px;
  border-top: 1px solid #e5e7eb;
  background-color: #f9fafb;
  font-size: 0.75rem;
  color: #6b7280;
}
</style>
