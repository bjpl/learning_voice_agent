<script setup lang="ts">
import { computed, ref } from 'vue'
import DoughnutChart from '../charts/DoughnutChart.vue'

interface QualityDimension {
  name: string
  score: number
  maxScore?: number
  description?: string
  color?: string
}

interface Props {
  dimensions: QualityDimension[]
  title?: string
  overallScore?: number
  overallLabel?: string
  showDetails?: boolean
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Quality Breakdown',
  overallLabel: 'Overall',
  showDetails: true,
  loading: false
})

const emit = defineEmits<{
  dimensionClick: [dimension: QualityDimension]
}>()

const defaultColors = [
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#06b6d4',
  '#ec4899',
  '#84cc16'
]

const chartLabels = computed(() => props.dimensions.map(d => d.name))
const chartData = computed(() => props.dimensions.map(d => d.score))
const chartColors = computed(() =>
  props.dimensions.map((d, i) => d.color || defaultColors[i % defaultColors.length])
)

const calculatedOverall = computed(() => {
  if (props.overallScore !== undefined) return props.overallScore
  if (props.dimensions.length === 0) return 0

  const total = props.dimensions.reduce((sum, d) => sum + d.score, 0)
  const maxTotal = props.dimensions.reduce((sum, d) => sum + (d.maxScore || 100), 0)
  return Math.round((total / maxTotal) * 100)
})

const centerText = computed(() => `${calculatedOverall.value}%`)

const getScoreColor = (score: number, maxScore = 100): string => {
  const percentage = (score / maxScore) * 100
  if (percentage >= 80) return '#10b981'
  if (percentage >= 60) return '#f59e0b'
  if (percentage >= 40) return '#f97316'
  return '#ef4444'
}

const getScoreLabel = (score: number, maxScore = 100): string => {
  const percentage = (score / maxScore) * 100
  if (percentage >= 90) return 'Excellent'
  if (percentage >= 75) return 'Good'
  if (percentage >= 60) return 'Fair'
  if (percentage >= 40) return 'Needs Work'
  return 'Poor'
}

const selectedDimension = ref<QualityDimension | null>(null)

const handleSegmentClick = (index: number) => {
  const dimension = props.dimensions[index]
  if (dimension) {
    selectedDimension.value = dimension
    emit('dimensionClick', dimension)
  }
}
</script>

<template>
  <div class="quality-breakdown">
    <div class="breakdown-header">
      <h3 class="breakdown-title">{{ title }}</h3>
      <div class="overall-badge" :style="{ backgroundColor: `${getScoreColor(calculatedOverall)}20`, color: getScoreColor(calculatedOverall) }">
        {{ getScoreLabel(calculatedOverall) }}
      </div>
    </div>

    <div class="breakdown-content">
      <div v-if="loading" class="loading-state">
        <div class="loading-spinner" />
        <span>Loading quality data...</span>
      </div>

      <div v-else-if="dimensions.length === 0" class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" stroke-width="2"/>
          <path d="M12 6v6l4 2" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <span>No quality data available</span>
      </div>

      <template v-else>
        <div class="chart-section">
          <DoughnutChart
            :labels="chartLabels"
            :data="chartData"
            :colors="chartColors"
            :center-text="centerText"
            :center-subtext="overallLabel"
            :show-legend="false"
            :cutout="'70%'"
            :aspect-ratio="1"
            @segment-click="handleSegmentClick"
          />
        </div>

        <div v-if="showDetails" class="details-section">
          <div
            v-for="(dimension, index) in dimensions"
            :key="dimension.name"
            class="dimension-item"
            :class="{ selected: selectedDimension?.name === dimension.name }"
            @click="handleSegmentClick(index)"
          >
            <div class="dimension-header">
              <div class="dimension-color" :style="{ backgroundColor: chartColors[index] }" />
              <span class="dimension-name">{{ dimension.name }}</span>
              <span class="dimension-score" :style="{ color: getScoreColor(dimension.score, dimension.maxScore) }">
                {{ dimension.score }}{{ dimension.maxScore ? `/${dimension.maxScore}` : '%' }}
              </span>
            </div>

            <div class="dimension-progress">
              <div
                class="progress-fill"
                :style="{
                  width: `${(dimension.score / (dimension.maxScore || 100)) * 100}%`,
                  backgroundColor: chartColors[index]
                }"
              />
            </div>

            <p v-if="dimension.description" class="dimension-description">
              {{ dimension.description }}
            </p>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.quality-breakdown {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
}

.breakdown-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.breakdown-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.overall-badge {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}

.breakdown-content {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.chart-section {
  flex: 0 0 200px;
}

.details-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 300px;
  overflow-y: auto;
}

.dimension-item {
  padding: 12px;
  border-radius: 8px;
  background: #f9fafb;
  cursor: pointer;
  transition: all 0.2s ease;
}

.dimension-item:hover {
  background: #f3f4f6;
}

.dimension-item.selected {
  background: #eff6ff;
  box-shadow: inset 0 0 0 1px #3b82f6;
}

.dimension-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.dimension-color {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dimension-name {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: #374151;
}

.dimension-score {
  font-size: 14px;
  font-weight: 600;
}

.dimension-progress {
  height: 4px;
  background: #e5e7eb;
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s ease;
}

.dimension-description {
  margin: 8px 0 0;
  font-size: 12px;
  color: #6b7280;
  line-height: 1.4;
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #9ca3af;
  font-size: 14px;
  padding: 40px;
  width: 100%;
}

.loading-spinner {
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

.empty-state svg {
  width: 48px;
  height: 48px;
  color: #d1d5db;
}

@media (max-width: 700px) {
  .breakdown-content {
    flex-direction: column;
    align-items: center;
  }

  .chart-section {
    flex: none;
    width: 100%;
    max-width: 250px;
  }

  .details-section {
    width: 100%;
    max-height: none;
  }
}
</style>
