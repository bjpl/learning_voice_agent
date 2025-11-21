<script setup lang="ts">
import { ref, computed } from 'vue'
import RadarChart from '../charts/RadarChart.vue'
import BarChart from '../charts/BarChart.vue'

type ChartMode = 'radar' | 'bar'

interface TopicData {
  topic: string
  mastery: number
  sessions?: number
  lastPracticed?: string
}

interface Props {
  topics: TopicData[]
  title?: string
  defaultMode?: ChartMode
  showModeToggle?: boolean
  targetMastery?: number
  loading?: boolean
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Topic Mastery',
  defaultMode: 'radar',
  showModeToggle: true,
  targetMastery: 80,
  loading: false,
  height: 350
})

const emit = defineEmits<{
  topicClick: [topic: string, mastery: number]
}>()

const chartMode = ref<ChartMode>(props.defaultMode)

const sortedTopics = computed(() => {
  return [...props.topics].sort((a, b) => b.mastery - a.mastery)
})

const topicLabels = computed(() => sortedTopics.value.map(t => t.topic))
const masteryData = computed(() => sortedTopics.value.map(t => t.mastery))

const radarDatasets = computed(() => [{
  label: 'Mastery Level',
  data: masteryData.value,
  backgroundColor: 'rgba(59, 130, 246, 0.2)',
  borderColor: '#3b82f6'
}])

const barColors = computed(() =>
  masteryData.value.map(mastery => {
    if (mastery >= props.targetMastery) return '#10b981'
    if (mastery >= props.targetMastery * 0.7) return '#f59e0b'
    return '#ef4444'
  })
)

const barDatasets = computed(() => [{
  label: 'Mastery %',
  data: masteryData.value,
  backgroundColor: barColors.value,
  borderRadius: 4
}])

const averageMastery = computed(() => {
  if (props.topics.length === 0) return 0
  const sum = props.topics.reduce((acc, t) => acc + t.mastery, 0)
  return Math.round(sum / props.topics.length)
})

const topicsAboveTarget = computed(() =>
  props.topics.filter(t => t.mastery >= props.targetMastery).length
)

const handleRadarPointClick = (_datasetIndex: number, index: number, label: string, value: number) => {
  emit('topicClick', label, value)
}

const handleBarClick = (_datasetIndex: number, index: number, value: number) => {
  const topic = sortedTopics.value[index]
  if (topic) {
    emit('topicClick', topic.topic, value)
  }
}

const getMasteryLevel = (mastery: number): string => {
  if (mastery >= 90) return 'Expert'
  if (mastery >= 75) return 'Advanced'
  if (mastery >= 50) return 'Intermediate'
  if (mastery >= 25) return 'Beginner'
  return 'Novice'
}

const getMasteryColor = (mastery: number): string => {
  if (mastery >= props.targetMastery) return '#10b981'
  if (mastery >= props.targetMastery * 0.7) return '#f59e0b'
  return '#ef4444'
}
</script>

<template>
  <div class="topic-mastery-chart">
    <div class="chart-header">
      <div class="header-left">
        <h3 class="chart-title">{{ title }}</h3>
        <div class="chart-summary">
          <span class="summary-item">
            <span class="summary-value">{{ averageMastery }}%</span>
            <span class="summary-label">Avg. Mastery</span>
          </span>
          <span class="summary-divider" />
          <span class="summary-item">
            <span class="summary-value">{{ topicsAboveTarget }}/{{ topics.length }}</span>
            <span class="summary-label">Above Target</span>
          </span>
        </div>
      </div>

      <div v-if="showModeToggle" class="mode-toggle">
        <button
          class="toggle-button"
          :class="{ active: chartMode === 'radar' }"
          @click="chartMode = 'radar'"
          title="Radar View"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <circle cx="12" cy="12" r="6"/>
            <circle cx="12" cy="12" r="2"/>
            <line x1="12" y1="2" x2="12" y2="6"/>
            <line x1="12" y1="18" x2="12" y2="22"/>
            <line x1="2" y1="12" x2="6" y2="12"/>
            <line x1="18" y1="12" x2="22" y2="12"/>
          </svg>
        </button>
        <button
          class="toggle-button"
          :class="{ active: chartMode === 'bar' }"
          @click="chartMode = 'bar'"
          title="Bar View"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="12" width="4" height="9" rx="1"/>
            <rect x="10" y="8" width="4" height="13" rx="1"/>
            <rect x="17" y="4" width="4" height="17" rx="1"/>
          </svg>
        </button>
      </div>
    </div>

    <div class="chart-body" :style="{ minHeight: `${height}px` }">
      <div v-if="loading" class="loading-state">
        <div class="loading-spinner" />
        <span>Loading topics...</span>
      </div>

      <div v-else-if="topics.length === 0" class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>No topics available yet</span>
        <span class="empty-hint">Start learning to see your progress here</span>
      </div>

      <template v-else>
        <RadarChart
          v-if="chartMode === 'radar'"
          :labels="topicLabels"
          :datasets="radarDatasets"
          :max-value="100"
          :step-size="25"
          :show-legend="false"
          :aspect-ratio="1"
          @point-click="handleRadarPointClick"
        />

        <BarChart
          v-else
          :labels="topicLabels"
          :datasets="barDatasets"
          :horizontal="true"
          :show-legend="false"
          :y-axis-max="100"
          y-axis-label="Mastery %"
          :aspect-ratio="topics.length > 6 ? 0.8 : 1.2"
          @bar-click="handleBarClick"
        />
      </template>
    </div>

    <div v-if="!loading && topics.length > 0" class="topic-legend">
      <div
        v-for="topic in sortedTopics.slice(0, 5)"
        :key="topic.topic"
        class="legend-item"
        @click="emit('topicClick', topic.topic, topic.mastery)"
      >
        <div class="legend-color" :style="{ backgroundColor: getMasteryColor(topic.mastery) }" />
        <span class="legend-topic">{{ topic.topic }}</span>
        <span class="legend-mastery">{{ topic.mastery }}%</span>
        <span class="legend-level">{{ getMasteryLevel(topic.mastery) }}</span>
      </div>
      <div v-if="topics.length > 5" class="legend-more">
        +{{ topics.length - 5 }} more topics
      </div>
    </div>
  </div>
</template>

<style scoped>
.topic-mastery-chart {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 12px;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.chart-summary {
  display: flex;
  align-items: center;
  gap: 12px;
}

.summary-item {
  display: flex;
  flex-direction: column;
}

.summary-value {
  font-size: 18px;
  font-weight: 700;
  color: #3b82f6;
}

.summary-label {
  font-size: 11px;
  color: #9ca3af;
}

.summary-divider {
  width: 1px;
  height: 32px;
  background: #e5e7eb;
}

.mode-toggle {
  display: flex;
  gap: 4px;
  background: #f3f4f6;
  padding: 4px;
  border-radius: 8px;
}

.toggle-button {
  width: 36px;
  height: 36px;
  border: none;
  background: transparent;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  transition: all 0.2s ease;
}

.toggle-button:hover {
  color: #374151;
}

.toggle-button.active {
  background: white;
  color: #3b82f6;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.toggle-button svg {
  width: 20px;
  height: 20px;
}

.chart-body {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
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
}

.empty-hint {
  font-size: 12px;
  color: #d1d5db;
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

.topic-legend {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #f3f4f6;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s ease;
}

.legend-item:hover {
  background: #f9fafb;
}

.legend-color {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.legend-topic {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: #374151;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.legend-mastery {
  font-size: 13px;
  font-weight: 600;
  color: #1f2937;
  min-width: 40px;
  text-align: right;
}

.legend-level {
  font-size: 11px;
  color: #9ca3af;
  padding: 2px 8px;
  background: #f3f4f6;
  border-radius: 4px;
}

.legend-more {
  font-size: 12px;
  color: #6b7280;
  text-align: center;
  padding: 8px;
}
</style>
