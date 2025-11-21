<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import LineChart from '../charts/LineChart.vue'

type Period = '7d' | '30d' | '90d' | '6m' | '1y' | 'all'

interface DataPoint {
  date: string
  value: number
}

interface DataSeries {
  label: string
  data: DataPoint[]
  color?: string
}

interface Props {
  series: DataSeries[]
  title?: string
  periods?: Period[]
  defaultPeriod?: Period
  yAxisLabel?: string
  showLegend?: boolean
  height?: number
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Progress Over Time',
  periods: () => ['7d', '30d', '90d', '6m', '1y'],
  defaultPeriod: '30d',
  showLegend: true,
  height: 300,
  loading: false
})

const emit = defineEmits<{
  periodChange: [period: Period]
  pointClick: [series: string, date: string, value: number]
}>()

const selectedPeriod = ref<Period>(props.defaultPeriod)

const periodLabels: Record<Period, string> = {
  '7d': '7 Days',
  '30d': '30 Days',
  '90d': '90 Days',
  '6m': '6 Months',
  '1y': '1 Year',
  'all': 'All Time'
}

const filterDataByPeriod = (data: DataPoint[], period: Period): DataPoint[] => {
  if (period === 'all') return data

  const now = new Date()
  let startDate: Date

  switch (period) {
    case '7d':
      startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
      break
    case '30d':
      startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
      break
    case '90d':
      startDate = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000)
      break
    case '6m':
      startDate = new Date(now.setMonth(now.getMonth() - 6))
      break
    case '1y':
      startDate = new Date(now.setFullYear(now.getFullYear() - 1))
      break
    default:
      return data
  }

  return data.filter(point => new Date(point.date) >= startDate)
}

const filteredSeries = computed(() => {
  return props.series.map(s => ({
    ...s,
    data: filterDataByPeriod(s.data, selectedPeriod.value)
  }))
})

const chartLabels = computed(() => {
  if (filteredSeries.value.length === 0) return []

  const allDates = new Set<string>()
  filteredSeries.value.forEach(s => {
    s.data.forEach(d => allDates.add(d.date))
  })

  return Array.from(allDates).sort()
})

const chartDatasets = computed(() => {
  return filteredSeries.value.map((s, index) => {
    const dataMap = new Map(s.data.map(d => [d.date, d.value]))
    const data = chartLabels.value.map(date => dataMap.get(date) ?? 0)

    return {
      label: s.label,
      data,
      borderColor: s.color,
      backgroundColor: s.color ? `${s.color}20` : undefined
    }
  })
})

const formatLabel = (date: string): string => {
  const d = new Date(date)

  if (selectedPeriod.value === '7d') {
    return d.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' })
  }
  if (selectedPeriod.value === '30d') {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }
  if (selectedPeriod.value === '90d' || selectedPeriod.value === '6m') {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }
  return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
}

const formattedLabels = computed(() => chartLabels.value.map(formatLabel))

const handlePeriodChange = (period: Period) => {
  selectedPeriod.value = period
  emit('periodChange', period)
}

const handlePointClick = (datasetIndex: number, index: number, value: number) => {
  const seriesLabel = props.series[datasetIndex]?.label || ''
  const date = chartLabels.value[index] || ''
  emit('pointClick', seriesLabel, date, value)
}

watch(() => props.defaultPeriod, (newPeriod) => {
  selectedPeriod.value = newPeriod
})
</script>

<template>
  <div class="progress-chart">
    <div class="chart-header">
      <h3 class="chart-title">{{ title }}</h3>

      <div class="period-selector">
        <button
          v-for="period in periods"
          :key="period"
          class="period-button"
          :class="{ active: selectedPeriod === period }"
          @click="handlePeriodChange(period)"
        >
          {{ periodLabels[period] }}
        </button>
      </div>
    </div>

    <div class="chart-body" :style="{ height: `${height}px` }">
      <div v-if="loading" class="loading-state">
        <div class="loading-spinner" />
        <span>Loading data...</span>
      </div>

      <div v-else-if="chartDatasets.length === 0 || chartLabels.length === 0" class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M3 3v18h18M9 9v9m4-6v6m4-12v12" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>No data available for this period</span>
      </div>

      <LineChart
        v-else
        :labels="formattedLabels"
        :datasets="chartDatasets"
        :show-legend="showLegend"
        :y-axis-label="yAxisLabel"
        :aspect-ratio="height / 150"
        @point-click="handlePointClick"
      />
    </div>
  </div>
</template>

<style scoped>
.progress-chart {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 12px;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.period-selector {
  display: flex;
  gap: 4px;
  background: #f3f4f6;
  padding: 4px;
  border-radius: 8px;
}

.period-button {
  padding: 6px 12px;
  border: none;
  background: transparent;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
}

.period-button:hover {
  color: #374151;
}

.period-button.active {
  background: white;
  color: #1f2937;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
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

@media (max-width: 600px) {
  .chart-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .period-selector {
    width: 100%;
    overflow-x: auto;
    justify-content: flex-start;
  }

  .period-button {
    white-space: nowrap;
    flex-shrink: 0;
  }
}
</style>
