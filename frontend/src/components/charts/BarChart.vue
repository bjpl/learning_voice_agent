<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  type ChartData,
  type ChartOptions
} from 'chart.js'
import { Bar } from 'vue-chartjs'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

interface Props {
  labels: string[]
  datasets: Array<{
    label: string
    data: number[]
    backgroundColor?: string | string[]
    borderColor?: string | string[]
    borderWidth?: number
    borderRadius?: number
  }>
  title?: string
  showLegend?: boolean
  showGrid?: boolean
  aspectRatio?: number
  horizontal?: boolean
  stacked?: boolean
  yAxisMin?: number
  yAxisMax?: number
  xAxisLabel?: string
  yAxisLabel?: string
  animated?: boolean
  barPercentage?: number
  categoryPercentage?: number
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  showLegend: true,
  showGrid: true,
  aspectRatio: 2,
  horizontal: false,
  stacked: false,
  animated: true,
  barPercentage: 0.8,
  categoryPercentage: 0.9
})

const emit = defineEmits<{
  barClick: [datasetIndex: number, index: number, value: number]
}>()

const chartRef = ref<InstanceType<typeof Bar> | null>(null)

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

const chartData = computed<ChartData<'bar'>>(() => ({
  labels: props.labels,
  datasets: props.datasets.map((dataset, index) => ({
    label: dataset.label,
    data: dataset.data,
    backgroundColor: dataset.backgroundColor || defaultColors[index % defaultColors.length],
    borderColor: dataset.borderColor || 'transparent',
    borderWidth: dataset.borderWidth ?? 0,
    borderRadius: dataset.borderRadius ?? 4,
    barPercentage: props.barPercentage,
    categoryPercentage: props.categoryPercentage
  }))
}))

const chartOptions = computed<ChartOptions<'bar'>>(() => ({
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: props.aspectRatio,
  indexAxis: props.horizontal ? 'y' : 'x',
  animation: props.animated ? { duration: 750, easing: 'easeInOutQuart' } : false,
  interaction: {
    mode: 'index',
    intersect: false
  },
  plugins: {
    legend: {
      display: props.showLegend && props.datasets.length > 1,
      position: 'top',
      labels: {
        usePointStyle: true,
        padding: 16,
        font: { size: 12, family: 'Inter, sans-serif' }
      }
    },
    title: {
      display: !!props.title,
      text: props.title,
      font: { size: 16, weight: 'bold', family: 'Inter, sans-serif' },
      padding: { bottom: 16 }
    },
    tooltip: {
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleFont: { size: 13 },
      bodyFont: { size: 12 },
      padding: 12,
      cornerRadius: 8
    }
  },
  scales: {
    x: {
      display: true,
      stacked: props.stacked,
      grid: { display: props.showGrid && !props.horizontal, color: 'rgba(0, 0, 0, 0.05)' },
      title: {
        display: !!props.xAxisLabel,
        text: props.xAxisLabel,
        font: { size: 12, weight: 'bold' }
      },
      ticks: { font: { size: 11 } }
    },
    y: {
      display: true,
      stacked: props.stacked,
      min: props.yAxisMin,
      max: props.yAxisMax,
      grid: { display: props.showGrid && props.horizontal, color: 'rgba(0, 0, 0, 0.05)' },
      title: {
        display: !!props.yAxisLabel,
        text: props.yAxisLabel,
        font: { size: 12, weight: 'bold' }
      },
      ticks: { font: { size: 11 } }
    }
  },
  onClick: (_event, elements) => {
    if (elements.length > 0) {
      const element = elements[0]
      const datasetIndex = element.datasetIndex
      const index = element.index
      const value = props.datasets[datasetIndex].data[index]
      emit('barClick', datasetIndex, index, value)
    }
  }
}))

const updateChart = () => {
  if (chartRef.value?.chart) {
    chartRef.value.chart.update()
  }
}

watch(() => props.datasets, updateChart, { deep: true })
watch(() => props.labels, updateChart)

defineExpose({ updateChart, chartRef })
</script>

<template>
  <div class="bar-chart-container">
    <Bar
      ref="chartRef"
      :data="chartData"
      :options="chartOptions"
    />
  </div>
</template>

<style scoped>
.bar-chart-container {
  position: relative;
  width: 100%;
  padding: 8px;
}
</style>
