<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  type ChartData,
  type ChartOptions
} from 'chart.js'
import { Line } from 'vue-chartjs'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface Props {
  labels: string[]
  datasets: Array<{
    label: string
    data: number[]
    borderColor?: string
    backgroundColor?: string
    fill?: boolean
    tension?: number
    pointRadius?: number
    borderWidth?: number
  }>
  title?: string
  showLegend?: boolean
  showGrid?: boolean
  aspectRatio?: number
  yAxisMin?: number
  yAxisMax?: number
  xAxisLabel?: string
  yAxisLabel?: string
  animated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  showLegend: true,
  showGrid: true,
  aspectRatio: 2,
  animated: true
})

const emit = defineEmits<{
  pointClick: [datasetIndex: number, index: number, value: number]
}>()

const chartRef = ref<InstanceType<typeof Line> | null>(null)

const defaultColors = [
  { border: '#3b82f6', background: 'rgba(59, 130, 246, 0.1)' },
  { border: '#10b981', background: 'rgba(16, 185, 129, 0.1)' },
  { border: '#f59e0b', background: 'rgba(245, 158, 11, 0.1)' },
  { border: '#ef4444', background: 'rgba(239, 68, 68, 0.1)' },
  { border: '#8b5cf6', background: 'rgba(139, 92, 246, 0.1)' }
]

const chartData = computed<ChartData<'line'>>(() => ({
  labels: props.labels,
  datasets: props.datasets.map((dataset, index) => ({
    label: dataset.label,
    data: dataset.data,
    borderColor: dataset.borderColor || defaultColors[index % defaultColors.length].border,
    backgroundColor: dataset.backgroundColor || defaultColors[index % defaultColors.length].background,
    fill: dataset.fill ?? true,
    tension: dataset.tension ?? 0.4,
    pointRadius: dataset.pointRadius ?? 4,
    pointHoverRadius: 6,
    borderWidth: dataset.borderWidth ?? 2
  }))
}))

const chartOptions = computed<ChartOptions<'line'>>(() => ({
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: props.aspectRatio,
  animation: props.animated ? { duration: 750, easing: 'easeInOutQuart' } : false,
  interaction: {
    mode: 'index',
    intersect: false
  },
  plugins: {
    legend: {
      display: props.showLegend,
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
      cornerRadius: 8,
      displayColors: true
    }
  },
  scales: {
    x: {
      display: true,
      grid: { display: props.showGrid, color: 'rgba(0, 0, 0, 0.05)' },
      title: {
        display: !!props.xAxisLabel,
        text: props.xAxisLabel,
        font: { size: 12, weight: 'bold' }
      },
      ticks: { font: { size: 11 } }
    },
    y: {
      display: true,
      min: props.yAxisMin,
      max: props.yAxisMax,
      grid: { display: props.showGrid, color: 'rgba(0, 0, 0, 0.05)' },
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
      emit('pointClick', datasetIndex, index, value)
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
  <div class="line-chart-container">
    <Line
      ref="chartRef"
      :data="chartData"
      :options="chartOptions"
    />
  </div>
</template>

<style scoped>
.line-chart-container {
  position: relative;
  width: 100%;
  padding: 8px;
}
</style>
