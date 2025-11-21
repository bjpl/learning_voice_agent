<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  type ChartData,
  type ChartOptions
} from 'chart.js'
import { Radar } from 'vue-chartjs'

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
)

interface Props {
  labels: string[]
  datasets: Array<{
    label: string
    data: number[]
    backgroundColor?: string
    borderColor?: string
    borderWidth?: number
    pointBackgroundColor?: string
    pointRadius?: number
  }>
  title?: string
  showLegend?: boolean
  maxValue?: number
  stepSize?: number
  aspectRatio?: number
  animated?: boolean
  showPointLabels?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  showLegend: true,
  maxValue: 100,
  stepSize: 20,
  aspectRatio: 1,
  animated: true,
  showPointLabels: true
})

const emit = defineEmits<{
  pointClick: [datasetIndex: number, index: number, label: string, value: number]
}>()

const chartRef = ref<InstanceType<typeof Radar> | null>(null)

const defaultColors = [
  { border: '#3b82f6', background: 'rgba(59, 130, 246, 0.2)' },
  { border: '#10b981', background: 'rgba(16, 185, 129, 0.2)' },
  { border: '#f59e0b', background: 'rgba(245, 158, 11, 0.2)' },
  { border: '#ef4444', background: 'rgba(239, 68, 68, 0.2)' },
  { border: '#8b5cf6', background: 'rgba(139, 92, 246, 0.2)' }
]

const chartData = computed<ChartData<'radar'>>(() => ({
  labels: props.labels,
  datasets: props.datasets.map((dataset, index) => ({
    label: dataset.label,
    data: dataset.data,
    backgroundColor: dataset.backgroundColor || defaultColors[index % defaultColors.length].background,
    borderColor: dataset.borderColor || defaultColors[index % defaultColors.length].border,
    borderWidth: dataset.borderWidth ?? 2,
    pointBackgroundColor: dataset.pointBackgroundColor || defaultColors[index % defaultColors.length].border,
    pointBorderColor: '#ffffff',
    pointBorderWidth: 2,
    pointRadius: dataset.pointRadius ?? 4,
    pointHoverRadius: 6,
    fill: true
  }))
}))

const chartOptions = computed<ChartOptions<'radar'>>(() => ({
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: props.aspectRatio,
  animation: props.animated ? { duration: 750, easing: 'easeInOutQuart' } : false,
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
      cornerRadius: 8,
      callbacks: {
        label: (context) => {
          return `${context.dataset.label}: ${context.raw}%`
        }
      }
    }
  },
  scales: {
    r: {
      beginAtZero: true,
      max: props.maxValue,
      ticks: {
        stepSize: props.stepSize,
        font: { size: 10 },
        backdropColor: 'transparent',
        showLabelBackdrop: false
      },
      pointLabels: {
        display: props.showPointLabels,
        font: { size: 11, family: 'Inter, sans-serif' },
        color: '#374151'
      },
      grid: {
        color: 'rgba(0, 0, 0, 0.1)',
        circular: true
      },
      angleLines: {
        color: 'rgba(0, 0, 0, 0.1)'
      }
    }
  },
  onClick: (_event, elements) => {
    if (elements.length > 0) {
      const element = elements[0]
      const datasetIndex = element.datasetIndex
      const index = element.index
      const label = props.labels[index]
      const value = props.datasets[datasetIndex].data[index]
      emit('pointClick', datasetIndex, index, label, value)
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
  <div class="radar-chart-container">
    <Radar
      ref="chartRef"
      :data="chartData"
      :options="chartOptions"
    />
  </div>
</template>

<style scoped>
.radar-chart-container {
  position: relative;
  width: 100%;
  padding: 8px;
}
</style>
