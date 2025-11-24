<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  type ChartData,
  type ChartOptions,
  type Plugin
} from 'chart.js'
import { Doughnut } from 'vue-chartjs'

ChartJS.register(ArcElement, Tooltip, Legend)

interface Props {
  labels: string[]
  data: number[]
  colors?: string[]
  title?: string
  showLegend?: boolean
  legendPosition?: 'top' | 'bottom' | 'left' | 'right'
  cutout?: string
  centerText?: string
  centerSubtext?: string
  aspectRatio?: number
  animated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  showLegend: true,
  legendPosition: 'right',
  cutout: '65%',
  centerText: '',
  centerSubtext: '',
  aspectRatio: 1,
  animated: true
})

const emit = defineEmits<{
  segmentClick: [index: number, label: string, value: number]
}>()

const chartRef = ref<InstanceType<typeof Doughnut> | null>(null)

const defaultColors = [
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#06b6d4',
  '#ec4899',
  '#84cc16',
  '#f97316',
  '#6366f1'
]

const chartData = computed<ChartData<'doughnut'>>(() => ({
  labels: props.labels,
  datasets: [{
    data: props.data,
    backgroundColor: props.colors || defaultColors.slice(0, props.data.length),
    borderColor: '#ffffff',
    borderWidth: 2,
    hoverBorderColor: '#ffffff',
    hoverBorderWidth: 3,
    hoverOffset: 8
  }]
}))

const centerTextPlugin: Plugin<'doughnut'> = {
  id: 'centerText',
  afterDraw(chart) {
    if (!props.centerText) return

    const { ctx, width, height } = chart
    ctx.save()

    const centerX = width / 2
    const centerY = height / 2

    // Main text
    ctx.font = 'bold 24px Inter, sans-serif'
    ctx.fillStyle = '#1f2937'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    if (props.centerSubtext) {
      ctx.fillText(props.centerText, centerX, centerY - 12)

      // Subtext
      ctx.font = '12px Inter, sans-serif'
      ctx.fillStyle = '#6b7280'
      ctx.fillText(props.centerSubtext, centerX, centerY + 12)
    } else {
      ctx.fillText(props.centerText, centerX, centerY)
    }

    ctx.restore()
  }
}

const chartOptions = computed<ChartOptions<'doughnut'>>(() => ({
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: props.aspectRatio,
  cutout: props.cutout,
  animation: props.animated ? {
    animateRotate: true,
    animateScale: true,
    duration: 750
  } : false,
  plugins: {
    legend: {
      display: props.showLegend,
      position: props.legendPosition,
      labels: {
        usePointStyle: true,
        pointStyle: 'circle',
        padding: 16,
        font: { size: 12, family: 'Inter, sans-serif' },
        generateLabels: (chart) => {
          const datasets = chart.data.datasets
          const labels = chart.data.labels as string[]
          return labels.map((label, i) => {
            const value = datasets[0].data[i] as number
            const total = (datasets[0].data as number[]).reduce((a, b) => a + b, 0)
            const percentage = ((value / total) * 100).toFixed(1)
            return {
              text: `${label} (${percentage}%)`,
              fillStyle: (datasets[0].backgroundColor as string[])[i],
              strokeStyle: 'transparent',
              lineWidth: 0,
              hidden: false,
              index: i
            }
          })
        }
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
          const value = context.raw as number
          const total = (context.dataset.data as number[]).reduce((a, b) => a + b, 0)
          const percentage = ((value / total) * 100).toFixed(1)
          return `${context.label}: ${value} (${percentage}%)`
        }
      }
    }
  },
  onClick: (_event, elements) => {
    if (elements.length > 0) {
      const index = elements[0].index
      emit('segmentClick', index, props.labels[index], props.data[index])
    }
  }
}))

const chartPlugins = computed(() =>
  props.centerText ? [centerTextPlugin] : []
)

const updateChart = () => {
  if (chartRef.value?.chart) {
    chartRef.value.chart.update()
  }
}

watch(() => props.data, updateChart, { deep: true })
watch(() => props.labels, updateChart)

defineExpose({ updateChart, chartRef })
</script>

<template>
  <div class="doughnut-chart-container">
    <Doughnut
      ref="chartRef"
      :data="chartData"
      :options="chartOptions"
      :plugins="chartPlugins"
    />
  </div>
</template>

<style scoped>
.doughnut-chart-container {
  position: relative;
  width: 100%;
  padding: 8px;
}
</style>
