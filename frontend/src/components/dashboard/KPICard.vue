<script setup lang="ts">
import { computed } from 'vue'
import TrendIndicator from '../charts/TrendIndicator.vue'

type CardVariant = 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info'

interface Props {
  title: string
  value: string | number
  trend?: number
  trendLabel?: string
  icon?: string
  iconColor?: string
  variant?: CardVariant
  loading?: boolean
  subtitle?: string
  format?: 'number' | 'percentage' | 'currency' | 'duration'
  invertTrend?: boolean
  clickable?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  loading: false,
  invertTrend: false,
  clickable: false
})

const emit = defineEmits<{
  click: []
}>()

const formattedValue = computed(() => {
  if (typeof props.value === 'string') return props.value

  switch (props.format) {
    case 'percentage':
      return `${props.value.toFixed(1)}%`
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(props.value)
    case 'duration':
      if (props.value < 60) return `${props.value}s`
      if (props.value < 3600) return `${Math.floor(props.value / 60)}m`
      return `${(props.value / 3600).toFixed(1)}h`
    case 'number':
    default:
      return new Intl.NumberFormat('en-US').format(props.value)
  }
})

const variantClasses = computed(() => ({
  'variant-default': props.variant === 'default',
  'variant-primary': props.variant === 'primary',
  'variant-success': props.variant === 'success',
  'variant-warning': props.variant === 'warning',
  'variant-danger': props.variant === 'danger',
  'variant-info': props.variant === 'info'
}))

const iconStyles = computed(() => {
  if (props.iconColor) {
    return { backgroundColor: props.iconColor }
  }
  return {}
})

const handleClick = () => {
  if (props.clickable) {
    emit('click')
  }
}

// Built-in icon mappings
const iconPaths: Record<string, string> = {
  'chart': 'M3 3v18h18M9 9v9m4-6v6m4-12v12',
  'users': 'M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8m13 10v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75',
  'clock': 'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 6v6l4 2',
  'star': 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z',
  'check': 'M9 11l3 3L22 4M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11',
  'trending': 'M23 6l-9.5 9.5-5-5L1 18M17 6h6v6',
  'book': 'M4 19.5A2.5 2.5 0 016.5 17H20M4 4.5A2.5 2.5 0 016.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15z',
  'target': 'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zm0-14a4 4 0 100 8 4 4 0 000-8zm0 2a2 2 0 110 4 2 2 0 010-4z',
  'activity': 'M22 12h-4l-3 9L9 3l-3 9H2',
  'award': 'M12 15a7 7 0 100-14 7 7 0 000 14zm0 0v9m-4-4l4-5 4 5'
}
</script>

<template>
  <div
    class="kpi-card"
    :class="[variantClasses, { 'is-loading': loading, 'is-clickable': clickable }]"
    @click="handleClick"
  >
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner" />
    </div>

    <div class="kpi-content">
      <div class="kpi-header">
        <div v-if="icon" class="kpi-icon" :style="iconStyles">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path :d="iconPaths[icon] || icon" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </div>
        <div class="kpi-info">
          <span class="kpi-title">{{ title }}</span>
          <span v-if="subtitle" class="kpi-subtitle">{{ subtitle }}</span>
        </div>
      </div>

      <div class="kpi-body">
        <span class="kpi-value">{{ formattedValue }}</span>

        <div v-if="trend !== undefined" class="kpi-trend">
          <TrendIndicator
            :value="trend"
            :invert-sentiment="invertTrend"
            size="sm"
          />
          <span v-if="trendLabel" class="trend-label">{{ trendLabel }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kpi-card {
  position: relative;
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  transition: all 0.2s ease;
  overflow: hidden;
}

.kpi-card.is-clickable {
  cursor: pointer;
}

.kpi-card.is-clickable:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.kpi-card.is-loading .kpi-content {
  opacity: 0.5;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.8);
  z-index: 1;
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.kpi-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.kpi-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.kpi-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  color: #6b7280;
  flex-shrink: 0;
}

.kpi-icon svg {
  width: 20px;
  height: 20px;
}

.kpi-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.kpi-title {
  font-size: 14px;
  font-weight: 500;
  color: #6b7280;
}

.kpi-subtitle {
  font-size: 12px;
  color: #9ca3af;
}

.kpi-body {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
}

.kpi-value {
  font-size: 28px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1.1;
}

.kpi-trend {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}

.trend-label {
  font-size: 11px;
  color: #9ca3af;
}

/* Variants */
.variant-primary .kpi-icon {
  background: #dbeafe;
  color: #2563eb;
}

.variant-success .kpi-icon {
  background: #d1fae5;
  color: #059669;
}

.variant-warning .kpi-icon {
  background: #fef3c7;
  color: #d97706;
}

.variant-danger .kpi-icon {
  background: #fee2e2;
  color: #dc2626;
}

.variant-info .kpi-icon {
  background: #e0e7ff;
  color: #4f46e5;
}
</style>
