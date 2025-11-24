<script setup lang="ts">
import { computed } from 'vue'

type TrendDirection = 'up' | 'down' | 'stable'
type TrendSentiment = 'positive' | 'negative' | 'neutral'

interface Props {
  value: number
  direction?: TrendDirection
  sentiment?: TrendSentiment
  format?: 'percentage' | 'number' | 'currency'
  prefix?: string
  suffix?: string
  size?: 'sm' | 'md' | 'lg'
  showIcon?: boolean
  showValue?: boolean
  invertSentiment?: boolean
  decimals?: number
  animated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  format: 'percentage',
  prefix: '',
  suffix: '',
  size: 'md',
  showIcon: true,
  showValue: true,
  invertSentiment: false,
  decimals: 1,
  animated: true
})

const computedDirection = computed<TrendDirection>(() => {
  if (props.direction) return props.direction
  if (props.value > 0) return 'up'
  if (props.value < 0) return 'down'
  return 'stable'
})

const computedSentiment = computed<TrendSentiment>(() => {
  if (props.sentiment) return props.sentiment

  const direction = computedDirection.value
  if (direction === 'stable') return 'neutral'

  const isPositive = direction === 'up'
  if (props.invertSentiment) {
    return isPositive ? 'negative' : 'positive'
  }
  return isPositive ? 'positive' : 'negative'
})

const formattedValue = computed(() => {
  const absValue = Math.abs(props.value)
  let formatted: string

  switch (props.format) {
    case 'percentage':
      formatted = `${absValue.toFixed(props.decimals)}%`
      break
    case 'currency':
      formatted = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: props.decimals,
        maximumFractionDigits: props.decimals
      }).format(absValue)
      break
    case 'number':
    default:
      formatted = absValue.toFixed(props.decimals)
      break
  }

  return `${props.prefix}${formatted}${props.suffix}`
})

const iconClass = computed(() => {
  switch (computedDirection.value) {
    case 'up':
      return 'icon-up'
    case 'down':
      return 'icon-down'
    default:
      return 'icon-stable'
  }
})

const sentimentClass = computed(() => {
  switch (computedSentiment.value) {
    case 'positive':
      return 'sentiment-positive'
    case 'negative':
      return 'sentiment-negative'
    default:
      return 'sentiment-neutral'
  }
})

const sizeClass = computed(() => `size-${props.size}`)
</script>

<template>
  <span
    class="trend-indicator"
    :class="[sentimentClass, sizeClass, { animated: animated }]"
  >
    <span v-if="showIcon" class="trend-icon" :class="iconClass">
      <svg v-if="computedDirection === 'up'" viewBox="0 0 24 24" fill="currentColor">
        <path d="M7 14l5-5 5 5H7z"/>
      </svg>
      <svg v-else-if="computedDirection === 'down'" viewBox="0 0 24 24" fill="currentColor">
        <path d="M7 10l5 5 5-5H7z"/>
      </svg>
      <svg v-else viewBox="0 0 24 24" fill="currentColor">
        <path d="M8 12h8v2H8z"/>
      </svg>
    </span>
    <span v-if="showValue" class="trend-value">
      {{ formattedValue }}
    </span>
  </span>
</template>

<style scoped>
.trend-indicator {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-weight: 600;
  border-radius: 4px;
  padding: 2px 6px;
  transition: all 0.2s ease;
}

.trend-indicator.animated {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Sentiment colors */
.sentiment-positive {
  color: #059669;
  background-color: #d1fae5;
}

.sentiment-negative {
  color: #dc2626;
  background-color: #fee2e2;
}

.sentiment-neutral {
  color: #6b7280;
  background-color: #f3f4f6;
}

/* Size variants */
.size-sm {
  font-size: 11px;
  padding: 1px 4px;
}

.size-sm .trend-icon svg {
  width: 12px;
  height: 12px;
}

.size-md {
  font-size: 13px;
  padding: 2px 6px;
}

.size-md .trend-icon svg {
  width: 16px;
  height: 16px;
}

.size-lg {
  font-size: 15px;
  padding: 4px 8px;
}

.size-lg .trend-icon svg {
  width: 20px;
  height: 20px;
}

/* Icon styles */
.trend-icon {
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-up svg {
  transform: translateY(-1px);
}

.icon-down svg {
  transform: translateY(1px);
}

.trend-value {
  white-space: nowrap;
}
</style>
