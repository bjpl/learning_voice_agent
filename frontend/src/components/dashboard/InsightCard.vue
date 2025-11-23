<script setup lang="ts">
import { computed } from 'vue'

type InsightPriority = 'high' | 'medium' | 'low'
type InsightType = 'improvement' | 'achievement' | 'warning' | 'tip' | 'milestone'

interface Props {
  title: string
  description: string
  priority?: InsightPriority
  type?: InsightType
  metric?: string
  metricValue?: string | number
  actionLabel?: string
  timestamp?: string | Date
  read?: boolean
  dismissible?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  priority: 'medium',
  type: 'tip',
  read: false,
  dismissible: true
})

const emit = defineEmits<{
  action: []
  dismiss: []
  click: []
}>()

const priorityConfig: Record<InsightPriority, { label: string; color: string; bg: string }> = {
  high: { label: 'High Priority', color: '#dc2626', bg: '#fee2e2' },
  medium: { label: 'Medium', color: '#d97706', bg: '#fef3c7' },
  low: { label: 'Low', color: '#6b7280', bg: '#f3f4f6' }
}

const typeConfig: Record<InsightType, { icon: string; color: string; bg: string }> = {
  improvement: {
    icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
    color: '#3b82f6',
    bg: '#dbeafe'
  },
  achievement: {
    icon: 'M12 15l8.385 4.89a.662.662 0 00.99-.654l-2.18-9.39 7.288-6.323a.662.662 0 00-.377-1.159l-9.619-.815L12.586.782a.662.662 0 00-1.172 0L7.513 7.549l-9.619.815a.662.662 0 00-.377 1.159l7.288 6.323-2.18 9.39a.662.662 0 00.99.654L12 21l8.385 4.89',
    color: '#f59e0b',
    bg: '#fef3c7'
  },
  warning: {
    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
    color: '#ef4444',
    bg: '#fee2e2'
  },
  tip: {
    icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
    color: '#8b5cf6',
    bg: '#ede9fe'
  },
  milestone: {
    icon: 'M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9',
    color: '#10b981',
    bg: '#d1fae5'
  }
}

const priorityInfo = computed(() => priorityConfig[props.priority])
const typeInfo = computed(() => typeConfig[props.type])

const formattedTimestamp = computed(() => {
  if (!props.timestamp) return null

  const date = typeof props.timestamp === 'string' ? new Date(props.timestamp) : props.timestamp
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days < 7) return `${days}d ago`

  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
})

const handleClick = () => {
  emit('click')
}

const handleAction = (event: Event) => {
  event.stopPropagation()
  emit('action')
}

const handleDismiss = (event: Event) => {
  event.stopPropagation()
  emit('dismiss')
}
</script>

<template>
  <div
    class="insight-card"
    :class="{ unread: !read }"
    @click="handleClick"
  >
    <div class="insight-icon" :style="{ backgroundColor: typeInfo.bg, color: typeInfo.color }">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path :d="typeInfo.icon" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </div>

    <div class="insight-content">
      <div class="insight-header">
        <h4 class="insight-title">{{ title }}</h4>
        <div class="insight-meta">
          <span
            class="priority-badge"
            :style="{ backgroundColor: priorityInfo.bg, color: priorityInfo.color }"
          >
            {{ priorityInfo.label }}
          </span>
          <span v-if="formattedTimestamp" class="insight-time">
            {{ formattedTimestamp }}
          </span>
        </div>
      </div>

      <p class="insight-description">{{ description }}</p>

      <div v-if="metric || actionLabel" class="insight-footer">
        <div v-if="metric" class="insight-metric">
          <span class="metric-label">{{ metric }}:</span>
          <span class="metric-value">{{ metricValue }}</span>
        </div>

        <button
          v-if="actionLabel"
          class="action-button"
          @click="handleAction"
        >
          {{ actionLabel }}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 5l7 7-7 7" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
      </div>
    </div>

    <button
      v-if="dismissible"
      class="dismiss-button"
      @click="handleDismiss"
      title="Dismiss"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>
  </div>
</template>

<style scoped>
.insight-card {
  display: flex;
  gap: 16px;
  padding: 16px;
  background: white;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
}

.insight-card:hover {
  border-color: #d1d5db;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.insight-card.unread {
  border-left: 3px solid #3b82f6;
}

.insight-icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.insight-icon svg {
  width: 22px;
  height: 22px;
}

.insight-content {
  flex: 1;
  min-width: 0;
}

.insight-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 8px;
}

.insight-title {
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
  line-height: 1.4;
}

.insight-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.priority-badge {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
}

.insight-time {
  font-size: 12px;
  color: #9ca3af;
}

.insight-description {
  font-size: 13px;
  color: #6b7280;
  line-height: 1.5;
  margin: 0 0 12px;
}

.insight-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.insight-metric {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
}

.metric-label {
  color: #9ca3af;
}

.metric-value {
  font-weight: 600;
  color: #374151;
}

.action-button {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  border: none;
  background: #f3f4f6;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-button:hover {
  background: #e5e7eb;
}

.action-button svg {
  width: 14px;
  height: 14px;
}

.dismiss-button {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 24px;
  height: 24px;
  border: none;
  background: transparent;
  border-radius: 4px;
  cursor: pointer;
  color: #9ca3af;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: all 0.2s ease;
}

.insight-card:hover .dismiss-button {
  opacity: 1;
}

.dismiss-button:hover {
  background: #f3f4f6;
  color: #6b7280;
}

.dismiss-button svg {
  width: 16px;
  height: 16px;
}

@media (max-width: 500px) {
  .insight-header {
    flex-direction: column;
    gap: 8px;
  }

  .insight-meta {
    align-self: flex-start;
  }
}
</style>
