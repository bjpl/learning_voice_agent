<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import InsightCard from './InsightCard.vue'

type InsightPriority = 'high' | 'medium' | 'low'
type InsightType = 'improvement' | 'achievement' | 'warning' | 'tip' | 'milestone'
type FilterOption = 'all' | InsightPriority | InsightType

interface Insight {
  id: string
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

interface Props {
  insights: Insight[]
  title?: string
  showFilters?: boolean
  showSearch?: boolean
  maxVisible?: number
  loading?: boolean
  emptyMessage?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Insights',
  showFilters: true,
  showSearch: true,
  maxVisible: 10,
  loading: false,
  emptyMessage: 'No insights available'
})

const emit = defineEmits<{
  action: [insight: Insight]
  dismiss: [id: string]
  click: [insight: Insight]
  loadMore: []
}>()

const searchQuery = ref('')
const activeFilter = ref<FilterOption>('all')
const showAll = ref(false)

const priorityFilters: { value: FilterOption; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'high', label: 'High Priority' },
  { value: 'medium', label: 'Medium' },
  { value: 'low', label: 'Low' }
]

const typeFilters: { value: InsightType; label: string; icon: string }[] = [
  { value: 'improvement', label: 'Improvements', icon: 'chart' },
  { value: 'achievement', label: 'Achievements', icon: 'star' },
  { value: 'warning', label: 'Warnings', icon: 'alert' },
  { value: 'tip', label: 'Tips', icon: 'lightbulb' },
  { value: 'milestone', label: 'Milestones', icon: 'flag' }
]

const filteredInsights = computed(() => {
  let result = [...props.insights]

  // Apply search filter
  if (searchQuery.value.trim()) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(insight =>
      insight.title.toLowerCase().includes(query) ||
      insight.description.toLowerCase().includes(query)
    )
  }

  // Apply category filter
  if (activeFilter.value !== 'all') {
    result = result.filter(insight =>
      insight.priority === activeFilter.value ||
      insight.type === activeFilter.value
    )
  }

  // Sort by priority and timestamp
  result.sort((a, b) => {
    const priorityOrder: Record<InsightPriority, number> = { high: 0, medium: 1, low: 2 }
    const aPriority = priorityOrder[a.priority || 'low']
    const bPriority = priorityOrder[b.priority || 'low']

    if (aPriority !== bPriority) return aPriority - bPriority

    const aTime = a.timestamp ? new Date(a.timestamp).getTime() : 0
    const bTime = b.timestamp ? new Date(b.timestamp).getTime() : 0
    return bTime - aTime
  })

  return result
})

const visibleInsights = computed(() => {
  if (showAll.value) return filteredInsights.value
  return filteredInsights.value.slice(0, props.maxVisible)
})

const hasMore = computed(() =>
  filteredInsights.value.length > props.maxVisible && !showAll.value
)

const remainingCount = computed(() =>
  filteredInsights.value.length - props.maxVisible
)

const unreadCount = computed(() =>
  props.insights.filter(i => !i.read).length
)

const filterCounts = computed(() => {
  const counts: Record<string, number> = { all: props.insights.length }

  props.insights.forEach(insight => {
    if (insight.priority) {
      counts[insight.priority] = (counts[insight.priority] || 0) + 1
    }
    if (insight.type) {
      counts[insight.type] = (counts[insight.type] || 0) + 1
    }
  })

  return counts
})

const handleAction = (insight: Insight) => {
  emit('action', insight)
}

const handleDismiss = (id: string) => {
  emit('dismiss', id)
}

const handleClick = (insight: Insight) => {
  emit('click', insight)
}

const handleLoadMore = () => {
  showAll.value = true
  emit('loadMore')
}

const clearSearch = () => {
  searchQuery.value = ''
}

watch(activeFilter, () => {
  showAll.value = false
})
</script>

<template>
  <div class="insights-list">
    <div class="list-header">
      <div class="header-title">
        <h3>{{ title }}</h3>
        <span v-if="unreadCount > 0" class="unread-badge">
          {{ unreadCount }} new
        </span>
      </div>

      <div v-if="showSearch" class="search-box">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/>
          <path d="M21 21l-4.35-4.35"/>
        </svg>
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search insights..."
          class="search-input"
        />
        <button
          v-if="searchQuery"
          class="clear-button"
          @click="clearSearch"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    </div>

    <div v-if="showFilters" class="filters-section">
      <div class="priority-filters">
        <button
          v-for="filter in priorityFilters"
          :key="filter.value"
          class="filter-button"
          :class="{ active: activeFilter === filter.value }"
          @click="activeFilter = filter.value"
        >
          {{ filter.label }}
          <span v-if="filterCounts[filter.value]" class="filter-count">
            {{ filterCounts[filter.value] }}
          </span>
        </button>
      </div>

      <div class="type-filters">
        <button
          v-for="filter in typeFilters"
          :key="filter.value"
          class="type-button"
          :class="{ active: activeFilter === filter.value }"
          :title="filter.label"
          @click="activeFilter = activeFilter === filter.value ? 'all' : filter.value"
        >
          <svg v-if="filter.value === 'improvement'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
          </svg>
          <svg v-else-if="filter.value === 'achievement'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
          </svg>
          <svg v-else-if="filter.value === 'warning'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
          </svg>
          <svg v-else-if="filter.value === 'tip'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
          </svg>
          <svg v-else-if="filter.value === 'milestone'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2z"/>
          </svg>
          <span class="type-label">{{ filter.label }}</span>
        </button>
      </div>
    </div>

    <div class="list-content">
      <div v-if="loading" class="loading-state">
        <div class="loading-spinner" />
        <span>Loading insights...</span>
      </div>

      <div v-else-if="filteredInsights.length === 0" class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3m.08 4h.01"/>
        </svg>
        <span>{{ searchQuery ? 'No insights match your search' : emptyMessage }}</span>
        <button
          v-if="searchQuery"
          class="clear-search-button"
          @click="clearSearch"
        >
          Clear search
        </button>
      </div>

      <TransitionGroup v-else name="insight" tag="div" class="insights-container">
        <InsightCard
          v-for="insight in visibleInsights"
          :key="insight.id"
          :title="insight.title"
          :description="insight.description"
          :priority="insight.priority"
          :type="insight.type"
          :metric="insight.metric"
          :metric-value="insight.metricValue"
          :action-label="insight.actionLabel"
          :timestamp="insight.timestamp"
          :read="insight.read"
          :dismissible="insight.dismissible"
          @action="handleAction(insight)"
          @dismiss="handleDismiss(insight.id)"
          @click="handleClick(insight)"
        />
      </TransitionGroup>

      <button
        v-if="hasMore"
        class="load-more-button"
        @click="handleLoadMore"
      >
        Show {{ remainingCount }} more insights
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.insights-list {
  background: white;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  overflow: hidden;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #f3f4f6;
  flex-wrap: wrap;
  gap: 12px;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.header-title h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.unread-badge {
  padding: 2px 8px;
  background: #dbeafe;
  color: #2563eb;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
}

.search-box {
  position: relative;
  display: flex;
  align-items: center;
  min-width: 200px;
  max-width: 300px;
}

.search-icon {
  position: absolute;
  left: 10px;
  width: 16px;
  height: 16px;
  color: #9ca3af;
}

.search-input {
  width: 100%;
  padding: 8px 32px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  font-size: 13px;
  color: #1f2937;
  transition: all 0.2s ease;
}

.search-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.search-input::placeholder {
  color: #9ca3af;
}

.clear-button {
  position: absolute;
  right: 8px;
  width: 20px;
  height: 20px;
  border: none;
  background: #f3f4f6;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
}

.clear-button:hover {
  background: #e5e7eb;
}

.clear-button svg {
  width: 12px;
  height: 12px;
}

.filters-section {
  padding: 12px 20px;
  border-bottom: 1px solid #f3f4f6;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.priority-filters {
  display: flex;
  gap: 8px;
  overflow-x: auto;
}

.filter-button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border: 1px solid #e5e7eb;
  background: white;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  color: #6b7280;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s ease;
}

.filter-button:hover {
  border-color: #d1d5db;
  background: #f9fafb;
}

.filter-button.active {
  border-color: #3b82f6;
  background: #eff6ff;
  color: #2563eb;
}

.filter-count {
  padding: 1px 6px;
  background: #f3f4f6;
  border-radius: 10px;
  font-size: 10px;
}

.filter-button.active .filter-count {
  background: #dbeafe;
}

.type-filters {
  display: flex;
  gap: 4px;
}

.type-button {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  border: none;
  background: transparent;
  border-radius: 6px;
  cursor: pointer;
  color: #9ca3af;
  transition: all 0.2s ease;
}

.type-button:hover {
  background: #f3f4f6;
  color: #6b7280;
}

.type-button.active {
  background: #f3f4f6;
  color: #374151;
}

.type-button svg {
  width: 16px;
  height: 16px;
}

.type-label {
  font-size: 12px;
  font-weight: 500;
}

.list-content {
  padding: 16px;
}

.insights-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 48px 20px;
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

.clear-search-button {
  padding: 8px 16px;
  border: none;
  background: #f3f4f6;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  color: #374151;
  cursor: pointer;
  transition: background 0.2s ease;
}

.clear-search-button:hover {
  background: #e5e7eb;
}

.load-more-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: 12px;
  margin-top: 12px;
  border: 1px dashed #e5e7eb;
  background: transparent;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
}

.load-more-button:hover {
  border-color: #d1d5db;
  background: #f9fafb;
}

.load-more-button svg {
  width: 16px;
  height: 16px;
}

/* Transition animations */
.insight-enter-active,
.insight-leave-active {
  transition: all 0.3s ease;
}

.insight-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.insight-leave-to {
  opacity: 0;
  transform: translateX(20px);
}

.insight-move {
  transition: transform 0.3s ease;
}

@media (max-width: 600px) {
  .list-header {
    flex-direction: column;
    align-items: stretch;
  }

  .search-box {
    max-width: none;
  }

  .filters-section {
    flex-direction: column;
    align-items: stretch;
  }

  .priority-filters {
    justify-content: flex-start;
  }

  .type-filters {
    justify-content: flex-start;
    overflow-x: auto;
  }

  .type-label {
    display: none;
  }
}
</style>
