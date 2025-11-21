<template>
  <div class="goal-list">
    <div class="goal-list__header">
      <h2 class="goal-list__title">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
          <path fill-rule="evenodd" d="M8.603 3.799A4.49 4.49 0 0112 2.25c1.357 0 2.573.6 3.397 1.549a4.49 4.49 0 013.498 1.307 4.491 4.491 0 011.307 3.497A4.49 4.49 0 0121.75 12a4.49 4.49 0 01-1.549 3.397 4.491 4.491 0 01-1.307 3.497 4.491 4.491 0 01-3.497 1.307A4.49 4.49 0 0112 21.75a4.49 4.49 0 01-3.397-1.549 4.49 4.49 0 01-3.498-1.306 4.491 4.491 0 01-1.307-3.498A4.49 4.49 0 012.25 12c0-1.357.6-2.573 1.549-3.397a4.49 4.49 0 011.307-3.497 4.49 4.49 0 013.497-1.307zm7.007 6.387a.75.75 0 10-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 00-1.06 1.06l2.25 2.25a.75.75 0 001.14-.094l3.75-5.25z" clip-rule="evenodd" />
        </svg>
        My Goals
      </h2>
      <button
        class="goal-list__add-btn"
        @click="$emit('create')"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
          <path d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z" />
        </svg>
        New Goal
      </button>
    </div>

    <GoalFilters
      v-model="activeFilter"
      :counts="filterCounts"
      class="mb-6"
    />

    <!-- Loading State -->
    <div v-if="loading" class="goal-list__loading">
      <div class="goal-list__skeleton" v-for="i in 3" :key="i">
        <div class="skeleton-icon"></div>
        <div class="skeleton-content">
          <div class="skeleton-title"></div>
          <div class="skeleton-bar"></div>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else-if="filteredGoals.length === 0" class="goal-list__empty">
      <div class="goal-list__empty-icon">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
        </svg>
      </div>
      <h3 class="goal-list__empty-title">
        {{ emptyStateTitle }}
      </h3>
      <p class="goal-list__empty-text">
        {{ emptyStateText }}
      </p>
      <button
        v-if="activeFilter === 'all'"
        class="goal-list__empty-btn"
        @click="$emit('create')"
      >
        Create Your First Goal
      </button>
    </div>

    <!-- Goals Grid -->
    <TransitionGroup
      v-else
      name="goal-list"
      tag="div"
      class="goal-list__grid"
    >
      <GoalCard
        v-for="goal in filteredGoals"
        :key="goal.id"
        :goal="goal"
        @edit="$emit('edit', $event)"
        @delete="$emit('delete', $event)"
        @increment="$emit('increment', $event)"
      />
    </TransitionGroup>

    <!-- Summary Stats -->
    <div v-if="goals.length > 0" class="goal-list__stats">
      <div class="goal-list__stat">
        <span class="goal-list__stat-value">{{ completedCount }}</span>
        <span class="goal-list__stat-label">Completed</span>
      </div>
      <div class="goal-list__stat">
        <span class="goal-list__stat-value">{{ activeCount }}</span>
        <span class="goal-list__stat-label">In Progress</span>
      </div>
      <div class="goal-list__stat">
        <span class="goal-list__stat-value">{{ averageProgress }}%</span>
        <span class="goal-list__stat-label">Avg Progress</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import GoalCard, { type Goal } from './GoalCard.vue'
import GoalFilters from './GoalFilters.vue'

const props = withDefaults(defineProps<{
  goals: Goal[]
  loading?: boolean
}>(), {
  loading: false
})

defineEmits<{
  create: []
  edit: [goal: Goal]
  delete: [goal: Goal]
  increment: [goal: Goal]
}>()

const activeFilter = ref<'all' | 'active' | 'completed'>('all')

const completedGoals = computed(() =>
  props.goals.filter(g => g.currentValue >= g.targetValue)
)

const activeGoals = computed(() =>
  props.goals.filter(g => g.currentValue < g.targetValue)
)

const filteredGoals = computed(() => {
  switch (activeFilter.value) {
    case 'active': return activeGoals.value
    case 'completed': return completedGoals.value
    default: return props.goals
  }
})

const filterCounts = computed(() => ({
  all: props.goals.length,
  active: activeGoals.value.length,
  completed: completedGoals.value.length
}))

const completedCount = computed(() => completedGoals.value.length)
const activeCount = computed(() => activeGoals.value.length)

const averageProgress = computed(() => {
  if (props.goals.length === 0) return 0
  const total = props.goals.reduce((sum, g) => {
    return sum + Math.min((g.currentValue / g.targetValue) * 100, 100)
  }, 0)
  return Math.round(total / props.goals.length)
})

const emptyStateTitle = computed(() => {
  switch (activeFilter.value) {
    case 'active': return 'No Active Goals'
    case 'completed': return 'No Completed Goals Yet'
    default: return 'Set Your Learning Goals'
  }
})

const emptyStateText = computed(() => {
  switch (activeFilter.value) {
    case 'active': return 'All your goals are completed! Time to set new challenges.'
    case 'completed': return 'Keep working on your active goals to see them here.'
    default: return 'Goals help you stay motivated and track your learning journey. Create your first goal to get started!'
  }
})
</script>

<style scoped>
.goal-list {
  @apply w-full;
}

.goal-list__header {
  @apply flex items-center justify-between mb-6;
}

.goal-list__title {
  @apply flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-white;
}

.goal-list__title svg {
  @apply text-indigo-500;
}

.goal-list__add-btn {
  @apply flex items-center gap-1.5 px-4 py-2 text-sm font-semibold;
  @apply bg-indigo-600 text-white rounded-lg;
  @apply hover:bg-indigo-700 transition-colors;
  @apply shadow-md hover:shadow-lg;
}

.goal-list__loading {
  @apply space-y-4;
}

.goal-list__skeleton {
  @apply flex gap-4 p-5 bg-gray-100 dark:bg-gray-800 rounded-xl animate-pulse;
}

.skeleton-icon {
  @apply w-10 h-10 bg-gray-300 dark:bg-gray-700 rounded-lg;
}

.skeleton-content {
  @apply flex-1 space-y-3;
}

.skeleton-title {
  @apply h-4 bg-gray-300 dark:bg-gray-700 rounded w-3/4;
}

.skeleton-bar {
  @apply h-2 bg-gray-300 dark:bg-gray-700 rounded w-full;
}

.goal-list__empty {
  @apply flex flex-col items-center justify-center py-16 px-4;
  @apply bg-gray-50 dark:bg-gray-800/50 rounded-2xl border-2 border-dashed border-gray-200 dark:border-gray-700;
}

.goal-list__empty-icon {
  @apply w-16 h-16 text-gray-300 dark:text-gray-600 mb-4;
}

.goal-list__empty-icon svg {
  @apply w-full h-full;
}

.goal-list__empty-title {
  @apply text-xl font-semibold text-gray-700 dark:text-gray-300 mb-2;
}

.goal-list__empty-text {
  @apply text-gray-500 dark:text-gray-400 text-center max-w-md mb-6;
}

.goal-list__empty-btn {
  @apply px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg;
  @apply hover:bg-indigo-700 transition-colors shadow-md;
}

.goal-list__grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
}

.goal-list__stats {
  @apply flex justify-center gap-8 mt-8 py-4 border-t border-gray-200 dark:border-gray-700;
}

.goal-list__stat {
  @apply flex flex-col items-center;
}

.goal-list__stat-value {
  @apply text-2xl font-bold text-indigo-600 dark:text-indigo-400;
}

.goal-list__stat-label {
  @apply text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide;
}

/* List transition animations */
.goal-list-enter-active {
  transition: all 0.4s ease-out;
}

.goal-list-leave-active {
  transition: all 0.3s ease-in;
}

.goal-list-enter-from {
  opacity: 0;
  transform: translateY(20px) scale(0.95);
}

.goal-list-leave-to {
  opacity: 0;
  transform: translateX(-20px) scale(0.95);
}

.goal-list-move {
  transition: transform 0.4s ease;
}
</style>
