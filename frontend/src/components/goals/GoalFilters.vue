<template>
  <div class="goal-filters">
    <button
      v-for="filter in filters"
      :key="filter.value"
      class="goal-filters__btn"
      :class="{ 'goal-filters__btn--active': modelValue === filter.value }"
      @click="$emit('update:modelValue', filter.value)"
    >
      <span class="goal-filters__label">{{ filter.label }}</span>
      <span
        v-if="counts && counts[filter.value] !== undefined"
        class="goal-filters__count"
        :class="{ 'goal-filters__count--active': modelValue === filter.value }"
      >
        {{ counts[filter.value] }}
      </span>
    </button>

    <!-- Active indicator -->
    <div
      class="goal-filters__indicator"
      :style="indicatorStyle"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'

type FilterValue = 'all' | 'active' | 'completed'

interface FilterCounts {
  all: number
  active: number
  completed: number
}

const props = withDefaults(defineProps<{
  modelValue: FilterValue
  counts?: FilterCounts
}>(), {
  modelValue: 'all'
})

defineEmits<{
  'update:modelValue': [value: FilterValue]
}>()

const filters: { value: FilterValue; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'Active' },
  { value: 'completed', label: 'Completed' }
]

const activeIndex = computed(() =>
  filters.findIndex(f => f.value === props.modelValue)
)

const indicatorStyle = computed(() => {
  const width = 100 / filters.length
  return {
    width: `${width}%`,
    transform: `translateX(${activeIndex.value * 100}%)`
  }
})
</script>

<style scoped>
.goal-filters {
  @apply relative flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1;
  @apply overflow-hidden;
}

.goal-filters__btn {
  @apply flex-1 flex items-center justify-center gap-2 py-2 px-4;
  @apply text-sm font-medium text-gray-600 dark:text-gray-400;
  @apply rounded-md transition-colors duration-200 z-10;
}

.goal-filters__btn--active {
  @apply text-gray-900 dark:text-white;
}

.goal-filters__btn:hover:not(.goal-filters__btn--active) {
  @apply text-gray-700 dark:text-gray-300;
}

.goal-filters__label {
  @apply whitespace-nowrap;
}

.goal-filters__count {
  @apply text-xs px-2 py-0.5 rounded-full;
  @apply bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400;
  @apply transition-colors duration-200;
}

.goal-filters__count--active {
  @apply bg-indigo-100 dark:bg-indigo-900/50 text-indigo-600 dark:text-indigo-400;
}

.goal-filters__indicator {
  @apply absolute top-1 bottom-1 left-1;
  @apply bg-white dark:bg-gray-700 rounded-md shadow-sm;
  @apply transition-transform duration-300 ease-out;
  @apply pointer-events-none;
}

/* Hover animations */
.goal-filters__btn {
  @apply relative;
}

.goal-filters__btn::after {
  @apply absolute inset-0 rounded-md opacity-0;
  content: '';
  background: radial-gradient(circle at center, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
  transition: opacity 0.3s ease;
}

.goal-filters__btn:hover::after {
  @apply opacity-100;
}
</style>
