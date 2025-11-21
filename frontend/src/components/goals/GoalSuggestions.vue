<template>
  <div class="goal-suggestions">
    <div class="goal-suggestions__header">
      <div class="goal-suggestions__title-group">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5 text-purple-500">
          <path fill-rule="evenodd" d="M9 4.5a.75.75 0 01.721.544l.813 2.846a3.75 3.75 0 002.576 2.576l2.846.813a.75.75 0 010 1.442l-2.846.813a3.75 3.75 0 00-2.576 2.576l-.813 2.846a.75.75 0 01-1.442 0l-.813-2.846a3.75 3.75 0 00-2.576-2.576l-2.846-.813a.75.75 0 010-1.442l2.846-.813A3.75 3.75 0 007.466 7.89l.813-2.846A.75.75 0 019 4.5zM18 1.5a.75.75 0 01.728.568l.258 1.036c.236.94.97 1.674 1.91 1.91l1.036.258a.75.75 0 010 1.456l-1.036.258c-.94.236-1.674.97-1.91 1.91l-.258 1.036a.75.75 0 01-1.456 0l-.258-1.036a2.625 2.625 0 00-1.91-1.91l-1.036-.258a.75.75 0 010-1.456l1.036-.258a2.625 2.625 0 001.91-1.91l.258-1.036A.75.75 0 0118 1.5z" clip-rule="evenodd" />
        </svg>
        <h3 class="goal-suggestions__title">AI Suggested Goals</h3>
      </div>
      <button
        class="goal-suggestions__refresh"
        @click="refresh"
        :disabled="loading"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          class="w-4 h-4"
          :class="{ 'animate-spin': loading }"
        >
          <path fill-rule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z" clip-rule="evenodd" />
        </svg>
        Refresh
      </button>
    </div>

    <div class="goal-suggestions__carousel" ref="carouselRef">
      <div
        class="goal-suggestions__track"
        :style="{ transform: `translateX(-${currentIndex * 100}%)` }"
      >
        <div
          v-for="(suggestion, index) in suggestions"
          :key="index"
          class="goal-suggestions__slide"
        >
          <div class="suggestion-card">
            <div class="suggestion-card__icon" :class="`icon--${suggestion.type}`">
              {{ getTypeEmoji(suggestion.type) }}
            </div>
            <div class="suggestion-card__content">
              <h4 class="suggestion-card__title">{{ suggestion.title }}</h4>
              <p class="suggestion-card__desc">{{ suggestion.description }}</p>
              <div class="suggestion-card__meta">
                <span class="suggestion-card__target">
                  Target: {{ suggestion.targetValue }} {{ suggestion.unit }}
                </span>
                <span v-if="suggestion.timeframe" class="suggestion-card__timeframe">
                  {{ suggestion.timeframe }}
                </span>
              </div>
            </div>
            <button
              class="suggestion-card__add"
              @click="$emit('select', suggestion)"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
                <path d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="goal-suggestions__nav">
      <button
        class="goal-suggestions__nav-btn"
        :disabled="currentIndex === 0"
        @click="prev"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
          <path fill-rule="evenodd" d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z" clip-rule="evenodd" />
        </svg>
      </button>
      <div class="goal-suggestions__dots">
        <button
          v-for="(_, index) in suggestions"
          :key="index"
          class="goal-suggestions__dot"
          :class="{ 'goal-suggestions__dot--active': index === currentIndex }"
          @click="currentIndex = index"
        />
      </div>
      <button
        class="goal-suggestions__nav-btn"
        :disabled="currentIndex === suggestions.length - 1"
        @click="next"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
          <path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clip-rule="evenodd" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { Goal } from './GoalCard.vue'

export interface GoalSuggestion {
  type: Goal['type']
  title: string
  description: string
  targetValue: number
  unit: string
  timeframe?: string
}

const props = withDefaults(defineProps<{
  suggestions?: GoalSuggestion[]
  loading?: boolean
}>(), {
  suggestions: () => [
    {
      type: 'streak',
      title: '7-Day Practice Streak',
      description: 'Build consistency by practicing every day for a week',
      targetValue: 7,
      unit: 'days',
      timeframe: '1 week'
    },
    {
      type: 'sessions',
      title: 'Complete 10 Sessions',
      description: 'Deepen your learning with dedicated practice sessions',
      targetValue: 10,
      unit: 'sessions',
      timeframe: '2 weeks'
    },
    {
      type: 'duration',
      title: '5 Hours of Practice',
      description: 'Accumulate focused practice time to improve faster',
      targetValue: 300,
      unit: 'minutes',
      timeframe: '1 month'
    },
    {
      type: 'topics',
      title: 'Master 5 Topics',
      description: 'Expand your knowledge across different subjects',
      targetValue: 5,
      unit: 'topics',
      timeframe: '3 weeks'
    },
    {
      type: 'quality',
      title: 'Achieve 90% Average Score',
      description: 'Focus on quality responses and accurate learning',
      targetValue: 90,
      unit: 'points',
      timeframe: 'Ongoing'
    }
  ],
  loading: false
})

defineEmits<{
  select: [suggestion: GoalSuggestion]
  refresh: []
}>()

const currentIndex = ref(0)
const carouselRef = ref<HTMLElement | null>(null)
let autoplayInterval: number | null = null

const typeEmojis: Record<Goal['type'], string> = {
  streak: '🔥',
  sessions: '📚',
  topics: '📖',
  quality: '⭐',
  exchanges: '💬',
  duration: '⏱️',
  feedback: '👍',
  custom: '⚙️'
}

function getTypeEmoji(type: Goal['type']) {
  return typeEmojis[type]
}

function prev() {
  if (currentIndex.value > 0) {
    currentIndex.value--
  }
}

function next() {
  if (currentIndex.value < props.suggestions.length - 1) {
    currentIndex.value++
  }
}

function refresh() {
  // Emit refresh event for parent to handle
}

function startAutoplay() {
  autoplayInterval = window.setInterval(() => {
    if (currentIndex.value < props.suggestions.length - 1) {
      currentIndex.value++
    } else {
      currentIndex.value = 0
    }
  }, 5000)
}

function stopAutoplay() {
  if (autoplayInterval) {
    clearInterval(autoplayInterval)
    autoplayInterval = null
  }
}

onMounted(() => {
  startAutoplay()
})

onUnmounted(() => {
  stopAutoplay()
})
</script>

<style scoped>
.goal-suggestions {
  @apply bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20;
  @apply rounded-xl p-4 border border-purple-100 dark:border-purple-800;
}

.goal-suggestions__header {
  @apply flex items-center justify-between mb-4;
}

.goal-suggestions__title-group {
  @apply flex items-center gap-2;
}

.goal-suggestions__title {
  @apply text-sm font-semibold text-gray-700 dark:text-gray-200;
}

.goal-suggestions__refresh {
  @apply flex items-center gap-1.5 px-2 py-1 text-xs font-medium;
  @apply text-purple-600 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-800/30;
  @apply rounded-lg transition-colors disabled:opacity-50;
}

.goal-suggestions__carousel {
  @apply overflow-hidden rounded-lg;
}

.goal-suggestions__track {
  @apply flex transition-transform duration-500 ease-out;
}

.goal-suggestions__slide {
  @apply w-full flex-shrink-0 px-1;
}

.suggestion-card {
  @apply flex items-start gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm;
}

.suggestion-card__icon {
  @apply w-10 h-10 rounded-lg flex items-center justify-center text-xl flex-shrink-0;
}

.icon--streak { @apply bg-orange-100 dark:bg-orange-900/30; }
.icon--sessions { @apply bg-blue-100 dark:bg-blue-900/30; }
.icon--topics { @apply bg-emerald-100 dark:bg-emerald-900/30; }
.icon--quality { @apply bg-yellow-100 dark:bg-yellow-900/30; }
.icon--exchanges { @apply bg-purple-100 dark:bg-purple-900/30; }
.icon--duration { @apply bg-cyan-100 dark:bg-cyan-900/30; }
.icon--feedback { @apply bg-pink-100 dark:bg-pink-900/30; }
.icon--custom { @apply bg-gray-100 dark:bg-gray-700; }

.suggestion-card__content {
  @apply flex-1 min-w-0;
}

.suggestion-card__title {
  @apply text-sm font-semibold text-gray-900 dark:text-white mb-1;
}

.suggestion-card__desc {
  @apply text-xs text-gray-600 dark:text-gray-400 mb-2 line-clamp-2;
}

.suggestion-card__meta {
  @apply flex items-center gap-2 text-xs;
}

.suggestion-card__target {
  @apply text-indigo-600 dark:text-indigo-400 font-medium;
}

.suggestion-card__timeframe {
  @apply text-gray-400 dark:text-gray-500;
}

.suggestion-card__add {
  @apply p-2 rounded-lg bg-indigo-100 text-indigo-600 hover:bg-indigo-200;
  @apply dark:bg-indigo-900/30 dark:text-indigo-400 dark:hover:bg-indigo-900/50;
  @apply transition-colors flex-shrink-0;
}

.goal-suggestions__nav {
  @apply flex items-center justify-center gap-4 mt-4;
}

.goal-suggestions__nav-btn {
  @apply p-1.5 rounded-full text-gray-400 hover:text-gray-600 hover:bg-white;
  @apply dark:hover:text-gray-300 dark:hover:bg-gray-700;
  @apply transition-colors disabled:opacity-30 disabled:cursor-not-allowed;
}

.goal-suggestions__dots {
  @apply flex items-center gap-1.5;
}

.goal-suggestions__dot {
  @apply w-2 h-2 rounded-full bg-gray-300 dark:bg-gray-600 transition-all;
}

.goal-suggestions__dot--active {
  @apply w-4 bg-indigo-500;
}
</style>
