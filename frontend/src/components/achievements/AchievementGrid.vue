<template>
  <div class="achievement-grid">
    <div class="achievement-grid__header">
      <h2 class="achievement-grid__title">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
          <path fill-rule="evenodd" d="M5.166 2.621v.858c-1.035.148-2.059.33-3.071.543a.75.75 0 00-.584.859 6.753 6.753 0 006.138 5.6 6.73 6.73 0 002.743 1.346A6.707 6.707 0 019.279 15H8.54c-1.036 0-1.875.84-1.875 1.875V19.5h-.75a2.25 2.25 0 00-2.25 2.25c0 .414.336.75.75.75h15a.75.75 0 00.75-.75 2.25 2.25 0 00-2.25-2.25h-.75v-2.625c0-1.036-.84-1.875-1.875-1.875h-.739a6.706 6.706 0 01-1.112-3.173 6.73 6.73 0 002.743-1.347 6.753 6.753 0 006.139-5.6.75.75 0 00-.585-.858 47.077 47.077 0 00-3.07-.543V2.62a.75.75 0 00-.658-.744 49.22 49.22 0 00-6.093-.377c-2.063 0-4.096.128-6.093.377a.75.75 0 00-.657.744zm0 2.629c0 1.196.312 2.32.857 3.294A5.266 5.266 0 013.16 5.337a45.6 45.6 0 012.006-.343v.256zm13.5 0v-.256c.674.1 1.343.214 2.006.343a5.265 5.265 0 01-2.863 3.207 6.72 6.72 0 00.857-3.294z" clip-rule="evenodd" />
        </svg>
        Achievements
      </h2>
      <div class="achievement-grid__stats">
        <span class="achievement-grid__stat">
          <strong>{{ unlockedCount }}</strong> / {{ achievements.length }} Unlocked
        </span>
      </div>
    </div>

    <!-- Category filters -->
    <div class="achievement-grid__filters">
      <button
        v-for="category in categories"
        :key="category.value"
        class="achievement-grid__filter"
        :class="{ 'achievement-grid__filter--active': activeCategory === category.value }"
        @click="activeCategory = category.value"
      >
        {{ category.label }}
      </button>
    </div>

    <!-- Achievement progress bar -->
    <div class="achievement-grid__progress">
      <div class="achievement-grid__progress-bar">
        <div
          class="achievement-grid__progress-fill"
          :style="{ width: `${progressPercent}%` }"
        />
      </div>
      <span class="achievement-grid__progress-text">
        {{ progressPercent }}% Complete
      </span>
    </div>

    <!-- Loading state -->
    <div v-if="loading" class="achievement-grid__loading">
      <div v-for="i in 8" :key="i" class="achievement-grid__skeleton">
        <div class="skeleton-icon" />
        <div class="skeleton-text" />
      </div>
    </div>

    <!-- Empty state -->
    <div v-else-if="filteredAchievements.length === 0" class="achievement-grid__empty">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-12 h-12">
        <path stroke-linecap="round" stroke-linejoin="round" d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M7.73 9.728a6.726 6.726 0 002.748 1.35m8.272-6.842V4.5c0 2.108-.966 3.99-2.48 5.228m2.48-5.492a46.32 46.32 0 012.916.52 6.003 6.003 0 01-5.395 4.972m0 0a6.726 6.726 0 01-2.749 1.35m0 0a6.772 6.772 0 01-3.044 0" />
      </svg>
      <p>No achievements in this category yet</p>
    </div>

    <!-- Achievements grid -->
    <TransitionGroup
      v-else
      name="achievement-grid"
      tag="div"
      class="achievement-grid__items"
    >
      <AchievementBadge
        v-for="achievement in filteredAchievements"
        :key="achievement.id"
        :achievement="achievement"
        :is-new="newAchievementIds.includes(achievement.id)"
        @click="openModal"
      />
    </TransitionGroup>

    <!-- Rarity legend -->
    <div class="achievement-grid__legend">
      <span class="achievement-grid__legend-title">Rarity:</span>
      <div class="achievement-grid__legend-items">
        <span v-for="rarity in rarities" :key="rarity" class="achievement-grid__legend-item">
          <RarityBadge :rarity="rarity" size="sm" />
        </span>
      </div>
    </div>

    <!-- Detail modal -->
    <AchievementModal
      v-model="modalOpen"
      :achievement="selectedAchievement"
      @share="handleShare"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import AchievementBadge, { type Achievement, type Rarity } from './AchievementBadge.vue'
import AchievementModal from './AchievementModal.vue'
import RarityBadge from './RarityBadge.vue'

const props = withDefaults(defineProps<{
  achievements: Achievement[]
  loading?: boolean
  newAchievementIds?: string[]
}>(), {
  loading: false,
  newAchievementIds: () => []
})

const emit = defineEmits<{
  share: [achievement: Achievement]
}>()

const activeCategory = ref<string>('all')
const modalOpen = ref(false)
const selectedAchievement = ref<Achievement | null>(null)

const categories = [
  { value: 'all', label: 'All' },
  { value: 'learning', label: 'Learning' },
  { value: 'streak', label: 'Streaks' },
  { value: 'social', label: 'Social' },
  { value: 'mastery', label: 'Mastery' },
  { value: 'special', label: 'Special' }
]

const rarities: Rarity[] = ['common', 'uncommon', 'rare', 'epic', 'legendary']

const filteredAchievements = computed(() => {
  if (activeCategory.value === 'all') {
    return [...props.achievements].sort((a, b) => {
      // Unlocked first, then by rarity
      if (a.unlockedAt && !b.unlockedAt) return -1
      if (!a.unlockedAt && b.unlockedAt) return 1
      return rarities.indexOf(b.rarity) - rarities.indexOf(a.rarity)
    })
  }
  return props.achievements.filter(a => a.category === activeCategory.value)
})

const unlockedCount = computed(() =>
  props.achievements.filter(a => a.unlockedAt).length
)

const progressPercent = computed(() => {
  if (props.achievements.length === 0) return 0
  return Math.round((unlockedCount.value / props.achievements.length) * 100)
})

function openModal(achievement: Achievement) {
  selectedAchievement.value = achievement
  modalOpen.value = true
}

function handleShare(achievement: Achievement) {
  emit('share', achievement)
}
</script>

<style scoped>
.achievement-grid {
  @apply w-full;
}

.achievement-grid__header {
  @apply flex items-center justify-between mb-6;
}

.achievement-grid__title {
  @apply flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-white;
}

.achievement-grid__title svg {
  @apply text-yellow-500;
}

.achievement-grid__stats {
  @apply text-sm text-gray-600 dark:text-gray-400;
}

.achievement-grid__stat strong {
  @apply text-indigo-600 dark:text-indigo-400;
}

.achievement-grid__filters {
  @apply flex flex-wrap gap-2 mb-4;
}

.achievement-grid__filter {
  @apply px-3 py-1.5 text-sm font-medium rounded-full;
  @apply bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400;
  @apply hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors;
}

.achievement-grid__filter--active {
  @apply bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-400;
}

.achievement-grid__progress {
  @apply mb-6;
}

.achievement-grid__progress-bar {
  @apply h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-1;
}

.achievement-grid__progress-fill {
  @apply h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full;
  @apply transition-all duration-500 ease-out;
}

.achievement-grid__progress-text {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.achievement-grid__loading {
  @apply grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4;
}

.achievement-grid__skeleton {
  @apply flex flex-col items-center gap-2 p-4 animate-pulse;
}

.skeleton-icon {
  @apply w-16 h-16 bg-gray-200 dark:bg-gray-700 rounded-full;
}

.skeleton-text {
  @apply w-20 h-4 bg-gray-200 dark:bg-gray-700 rounded;
}

.achievement-grid__empty {
  @apply flex flex-col items-center justify-center py-16 text-gray-400 dark:text-gray-600;
}

.achievement-grid__empty p {
  @apply mt-4 text-sm;
}

.achievement-grid__items {
  @apply grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2;
}

.achievement-grid__legend {
  @apply flex items-center gap-3 mt-8 pt-4 border-t border-gray-200 dark:border-gray-700;
}

.achievement-grid__legend-title {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.achievement-grid__legend-items {
  @apply flex items-center gap-2;
}

/* Grid transition animations */
.achievement-grid-enter-active {
  transition: all 0.4s ease-out;
}

.achievement-grid-leave-active {
  transition: all 0.3s ease-in;
}

.achievement-grid-enter-from {
  opacity: 0;
  transform: scale(0.8);
}

.achievement-grid-leave-to {
  opacity: 0;
  transform: scale(0.8);
}

.achievement-grid-move {
  transition: transform 0.4s ease;
}
</style>
