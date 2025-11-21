<template>
  <div class="achievement-progress">
    <div class="achievement-progress__header">
      <h3 class="achievement-progress__title">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
          <path fill-rule="evenodd" d="M8.603 3.799A4.49 4.49 0 0112 2.25c1.357 0 2.573.6 3.397 1.549a4.49 4.49 0 013.498 1.307 4.491 4.491 0 011.307 3.497A4.49 4.49 0 0121.75 12a4.49 4.49 0 01-1.549 3.397 4.491 4.491 0 01-1.307 3.497 4.491 4.491 0 01-3.497 1.307A4.49 4.49 0 0112 21.75a4.49 4.49 0 01-3.397-1.549 4.49 4.49 0 01-3.498-1.306 4.491 4.491 0 01-1.307-3.498A4.49 4.49 0 012.25 12c0-1.357.6-2.573 1.549-3.397a4.49 4.49 0 011.307-3.497 4.49 4.49 0 013.497-1.307zm7.007 6.387a.75.75 0 10-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 00-1.06 1.06l2.25 2.25a.75.75 0 001.14-.094l3.75-5.25z" clip-rule="evenodd" />
        </svg>
        Next Achievement
      </h3>
      <span class="achievement-progress__count">
        {{ unlockedCount }}/{{ totalCount }} Unlocked
      </span>
    </div>

    <!-- Next achievements list -->
    <div v-if="nextAchievements.length > 0" class="achievement-progress__list">
      <div
        v-for="achievement in nextAchievements"
        :key="achievement.id"
        class="achievement-progress__item"
        :class="`achievement-progress__item--${achievement.rarity}`"
      >
        <div class="achievement-progress__icon">
          <span class="achievement-progress__emoji">{{ achievement.icon }}</span>
          <div class="achievement-progress__ring">
            <svg viewBox="0 0 36 36">
              <circle
                class="achievement-progress__ring-bg"
                cx="18"
                cy="18"
                r="16"
                fill="none"
                stroke-width="3"
              />
              <circle
                class="achievement-progress__ring-fill"
                cx="18"
                cy="18"
                r="16"
                fill="none"
                stroke-width="3"
                :stroke-dasharray="circumference"
                :stroke-dashoffset="getOffset(achievement)"
              />
            </svg>
          </div>
        </div>

        <div class="achievement-progress__info">
          <div class="achievement-progress__top">
            <h4 class="achievement-progress__name">{{ achievement.name }}</h4>
            <RarityBadge :rarity="achievement.rarity" size="xs" />
          </div>
          <p class="achievement-progress__desc">{{ achievement.description }}</p>
          <div class="achievement-progress__bar-container">
            <div class="achievement-progress__bar">
              <div
                class="achievement-progress__bar-fill"
                :class="`bar-fill--${achievement.rarity}`"
                :style="{ width: `${getProgressPercent(achievement)}%` }"
              />
            </div>
            <span class="achievement-progress__percent">
              {{ achievement.progress }}/{{ achievement.maxProgress }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- All unlocked state -->
    <div v-else class="achievement-progress__complete">
      <div class="achievement-progress__complete-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-12 h-12">
          <path fill-rule="evenodd" d="M5.166 2.621v.858c-1.035.148-2.059.33-3.071.543a.75.75 0 00-.584.859 6.753 6.753 0 006.138 5.6 6.73 6.73 0 002.743 1.346A6.707 6.707 0 019.279 15H8.54c-1.036 0-1.875.84-1.875 1.875V19.5h-.75a2.25 2.25 0 00-2.25 2.25c0 .414.336.75.75.75h15a.75.75 0 00.75-.75 2.25 2.25 0 00-2.25-2.25h-.75v-2.625c0-1.036-.84-1.875-1.875-1.875h-.739a6.706 6.706 0 01-1.112-3.173 6.73 6.73 0 002.743-1.347 6.753 6.753 0 006.139-5.6.75.75 0 00-.585-.858 47.077 47.077 0 00-3.07-.543V2.62a.75.75 0 00-.658-.744 49.22 49.22 0 00-6.093-.377c-2.063 0-4.096.128-6.093.377a.75.75 0 00-.657.744zm0 2.629c0 1.196.312 2.32.857 3.294A5.266 5.266 0 013.16 5.337a45.6 45.6 0 012.006-.343v.256zm13.5 0v-.256c.674.1 1.343.214 2.006.343a5.265 5.265 0 01-2.863 3.207 6.72 6.72 0 00.857-3.294z" clip-rule="evenodd" />
        </svg>
      </div>
      <h4 class="achievement-progress__complete-title">All Achievements Unlocked!</h4>
      <p class="achievement-progress__complete-text">
        Congratulations! You've unlocked all available achievements.
      </p>
    </div>

    <!-- Milestone progress -->
    <div v-if="nextMilestone" class="achievement-progress__milestone">
      <div class="achievement-progress__milestone-header">
        <span class="achievement-progress__milestone-label">Next Milestone</span>
        <span class="achievement-progress__milestone-value">
          {{ nextMilestone.current }}/{{ nextMilestone.target }}
        </span>
      </div>
      <div class="achievement-progress__milestone-bar">
        <div
          class="achievement-progress__milestone-fill"
          :style="{ width: `${milestonePercent}%` }"
        />
      </div>
      <p class="achievement-progress__milestone-reward">
        Reward: {{ nextMilestone.reward }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Achievement } from './AchievementBadge.vue'
import RarityBadge from './RarityBadge.vue'

interface Milestone {
  target: number
  current: number
  reward: string
}

const props = withDefaults(defineProps<{
  achievements: Achievement[]
  maxDisplay?: number
  nextMilestone?: Milestone | null
}>(), {
  maxDisplay: 3,
  nextMilestone: null
})

const circumference = 2 * Math.PI * 16 // radius = 16

const unlockedCount = computed(() =>
  props.achievements.filter(a => a.unlockedAt).length
)

const totalCount = computed(() => props.achievements.length)

const nextAchievements = computed(() => {
  return props.achievements
    .filter(a => !a.unlockedAt && a.progress !== undefined && a.maxProgress !== undefined)
    .sort((a, b) => {
      // Sort by progress percentage (highest first)
      const aPercent = (a.progress || 0) / (a.maxProgress || 1)
      const bPercent = (b.progress || 0) / (b.maxProgress || 1)
      return bPercent - aPercent
    })
    .slice(0, props.maxDisplay)
})

const milestonePercent = computed(() => {
  if (!props.nextMilestone) return 0
  return Math.round((props.nextMilestone.current / props.nextMilestone.target) * 100)
})

function getProgressPercent(achievement: Achievement): number {
  if (!achievement.progress || !achievement.maxProgress) return 0
  return Math.round((achievement.progress / achievement.maxProgress) * 100)
}

function getOffset(achievement: Achievement): number {
  const percent = getProgressPercent(achievement)
  return circumference - (percent / 100) * circumference
}
</script>

<style scoped>
.achievement-progress {
  @apply bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md;
  @apply border border-gray-100 dark:border-gray-700;
}

.achievement-progress__header {
  @apply flex items-center justify-between mb-4;
}

.achievement-progress__title {
  @apply flex items-center gap-2 text-lg font-semibold text-gray-900 dark:text-white;
}

.achievement-progress__title svg {
  @apply text-indigo-500;
}

.achievement-progress__count {
  @apply text-sm text-gray-500 dark:text-gray-400;
}

.achievement-progress__list {
  @apply space-y-4;
}

.achievement-progress__item {
  @apply flex gap-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50;
  @apply transition-all duration-300 hover:shadow-md;
}

.achievement-progress__icon {
  @apply relative w-12 h-12 flex-shrink-0;
}

.achievement-progress__emoji {
  @apply absolute inset-0 flex items-center justify-center text-xl z-10;
}

.achievement-progress__ring {
  @apply absolute inset-0;
}

.achievement-progress__ring svg {
  @apply w-full h-full transform -rotate-90;
}

.achievement-progress__ring-bg {
  @apply stroke-gray-200 dark:stroke-gray-600;
}

.achievement-progress__ring-fill {
  @apply transition-all duration-500 ease-out;
  stroke-linecap: round;
}

/* Ring colors by rarity */
.achievement-progress__item--common .achievement-progress__ring-fill {
  @apply stroke-gray-400;
}

.achievement-progress__item--uncommon .achievement-progress__ring-fill {
  @apply stroke-green-500;
}

.achievement-progress__item--rare .achievement-progress__ring-fill {
  @apply stroke-blue-500;
}

.achievement-progress__item--epic .achievement-progress__ring-fill {
  @apply stroke-purple-500;
}

.achievement-progress__item--legendary .achievement-progress__ring-fill {
  @apply stroke-yellow-500;
}

.achievement-progress__info {
  @apply flex-1 min-w-0;
}

.achievement-progress__top {
  @apply flex items-center gap-2 mb-1;
}

.achievement-progress__name {
  @apply text-sm font-semibold text-gray-900 dark:text-white truncate;
}

.achievement-progress__desc {
  @apply text-xs text-gray-500 dark:text-gray-400 mb-2 line-clamp-1;
}

.achievement-progress__bar-container {
  @apply flex items-center gap-2;
}

.achievement-progress__bar {
  @apply flex-1 h-1.5 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden;
}

.achievement-progress__bar-fill {
  @apply h-full rounded-full transition-all duration-500;
}

.bar-fill--common { @apply bg-gray-400; }
.bar-fill--uncommon { @apply bg-green-500; }
.bar-fill--rare { @apply bg-blue-500; }
.bar-fill--epic { @apply bg-purple-500; }
.bar-fill--legendary { @apply bg-gradient-to-r from-yellow-400 to-amber-500; }

.achievement-progress__percent {
  @apply text-xs font-medium text-gray-500 dark:text-gray-400 whitespace-nowrap;
}

/* Complete state */
.achievement-progress__complete {
  @apply flex flex-col items-center py-8 text-center;
}

.achievement-progress__complete-icon {
  @apply text-yellow-500 mb-3;
  animation: trophy-bounce 1s ease-in-out infinite;
}

@keyframes trophy-bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.achievement-progress__complete-title {
  @apply text-lg font-bold text-gray-900 dark:text-white mb-1;
}

.achievement-progress__complete-text {
  @apply text-sm text-gray-500 dark:text-gray-400;
}

/* Milestone section */
.achievement-progress__milestone {
  @apply mt-5 pt-4 border-t border-gray-200 dark:border-gray-700;
}

.achievement-progress__milestone-header {
  @apply flex items-center justify-between mb-2;
}

.achievement-progress__milestone-label {
  @apply text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide;
}

.achievement-progress__milestone-value {
  @apply text-sm font-semibold text-indigo-600 dark:text-indigo-400;
}

.achievement-progress__milestone-bar {
  @apply h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-2;
}

.achievement-progress__milestone-fill {
  @apply h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full;
  @apply transition-all duration-500 ease-out;
}

.achievement-progress__milestone-reward {
  @apply text-xs text-gray-500 dark:text-gray-400;
}
</style>
