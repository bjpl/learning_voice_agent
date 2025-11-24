<template>
  <span
    class="rarity-badge"
    :class="[
      `rarity-badge--${rarity}`,
      `rarity-badge--${size}`
    ]"
  >
    <span class="rarity-badge__dot" />
    <span class="rarity-badge__label">{{ rarityLabel }}</span>
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Rarity } from './AchievementBadge.vue'

const props = withDefaults(defineProps<{
  rarity: Rarity
  size?: 'xs' | 'sm' | 'md'
}>(), {
  size: 'sm'
})

const rarityLabels: Record<Rarity, string> = {
  common: 'Common',
  uncommon: 'Uncommon',
  rare: 'Rare',
  epic: 'Epic',
  legendary: 'Legendary'
}

const rarityLabel = computed(() => rarityLabels[props.rarity])
</script>

<style scoped>
.rarity-badge {
  @apply inline-flex items-center gap-1 font-medium rounded-full;
}

/* Sizes */
.rarity-badge--xs {
  @apply text-[10px] px-1.5 py-0.5;
}

.rarity-badge--xs .rarity-badge__dot {
  @apply w-1.5 h-1.5;
}

.rarity-badge--sm {
  @apply text-xs px-2 py-0.5;
}

.rarity-badge--sm .rarity-badge__dot {
  @apply w-2 h-2;
}

.rarity-badge--md {
  @apply text-sm px-2.5 py-1;
}

.rarity-badge--md .rarity-badge__dot {
  @apply w-2.5 h-2.5;
}

.rarity-badge__dot {
  @apply rounded-full;
}

/* Common - Gray */
.rarity-badge--common {
  @apply bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300;
}

.rarity-badge--common .rarity-badge__dot {
  @apply bg-gray-400;
}

/* Uncommon - Green */
.rarity-badge--uncommon {
  @apply bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400;
}

.rarity-badge--uncommon .rarity-badge__dot {
  @apply bg-green-500;
}

/* Rare - Blue */
.rarity-badge--rare {
  @apply bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400;
}

.rarity-badge--rare .rarity-badge__dot {
  @apply bg-blue-500;
  animation: pulse-blue 2s ease-in-out infinite;
}

@keyframes pulse-blue {
  0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
  50% { box-shadow: 0 0 0 4px rgba(59, 130, 246, 0); }
}

/* Epic - Purple */
.rarity-badge--epic {
  @apply bg-purple-50 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400;
}

.rarity-badge--epic .rarity-badge__dot {
  @apply bg-purple-500;
  animation: pulse-purple 2s ease-in-out infinite;
}

@keyframes pulse-purple {
  0%, 100% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.4); }
  50% { box-shadow: 0 0 0 4px rgba(168, 85, 247, 0); }
}

/* Legendary - Gold */
.rarity-badge--legendary {
  @apply bg-gradient-to-r from-yellow-50 to-amber-50 text-amber-700;
  @apply dark:from-yellow-900/30 dark:to-amber-900/30 dark:text-yellow-400;
  animation: legendary-glow 3s ease-in-out infinite;
}

@keyframes legendary-glow {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.2);
  }
  50% {
    box-shadow: 0 0 8px 2px rgba(251, 191, 36, 0.3);
  }
}

.rarity-badge--legendary .rarity-badge__dot {
  @apply bg-gradient-to-r from-yellow-400 to-amber-500;
  animation: legendary-pulse 1.5s ease-in-out infinite;
}

@keyframes legendary-pulse {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.6);
  }
  50% {
    transform: scale(1.2);
    box-shadow: 0 0 6px 2px rgba(251, 191, 36, 0.4);
  }
}

.rarity-badge--legendary .rarity-badge__label {
  background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 50%, #f59e0b 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer-text 3s linear infinite;
}

@keyframes shimmer-text {
  to { background-position: 200% center; }
}
</style>
