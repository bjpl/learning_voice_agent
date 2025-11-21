<template>
  <div
    class="achievement-badge"
    :class="[
      `achievement-badge--${achievement.rarity}`,
      {
        'achievement-badge--locked': !achievement.unlockedAt,
        'achievement-badge--new': isNew
      }
    ]"
    @click="$emit('click', achievement)"
  >
    <!-- Glow effect -->
    <div class="achievement-badge__glow" />

    <!-- Badge container -->
    <div class="achievement-badge__container">
      <!-- Icon -->
      <div class="achievement-badge__icon">
        <span v-if="achievement.unlockedAt" class="achievement-badge__emoji">
          {{ achievement.icon }}
        </span>
        <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-gray-400">
          <path fill-rule="evenodd" d="M12 1.5a5.25 5.25 0 00-5.25 5.25v3a3 3 0 00-3 3v6.75a3 3 0 003 3h10.5a3 3 0 003-3v-6.75a3 3 0 00-3-3v-3c0-2.9-2.35-5.25-5.25-5.25zm3.75 8.25v-3a3.75 3.75 0 10-7.5 0v3h7.5z" clip-rule="evenodd" />
        </svg>
      </div>

      <!-- Rarity ring -->
      <svg class="achievement-badge__ring" viewBox="0 0 100 100">
        <circle
          class="achievement-badge__ring-bg"
          cx="50"
          cy="50"
          r="46"
          fill="none"
          stroke-width="4"
        />
        <circle
          v-if="achievement.unlockedAt"
          class="achievement-badge__ring-fill"
          cx="50"
          cy="50"
          r="46"
          fill="none"
          stroke-width="4"
        />
      </svg>
    </div>

    <!-- Badge info -->
    <div class="achievement-badge__info">
      <h4 class="achievement-badge__name">{{ achievement.name }}</h4>
      <RarityBadge :rarity="achievement.rarity" size="sm" />
    </div>

    <!-- Unlock particles -->
    <div v-if="isNew" class="achievement-badge__particles">
      <span v-for="i in 12" :key="i" class="particle" :style="particleStyle(i)" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import RarityBadge from './RarityBadge.vue'

export type Rarity = 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary'

export interface Achievement {
  id: string
  name: string
  description: string
  icon: string
  rarity: Rarity
  category: string
  unlockedAt?: string
  progress?: number
  maxProgress?: number
}

const props = withDefaults(defineProps<{
  achievement: Achievement
  isNew?: boolean
}>(), {
  isNew: false
})

defineEmits<{
  click: [achievement: Achievement]
}>()

function particleStyle(index: number) {
  const angle = (index / 12) * 360
  const delay = (index / 12) * 0.5
  const distance = 30 + Math.random() * 20
  return {
    '--angle': `${angle}deg`,
    '--delay': `${delay}s`,
    '--distance': `${distance}px`
  }
}
</script>

<style scoped>
.achievement-badge {
  @apply relative flex flex-col items-center gap-2 p-4 cursor-pointer;
  @apply transition-all duration-300;
}

.achievement-badge:hover {
  @apply transform -translate-y-1;
}

.achievement-badge--locked {
  @apply opacity-60;
}

.achievement-badge--locked:hover {
  @apply opacity-80;
}

/* Glow effect by rarity */
.achievement-badge__glow {
  @apply absolute inset-0 rounded-xl opacity-0 transition-opacity duration-300;
}

.achievement-badge:hover .achievement-badge__glow {
  @apply opacity-100;
}

.achievement-badge--common .achievement-badge__glow {
  background: radial-gradient(circle, rgba(156, 163, 175, 0.3) 0%, transparent 70%);
}

.achievement-badge--uncommon .achievement-badge__glow {
  background: radial-gradient(circle, rgba(34, 197, 94, 0.3) 0%, transparent 70%);
}

.achievement-badge--rare .achievement-badge__glow {
  background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, transparent 70%);
}

.achievement-badge--epic .achievement-badge__glow {
  background: radial-gradient(circle, rgba(168, 85, 247, 0.4) 0%, transparent 70%);
}

.achievement-badge--legendary .achievement-badge__glow {
  background: radial-gradient(circle, rgba(234, 179, 8, 0.4) 0%, transparent 70%);
}

/* Badge container */
.achievement-badge__container {
  @apply relative w-20 h-20;
}

.achievement-badge__icon {
  @apply absolute inset-0 flex items-center justify-center;
  @apply text-3xl z-10;
}

.achievement-badge__emoji {
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

/* Ring styles */
.achievement-badge__ring {
  @apply absolute inset-0 w-full h-full transform -rotate-90;
}

.achievement-badge__ring-bg {
  @apply stroke-gray-200 dark:stroke-gray-700;
}

.achievement-badge__ring-fill {
  stroke-dasharray: 289;
  stroke-dashoffset: 0;
  animation: ring-fill 1s ease-out;
}

@keyframes ring-fill {
  from { stroke-dashoffset: 289; }
}

/* Rarity ring colors */
.achievement-badge--common .achievement-badge__ring-fill {
  @apply stroke-gray-400;
}

.achievement-badge--uncommon .achievement-badge__ring-fill {
  @apply stroke-green-500;
}

.achievement-badge--rare .achievement-badge__ring-fill {
  @apply stroke-blue-500;
}

.achievement-badge--epic .achievement-badge__ring-fill {
  @apply stroke-purple-500;
}

.achievement-badge--legendary .achievement-badge__ring-fill {
  @apply stroke-yellow-500;
  filter: drop-shadow(0 0 6px rgba(234, 179, 8, 0.6));
}

/* Info section */
.achievement-badge__info {
  @apply flex flex-col items-center gap-1;
}

.achievement-badge__name {
  @apply text-sm font-semibold text-gray-800 dark:text-gray-200 text-center;
  @apply line-clamp-2;
}

/* New achievement animation */
.achievement-badge--new {
  animation: badge-unlock 0.6s ease-out;
}

@keyframes badge-unlock {
  0% { transform: scale(0.5); opacity: 0; }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); opacity: 1; }
}

/* Particles */
.achievement-badge__particles {
  @apply absolute inset-0 pointer-events-none overflow-visible;
}

.particle {
  @apply absolute w-2 h-2 rounded-full;
  top: 50%;
  left: 50%;
  animation: particle-burst 1s ease-out forwards;
  animation-delay: var(--delay);
}

.achievement-badge--common .particle { @apply bg-gray-400; }
.achievement-badge--uncommon .particle { @apply bg-green-400; }
.achievement-badge--rare .particle { @apply bg-blue-400; }
.achievement-badge--epic .particle { @apply bg-purple-400; }
.achievement-badge--legendary .particle { @apply bg-yellow-400; }

@keyframes particle-burst {
  0% {
    transform: translate(-50%, -50%) rotate(var(--angle)) translateY(0) scale(1);
    opacity: 1;
  }
  100% {
    transform: translate(-50%, -50%) rotate(var(--angle)) translateY(var(--distance)) scale(0);
    opacity: 0;
  }
}

/* Legendary shimmer effect */
.achievement-badge--legendary .achievement-badge__container::after {
  @apply absolute inset-0 rounded-full;
  content: '';
  background: linear-gradient(
    45deg,
    transparent 0%,
    rgba(255, 255, 255, 0.4) 50%,
    transparent 100%
  );
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%) rotate(45deg); }
  100% { transform: translateX(100%) rotate(45deg); }
}
</style>
