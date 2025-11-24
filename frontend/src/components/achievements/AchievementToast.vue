<template>
  <Teleport to="body">
    <TransitionGroup
      name="toast"
      tag="div"
      class="achievement-toast-container"
    >
      <div
        v-for="toast in toasts"
        :key="toast.id"
        class="achievement-toast"
        :class="`achievement-toast--${toast.achievement.rarity}`"
        @click="dismiss(toast.id)"
      >
        <!-- Glow effect -->
        <div class="achievement-toast__glow" />

        <!-- Content -->
        <div class="achievement-toast__content">
          <div class="achievement-toast__header">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
              <path fill-rule="evenodd" d="M5.166 2.621v.858c-1.035.148-2.059.33-3.071.543a.75.75 0 00-.584.859 6.753 6.753 0 006.138 5.6 6.73 6.73 0 002.743 1.346A6.707 6.707 0 019.279 15H8.54c-1.036 0-1.875.84-1.875 1.875V19.5h-.75a2.25 2.25 0 00-2.25 2.25c0 .414.336.75.75.75h15a.75.75 0 00.75-.75 2.25 2.25 0 00-2.25-2.25h-.75v-2.625c0-1.036-.84-1.875-1.875-1.875h-.739a6.706 6.706 0 01-1.112-3.173 6.73 6.73 0 002.743-1.347 6.753 6.753 0 006.139-5.6.75.75 0 00-.585-.858 47.077 47.077 0 00-3.07-.543V2.62a.75.75 0 00-.658-.744 49.22 49.22 0 00-6.093-.377c-2.063 0-4.096.128-6.093.377a.75.75 0 00-.657.744zm0 2.629c0 1.196.312 2.32.857 3.294A5.266 5.266 0 013.16 5.337a45.6 45.6 0 012.006-.343v.256zm13.5 0v-.256c.674.1 1.343.214 2.006.343a5.265 5.265 0 01-2.863 3.207 6.72 6.72 0 00.857-3.294z" clip-rule="evenodd" />
            </svg>
            <span class="achievement-toast__label">Achievement Unlocked!</span>
          </div>

          <div class="achievement-toast__body">
            <div class="achievement-toast__icon">
              {{ toast.achievement.icon }}
            </div>
            <div class="achievement-toast__info">
              <h4 class="achievement-toast__name">{{ toast.achievement.name }}</h4>
              <RarityBadge :rarity="toast.achievement.rarity" size="xs" />
            </div>
          </div>
        </div>

        <!-- Progress bar for auto-dismiss -->
        <div class="achievement-toast__timer">
          <div
            class="achievement-toast__timer-fill"
            :style="{ animationDuration: `${duration}ms` }"
          />
        </div>

        <!-- Confetti particles -->
        <div class="achievement-toast__confetti">
          <span
            v-for="i in 20"
            :key="i"
            class="confetti"
            :style="confettiStyle(i, toast.achievement.rarity)"
          />
        </div>
      </div>
    </TransitionGroup>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import type { Achievement } from './AchievementBadge.vue'
import RarityBadge from './RarityBadge.vue'

interface ToastItem {
  id: string
  achievement: Achievement
}

const props = withDefaults(defineProps<{
  duration?: number
}>(), {
  duration: 5000
})

const emit = defineEmits<{
  dismiss: [achievementId: string]
}>()

const toasts = ref<ToastItem[]>([])
const timers = new Map<string, number>()

const rarityColors = {
  common: ['#9ca3af', '#6b7280', '#4b5563'],
  uncommon: ['#22c55e', '#16a34a', '#15803d'],
  rare: ['#3b82f6', '#2563eb', '#1d4ed8'],
  epic: ['#a855f7', '#9333ea', '#7c3aed'],
  legendary: ['#fbbf24', '#f59e0b', '#d97706']
}

function show(achievement: Achievement) {
  const id = `${achievement.id}-${Date.now()}`
  toasts.value.push({ id, achievement })

  // Auto dismiss after duration
  const timer = window.setTimeout(() => {
    dismiss(id)
  }, props.duration)

  timers.set(id, timer)
}

function dismiss(id: string) {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index > -1) {
    const toast = toasts.value[index]
    toasts.value.splice(index, 1)
    emit('dismiss', toast.achievement.id)

    const timer = timers.get(id)
    if (timer) {
      clearTimeout(timer)
      timers.delete(id)
    }
  }
}

function confettiStyle(index: number, rarity: Achievement['rarity']) {
  const colors = rarityColors[rarity]
  const color = colors[index % colors.length]
  const left = Math.random() * 100
  const delay = Math.random() * 0.5
  const duration = 1 + Math.random() * 0.5
  const size = 4 + Math.random() * 4

  return {
    '--confetti-color': color,
    '--confetti-left': `${left}%`,
    '--confetti-delay': `${delay}s`,
    '--confetti-duration': `${duration}s`,
    '--confetti-size': `${size}px`
  }
}

onUnmounted(() => {
  timers.forEach(timer => clearTimeout(timer))
  timers.clear()
})

// Expose show method for parent components
defineExpose({ show })
</script>

<style scoped>
.achievement-toast-container {
  @apply fixed top-4 right-4 z-50 space-y-3;
  @apply pointer-events-none;
}

.achievement-toast {
  @apply relative w-80 rounded-xl shadow-2xl overflow-hidden cursor-pointer;
  @apply pointer-events-auto;
  animation: toast-entrance 0.6s ease-out;
}

@keyframes toast-entrance {
  0% {
    transform: translateX(100%) scale(0.8);
    opacity: 0;
  }
  50% {
    transform: translateX(-10px) scale(1.02);
  }
  100% {
    transform: translateX(0) scale(1);
    opacity: 1;
  }
}

/* Rarity backgrounds */
.achievement-toast--common {
  @apply bg-gradient-to-r from-gray-600 to-gray-500;
}

.achievement-toast--uncommon {
  @apply bg-gradient-to-r from-green-600 to-green-500;
}

.achievement-toast--rare {
  @apply bg-gradient-to-r from-blue-600 to-blue-500;
}

.achievement-toast--epic {
  @apply bg-gradient-to-r from-purple-600 to-purple-500;
}

.achievement-toast--legendary {
  @apply bg-gradient-to-r from-yellow-500 to-amber-500;
  animation: toast-entrance 0.6s ease-out, legendary-pulse 2s ease-in-out infinite;
}

@keyframes legendary-pulse {
  0%, 100% { box-shadow: 0 0 20px rgba(251, 191, 36, 0.4); }
  50% { box-shadow: 0 0 40px rgba(251, 191, 36, 0.6); }
}

/* Glow effect */
.achievement-toast__glow {
  @apply absolute inset-0 opacity-50;
  background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.3) 0%, transparent 60%);
}

.achievement-toast__content {
  @apply relative p-4 text-white z-10;
}

.achievement-toast__header {
  @apply flex items-center gap-2 mb-3 text-sm font-medium opacity-90;
}

.achievement-toast__body {
  @apply flex items-center gap-3;
}

.achievement-toast__icon {
  @apply w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm;
  @apply flex items-center justify-center text-2xl;
  animation: icon-pop 0.4s ease-out 0.3s both;
}

@keyframes icon-pop {
  0% { transform: scale(0) rotate(-20deg); }
  60% { transform: scale(1.2) rotate(5deg); }
  100% { transform: scale(1) rotate(0); }
}

.achievement-toast__info {
  @apply flex-1;
}

.achievement-toast__name {
  @apply font-bold text-lg mb-1 text-white;
}

/* Timer bar */
.achievement-toast__timer {
  @apply h-1 bg-black/20;
}

.achievement-toast__timer-fill {
  @apply h-full bg-white/50;
  animation: timer-shrink linear forwards;
  animation-duration: var(--duration, 5000ms);
}

@keyframes timer-shrink {
  from { width: 100%; }
  to { width: 0%; }
}

/* Confetti */
.achievement-toast__confetti {
  @apply absolute inset-0 overflow-hidden pointer-events-none;
}

.confetti {
  @apply absolute top-0 rounded-sm;
  width: var(--confetti-size);
  height: var(--confetti-size);
  left: var(--confetti-left);
  background-color: var(--confetti-color);
  animation: confetti-fall var(--confetti-duration) ease-out var(--confetti-delay) forwards;
  opacity: 0;
}

@keyframes confetti-fall {
  0% {
    transform: translateY(-10px) rotate(0deg);
    opacity: 1;
  }
  100% {
    transform: translateY(100px) rotate(720deg);
    opacity: 0;
  }
}

/* Toast transitions */
.toast-enter-active {
  animation: toast-entrance 0.6s ease-out;
}

.toast-leave-active {
  transition: all 0.4s ease-in;
}

.toast-leave-to {
  transform: translateX(100%);
  opacity: 0;
}

.toast-move {
  transition: transform 0.4s ease;
}
</style>
