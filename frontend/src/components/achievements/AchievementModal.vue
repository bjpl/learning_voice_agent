<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="modelValue && achievement" class="achievement-modal-overlay" @click.self="close">
        <div class="achievement-modal" :class="`achievement-modal--${achievement.rarity}`">
          <!-- Close button -->
          <button class="achievement-modal__close" @click="close">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>

          <!-- Decorative background -->
          <div class="achievement-modal__bg" />

          <!-- Content -->
          <div class="achievement-modal__content">
            <!-- Badge display -->
            <div class="achievement-modal__badge">
              <div class="achievement-modal__icon-ring">
                <span class="achievement-modal__icon">{{ achievement.icon }}</span>
              </div>
              <div v-if="achievement.unlockedAt" class="achievement-modal__sparkles">
                <span v-for="i in 8" :key="i" class="sparkle" :style="sparkleStyle(i)" />
              </div>
            </div>

            <!-- Info -->
            <h2 class="achievement-modal__name">{{ achievement.name }}</h2>
            <RarityBadge :rarity="achievement.rarity" size="md" class="mb-4" />

            <p class="achievement-modal__description">{{ achievement.description }}</p>

            <!-- Status -->
            <div v-if="achievement.unlockedAt" class="achievement-modal__unlocked">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clip-rule="evenodd" />
              </svg>
              <span>Unlocked on {{ formatDate(achievement.unlockedAt) }}</span>
            </div>

            <div v-else-if="achievement.progress !== undefined" class="achievement-modal__progress">
              <div class="achievement-modal__progress-header">
                <span>Progress</span>
                <span>{{ achievement.progress }} / {{ achievement.maxProgress }}</span>
              </div>
              <div class="achievement-modal__progress-bar">
                <div
                  class="achievement-modal__progress-fill"
                  :style="{ width: `${progressPercent}%` }"
                />
              </div>
            </div>

            <div v-else class="achievement-modal__locked">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
                <path fill-rule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clip-rule="evenodd" />
              </svg>
              <span>Keep learning to unlock this achievement!</span>
            </div>

            <!-- Actions -->
            <div v-if="achievement.unlockedAt" class="achievement-modal__actions">
              <button class="achievement-modal__share-btn" @click="share">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
                  <path d="M13 4.5a2.5 2.5 0 11.702 1.737L6.97 9.604a2.518 2.518 0 010 .792l6.733 3.367a2.5 2.5 0 11-.671 1.341l-6.733-3.367a2.5 2.5 0 110-3.475l6.733-3.366A2.52 2.52 0 0113 4.5z" />
                </svg>
                Share Achievement
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Achievement } from './AchievementBadge.vue'
import RarityBadge from './RarityBadge.vue'

const props = defineProps<{
  modelValue: boolean
  achievement: Achievement | null
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  share: [achievement: Achievement]
}>()

const progressPercent = computed(() => {
  if (!props.achievement?.progress || !props.achievement?.maxProgress) return 0
  return Math.round((props.achievement.progress / props.achievement.maxProgress) * 100)
})

function close() {
  emit('update:modelValue', false)
}

function share() {
  if (props.achievement) {
    emit('share', props.achievement)
  }
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}

function sparkleStyle(index: number) {
  const angle = (index / 8) * 360
  const delay = (index / 8) * 0.3
  return {
    '--angle': `${angle}deg`,
    '--delay': `${delay}s`
  }
}
</script>

<style scoped>
.achievement-modal-overlay {
  @apply fixed inset-0 bg-black/60 backdrop-blur-sm z-50;
  @apply flex items-center justify-center p-4;
}

.achievement-modal {
  @apply relative bg-white dark:bg-gray-800 rounded-2xl shadow-2xl;
  @apply w-full max-w-md overflow-hidden;
}

.achievement-modal__close {
  @apply absolute top-4 right-4 p-2 rounded-full z-20;
  @apply text-gray-400 hover:text-gray-600 hover:bg-gray-100;
  @apply dark:hover:text-gray-300 dark:hover:bg-gray-700;
  @apply transition-colors;
}

/* Rarity-based backgrounds */
.achievement-modal__bg {
  @apply absolute top-0 left-0 right-0 h-32;
}

.achievement-modal--common .achievement-modal__bg {
  background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
}

.achievement-modal--uncommon .achievement-modal__bg {
  background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
}

.achievement-modal--rare .achievement-modal__bg {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}

.achievement-modal--epic .achievement-modal__bg {
  background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%);
}

.achievement-modal--legendary .achievement-modal__bg {
  background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
  animation: legendary-shimmer 3s ease-in-out infinite;
}

@keyframes legendary-shimmer {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.2); }
}

.achievement-modal__content {
  @apply relative pt-20 pb-6 px-6 text-center z-10;
}

.achievement-modal__badge {
  @apply relative -mt-32 mb-4;
}

.achievement-modal__icon-ring {
  @apply w-24 h-24 mx-auto rounded-full bg-white dark:bg-gray-800;
  @apply flex items-center justify-center shadow-lg;
  @apply border-4 border-white dark:border-gray-700;
}

.achievement-modal__icon {
  @apply text-5xl;
  filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
}

.achievement-modal__sparkles {
  @apply absolute inset-0 pointer-events-none;
}

.achievement-modal__sparkles .sparkle {
  @apply absolute w-2 h-2 rounded-full bg-yellow-400;
  top: 50%;
  left: 50%;
  animation: modal-sparkle 2s ease-in-out infinite;
  animation-delay: var(--delay);
}

@keyframes modal-sparkle {
  0%, 100% {
    transform: translate(-50%, -50%) rotate(var(--angle)) translateY(40px) scale(0);
    opacity: 0;
  }
  50% {
    transform: translate(-50%, -50%) rotate(var(--angle)) translateY(50px) scale(1);
    opacity: 1;
  }
}

.achievement-modal__name {
  @apply text-2xl font-bold text-gray-900 dark:text-white mb-2;
}

.achievement-modal__description {
  @apply text-gray-600 dark:text-gray-400 mb-6;
}

.achievement-modal__unlocked {
  @apply flex items-center justify-center gap-2 py-3 px-4;
  @apply bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400;
  @apply rounded-lg text-sm font-medium mb-4;
}

.achievement-modal__locked {
  @apply flex items-center justify-center gap-2 py-3 px-4;
  @apply bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400;
  @apply rounded-lg text-sm mb-4;
}

.achievement-modal__progress {
  @apply mb-6;
}

.achievement-modal__progress-header {
  @apply flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2;
}

.achievement-modal__progress-bar {
  @apply h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.achievement-modal__progress-fill {
  @apply h-full bg-indigo-500 rounded-full transition-all duration-500;
}

.achievement-modal__actions {
  @apply pt-4 border-t border-gray-200 dark:border-gray-700;
}

.achievement-modal__share-btn {
  @apply flex items-center justify-center gap-2 w-full py-3 px-4;
  @apply bg-indigo-600 text-white rounded-lg font-semibold;
  @apply hover:bg-indigo-700 transition-colors;
}

/* Modal transitions */
.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.3s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-active .achievement-modal {
  animation: modal-enter 0.4s ease-out;
}

.modal-leave-active .achievement-modal {
  animation: modal-enter 0.3s ease-in reverse;
}

@keyframes modal-enter {
  from {
    opacity: 0;
    transform: scale(0.9) translateY(20px);
  }
}
</style>
