<template>
  <div class="goal-progress" :class="`goal-progress--${size}`">
    <svg class="goal-progress__svg" :viewBox="`0 0 ${viewBoxSize} ${viewBoxSize}`">
      <!-- Background circle -->
      <circle
        class="goal-progress__bg"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth"
      />
      <!-- Progress circle -->
      <circle
        class="goal-progress__fill"
        :class="progressColorClass"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
        stroke-linecap="round"
      />
      <!-- Glow effect for high progress -->
      <circle
        v-if="percent >= 75"
        class="goal-progress__glow"
        :class="progressColorClass"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth + 4"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
        stroke-linecap="round"
      />
    </svg>

    <div class="goal-progress__content">
      <span class="goal-progress__percent">{{ displayPercent }}%</span>
      <span v-if="showLabel" class="goal-progress__label">{{ label }}</span>
    </div>

    <!-- Completion sparkles -->
    <div v-if="percent >= 100 && showSparkles" class="goal-progress__sparkles">
      <span v-for="i in 6" :key="i" class="sparkle" :style="sparkleStyle(i)" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const props = withDefaults(defineProps<{
  percent: number
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showLabel?: boolean
  label?: string
  animated?: boolean
  showSparkles?: boolean
}>(), {
  size: 'md',
  showLabel: false,
  label: 'Complete',
  animated: true,
  showSparkles: true
})

const sizeConfig = {
  sm: { viewBox: 60, stroke: 4 },
  md: { viewBox: 80, stroke: 6 },
  lg: { viewBox: 100, stroke: 8 },
  xl: { viewBox: 140, stroke: 10 }
}

const viewBoxSize = computed(() => sizeConfig[props.size].viewBox)
const strokeWidth = computed(() => sizeConfig[props.size].stroke)
const center = computed(() => viewBoxSize.value / 2)
const radius = computed(() => (viewBoxSize.value - strokeWidth.value) / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)

const displayPercent = ref(0)

// Animate the percentage
watch(() => props.percent, (newVal) => {
  if (props.animated) {
    const start = displayPercent.value
    const end = Math.min(newVal, 100)
    const duration = 800
    const startTime = performance.now()

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)

      // Easing function
      const eased = 1 - Math.pow(1 - progress, 3)
      displayPercent.value = Math.round(start + (end - start) * eased)

      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }

    requestAnimationFrame(animate)
  } else {
    displayPercent.value = Math.min(newVal, 100)
  }
}, { immediate: true })

const dashOffset = computed(() => {
  const normalizedPercent = Math.min(displayPercent.value, 100) / 100
  return circumference.value * (1 - normalizedPercent)
})

const progressColorClass = computed(() => {
  const p = props.percent
  if (p >= 100) return 'stroke-green-500'
  if (p >= 75) return 'stroke-emerald-500'
  if (p >= 50) return 'stroke-blue-500'
  if (p >= 25) return 'stroke-yellow-500'
  return 'stroke-orange-500'
})

function sparkleStyle(index: number) {
  const angle = (index / 6) * 360
  const delay = index * 0.1
  return {
    '--angle': `${angle}deg`,
    '--delay': `${delay}s`
  }
}
</script>

<style scoped>
.goal-progress {
  @apply relative inline-flex items-center justify-center;
}

.goal-progress--sm { @apply w-14 h-14; }
.goal-progress--md { @apply w-20 h-20; }
.goal-progress--lg { @apply w-28 h-28; }
.goal-progress--xl { @apply w-36 h-36; }

.goal-progress__svg {
  @apply w-full h-full transform -rotate-90;
}

.goal-progress__bg {
  @apply stroke-gray-200 dark:stroke-gray-700;
}

.goal-progress__fill {
  transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.goal-progress__glow {
  opacity: 0.3;
  filter: blur(4px);
  transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.goal-progress__content {
  @apply absolute inset-0 flex flex-col items-center justify-center;
}

.goal-progress__percent {
  @apply font-bold text-gray-900 dark:text-white;
}

.goal-progress--sm .goal-progress__percent { @apply text-xs; }
.goal-progress--md .goal-progress__percent { @apply text-sm; }
.goal-progress--lg .goal-progress__percent { @apply text-lg; }
.goal-progress--xl .goal-progress__percent { @apply text-2xl; }

.goal-progress__label {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.goal-progress--sm .goal-progress__label { @apply text-[10px]; }

/* Sparkle animation for completion */
.goal-progress__sparkles {
  @apply absolute inset-0 pointer-events-none;
}

.sparkle {
  @apply absolute w-1.5 h-1.5 bg-yellow-400 rounded-full;
  top: 50%;
  left: 50%;
  animation: sparkle 1.5s ease-out infinite;
  animation-delay: var(--delay);
  transform-origin: center;
}

@keyframes sparkle {
  0% {
    transform: translate(-50%, -50%) rotate(var(--angle)) translateY(0) scale(0);
    opacity: 1;
  }
  50% {
    opacity: 1;
  }
  100% {
    transform: translate(-50%, -50%) rotate(var(--angle)) translateY(-40px) scale(1);
    opacity: 0;
  }
}

/* Pulse animation on completion */
.goal-progress__fill.stroke-green-500 {
  animation: pulse-green 2s ease-in-out infinite;
}

@keyframes pulse-green {
  0%, 100% {
    filter: drop-shadow(0 0 2px rgb(34 197 94 / 0.5));
  }
  50% {
    filter: drop-shadow(0 0 8px rgb(34 197 94 / 0.8));
  }
}
</style>
