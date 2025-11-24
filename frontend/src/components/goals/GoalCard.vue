<template>
  <div
    class="goal-card"
    :class="[
      `goal-card--${goal.type}`,
      { 'goal-card--completed': isCompleted, 'goal-card--overdue': isOverdue }
    ]"
  >
    <div class="goal-card__header">
      <div class="goal-card__icon" :class="`goal-icon--${goal.type}`">
        <component :is="goalIcon" />
      </div>
      <div class="goal-card__meta">
        <span class="goal-card__type">{{ goalTypeLabel }}</span>
        <span v-if="goal.deadline" class="goal-card__deadline" :class="{ 'text-red-500': isOverdue }">
          {{ deadlineText }}
        </span>
      </div>
      <div class="goal-card__actions">
        <button
          class="goal-card__action-btn"
          @click.stop="$emit('edit', goal)"
          title="Edit goal"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
            <path d="M5.433 13.917l1.262-3.155A4 4 0 017.58 9.42l6.92-6.918a2.121 2.121 0 013 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 01-.65-.65z" />
            <path d="M3.5 5.75c0-.69.56-1.25 1.25-1.25H10A.75.75 0 0010 3H4.75A2.75 2.75 0 002 5.75v9.5A2.75 2.75 0 004.75 18h9.5A2.75 2.75 0 0017 15.25V10a.75.75 0 00-1.5 0v5.25c0 .69-.56 1.25-1.25 1.25h-9.5c-.69 0-1.25-.56-1.25-1.25v-9.5z" />
          </svg>
        </button>
        <button
          class="goal-card__action-btn goal-card__action-btn--danger"
          @click.stop="$emit('delete', goal)"
          title="Delete goal"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
            <path fill-rule="evenodd" d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.519.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z" clip-rule="evenodd" />
          </svg>
        </button>
      </div>
    </div>

    <h3 class="goal-card__title">{{ goal.title }}</h3>
    <p v-if="goal.description" class="goal-card__description">{{ goal.description }}</p>

    <div class="goal-card__progress-section">
      <div class="goal-card__progress-header">
        <span class="goal-card__progress-label">Progress</span>
        <span class="goal-card__progress-value">{{ goal.currentValue }} / {{ goal.targetValue }} {{ goal.unit }}</span>
      </div>
      <div class="goal-card__progress-bar">
        <div
          class="goal-card__progress-fill"
          :style="{ width: `${progressPercent}%` }"
          :class="progressColorClass"
        />
      </div>
      <span class="goal-card__progress-percent">{{ progressPercent }}%</span>
    </div>

    <div class="goal-card__footer">
      <button
        v-if="!isCompleted"
        class="goal-card__increment-btn"
        @click.stop="$emit('increment', goal)"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
          <path d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z" />
        </svg>
        Log Progress
      </button>
      <div v-else class="goal-card__completed-badge">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clip-rule="evenodd" />
        </svg>
        Completed!
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, h } from 'vue'

export interface Goal {
  id: string
  title: string
  description?: string
  type: 'streak' | 'sessions' | 'topics' | 'quality' | 'exchanges' | 'duration' | 'feedback' | 'custom'
  currentValue: number
  targetValue: number
  unit: string
  deadline?: string
  createdAt: string
  completedAt?: string
}

const props = defineProps<{
  goal: Goal
}>()

defineEmits<{
  edit: [goal: Goal]
  delete: [goal: Goal]
  increment: [goal: Goal]
}>()

const goalTypeLabels: Record<Goal['type'], string> = {
  streak: 'Daily Streak',
  sessions: 'Sessions',
  topics: 'Topics Covered',
  quality: 'Quality Score',
  exchanges: 'Exchanges',
  duration: 'Practice Time',
  feedback: 'Feedback Score',
  custom: 'Custom Goal'
}

const goalTypeLabel = computed(() => goalTypeLabels[props.goal.type])

const progressPercent = computed(() => {
  const percent = Math.round((props.goal.currentValue / props.goal.targetValue) * 100)
  return Math.min(percent, 100)
})

const isCompleted = computed(() => props.goal.currentValue >= props.goal.targetValue)

const isOverdue = computed(() => {
  if (!props.goal.deadline || isCompleted.value) return false
  return new Date(props.goal.deadline) < new Date()
})

const deadlineText = computed(() => {
  if (!props.goal.deadline) return ''
  const deadline = new Date(props.goal.deadline)
  const now = new Date()
  const diffDays = Math.ceil((deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24))

  if (diffDays < 0) return `${Math.abs(diffDays)} days overdue`
  if (diffDays === 0) return 'Due today'
  if (diffDays === 1) return 'Due tomorrow'
  return `${diffDays} days left`
})

const progressColorClass = computed(() => {
  if (isCompleted.value) return 'bg-green-500'
  if (progressPercent.value >= 75) return 'bg-emerald-500'
  if (progressPercent.value >= 50) return 'bg-blue-500'
  if (progressPercent.value >= 25) return 'bg-yellow-500'
  return 'bg-orange-500'
})

// Goal type icons as render functions
const goalIcons: Record<Goal['type'], () => any> = {
  streak: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { 'fill-rule': 'evenodd', d: 'M12.963 2.286a.75.75 0 00-1.071-.136 9.742 9.742 0 00-3.539 6.177A7.547 7.547 0 016.648 6.61a.75.75 0 00-1.152.082A9 9 0 1015.68 4.534a7.46 7.46 0 01-2.717-2.248zM15.75 14.25a3.75 3.75 0 11-7.313-1.172c.628.465 1.35.81 2.133 1a5.99 5.99 0 011.925-3.545 3.75 3.75 0 013.255 3.717z', 'clip-rule': 'evenodd' })
  ]),
  sessions: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { 'fill-rule': 'evenodd', d: 'M4.125 3C3.089 3 2.25 3.84 2.25 4.875V18a3 3 0 003 3h15a3 3 0 01-3-3V4.875C17.25 3.839 16.41 3 15.375 3H4.125zM12 9.75a.75.75 0 000 1.5h1.5a.75.75 0 000-1.5H12zm-.75-2.25a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5H12a.75.75 0 01-.75-.75zM6 12.75a.75.75 0 000 1.5h7.5a.75.75 0 000-1.5H6zm-.75 3.75a.75.75 0 01.75-.75h7.5a.75.75 0 010 1.5H6a.75.75 0 01-.75-.75zM6 6.75a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h3a.75.75 0 00.75-.75v-3A.75.75 0 009 6.75H6z', 'clip-rule': 'evenodd' }),
    h('path', { d: 'M18.75 6.75h1.875c.621 0 1.125.504 1.125 1.125V18a1.5 1.5 0 01-3 0V6.75z' })
  ]),
  topics: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { d: 'M11.25 4.533A9.707 9.707 0 006 3a9.735 9.735 0 00-3.25.555.75.75 0 00-.5.707v14.25a.75.75 0 001 .707A8.237 8.237 0 016 18.75c1.995 0 3.823.707 5.25 1.886V4.533zM12.75 20.636A8.214 8.214 0 0118 18.75c.966 0 1.89.166 2.75.47a.75.75 0 001-.708V4.262a.75.75 0 00-.5-.707A9.735 9.735 0 0018 3a9.707 9.707 0 00-5.25 1.533v16.103z' })
  ]),
  quality: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { 'fill-rule': 'evenodd', d: 'M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z', 'clip-rule': 'evenodd' })
  ]),
  exchanges: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { 'fill-rule': 'evenodd', d: 'M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97zM6.75 8.25a.75.75 0 01.75-.75h9a.75.75 0 010 1.5h-9a.75.75 0 01-.75-.75zm.75 2.25a.75.75 0 000 1.5H12a.75.75 0 000-1.5H7.5z', 'clip-rule': 'evenodd' })
  ]),
  duration: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { 'fill-rule': 'evenodd', d: 'M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM12.75 6a.75.75 0 00-1.5 0v6c0 .414.336.75.75.75h4.5a.75.75 0 000-1.5h-3.75V6z', 'clip-rule': 'evenodd' })
  ]),
  feedback: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { d: 'M7.493 18.75c-.425 0-.82-.236-.975-.632A7.48 7.48 0 016 15.375c0-1.75.599-3.358 1.602-4.634.151-.192.373-.309.6-.397.473-.183.89-.514 1.212-.924a9.042 9.042 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75 2.25 2.25 0 012.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H14.23c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23h-.777zM2.331 10.977a11.969 11.969 0 00-.831 4.398 12 12 0 00.52 3.507c.26.85 1.084 1.368 1.973 1.368H4.9c.445 0 .72-.498.523-.898a8.963 8.963 0 01-.924-3.977c0-1.708.476-3.305 1.302-4.666.245-.403-.028-.959-.5-.959H4.25c-.832 0-1.612.453-1.918 1.227z' })
  ]),
  custom: () => h('svg', { xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', class: 'w-6 h-6' }, [
    h('path', { 'fill-rule': 'evenodd', d: 'M11.078 2.25c-.917 0-1.699.663-1.85 1.567L9.05 4.889c-.02.12-.115.26-.297.348a7.493 7.493 0 00-.986.57c-.166.115-.334.126-.45.083L6.3 5.508a1.875 1.875 0 00-2.282.819l-.922 1.597a1.875 1.875 0 00.432 2.385l.84.692c.095.078.17.229.154.43a7.598 7.598 0 000 1.139c.015.2-.059.352-.153.43l-.841.692a1.875 1.875 0 00-.432 2.385l.922 1.597a1.875 1.875 0 002.282.818l1.019-.382c.115-.043.283-.031.45.082.312.214.641.405.985.57.182.088.277.228.297.35l.178 1.071c.151.904.933 1.567 1.85 1.567h1.844c.916 0 1.699-.663 1.85-1.567l.178-1.072c.02-.12.114-.26.297-.349.344-.165.673-.356.985-.57.167-.114.335-.125.45-.082l1.02.382a1.875 1.875 0 002.28-.819l.923-1.597a1.875 1.875 0 00-.432-2.385l-.84-.692c-.095-.078-.17-.229-.154-.43a7.614 7.614 0 000-1.139c-.016-.2.059-.352.153-.43l.84-.692c.708-.582.891-1.59.433-2.385l-.922-1.597a1.875 1.875 0 00-2.282-.818l-1.02.382c-.114.043-.282.031-.449-.083a7.49 7.49 0 00-.985-.57c-.183-.087-.277-.227-.297-.348l-.179-1.072a1.875 1.875 0 00-1.85-1.567h-1.843zM12 15.75a3.75 3.75 0 100-7.5 3.75 3.75 0 000 7.5z', 'clip-rule': 'evenodd' })
  ])
}

const goalIcon = computed(() => goalIcons[props.goal.type])
</script>

<style scoped>
.goal-card {
  @apply bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-gray-100 dark:border-gray-700;
  @apply transition-all duration-300 hover:shadow-lg hover:-translate-y-1;
}

.goal-card--completed {
  @apply border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20;
}

.goal-card--overdue {
  @apply border-red-200 dark:border-red-800;
}

.goal-card__header {
  @apply flex items-start gap-3 mb-3;
}

.goal-card__icon {
  @apply w-10 h-10 rounded-lg flex items-center justify-center text-white flex-shrink-0;
}

.goal-icon--streak { @apply bg-gradient-to-br from-orange-400 to-red-500; }
.goal-icon--sessions { @apply bg-gradient-to-br from-blue-400 to-indigo-500; }
.goal-icon--topics { @apply bg-gradient-to-br from-emerald-400 to-teal-500; }
.goal-icon--quality { @apply bg-gradient-to-br from-yellow-400 to-amber-500; }
.goal-icon--exchanges { @apply bg-gradient-to-br from-purple-400 to-violet-500; }
.goal-icon--duration { @apply bg-gradient-to-br from-cyan-400 to-blue-500; }
.goal-icon--feedback { @apply bg-gradient-to-br from-pink-400 to-rose-500; }
.goal-icon--custom { @apply bg-gradient-to-br from-gray-400 to-slate-500; }

.goal-card__meta {
  @apply flex-1 flex flex-col gap-0.5;
}

.goal-card__type {
  @apply text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide;
}

.goal-card__deadline {
  @apply text-xs text-gray-400 dark:text-gray-500;
}

.goal-card__actions {
  @apply flex gap-1;
}

.goal-card__action-btn {
  @apply p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100;
  @apply dark:hover:text-gray-300 dark:hover:bg-gray-700 transition-colors;
}

.goal-card__action-btn--danger:hover {
  @apply text-red-500 bg-red-50 dark:bg-red-900/20;
}

.goal-card__title {
  @apply text-lg font-semibold text-gray-900 dark:text-white mb-1;
}

.goal-card__description {
  @apply text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-2;
}

.goal-card__progress-section {
  @apply mb-4;
}

.goal-card__progress-header {
  @apply flex justify-between items-center mb-2;
}

.goal-card__progress-label {
  @apply text-xs font-medium text-gray-500 dark:text-gray-400;
}

.goal-card__progress-value {
  @apply text-xs font-semibold text-gray-700 dark:text-gray-300;
}

.goal-card__progress-bar {
  @apply h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.goal-card__progress-fill {
  @apply h-full rounded-full transition-all duration-500 ease-out;
  animation: progress-fill 0.8s ease-out;
}

@keyframes progress-fill {
  from { width: 0; }
}

.goal-card__progress-percent {
  @apply text-xs font-bold text-gray-600 dark:text-gray-400 mt-1 block text-right;
}

.goal-card__footer {
  @apply flex justify-end;
}

.goal-card__increment-btn {
  @apply flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium;
  @apply bg-indigo-100 text-indigo-700 rounded-lg;
  @apply hover:bg-indigo-200 transition-colors;
  @apply dark:bg-indigo-900/30 dark:text-indigo-400 dark:hover:bg-indigo-900/50;
}

.goal-card__completed-badge {
  @apply flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold;
  @apply text-green-600 dark:text-green-400;
  animation: badge-pop 0.5s ease-out;
}

@keyframes badge-pop {
  0% { transform: scale(0.8); opacity: 0; }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); opacity: 1; }
}
</style>
