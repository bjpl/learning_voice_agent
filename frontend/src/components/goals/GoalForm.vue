<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="modelValue" class="goal-form-overlay" @click.self="close">
        <div class="goal-form" @click.stop>
          <div class="goal-form__header">
            <h2 class="goal-form__title">
              {{ isEditing ? 'Edit Goal' : 'Create New Goal' }}
            </h2>
            <button class="goal-form__close" @click="close">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
                <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
              </svg>
            </button>
          </div>

          <form @submit.prevent="submit" class="goal-form__body">
            <!-- Goal Type Selection -->
            <div class="goal-form__field">
              <label class="goal-form__label">Goal Type</label>
              <div class="goal-form__type-grid">
                <button
                  v-for="type in goalTypes"
                  :key="type.value"
                  type="button"
                  class="goal-form__type-btn"
                  :class="{ 'goal-form__type-btn--active': form.type === type.value }"
                  @click="form.type = type.value"
                >
                  <span class="goal-form__type-icon" :class="`type-icon--${type.value}`">
                    {{ type.icon }}
                  </span>
                  <span class="goal-form__type-name">{{ type.label }}</span>
                </button>
              </div>
            </div>

            <!-- Title -->
            <div class="goal-form__field">
              <label class="goal-form__label" for="goal-title">Title</label>
              <input
                id="goal-title"
                v-model="form.title"
                type="text"
                class="goal-form__input"
                placeholder="e.g., Practice for 30 days straight"
                required
              />
            </div>

            <!-- Description -->
            <div class="goal-form__field">
              <label class="goal-form__label" for="goal-desc">Description (optional)</label>
              <textarea
                id="goal-desc"
                v-model="form.description"
                class="goal-form__textarea"
                placeholder="Add more details about your goal..."
                rows="3"
              />
            </div>

            <!-- Target Value -->
            <div class="goal-form__row">
              <div class="goal-form__field goal-form__field--half">
                <label class="goal-form__label" for="goal-target">Target</label>
                <input
                  id="goal-target"
                  v-model.number="form.targetValue"
                  type="number"
                  min="1"
                  class="goal-form__input"
                  placeholder="10"
                  required
                />
              </div>
              <div class="goal-form__field goal-form__field--half">
                <label class="goal-form__label" for="goal-unit">Unit</label>
                <input
                  id="goal-unit"
                  v-model="form.unit"
                  type="text"
                  class="goal-form__input"
                  :placeholder="suggestedUnit"
                />
              </div>
            </div>

            <!-- Current Progress (Edit mode) -->
            <div v-if="isEditing" class="goal-form__field">
              <label class="goal-form__label" for="goal-current">Current Progress</label>
              <input
                id="goal-current"
                v-model.number="form.currentValue"
                type="number"
                min="0"
                class="goal-form__input"
              />
            </div>

            <!-- Deadline -->
            <div class="goal-form__field">
              <label class="goal-form__label" for="goal-deadline">Deadline (optional)</label>
              <input
                id="goal-deadline"
                v-model="form.deadline"
                type="date"
                class="goal-form__input"
                :min="minDate"
              />
            </div>

            <!-- Actions -->
            <div class="goal-form__actions">
              <button type="button" class="goal-form__btn goal-form__btn--secondary" @click="close">
                Cancel
              </button>
              <button type="submit" class="goal-form__btn goal-form__btn--primary" :disabled="!isValid">
                {{ isEditing ? 'Save Changes' : 'Create Goal' }}
              </button>
            </div>
          </form>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import type { Goal } from './GoalCard.vue'

const props = defineProps<{
  modelValue: boolean
  goal?: Goal | null
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  submit: [goal: Omit<Goal, 'id' | 'createdAt'>]
}>()

const goalTypes = [
  { value: 'streak', label: 'Streak', icon: '🔥' },
  { value: 'sessions', label: 'Sessions', icon: '📚' },
  { value: 'topics', label: 'Topics', icon: '📖' },
  { value: 'quality', label: 'Quality', icon: '⭐' },
  { value: 'exchanges', label: 'Exchanges', icon: '💬' },
  { value: 'duration', label: 'Duration', icon: '⏱️' },
  { value: 'feedback', label: 'Feedback', icon: '👍' },
  { value: 'custom', label: 'Custom', icon: '⚙️' }
] as const

const defaultForm = {
  type: 'sessions' as Goal['type'],
  title: '',
  description: '',
  targetValue: 10,
  currentValue: 0,
  unit: '',
  deadline: ''
}

const form = reactive({ ...defaultForm })

const isEditing = computed(() => !!props.goal)

const isValid = computed(() => {
  return form.title.trim() && form.targetValue > 0
})

const minDate = computed(() => {
  const today = new Date()
  return today.toISOString().split('T')[0]
})

const suggestedUnits: Record<Goal['type'], string> = {
  streak: 'days',
  sessions: 'sessions',
  topics: 'topics',
  quality: 'points',
  exchanges: 'exchanges',
  duration: 'minutes',
  feedback: 'ratings',
  custom: 'units'
}

const suggestedUnit = computed(() => suggestedUnits[form.type])

// Reset form when modal opens
watch(() => props.modelValue, (isOpen) => {
  if (isOpen) {
    if (props.goal) {
      // Edit mode - populate form
      Object.assign(form, {
        type: props.goal.type,
        title: props.goal.title,
        description: props.goal.description || '',
        targetValue: props.goal.targetValue,
        currentValue: props.goal.currentValue,
        unit: props.goal.unit,
        deadline: props.goal.deadline || ''
      })
    } else {
      // Create mode - reset form
      Object.assign(form, defaultForm)
    }
  }
})

// Auto-suggest unit when type changes
watch(() => form.type, (newType) => {
  if (!form.unit || Object.values(suggestedUnits).includes(form.unit)) {
    form.unit = suggestedUnits[newType]
  }
})

function close() {
  emit('update:modelValue', false)
}

function submit() {
  if (!isValid.value) return

  emit('submit', {
    type: form.type,
    title: form.title.trim(),
    description: form.description.trim() || undefined,
    targetValue: form.targetValue,
    currentValue: form.currentValue,
    unit: form.unit || suggestedUnit.value,
    deadline: form.deadline || undefined
  })

  close()
}
</script>

<style scoped>
.goal-form-overlay {
  @apply fixed inset-0 bg-black/50 backdrop-blur-sm z-50;
  @apply flex items-center justify-center p-4;
}

.goal-form {
  @apply bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-lg;
  @apply max-h-[90vh] overflow-y-auto;
  animation: form-enter 0.3s ease-out;
}

@keyframes form-enter {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(10px);
  }
}

.goal-form__header {
  @apply flex items-center justify-between p-5 border-b border-gray-200 dark:border-gray-700;
  @apply sticky top-0 bg-white dark:bg-gray-800 z-10;
}

.goal-form__title {
  @apply text-xl font-bold text-gray-900 dark:text-white;
}

.goal-form__close {
  @apply p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100;
  @apply dark:hover:text-gray-300 dark:hover:bg-gray-700 transition-colors;
}

.goal-form__body {
  @apply p-5 space-y-5;
}

.goal-form__field {
  @apply flex flex-col gap-2;
}

.goal-form__field--half {
  @apply flex-1;
}

.goal-form__row {
  @apply flex gap-4;
}

.goal-form__label {
  @apply text-sm font-medium text-gray-700 dark:text-gray-300;
}

.goal-form__input,
.goal-form__textarea {
  @apply w-full px-4 py-2.5 rounded-lg border border-gray-300 dark:border-gray-600;
  @apply bg-white dark:bg-gray-700 text-gray-900 dark:text-white;
  @apply focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none;
  @apply transition-colors placeholder:text-gray-400;
}

.goal-form__textarea {
  @apply resize-none;
}

.goal-form__type-grid {
  @apply grid grid-cols-4 gap-2;
}

.goal-form__type-btn {
  @apply flex flex-col items-center gap-1 p-3 rounded-lg border-2 border-gray-200 dark:border-gray-600;
  @apply hover:border-indigo-300 dark:hover:border-indigo-600 transition-all;
}

.goal-form__type-btn--active {
  @apply border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30;
}

.goal-form__type-icon {
  @apply text-xl;
}

.goal-form__type-name {
  @apply text-xs font-medium text-gray-600 dark:text-gray-400;
}

.goal-form__type-btn--active .goal-form__type-name {
  @apply text-indigo-600 dark:text-indigo-400;
}

.goal-form__actions {
  @apply flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700;
}

.goal-form__btn {
  @apply flex-1 py-2.5 px-4 rounded-lg font-semibold transition-colors;
}

.goal-form__btn--secondary {
  @apply bg-gray-100 text-gray-700 hover:bg-gray-200;
  @apply dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600;
}

.goal-form__btn--primary {
  @apply bg-indigo-600 text-white hover:bg-indigo-700;
  @apply disabled:opacity-50 disabled:cursor-not-allowed;
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

.modal-enter-active .goal-form {
  animation: form-enter 0.3s ease-out;
}

.modal-leave-active .goal-form {
  animation: form-enter 0.2s ease-in reverse;
}
</style>
