<script setup lang="ts">
import { ref } from 'vue';
import { v4 as uuidv4 } from 'uuid';

interface Goal {
  id: string;
  title: string;
  description: string;
  target: number;
  current: number;
  unit: string;
  deadline: string;
  status: 'active' | 'completed' | 'paused';
}

const goals = ref<Goal[]>([
  {
    id: uuidv4(),
    title: 'Learn 100 New Words',
    description: 'Expand vocabulary by learning new words each day',
    target: 100,
    current: 0,
    unit: 'words',
    deadline: '2024-12-31',
    status: 'active'
  },
  {
    id: uuidv4(),
    title: 'Complete 30 Sessions',
    description: 'Practice consistently with daily learning sessions',
    target: 30,
    current: 0,
    unit: 'sessions',
    deadline: '2024-12-31',
    status: 'active'
  }
]);

const showAddModal = ref(false);
const newGoal = ref({
  title: '',
  description: '',
  target: 10,
  unit: 'items',
  deadline: ''
});

const calculateProgress = (goal: Goal): number => {
  return Math.round((goal.current / goal.target) * 100);
};

const addGoal = (): void => {
  if (!newGoal.value.title.trim()) return;

  goals.value.push({
    id: uuidv4(),
    title: newGoal.value.title,
    description: newGoal.value.description,
    target: newGoal.value.target,
    current: 0,
    unit: newGoal.value.unit,
    deadline: newGoal.value.deadline,
    status: 'active'
  });

  newGoal.value = { title: '', description: '', target: 10, unit: 'items', deadline: '' };
  showAddModal.value = false;
};

const deleteGoal = (id: string): void => {
  goals.value = goals.value.filter(g => g.id !== id);
};
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-lg font-semibold text-gray-900">Learning Goals</h2>
        <p class="text-sm text-gray-500">Track and manage your learning objectives</p>
      </div>
      <button @click="showAddModal = true" class="btn-primary">
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
        </svg>
        Add Goal
      </button>
    </div>

    <!-- Goals List -->
    <div class="space-y-4">
      <div
        v-for="goal in goals"
        :key="goal.id"
        class="card"
      >
        <div class="flex items-start justify-between mb-4">
          <div>
            <h3 class="font-semibold text-gray-900">{{ goal.title }}</h3>
            <p class="text-sm text-gray-500 mt-1">{{ goal.description }}</p>
          </div>
          <div class="flex items-center gap-2">
            <span
              :class="[
                'px-2 py-1 text-xs font-medium rounded-full',
                goal.status === 'active' ? 'bg-green-100 text-green-700' :
                goal.status === 'completed' ? 'bg-indigo-100 text-indigo-700' :
                'bg-gray-100 text-gray-700'
              ]"
            >
              {{ goal.status }}
            </span>
            <button
              @click="deleteGoal(goal.id)"
              class="p-1 text-gray-400 hover:text-red-500 transition-colors"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>

        <!-- Progress Bar -->
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span class="text-gray-600">{{ goal.current }} / {{ goal.target }} {{ goal.unit }}</span>
            <span class="font-medium text-gray-900">{{ calculateProgress(goal) }}%</span>
          </div>
          <div class="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              class="h-full bg-indigo-600 rounded-full transition-all duration-300"
              :style="{ width: `${calculateProgress(goal)}%` }"
            ></div>
          </div>
          <div class="text-xs text-gray-500">
            Deadline: {{ goal.deadline || 'No deadline set' }}
          </div>
        </div>
      </div>

      <div v-if="goals.length === 0" class="card text-center py-12">
        <p class="text-gray-500">No goals yet. Add your first learning goal!</p>
      </div>
    </div>

    <!-- Add Goal Modal -->
    <div v-if="showAddModal" class="fixed inset-0 z-50 flex items-center justify-center">
      <div class="absolute inset-0 bg-black/50" @click="showAddModal = false"></div>
      <div class="relative bg-white rounded-xl shadow-xl p-6 w-full max-w-md mx-4">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Add New Goal</h3>

        <div class="space-y-4">
          <div>
            <label class="label">Title</label>
            <input v-model="newGoal.title" type="text" class="input" placeholder="e.g., Learn 50 new words">
          </div>
          <div>
            <label class="label">Description</label>
            <textarea v-model="newGoal.description" class="input" rows="2" placeholder="Describe your goal..."></textarea>
          </div>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="label">Target</label>
              <input v-model.number="newGoal.target" type="number" class="input" min="1">
            </div>
            <div>
              <label class="label">Unit</label>
              <input v-model="newGoal.unit" type="text" class="input" placeholder="e.g., words">
            </div>
          </div>
          <div>
            <label class="label">Deadline</label>
            <input v-model="newGoal.deadline" type="date" class="input">
          </div>
        </div>

        <div class="flex justify-end gap-3 mt-6">
          <button @click="showAddModal = false" class="btn-secondary">Cancel</button>
          <button @click="addGoal" class="btn-primary">Add Goal</button>
        </div>
      </div>
    </div>
  </div>
</template>
