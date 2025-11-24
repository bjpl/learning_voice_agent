<script setup lang="ts">
import { ref, onMounted } from 'vue';

interface TimeRange {
  label: string;
  value: 'week' | 'month' | 'year';
}

const timeRanges: TimeRange[] = [
  { label: 'This Week', value: 'week' },
  { label: 'This Month', value: 'month' },
  { label: 'This Year', value: 'year' }
];

const selectedRange = ref<TimeRange['value']>('week');
const isLoading = ref(true);

onMounted(() => {
  setTimeout(() => {
    isLoading.value = false;
  }, 500);
});
</script>

<template>
  <div class="space-y-6">
    <!-- Time Range Selector -->
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold text-gray-900">Learning Analytics</h2>
      <div class="flex gap-2">
        <button
          v-for="range in timeRanges"
          :key="range.value"
          @click="selectedRange = range.value"
          :class="[
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            selectedRange === range.value
              ? 'bg-indigo-600 text-white'
              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
          ]"
        >
          {{ range.label }}
        </button>
      </div>
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Learning Progress Chart Placeholder -->
      <div class="card">
        <h3 class="text-md font-medium text-gray-900 mb-4">Learning Progress</h3>
        <div v-if="isLoading" class="animate-pulse h-64 bg-gray-100 rounded-lg"></div>
        <div v-else class="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
          <p class="text-gray-500">Chart will be rendered here with Chart.js</p>
        </div>
      </div>

      <!-- Session Duration Chart Placeholder -->
      <div class="card">
        <h3 class="text-md font-medium text-gray-900 mb-4">Session Duration</h3>
        <div v-if="isLoading" class="animate-pulse h-64 bg-gray-100 rounded-lg"></div>
        <div v-else class="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
          <p class="text-gray-500">Chart will be rendered here with Chart.js</p>
        </div>
      </div>

      <!-- Vocabulary Growth Chart Placeholder -->
      <div class="card">
        <h3 class="text-md font-medium text-gray-900 mb-4">Vocabulary Growth</h3>
        <div v-if="isLoading" class="animate-pulse h-64 bg-gray-100 rounded-lg"></div>
        <div v-else class="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
          <p class="text-gray-500">Chart will be rendered here with Chart.js</p>
        </div>
      </div>

      <!-- Accuracy Metrics Chart Placeholder -->
      <div class="card">
        <h3 class="text-md font-medium text-gray-900 mb-4">Accuracy Metrics</h3>
        <div v-if="isLoading" class="animate-pulse h-64 bg-gray-100 rounded-lg"></div>
        <div v-else class="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
          <p class="text-gray-500">Chart will be rendered here with Chart.js</p>
        </div>
      </div>
    </div>

    <!-- Summary Stats -->
    <div class="card">
      <h3 class="text-md font-medium text-gray-900 mb-4">Summary Statistics</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-indigo-600">0</div>
          <div class="text-sm text-gray-500">Total Sessions</div>
        </div>
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-indigo-600">0h</div>
          <div class="text-sm text-gray-500">Total Time</div>
        </div>
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-indigo-600">0%</div>
          <div class="text-sm text-gray-500">Avg. Accuracy</div>
        </div>
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-indigo-600">0</div>
          <div class="text-sm text-gray-500">Words Mastered</div>
        </div>
      </div>
    </div>
  </div>
</template>
