<script setup lang="ts">
import { ref, onMounted } from 'vue';

interface StatCard {
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down' | 'neutral';
}

const stats = ref<StatCard[]>([
  { title: 'Total Sessions', value: '0', change: '+0%', trend: 'neutral' },
  { title: 'Learning Time', value: '0h', change: '+0%', trend: 'neutral' },
  { title: 'Words Learned', value: '0', change: '+0%', trend: 'neutral' },
  { title: 'Streak Days', value: '0', change: '+0%', trend: 'neutral' }
]);

const isLoading = ref(true);

onMounted(() => {
  setTimeout(() => {
    isLoading.value = false;
  }, 500);
});
</script>

<template>
  <div class="space-y-6">
    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div
        v-for="stat in stats"
        :key="stat.title"
        class="card"
      >
        <div class="text-sm font-medium text-gray-500">{{ stat.title }}</div>
        <div class="mt-2 flex items-baseline gap-2">
          <span class="text-2xl font-semibold text-gray-900">{{ stat.value }}</span>
          <span
            :class="[
              'text-sm font-medium',
              stat.trend === 'up' ? 'text-green-600' : stat.trend === 'down' ? 'text-red-600' : 'text-gray-500'
            ]"
          >
            {{ stat.change }}
          </span>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="card">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
      <div class="flex flex-wrap gap-3">
        <router-link to="/conversation" class="btn-primary">
          Start Learning Session
        </router-link>
        <router-link to="/goals" class="btn-secondary">
          View Goals
        </router-link>
        <router-link to="/analytics" class="btn-secondary">
          Check Progress
        </router-link>
      </div>
    </div>

    <!-- Recent Activity Placeholder -->
    <div class="card">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h2>
      <div v-if="isLoading" class="animate-pulse space-y-3">
        <div class="h-4 bg-gray-200 rounded w-3/4"></div>
        <div class="h-4 bg-gray-200 rounded w-1/2"></div>
        <div class="h-4 bg-gray-200 rounded w-2/3"></div>
      </div>
      <div v-else class="text-gray-500 text-center py-8">
        <p>No recent activity. Start a conversation to begin learning!</p>
      </div>
    </div>
  </div>
</template>
