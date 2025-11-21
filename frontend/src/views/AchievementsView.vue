<script setup lang="ts">
import { ref, computed } from 'vue';

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  category: 'learning' | 'consistency' | 'mastery' | 'social';
  unlocked: boolean;
  unlockedAt?: string;
  progress?: number;
  requirement: number;
}

const achievements = ref<Achievement[]>([
  {
    id: '1',
    title: 'First Steps',
    description: 'Complete your first learning session',
    icon: 'M13 10V3L4 14h7v7l9-11h-7z',
    category: 'learning',
    unlocked: false,
    progress: 0,
    requirement: 1
  },
  {
    id: '2',
    title: 'Word Collector',
    description: 'Learn 50 new vocabulary words',
    icon: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253',
    category: 'learning',
    unlocked: false,
    progress: 0,
    requirement: 50
  },
  {
    id: '3',
    title: 'Streak Starter',
    description: 'Maintain a 7-day learning streak',
    icon: 'M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z',
    category: 'consistency',
    unlocked: false,
    progress: 0,
    requirement: 7
  },
  {
    id: '4',
    title: 'Dedicated Learner',
    description: 'Complete 30 learning sessions',
    icon: 'M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z',
    category: 'mastery',
    unlocked: false,
    progress: 0,
    requirement: 30
  },
  {
    id: '5',
    title: 'Quick Study',
    description: 'Complete 5 sessions in one day',
    icon: 'M13 10V3L4 14h7v7l9-11h-7z',
    category: 'consistency',
    unlocked: false,
    progress: 0,
    requirement: 5
  },
  {
    id: '6',
    title: 'Vocabulary Master',
    description: 'Learn 500 vocabulary words',
    icon: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z',
    category: 'mastery',
    unlocked: false,
    progress: 0,
    requirement: 500
  }
]);

const selectedCategory = ref<string>('all');

const categories = [
  { value: 'all', label: 'All' },
  { value: 'learning', label: 'Learning' },
  { value: 'consistency', label: 'Consistency' },
  { value: 'mastery', label: 'Mastery' },
  { value: 'social', label: 'Social' }
];

const filteredAchievements = computed(() => {
  if (selectedCategory.value === 'all') {
    return achievements.value;
  }
  return achievements.value.filter(a => a.category === selectedCategory.value);
});

const unlockedCount = computed(() => achievements.value.filter(a => a.unlocked).length);

const calculateProgress = (achievement: Achievement): number => {
  if (achievement.unlocked) return 100;
  return Math.round(((achievement.progress || 0) / achievement.requirement) * 100);
};
</script>

<template>
  <div class="space-y-6">
    <!-- Header Stats -->
    <div class="card bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-2xl font-bold">{{ unlockedCount }} / {{ achievements.length }}</h2>
          <p class="text-indigo-100">Achievements Unlocked</p>
        </div>
        <div class="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center">
          <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
          </svg>
        </div>
      </div>
    </div>

    <!-- Category Filter -->
    <div class="flex gap-2 overflow-x-auto pb-2">
      <button
        v-for="cat in categories"
        :key="cat.value"
        @click="selectedCategory = cat.value"
        :class="[
          'px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors',
          selectedCategory === cat.value
            ? 'bg-indigo-600 text-white'
            : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
        ]"
      >
        {{ cat.label }}
      </button>
    </div>

    <!-- Achievements Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <div
        v-for="achievement in filteredAchievements"
        :key="achievement.id"
        :class="[
          'card relative overflow-hidden transition-all duration-300',
          achievement.unlocked ? 'border-indigo-200 bg-indigo-50/50' : 'opacity-75'
        ]"
      >
        <!-- Unlocked Badge -->
        <div
          v-if="achievement.unlocked"
          class="absolute top-2 right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center"
        >
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
        </div>

        <div class="flex items-start gap-4">
          <!-- Icon -->
          <div
            :class="[
              'w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0',
              achievement.unlocked ? 'bg-indigo-600 text-white' : 'bg-gray-200 text-gray-400'
            ]"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d="achievement.icon" />
            </svg>
          </div>

          <!-- Content -->
          <div class="flex-1 min-w-0">
            <h3 class="font-semibold text-gray-900">{{ achievement.title }}</h3>
            <p class="text-sm text-gray-500 mt-1">{{ achievement.description }}</p>

            <!-- Progress -->
            <div v-if="!achievement.unlocked" class="mt-3">
              <div class="flex justify-between text-xs text-gray-500 mb-1">
                <span>Progress</span>
                <span>{{ achievement.progress || 0 }} / {{ achievement.requirement }}</span>
              </div>
              <div class="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  class="h-full bg-indigo-600 rounded-full transition-all duration-300"
                  :style="{ width: `${calculateProgress(achievement)}%` }"
                ></div>
              </div>
            </div>

            <!-- Unlocked Date -->
            <div v-else-if="achievement.unlockedAt" class="mt-2 text-xs text-indigo-600">
              Unlocked: {{ achievement.unlockedAt }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
