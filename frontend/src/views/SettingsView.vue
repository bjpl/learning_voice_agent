<script setup lang="ts">
import { ref, reactive } from 'vue';

interface UserSettings {
  profile: {
    name: string;
    email: string;
    language: string;
    timezone: string;
  };
  learning: {
    dailyGoal: number;
    difficultyLevel: 'beginner' | 'intermediate' | 'advanced';
    autoPlayAudio: boolean;
    showTranslations: boolean;
  };
  notifications: {
    emailReminders: boolean;
    pushNotifications: boolean;
    streakReminders: boolean;
    weeklyReport: boolean;
  };
  audio: {
    microphoneEnabled: boolean;
    speakerEnabled: boolean;
    voiceSpeed: number;
  };
}

const settings = reactive<UserSettings>({
  profile: {
    name: '',
    email: '',
    language: 'en',
    timezone: 'UTC'
  },
  learning: {
    dailyGoal: 15,
    difficultyLevel: 'intermediate',
    autoPlayAudio: true,
    showTranslations: true
  },
  notifications: {
    emailReminders: true,
    pushNotifications: false,
    streakReminders: true,
    weeklyReport: true
  },
  audio: {
    microphoneEnabled: true,
    speakerEnabled: true,
    voiceSpeed: 1.0
  }
});

const isSaving = ref(false);
const saveMessage = ref<{ type: 'success' | 'error'; text: string } | null>(null);

const languages = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' }
];

const difficultyLevels = [
  { value: 'beginner', label: 'Beginner' },
  { value: 'intermediate', label: 'Intermediate' },
  { value: 'advanced', label: 'Advanced' }
];

const saveSettings = async (): Promise<void> => {
  isSaving.value = true;
  saveMessage.value = null;

  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    saveMessage.value = { type: 'success', text: 'Settings saved successfully!' };
  } catch {
    saveMessage.value = { type: 'error', text: 'Failed to save settings. Please try again.' };
  } finally {
    isSaving.value = false;
    setTimeout(() => {
      saveMessage.value = null;
    }, 3000);
  }
};
</script>

<template>
  <div class="max-w-3xl space-y-6">
    <!-- Save Message -->
    <div
      v-if="saveMessage"
      :class="[
        'p-4 rounded-lg',
        saveMessage.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
      ]"
    >
      {{ saveMessage.text }}
    </div>

    <!-- Profile Settings -->
    <div class="card">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Profile</h2>
      <div class="space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="label">Name</label>
            <input v-model="settings.profile.name" type="text" class="input" placeholder="Your name">
          </div>
          <div>
            <label class="label">Email</label>
            <input v-model="settings.profile.email" type="email" class="input" placeholder="your@email.com">
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="label">Learning Language</label>
            <select v-model="settings.profile.language" class="input">
              <option v-for="lang in languages" :key="lang.value" :value="lang.value">
                {{ lang.label }}
              </option>
            </select>
          </div>
          <div>
            <label class="label">Timezone</label>
            <input v-model="settings.profile.timezone" type="text" class="input" placeholder="UTC">
          </div>
        </div>
      </div>
    </div>

    <!-- Learning Settings -->
    <div class="card">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Learning Preferences</h2>
      <div class="space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="label">Daily Goal (minutes)</label>
            <input v-model.number="settings.learning.dailyGoal" type="number" class="input" min="5" max="120">
          </div>
          <div>
            <label class="label">Difficulty Level</label>
            <select v-model="settings.learning.difficultyLevel" class="input">
              <option v-for="level in difficultyLevels" :key="level.value" :value="level.value">
                {{ level.label }}
              </option>
            </select>
          </div>
        </div>
        <div class="space-y-3">
          <label class="flex items-center gap-3">
            <input v-model="settings.learning.autoPlayAudio" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
            <span class="text-sm text-gray-700">Auto-play audio for new words</span>
          </label>
          <label class="flex items-center gap-3">
            <input v-model="settings.learning.showTranslations" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
            <span class="text-sm text-gray-700">Show translations by default</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Audio Settings -->
    <div class="card">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Audio Settings</h2>
      <div class="space-y-4">
        <div class="space-y-3">
          <label class="flex items-center gap-3">
            <input v-model="settings.audio.microphoneEnabled" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
            <span class="text-sm text-gray-700">Enable microphone for voice input</span>
          </label>
          <label class="flex items-center gap-3">
            <input v-model="settings.audio.speakerEnabled" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
            <span class="text-sm text-gray-700">Enable audio playback</span>
          </label>
        </div>
        <div>
          <label class="label">Voice Speed: {{ settings.audio.voiceSpeed }}x</label>
          <input
            v-model.number="settings.audio.voiceSpeed"
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          >
        </div>
      </div>
    </div>

    <!-- Notification Settings -->
    <div class="card">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Notifications</h2>
      <div class="space-y-3">
        <label class="flex items-center gap-3">
          <input v-model="settings.notifications.emailReminders" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
          <span class="text-sm text-gray-700">Email reminders to practice</span>
        </label>
        <label class="flex items-center gap-3">
          <input v-model="settings.notifications.pushNotifications" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
          <span class="text-sm text-gray-700">Push notifications</span>
        </label>
        <label class="flex items-center gap-3">
          <input v-model="settings.notifications.streakReminders" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
          <span class="text-sm text-gray-700">Streak maintenance reminders</span>
        </label>
        <label class="flex items-center gap-3">
          <input v-model="settings.notifications.weeklyReport" type="checkbox" class="w-4 h-4 text-indigo-600 rounded">
          <span class="text-sm text-gray-700">Weekly progress report</span>
        </label>
      </div>
    </div>

    <!-- Save Button -->
    <div class="flex justify-end">
      <button
        @click="saveSettings"
        :disabled="isSaving"
        class="btn-primary"
      >
        <svg v-if="isSaving" class="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        {{ isSaving ? 'Saving...' : 'Save Settings' }}
      </button>
    </div>
  </div>
</template>
