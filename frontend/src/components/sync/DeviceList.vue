<script setup lang="ts">
/**
 * DeviceList - Display and manage registered devices
 *
 * PATTERN: List with action buttons
 * WHY: Allow users to see and manage where their data is synced
 */
import { ref, computed, onMounted } from 'vue';
import { useSync } from '@/composables/useSync';
import type { DeviceInfo, DeviceType } from '@/types/sync';

const { devices, listDevices, removeDevice, registerDevice } = useSync();

const showConfirmRemove = ref(false);
const deviceToRemove = ref<DeviceInfo | null>(null);

onMounted(() => {
  listDevices();
});

const sortedDevices = computed(() => {
  return [...devices.value].sort((a, b) => {
    // Current device first
    if (a.isCurrentDevice) return -1;
    if (b.isCurrentDevice) return 1;
    // Then by last seen
    return new Date(b.lastSeen).getTime() - new Date(a.lastSeen).getTime();
  });
});

const getPlatformIcon = (deviceType: DeviceType): string => {
  switch (deviceType) {
    case 'mobile':
      return 'M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z';
    case 'tablet':
      return 'M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z';
    case 'desktop':
      return 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z';
    default:
      return 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9';
  }
};

const getPlatformLabel = (deviceType: DeviceType): string => {
  switch (deviceType) {
    case 'mobile':
      return 'Mobile';
    case 'tablet':
      return 'Tablet';
    case 'desktop':
      return 'Desktop';
    default:
      return 'Web';
  }
};

const formatLastSeen = (dateStr: string): string => {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)} days ago`;
  return date.toLocaleDateString();
};

const confirmRemove = (device: DeviceInfo) => {
  deviceToRemove.value = device;
  showConfirmRemove.value = true;
};

const handleRemove = () => {
  if (deviceToRemove.value) {
    removeDevice(deviceToRemove.value.id);
    showConfirmRemove.value = false;
    deviceToRemove.value = null;
  }
};

const cancelRemove = () => {
  showConfirmRemove.value = false;
  deviceToRemove.value = null;
};
</script>

<template>
  <div class="rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            Registered Devices
          </h3>
          <p class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            Devices that have synced your data
          </p>
        </div>
        <span class="px-2.5 py-1 text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full">
          {{ devices.length }} device{{ devices.length !== 1 ? 's' : '' }}
        </span>
      </div>
    </div>

    <!-- Device List -->
    <div class="divide-y divide-gray-200 dark:divide-gray-700">
      <div
        v-for="device in sortedDevices"
        :key="device.id"
        class="px-6 py-4 flex items-center gap-4"
      >
        <!-- Platform Icon -->
        <div
          :class="[
            'p-3 rounded-lg',
            device.isCurrentDevice
              ? 'bg-blue-100 dark:bg-blue-900/30'
              : 'bg-gray-100 dark:bg-gray-700'
          ]"
        >
          <svg
            :class="[
              'w-6 h-6',
              device.isCurrentDevice
                ? 'text-blue-600 dark:text-blue-400'
                : 'text-gray-500 dark:text-gray-400'
            ]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              :d="getPlatformIcon(device.type)"
            />
          </svg>
        </div>

        <!-- Device Info -->
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            <span class="font-medium text-gray-900 dark:text-white truncate">
              {{ device.name }}
            </span>
            <span
              v-if="device.isCurrentDevice"
              class="px-2 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full"
            >
              Current
            </span>
          </div>
          <div class="flex items-center gap-3 mt-1">
            <span class="text-sm text-gray-500 dark:text-gray-400">
              {{ getPlatformLabel(device.type) }} - {{ device.platform }}
            </span>
            <span class="text-gray-300 dark:text-gray-600">|</span>
            <span class="text-sm text-gray-500 dark:text-gray-400">
              Last active {{ formatLastSeen(device.lastSeen) }}
            </span>
          </div>
        </div>

        <!-- Actions -->
        <button
          v-if="!device.isCurrentDevice"
          @click="confirmRemove(device)"
          class="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          title="Remove device"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>

      <!-- Empty State -->
      <div
        v-if="devices.length === 0"
        class="px-6 py-12 text-center"
      >
        <svg class="w-12 h-12 mx-auto text-gray-300 dark:text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
        <p class="text-gray-500 dark:text-gray-400 mb-3">No devices registered yet</p>
        <button
          @click="registerDevice"
          class="text-blue-600 dark:text-blue-400 hover:underline text-sm font-medium"
        >
          Register this device
        </button>
      </div>
    </div>

    <!-- Confirm Remove Dialog -->
    <Teleport to="body">
      <Transition
        enter-active-class="transition-opacity duration-200 ease-out"
        enter-from-class="opacity-0"
        enter-to-class="opacity-100"
        leave-active-class="transition-opacity duration-150 ease-in"
        leave-from-class="opacity-100"
        leave-to-class="opacity-0"
      >
        <div
          v-if="showConfirmRemove"
          class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          @click.self="cancelRemove"
        >
          <Transition
            enter-active-class="transition-all duration-200 ease-out"
            enter-from-class="opacity-0 scale-95"
            enter-to-class="opacity-100 scale-100"
            leave-active-class="transition-all duration-150 ease-in"
            leave-from-class="opacity-100 scale-100"
            leave-to-class="opacity-0 scale-95"
          >
            <div
              v-if="showConfirmRemove"
              class="w-full max-w-md rounded-xl bg-white dark:bg-gray-800 shadow-xl"
            >
              <div class="p-6">
                <div class="flex items-start gap-4">
                  <div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
                    <svg class="h-6 w-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <div class="flex-1">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
                      Remove Device
                    </h3>
                    <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
                      Are you sure you want to remove <strong>{{ deviceToRemove?.name }}</strong>?
                      This device will no longer be able to sync data.
                    </p>
                  </div>
                </div>
              </div>
              <div class="flex justify-end gap-3 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
                <button
                  @click="cancelRemove"
                  class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  @click="handleRemove"
                  class="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  Remove Device
                </button>
              </div>
            </div>
          </Transition>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>
