<script setup lang="ts">
/**
 * OfflineIndicator - Network Status Banner
 *
 * PATTERN: Non-intrusive offline notification
 * WHY: Inform users when they're offline without disrupting workflow
 */
import { usePWA } from '@/composables/usePWA';

const { isOnline } = usePWA();
</script>

<template>
  <Transition name="slide-down">
    <div v-if="!isOnline" class="offline-indicator">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M24 8.98C20.93 5.9 16.69 4 12 4c-1.98 0-3.86.35-5.61.98l7.62 7.62 9.99-3.62zm-13.03 7.06l2.62 2.62c.39.39 1.02.39 1.41 0l2.62-2.62a5.003 5.003 0 00-6.65 0zm-8.56 2.24l19.29-19.29c.39-.39.39-1.02 0-1.41a.9959.9959 0 00-1.41 0L.29 17.87c-.39.39-.39 1.02 0 1.41.39.39 1.02.39 1.41 0z"/>
      </svg>
      <span>You're offline. Some features may be limited.</span>
    </div>
  </Transition>
</template>

<style scoped>
.offline-indicator {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: #fef3c7;
  color: #92400e;
  padding: 0.5rem 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  z-index: 1001;
  border-bottom: 1px solid #fcd34d;
}

.offline-indicator svg {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}

/* Transition */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.slide-down-enter-from,
.slide-down-leave-to {
  transform: translateY(-100%);
  opacity: 0;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .offline-indicator {
    background: #78350f;
    color: #fef3c7;
    border-bottom-color: #92400e;
  }
}
</style>
