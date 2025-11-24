<script setup lang="ts">
/**
 * UpdatePrompt - App Update Notification
 *
 * PATTERN: Non-blocking update notification
 * WHY: Inform users of updates without forcing immediate reload
 */
import { usePWA } from '@/composables/usePWA';

const { updateAvailable, applyUpdate, reloadPage } = usePWA();

const handleUpdate = () => {
  applyUpdate();
  // Small delay to allow SW to activate
  setTimeout(() => {
    reloadPage();
  }, 100);
};
</script>

<template>
  <Transition name="slide-up">
    <div v-if="updateAvailable" class="update-prompt">
      <div class="update-content">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
          <path d="M21 10.12h-6.78l2.74-2.82c-2.73-2.7-7.15-2.8-9.88-.1-2.73 2.71-2.73 7.08 0 9.79s7.15 2.71 9.88 0C18.32 15.65 19 14.08 19 12.1h2c0 1.98-.88 4.55-2.64 6.29-3.51 3.48-9.21 3.48-12.72 0-3.5-3.47-3.53-9.11-.02-12.58s9.14-3.47 12.65 0L21 3v7.12zM12.5 8v4.25l3.5 2.08-.72 1.21L11 13V8h1.5z"/>
        </svg>
        <span>A new version is available</span>
      </div>
      <button class="btn-update" @click="handleUpdate">
        Update now
      </button>
    </div>
  </Transition>
</template>

<style scoped>
.update-prompt {
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  background: #1f2937;
  color: white;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  display: flex;
  align-items: center;
  gap: 1rem;
  z-index: 1000;
  max-width: 90vw;
}

@media (min-width: 640px) {
  .update-prompt {
    max-width: 400px;
  }
}

.update-content {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.update-content svg {
  width: 20px;
  height: 20px;
  color: #60a5fa;
  flex-shrink: 0;
}

.btn-update {
  padding: 0.375rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: #1f2937;
  background: #60a5fa;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
  white-space: nowrap;
}

.btn-update:hover {
  background: #93c5fd;
}

/* Transition */
.slide-up-enter-active,
.slide-up-leave-active {
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.slide-up-enter-from,
.slide-up-leave-to {
  transform: translateY(100%);
  opacity: 0;
}
</style>
