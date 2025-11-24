<script setup lang="ts">
/**
 * InstallPrompt - PWA Install Banner Component
 *
 * PATTERN: Non-intrusive install promotion
 * WHY: Encourage app installation without being annoying
 */
import { ref, computed } from 'vue';
import { usePWA } from '@/composables/usePWA';

const { isInstallable, isInstalled, installApp, dismissInstallPrompt } = usePWA();

const isVisible = ref(true);
const isInstalling = ref(false);

const showPrompt = computed(() => {
  return isInstallable.value && isVisible.value && !isInstalled.value;
});

const handleInstall = async () => {
  isInstalling.value = true;
  try {
    const result = await installApp();
    if (result.outcome === 'accepted') {
      isVisible.value = false;
    }
  } finally {
    isInstalling.value = false;
  }
};

const handleDismiss = () => {
  dismissInstallPrompt();
  isVisible.value = false;
  // Store dismissal in localStorage to not show again for a while
  localStorage.setItem('pwa-install-dismissed', Date.now().toString());
};

// Check if recently dismissed
const checkDismissed = () => {
  const dismissed = localStorage.getItem('pwa-install-dismissed');
  if (dismissed) {
    const dismissedAt = parseInt(dismissed, 10);
    const daysSinceDismissed = (Date.now() - dismissedAt) / (1000 * 60 * 60 * 24);
    // Show again after 7 days
    if (daysSinceDismissed < 7) {
      isVisible.value = false;
    }
  }
};

checkDismissed();
</script>

<template>
  <Transition name="slide-up">
    <div v-if="showPrompt" class="install-prompt">
      <div class="install-content">
        <div class="install-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
          </svg>
        </div>
        <div class="install-text">
          <h3>Install Learning Voice Agent</h3>
          <p>Add to your home screen for quick access and offline support</p>
        </div>
      </div>
      <div class="install-actions">
        <button
          class="btn-dismiss"
          @click="handleDismiss"
          :disabled="isInstalling"
        >
          Not now
        </button>
        <button
          class="btn-install"
          @click="handleInstall"
          :disabled="isInstalling"
        >
          <span v-if="isInstalling">Installing...</span>
          <span v-else>Install</span>
        </button>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.install-prompt {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  border-top: 1px solid #e5e7eb;
  box-shadow: 0 -4px 6px -1px rgb(0 0 0 / 0.1);
  padding: 1rem;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

@media (min-width: 640px) {
  .install-prompt {
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
  }
}

.install-content {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.install-icon {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  background: #eff6ff;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #3b82f6;
}

.install-icon svg {
  width: 28px;
  height: 28px;
}

.install-text h3 {
  font-size: 1rem;
  font-weight: 600;
  color: #111827;
  margin: 0;
}

.install-text p {
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0.25rem 0 0;
}

.install-actions {
  display: flex;
  gap: 0.75rem;
  flex-shrink: 0;
}

.btn-dismiss {
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #6b7280;
  background: transparent;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-dismiss:hover {
  background: #f9fafb;
}

.btn-dismiss:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-install {
  padding: 0.5rem 1.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: white;
  background: #3b82f6;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-install:hover {
  background: #2563eb;
}

.btn-install:disabled {
  opacity: 0.7;
  cursor: not-allowed;
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

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .install-prompt {
    background: #1f2937;
    border-top-color: #374151;
  }

  .install-icon {
    background: #1e3a5f;
  }

  .install-text h3 {
    color: #f9fafb;
  }

  .install-text p {
    color: #9ca3af;
  }

  .btn-dismiss {
    color: #9ca3af;
    border-color: #4b5563;
  }

  .btn-dismiss:hover {
    background: #374151;
  }
}
</style>
