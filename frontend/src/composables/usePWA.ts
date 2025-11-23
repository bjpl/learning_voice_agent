/**
 * PWA Composable - Install Prompt and Update Handling
 *
 * PATTERN: Reactive PWA state management
 * WHY: Centralized PWA functionality for install prompts and updates
 */
import { ref, computed, onMounted, onUnmounted } from 'vue';

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed';
    platform: string;
  }>;
  prompt(): Promise<void>;
}

// PWA state
const deferredPrompt = ref<BeforeInstallPromptEvent | null>(null);
const isInstallable = ref(false);
const isInstalled = ref(false);
const isOnline = ref(navigator.onLine);
const needsRefresh = ref(false);
const updateAvailable = ref(false);

// Check if app is already installed
const checkInstalled = () => {
  // Check display-mode
  if (window.matchMedia('(display-mode: standalone)').matches) {
    isInstalled.value = true;
    return true;
  }
  // Check iOS standalone
  if ((navigator as any).standalone === true) {
    isInstalled.value = true;
    return true;
  }
  return false;
};

export function usePWA() {
  let registration: ServiceWorkerRegistration | null = null;

  // Handle install prompt
  const handleBeforeInstallPrompt = (e: Event) => {
    e.preventDefault();
    deferredPrompt.value = e as BeforeInstallPromptEvent;
    isInstallable.value = true;
  };

  // Handle app installed
  const handleAppInstalled = () => {
    deferredPrompt.value = null;
    isInstallable.value = false;
    isInstalled.value = true;
  };

  // Handle online/offline
  const handleOnline = () => {
    isOnline.value = true;
  };

  const handleOffline = () => {
    isOnline.value = false;
  };

  // Trigger install prompt
  const installApp = async () => {
    if (!deferredPrompt.value) {
      return { outcome: 'unavailable' };
    }

    await deferredPrompt.value.prompt();
    const result = await deferredPrompt.value.userChoice;

    if (result.outcome === 'accepted') {
      deferredPrompt.value = null;
      isInstallable.value = false;
    }

    return result;
  };

  // Dismiss install prompt
  const dismissInstallPrompt = () => {
    deferredPrompt.value = null;
    isInstallable.value = false;
  };

  // Check for service worker updates
  const checkForUpdates = async () => {
    if (registration) {
      await registration.update();
    }
  };

  // Apply pending update
  const applyUpdate = () => {
    if (registration?.waiting) {
      registration.waiting.postMessage({ type: 'SKIP_WAITING' });
      needsRefresh.value = true;
    }
  };

  // Reload page to get new version
  const reloadPage = () => {
    window.location.reload();
  };

  // Register service worker and handle updates
  const registerSW = async () => {
    if ('serviceWorker' in navigator) {
      try {
        registration = await navigator.serviceWorker.register('/sw.js', {
          scope: '/'
        });

        // Check for updates on registration
        registration.addEventListener('updatefound', () => {
          const newWorker = registration?.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                updateAvailable.value = true;
              }
            });
          }
        });

        // Handle controller change (new SW activated)
        navigator.serviceWorker.addEventListener('controllerchange', () => {
          if (needsRefresh.value) {
            window.location.reload();
          }
        });

      } catch (error) {
        console.error('Service worker registration failed:', error);
      }
    }
  };

  onMounted(() => {
    // Check if already installed
    checkInstalled();

    // Listen for install prompt
    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    // Listen for online/offline
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Register service worker
    registerSW();
  });

  onUnmounted(() => {
    window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.removeEventListener('appinstalled', handleAppInstalled);
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
  });

  return {
    // State
    isInstallable: computed(() => isInstallable.value),
    isInstalled: computed(() => isInstalled.value),
    isOnline: computed(() => isOnline.value),
    updateAvailable: computed(() => updateAvailable.value),
    needsRefresh: computed(() => needsRefresh.value),

    // Actions
    installApp,
    dismissInstallPrompt,
    checkForUpdates,
    applyUpdate,
    reloadPage
  };
}
