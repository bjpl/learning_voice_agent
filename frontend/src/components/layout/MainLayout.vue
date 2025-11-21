<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import { useUIStore } from '@/stores/ui'
import AppHeader from './AppHeader.vue'
import AppSidebar from './AppSidebar.vue'
import AppFooter from './AppFooter.vue'
import Toast from '@/components/common/Toast.vue'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'

const uiStore = useUIStore()

function handleResize(): void {
  uiStore.setMobile(window.innerWidth < 1024)
}

function handleOnline(): void {
  uiStore.setOnlineStatus(true)
}

function handleOffline(): void {
  uiStore.setOnlineStatus(false)
}

onMounted(() => {
  handleResize()
  window.addEventListener('resize', handleResize)
  window.addEventListener('online', handleOnline)
  window.addEventListener('offline', handleOffline)

  // Check initial dark mode preference
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    uiStore.setDarkMode(true)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  window.removeEventListener('online', handleOnline)
  window.removeEventListener('offline', handleOffline)
})
</script>

<template>
  <div class="flex h-screen bg-gray-50 dark:bg-gray-950">
    <AppSidebar />

    <div class="flex flex-1 flex-col overflow-hidden">
      <AppHeader />

      <main class="flex-1 overflow-y-auto">
        <slot />
      </main>

      <AppFooter />
    </div>

    <!-- Toast notifications -->
    <Teleport to="body">
      <div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
        <TransitionGroup
          enter-active-class="transition-all duration-300 ease-out"
          enter-from-class="opacity-0 translate-x-8"
          enter-to-class="opacity-100 translate-x-0"
          leave-active-class="transition-all duration-200 ease-in"
          leave-from-class="opacity-100 translate-x-0"
          leave-to-class="opacity-0 translate-x-8"
        >
          <Toast
            v-for="toast in uiStore.activeToasts"
            :key="toast.id"
            :toast="toast"
            @dismiss="uiStore.dismissToast"
          />
        </TransitionGroup>
      </div>
    </Teleport>

    <!-- Confirm dialog -->
    <ConfirmDialog
      v-if="uiStore.confirmDialog.options"
      :is-open="uiStore.confirmDialog.isOpen"
      :title="uiStore.confirmDialog.options.title"
      :message="uiStore.confirmDialog.options.message"
      :confirm-text="uiStore.confirmDialog.options.confirmText"
      :cancel-text="uiStore.confirmDialog.options.cancelText"
      :type="uiStore.confirmDialog.options.type"
      @confirm="uiStore.resolveConfirm(true)"
      @cancel="uiStore.resolveConfirm(false)"
    />
  </div>
</template>
