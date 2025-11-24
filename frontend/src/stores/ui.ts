import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
  dismissible?: boolean
}

export interface ConfirmDialogOptions {
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  type?: 'danger' | 'warning' | 'info'
}

export const useUIStore = defineStore('ui', () => {
  const sidebarOpen = ref(true)
  const sidebarCollapsed = ref(false)
  const isDarkMode = ref(false)
  const isMobile = ref(false)
  const isOnline = ref(true)
  const toasts = ref<Toast[]>([])
  const confirmDialog = ref<{
    isOpen: boolean
    options: ConfirmDialogOptions | null
    resolve: ((value: boolean) => void) | null
  }>({
    isOpen: false,
    options: null,
    resolve: null
  })
  const globalLoading = ref(false)
  const loadingMessage = ref('')

  const activeToasts = computed(() => toasts.value.slice(-5))

  function toggleSidebar(): void {
    if (isMobile.value) {
      sidebarOpen.value = !sidebarOpen.value
    } else {
      sidebarCollapsed.value = !sidebarCollapsed.value
    }
  }

  function setSidebarOpen(value: boolean): void {
    sidebarOpen.value = value
  }

  function setMobile(value: boolean): void {
    isMobile.value = value
    if (value) {
      sidebarOpen.value = false
    }
  }

  function setOnlineStatus(value: boolean): void {
    isOnline.value = value
    if (!value) {
      showToast({
        type: 'warning',
        title: 'Connection Lost',
        message: 'You are currently offline. Some features may be unavailable.'
      })
    } else {
      showToast({
        type: 'success',
        title: 'Connected',
        message: 'Your connection has been restored.'
      })
    }
  }

  function setDarkMode(value: boolean): void {
    isDarkMode.value = value
    document.documentElement.classList.toggle('dark', value)
  }

  function showToast(options: Omit<Toast, 'id'>): string {
    const id = crypto.randomUUID()
    const toast: Toast = {
      id,
      duration: 5000,
      dismissible: true,
      ...options
    }
    toasts.value.push(toast)

    if (toast.duration && toast.duration > 0) {
      setTimeout(() => {
        dismissToast(id)
      }, toast.duration)
    }

    return id
  }

  function dismissToast(id: string): void {
    const index = toasts.value.findIndex(t => t.id === id)
    if (index !== -1) {
      toasts.value.splice(index, 1)
    }
  }

  function clearAllToasts(): void {
    toasts.value = []
  }

  async function confirm(options: ConfirmDialogOptions): Promise<boolean> {
    return new Promise((resolve) => {
      confirmDialog.value = {
        isOpen: true,
        options,
        resolve
      }
    })
  }

  function resolveConfirm(value: boolean): void {
    if (confirmDialog.value.resolve) {
      confirmDialog.value.resolve(value)
    }
    confirmDialog.value = {
      isOpen: false,
      options: null,
      resolve: null
    }
  }

  function setGlobalLoading(loading: boolean, message?: string): void {
    globalLoading.value = loading
    loadingMessage.value = message || ''
  }

  return {
    sidebarOpen,
    sidebarCollapsed,
    isDarkMode,
    isMobile,
    isOnline,
    toasts,
    confirmDialog,
    globalLoading,
    loadingMessage,
    activeToasts,
    toggleSidebar,
    setSidebarOpen,
    setMobile,
    setOnlineStatus,
    setDarkMode,
    showToast,
    dismissToast,
    clearAllToasts,
    confirm,
    resolveConfirm,
    setGlobalLoading
  }
})
