<script setup lang="ts">
interface Props {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  color?: string
  label?: string
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  color: 'text-blue-600',
  label: 'Loading...'
})

const sizeClasses: Record<string, string> = {
  sm: 'h-4 w-4',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
  xl: 'h-16 w-16'
}
</script>

<template>
  <div class="flex flex-col items-center justify-center gap-2" role="status" aria-live="polite">
    <svg
      :class="[sizeClasses[props.size], props.color, 'animate-spin']"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        class="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        stroke-width="4"
      />
      <path
        class="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
    <span v-if="label" class="text-sm text-gray-500 dark:text-gray-400">
      {{ label }}
    </span>
    <span class="sr-only">{{ label }}</span>
  </div>
</template>
