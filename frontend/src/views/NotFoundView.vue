<script setup lang="ts">
import { useRouter } from 'vue-router';

const router = useRouter();

const goHome = (): void => {
  router.push('/');
};

const goBack = (): void => {
  router.back();
};
</script>

<template>
  <div class="flex flex-col items-center justify-center min-h-[60vh] text-center">
    <div class="text-6xl font-bold text-gray-200 mb-4">404</div>
    <h1 class="text-2xl font-semibold text-gray-900 mb-2">Page Not Found</h1>
    <p class="text-gray-500 mb-8 max-w-md">
      The page you're looking for doesn't exist or has been moved.
    </p>
    <div class="flex gap-4">
      <button @click="goBack" class="btn-secondary">
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        Go Back
      </button>
      <button @click="goHome" class="btn-primary">
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
        Go Home
      </button>
    </div>
  </div>
</template>
