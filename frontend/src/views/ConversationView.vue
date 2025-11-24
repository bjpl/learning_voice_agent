<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue';
import { v4 as uuidv4 } from 'uuid';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const messages = ref<Message[]>([]);
const inputText = ref('');
const isRecording = ref(false);
const isProcessing = ref(false);
const messagesContainer = ref<HTMLElement | null>(null);

const scrollToBottom = async (): Promise<void> => {
  await nextTick();
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
};

const sendMessage = async (): Promise<void> => {
  const text = inputText.value.trim();
  if (!text || isProcessing.value) return;

  const userMessage: Message = {
    id: uuidv4(),
    role: 'user',
    content: text,
    timestamp: new Date()
  };

  messages.value.push(userMessage);
  inputText.value = '';
  isProcessing.value = true;
  await scrollToBottom();

  // Placeholder response - will be replaced with actual API call
  setTimeout(() => {
    const assistantMessage: Message = {
      id: uuidv4(),
      role: 'assistant',
      content: 'This is a placeholder response. The voice learning agent backend will provide actual responses.',
      timestamp: new Date()
    };
    messages.value.push(assistantMessage);
    isProcessing.value = false;
    scrollToBottom();
  }, 1000);
};

const toggleRecording = (): void => {
  isRecording.value = !isRecording.value;
  // Voice recording will be implemented with Web Audio API
};

const handleKeydown = (event: KeyboardEvent): void => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
};

onMounted(() => {
  messages.value.push({
    id: uuidv4(),
    role: 'assistant',
    content: 'Hello! I\'m your voice learning assistant. How can I help you learn today?',
    timestamp: new Date()
  });
});
</script>

<template>
  <div class="flex flex-col h-[calc(100vh-8rem)]">
    <!-- Messages Area -->
    <div
      ref="messagesContainer"
      class="flex-1 overflow-y-auto scrollbar-thin space-y-4 pb-4"
    >
      <div
        v-for="message in messages"
        :key="message.id"
        :class="[
          'flex',
          message.role === 'user' ? 'justify-end' : 'justify-start'
        ]"
      >
        <div
          :class="[
            'max-w-[70%] rounded-2xl px-4 py-3',
            message.role === 'user'
              ? 'bg-indigo-600 text-white'
              : 'bg-white border border-gray-200 text-gray-900'
          ]"
        >
          <p class="whitespace-pre-wrap">{{ message.content }}</p>
          <span
            :class="[
              'text-xs mt-1 block',
              message.role === 'user' ? 'text-indigo-200' : 'text-gray-400'
            ]"
          >
            {{ message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}
          </span>
        </div>
      </div>

      <!-- Typing Indicator -->
      <div v-if="isProcessing" class="flex justify-start">
        <div class="bg-white border border-gray-200 rounded-2xl px-4 py-3">
          <div class="flex space-x-2">
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="border-t border-gray-200 bg-white p-4 rounded-b-xl">
      <div class="flex items-end gap-3">
        <!-- Voice Record Button -->
        <button
          @click="toggleRecording"
          :class="[
            'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-colors',
            isRecording
              ? 'bg-red-500 text-white animate-pulse'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          ]"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        </button>

        <!-- Text Input -->
        <div class="flex-1 relative">
          <textarea
            v-model="inputText"
            @keydown="handleKeydown"
            placeholder="Type a message or use voice..."
            rows="1"
            class="input resize-none pr-12"
          ></textarea>
        </div>

        <!-- Send Button -->
        <button
          @click="sendMessage"
          :disabled="!inputText.trim() || isProcessing"
          class="btn-primary flex-shrink-0"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>
