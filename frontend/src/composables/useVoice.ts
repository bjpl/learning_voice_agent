/**
 * Voice Recording Composable
 * Provides voice recording functionality using MediaRecorder API
 * with state management and audio processing
 */

import { ref, computed, onUnmounted, type Ref } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface VoiceState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  audioBlob: Blob | null;
  audioUrl: string | null;
  error: string | null;
}

export interface VoiceRecordingOptions {
  mimeType?: string;
  audioBitsPerSecond?: number;
  maxDuration?: number;
  onDataAvailable?: (data: Blob) => void;
  onStop?: (blob: Blob) => void;
  onError?: (error: Error) => void;
}

export interface AudioAnalyzerData {
  frequencyData: Uint8Array;
  timeDomainData: Uint8Array;
  volume: number;
}

export interface UseVoiceReturn {
  // State
  isRecording: Ref<boolean>;
  isPaused: Ref<boolean>;
  duration: Ref<number>;
  audioBlob: Ref<Blob | null>;
  audioUrl: Ref<string | null>;
  error: Ref<string | null>;
  isSupported: Ref<boolean>;
  analyzerData: Ref<AudioAnalyzerData | null>;

  // Computed
  formattedDuration: Ref<string>;

  // Methods
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<Blob | null>;
  pauseRecording: () => void;
  resumeRecording: () => void;
  resetRecording: () => void;
  getAnalyzerData: () => AudioAnalyzerData | null;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_MIME_TYPE = 'audio/webm;codecs=opus';
const FALLBACK_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4',
  'audio/wav'
];

// ============================================================================
// Composable Implementation
// ============================================================================

export function useVoice(options: VoiceRecordingOptions = {}): UseVoiceReturn {
  // Reactive state
  const isRecording = ref(false);
  const isPaused = ref(false);
  const duration = ref(0);
  const audioBlob = ref<Blob | null>(null);
  const audioUrl = ref<string | null>(null);
  const error = ref<string | null>(null);
  const isSupported = ref(checkSupport());
  const analyzerData = ref<AudioAnalyzerData | null>(null);

  // Internal state
  let mediaRecorder: MediaRecorder | null = null;
  let audioChunks: Blob[] = [];
  let durationInterval: ReturnType<typeof setInterval> | null = null;
  let audioContext: AudioContext | null = null;
  let analyserNode: AnalyserNode | null = null;
  let mediaStream: MediaStream | null = null;
  let animationFrameId: number | null = null;

  // ============================================================================
  // Computed Properties
  // ============================================================================

  const formattedDuration = computed(() => {
    const totalSeconds = Math.floor(duration.value);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  });

  // ============================================================================
  // Support Check
  // ============================================================================

  function checkSupport(): boolean {
    return !!(
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia &&
      window.MediaRecorder
    );
  }

  function getSupportedMimeType(): string {
    const preferred = options.mimeType || DEFAULT_MIME_TYPE;
    if (MediaRecorder.isTypeSupported(preferred)) {
      return preferred;
    }

    for (const mimeType of FALLBACK_MIME_TYPES) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        return mimeType;
      }
    }

    return '';
  }

  // ============================================================================
  // Audio Analysis
  // ============================================================================

  function setupAudioAnalyzer(stream: MediaStream): void {
    try {
      audioContext = new AudioContext();
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 256;
      analyserNode.smoothingTimeConstant = 0.8;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyserNode);

      updateAnalyzerData();
    } catch (err) {
      console.warn('Failed to setup audio analyzer:', err);
    }
  }

  function updateAnalyzerData(): void {
    if (!analyserNode || !isRecording.value) return;

    const frequencyData = new Uint8Array(analyserNode.frequencyBinCount);
    const timeDomainData = new Uint8Array(analyserNode.fftSize);

    analyserNode.getByteFrequencyData(frequencyData);
    analyserNode.getByteTimeDomainData(timeDomainData);

    // Calculate volume (RMS)
    let sum = 0;
    for (let i = 0; i < timeDomainData.length; i++) {
      const normalized = (timeDomainData[i] - 128) / 128;
      sum += normalized * normalized;
    }
    const volume = Math.sqrt(sum / timeDomainData.length);

    analyzerData.value = {
      frequencyData,
      timeDomainData,
      volume
    };

    animationFrameId = requestAnimationFrame(updateAnalyzerData);
  }

  function getAnalyzerData(): AudioAnalyzerData | null {
    return analyzerData.value;
  }

  // ============================================================================
  // Recording Methods
  // ============================================================================

  async function startRecording(): Promise<void> {
    if (!isSupported.value) {
      error.value = 'Voice recording is not supported in this browser';
      options.onError?.(new Error(error.value));
      return;
    }

    try {
      error.value = null;
      audioChunks = [];

      // Get microphone access
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Setup audio analyzer for visualization
      setupAudioAnalyzer(mediaStream);

      // Get supported MIME type
      const mimeType = getSupportedMimeType();
      const recorderOptions: MediaRecorderOptions = {};

      if (mimeType) {
        recorderOptions.mimeType = mimeType;
      }
      if (options.audioBitsPerSecond) {
        recorderOptions.audioBitsPerSecond = options.audioBitsPerSecond;
      }

      // Create MediaRecorder
      mediaRecorder = new MediaRecorder(mediaStream, recorderOptions);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
          options.onDataAvailable?.(event.data);
        }
      };

      mediaRecorder.onerror = (event) => {
        const recorderError = new Error('Recording error occurred');
        error.value = recorderError.message;
        options.onError?.(recorderError);
        stopRecording();
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
        audioBlob.value = blob;

        // Revoke previous URL if exists
        if (audioUrl.value) {
          URL.revokeObjectURL(audioUrl.value);
        }
        audioUrl.value = URL.createObjectURL(blob);

        options.onStop?.(blob);
      };

      // Start recording with timeslice for live data
      mediaRecorder.start(100);
      isRecording.value = true;
      isPaused.value = false;
      duration.value = 0;

      // Start duration timer
      durationInterval = setInterval(() => {
        if (!isPaused.value) {
          duration.value += 0.1;

          // Check max duration
          if (options.maxDuration && duration.value >= options.maxDuration) {
            stopRecording();
          }
        }
      }, 100);

    } catch (err) {
      const recordError = err instanceof Error ? err : new Error('Failed to start recording');
      error.value = recordError.message;
      options.onError?.(recordError);
    }
  }

  async function stopRecording(): Promise<Blob | null> {
    return new Promise((resolve) => {
      if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        resolve(audioBlob.value);
        return;
      }

      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder?.mimeType || 'audio/webm';
        const blob = new Blob(audioChunks, { type: mimeType });
        audioBlob.value = blob;

        if (audioUrl.value) {
          URL.revokeObjectURL(audioUrl.value);
        }
        audioUrl.value = URL.createObjectURL(blob);

        options.onStop?.(blob);
        resolve(blob);
      };

      mediaRecorder.stop();
      isRecording.value = false;
      isPaused.value = false;

      // Cleanup
      if (durationInterval) {
        clearInterval(durationInterval);
        durationInterval = null;
      }

      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }

      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
      }

      if (audioContext) {
        audioContext.close();
        audioContext = null;
        analyserNode = null;
      }
    });
  }

  function pauseRecording(): void {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.pause();
      isPaused.value = true;
    }
  }

  function resumeRecording(): void {
    if (mediaRecorder && mediaRecorder.state === 'paused') {
      mediaRecorder.resume();
      isPaused.value = false;
    }
  }

  function resetRecording(): void {
    stopRecording();

    if (audioUrl.value) {
      URL.revokeObjectURL(audioUrl.value);
    }

    audioBlob.value = null;
    audioUrl.value = null;
    duration.value = 0;
    error.value = null;
    analyzerData.value = null;
    audioChunks = [];
  }

  // ============================================================================
  // Cleanup
  // ============================================================================

  onUnmounted(() => {
    resetRecording();
  });

  // ============================================================================
  // Return
  // ============================================================================

  return {
    // State
    isRecording,
    isPaused,
    duration,
    audioBlob,
    audioUrl,
    error,
    isSupported,
    analyzerData,

    // Computed
    formattedDuration,

    // Methods
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    resetRecording,
    getAnalyzerData
  };
}

export default useVoice;
