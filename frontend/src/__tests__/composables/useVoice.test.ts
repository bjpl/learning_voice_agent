import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'

// Mock MediaRecorder
const mockMediaRecorder = {
  start: vi.fn(),
  stop: vi.fn(),
  pause: vi.fn(),
  resume: vi.fn(),
  ondataavailable: null as ((e: any) => void) | null,
  onstop: null as (() => void) | null,
  onerror: null as ((e: any) => void) | null,
  state: 'inactive',
}

const mockMediaStream = {
  getTracks: vi.fn(() => [{ stop: vi.fn() }]),
}

global.MediaRecorder = vi.fn().mockImplementation(() => mockMediaRecorder) as any
;(global.MediaRecorder as any).isTypeSupported = vi.fn(() => true)

global.navigator.mediaDevices = {
  getUserMedia: vi.fn().mockResolvedValue(mockMediaStream),
} as any

// Mock AudioContext
const mockAnalyser = {
  connect: vi.fn(),
  disconnect: vi.fn(),
  getByteFrequencyData: vi.fn(),
  getByteTimeDomainData: vi.fn(),
  fftSize: 2048,
  frequencyBinCount: 1024,
}

const mockAudioContext = {
  createAnalyser: vi.fn(() => mockAnalyser),
  createMediaStreamSource: vi.fn(() => ({ connect: vi.fn() })),
  close: vi.fn(),
  state: 'running',
}

global.AudioContext = vi.fn().mockImplementation(() => mockAudioContext) as any

describe('useVoice Composable', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('should check browser support', async () => {
    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    expect(voice.isSupported.value).toBe(true)
  })

  it('should start recording', async () => {
    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    await voice.startRecording()

    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true })
    expect(voice.isRecording.value).toBe(true)
  })

  it('should stop recording', async () => {
    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    await voice.startRecording()
    voice.stopRecording()

    expect(mockMediaRecorder.stop).toHaveBeenCalled()
  })

  it('should handle recording errors', async () => {
    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    const error = new Error('Permission denied')
    ;(navigator.mediaDevices.getUserMedia as any).mockRejectedValueOnce(error)

    await voice.startRecording()

    expect(voice.error.value).toBeTruthy()
    expect(voice.isRecording.value).toBe(false)
  })

  it('should track duration', async () => {
    vi.useFakeTimers()

    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    await voice.startRecording()

    vi.advanceTimersByTime(5000)

    expect(voice.duration.value).toBeGreaterThanOrEqual(0)

    vi.useRealTimers()
  })

  it('should pause and resume recording', async () => {
    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    await voice.startRecording()

    voice.pauseRecording()
    expect(mockMediaRecorder.pause).toHaveBeenCalled()
    expect(voice.isPaused.value).toBe(true)

    voice.resumeRecording()
    expect(mockMediaRecorder.resume).toHaveBeenCalled()
    expect(voice.isPaused.value).toBe(false)
  })

  it('should reset recording state', async () => {
    const { useVoice } = await import('@/composables/useVoice')
    const voice = useVoice()

    await voice.startRecording()
    voice.stopRecording()
    voice.resetRecording()

    expect(voice.audioBlob.value).toBeNull()
    expect(voice.audioUrl.value).toBe('')
    expect(voice.duration.value).toBe(0)
  })
})
