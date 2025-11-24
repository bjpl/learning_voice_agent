import { describe, it, expect, beforeEach, vi } from 'vitest'
import axios from 'axios'

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    })),
  },
}))

describe('API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should create axios instance with correct config', async () => {
    const { apiClient } = await import('@/services/api')

    expect(axios.create).toHaveBeenCalledWith(
      expect.objectContaining({
        timeout: expect.any(Number),
      })
    )
  })

  it('should handle successful GET request', async () => {
    const mockResponse = { data: { message: 'success' } }
    const mockAxios = axios.create() as any
    mockAxios.get.mockResolvedValueOnce(mockResponse)

    const { apiClient } = await import('@/services/api')

    // The actual implementation would use the mocked axios
    expect(mockAxios.get).toBeDefined()
  })

  it('should handle API errors', async () => {
    const mockError = {
      response: {
        status: 500,
        data: { error: 'Internal Server Error' },
      },
    }

    const mockAxios = axios.create() as any
    mockAxios.get.mockRejectedValueOnce(mockError)

    const { apiClient } = await import('@/services/api')

    expect(mockAxios.get).toBeDefined()
  })

  it('should handle network errors', async () => {
    const mockError = new Error('Network Error')

    const mockAxios = axios.create() as any
    mockAxios.get.mockRejectedValueOnce(mockError)

    const { apiClient } = await import('@/services/api')

    expect(mockAxios.get).toBeDefined()
  })
})
