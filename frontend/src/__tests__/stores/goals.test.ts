import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

describe('Goals Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('should initialize with empty goals', async () => {
    const { useGoalsStore } = await import('@/stores/goals')
    const store = useGoalsStore()

    expect(store.goals).toEqual([])
    expect(store.isLoading).toBe(false)
  })

  it('should add a goal', async () => {
    const { useGoalsStore } = await import('@/stores/goals')
    const store = useGoalsStore()

    const goal = {
      id: '1',
      title: 'Complete 5 sessions',
      description: 'Have 5 learning sessions',
      type: 'sessions' as const,
      targetValue: 5,
      currentValue: 0,
      status: 'active' as const,
      createdAt: new Date().toISOString(),
    }

    store.addGoal(goal)

    expect(store.goals).toHaveLength(1)
    expect(store.goals[0].title).toBe('Complete 5 sessions')
  })

  it('should update goal progress', async () => {
    const { useGoalsStore } = await import('@/stores/goals')
    const store = useGoalsStore()

    store.addGoal({
      id: '1',
      title: 'Complete 5 sessions',
      type: 'sessions' as const,
      targetValue: 5,
      currentValue: 0,
      status: 'active' as const,
      createdAt: new Date().toISOString(),
    })

    store.updateProgress('1', 3)

    expect(store.goals[0].currentValue).toBe(3)
  })

  it('should mark goal as completed when target reached', async () => {
    const { useGoalsStore } = await import('@/stores/goals')
    const store = useGoalsStore()

    store.addGoal({
      id: '1',
      title: 'Complete 5 sessions',
      type: 'sessions' as const,
      targetValue: 5,
      currentValue: 0,
      status: 'active' as const,
      createdAt: new Date().toISOString(),
    })

    store.updateProgress('1', 5)

    expect(store.goals[0].status).toBe('completed')
  })

  it('should filter goals by status', async () => {
    const { useGoalsStore } = await import('@/stores/goals')
    const store = useGoalsStore()

    store.addGoal({
      id: '1',
      title: 'Goal 1',
      type: 'sessions' as const,
      targetValue: 5,
      currentValue: 0,
      status: 'active' as const,
      createdAt: new Date().toISOString(),
    })

    store.addGoal({
      id: '2',
      title: 'Goal 2',
      type: 'streak' as const,
      targetValue: 7,
      currentValue: 7,
      status: 'completed' as const,
      createdAt: new Date().toISOString(),
    })

    expect(store.activeGoals).toHaveLength(1)
    expect(store.completedGoals).toHaveLength(1)
  })

  it('should delete a goal', async () => {
    const { useGoalsStore } = await import('@/stores/goals')
    const store = useGoalsStore()

    store.addGoal({
      id: '1',
      title: 'Goal 1',
      type: 'sessions' as const,
      targetValue: 5,
      currentValue: 0,
      status: 'active' as const,
      createdAt: new Date().toISOString(),
    })

    expect(store.goals).toHaveLength(1)

    store.deleteGoal('1')

    expect(store.goals).toHaveLength(0)
  })
})
