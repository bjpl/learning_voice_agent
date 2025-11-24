import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

describe('Achievements Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('should initialize with default achievements', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    expect(store.achievements).toBeDefined()
    expect(Array.isArray(store.achievements)).toBe(true)
  })

  it('should unlock an achievement', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    // Add a locked achievement
    store.achievements = [{
      id: 'first_session',
      title: 'First Steps',
      description: 'Complete your first session',
      icon: '🎯',
      rarity: 'common',
      category: 'learning',
      unlocked: false,
      progress: 0,
      maxProgress: 1,
    }]

    store.unlockAchievement('first_session')

    const achievement = store.achievements.find(a => a.id === 'first_session')
    expect(achievement?.unlocked).toBe(true)
    expect(achievement?.unlockedAt).toBeDefined()
  })

  it('should calculate total unlocked count', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    store.achievements = [
      { id: '1', title: 'A1', unlocked: true, rarity: 'common', category: 'learning' },
      { id: '2', title: 'A2', unlocked: true, rarity: 'rare', category: 'learning' },
      { id: '3', title: 'A3', unlocked: false, rarity: 'epic', category: 'learning' },
    ]

    expect(store.unlockedCount).toBe(2)
    expect(store.totalCount).toBe(3)
  })

  it('should filter achievements by category', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    store.achievements = [
      { id: '1', title: 'A1', unlocked: true, rarity: 'common', category: 'learning' },
      { id: '2', title: 'A2', unlocked: true, rarity: 'rare', category: 'streaks' },
      { id: '3', title: 'A3', unlocked: false, rarity: 'epic', category: 'learning' },
    ]

    const learningAchievements = store.getByCategory('learning')
    expect(learningAchievements).toHaveLength(2)
  })

  it('should filter achievements by rarity', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    store.achievements = [
      { id: '1', title: 'A1', unlocked: true, rarity: 'common', category: 'learning' },
      { id: '2', title: 'A2', unlocked: true, rarity: 'rare', category: 'learning' },
      { id: '3', title: 'A3', unlocked: false, rarity: 'rare', category: 'learning' },
    ]

    const rareAchievements = store.getByRarity('rare')
    expect(rareAchievements).toHaveLength(2)
  })

  it('should update achievement progress', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    store.achievements = [{
      id: 'streak_7',
      title: 'Week Warrior',
      description: 'Maintain a 7-day streak',
      rarity: 'uncommon',
      category: 'streaks',
      unlocked: false,
      progress: 3,
      maxProgress: 7,
    }]

    store.updateProgress('streak_7', 5)

    const achievement = store.achievements.find(a => a.id === 'streak_7')
    expect(achievement?.progress).toBe(5)
  })

  it('should auto-unlock when progress reaches max', async () => {
    const { useAchievementsStore } = await import('@/stores/achievements')
    const store = useAchievementsStore()

    store.achievements = [{
      id: 'streak_7',
      title: 'Week Warrior',
      rarity: 'uncommon',
      category: 'streaks',
      unlocked: false,
      progress: 6,
      maxProgress: 7,
    }]

    store.updateProgress('streak_7', 7)

    const achievement = store.achievements.find(a => a.id === 'streak_7')
    expect(achievement?.unlocked).toBe(true)
  })
})
