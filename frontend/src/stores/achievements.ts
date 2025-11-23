import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Achievement {
  id: string
  title: string
  description: string
  icon: string
  category: string
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary'
  unlockedAt?: Date
  progress: number
  maxProgress: number
  reward?: string
  isSecret: boolean
}

export interface AchievementCategory {
  id: string
  name: string
  icon: string
  achievements: Achievement[]
}

export const useAchievementsStore = defineStore('achievements', () => {
  const achievements = ref<Achievement[]>([])
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const recentUnlock = ref<Achievement | null>(null)

  const unlockedAchievements = computed(() =>
    achievements.value.filter(a => a.unlockedAt)
  )

  const lockedAchievements = computed(() =>
    achievements.value.filter(a => !a.unlockedAt && !a.isSecret)
  )

  const secretAchievements = computed(() =>
    achievements.value.filter(a => a.isSecret && !a.unlockedAt)
  )

  const totalPoints = computed(() => {
    const rarityPoints: Record<string, number> = {
      common: 10,
      uncommon: 25,
      rare: 50,
      epic: 100,
      legendary: 250
    }
    return unlockedAchievements.value.reduce((sum, a) =>
      sum + (rarityPoints[a.rarity] || 0), 0
    )
  })

  const completionPercentage = computed(() => {
    if (achievements.value.length === 0) return 0
    return Math.round(
      (unlockedAchievements.value.length / achievements.value.length) * 100
    )
  })

  const achievementsByCategory = computed(() => {
    const grouped: Record<string, Achievement[]> = {}
    achievements.value.forEach(a => {
      if (!grouped[a.category]) {
        grouped[a.category] = []
      }
      grouped[a.category].push(a)
    })
    return grouped
  })

  const rarityColors: Record<string, string> = {
    common: 'gray',
    uncommon: 'green',
    rare: 'blue',
    epic: 'purple',
    legendary: 'yellow'
  }

  async function fetchAchievements(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      // API call would go here - using mock data
      achievements.value = [
        {
          id: '1',
          title: 'First Steps',
          description: 'Complete your first conversation',
          icon: 'star',
          category: 'Milestones',
          rarity: 'common',
          unlockedAt: new Date(),
          progress: 1,
          maxProgress: 1,
          isSecret: false
        },
        {
          id: '2',
          title: 'Chatterbox',
          description: 'Send 100 messages',
          icon: 'message-circle',
          category: 'Communication',
          rarity: 'uncommon',
          unlockedAt: new Date(),
          progress: 100,
          maxProgress: 100,
          isSecret: false
        },
        {
          id: '3',
          title: 'Wordsmith',
          description: 'Learn 500 vocabulary words',
          icon: 'book-open',
          category: 'Vocabulary',
          rarity: 'rare',
          progress: 234,
          maxProgress: 500,
          isSecret: false
        },
        {
          id: '4',
          title: 'Marathon Learner',
          description: 'Study for 50 hours total',
          icon: 'clock',
          category: 'Dedication',
          rarity: 'epic',
          progress: 28,
          maxProgress: 50,
          isSecret: false
        },
        {
          id: '5',
          title: 'Perfect Week',
          description: 'Complete all daily goals for 7 days',
          icon: 'award',
          category: 'Streaks',
          rarity: 'rare',
          progress: 4,
          maxProgress: 7,
          isSecret: false
        },
        {
          id: '6',
          title: 'The Hidden Path',
          description: '???',
          icon: 'help-circle',
          category: 'Secrets',
          rarity: 'legendary',
          progress: 0,
          maxProgress: 1,
          isSecret: true
        }
      ]
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch achievements'
    } finally {
      isLoading.value = false
    }
  }

  function updateProgress(id: string, progress: number): void {
    const achievement = achievements.value.find(a => a.id === id)
    if (achievement) {
      achievement.progress = Math.min(progress, achievement.maxProgress)
      if (achievement.progress >= achievement.maxProgress && !achievement.unlockedAt) {
        unlockAchievement(id)
      }
    }
  }

  function unlockAchievement(id: string): void {
    const achievement = achievements.value.find(a => a.id === id)
    if (achievement && !achievement.unlockedAt) {
      achievement.unlockedAt = new Date()
      achievement.progress = achievement.maxProgress
      recentUnlock.value = achievement
    }
  }

  function clearRecentUnlock(): void {
    recentUnlock.value = null
  }

  function getAchievementsByRarity(rarity: Achievement['rarity']): Achievement[] {
    return achievements.value.filter(a => a.rarity === rarity)
  }

  return {
    achievements,
    isLoading,
    error,
    recentUnlock,
    unlockedAchievements,
    lockedAchievements,
    secretAchievements,
    totalPoints,
    completionPercentage,
    achievementsByCategory,
    rarityColors,
    fetchAchievements,
    updateProgress,
    unlockAchievement,
    clearRecentUnlock,
    getAchievementsByRarity
  }
})
