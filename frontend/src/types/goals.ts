/**
 * Goals and achievements types
 */

export interface Goal {
  id: string;
  user_id: string;
  type: GoalType;
  title: string;
  description: string;
  target_value: number;
  current_value: number;
  unit: string;
  status: GoalStatus;
  priority: GoalPriority;
  deadline?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  metadata?: GoalMetadata;
}

export type GoalType =
  | 'practice_time'
  | 'session_count'
  | 'streak_days'
  | 'vocabulary_count'
  | 'fluency_score'
  | 'accuracy_score'
  | 'correction_reduction'
  | 'custom';

export type GoalStatus =
  | 'active'
  | 'completed'
  | 'failed'
  | 'paused'
  | 'cancelled';

export type GoalPriority = 'low' | 'medium' | 'high';

export interface GoalMetadata {
  category?: string;
  difficulty?: 'easy' | 'medium' | 'hard';
  suggested?: boolean;
  auto_generated?: boolean;
}

export interface GoalProgress {
  goal_id: string;
  percentage: number;
  remaining: number;
  estimated_completion?: string;
  pace: 'ahead' | 'on_track' | 'behind';
}

export interface GoalCreateRequest {
  type: GoalType;
  title: string;
  description?: string;
  target_value: number;
  unit: string;
  deadline?: string;
  priority?: GoalPriority;
}

export interface GoalUpdateRequest {
  title?: string;
  description?: string;
  target_value?: number;
  deadline?: string;
  status?: GoalStatus;
  priority?: GoalPriority;
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: AchievementRarity;
  category: AchievementCategory;
  xp_reward: number;
  unlocked: boolean;
  unlocked_at?: string;
  progress?: AchievementProgress;
  criteria: AchievementCriteria;
}

export type AchievementRarity =
  | 'common'
  | 'uncommon'
  | 'rare'
  | 'epic'
  | 'legendary';

export type AchievementCategory =
  | 'practice'
  | 'streak'
  | 'accuracy'
  | 'fluency'
  | 'vocabulary'
  | 'social'
  | 'milestone'
  | 'special';

export interface AchievementProgress {
  current: number;
  required: number;
  percentage: number;
}

export interface AchievementCriteria {
  type: string;
  threshold: number;
  time_limit?: string;
  conditions?: Record<string, unknown>;
}

export interface AchievementUnlockedEvent {
  achievement: Achievement;
  unlocked_at: string;
  celebration_shown: boolean;
}

export interface GoalSuggestion {
  type: GoalType;
  title: string;
  description: string;
  target_value: number;
  unit: string;
  reasoning: string;
  difficulty: 'easy' | 'medium' | 'hard';
}
