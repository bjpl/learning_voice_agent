/**
 * Analytics and dashboard data types
 */

export interface OverviewResponse {
  session_count: number;
  total_practice_minutes: number;
  messages_exchanged: number;
  corrections_received: number;
  current_streak: number;
  longest_streak: number;
  average_session_duration: number;
  improvement_rate: number;
  level: UserLevel;
  xp: ExperiencePoints;
}

export interface UserLevel {
  current: number;
  name: string;
  progress_to_next: number;
  xp_required: number;
}

export interface ExperiencePoints {
  total: number;
  today: number;
  this_week: number;
  this_month: number;
}

export interface ProgressChartResponse {
  period: 'day' | 'week' | 'month' | 'year';
  data_points: ProgressDataPoint[];
  summary: ProgressSummary;
}

export interface ProgressDataPoint {
  date: string;
  fluency_score: number;
  accuracy_score: number;
  vocabulary_score: number;
  practice_minutes: number;
  corrections: number;
  messages: number;
}

export interface ProgressSummary {
  fluency_change: number;
  accuracy_change: number;
  vocabulary_change: number;
  total_improvement: number;
  trend: 'improving' | 'stable' | 'declining';
}

export interface TrendResponse {
  category: TrendCategory;
  data: TrendDataPoint[];
  insights: TrendInsight[];
}

export type TrendCategory =
  | 'fluency'
  | 'accuracy'
  | 'vocabulary'
  | 'grammar'
  | 'pronunciation';

export interface TrendDataPoint {
  timestamp: string;
  value: number;
  baseline: number;
  target?: number;
}

export interface TrendInsight {
  type: 'strength' | 'weakness' | 'opportunity' | 'milestone';
  title: string;
  description: string;
  action_suggested?: string;
  priority: 'low' | 'medium' | 'high';
}

export interface InsightResponse {
  insights: Insight[];
  generated_at: string;
  valid_until: string;
}

export interface Insight {
  id: string;
  type: InsightType;
  title: string;
  description: string;
  data?: Record<string, unknown>;
  recommendations: string[];
  created_at: string;
}

export type InsightType =
  | 'learning_pattern'
  | 'common_mistake'
  | 'improvement_area'
  | 'achievement_nearby'
  | 'streak_risk'
  | 'personalized_tip';

export interface SkillBreakdown {
  skill: string;
  score: number;
  trend: 'up' | 'down' | 'stable';
  change: number;
  sub_skills?: SkillBreakdown[];
}

export interface AnalyticsPeriod {
  start: string;
  end: string;
  label: string;
}
