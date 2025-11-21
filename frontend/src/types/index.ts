/**
 * Central type exports for the application
 */

// API types
export type {
  ApiResponse,
  ApiError,
  RateLimitInfo,
  HealthCheck,
  ServiceStatus,
  PaginationParams,
  PaginatedResponse,
  RequestConfig,
} from './api';

// Conversation types
export type {
  ConversationRequest,
  ConversationResponse,
  Message,
  MessageMetadata,
  ConversationContext,
  CorrectionItem,
  CorrectionType,
  ConversationMetrics,
  ConversationSession,
  ConversationHistory,
} from './conversation';

// Feedback types
export type {
  ExplicitFeedback,
  FeedbackCategory,
  ImplicitFeedback,
  ImplicitEventType,
  ImplicitFeedbackMetadata,
  CorrectionFeedback,
  CorrectionRejectionReason,
  FeedbackSummary,
  FeedbackQueueItem,
} from './feedback';

// Analytics types
export type {
  OverviewResponse,
  UserLevel,
  ExperiencePoints,
  ProgressChartResponse,
  ProgressDataPoint,
  ProgressSummary,
  TrendResponse,
  TrendCategory,
  TrendDataPoint,
  TrendInsight,
  InsightResponse,
  Insight,
  InsightType,
  SkillBreakdown,
  AnalyticsPeriod,
} from './analytics';

// Goals types
export type {
  Goal,
  GoalType,
  GoalStatus,
  GoalPriority,
  GoalMetadata,
  GoalProgress,
  GoalCreateRequest,
  GoalUpdateRequest,
  Achievement,
  AchievementRarity,
  AchievementCategory,
  AchievementProgress,
  AchievementCriteria,
  AchievementUnlockedEvent,
  GoalSuggestion,
} from './goals';
