/**
 * Feedback types for user interactions
 */

export interface ExplicitFeedback {
  id?: string;
  session_id: string;
  message_id: string;
  rating: 1 | 2 | 3 | 4 | 5;
  category: FeedbackCategory;
  comment?: string;
  tags?: string[];
  timestamp?: string;
}

export type FeedbackCategory =
  | 'response_quality'
  | 'correction_accuracy'
  | 'explanation_clarity'
  | 'voice_quality'
  | 'difficulty_level'
  | 'general';

export interface ImplicitFeedback {
  id?: string;
  session_id: string;
  message_id: string;
  event_type: ImplicitEventType;
  duration_ms?: number;
  metadata?: ImplicitFeedbackMetadata;
  timestamp?: string;
}

export type ImplicitEventType =
  | 'message_replay'
  | 'correction_viewed'
  | 'explanation_expanded'
  | 'audio_replayed'
  | 'response_skipped'
  | 'session_abandoned'
  | 'long_pause'
  | 'retry_speech';

export interface ImplicitFeedbackMetadata {
  replay_count?: number;
  view_duration_ms?: number;
  scroll_depth?: number;
  interaction_count?: number;
}

export interface CorrectionFeedback {
  id?: string;
  session_id: string;
  correction_id: string;
  accepted: boolean;
  user_alternative?: string;
  rejection_reason?: CorrectionRejectionReason;
  timestamp?: string;
}

export type CorrectionRejectionReason =
  | 'incorrect_correction'
  | 'prefer_original'
  | 'context_specific'
  | 'regional_variation'
  | 'informal_acceptable'
  | 'other';

export interface FeedbackSummary {
  session_id: string;
  total_explicit: number;
  total_implicit: number;
  total_corrections: number;
  average_rating: number;
  correction_acceptance_rate: number;
  common_issues: string[];
}

export interface FeedbackQueueItem {
  type: 'explicit' | 'implicit' | 'correction';
  payload: ExplicitFeedback | ImplicitFeedback | CorrectionFeedback;
  retries: number;
  created_at: string;
}
