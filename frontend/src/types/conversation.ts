/**
 * Conversation and message types
 */

export interface ConversationRequest {
  text?: string;
  audio_base64?: string;
  session_id: string;
  language?: string;
  context?: ConversationContext;
}

export interface ConversationResponse {
  id: string;
  session_id: string;
  user_message: Message;
  assistant_message: Message;
  audio_response_base64?: string;
  corrections?: CorrectionItem[];
  suggestions?: string[];
  metrics: ConversationMetrics;
  timestamp: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  audio_duration?: number;
  confidence?: number;
  language_detected?: string;
  tokens_used?: number;
  processing_time_ms?: number;
}

export interface ConversationContext {
  topic?: string;
  difficulty_level?: 'beginner' | 'intermediate' | 'advanced';
  focus_areas?: string[];
  previous_corrections?: string[];
}

export interface CorrectionItem {
  original: string;
  corrected: string;
  type: CorrectionType;
  explanation?: string;
  severity: 'minor' | 'moderate' | 'major';
}

export type CorrectionType =
  | 'grammar'
  | 'vocabulary'
  | 'pronunciation'
  | 'syntax'
  | 'style'
  | 'idiom';

export interface ConversationMetrics {
  response_time_ms: number;
  words_spoken: number;
  errors_detected: number;
  fluency_score?: number;
  accuracy_score?: number;
}

export interface ConversationSession {
  id: string;
  started_at: string;
  ended_at?: string;
  message_count: number;
  total_corrections: number;
  average_fluency?: number;
  topics_covered: string[];
}

export interface ConversationHistory {
  sessions: ConversationSession[];
  total_sessions: number;
  total_messages: number;
  total_practice_time_minutes: number;
}
