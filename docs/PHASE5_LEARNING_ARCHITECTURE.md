# Phase 5: Real-Time Learning Architecture

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Architecture Design
**Target:** Production-ready real-time learning system for single-user Railway deployment

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Critical Design Decisions](#critical-design-decisions)
4. [Feedback Collection System](#feedback-collection-system)
5. [Quality Scoring Engine](#quality-scoring-engine)
6. [Learning Analytics Pipeline](#learning-analytics-pipeline)
7. [Personalization Engine](#personalization-engine)
8. [Integration Architecture](#integration-architecture)
9. [Database Schema](#database-schema)
10. [Privacy and Data Retention](#privacy-and-data-retention)
11. [Performance Targets](#performance-targets)
12. [A/B Testing Framework](#ab-testing-framework)
13. [Implementation Plan](#implementation-plan)

---

## Executive Summary

### Objectives

Build a **real-time learning system** that improves from user interactions:

- **Collect** explicit and implicit feedback on response quality
- **Score** conversation quality using multi-dimensional metrics
- **Detect** patterns in user preferences and learning behavior
- **Adapt** responses through prompt tuning and retrieval augmentation
- **Personalize** content style, depth, and topics based on learned preferences

### Key Principles

1. **Simplicity First**: Single SQLite database, no external ML services
2. **Privacy Aware**: Local processing, configurable retention, no external data sharing
3. **Non-Blocking**: Learning loop must not degrade response latency
4. **Graceful Degradation**: System works without learning data; improvements are additive
5. **Observable**: All learning decisions are logged and traceable

### Success Criteria

- Feedback collection latency: < 10ms overhead
- Learning loop execution: < 100ms (background)
- Quality improvement: > 10% positive feedback rate increase over baseline
- Personalization accuracy: > 80% preference prediction
- Zero regression in response latency (P95 < 1.5s maintained)

---

## System Architecture Overview

### High-Level Learning Architecture

```
+-----------------------------------------------------------------------+
|                         USER INTERACTION                               |
+------------------------------------+----------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                    FEEDBACK COLLECTION LAYER                           |
+----------+------------+------------+------------+------------+---------+
|  Explicit |  Implicit  | Behavioral | Contextual | Correction |         |
|  Ratings  | Engagement | Patterns   | Signals    | Events     |         |
| (thumbs)  | (timing)   | (follow-up)| (session)  | (edits)    |         |
+-----+-----+-----+------+-----+------+-----+------+-----+------+---------+
      |           |            |            |            |
      +-----------+------------+------------+------------+
                               |
                               v
+-----------------------------------------------------------------------+
|                    LEARNING ANALYTICS ENGINE                           |
+-----------------------------------------------------------------------+
|                                                                        |
|   +------------------+   +-------------------+   +------------------+  |
|   | Quality Scoring  |   | Pattern Detection |   | Preference       |  |
|   |                  |   |                   |   | Learning         |  |
|   | - Response score |   | - Topic clusters  |   | - Style prefs    |  |
|   | - Engagement     |   | - Time patterns   |   | - Depth prefs    |  |
|   | - Helpfulness    |   | - Error patterns  |   | - Topic prefs    |  |
|   +------------------+   +-------------------+   +------------------+  |
|                                                                        |
+------------------------------------+----------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                ADAPTATION & PERSONALIZATION LAYER                      |
+-----------------------------------------------------------------------+
|                                                                        |
|   +------------------+   +-------------------+   +------------------+  |
|   | Prompt Tuning    |   | Context           |   | Response         |  |
|   |                  |   | Enhancement       |   | Calibration      |  |
|   | - Style params   |   | - Pref injection  |   | - Length adjust  |  |
|   | - Depth params   |   | - Topic boost     |   | - Detail level   |  |
|   | - Format params  |   | - History weight  |   | - Tone matching  |  |
|   +------------------+   +-------------------+   +------------------+  |
|                                                                        |
+-----------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                    AGENT SYSTEM (Phase 2)                              |
+-----------------------------------------------------------------------+
|  ConversationAgent | AnalysisAgent | ResearchAgent | SynthesisAgent   |
+-----------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                    RAG SYSTEM (Phase 3)                                |
+-----------------------------------------------------------------------+
|  Retriever (with preference boost) | Context Builder | Generator      |
+-----------------------------------------------------------------------+
```

### Component Interaction Flow

```
User Message
     |
     v
+--------------------+     +--------------------+
| 1. CAPTURE CONTEXT |---->| 2. LOAD PREFERENCES|
|    (implicit)      |     |    (< 5ms)         |
+--------------------+     +--------------------+
     |                              |
     v                              v
+--------------------+     +--------------------+
| 3. ENHANCE PROMPT  |<----| 4. INJECT CONTEXT  |
|    (style/depth)   |     |    (preferences)   |
+--------------------+     +--------------------+
     |
     v
+--------------------+
| 5. GENERATE        |
|    RESPONSE        |
|    (agents + RAG)  |
+--------------------+
     |
     v
+--------------------+     +--------------------+
| 6. CAPTURE         |---->| 7. SCORE QUALITY   |
|    RESPONSE META   |     |    (background)    |
+--------------------+     +--------------------+
     |                              |
     v                              v
+--------------------+     +--------------------+
| 8. AWAIT FEEDBACK  |     | 9. UPDATE LEARNING |
|    (30s window)    |     |    (async)         |
+--------------------+     +--------------------+
```

---

## Critical Design Decisions

### Decision 1: Feedback Mechanism

**Choice: BOTH Explicit AND Implicit**

| Mechanism | Pros | Cons | Implementation |
|-----------|------|------|----------------|
| **Explicit** (thumbs up/down) | Clear signal, actionable | Low participation rate (~5%) | Simple UI buttons |
| **Implicit** (engagement) | Always available, high volume | Noisy, requires interpretation | Automatic collection |

**Rationale**: Explicit feedback provides high-quality training signal for the quality scorer. Implicit feedback provides volume for pattern detection. Combined approach maximizes learning signal while minimizing user burden.

**Explicit Signals Collected:**
- Thumbs up/down rating (binary)
- Optional text feedback ("what went wrong?")
- Copy action (indicates usefulness)
- Regenerate action (indicates dissatisfaction)

**Implicit Signals Collected:**
- Response time (time to next user message)
- Follow-up type (clarification vs new topic)
- Session depth (exchanges in session)
- Message length (short vs detailed user input)
- Edit/retry patterns

---

### Decision 2: Learning Storage

**Choice: SQLite Tables (Extend Existing Database)**

| Option | Pros | Cons | Railway Cost |
|--------|------|------|--------------|
| **SQLite** | Simple, no external deps, ACID | Limited concurrent writes | $0/mo |
| PostgreSQL | Better concurrency, richer types | External service complexity | ~$7/mo |
| Separate DB | Isolation, schema flexibility | Complexity, coordination | Variable |

**Rationale**: Single-user Railway deployment means write concurrency is not a concern. SQLite's simplicity and zero operational overhead outweighs the benefits of a separate database for the learning data.

**Schema Extension:**
- Add `feedback` table
- Add `learning_metrics` table
- Add `user_preferences` table
- Add `quality_scores` table
- All in existing `learning_captures.db`

---

### Decision 3: Adaptation Strategy

**Choice: Retrieval Augmentation + Dynamic Prompt Tuning**

| Strategy | Pros | Cons | Fit |
|----------|------|------|-----|
| Fine-tuning | Best quality | Expensive, API limitation | Not possible with Claude |
| **Prompt Tuning** | Fast, no model access needed | Limited scope | Excellent |
| **Retrieval Aug** | Uses existing RAG, flexible | Requires good retrieval | Excellent |
| Few-shot learning | Simple, effective | Context window limits | Good |

**Rationale**: Claude API does not support fine-tuning. Prompt tuning (adjusting system prompts with learned parameters) and retrieval augmentation (boosting relevant past interactions) provide effective personalization without model access.

**Implementation:**
1. **Prompt Parameters**: Inject learned style/depth/format preferences into system prompt
2. **Context Enhancement**: Boost retrieval of high-rated past responses as examples
3. **Preference Injection**: Include explicit user preferences in conversation context

---

### Decision 4: Pattern Detection

**Choice: Rule-Based with Heuristics (Start Simple)**

| Approach | Pros | Cons | Complexity |
|----------|------|------|------------|
| **Rule-based** | Interpretable, fast, deterministic | Less adaptive | Low |
| ML-based | Adaptive, handles complexity | Training data needed, opaque | High |
| Hybrid | Best of both | Implementation complexity | Medium |

**Rationale**: For a single-user system, rule-based pattern detection is sufficient and more maintainable. Complex ML models require significant training data that a single user won't generate quickly. Start simple, add ML later if needed.

**Rule-Based Patterns:**
- Topic affinity: Track topic frequencies, boost preferred topics
- Style preference: Detect preferred response length/detail from engagement
- Time patterns: Identify peak usage times, session duration patterns
- Error patterns: Track common misunderstandings for correction

---

### Decision 5: Personalization Scope

**Choice: Progressive Enhancement (All, Phased)**

| Scope | Phase | Complexity | Value |
|-------|-------|------------|-------|
| **Response Style** | 1 | Low | High |
| **Topic Focus** | 1 | Low | Medium |
| **Vocabulary** | 2 | Medium | Medium |
| **Depth/Detail** | 1 | Low | High |
| **Follow-up Style** | 2 | Medium | Medium |

**Phase 1 Personalization:**
1. Response length preference (concise vs detailed)
2. Technical depth preference (simple vs advanced)
3. Topic boosting (preferred subjects in retrieval)
4. Format preference (lists vs prose vs examples)

**Phase 2 Personalization (Future):**
1. Vocabulary adaptation
2. Follow-up question style
3. Proactive suggestions
4. Learning pace adaptation

---

## Feedback Collection System

### Feedback Data Model

```
+-----------------------------------------------------------------------+
|                         FEEDBACK TYPES                                 |
+-----------------------------------------------------------------------+

EXPLICIT FEEDBACK
+------------------------+
| exchange_id    (FK)    |
| feedback_type  (enum)  |  --> thumbs_up, thumbs_down, copy, regenerate
| feedback_text  (text)  |  --> optional user comment
| timestamp      (dt)    |
+------------------------+

IMPLICIT FEEDBACK
+------------------------+
| exchange_id    (FK)    |
| response_time_ms (int) |  --> time to next user message
| follow_up_type (enum)  |  --> clarification, continuation, new_topic, end
| user_msg_length (int)  |  --> length of user's next message
| session_position (int) |  --> position in session (1st, 5th, etc.)
| timestamp      (dt)    |
+------------------------+
```

### Feedback Collection Architecture

```
+-----------------------------------------------------------------------+
|                    FEEDBACK COLLECTOR                                  |
+-----------------------------------------------------------------------+

                    User Interaction
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
+----------------+                  +--------------------+
| ExplicitCollector |              | ImplicitCollector   |
|                |                  |                    |
| - Button clicks|                  | - Timing signals   |
| - Copy events  |                  | - Session tracking |
| - Regen events |                  | - Length analysis  |
| - Text feedback|                  | - Follow-up detect |
+-------+--------+                  +---------+----------+
        |                                     |
        +------------------+------------------+
                           |
                           v
                 +-------------------+
                 | FeedbackNormalizer|
                 |                   |
                 | - Deduplicate     |
                 | - Validate        |
                 | - Enrich metadata |
                 +--------+----------+
                          |
                          v
                 +-------------------+
                 | FeedbackStorage   |
                 |                   |
                 | - SQLite write    |
                 | - Background queue|
                 +-------------------+
```

### Feedback Collection Interface

```python
@dataclass
class ExplicitFeedback:
    """Explicit user feedback on a response"""
    exchange_id: int
    feedback_type: FeedbackType  # thumbs_up, thumbs_down, copy, regenerate
    feedback_text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ImplicitFeedback:
    """Automatically collected engagement signals"""
    exchange_id: int
    response_time_ms: int          # Time until user's next message
    follow_up_type: FollowUpType   # clarification, continuation, new_topic, end
    user_message_length: int       # Length of next user message
    session_position: int          # Exchange number in session
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    COPY = "copy"
    REGENERATE = "regenerate"

class FollowUpType(Enum):
    CLARIFICATION = "clarification"   # User asks for clarification
    CONTINUATION = "continuation"      # User continues on same topic
    NEW_TOPIC = "new_topic"           # User changes topic
    END = "end"                       # Session ends
```

### Feedback API Endpoints

```
POST /api/feedback
    Body: { exchange_id, feedback_type, feedback_text? }
    Returns: { success, feedback_id }

GET /api/feedback/stats
    Returns: { total_feedback, positive_rate, recent_trend }

GET /api/feedback/exchange/{exchange_id}
    Returns: { explicit_feedback?, implicit_feedback?, quality_score }
```

---

## Quality Scoring Engine

### Multi-Dimensional Quality Model

```
+-----------------------------------------------------------------------+
|                    QUALITY SCORING MODEL                               |
+-----------------------------------------------------------------------+

QUALITY_SCORE = weighted_sum([
    RELEVANCE_SCORE,      # 0.30 - Did response address the question?
    HELPFULNESS_SCORE,    # 0.25 - Did user find it useful?
    ENGAGEMENT_SCORE,     # 0.20 - Did user engage further?
    CLARITY_SCORE,        # 0.15 - Was response clear and well-structured?
    ACCURACY_SCORE        # 0.10 - Was information correct? (hard to measure)
])

Each sub-score: 0.0 to 1.0
Final score: 0.0 to 1.0
```

### Scoring Algorithm

```python
@dataclass
class QualityScore:
    """Multi-dimensional quality assessment"""
    exchange_id: int

    # Component scores (0.0 - 1.0)
    relevance_score: float       # Based on semantic similarity
    helpfulness_score: float     # Based on explicit feedback
    engagement_score: float      # Based on implicit signals
    clarity_score: float         # Based on response structure
    accuracy_score: float        # Based on corrections (if any)

    # Composite score
    overall_score: float         # Weighted combination

    # Metadata
    confidence: float            # Confidence in this score
    scored_at: datetime
    scoring_version: str         # Algorithm version

class QualityScoringEngine:
    """
    Calculate quality scores for exchanges.

    PATTERN: Multi-signal fusion
    WHY: No single signal captures quality; combine multiple perspectives
    """

    WEIGHTS = {
        "relevance": 0.30,
        "helpfulness": 0.25,
        "engagement": 0.20,
        "clarity": 0.15,
        "accuracy": 0.10
    }

    async def score_exchange(
        self,
        exchange_id: int,
        explicit_feedback: Optional[ExplicitFeedback] = None,
        implicit_feedback: Optional[ImplicitFeedback] = None
    ) -> QualityScore:
        """
        Score an exchange using available signals.

        Scoring logic:
        1. Relevance: Semantic similarity between query and response
        2. Helpfulness: Explicit feedback (if available)
        3. Engagement: Implicit signals (response time, follow-up)
        4. Clarity: Response structure analysis
        5. Accuracy: Correction events (if any)
        """
        pass
```

### Relevance Scoring

```python
def calculate_relevance_score(
    user_text: str,
    agent_text: str,
    embeddings_engine: EmbeddingEngine
) -> float:
    """
    Calculate relevance using semantic similarity.

    Algorithm:
    1. Embed user query
    2. Embed agent response
    3. Calculate cosine similarity
    4. Apply threshold and scaling
    """
    user_embedding = embeddings_engine.embed(user_text)
    response_embedding = embeddings_engine.embed(agent_text)

    similarity = cosine_similarity(user_embedding, response_embedding)

    # Scale: 0.0-0.5 -> 0.0, 0.5-0.9 -> 0.0-1.0, 0.9-1.0 -> 1.0
    if similarity < 0.5:
        return 0.0
    elif similarity > 0.9:
        return 1.0
    else:
        return (similarity - 0.5) / 0.4
```

### Engagement Scoring

```python
def calculate_engagement_score(
    implicit: ImplicitFeedback,
    session_avg_response_time: float
) -> float:
    """
    Calculate engagement from implicit signals.

    Signals (each 0.0-1.0):
    - response_time: Faster response = higher engagement
    - follow_up_type: Continuation > Clarification > New Topic > End
    - message_length: Longer follow-up = more engaged
    - session_depth: Later in session = more engaged
    """
    # Response time signal (faster = more engaged)
    time_ratio = implicit.response_time_ms / session_avg_response_time
    time_signal = max(0, min(1, 2 - time_ratio))  # 1.0 if twice as fast

    # Follow-up type signal
    follow_up_signals = {
        FollowUpType.CONTINUATION: 1.0,
        FollowUpType.CLARIFICATION: 0.6,
        FollowUpType.NEW_TOPIC: 0.4,
        FollowUpType.END: 0.2
    }
    follow_up_signal = follow_up_signals.get(implicit.follow_up_type, 0.5)

    # Message length signal (longer = more engaged)
    length_signal = min(1.0, implicit.user_message_length / 200)

    # Combine signals
    return (time_signal * 0.4 + follow_up_signal * 0.4 + length_signal * 0.2)
```

### Helpfulness Scoring

```python
def calculate_helpfulness_score(
    explicit: Optional[ExplicitFeedback],
    prior_scores: List[float]  # Historical scores for this topic
) -> float:
    """
    Calculate helpfulness from explicit feedback.

    Scoring:
    - thumbs_up: 1.0
    - copy: 0.9 (useful enough to copy)
    - no feedback: 0.5 (neutral) or use prior average
    - regenerate: 0.3 (not satisfied but trying)
    - thumbs_down: 0.1
    """
    if explicit is None:
        # Use historical average if available
        return sum(prior_scores) / len(prior_scores) if prior_scores else 0.5

    feedback_scores = {
        FeedbackType.THUMBS_UP: 1.0,
        FeedbackType.COPY: 0.9,
        FeedbackType.REGENERATE: 0.3,
        FeedbackType.THUMBS_DOWN: 0.1
    }

    return feedback_scores.get(explicit.feedback_type, 0.5)
```

---

## Learning Analytics Pipeline

### Pattern Detection Architecture

```
+-----------------------------------------------------------------------+
|                    LEARNING ANALYTICS PIPELINE                         |
+-----------------------------------------------------------------------+

Raw Feedback Data
       |
       v
+------------------+
| Data Aggregator  |
|                  |
| - Batch feedback |
| - Time windows   |
| - Session groups |
+--------+---------+
         |
         v
+------------------+    +------------------+    +------------------+
| Topic Analyzer   |    | Style Analyzer   |    | Error Analyzer   |
|                  |    |                  |    |                  |
| - Topic clusters |    | - Length prefs   |    | - Common errors  |
| - Topic affinty  |    | - Detail prefs   |    | - Misunderstand  |
| - Topic trends   |    | - Format prefs   |    | - Correction pat |
+--------+---------+    +--------+---------+    +--------+---------+
         |                       |                       |
         +-----------------------------------------------+
                                 |
                                 v
                      +--------------------+
                      | Pattern Aggregator |
                      |                    |
                      | - Confidence calc  |
                      | - Trend detection  |
                      | - Anomaly flagging |
                      +----------+---------+
                                 |
                                 v
                      +--------------------+
                      | Preference Store   |
                      |                    |
                      | - User preferences |
                      | - Model parameters |
                      | - Learning history |
                      +--------------------+
```

### Topic Affinity Detection

```python
class TopicAffinityAnalyzer:
    """
    Detect preferred topics from interaction patterns.

    ALGORITHM:
    1. Extract topics from exchanges using AnalysisAgent
    2. Calculate engagement score per topic
    3. Identify high-affinity topics (top 20%)
    4. Track affinity changes over time
    """

    def analyze_topic_affinity(
        self,
        exchanges: List[Exchange],
        quality_scores: List[QualityScore]
    ) -> Dict[str, float]:
        """
        Calculate topic affinity scores.

        Returns: {topic: affinity_score} where score is 0.0-1.0
        """
        topic_engagement: Dict[str, List[float]] = defaultdict(list)

        for exchange, score in zip(exchanges, quality_scores):
            # Extract topics (using cached analysis or re-analyze)
            topics = extract_topics(exchange.user_text)

            for topic in topics:
                topic_engagement[topic].append(score.engagement_score)

        # Calculate average engagement per topic
        topic_affinity = {
            topic: sum(scores) / len(scores)
            for topic, scores in topic_engagement.items()
            if len(scores) >= 3  # Minimum sample size
        }

        # Normalize to 0.0-1.0
        if topic_affinity:
            max_score = max(topic_affinity.values())
            min_score = min(topic_affinity.values())
            range_score = max_score - min_score or 1

            topic_affinity = {
                topic: (score - min_score) / range_score
                for topic, score in topic_affinity.items()
            }

        return topic_affinity
```

### Style Preference Learning

```python
@dataclass
class StylePreferences:
    """Learned style preferences for response generation"""

    # Length preference (0.0 = concise, 1.0 = detailed)
    length_preference: float = 0.5

    # Technical depth (0.0 = simple, 1.0 = advanced)
    depth_preference: float = 0.5

    # Format preference
    preferred_format: str = "balanced"  # concise, balanced, detailed, lists

    # Examples preference (0.0 = no examples, 1.0 = many examples)
    examples_preference: float = 0.5

    # Confidence in these preferences
    confidence: float = 0.0

    # Data points used to learn
    sample_size: int = 0

    def to_prompt_params(self) -> Dict[str, str]:
        """Convert preferences to prompt parameters"""
        return {
            "length": self._length_instruction(),
            "depth": self._depth_instruction(),
            "format": self._format_instruction(),
            "examples": self._examples_instruction()
        }

    def _length_instruction(self) -> str:
        if self.length_preference < 0.3:
            return "Keep responses concise and to the point (2-3 sentences max)."
        elif self.length_preference > 0.7:
            return "Provide detailed, thorough explanations with full context."
        else:
            return "Balance brevity with thoroughness in responses."

class StyleAnalyzer:
    """
    Learn style preferences from feedback patterns.

    ALGORITHM:
    1. Correlate response characteristics with quality scores
    2. Identify response length correlation with engagement
    3. Detect format preferences (lists vs prose)
    4. Learn technical depth preferences
    """

    def learn_style_preferences(
        self,
        exchanges: List[Exchange],
        quality_scores: List[QualityScore]
    ) -> StylePreferences:
        """
        Learn style preferences from historical data.
        """
        if len(exchanges) < 10:
            return StylePreferences()  # Not enough data

        # Analyze length preference
        length_scores = []
        for exchange, score in zip(exchanges, quality_scores):
            response_length = len(exchange.agent_text)
            length_scores.append((response_length, score.overall_score))

        # Calculate correlation between length and quality
        length_preference = self._calculate_length_preference(length_scores)

        # Similar analysis for other dimensions...

        return StylePreferences(
            length_preference=length_preference,
            confidence=min(1.0, len(exchanges) / 50),  # Max confidence at 50 samples
            sample_size=len(exchanges)
        )
```

### Error Pattern Detection

```python
class ErrorPatternAnalyzer:
    """
    Detect patterns in misunderstandings and errors.

    PATTERNS TO DETECT:
    1. Common clarification requests (what did you mean by X?)
    2. Regeneration patterns (same question, different phrasing)
    3. Negative feedback patterns (topics/styles that fail)
    4. Correction sequences (user provides correct info after response)
    """

    def detect_error_patterns(
        self,
        exchanges: List[Exchange],
        feedback: List[ExplicitFeedback]
    ) -> List[ErrorPattern]:
        """
        Identify recurring error patterns.
        """
        patterns = []

        # Pattern 1: Clarification requests
        clarification_topics = self._find_clarification_topics(exchanges)

        # Pattern 2: Regeneration patterns
        regeneration_patterns = self._find_regeneration_patterns(exchanges, feedback)

        # Pattern 3: Negative feedback clusters
        negative_clusters = self._find_negative_clusters(exchanges, feedback)

        return patterns

@dataclass
class ErrorPattern:
    """Detected error pattern for improvement"""
    pattern_type: str  # clarification, regeneration, negative_cluster
    description: str
    frequency: int
    examples: List[int]  # Exchange IDs
    suggested_fix: Optional[str] = None
```

---

## Personalization Engine

### Personalization Architecture

```
+-----------------------------------------------------------------------+
|                    PERSONALIZATION ENGINE                              |
+-----------------------------------------------------------------------+

               Incoming Request
                      |
                      v
            +-------------------+
            | Preference Loader |
            |                   |
            | - Load user prefs |
            | - Load topic aff  |
            | - Load style pref |
            +--------+----------+
                     |
                     v
        +------------+------------+
        |                         |
        v                         v
+----------------+      +------------------+
| Prompt Tuner   |      | Context Enhancer |
|                |      |                  |
| - Style params |      | - Pref injection |
| - Depth params |      | - Topic boosting |
| - Format guide |      | - Example select |
+-------+--------+      +--------+---------+
        |                        |
        +------------+-----------+
                     |
                     v
            +-------------------+
            | Response Generator|
            | (ConversationAgent|
            |  + RAG pipeline)  |
            +-------------------+
```

### Prompt Tuning System

```python
class PromptTuner:
    """
    Dynamically tune system prompts based on learned preferences.

    APPROACH: Inject personalization parameters into system prompt
    WHY: No model fine-tuning available; prompt engineering is our tool
    """

    BASE_SYSTEM_PROMPT = """You are an advanced learning companion helping users capture and develop ideas through intelligent conversation.

{personalization_section}

Core behaviors:
- Ask ONE clarifying question when responses are vague
- Connect new topics to previous conversation themes
- Keep responses {length_style}
- {examples_instruction}
- {format_instruction}

{topic_focus_section}
"""

    def generate_personalized_prompt(
        self,
        style_prefs: StylePreferences,
        topic_affinities: Dict[str, float]
    ) -> str:
        """
        Generate personalized system prompt.
        """
        # Style parameters
        length_style = style_prefs._length_instruction()

        # Examples instruction
        if style_prefs.examples_preference > 0.6:
            examples_instruction = "Include concrete examples to illustrate points"
        else:
            examples_instruction = "Focus on explanations over examples unless asked"

        # Format instruction
        format_instructions = {
            "lists": "Use bullet points and lists for organization",
            "prose": "Write in flowing prose without bullet points",
            "balanced": "Balance prose with occasional bullet points"
        }
        format_instruction = format_instructions.get(
            style_prefs.preferred_format,
            format_instructions["balanced"]
        )

        # Topic focus section
        top_topics = sorted(
            topic_affinities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        if top_topics:
            topic_focus = f"""
The user has shown particular interest in: {', '.join(t[0] for t in top_topics)}.
When relevant, connect discussions to these areas of interest."""
        else:
            topic_focus = ""

        # Personalization section
        if style_prefs.confidence > 0.3:
            personalization = f"""
Based on our conversations, I've learned you prefer:
- {length_style}
- Technical depth: {'advanced explanations' if style_prefs.depth_preference > 0.6 else 'accessible explanations'}"""
        else:
            personalization = ""

        return self.BASE_SYSTEM_PROMPT.format(
            personalization_section=personalization,
            length_style=length_style,
            examples_instruction=examples_instruction,
            format_instruction=format_instruction,
            topic_focus_section=topic_focus
        )
```

### Context Enhancement for RAG

```python
class ContextEnhancer:
    """
    Enhance RAG retrieval with personalization signals.

    APPROACH: Boost retrieval scores for preferred topics
              and high-rated past responses
    """

    def enhance_retrieval(
        self,
        query: str,
        base_results: List[RetrievalResult],
        topic_affinities: Dict[str, float],
        quality_scores: Dict[int, float]  # exchange_id -> score
    ) -> List[RetrievalResult]:
        """
        Re-rank retrieval results based on personalization.

        Algorithm:
        1. Start with base relevance scores
        2. Boost by topic affinity (if result matches preferred topic)
        3. Boost by historical quality score (if result was rated highly)
        4. Re-rank and return
        """
        for result in base_results:
            # Base score from retrieval
            score = result.score

            # Topic affinity boost (up to +20%)
            result_topics = extract_topics(result.text)
            topic_boost = max(
                topic_affinities.get(topic, 0) * 0.2
                for topic in result_topics
            ) if result_topics else 0

            # Quality score boost (up to +15%)
            quality_boost = 0
            if result.exchange_id in quality_scores:
                quality_boost = (quality_scores[result.exchange_id] - 0.5) * 0.3

            # Apply boosts
            result.score = score + topic_boost + quality_boost

        # Re-sort by adjusted score
        return sorted(base_results, key=lambda r: r.score, reverse=True)
```

### Response Calibration

```python
class ResponseCalibrator:
    """
    Post-process responses to match preferences.

    APPROACH: Light editing of generated responses to better match style
    NOTE: Avoid heavy modification; trust the prompt tuning
    """

    def calibrate_response(
        self,
        response: str,
        style_prefs: StylePreferences
    ) -> str:
        """
        Apply light calibration to response.

        Calibrations:
        1. Length check: Warn if significantly outside preference
        2. Format check: Convert to preferred format if needed
        3. Add examples if preference high and none present
        """
        # This is intentionally light-touch
        # Heavy modification should be done via prompt tuning

        calibrated = response

        # Check if response matches length preference
        word_count = len(response.split())

        if style_prefs.length_preference < 0.3 and word_count > 200:
            # Response too long for concise preference
            # Log for learning, but don't truncate
            log_calibration_mismatch("length", "long", word_count)

        return calibrated
```

---

## Integration Architecture

### Integration with Phase 2 Agents

```
+-----------------------------------------------------------------------+
|                    AGENT INTEGRATION                                   |
+-----------------------------------------------------------------------+

ConversationAgent
       |
       +----> PreferencesMiddleware (injected)
       |              |
       |              v
       |      +------------------+
       |      | Load preferences |
       |      | Tune system prompt|
       |      +------------------+
       |
       +----> Original process() method
       |
       +----> FeedbackMiddleware (injected)
                      |
                      v
              +------------------+
              | Collect implicit |
              | Start feedback   |
              | window timer     |
              +------------------+
```

```python
class LearningMiddleware:
    """
    Middleware to inject learning capabilities into agents.

    PATTERN: Decorator/middleware pattern
    WHY: Non-invasive integration with existing agent system
    """

    def __init__(
        self,
        preference_store: PreferenceStore,
        feedback_collector: FeedbackCollector,
        prompt_tuner: PromptTuner
    ):
        self.preference_store = preference_store
        self.feedback_collector = feedback_collector
        self.prompt_tuner = prompt_tuner

    async def pre_process(
        self,
        session_id: str,
        user_input: str,
        base_system_prompt: str
    ) -> str:
        """
        Pre-process: Load preferences and tune prompt.
        """
        # Load user preferences (fast, cached)
        prefs = await self.preference_store.get_preferences(session_id)

        # Generate personalized prompt
        if prefs.confidence > 0.3:
            return self.prompt_tuner.generate_personalized_prompt(
                prefs.style,
                prefs.topic_affinities
            )

        return base_system_prompt

    async def post_process(
        self,
        session_id: str,
        exchange_id: int,
        user_input: str,
        agent_response: str,
        metadata: Dict
    ) -> None:
        """
        Post-process: Collect implicit feedback, start feedback window.
        """
        # Record exchange metadata for implicit feedback
        await self.feedback_collector.record_exchange(
            exchange_id=exchange_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            response_length=len(agent_response),
            processing_time_ms=metadata.get("processing_time_ms", 0)
        )

        # Start 30-second feedback window (non-blocking)
        asyncio.create_task(
            self.feedback_collector.start_feedback_window(exchange_id)
        )
```

### Integration with Phase 3 RAG

```python
class LearningEnhancedRAG:
    """
    RAG system enhanced with learning signals.

    MODIFICATIONS:
    1. Retriever: Boost by topic affinity and quality scores
    2. ContextBuilder: Include preference context
    3. Generator: Use personalized prompts
    """

    def __init__(
        self,
        base_rag: RAGPipeline,
        context_enhancer: ContextEnhancer,
        preference_store: PreferenceStore
    ):
        self.base_rag = base_rag
        self.context_enhancer = context_enhancer
        self.preference_store = preference_store

    async def retrieve_with_personalization(
        self,
        query: str,
        session_id: str
    ) -> List[RetrievalResult]:
        """
        Retrieve with personalization boost.
        """
        # Get base retrieval results
        base_results = await self.base_rag.retriever.retrieve(query)

        # Load personalization data
        prefs = await self.preference_store.get_preferences(session_id)
        quality_scores = await self.preference_store.get_quality_scores()

        # Enhance results with personalization
        if prefs.confidence > 0.3:
            enhanced_results = self.context_enhancer.enhance_retrieval(
                query=query,
                base_results=base_results,
                topic_affinities=prefs.topic_affinities,
                quality_scores=quality_scores
            )
            return enhanced_results

        return base_results
```

---

## Database Schema

### Schema Design

```sql
-- ============================================================================
-- PHASE 5: Learning Tables
-- Added to existing learning_captures.db
-- ============================================================================

-- Explicit feedback from users
CREATE TABLE IF NOT EXISTS feedback_explicit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL,  -- thumbs_up, thumbs_down, copy, regenerate
    feedback_text TEXT,           -- Optional user comment
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (exchange_id) REFERENCES captures(id)
);

CREATE INDEX idx_feedback_explicit_exchange ON feedback_explicit(exchange_id);
CREATE INDEX idx_feedback_explicit_type ON feedback_explicit(feedback_type);

-- Implicit engagement signals
CREATE TABLE IF NOT EXISTS feedback_implicit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    response_time_ms INTEGER,     -- Time to next user message
    follow_up_type TEXT,          -- clarification, continuation, new_topic, end
    user_message_length INTEGER,  -- Length of next user message
    session_position INTEGER,     -- Position in session
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (exchange_id) REFERENCES captures(id)
);

CREATE INDEX idx_feedback_implicit_exchange ON feedback_implicit(exchange_id);
CREATE INDEX idx_feedback_implicit_session ON feedback_implicit(session_id);

-- Computed quality scores per exchange
CREATE TABLE IF NOT EXISTS quality_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL UNIQUE,

    -- Component scores (0.0 - 1.0)
    relevance_score REAL,
    helpfulness_score REAL,
    engagement_score REAL,
    clarity_score REAL,
    accuracy_score REAL,

    -- Composite
    overall_score REAL NOT NULL,

    -- Metadata
    confidence REAL NOT NULL,
    scoring_version TEXT NOT NULL,
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (exchange_id) REFERENCES captures(id)
);

CREATE INDEX idx_quality_scores_exchange ON quality_scores(exchange_id);
CREATE INDEX idx_quality_scores_overall ON quality_scores(overall_score);

-- Learned user preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    preference_type TEXT NOT NULL,  -- style, topic_affinity, error_pattern
    preference_key TEXT NOT NULL,   -- e.g., 'length_preference', 'machine_learning'
    preference_value REAL NOT NULL, -- Numeric value (0.0 - 1.0)
    confidence REAL NOT NULL,
    sample_size INTEGER NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(preference_type, preference_key)
);

CREATE INDEX idx_preferences_type ON user_preferences(preference_type);

-- Learning metrics over time
CREATE TABLE IF NOT EXISTS learning_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,      -- quality_trend, feedback_rate, etc.
    metric_value REAL NOT NULL,
    period_start DATETIME NOT NULL,
    period_end DATETIME NOT NULL,
    sample_count INTEGER NOT NULL,
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_learning_metrics_type ON learning_metrics(metric_type);
CREATE INDEX idx_learning_metrics_period ON learning_metrics(period_start, period_end);

-- A/B test results
CREATE TABLE IF NOT EXISTS ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT NOT NULL,
    variant TEXT NOT NULL,          -- control, treatment_a, treatment_b
    exchange_id INTEGER NOT NULL,
    outcome_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (exchange_id) REFERENCES captures(id)
);

CREATE INDEX idx_ab_tests_name ON ab_tests(test_name);
CREATE INDEX idx_ab_tests_variant ON ab_tests(test_name, variant);
```

### Data Access Layer

```python
class LearningDatabase:
    """
    Data access layer for learning tables.

    PATTERN: Repository pattern with async operations
    WHY: Clean separation, testability, consistent error handling
    """

    def __init__(self, db_path: str = "learning_captures.db"):
        self.db_path = db_path

    async def save_explicit_feedback(
        self,
        feedback: ExplicitFeedback
    ) -> int:
        """Save explicit feedback, return feedback ID"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO feedback_explicit
                (exchange_id, feedback_type, feedback_text)
                VALUES (?, ?, ?)
                """,
                (feedback.exchange_id, feedback.feedback_type.value, feedback.feedback_text)
            )
            await db.commit()
            return cursor.lastrowid

    async def save_implicit_feedback(
        self,
        feedback: ImplicitFeedback
    ) -> int:
        """Save implicit feedback signals"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO feedback_implicit
                (exchange_id, session_id, response_time_ms, follow_up_type,
                 user_message_length, session_position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.exchange_id,
                    feedback.session_id,
                    feedback.response_time_ms,
                    feedback.follow_up_type.value,
                    feedback.user_message_length,
                    feedback.session_position
                )
            )
            await db.commit()
            return cursor.lastrowid

    async def get_quality_scores(
        self,
        min_score: float = 0.0,
        limit: int = 100
    ) -> Dict[int, float]:
        """Get quality scores as {exchange_id: score}"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT exchange_id, overall_score
                FROM quality_scores
                WHERE overall_score >= ?
                ORDER BY computed_at DESC
                LIMIT ?
                """,
                (min_score, limit)
            )
            rows = await cursor.fetchall()
            return {row["exchange_id"]: row["overall_score"] for row in rows}

    async def get_preferences(self) -> Dict[str, Dict[str, float]]:
        """Get all user preferences grouped by type"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT preference_type, preference_key, preference_value, confidence
                FROM user_preferences
                WHERE confidence >= 0.3
                """
            )
            rows = await cursor.fetchall()

            prefs = defaultdict(dict)
            for row in rows:
                prefs[row["preference_type"]][row["preference_key"]] = row["preference_value"]

            return dict(prefs)
```

---

## Privacy and Data Retention

### Privacy Architecture

```
+-----------------------------------------------------------------------+
|                    PRIVACY PROTECTION LAYERS                           |
+-----------------------------------------------------------------------+

Layer 1: Data Collection Controls
+----------------------------------+
| - Opt-out for explicit feedback  |
| - Opt-out for implicit tracking  |
| - Opt-out for learning           |
+----------------------------------+
              |
              v
Layer 2: Data Minimization
+----------------------------------+
| - Store only necessary signals   |
| - No raw conversation storage    |
|   (use existing captures table)  |
| - Aggregate where possible       |
+----------------------------------+
              |
              v
Layer 3: Retention Limits
+----------------------------------+
| - Feedback: 90 days              |
| - Quality scores: 180 days       |
| - Preferences: indefinite (agg)  |
| - Metrics: 365 days              |
+----------------------------------+
              |
              v
Layer 4: Local Processing
+----------------------------------+
| - All learning on-device         |
| - No external ML services        |
| - No data export to third parties|
+----------------------------------+
```

### Data Retention Policy

```python
class DataRetentionPolicy:
    """
    Data retention configuration and enforcement.

    POLICY:
    - feedback_explicit: 90 days (then delete)
    - feedback_implicit: 90 days (then delete)
    - quality_scores: 180 days (then delete)
    - user_preferences: indefinite (aggregated data)
    - learning_metrics: 365 days (then delete)
    - ab_tests: 180 days (then delete)
    """

    RETENTION_DAYS = {
        "feedback_explicit": 90,
        "feedback_implicit": 90,
        "quality_scores": 180,
        "learning_metrics": 365,
        "ab_tests": 180
    }

    async def enforce_retention(self, db: LearningDatabase) -> Dict[str, int]:
        """
        Delete data older than retention period.
        Returns count of deleted records per table.
        """
        deleted = {}

        for table, days in self.RETENTION_DAYS.items():
            cutoff = datetime.utcnow() - timedelta(days=days)
            count = await db.delete_before_date(table, cutoff)
            deleted[table] = count

            if count > 0:
                logger.info(
                    "retention_enforced",
                    table=table,
                    deleted_count=count,
                    cutoff_date=cutoff.isoformat()
                )

        return deleted

class PrivacySettings:
    """User-configurable privacy settings"""

    # Feedback collection
    enable_explicit_feedback: bool = True
    enable_implicit_tracking: bool = True

    # Learning features
    enable_personalization: bool = True
    enable_quality_scoring: bool = True

    # Data retention overrides
    custom_retention_days: Optional[int] = None  # Override all retention

    def to_dict(self) -> Dict:
        return asdict(self)
```

### Privacy Configuration

```python
# Environment variables for privacy configuration
PRIVACY_CONFIG = {
    # Feature toggles
    "LEARNING_ENABLE_EXPLICIT_FEEDBACK": True,
    "LEARNING_ENABLE_IMPLICIT_TRACKING": True,
    "LEARNING_ENABLE_PERSONALIZATION": True,

    # Retention periods (days)
    "LEARNING_RETENTION_FEEDBACK": 90,
    "LEARNING_RETENTION_SCORES": 180,
    "LEARNING_RETENTION_METRICS": 365,

    # Data minimization
    "LEARNING_STORE_RAW_FEEDBACK_TEXT": False,  # If False, only store structured data

    # Aggregation
    "LEARNING_AGGREGATE_AFTER_DAYS": 30,  # Aggregate detailed data after N days
}
```

---

## Performance Targets

### Latency Budgets

```
+-----------------------------------------------------------------------+
|                    LATENCY BUDGET ALLOCATION                           |
+-----------------------------------------------------------------------+

Total Request (P95): 1500ms (unchanged from Phase 2)

Learning Overhead (must fit within existing budget):

PRE-PROCESSING (sync, blocking):
+--------------------------------+
| Load preferences    | 5ms     |  <- Redis cache hit
| Tune prompt         | 2ms     |  <- String formatting
| Enhance context     | 3ms     |  <- Score lookup
+--------------------------------+
| Total               | 10ms    |
+--------------------------------+

POST-PROCESSING (async, non-blocking):
+--------------------------------+
| Collect implicit    | 1ms     |  <- Fire and forget
| Start feedback timer| 0ms     |  <- Async task spawn
| Queue scoring       | 1ms     |  <- Background queue
+--------------------------------+
| Total (blocking)    | 2ms     |
+--------------------------------+

BACKGROUND TASKS (async):
+--------------------------------+
| Score exchange      | 50ms    |  <- Embedding + calc
| Update preferences  | 20ms    |  <- DB write
| Aggregate metrics   | 30ms    |  <- Periodic batch
+--------------------------------+
| Total (background)  | 100ms   |
+--------------------------------+

Total Overhead: 12ms (blocking) + 100ms (background)
```

### Throughput Requirements

```
+-----------------------------------------------------------------------+
|                    THROUGHPUT REQUIREMENTS                             |
+-----------------------------------------------------------------------+

Feedback Collection:
- Explicit feedback: < 10ms write latency
- Implicit signals: < 5ms write latency
- Throughput: 100+ feedback/sec (background)

Quality Scoring:
- Score calculation: < 50ms per exchange
- Throughput: 20+ scores/sec (background)
- Batch scoring: 100 exchanges in < 5sec

Preference Learning:
- Preference update: < 20ms
- Full re-learn: < 5sec (periodic batch)
- Preference load: < 5ms (cached)

Analytics:
- Metric aggregation: < 100ms
- Pattern detection: < 500ms
- Report generation: < 2sec
```

### Resource Limits

```python
LEARNING_RESOURCE_LIMITS = {
    # Memory
    "max_preference_cache_mb": 10,      # Max memory for preference cache
    "max_quality_scores_cached": 1000,  # Max scores in memory

    # Database
    "max_feedback_per_session": 100,    # Max feedback records per session
    "max_preferences_total": 500,       # Max preference records

    # Processing
    "max_scoring_queue_size": 100,      # Max pending scores
    "scoring_batch_size": 20,           # Scores per batch

    # Background tasks
    "learning_update_interval_sec": 300,  # 5 minutes
    "retention_check_interval_sec": 86400,  # 24 hours
}
```

---

## A/B Testing Framework

### A/B Test Architecture

```
+-----------------------------------------------------------------------+
|                    A/B TESTING FRAMEWORK                               |
+-----------------------------------------------------------------------+

               User Request
                    |
                    v
          +------------------+
          | Test Assignment  |
          |                  |
          | - Check active   |
          |   tests          |
          | - Assign variant |
          | - Persist        |
          +--------+---------+
                   |
        +----------+----------+
        |                     |
        v                     v
+---------------+     +---------------+
| Control (A)   |     | Treatment (B) |
|               |     |               |
| - Baseline    |     | - New feature |
|   behavior    |     |   or param    |
+-------+-------+     +-------+-------+
        |                     |
        +----------+----------+
                   |
                   v
          +------------------+
          | Outcome Tracking |
          |                  |
          | - Quality score  |
          | - Feedback       |
          | - Engagement     |
          +------------------+
                   |
                   v
          +------------------+
          | Statistical      |
          | Analysis         |
          |                  |
          | - Significance   |
          | - Effect size    |
          | - Confidence     |
          +------------------+
```

### A/B Test Configuration

```python
@dataclass
class ABTest:
    """A/B test configuration"""
    name: str
    description: str

    # Variants
    control_config: Dict[str, Any]
    treatment_config: Dict[str, Any]

    # Traffic allocation
    treatment_percentage: float = 50.0  # % of traffic to treatment

    # Duration
    start_date: datetime
    end_date: Optional[datetime] = None

    # Success metrics
    primary_metric: str = "overall_quality_score"
    secondary_metrics: List[str] = field(default_factory=list)

    # Statistical requirements
    min_sample_size: int = 100
    confidence_level: float = 0.95

    # Status
    is_active: bool = True

class ABTestManager:
    """
    Manage A/B tests for learning improvements.

    EXAMPLE TESTS:
    1. Prompt tuning variants (different personalization approaches)
    2. Retrieval boost weights (topic affinity weight)
    3. Scoring algorithm variants (weight combinations)
    """

    def __init__(self, db: LearningDatabase):
        self.db = db
        self.active_tests: Dict[str, ABTest] = {}

    def assign_variant(
        self,
        test_name: str,
        session_id: str
    ) -> str:
        """
        Assign user to test variant.
        Uses consistent hashing for stable assignment.
        """
        test = self.active_tests.get(test_name)
        if not test or not test.is_active:
            return "control"

        # Consistent hash based on session_id
        hash_value = int(hashlib.md5(
            f"{test_name}:{session_id}".encode()
        ).hexdigest(), 16) % 100

        if hash_value < test.treatment_percentage:
            return "treatment"
        return "control"

    def record_outcome(
        self,
        test_name: str,
        variant: str,
        exchange_id: int,
        outcome_score: float
    ) -> None:
        """Record test outcome for analysis"""
        # Async write to ab_tests table
        pass

    async def analyze_test(
        self,
        test_name: str
    ) -> ABTestResult:
        """
        Analyze test results.

        Returns statistical analysis of test performance.
        """
        test = self.active_tests.get(test_name)
        if not test:
            raise ValueError(f"Unknown test: {test_name}")

        # Get outcomes
        control_scores = await self.db.get_test_outcomes(test_name, "control")
        treatment_scores = await self.db.get_test_outcomes(test_name, "treatment")

        # Statistical analysis
        result = self._analyze_significance(
            control_scores,
            treatment_scores,
            test.confidence_level
        )

        return result

@dataclass
class ABTestResult:
    """A/B test analysis result"""
    test_name: str

    # Sample sizes
    control_n: int
    treatment_n: int

    # Means
    control_mean: float
    treatment_mean: float

    # Effect
    effect_size: float  # (treatment - control) / control

    # Significance
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]

    # Recommendation
    recommendation: str  # "adopt_treatment", "keep_control", "continue_test"
```

### Example A/B Tests

```python
# Test 1: Personalization depth
LENGTH_PREFERENCE_TEST = ABTest(
    name="length_preference_weight",
    description="Test impact of length preference on quality",
    control_config={"length_weight": 0.0},  # Ignore length preference
    treatment_config={"length_weight": 1.0},  # Full length preference
    primary_metric="overall_quality_score",
    min_sample_size=50
)

# Test 2: Topic boost strength
TOPIC_BOOST_TEST = ABTest(
    name="topic_affinity_boost",
    description="Test topic affinity boost in retrieval",
    control_config={"topic_boost": 0.0},   # No boost
    treatment_config={"topic_boost": 0.2},  # 20% boost
    primary_metric="engagement_score",
    min_sample_size=50
)

# Test 3: Quality score weights
QUALITY_WEIGHTS_TEST = ABTest(
    name="quality_score_weights",
    description="Test alternative quality score weights",
    control_config={
        "weights": {"relevance": 0.30, "helpfulness": 0.25, "engagement": 0.20, "clarity": 0.15, "accuracy": 0.10}
    },
    treatment_config={
        "weights": {"relevance": 0.25, "helpfulness": 0.30, "engagement": 0.25, "clarity": 0.10, "accuracy": 0.10}
    },
    primary_metric="explicit_positive_rate",
    min_sample_size=100
)
```

---

## Implementation Plan

### Phase 5.1: Foundation (Week 1)

**Objectives:**
- Set up learning database schema
- Implement feedback collection
- Create quality scoring engine

**Tasks:**
1. Create database migration for learning tables
2. Implement FeedbackCollector class
3. Implement QualityScoringEngine class
4. Add feedback API endpoints
5. Write unit tests for all components
6. Add feature flags for learning system

**Deliverables:**
- `/app/learning/__init__.py`
- `/app/learning/feedback.py`
- `/app/learning/scoring.py`
- `/app/learning/database.py`
- `/tests/learning/test_feedback.py`
- `/tests/learning/test_scoring.py`

**Validation:**
- [ ] Feedback collection < 10ms
- [ ] Quality scoring < 50ms
- [ ] All tests passing
- [ ] No latency regression

---

### Phase 5.2: Analytics (Week 2)

**Objectives:**
- Implement learning analytics pipeline
- Build pattern detection algorithms
- Create preference learning system

**Tasks:**
1. Implement TopicAffinityAnalyzer
2. Implement StyleAnalyzer
3. Implement ErrorPatternAnalyzer
4. Create PreferenceStore with caching
5. Add background task scheduler
6. Write integration tests

**Deliverables:**
- `/app/learning/analytics.py`
- `/app/learning/patterns.py`
- `/app/learning/preferences.py`
- `/tests/learning/test_analytics.py`

**Validation:**
- [ ] Pattern detection running
- [ ] Preferences updating correctly
- [ ] Background tasks stable
- [ ] Memory usage within limits

---

### Phase 5.3: Personalization (Week 3)

**Objectives:**
- Build personalization engine
- Integrate with agents and RAG
- Implement prompt tuning

**Tasks:**
1. Implement PromptTuner class
2. Implement ContextEnhancer class
3. Create LearningMiddleware for agents
4. Integrate with RAG retrieval
5. Add personalization API endpoints
6. Write end-to-end tests

**Deliverables:**
- `/app/learning/personalization.py`
- `/app/learning/middleware.py`
- `/app/learning/integration.py`
- `/tests/learning/test_personalization.py`
- `/tests/learning/test_integration.py`

**Validation:**
- [ ] Prompts adapting to preferences
- [ ] Retrieval boosting working
- [ ] No latency regression
- [ ] Personalization visible in responses

---

### Phase 5.4: Testing & Optimization (Week 4)

**Objectives:**
- Implement A/B testing framework
- Optimize performance
- Add privacy controls

**Tasks:**
1. Implement ABTestManager
2. Add statistical analysis functions
3. Implement DataRetentionPolicy
4. Add privacy configuration
5. Performance optimization
6. Documentation and examples

**Deliverables:**
- `/app/learning/ab_testing.py`
- `/app/learning/privacy.py`
- `/docs/PHASE5_USAGE_GUIDE.md`
- `/examples/learning_demo.py`

**Validation:**
- [ ] A/B tests assignable and trackable
- [ ] Retention policy enforced
- [ ] Privacy controls working
- [ ] All performance targets met

---

## File Structure

```
learning_voice_agent/
+-- app/
|   +-- learning/                     # NEW: Learning system
|   |   +-- __init__.py              # Public API exports
|   |   +-- config.py                # Learning configuration
|   |   +-- feedback.py              # Feedback collection
|   |   +-- scoring.py               # Quality scoring engine
|   |   +-- analytics.py             # Learning analytics pipeline
|   |   +-- patterns.py              # Pattern detection
|   |   +-- preferences.py           # Preference store
|   |   +-- personalization.py       # Personalization engine
|   |   +-- middleware.py            # Agent integration middleware
|   |   +-- ab_testing.py            # A/B testing framework
|   |   +-- privacy.py               # Privacy and retention
|   |   +-- database.py              # Learning database layer
|   |   +-- models.py                # Pydantic models
|   |
|   +-- agents/                       # MODIFIED: Add learning integration
|   |   +-- conversation_agent.py    # Add LearningMiddleware
|   |
|   +-- rag/                          # MODIFIED: Add personalization
|   |   +-- retriever.py             # Add preference boosting
|   |
|   +-- main.py                       # Add feedback endpoints
|
+-- tests/
|   +-- learning/                     # NEW: Learning tests
|   |   +-- test_feedback.py
|   |   +-- test_scoring.py
|   |   +-- test_analytics.py
|   |   +-- test_personalization.py
|   |   +-- test_ab_testing.py
|   |   +-- test_integration.py
|
+-- docs/
|   +-- PHASE5_LEARNING_ARCHITECTURE.md  # This document
|   +-- PHASE5_USAGE_GUIDE.md            # Usage documentation
|
+-- examples/
    +-- learning_demo.py                 # Learning system demo
```

---

## Success Metrics

### Code Quality

- [ ] Type hints on all functions
- [ ] Docstrings on all classes/methods
- [ ] Comprehensive error handling
- [ ] Structured logging throughout
- [ ] Configuration-driven design

### Test Coverage

- [ ] Unit tests for all components
- [ ] Integration tests for pipelines
- [ ] End-to-end learning loop test
- [ ] Performance regression tests

### Documentation

- [ ] Architecture document (this file)
- [ ] Usage guide with examples
- [ ] API reference
- [ ] Privacy documentation

### Performance

- [ ] Pre-processing overhead < 10ms
- [ ] Post-processing overhead < 2ms (blocking)
- [ ] Background processing < 100ms
- [ ] No P95 latency regression

### Functionality

- [ ] Explicit feedback collected and stored
- [ ] Implicit signals captured automatically
- [ ] Quality scores computed for all exchanges
- [ ] Preferences learned and applied
- [ ] A/B tests runnable and analyzable

---

## Conclusion

This architecture provides a comprehensive real-time learning system that:

1. **Collects rich feedback** through both explicit (ratings) and implicit (engagement) signals
2. **Scores response quality** using multi-dimensional metrics with explainable components
3. **Detects patterns** in user preferences through rule-based analytics
4. **Personalizes responses** via dynamic prompt tuning and retrieval augmentation
5. **Enables experimentation** through a built-in A/B testing framework

**Key Design Philosophy:**

- **Simplicity**: Single SQLite database, no external ML services
- **Privacy-first**: Local processing, configurable retention
- **Non-invasive**: Middleware pattern for agent integration
- **Measurable**: Quality metrics and A/B testing for validation

The implementation plan spans 4 weeks, with clear deliverables and validation criteria for each phase. The system is designed to work incrementally - even partial implementation provides value, and full implementation enables sophisticated personalization.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Status:** Architecture Design Complete - Ready for Implementation
**Approvers:** [To be filled]
