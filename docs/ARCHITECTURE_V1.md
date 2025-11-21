# Learning Voice Agent - v1.0 Architecture Documentation

**Version:** 1.0.0
**Last Updated:** 2025-11-21
**Status:** Production-ready with known limitations

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Descriptions](#component-descriptions)
4. [Data Flow](#data-flow)
5. [API Contracts](#api-contracts)
6. [Database Schema](#database-schema)
7. [Deployment Architecture](#deployment-architecture)
8. [Security Model](#security-model)
9. [Performance Characteristics](#performance-characteristics)
10. [Known Limitations](#known-limitations)

---

## System Overview

### Purpose
Learning Voice Agent is an AI-powered voice conversation system designed to capture and develop learning insights through natural voice interactions. It combines modern web technologies with state-of-the-art AI models to provide a seamless learning companion experience.

### Core Value Propositions
- **Sub-2-second conversation loops** for natural interaction
- **Multi-channel support** (browser WebSocket, phone via Twilio)
- **Intelligent context management** using Redis
- **Instant semantic search** with SQLite FTS5
- **Offline-capable PWA** for continuous availability

### Technology Stack

**Backend:**
- Python 3.11+ with async/await
- FastAPI 0.109.0 for REST and WebSocket APIs
- Uvicorn with standard extensions for ASGI server

**AI Models:**
- Anthropic Claude Haiku (claude-3-haiku-20240307) for conversation
- OpenAI Whisper (whisper-1) for audio transcription

**Data Layer:**
- Redis 5.0.1 for session state (30-minute TTL)
- SQLite 3 with FTS5 for persistent storage and full-text search
- aiosqlite 0.19.0 for async database operations

**Frontend:**
- Vue 3 PWA with Composition API
- Service Workers for offline support
- MediaRecorder API for audio capture

**Integrations:**
- Twilio 8.11.0 for phone call handling
- TwiML for voice responses

---

## Architecture Diagrams

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Web Browser │  Mobile PWA  │  Phone Call  │   REST Client      │
│  (Vue 3)     │  (Installed) │  (Twilio)    │   (API)            │
│              │              │              │                    │
│  WebSocket   │  WebSocket   │  Webhooks    │   HTTP/REST        │
└──────┬───────┴───────┬──────┴───────┬──────┴────────┬───────────┘
       │               │              │               │
       └───────────────┴──────────────┴───────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────────────┐
       │              FASTAPI APPLICATION LAYER                    │
       │                  (app/main.py)                           │
       ├──────────────────────────────────────────────────────────┤
       │  • CORS Middleware                                       │
       │  • Lifecycle Management (startup/shutdown)               │
       │  • Route Handling (REST + WebSocket + Twilio)            │
       │  • Dependency Injection                                  │
       │  • Background Tasks                                      │
       └──────┬───────────────────────┬────────────────┬──────────┘
              │                       │                │
              ▼                       ▼                ▼
    ┌─────────────────┐   ┌──────────────────┐  ┌─────────────────┐
    │  Conversation   │   │  Audio Pipeline  │  │  State Manager  │
    │  Handler        │   │  (Whisper)       │  │  (Redis)        │
    │  (Claude)       │   │                  │  │                 │
    └────────┬────────┘   └─────────┬────────┘  └────────┬────────┘
             │                      │                     │
             └──────────────────────┴─────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Database Layer     │
                   │   (SQLite + FTS5)    │
                   └──────────────────────┘
```

### Request Flow Diagram

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     │ 1. Send audio/text
     ▼
┌─────────────────┐
│  FastAPI Route  │ ──────────────┐
└────┬────────────┘               │
     │                            │
     │ 2. Transcribe (if audio)   │ 5. Background:
     ▼                            │    Update state
┌──────────────────┐              │    Save to DB
│  Audio Pipeline  │              │
└────┬─────────────┘              │
     │                            │
     │ 3. Get context             │
     ▼                            ▼
┌──────────────────┐         ┌────────────────┐
│  State Manager   │<────────│  Update State  │
│  (Redis Cache)   │         │  (Background)  │
└────┬─────────────┘         └────────────────┘
     │                                 │
     │ 4. Generate response            │
     ▼                                 ▼
┌──────────────────────┐         ┌────────────┐
│ Conversation Handler │         │  Database  │
│  (Claude Haiku)      │         │  (SQLite)  │
└────┬─────────────────┘         └────────────┘
     │
     │ 5. Return response
     ▼
┌──────────┐
│  Client  │
└──────────┘

Total Time: < 2 seconds
```

### Data Flow: Voice Conversation

```
[User Voice]
    │
    │ MediaRecorder → Base64
    ▼
[WebSocket /ws/{session_id}]
    │
    │ {"type": "audio", "audio": "base64..."}
    ▼
[Audio Pipeline]
    │
    ├─► Detect format (WAV/MP3/OGG)
    ├─► Validate size/duration
    └─► Whisper API transcription
         │
         │ < 800ms
         ▼
    [Transcribed Text]
         │
         ▼
[State Manager - Get Context]
    │
    ├─► Redis: session:{id}:context
    └─► Returns last 5 exchanges
         │
         ▼
[Conversation Handler]
    │
    ├─► Format system prompt
    ├─► Add conversation context
    ├─► Claude Haiku API call
    └─► Post-process response
         │
         │ < 900ms
         ▼
    [Agent Response]
         │
         ├─► Update Redis context
         ├─► Save to SQLite (background)
         └─► Return to client
              │
              ▼
         [WebSocket Send]
              │
              ▼
         [Client Receives]
```

---

## Component Descriptions

### 1. FastAPI Application (`app/main.py`)

**Purpose:** Integration layer and entry point for all requests.

**Key Features:**
- Lifecycle management with `@asynccontextmanager`
- CORS middleware for cross-origin requests
- REST endpoints for API clients
- WebSocket endpoint for real-time communication
- Twilio webhook integration
- Background task processing

**Dependencies:**
- `app.config.settings` - Configuration singleton
- `app.database.db` - Database instance
- `app.state_manager.state_manager` - Redis state
- `app.conversation_handler.conversation_handler` - AI logic
- `app.audio_pipeline.audio_pipeline` - Audio processing

**Pattern:** Dependency injection with singleton services

### 2. Conversation Handler (`app/conversation_handler.py`)

**Purpose:** Claude API integration for intelligent conversation.

**System Prompt Philosophy:**
```
"You are a personal learning companion helping capture and develop ideas."

Core Behaviors:
- Ask ONE clarifying question for vague responses
- Connect new ideas to previous topics
- Keep responses under 3 sentences
- Never lecture unless explicitly asked
- Mirror user's energy level
```

**Key Methods:**
- `generate_response(user_text, context, metadata)` - Main conversation logic
- `detect_intent(text)` - Simple intent classification
- `create_summary(exchanges)` - End-of-session summary
- `_should_add_followup(user_text, response)` - Intelligent fallback

**Error Handling:**
- Rate limit errors → "I need a moment to catch up"
- API errors → "I'm having trouble connecting"
- Unexpected errors → "Something went wrong on my end"

**Configuration:**
- Model: claude-3-haiku-20240307
- Max tokens: 150
- Temperature: 0.7
- Target latency: < 900ms

### 3. Audio Pipeline (`app/audio_pipeline.py`)

**Purpose:** Audio transcription with format detection and validation.

**Architecture Pattern:** Strategy pattern for multiple transcription providers.

**Audio Format Support:**
- WAV (RIFF header)
- MP3 (ID3 or 0xFFxFB header)
- OGG (OggS header)
- WebM (webm in first 40 bytes)

**Processing Flow:**
1. **Format Detection** - Magic byte analysis
2. **Validation** - Size (< 25MB), format support
3. **Transcription** - Whisper API call
4. **Post-processing** - Clean artifacts, normalize text

**Key Classes:**
- `AudioData` - Data class for audio metadata
- `TranscriptionStrategy` - Abstract base for providers
- `WhisperStrategy` - OpenAI Whisper implementation
- `AudioPipeline` - Facade for audio operations

**Performance:**
- Target: < 800ms for typical 5-second audio
- Actual: Varies with Whisper API latency

### 4. State Manager (`app/state_manager.py`)

**Purpose:** Redis-based conversation context and session management.

**Pattern:** Cache-aside with TTL-based expiration.

**Key Responsibilities:**
- Store conversation context (last N exchanges)
- Track session metadata (created_at, exchange_count)
- Validate session activity
- Clean up expired sessions

**Redis Keys:**
```
session:{session_id}:context     → JSON array of exchanges
session:{session_id}:metadata    → JSON object of metadata
```

**Configuration:**
- TTL: 1800 seconds (30 minutes)
- Max context exchanges: 5
- Session timeout: 180 seconds (3 minutes)
- Max connections: 50

**Methods:**
- `get_conversation_context(session_id)` - Retrieve recent exchanges
- `update_conversation_context(session_id, user, agent)` - Add exchange
- `is_session_active(session_id)` - Check activity timeout
- `end_session(session_id)` - Clean up session data

### 5. Database Layer (`app/database.py`)

**Purpose:** Persistent storage with full-text search.

**Pattern:** Repository pattern with async operations.

**Key Features:**
- SQLite with FTS5 virtual table
- Automatic FTS index synchronization via triggers
- BM25 ranking for search relevance
- Async connection pooling

**Tables:**

```sql
-- Main captures table
CREATE TABLE captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_text TEXT NOT NULL,
    agent_text TEXT NOT NULL,
    metadata TEXT  -- JSON string
);

-- FTS5 virtual table
CREATE VIRTUAL TABLE captures_fts USING fts5(
    session_id UNINDEXED,
    user_text,
    agent_text,
    content=captures,
    content_rowid=id
);

-- Index for session queries
CREATE INDEX idx_session_timestamp
ON captures(session_id, timestamp DESC);
```

**Methods:**
- `save_exchange()` - Insert with automatic FTS indexing
- `get_session_history()` - Retrieve by session with limit
- `search_captures()` - Full-text search with snippets
- `get_stats()` - System statistics

### 6. Twilio Handler (`app/twilio_handler.py`)

**Purpose:** Phone call integration via Twilio webhooks.

**Pattern:** Webhook handler with TwiML generation.

**Key Features:**
- Request signature validation
- TwiML response generation
- Speech recognition integration
- Call state management

**Endpoints:**
- `POST /twilio/voice` - Incoming call handler
- `POST /twilio/process-speech` - Speech result processing
- `POST /twilio/recording` - Recording handler

**TwiML Configuration:**
- Voice: Polly.Joanna-Neural (high quality)
- Language: en-US
- Enhanced speech recognition
- Auto timeout detection

**Session IDs:** `twilio_{CallSid}` for tracking

### 7. Configuration (`app/config.py`)

**Purpose:** Centralized configuration with environment validation.

**Pattern:** Singleton with Pydantic validation.

**Environment Variables:**

| Category | Variable | Default | Required |
|----------|----------|---------|----------|
| **AI APIs** | ANTHROPIC_API_KEY | - | Yes |
| | OPENAI_API_KEY | - | Yes |
| **Twilio** | TWILIO_ACCOUNT_SID | None | No |
| | TWILIO_AUTH_TOKEN | None | No |
| | TWILIO_PHONE_NUMBER | None | No |
| **Database** | DATABASE_URL | sqlite:///./learning_captures.db | No |
| **Redis** | REDIS_URL | redis://localhost:6379 | Yes |
| | REDIS_TTL | 1800 | No |
| **Server** | HOST | 0.0.0.0 | No |
| | PORT | 8000 | No |
| **Audio** | WHISPER_MODEL | whisper-1 | No |
| | MAX_AUDIO_DURATION | 60 | No |
| **Claude** | CLAUDE_MODEL | claude-3-haiku-20240307 | No |
| | CLAUDE_MAX_TOKENS | 150 | No |
| | CLAUDE_TEMPERATURE | 0.7 | No |
| **Session** | SESSION_TIMEOUT | 180 | No |
| | MAX_CONTEXT_EXCHANGES | 5 | No |

### 8. Frontend PWA (`static/index.html`)

**Purpose:** Vue 3 PWA for voice interaction.

**Features:**
- MediaRecorder API for voice capture
- WebSocket for real-time communication
- Service Worker for offline support
- InstallPrompt for app installation
- Search interface with Cmd+K shortcut

**Architecture:**
- Vue 3 Composition API
- No build step (CDN-based)
- Progressive enhancement

---

## Data Flow

### Voice Conversation Flow (Detailed)

1. **User Interaction**
   - User clicks "Hold to Talk" button
   - MediaRecorder starts capturing audio
   - Audio chunks buffered in memory

2. **Audio Transmission**
   - User releases button
   - Audio blob converted to Base64
   - WebSocket sends: `{"type": "audio", "audio": "..."}`

3. **Server Processing**
   - FastAPI WebSocket handler receives message
   - Extracts Base64 audio data
   - Passes to `audio_pipeline.transcribe_base64()`

4. **Audio Transcription**
   - Decode Base64 to bytes
   - Detect audio format (magic bytes)
   - Validate size and format
   - Call Whisper API
   - Clean transcript (artifacts, punctuation)
   - Return text

5. **Context Retrieval**
   - Query Redis: `session:{id}:context`
   - Get last 5 exchanges
   - Format for Claude prompt

6. **AI Response Generation**
   - Build system prompt
   - Format conversation context
   - Call Claude Haiku API
   - Post-process response
   - Add follow-up if needed

7. **State Updates (Background)**
   - Add exchange to Redis context
   - Trim to max exchanges (5)
   - Save to SQLite database
   - Update session metadata

8. **Response Delivery**
   - WebSocket sends: `{"type": "response", "user_text": "...", "agent_text": "...", "intent": "..."}`
   - Client displays response
   - Ready for next interaction

**Performance Checkpoints:**
- Transcription: < 800ms
- Claude response: < 900ms
- Total loop: < 2 seconds

---

## API Contracts

### REST Endpoints

#### POST /api/conversation

Process text or audio input and generate response.

**Request:**
```json
{
  "session_id": "abc-123",  // optional
  "text": "I'm learning about AI",  // optional
  "audio_base64": "UklGR..."  // optional
}
```

**Response:**
```json
{
  "session_id": "abc-123",
  "user_text": "I'm learning about AI",
  "agent_text": "That's exciting! What aspect of AI interests you most?",
  "intent": "statement",
  "timestamp": "2025-11-21T10:30:00Z"
}
```

**Status Codes:**
- 200: Success
- 400: No input provided
- 500: Processing error

#### POST /api/search

Search conversation history with FTS5.

**Request:**
```json
{
  "query": "machine learning",
  "limit": 20  // optional, default 20
}
```

**Response:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "id": 42,
      "session_id": "abc-123",
      "timestamp": "2025-11-21T10:30:00",
      "user_text": "I'm studying machine learning",
      "agent_text": "What aspect interests you?",
      "user_snippet": "studying <mark>machine learning</mark>",
      "agent_snippet": "What aspect interests you?"
    }
  ],
  "count": 1
}
```

#### GET /api/stats

System statistics and monitoring.

**Response:**
```json
{
  "database": {
    "total_captures": 1234,
    "unique_sessions": 45,
    "last_capture": "2025-11-21T10:30:00"
  },
  "sessions": {
    "active": 3,
    "ids": ["abc-123", "def-456", "ghi-789"]
  }
}
```

#### GET /api/session/{session_id}/history

Get conversation history for a session.

**Parameters:**
- `session_id` (path): Session identifier
- `limit` (query): Max results (default 20)

**Response:**
```json
{
  "session_id": "abc-123",
  "history": [
    {
      "id": 42,
      "timestamp": "2025-11-21T10:30:00",
      "user_text": "Hello",
      "agent_text": "Hi there!",
      "metadata": "{\"source\": \"api\"}"
    }
  ],
  "count": 1
}
```

### WebSocket Protocol

#### Connection

```
ws://localhost:8000/ws/{session_id}
```

#### Message Types

**Audio Input:**
```json
{
  "type": "audio",
  "audio": "UklGR..."  // Base64 encoded
}
```

**Text Input:**
```json
{
  "type": "text",
  "text": "Hello, I'm learning about AI"
}
```

**End Conversation:**
```json
{
  "type": "end"
}
```

**Ping:**
```json
{
  "type": "ping"
}
```

#### Server Responses

**Conversation Response:**
```json
{
  "type": "response",
  "user_text": "Hello",
  "agent_text": "Hi there!",
  "intent": "statement"
}
```

**Summary:**
```json
{
  "type": "summary",
  "text": "We explored: learning, AI, concepts. Great conversation!"
}
```

### Twilio Webhooks

#### POST /twilio/voice

Incoming call handler.

**Request (Form Data):**
- `CallSid`: Unique call identifier
- `From`: Caller phone number
- `To`: Twilio number called
- `CallStatus`: ringing | in-progress | completed

**Response (TwiML):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" language="en-US" timeout="3"
          action="/twilio/process-speech?session_id=twilio_CA123"
          enhanced="true">
    <Say voice="Polly.Joanna-Neural">
      Hello! I'm your learning companion.
      Tell me what you're working on today.
    </Say>
  </Gather>
  <Say>I didn't catch that. Let's try again.</Say>
  <Redirect>/twilio/voice?session_id=twilio_CA123</Redirect>
</Response>
```

#### POST /twilio/process-speech

Speech recognition result handler.

**Request (Form Data):**
- `SpeechResult`: Transcribed text
- `Confidence`: Recognition confidence (0-1)

**Response:** TwiML with agent response or error handling

---

## Database Schema

### Tables

```sql
-- Main conversation captures
CREATE TABLE captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_text TEXT NOT NULL,
    agent_text TEXT NOT NULL,
    metadata TEXT  -- JSON: {"source": "api|websocket|twilio"}
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE captures_fts USING fts5(
    session_id UNINDEXED,  -- Not searchable, but included
    user_text,              -- Searchable with BM25 ranking
    agent_text,             -- Searchable with BM25 ranking
    content=captures,       -- External content table
    content_rowid=id        -- Link to captures.id
);
```

### Indexes

```sql
-- Optimize session history queries
CREATE INDEX idx_session_timestamp
ON captures(session_id, timestamp DESC);
```

### Triggers (FTS Synchronization)

```sql
-- Insert trigger
CREATE TRIGGER captures_ai AFTER INSERT ON captures BEGIN
    INSERT INTO captures_fts(rowid, session_id, user_text, agent_text)
    VALUES (new.id, new.session_id, new.user_text, new.agent_text);
END;

-- Delete trigger
CREATE TRIGGER captures_ad AFTER DELETE ON captures BEGIN
    DELETE FROM captures_fts WHERE rowid = old.id;
END;

-- Update trigger
CREATE TRIGGER captures_au AFTER UPDATE ON captures BEGIN
    UPDATE captures_fts
    SET user_text = new.user_text, agent_text = new.agent_text
    WHERE rowid = new.id;
END;
```

### Relationships

```
┌─────────────┐
│  captures   │
│─────────────│
│ id (PK)     │◄────────┐
│ session_id  │         │
│ timestamp   │         │ content_rowid
│ user_text   │         │ (virtual FK)
│ agent_text  │         │
│ metadata    │         │
└─────────────┘         │
                        │
               ┌────────┴────────┐
               │  captures_fts   │
               │─────────────────│
               │ rowid           │
               │ session_id      │
               │ user_text       │
               │ agent_text      │
               └─────────────────┘
```

### Redis Schema

```
Key Pattern: session:{session_id}:context
Type: String (JSON)
TTL: 1800 seconds
Value: [
  {
    "timestamp": "2025-11-21T10:30:00Z",
    "user": "Hello",
    "agent": "Hi there!"
  },
  ...
]

Key Pattern: session:{session_id}:metadata
Type: String (JSON)
TTL: 1800 seconds
Value: {
  "created_at": "2025-11-21T10:00:00Z",
  "last_activity": "2025-11-21T10:30:00Z",
  "exchange_count": 5,
  "source": "api|websocket|twilio"
}
```

---

## Deployment Architecture

### Development Environment

```
┌──────────────────────────────────────┐
│  Developer Machine                   │
├──────────────────────────────────────┤
│  Python 3.11+                        │
│  Redis (local or Docker)             │
│  SQLite (file-based)                 │
│                                      │
│  uvicorn app.main:app --reload       │
│  Port: 8000                          │
└──────────────────────────────────────┘
```

### Docker Compose Deployment

```
┌─────────────────────────────────────────────────────┐
│  Docker Host                                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐      ┌──────────────┐            │
│  │   FastAPI    │      │    Redis     │            │
│  │  Container   │◄────►│  Container   │            │
│  │              │      │              │            │
│  │  Port: 8000  │      │  Port: 6379  │            │
│  └──────┬───────┘      └──────────────┘            │
│         │                                           │
│         │ Volume Mount                              │
│         ▼                                           │
│  ┌──────────────┐                                   │
│  │ SQLite File  │                                   │
│  │ (persistent) │                                   │
│  └──────────────┘                                   │
│                                                     │
│  Optional:                                          │
│  ┌──────────────┐      ┌──────────────┐            │
│  │ Cloudflare   │      │ Litestream   │            │
│  │   Tunnel     │      │   Backup     │            │
│  └──────────────┘      └──────────────┘            │
└─────────────────────────────────────────────────────┘
```

### Railway Production Deployment

```
┌─────────────────────────────────────────────────────────┐
│  Railway.app                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────────────────────────┐                 │
│  │  FastAPI Service                   │                 │
│  │  ─────────────────────────────────│                 │
│  │  • Dockerfile build                │                 │
│  │  • Auto-scaling (1-5 instances)    │                 │
│  │  • Health checks on /              │                 │
│  │  • Restart on failure (max 10)     │                 │
│  │  • Environment variables injected  │                 │
│  └────────────┬───────────────────────┘                 │
│               │                                         │
│               │                                         │
│  ┌────────────▼───────────────────────┐                 │
│  │  Redis Plugin                      │                 │
│  │  ─────────────────────────────────│                 │
│  │  • Managed Redis instance          │                 │
│  │  • Automatic REDIS_URL injection   │                 │
│  │  • Persistent storage              │                 │
│  └────────────────────────────────────┘                 │
│                                                         │
│  ┌────────────────────────────────────┐                 │
│  │  Volumes (Persistent)              │                 │
│  │  ─────────────────────────────────│                 │
│  │  /app/learning_captures.db         │                 │
│  └────────────────────────────────────┘                 │
│                                                         │
│  ┌────────────────────────────────────┐                 │
│  │  Networking                        │                 │
│  │  ─────────────────────────────────│                 │
│  │  • HTTPS (automatic SSL)           │                 │
│  │  • Custom domain support           │                 │
│  │  • WebSocket support               │                 │
│  └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
                     │
                     │ External APIs
                     ▼
┌─────────────────────────────────────────────────────────┐
│  External Services                                      │
├─────────────────────────────────────────────────────────┤
│  • Anthropic Claude API (conversation)                  │
│  • OpenAI Whisper API (transcription)                   │
│  • Twilio (optional phone integration)                  │
└─────────────────────────────────────────────────────────┘
```

---

## Security Model

### Authentication & Authorization

**Current State (v1.0):**
- ⚠️ **No authentication implemented**
- Public API endpoints
- Session-based tracking via UUIDs
- Twilio request signature validation

**Planned (v2.0):**
- JWT-based authentication
- User accounts
- API key management
- Rate limiting per user

### Data Security

**API Keys:**
- Stored in environment variables
- Never committed to repository
- Loaded via python-dotenv

**Network Security:**
- HTTPS in production (Railway automatic)
- CORS configured for allowed origins
- WebSocket connections over WSS

**Data Privacy:**
- No PII collected by default
- Conversation data stored locally
- Phone numbers hashed in logs (Twilio)

**Twilio Webhook Validation:**
```python
# Request signature validation
validator = RequestValidator(TWILIO_AUTH_TOKEN)
is_valid = validator.validate(
    url=request.url,
    params=request.form,
    signature=request.headers['X-Twilio-Signature']
)
```

### Known Security Limitations

1. **No Rate Limiting** - Can be abused for API costs
2. **No Input Sanitization** - XSS potential in stored text
3. **No SQL Injection Protection** - Using parameterized queries (safe)
4. **No Authentication** - Anyone can access API
5. **No Audit Logging** - No tracking of access patterns

---

## Performance Characteristics

### Measured Performance (v1.0)

**API Response Times:**
- Health check (`/`): < 10ms
- Stats endpoint (`/api/stats`): 20-50ms
- Search endpoint (`/api/search`): 50-200ms (depends on index size)
- Conversation endpoint (`/api/conversation`): 1500-2500ms

**Conversation Loop Breakdown:**
```
Total: ~1800ms
├─ Audio transcription: 600-800ms (Whisper API)
├─ Context retrieval: 5-10ms (Redis)
├─ Claude response: 700-900ms (Haiku API)
├─ State update: 5-10ms (Redis)
└─ DB save (background): 20-50ms (non-blocking)
```

**WebSocket Latency:**
- Connection establishment: < 50ms
- Message round-trip (ping): < 20ms
- Audio processing: Same as REST (1800ms avg)

**Database Performance:**
- Insert: 5-15ms
- Session history query: 5-20ms
- FTS5 search: 20-100ms (grows with data)
- Stats query: 10-30ms

**Memory Usage:**
- Base application: ~50MB
- Per active session: ~2MB (Redis + in-memory)
- With 10 active sessions: ~70MB

**Redis Performance:**
- Get operation: < 1ms
- Set operation: < 2ms
- Scan operation: 5-20ms

### Scalability Limits

**Current Constraints:**
1. **Single Redis Instance** - Max ~50 concurrent sessions
2. **SQLite Database** - Concurrent writes limited
3. **File-based Storage** - No horizontal scaling
4. **Single Process** - Limited to one CPU core (async helps)

**Bottlenecks:**
1. **AI API Latency** - Claude + Whisper determine response time
2. **Redis Network** - RTT for context retrieval
3. **SQLite Writes** - Can become slow with large dataset
4. **Memory** - Context accumulation in Redis

**Recommended Limits (v1.0):**
- Concurrent sessions: < 50
- Total captures: < 1M (SQLite FTS5 limit)
- Audio file size: < 25MB
- Audio duration: < 60 seconds

### Performance Optimization Strategies

**Already Implemented:**
- Async/await throughout
- Connection pooling (Redis)
- Background tasks for DB writes
- FTS5 for fast search
- Index on session+timestamp
- TTL-based cleanup in Redis

**Future Optimizations:**
- Redis caching for frequent queries
- Response streaming for Claude
- Audio chunking for long files
- CDN for static assets
- PostgreSQL for better concurrency

---

## Known Limitations

### Functional Limitations

1. **No User Accounts**
   - All conversations anonymous
   - No cross-session user tracking
   - Cannot sync across devices

2. **Limited Context Window**
   - Only last 5 exchanges retained in context
   - No long-term memory beyond 30 minutes
   - No semantic memory or RAG

3. **Single Language Support**
   - English only (hardcoded in Whisper and prompts)
   - No internationalization

4. **Audio Format Constraints**
   - Max 60 seconds duration
   - Max 25MB file size
   - Limited format support (WAV, MP3, OGG, WebM)

5. **No Vision Support**
   - Text and audio only
   - Cannot process images or documents

### Technical Limitations

1. **SQLite Concurrency**
   - Limited concurrent writes
   - Not suitable for high-traffic production
   - File locking issues possible

2. **Redis as SPOF**
   - No redundancy
   - Session loss on Redis failure
   - No persistence configured

3. **No Horizontal Scaling**
   - Single process deployment
   - Cannot load balance across instances
   - Stateful WebSocket connections

4. **Logging**
   - Using print() statements instead of structured logging
   - No log aggregation
   - Difficult to debug in production

5. **Error Handling**
   - Basic exception handling
   - No retry logic for transient failures
   - No circuit breakers for API calls

### Integration Limitations

1. **Twilio**
   - Phone integration optional
   - No call recording storage
   - No voicemail support

2. **PWA**
   - No push notifications
   - Limited offline capabilities
   - No background sync

3. **Search**
   - Simple keyword matching
   - No semantic search
   - No faceted search
   - No relevance tuning

### Cost Considerations

**Per-conversation costs (approximate):**
- Whisper: $0.006 per minute (~$0.0005 per 5-second clip)
- Claude Haiku: $0.00025 per request (150 tokens)
- **Total per exchange: ~$0.00075**

**Monthly costs at moderate usage (100 exchanges/day):**
- AI APIs: ~$2.25/month
- Railway hosting: $5-10/month
- Twilio (optional): $20/month
- **Total: $27-32/month**

### Migration Path to v2.0

See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for detailed migration strategy to address these limitations.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-21 | Initial architecture documentation |

---

**Next Steps:**
- Review [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for v2.0 strategy
- See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for detailed API reference
- Check [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) for local setup
