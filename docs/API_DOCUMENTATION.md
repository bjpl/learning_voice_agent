# API Documentation - Learning Voice Agent v1.0

**Version:** 1.0.0
**Base URL:** `https://your-domain.railway.app`
**Protocol:** HTTPS
**Format:** JSON

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket Protocol](#websocket-protocol)
5. [Twilio Webhooks](#twilio-webhooks)
6. [Request/Response Schemas](#requestresponse-schemas)
7. [Error Handling](#error-handling)
8. [Rate Limits](#rate-limits)
9. [Example Usage](#example-usage)

---

## Overview

The Learning Voice Agent provides three types of APIs:

1. **REST API** - Traditional HTTP endpoints for conversation, search, and stats
2. **WebSocket API** - Real-time bidirectional communication for voice interactions
3. **Twilio Webhooks** - Phone call integration (optional)

### Base URL

```
Production: https://your-app.railway.app
Development: http://localhost:8000
```

### Content Types

All REST endpoints accept and return `application/json` unless specified otherwise.

---

## Authentication

### v1.0 Status

**Authentication:** ⚠️ Not implemented in v1.0

All endpoints are currently public. Authentication will be added in v2.0.

### Planned v2.0

```http
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <API_KEY>
```

---

## REST API Endpoints

### Health Check

Check system health and API status.

```http
GET /
```

**Response:**

```json
{
  "status": "healthy",
  "service": "Learning Voice Agent",
  "version": "1.0.0",
  "endpoints": {
    "websocket": "/ws/{session_id}",
    "twilio": "/twilio/voice",
    "search": "/api/search",
    "stats": "/api/stats"
  }
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is down

---

### POST /api/conversation

Process text or audio input and generate an AI response.

**Request Body:**

```json
{
  "session_id": "abc-123",        // Optional: Session identifier
  "text": "Hello, I'm learning",  // Optional: Text input
  "audio_base64": "UklGR..."      // Optional: Base64 encoded audio
}
```

**Notes:**
- Must provide either `text` or `audio_base64`
- If both provided, `audio_base64` takes precedence
- `session_id` auto-generated if not provided

**Response:**

```json
{
  "session_id": "abc-123",
  "user_text": "Hello, I'm learning about AI",
  "agent_text": "That's exciting! What aspect of AI interests you most?",
  "intent": "statement",
  "timestamp": "2025-11-21T10:30:00.123Z"
}
```

**Intent Types:**
- `statement` - General statement
- `question` - User asking a question
- `listing` - User listing items
- `reflection` - Thinking/feeling statement
- `end_conversation` - Goodbye/ending

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - No input provided
- `422 Unprocessable Entity` - Invalid request format
- `500 Internal Server Error` - Processing error

**Example:**

```bash
curl -X POST https://app.railway.app/api/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-123",
    "text": "I am learning about machine learning"
  }'
```

---

### POST /api/search

Search conversation history using full-text search.

**Request Body:**

```json
{
  "query": "machine learning",  // Required: Search query
  "limit": 20                   // Optional: Max results (1-100, default 20)
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
      "user_text": "I'm studying machine learning algorithms",
      "agent_text": "What type of algorithms interest you?",
      "user_snippet": "studying <mark>machine learning</mark> algorithms",
      "agent_snippet": "What type of algorithms interest you?"
    }
  ],
  "count": 1
}
```

**Search Features:**
- Full-text search with BM25 ranking
- Searches both user and agent text
- Highlighted snippets with `<mark>` tags
- Results ordered by relevance

**Status Codes:**
- `200 OK` - Success (even if no results)
- `400 Bad Request` - Invalid query
- `422 Unprocessable Entity` - Validation error

**Example:**

```bash
curl -X POST https://app.railway.app/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "limit": 10
  }'
```

---

### GET /api/stats

Get system statistics and monitoring data.

**Request:**

```http
GET /api/stats
```

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

**Fields:**
- `database.total_captures` - Total conversation exchanges stored
- `database.unique_sessions` - Number of unique sessions
- `database.last_capture` - Timestamp of most recent capture
- `sessions.active` - Currently active sessions (< 30 min old)
- `sessions.ids` - List of active session IDs

**Status Codes:**
- `200 OK` - Success

**Example:**

```bash
curl https://app.railway.app/api/stats
```

---

### GET /api/session/{session_id}/history

Retrieve conversation history for a specific session.

**Parameters:**
- `session_id` (path) - Session identifier
- `limit` (query) - Maximum results (default: 20)

**Request:**

```http
GET /api/session/abc-123/history?limit=10
```

**Response:**

```json
{
  "session_id": "abc-123",
  "history": [
    {
      "id": 1,
      "timestamp": "2025-11-21T10:25:00",
      "user_text": "Hello",
      "agent_text": "Hi there! What are you learning today?",
      "metadata": "{\"source\": \"api\"}"
    },
    {
      "id": 2,
      "timestamp": "2025-11-21T10:26:00",
      "user_text": "I'm studying AI",
      "agent_text": "That's exciting! What aspect interests you?",
      "metadata": "{\"source\": \"api\"}"
    }
  ],
  "count": 2
}
```

**Notes:**
- Results ordered chronologically (oldest first)
- Metadata is JSON string (parse for structured data)

**Status Codes:**
- `200 OK` - Success (even if empty)
- `404 Not Found` - Session not found

**Example:**

```bash
curl https://app.railway.app/api/session/abc-123/history?limit=5
```

---

## WebSocket Protocol

### Connection

Establish a WebSocket connection to receive real-time updates.

**URL:**

```
ws://localhost:8000/ws/{session_id}
wss://app.railway.app/ws/{session_id}
```

**Example (JavaScript):**

```javascript
const ws = new WebSocket('wss://app.railway.app/ws/my-session');

ws.onopen = () => {
  console.log('Connected!');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

---

### Client → Server Messages

#### Send Audio

```json
{
  "type": "audio",
  "audio": "UklGRiQAAABXQVZFZm10..."  // Base64 encoded audio
}
```

**Audio Requirements:**
- Format: WAV, MP3, OGG, or WebM
- Max duration: 60 seconds
- Max size: 25MB (after encoding)

**Example (JavaScript):**

```javascript
// Record audio with MediaRecorder
const mediaRecorder = new MediaRecorder(stream);
const chunks = [];

mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

mediaRecorder.onstop = async () => {
  const blob = new Blob(chunks);
  const reader = new FileReader();

  reader.onload = () => {
    const base64 = reader.result.split(',')[1];
    ws.send(JSON.stringify({
      type: 'audio',
      audio: base64
    }));
  };

  reader.readAsDataURL(blob);
};
```

#### Send Text

```json
{
  "type": "text",
  "text": "Hello, I'm learning about distributed systems"
}
```

#### End Conversation

```json
{
  "type": "end"
}
```

**Response:** Conversation summary and WebSocket close

#### Ping (Keep-Alive)

```json
{
  "type": "ping"
}
```

**Response:** None (connection maintained)

---

### Server → Client Messages

#### Conversation Response

```json
{
  "type": "response",
  "user_text": "Hello, I'm learning about AI",
  "agent_text": "That's exciting! What aspect interests you?",
  "intent": "statement"
}
```

**Typical Flow:**
1. Client sends audio/text
2. Server transcribes (if audio)
3. Server generates response
4. Server sends response message
5. Client displays to user

#### Summary (End of Conversation)

```json
{
  "type": "summary",
  "text": "We explored: machine learning, neural networks, transformers. Great conversation!"
}
```

**Triggered by:**
- Client sends `{"type": "end"}`
- Client says "goodbye" or similar

---

## Twilio Webhooks

Twilio integration for phone call handling (optional feature).

### POST /twilio/voice

Handle incoming Twilio phone calls.

**Request (Form Data):**

```
CallSid=CA1234567890abcdef
From=+1234567890
To=+0987654321
CallStatus=ringing
```

**Response (TwiML XML):**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech"
          language="en-US"
          timeout="3"
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

**Notes:**
- Session ID: `twilio_{CallSid}`
- Uses Polly.Joanna-Neural voice
- Enhanced speech recognition

---

### POST /twilio/process-speech

Process speech recognition results from Twilio.

**Request (Form Data):**

```
SpeechResult=I am learning about machine learning
Confidence=0.95
CallSid=CA1234567890abcdef
```

**Response (TwiML XML):**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech"
          language="en-US"
          timeout="5"
          action="/twilio/process-speech?session_id=twilio_CA123">
    <Say voice="Polly.Joanna-Neural">
      That's exciting! What aspect of machine learning interests you?
    </Say>
  </Gather>
</Response>
```

**Low Confidence Handling:**

If `Confidence < 0.5`:

```xml
<Response>
  <Gather ...>
    <Say>I'm not sure I understood. Could you say that again?</Say>
  </Gather>
</Response>
```

---

### POST /twilio/recording

Handle recorded audio from Twilio.

**Request (Form Data):**

```
RecordingUrl=https://api.twilio.com/recordings/RE123
CallSid=CA1234567890abcdef
RecordingDuration=15
```

**Response:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Got it. Processing your recording.</Say>
</Response>
```

**Notes:**
- Recording processed asynchronously
- URL is temporary (download ASAP)

---

## Request/Response Schemas

### ConversationRequest

```typescript
interface ConversationRequest {
  session_id?: string;      // Optional session ID
  text?: string;            // Text input (alternative to audio)
  audio_base64?: string;    // Base64 encoded audio (alternative to text)
}
```

**Validation:**
- At least one of `text` or `audio_base64` required
- `session_id` must be valid UUID or alphanumeric string

### ConversationResponse

```typescript
interface ConversationResponse {
  session_id: string;       // Session identifier
  user_text: string;        // Transcribed or provided text
  agent_text: string;       // AI-generated response
  intent: string;           // Detected intent type
  timestamp: string;        // ISO 8601 timestamp
}
```

### SearchRequest

```typescript
interface SearchRequest {
  query: string;            // Search query (min 1 char)
  limit?: number;           // Max results (1-100, default 20)
}
```

### SearchResponse

```typescript
interface SearchResponse {
  query: string;            // Original query
  results: SearchResult[];  // Array of results
  count: number;            // Number of results returned
}

interface SearchResult {
  id: number;               // Capture ID
  session_id: string;       // Session ID
  timestamp: string;        // ISO 8601 timestamp
  user_text: string;        // Full user text
  agent_text: string;       // Full agent text
  user_snippet: string;     // Highlighted snippet with <mark>
  agent_snippet: string;    // Highlighted snippet with <mark>
}
```

### WebSocketMessage (Client → Server)

```typescript
type WebSocketMessage =
  | { type: 'audio'; audio: string }
  | { type: 'text'; text: string }
  | { type: 'end' }
  | { type: 'ping' };
```

### WebSocketMessage (Server → Client)

```typescript
type WebSocketMessage =
  | {
      type: 'response';
      user_text: string;
      agent_text: string;
      intent: string;
    }
  | {
      type: 'summary';
      text: string;
    };
```

---

## Error Handling

### Error Response Format

All errors return JSON with consistent structure:

```json
{
  "detail": "Error message here"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Successful request |
| 400 | Bad Request | Missing required fields, invalid data |
| 404 | Not Found | Resource doesn't exist |
| 422 | Unprocessable Entity | Validation failed |
| 500 | Internal Server Error | Server-side error, API failures |
| 503 | Service Unavailable | System down, maintenance |

### Error Examples

**Missing Input:**

```json
{
  "detail": "No input provided"
}
```

**Invalid Audio:**

```json
{
  "detail": "Audio too large: 30000000 bytes"
}
```

**API Failure:**

```json
{
  "detail": "I'm having trouble connecting right now. Let's try again - what were you saying?"
}
```

### WebSocket Errors

WebSocket errors close the connection with status codes:

- `1000` - Normal closure
- `1001` - Going away (server restart)
- `1006` - Abnormal closure (network issue)
- `1011` - Server error

---

## Rate Limits

### v1.0 Status

**Rate Limiting:** ⚠️ Not implemented in v1.0

No rate limits currently enforced. Use responsibly to avoid API cost overruns.

### Planned v2.0

```
Rate Limits (per API key):
- 100 requests/minute
- 1000 requests/hour
- 10,000 requests/day

Response Headers:
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635724800
```

**Rate Limit Exceeded:**

```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds.",
  "retry_after": 60
}
```

---

## Example Usage

### Python (REST API)

```python
import requests

BASE_URL = "https://app.railway.app"

# Start a conversation
response = requests.post(
    f"{BASE_URL}/api/conversation",
    json={
        "session_id": "python-client-123",
        "text": "I'm learning about Python decorators"
    }
)

data = response.json()
print(f"User: {data['user_text']}")
print(f"Agent: {data['agent_text']}")

# Search previous conversations
search_response = requests.post(
    f"{BASE_URL}/api/search",
    json={
        "query": "decorators",
        "limit": 5
    }
)

results = search_response.json()
print(f"Found {results['count']} results")
for result in results['results']:
    print(f"- {result['user_text']}")
```

### Python (WebSocket)

```python
import asyncio
import websockets
import json

async def conversation():
    uri = "wss://app.railway.app/ws/python-ws-123"

    async with websockets.connect(uri) as websocket:
        # Send text message
        await websocket.send(json.dumps({
            "type": "text",
            "text": "Hello, I'm learning about async Python"
        }))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)

        print(f"Agent: {data['agent_text']}")

        # End conversation
        await websocket.send(json.dumps({"type": "end"}))

        # Receive summary
        summary = await websocket.recv()
        print(f"Summary: {json.loads(summary)['text']}")

asyncio.run(conversation())
```

### JavaScript (Fetch API)

```javascript
const BASE_URL = 'https://app.railway.app';

async function sendMessage(text) {
  const response = await fetch(`${BASE_URL}/api/conversation`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: 'js-client-123',
      text: text,
    }),
  });

  const data = await response.json();
  console.log('User:', data.user_text);
  console.log('Agent:', data.agent_text);

  return data;
}

// Usage
sendMessage("I'm learning JavaScript promises");
```

### JavaScript (WebSocket with Audio)

```javascript
const ws = new WebSocket('wss://app.railway.app/ws/js-ws-123');

// Get microphone access
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream);
const chunks = [];

mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

mediaRecorder.onstop = async () => {
  const blob = new Blob(chunks, { type: 'audio/webm' });
  const reader = new FileReader();

  reader.onload = () => {
    const base64 = reader.result.split(',')[1];

    // Send audio to server
    ws.send(JSON.stringify({
      type: 'audio',
      audio: base64,
    }));
  };

  reader.readAsDataURL(blob);
  chunks.length = 0;
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'response') {
    console.log('You said:', data.user_text);
    console.log('Agent:', data.agent_text);
  }
};

// Start recording
mediaRecorder.start();

// Stop after 5 seconds
setTimeout(() => mediaRecorder.stop(), 5000);
```

### curl Examples

**Simple Conversation:**

```bash
curl -X POST https://app.railway.app/api/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, I am learning about APIs"
  }'
```

**Search:**

```bash
curl -X POST https://app.railway.app/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "APIs",
    "limit": 10
  }'
```

**Stats:**

```bash
curl https://app.railway.app/api/stats
```

**Session History:**

```bash
curl "https://app.railway.app/api/session/abc-123/history?limit=5"
```

---

## Postman Collection

Import this collection for easy testing:

```json
{
  "info": {
    "name": "Learning Voice Agent API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/"
      }
    },
    {
      "name": "Send Conversation",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/api/conversation",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"text\": \"I am learning about machine learning\"\n}"
        }
      }
    },
    {
      "name": "Search",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/api/search",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"query\": \"machine learning\",\n  \"limit\": 10\n}"
        }
      }
    },
    {
      "name": "Get Stats",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/api/stats"
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "https://app.railway.app"
    }
  ]
}
```

---

## Next Steps

- See [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) for local setup
- Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for production deployment
- Review [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for v2.0 roadmap

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
