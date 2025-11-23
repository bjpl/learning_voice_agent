# Learning Voice Agent - API Documentation

## Overview

The Learning Voice Agent API provides endpoints for voice conversation capture, AI-powered responses, and semantic search across captured learnings.

**Base URL:** `https://api.yourdomain.com`

**API Version:** 1.0.0

**Authentication:** API keys via environment variables (server-side)

## Endpoints

### Health & Status

#### GET /
Health check and API information.

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

#### GET /api/stats
System statistics and monitoring.

**Response:**
```json
{
  "database": {
    "total_exchanges": 1234,
    "total_sessions": 56
  },
  "sessions": {
    "active": 3,
    "ids": ["session-1", "session-2", "session-3"]
  }
}
```

---

### Conversations

#### POST /api/conversation
Handle a single conversation exchange.

**Request Body:**
```json
{
  "session_id": "optional-uuid",
  "text": "What did I learn about Python yesterday?",
  "audio_base64": "optional-base64-encoded-audio"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| session_id | string | No | Session identifier (auto-generated if not provided) |
| text | string | No* | Text input for conversation |
| audio_base64 | string | No* | Base64-encoded audio for transcription |

*Either `text` or `audio_base64` must be provided.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_text": "What did I learn about Python yesterday?",
  "agent_text": "Based on your previous conversations, you learned about Python decorators and how they can be used to add functionality to functions without modifying their code.",
  "intent": "recall"
}
```

**Intent Values:**
- `capture` - User is providing new information
- `recall` - User is trying to remember something
- `search` - User is looking for specific information
- `summary` - User wants a summary
- `general` - General conversation

**Error Responses:**
- `400 Bad Request` - No input provided
- `500 Internal Server Error` - Processing error

---

#### GET /api/session/{session_id}/history
Get conversation history for a session.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| session_id | string | Session identifier |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 20 | Maximum exchanges to return |

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "history": [
    {
      "id": 1,
      "user_text": "Hello",
      "agent_text": "Hi there! What would you like to capture today?",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

---

### Search

#### POST /api/search
Search across captured learnings using full-text search.

**Request Body:**
```json
{
  "query": "python decorators",
  "limit": 10
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| limit | integer | No | 10 | Maximum results |

**Response:**
```json
{
  "query": "python decorators",
  "results": [
    {
      "id": 123,
      "session_id": "550e8400-...",
      "user_text": "Python decorators are functions that modify other functions",
      "agent_text": "Great explanation! Decorators use the @syntax...",
      "timestamp": "2024-01-14T15:20:00Z",
      "relevance_score": 0.95
    }
  ],
  "count": 1
}
```

---

### WebSocket

#### WS /ws/{session_id}
Real-time bidirectional communication for streaming audio.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.yourdomain.com/ws/my-session-id');
```

**Client Messages:**

1. **Audio Data:**
```json
{
  "type": "audio",
  "audio": "base64-encoded-audio-data"
}
```

2. **End Session:**
```json
{
  "type": "end"
}
```

**Server Messages:**

1. **Response:**
```json
{
  "type": "response",
  "user_text": "transcribed user input",
  "agent_text": "AI response",
  "intent": "recall"
}
```

2. **Summary (on end):**
```json
{
  "type": "summary",
  "text": "Session summary..."
}
```

---

### Admin Dashboard

#### GET /admin/dashboard
HTML admin dashboard with real-time metrics.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| admin_key | string | Admin API key (if configured) |

**Headers:**
| Header | Description |
|--------|-------------|
| X-Admin-Key | Admin API key (alternative to query param) |

---

#### GET /admin/api/metrics
Complete dashboard metrics as JSON.

**Response:**
```json
{
  "overview": {
    "total_requests": 10000,
    "total_errors": 5,
    "uptime_seconds": 86400,
    "error_rate": 0.05
  },
  "request_stats": {
    "window_minutes": 5,
    "total_requests": 500,
    "requests_per_second": 1.67,
    "error_rate": 0.02,
    "avg_response_ms": 45.5,
    "p50_ms": 35,
    "p95_ms": 120,
    "p99_ms": 250
  },
  "system_stats": {
    "current": {
      "cpu_percent": 25.5,
      "memory_percent": 45.2,
      "memory_used_mb": 512,
      "memory_available_mb": 1024,
      "disk_percent": 30.0,
      "active_connections": 15
    }
  },
  "top_endpoints": [...],
  "recent_errors": [...]
}
```

---

#### GET /admin/api/health/detailed
Detailed component health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "total_exchanges": 1234
    },
    "redis": {
      "status": "healthy",
      "active_sessions": 5
    },
    "system": {
      "status": "healthy",
      "cpu_percent": 25.5,
      "memory_percent": 45.2
    }
  }
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Some components have issues
- `unhealthy` - Critical component failure

---

## Error Handling

All errors follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

**HTTP Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 403 | Forbidden - Invalid admin key |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently no rate limiting is enforced. For production deployments, consider implementing:
- 100 requests/minute for conversation endpoints
- 1000 requests/minute for search endpoints
- 10 requests/minute for admin endpoints

---

## OpenAPI/Swagger

Interactive API documentation is available at:
- **Swagger UI:** `https://api.yourdomain.com/docs`
- **ReDoc:** `https://api.yourdomain.com/redoc`
- **OpenAPI JSON:** `https://api.yourdomain.com/openapi.json`

---

## Code Examples

### Python
```python
import httpx

async def send_message(text: str, session_id: str = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.yourdomain.com/api/conversation",
            json={"text": text, "session_id": session_id}
        )
        return response.json()
```

### JavaScript
```javascript
async function sendMessage(text, sessionId = null) {
  const response = await fetch('https://api.yourdomain.com/api/conversation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, session_id: sessionId })
  });
  return response.json();
}
```

### cURL
```bash
curl -X POST https://api.yourdomain.com/api/conversation \
  -H "Content-Type: application/json" \
  -d '{"text": "What did I learn today?"}'
```
