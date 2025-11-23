# Learning Voice Agent - User Guide

## Overview

The Learning Voice Agent is an AI-powered voice conversation system designed to capture and recall your learnings through natural conversation. Think of it as a voice-enabled second brain that remembers what you tell it and helps you recall information when needed.

## Getting Started

### Accessing the Application

1. **Web Interface**: Navigate to `https://yourdomain.com/static/index.html`
2. **API**: Use the REST API at `https://yourdomain.com/api/`
3. **Phone** (if configured): Call your Twilio number

### First Conversation

1. Click the microphone button or type in the text box
2. Share something you want to remember:
   - "Save this: Python list comprehensions are faster than traditional loops"
   - "I learned today that Redis uses single-threaded processing"
3. The agent will acknowledge and categorize your input

### Recalling Information

Ask the agent to help you remember:
- "What did I learn about Python?"
- "Remind me about that meeting from last week"
- "Search for anything about databases"

## Features

### Voice Input
- Click and hold the microphone to record
- Release to send for processing
- Supports up to 60 seconds of audio

### Text Input
- Type your message in the text box
- Press Enter or click Send
- Supports markdown formatting

### Search
- **Keyword Search**: `/api/search` - Fast text matching
- **Semantic Search**: `/api/semantic-search` - Find conceptually similar content

### Offline Mode
The app works offline with limited functionality:
- View recently cached conversations
- Queue messages for later sync
- Automatic reconnection when online

## API Quick Reference

### Conversation
```bash
# Send a text message
curl -X POST https://api.yourdomain.com/api/conversation \
  -H "Content-Type: application/json" \
  -d '{"text": "Remember that Python is awesome"}'

# Response
{
  "session_id": "uuid",
  "user_text": "Remember that Python is awesome",
  "agent_text": "Got it! I've saved that Python is awesome.",
  "intent": "capture"
}
```

### Search
```bash
# Keyword search
curl -X POST https://api.yourdomain.com/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python", "limit": 10}'

# Semantic search (find similar concepts)
curl -X POST https://api.yourdomain.com/api/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "programming languages", "limit": 5}'
```

### History
```bash
# Get conversation history
curl https://api.yourdomain.com/api/session/YOUR_SESSION_ID/history?limit=20
```

## Tips for Best Results

### Capturing Information
- Be specific: "I learned that X does Y because of Z"
- Use keywords that you'll search for later
- Categorize implicitly: "Meeting note:", "Book summary:", "Idea:"

### Recalling Information
- Start with broad searches, then narrow down
- Try different phrasings if semantic search doesn't find it
- Use the "similar" feature to explore related concepts

### Session Management
- Sessions last 30 minutes of inactivity
- End a session explicitly to get a summary
- Each session maintains conversation context

## Troubleshooting

### Audio Not Working
1. Check browser microphone permissions
2. Ensure you're on HTTPS
3. Try a different browser (Chrome/Firefox recommended)

### Slow Responses
1. Check your internet connection
2. Longer inputs take more time to process
3. Peak times may have higher latency

### Search Not Finding Content
1. Try semantic search for concept-based queries
2. Use different keywords or phrases
3. Check if the content was saved in a recent session

### Offline Issues
1. The app caches limited data offline
2. Complex operations require connectivity
3. Queued messages sync when reconnected

## Privacy & Data

- All conversations are stored securely
- Data is encrypted in transit (HTTPS)
- Session data expires after configured period
- Search indexes are maintained for fast retrieval

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Enter | Send message |
| Shift+Enter | New line in message |
| Ctrl+/ | Focus search |
| Escape | Cancel recording |

## Support

- API Documentation: `/docs`
- System Status: `/api/stats`
- Health Check: `/health`

For technical issues, contact your system administrator.
