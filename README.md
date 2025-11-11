# Learning Voice Agent 🎙️

An AI-powered voice conversation system for capturing and developing learning insights. Built with FastAPI, Claude Haiku, and modern web technologies.

## ✨ Features

- **Voice Conversations**: Natural voice interaction through browser or phone (Twilio)
- **AI Intelligence**: Claude Haiku provides thoughtful responses and follow-up questions
- **Real-time Processing**: Sub-2-second conversation loops with WebSocket support
- **Smart Search**: FTS5-powered instant search across all captures
- **PWA Support**: Works offline, installable as an app
- **Multi-channel**: Browser WebSocket and Twilio phone support

## 🏗️ Architecture

See [Architecture Documentation](docs/ARCHITECTURE.md) for detailed system design.

### Core Components (SPARC Methodology Applied)

1. **Conversation Handler**: Claude Haiku integration with intelligent prompting
2. **Audio Pipeline**: Whisper transcription with format detection
3. **State Management**: Redis for conversation context (30-min TTL)
4. **Database Layer**: SQLite with FTS5 for instant search
5. **Frontend PWA**: Vue 3 with real-time updates

### Performance Targets

- Audio transcription: < 800ms (Whisper API)
- Claude response: < 900ms (Haiku model)
- Total loop: < 2 seconds end-to-end
- Session timeout: 3 minutes of inactivity

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis server
- API Keys: Anthropic, OpenAI
- (Optional) Twilio account for phone support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bjpl/learning_voice_agent.git
cd learning_voice_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
python -m app.main
```

5. Open browser to `http://localhost:8000/static/index.html`

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or deploy to Railway
railway up
```

## 📱 PWA Installation

1. Open the app in Chrome/Edge
2. Click "Install" when prompted
3. Or use menu → "Install app"

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | Yes |
| `OPENAI_API_KEY` | Whisper API key | Yes |
| `TWILIO_ACCOUNT_SID` | Twilio account | No |
| `TWILIO_AUTH_TOKEN` | Twilio auth | No |
| `REDIS_URL` | Redis connection | Yes |

### Twilio Setup (Optional)

1. Get a Twilio phone number
2. Configure webhook URL: `https://your-domain/twilio/voice`
3. Set environment variables
4. Test with a phone call

## 📊 API Endpoints

### REST API

- `POST /api/conversation` - Process text/audio input
- `POST /api/search` - Search captures with FTS5
- `GET /api/stats` - System statistics
- `GET /api/session/{id}/history` - Session history

### WebSocket

- `/ws/{session_id}` - Real-time conversation stream

### Twilio Webhooks

- `POST /twilio/voice` - Handle incoming calls
- `POST /twilio/process-speech` - Process speech input

## 🧠 Claude System Prompt

The system uses a carefully crafted prompt to make Claude act as a learning companion:

- Asks clarifying questions for vague inputs
- Connects new ideas to previous topics
- Keeps responses under 3 sentences
- Never lectures unless asked
- Maintains conversation context

## 💾 Database Schema

```sql
CREATE TABLE captures (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_text TEXT NOT NULL,
    agent_text TEXT NOT NULL,
    metadata TEXT
);

-- FTS5 virtual table for search
CREATE VIRTUAL TABLE captures_fts USING fts5(
    session_id UNINDEXED,
    user_text,
    agent_text,
    content=captures
);
```

## 🔍 Search Features

- Full-text search with BM25 ranking
- Highlighted snippets in results
- Instant search-as-you-type
- Keyboard shortcut: Cmd+K

## 📈 Monitoring

- Active session tracking
- Database statistics
- Redis connection health
- Conversation metrics

## 🚢 Deployment Options

### Railway (Recommended)
- Push to deploy with `railway up`
- Automatic SSL and scaling
- Built-in Redis support

### Cloudflare Tunnel
- Secure HTTPS without ports
- Zero-trust networking
- Included in docker-compose

### Litestream Backup
- Continuous SQLite replication
- Backup to R2/S3
- Point-in-time recovery

## 💰 Estimated Costs

- Twilio: ~$20/month (phone number + minutes)
- Claude Haiku: ~$5/month (at moderate usage)
- Whisper API: ~$3/month
- Hosting: ~$5/month (Railway)
- **Total: ~$33/month**

## 🛠️ Development

### Project Structure
```
learning_voice_agent/
├── app/
│   ├── main.py              # FastAPI application
│   ├── conversation_handler.py  # Claude integration
│   ├── audio_pipeline.py    # Audio processing
│   ├── database.py          # SQLite + FTS5
│   ├── state_manager.py     # Redis state
│   └── twilio_handler.py    # Twilio webhooks
├── static/
│   ├── index.html           # Vue 3 PWA
│   ├── manifest.json        # PWA manifest
│   └── sw.js               # Service worker
└── requirements.txt
```

### Testing

```bash
# Run tests
pytest tests/

# Test WebSocket connection
wscat -c ws://localhost:8000/ws/test-session

# Test Twilio webhook
curl -X POST http://localhost:8000/twilio/voice \
  -d "CallSid=test&From=+1234567890&CallStatus=ringing"
```

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Apply SPARC methodology to new features
4. Submit a pull request

## 🙏 Acknowledgments

- Built with Flow Nexus orchestration
- SPARC methodology for efficient development
- Claude Haiku for intelligent conversations
- Whisper for accurate transcription

---

**Built with ❤️ using SPARC methodology and Flow Nexus**