# Learning Voice Agent 🎙️

[![Tests](https://github.com/bjpl/learning_voice_agent/workflows/Test%20Suite/badge.svg)](https://github.com/bjpl/learning_voice_agent/actions/workflows/test.yml)
[![Code Quality](https://github.com/bjpl/learning_voice_agent/workflows/Code%20Quality/badge.svg)](https://github.com/bjpl/learning_voice_agent/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/bjpl/learning_voice_agent/branch/main/graph/badge.svg)](https://codecov.io/gh/bjpl/learning_voice_agent)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version:** 1.0.0 | **Health Score:** 75/100 🟡 | **Status:** Production-ready (with known limitations)

An AI-powered voice conversation system for capturing and developing learning insights. Built with FastAPI, Claude Haiku, Whisper, and modern web technologies using SPARC methodology.

---

## ✨ Features

### Core Capabilities

- **Voice Conversations** - Natural voice interaction through browser or phone (Twilio)
- **AI Intelligence** - Claude Haiku provides thoughtful responses and follow-up questions
- **Real-time Processing** - Sub-2-second conversation loops with WebSocket support
- **Smart Search** - SQLite FTS5-powered instant search across all captures
- **PWA Support** - Works offline, installable as an app
- **Multi-channel** - Browser WebSocket and Twilio phone support

### Technical Highlights

- **Async Python** - Built with FastAPI and async/await throughout
- **SPARC Architecture** - Clean code following Specification, Pseudocode, Architecture, Refinement, Completion methodology
- **Full-Text Search** - BM25 ranking with SQLite FTS5
- **Session Management** - Redis-based context with 30-minute TTL
- **Type Safety** - Pydantic models for request/response validation

---

## 📚 Complete Documentation

### 🚀 Getting Started

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in 30 minutes
- **[Development Guide](docs/DEVELOPMENT_GUIDE.md)** - Local setup, debugging, code style, testing
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Railway, Docker, cloud deployment

### 📖 Architecture & Reference

- **[Architecture v1.0](docs/ARCHITECTURE_V1.md)** - Complete system architecture with diagrams
- **[API Documentation](docs/API_DOCUMENTATION.md)** - REST, WebSocket, Twilio APIs with examples
- **[Database Schema](docs/ARCHITECTURE_V1.md#database-schema)** - SQLite + FTS5 + Redis

### 🗺️ Planning & Roadmap

- **[Migration Plan](docs/MIGRATION_PLAN.md)** - v1.0 → v2.0 migration strategy (20 weeks)
- **[Rebuild Strategy](docs/REBUILD_STRATEGY.md)** - Comprehensive v2.0 rebuild with multi-agent orchestration
- **[Project Status](PROJECT_STATUS.md)** - Current health metrics and next actions
- **[Tech Debt](docs/TECH_DEBT.md)** - Known issues and improvements

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                              │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Web Browser │  Mobile PWA  │  Phone Call  │   REST Client  │
│  (Vue 3)     │  (Installed) │  (Twilio)    │   (API)        │
└──────┬───────┴───────┬──────┴───────┬──────┴────────┬───────┘
       │               │              │               │
       └───────────────┴──────────────┴───────────────┘
                              │
       ┌──────────────────────────────────────────────────────┐
       │           FASTAPI APPLICATION LAYER                   │
       ├──────────────────────────────────────────────────────┤
       │  • Conversation Handler (Claude Haiku)               │
       │  • Audio Pipeline (Whisper)                          │
       │  • State Manager (Redis)                             │
       │  • Database (SQLite + FTS5)                          │
       └──────────────────────────────────────────────────────┘
```

**See [docs/ARCHITECTURE_V1.md](docs/ARCHITECTURE_V1.md) for detailed architecture documentation.**

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis server (or Docker)
- API Keys: [Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/)
- (Optional) [Twilio](https://www.twilio.com/) account for phone support

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/bjpl/learning_voice_agent.git
cd learning_voice_agent

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 6. Run application
python -m app.main

# 7. Open browser
# http://localhost:8000/static/index.html
```

**For detailed setup instructions, see [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md).**

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

## 🚂 Railway Deployment (Recommended)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway add redis
railway up

# Your app is live at https://your-app.railway.app
```

**For complete deployment instructions, see [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).**

---

## 📊 Technology Stack

### Backend
- **Framework:** FastAPI 0.109.0 (async Python 3.11+)
- **AI Models:** Anthropic Claude Haiku + OpenAI Whisper
- **Database:** SQLite 3 with FTS5 full-text search
- **State:** Redis 5.0.1 (session management)
- **Validation:** Pydantic 2.5.3

### Frontend
- **Framework:** Vue 3 (Composition API)
- **PWA:** Service Workers for offline support
- **Audio:** MediaRecorder API
- **WebSocket:** Real-time bidirectional communication

### Performance Targets
- **Audio transcription:** < 800ms (Whisper API)
- **Claude response:** < 900ms (Haiku model)
- **Total conversation loop:** < 2 seconds end-to-end
- **Session timeout:** 3 minutes of inactivity

---

## 📊 API Endpoints

### REST API

```http
POST /api/conversation      # Process text/audio, get AI response
POST /api/search           # Full-text search across captures
GET  /api/stats            # System statistics
GET  /api/session/{id}/history  # Conversation history
```

### WebSocket

```
ws://localhost:8000/ws/{session_id}
```

**Messages:**
- Client → Server: `{"type": "audio", "audio": "base64..."}`
- Server → Client: `{"type": "response", "user_text": "...", "agent_text": "..."}`

### Twilio Webhooks (Optional)

```http
POST /twilio/voice          # Incoming call handler
POST /twilio/process-speech # Speech recognition results
```

**Complete API documentation with examples: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)**

---

## 🧠 Claude System Prompt

The system uses a carefully crafted prompt to make Claude act as a learning companion:

```
You are a personal learning companion helping capture and develop ideas.

Your role:
- Ask ONE clarifying question when responses are vague
- Connect new ideas to previously mentioned topics
- Keep responses under 3 sentences
- Never lecture unless explicitly asked
- Mirror the user's energy level
```

This prompt-first approach means the intelligence comes from the prompt, not complex logic.

---

## 💾 Database Schema

```sql
-- Conversation captures
CREATE TABLE captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_text TEXT NOT NULL,
    agent_text TEXT NOT NULL,
    metadata TEXT
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE captures_fts USING fts5(
    session_id UNINDEXED,
    user_text,
    agent_text,
    content=captures
);
```

**See [docs/ARCHITECTURE_V1.md#database-schema](docs/ARCHITECTURE_V1.md#database-schema) for complete schema documentation.**

---

## 🔍 Search Features

- **Full-text search** with BM25 ranking (SQLite FTS5)
- **Highlighted snippets** in results (`<mark>` tags)
- **Instant search** as you type
- **Keyboard shortcut:** Cmd+K or Ctrl+K

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 10}'
```

---

## 💰 Estimated Costs

### Per-Conversation
- **Whisper:** $0.0005 per 5-second clip
- **Claude Haiku:** $0.00025 per response
- **Total:** ~$0.00075 per exchange

### Monthly Costs

**Light Usage (100 exchanges/day):**
- AI APIs: ~$2.25/month
- Railway: $5/month
- **Total: ~$7.25/month**

**Moderate Usage (1000 exchanges/day):**
- AI APIs: ~$22.50/month
- Railway: $20/month
- Optional Twilio: $20/month
- **Total: ~$42.50-62.50/month**

---

## 🛠️ Development

### Project Structure

```
learning_voice_agent/
├── app/                    # Main application
│   ├── main.py            # FastAPI app & routes
│   ├── conversation_handler.py  # Claude integration
│   ├── audio_pipeline.py  # Whisper transcription
│   ├── database.py        # SQLite + FTS5
│   ├── state_manager.py   # Redis state
│   └── twilio_handler.py  # Phone integration
├── static/                # Frontend PWA
├── tests/                 # Test suite
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# View coverage
open htmlcov/index.html
```

**See [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) for complete development workflow.**

---

## ⚠️ Known Limitations (v1.0)

### Functional
- ❌ No user authentication
- ❌ No cross-device synchronization
- ❌ Limited context window (5 exchanges)
- ❌ English only
- ❌ No semantic search (keyword only)

### Technical
- ❌ SQLite doesn't scale horizontally
- ❌ Single Redis instance (no redundancy)
- ❌ No rate limiting
- ❌ Low test coverage (~10%)

### Migration to v2.0

We're planning a comprehensive 20-week rebuild to address these limitations:

- ✅ Multi-agent orchestration (LangGraph)
- ✅ Semantic memory (ChromaDB + vector search)
- ✅ Multi-modal support (voice + vision + documents)
- ✅ Real-time learning and model improvement
- ✅ Cross-device sync (web + mobile)
- ✅ Production-ready infrastructure

**See [docs/REBUILD_STRATEGY.md](docs/REBUILD_STRATEGY.md) and [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md) for the complete v2.0 plan.**

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow SPARC methodology
4. Add tests for new functionality
5. Commit: `git commit -m "feat: add amazing feature"`
6. Push and create PR

**See [docs/DEVELOPMENT_GUIDE.md#contributing-guidelines](docs/DEVELOPMENT_GUIDE.md#contributing-guidelines) for detailed guidelines.**

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Anthropic](https://www.anthropic.com/)** - Claude Haiku API
- **[OpenAI](https://openai.com/)** - Whisper API
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Railway](https://railway.app/)** - Simplified deployment
- **SPARC Methodology** - Systematic development approach
- **Flow Nexus** - Agent orchestration framework

---

## 📞 Support & Resources

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/bjpl/learning_voice_agent/issues)
- **Discussions:** [GitHub Discussions](https://github.com/bjpl/learning_voice_agent/discussions)

---

## 🗺️ Roadmap

### v1.0 (Current - Production)
- ✅ Voice conversation with Claude Haiku
- ✅ Audio transcription with Whisper
- ✅ Full-text search with SQLite FTS5
- ✅ WebSocket real-time updates
- ✅ PWA support
- ✅ Twilio phone integration
- ✅ Comprehensive documentation

### v2.0 (Planning - 20 weeks)
- 🔄 Multi-agent orchestration (LangGraph/CrewAI)
- 🔄 Semantic memory (ChromaDB + embeddings)
- 🔄 Multi-modal (vision + documents)
- 🔄 Real-time learning
- 🔄 Mobile apps (iOS + Android)
- 🔄 Cross-device sync
- 🔄 Analytics engine

**Read the full v2.0 plan: [docs/REBUILD_STRATEGY.md](docs/REBUILD_STRATEGY.md)**

---

**Built with ❤️ using SPARC methodology and modern AI technologies**

**Current Status:** ✅ Production-ready v1.0 with comprehensive documentation | 🔄 v2.0 rebuild planned
