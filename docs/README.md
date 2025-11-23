# 📚 Learning Voice Agent Documentation

## Quick Links

- [🚀 Quick Start Guide](../QUICK_START.md) - Get running in 30 minutes
- [📖 Main README](../README.md) - Project overview and features

## Architecture & Design

- [🏗️ Architecture](ARCHITECTURE.md) - System design and SPARC methodology
- [🔧 Tech Debt Tracker](TECH_DEBT.md) - Known issues and technical debt
- [🗺️ Development Roadmap](DEVELOPMENT_ROADMAP.md) - 8-week development plan

## API Documentation

- [API Endpoints](API.md) - REST API reference *(coming soon)*
- [WebSocket Protocol](WEBSOCKET.md) - Real-time communication *(coming soon)*
- [Twilio Integration](TWILIO.md) - Phone webhook setup *(coming soon)*

## Development Guides

- [Testing Guide](TESTING.md) - How to write and run tests *(coming soon)*
- [Deployment Guide](DEPLOYMENT.md) - Production deployment *(coming soon)*
- [Contributing Guide](CONTRIBUTING.md) - How to contribute *(coming soon)*

## Advanced Topics

- [Prompt Engineering](PROMPT_ENGINEERING.md) - Claude optimization *(coming soon)*
- [Vector Databases](VECTOR_DB.md) - Semantic search implementation *(coming soon)*
- [WebRTC Integration](WEBRTC.md) - Peer-to-peer audio *(coming soon)*
- [Edge Computing](EDGE_COMPUTING.md) - Local Whisper with ONNX *(coming soon)*

## Project Structure

```
learning_voice_agent/
├── app/                    # Core application code
│   ├── audio_pipeline.py   # Audio processing (Whisper)
│   ├── config.py           # Configuration management
│   ├── conversation_handler.py # Claude AI integration
│   ├── database.py         # SQLite + FTS5
│   ├── logger.py           # Logging configuration
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── state_manager.py   # Redis state management
│   └── twilio_handler.py  # Twilio webhooks
├── static/                # Frontend files
│   ├── index.html         # Vue 3 PWA interface
│   ├── manifest.json      # PWA manifest
│   └── sw.js             # Service worker
├── tests/                 # Test suite
│   ├── test_imports.py    # Import verification
│   └── test_conversation.py # Conversation tests
├── scripts/               # Utility scripts
│   └── system_audit.py    # Health check script
├── docs/                  # Documentation
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-container setup
└── railway.json         # Railway deployment
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | Claude API key |
| `OPENAI_API_KEY` | ✅ | Whisper API key |
| `TWILIO_ACCOUNT_SID` | ❌ | Twilio account (optional) |
| `TWILIO_AUTH_TOKEN` | ❌ | Twilio auth (optional) |
| `REDIS_URL` | ✅ | Redis connection URL |
| `DATABASE_URL` | ✅ | SQLite database path |

## Common Tasks

### Run Tests
```bash
python tests/test_imports.py
python tests/test_conversation.py
```

### Start Development Server
```bash
uvicorn app.main:app --reload
```

### Run System Audit
```bash
python scripts/system_audit.py
```

### Build Docker Image
```bash
docker build -t learning-voice-agent .
```

### Deploy to Railway
```bash
railway up
```

## Support

- **Issues**: [GitHub Issues](https://github.com/bjpl/learning_voice_agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bjpl/learning_voice_agent/discussions)

---

*Last updated: Documentation is actively being developed. Check back for updates!*