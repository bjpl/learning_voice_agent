# Learning Voice Agent Documentation

Welcome to the Learning Voice Agent documentation. This comprehensive AI-powered voice agent platform supports multiple phases of development.

## Quick Links

- [Getting Started](./guides/GET_STARTED.md) - Get running in 30 minutes
- [Quick Start](./guides/QUICK_START.md) - Quick setup guide
- [Main README](../README.md) - Project overview and features

## Documentation Structure

```
docs/
├── README.md              # This file - Documentation index
├── api/                   # API documentation
├── architecture/          # System architecture
│   ├── ARCHITECTURE.md
│   └── ARCHITECTURE_V1.md
├── guides/               # User guides
│   ├── GET_STARTED.md
│   ├── QUICK_START.md
│   └── README_CONVERSATION_AGENT.md
├── development/          # Development docs
│   ├── PROJECT_STATUS.md
│   ├── daily-reports/
│   └── startup-reports/
├── testing/              # Test documentation
├── deployment/           # Deployment guides
│   ├── DEPLOYMENT_GUIDE.md
│   └── DEPLOYMENT_MONITORING.md
├── archive/              # Historical documentation
│   ├── completion/       # Phase completion reports
│   ├── completion-reports/
│   ├── CHANGELOG.md
│   └── README_PHASE4.md
├── plans/                # Planning documentation
├── production/           # Production docs
├── security/             # Security documentation
├── sparc/                # SPARC methodology docs
├── examples/             # Code examples
└── vision/               # Vision documentation
```

## Architecture & Design

- [Architecture](./ARCHITECTURE.md) - System design and SPARC methodology
- [Tech Debt Tracker](./TECH_DEBT.md) - Known issues and technical debt
- [Development Roadmap](./DEVELOPMENT_ROADMAP.md) - Development plan

## Phase Documentation

### Phase 1 - Foundation
- [Audit Report](./PHASE1_AUDIT_REPORT.md)
- [Completion Summary](./PHASE1_COMPLETION_SUMMARY.md)

### Phase 2 - Agent Architecture
- [Agent Architecture](./PHASE2_AGENT_ARCHITECTURE.md)
- [Implementation Guide](./PHASE2_IMPLEMENTATION_GUIDE.md)
- [Testing Guide](./PHASE2_TESTING_GUIDE.md)

### Phase 3 - Vector Search
- [Vector Architecture](./PHASE3_VECTOR_ARCHITECTURE.md)
- [Knowledge Graph](./PHASE3_KNOWLEDGE_GRAPH.md)
- [Quickstart](./PHASE3_QUICKSTART.md)

### Phase 4 - Multimodal
- [Multimodal Architecture](./PHASE4_MULTIMODAL_ARCHITECTURE.md)
- [Implementation Guide](./PHASE4_IMPLEMENTATION_GUIDE.md)
- [API Reference](./PHASE4_API_REFERENCE.md)

### Phase 5 - Learning
- [Learning Architecture](./PHASE5_LEARNING_ARCHITECTURE.md)
- [Learning Guide](./PHASE5_LEARNING_GUIDE.md)

### Phase 6 - Analytics
- [Analytics Architecture](./PHASE6_ANALYTICS_ARCHITECTURE.md)
- [Dashboard Guide](./PHASE6_DASHBOARD_GUIDE.md)

## API Documentation

- [API Documentation](./API_DOCUMENTATION.md) - REST API reference
- [Agent API Reference](./AGENT_API_REFERENCE.md) - Agent API specs

## Development Guides

- [Development Guide](./DEVELOPMENT_GUIDE.md) - Development setup
- [Testing Guide](./TESTING.md) - How to write and run tests
- [Deployment Guide](./DEPLOYMENT_GUIDE.md) - Production deployment

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | Yes | Whisper API key |
| `TWILIO_ACCOUNT_SID` | No | Twilio account (optional) |
| `TWILIO_AUTH_TOKEN` | No | Twilio auth (optional) |
| `REDIS_URL` | Yes | Redis connection URL |
| `DATABASE_URL` | Yes | SQLite database path |

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

### Build Docker Image
```bash
docker build -t learning-voice-agent .
```

## Getting Started

1. Review [Getting Started Guide](./guides/GET_STARTED.md)
2. Follow the [Quick Start](./guides/QUICK_START.md)
3. Check [Architecture Overview](./ARCHITECTURE.md)
