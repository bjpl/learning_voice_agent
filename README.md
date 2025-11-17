# Learning Voice Agent

An AI-powered voice conversation system designed for capturing and developing learning insights through natural voice interaction.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

Learning Voice Agent is a comprehensive voice conversation system built with FastAPI, Claude Haiku, and modern web technologies. The platform provides an interface for capturing learning insights through voice interactions, offering both browser-based WebSocket connections and optional Twilio phone integration for flexible accessibility.

The system is optimized for real-time performance with sub-2-second conversation loops, full-text search capabilities across all conversations, and Progressive Web App (PWA) support for offline functionality.

## Features

- Voice conversation capability through browser WebSocket or Twilio phone integration
- Claude Haiku AI providing intelligent responses and follow-up questions
- Real-time audio processing with sub-2-second response times
- FTS5-powered full-text search across conversation captures
- Progressive Web App support with offline functionality and app installation
- Redis-based session management with 30-minute conversation context retention
- SQLite database with FTS5 for instant search and retrieval

## Installation

### Prerequisites

- Python 3.11 or higher
- Redis server
- API keys for Anthropic Claude and OpenAI Whisper
- Twilio account (optional, for phone support)

### Setup

Clone the repository:
```bash
git clone https://github.com/bjpl/learning_voice_agent.git
cd learning_voice_agent
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure environment variables:
```bash
cp .env.example .env
# Edit .env with required API keys
```

Initialize the application:
```bash
python -m app.main
```

Access the application at `http://localhost:8000/static/index.html`

## Usage

### Browser-Based Voice Interaction

Open the application in a web browser and grant microphone permissions to begin voice conversations. The system supports real-time WebSocket connections for immediate audio processing and response.

### Phone Integration (Optional)

Configure Twilio credentials in environment variables and set up webhook URL to enable phone-based voice interactions.

### Progressive Web App Installation

The application can be installed as a standalone app on supported browsers. Click the installation prompt or use the browser menu to install the app.

## Project Structure

```
learning_voice_agent/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── conversation_handler.py  # Claude Haiku integration
│   ├── audio_pipeline.py        # Audio transcription processing
│   ├── database.py              # SQLite database with FTS5
│   ├── state_manager.py         # Redis session management
│   └── twilio_handler.py        # Twilio webhook handlers
├── static/
│   ├── index.html               # Vue 3 PWA interface
│   ├── manifest.json            # PWA manifest configuration
│   └── sw.js                    # Service worker for offline support
├── requirements.txt             # Python dependencies
└── docker-compose.yml           # Docker deployment configuration
```

## Development

### Running Locally

Start the development server:
```bash
python -m app.main
```

The application runs on port 8000 by default. API documentation is available at `http://localhost:8000/docs`.

### Testing

Run the test suite:
```bash
pytest tests/
```

Test WebSocket connections:
```bash
wscat -c ws://localhost:8000/ws/test-session
```

Test Twilio webhooks:
```bash
curl -X POST http://localhost:8000/twilio/voice \
  -d "CallSid=test&From=+1234567890&CallStatus=ringing"
```

### Docker Deployment

Build and run with Docker Compose:
```bash
docker-compose up -d
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| ANTHROPIC_API_KEY | Claude API key | Yes |
| OPENAI_API_KEY | Whisper API key | Yes |
| REDIS_URL | Redis connection URL | Yes |
| TWILIO_ACCOUNT_SID | Twilio account identifier | No |
| TWILIO_AUTH_TOKEN | Twilio authentication token | No |

### Performance Targets

- Audio transcription: under 800ms (Whisper API)
- Claude response generation: under 900ms (Haiku model)
- Total conversation loop: under 2 seconds end-to-end
- Session timeout: 3 minutes of inactivity

## API Endpoints

### REST API

- `POST /api/conversation` - Process text or audio input
- `POST /api/search` - Search conversation captures with full-text search
- `GET /api/stats` - Retrieve system statistics
- `GET /api/session/{id}/history` - Retrieve session conversation history

### WebSocket

- `/ws/{session_id}` - Real-time conversation stream

### Twilio Webhooks

- `POST /twilio/voice` - Handle incoming phone calls
- `POST /twilio/process-speech` - Process speech input from calls

## Deployment

### Railway (Recommended)

Deploy using Railway CLI:
```bash
railway up
```

The platform provides automatic SSL, scaling, and built-in Redis support.

### Cloudflare Tunnel

Secure HTTPS deployment without port configuration is available through Cloudflare Tunnel, included in the docker-compose configuration.

### Backup Strategy

Continuous SQLite replication to cloud storage (R2/S3) is supported via Litestream for point-in-time recovery capabilities.

## Contributing

Contributions are welcome. Please follow the SPARC methodology for new features and submit pull requests with clear descriptions of changes.

## License

MIT License - See LICENSE file for details
