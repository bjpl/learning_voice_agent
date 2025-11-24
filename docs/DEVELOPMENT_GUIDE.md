# Development Guide - Learning Voice Agent

**Version:** 1.0.0
**Last Updated:** 2025-11-21
**Target Audience:** Developers contributing to the project

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Local Setup](#local-setup)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Running Tests](#running-tests)
6. [Debugging](#debugging)
7. [Code Style Guide](#code-style-guide)
8. [Contributing Guidelines](#contributing-guidelines)
9. [Common Issues](#common-issues)
10. [Useful Commands](#useful-commands)

---

## Getting Started

### Prerequisites

**Required:**
- Python 3.11 or higher
- Redis server (local or Docker)
- Git

**API Keys:**
- Anthropic API key (for Claude)
- OpenAI API key (for Whisper)

**Optional:**
- Twilio account (for phone integration)
- Docker & Docker Compose

### Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/bjpl/learning_voice_agent.git
cd learning_voice_agent

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start Redis (option 1: Docker)
docker run -d -p 6379:6379 redis:7-alpine

# OR (option 2: local Redis)
redis-server

# 6. Run application
python -m app.main

# 7. Test in browser
# Open http://localhost:8000/static/index.html
```

---

## Local Setup

### Python Environment

**Using venv (recommended):**

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print(fastapi.__version__)"
```

**Using conda:**

```bash
# Create environment
conda create -n voice_agent python=3.11

# Activate
conda activate voice_agent

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

**Create .env file:**

```bash
cp .env.example .env
```

**Edit .env:**

```bash
# Required API Keys
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-proj-...

# Redis (local development)
REDIS_URL=redis://localhost:6379

# Database
DATABASE_URL=sqlite:///./learning_captures.db

# Server
HOST=0.0.0.0
PORT=8000

# Optional: Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890

# Claude Configuration
CLAUDE_MODEL=claude-3-haiku-20240307
CLAUDE_MAX_TOKENS=150
CLAUDE_TEMPERATURE=0.7

# Audio
WHISPER_MODEL=whisper-1
MAX_AUDIO_DURATION=60

# Session
SESSION_TIMEOUT=180
MAX_CONTEXT_EXCHANGES=5
REDIS_TTL=1800
```

**Environment Variables Reference:**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| ANTHROPIC_API_KEY | Claude API key | - | Yes |
| OPENAI_API_KEY | Whisper API key | - | Yes |
| REDIS_URL | Redis connection | redis://localhost:6379 | Yes |
| DATABASE_URL | SQLite path | sqlite:///./learning_captures.db | No |
| HOST | Server host | 0.0.0.0 | No |
| PORT | Server port | 8000 | No |
| CLAUDE_MODEL | Claude model | claude-3-haiku-20240307 | No |
| CLAUDE_MAX_TOKENS | Response length | 150 | No |
| CLAUDE_TEMPERATURE | Creativity | 0.7 | No |
| WHISPER_MODEL | Whisper model | whisper-1 | No |
| MAX_AUDIO_DURATION | Max audio (sec) | 60 | No |
| SESSION_TIMEOUT | Inactivity timeout | 180 | No |
| MAX_CONTEXT_EXCHANGES | Context window | 5 | No |
| REDIS_TTL | Redis expiry (sec) | 1800 | No |

### Redis Setup

**Option 1: Docker (easiest)**

```bash
# Run Redis container
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Verify
docker ps | grep redis

# Connect with CLI
docker exec -it redis redis-cli
127.0.0.1:6379> ping
PONG
```

**Option 2: Local Installation**

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis

# Verify
redis-cli ping
```

**Option 3: Railway (cloud)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link project
railway link

# Add Redis plugin
railway add redis

# Get Redis URL
railway variables
```

### Database Setup

SQLite is file-based and auto-initializes on first run. No setup needed!

**Verify database:**

```bash
# Start the app once
python -m app.main

# Check database file
ls -lh learning_captures.db

# Inspect schema
sqlite3 learning_captures.db ".schema"
```

**Reset database:**

```bash
# Delete database
rm learning_captures.db

# Restart app (will recreate)
python -m app.main
```

---

## Project Structure

```
learning_voice_agent/
├── app/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI app & routes (280 lines)
│   ├── config.py                # Configuration (52 lines)
│   ├── models.py                # Pydantic models (57 lines)
│   ├── conversation_handler.py  # Claude integration (202 lines)
│   ├── audio_pipeline.py        # Whisper transcription (255 lines)
│   ├── state_manager.py         # Redis state (149 lines)
│   ├── database.py              # SQLite + FTS5 (177 lines)
│   ├── twilio_handler.py        # Twilio webhooks (319 lines)
│   └── logger.py                # Logging configuration
│
├── static/                       # Frontend PWA
│   ├── index.html               # Vue 3 app (22KB)
│   ├── manifest.json            # PWA manifest
│   └── sw.js                    # Service worker
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_imports.py          # Import validation
│   └── test_conversation.py     # Conversation tests
│
├── scripts/                      # Utility scripts
│   └── system_audit.py          # System diagnostics
│
├── docs/                         # Documentation
│   ├── README.md                # Docs index
│   ├── ARCHITECTURE_V1.md       # Current architecture
│   ├── MIGRATION_PLAN.md        # v1 → v2 migration
│   ├── API_DOCUMENTATION.md     # API reference
│   ├── DEVELOPMENT_GUIDE.md     # This file
│   ├── DEPLOYMENT_GUIDE.md      # Deployment procedures
│   ├── TECH_DEBT.md             # Technical debt tracking
│   └── DEVELOPMENT_ROADMAP.md   # Future plans
│
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container
├── docker-compose.yml            # Multi-container setup
├── railway.json                  # Railway deployment config
├── README.md                     # Project overview
├── PROJECT_STATUS.md             # Current status
├── QUICK_START.md                # Quick start guide
└── GET_STARTED.md                # Getting started guide
```

### Module Responsibilities

**app/main.py** - Application entry point
- FastAPI app initialization
- Lifecycle management (startup/shutdown)
- REST endpoints
- WebSocket handler
- Background task coordination

**app/conversation_handler.py** - AI conversation logic
- Claude API integration
- System prompt management
- Intent detection
- Conversation summarization
- Error handling for AI APIs

**app/audio_pipeline.py** - Audio processing
- Format detection (magic bytes)
- Whisper API transcription
- Audio validation
- Post-processing (cleaning)
- Streaming support (future)

**app/state_manager.py** - Session state
- Redis connection pooling
- Conversation context (FIFO queue)
- Session metadata
- Activity tracking
- TTL-based cleanup

**app/database.py** - Persistent storage
- SQLite with FTS5
- Async operations
- Capture CRUD
- Full-text search
- Statistics queries

**app/twilio_handler.py** - Phone integration
- Twilio webhook handling
- TwiML generation
- Speech recognition
- Call state management

---

## Development Workflow

### SPARC Methodology

This project uses **SPARC** (Specification, Pseudocode, Architecture, Refinement, Completion) for development:

1. **Specification** - Define requirements clearly
2. **Pseudocode** - Plan the logic
3. **Architecture** - Design the solution
4. **Refinement** - Optimize and test
5. **Completion** - Integrate and document

**Example (adding a new feature):**

```
Feature: Export conversation as PDF

SPECIFICATION:
- User clicks "Export" button
- System generates PDF of conversation
- Downloads to user's device
- Format: Chat-style layout

PSEUDOCODE:
1. GET /api/session/{id}/export
2. Fetch conversation history from DB
3. Generate PDF using reportlab
4. Return as downloadable file

ARCHITECTURE:
- New endpoint in main.py
- PDF generator service
- Template for chat layout
- Stream response for large PDFs

REFINEMENT:
- Add pagination for long conversations
- Include metadata (date, session)
- Optimize for large exports
- Add unit tests

COMPLETION:
- Integrate with frontend
- Add to API documentation
- Update user guide
```

### Git Workflow

**Branching Strategy:**

```
main                 (production-ready code)
  │
  ├── feature/xyz    (new features)
  ├── bugfix/abc     (bug fixes)
  └── hotfix/urgent  (critical fixes)
```

**Feature Development:**

```bash
# 1. Create feature branch
git checkout -b feature/export-pdf

# 2. Make changes
# ... edit files ...

# 3. Commit often with clear messages
git add app/export.py tests/test_export.py
git commit -m "feat: add PDF export functionality

- Add /api/session/{id}/export endpoint
- Implement PDF generation with reportlab
- Add tests for export service
- Update API documentation

Closes #42"

# 4. Push to remote
git push origin feature/export-pdf

# 5. Create Pull Request
# ... use GitHub UI ...

# 6. After review, merge to main
git checkout main
git pull origin main
git merge feature/export-pdf
git push origin main

# 7. Delete feature branch
git branch -d feature/export-pdf
git push origin --delete feature/export-pdf
```

**Commit Message Format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**

```
feat(api): add semantic search endpoint

- Implement vector similarity search
- Add ChromaDB integration
- Update search response schema

Closes #123

---

fix(audio): handle WebM format correctly

The audio pipeline was failing on WebM files due to
incorrect magic byte detection. Updated _detect_format()
to properly identify WebM headers.

Fixes #456

---

docs: update API documentation with examples

Added curl and Python examples for all endpoints.
Improved error response documentation.
```

### Code Review Process

1. **Create PR with:**
   - Clear description
   - Screenshots (if UI changes)
   - Test results
   - Link to issue

2. **Reviewers check:**
   - Code follows style guide
   - Tests included and passing
   - Documentation updated
   - No security issues
   - Performance acceptable

3. **Approval:**
   - Minimum 1 approval required
   - All CI checks passing
   - No merge conflicts

4. **Merge:**
   - Squash and merge (clean history)
   - Delete branch after merge

---

## Running Tests

### Test Structure

```
tests/
├── __init__.py
├── test_imports.py           # Import validation
├── test_conversation.py      # Conversation handler tests
├── test_audio.py            # Audio pipeline tests (TODO)
├── test_database.py         # Database tests (TODO)
└── test_api.py              # API endpoint tests (TODO)
```

### Running Tests

**All tests:**

```bash
# Using pytest
pytest

# With coverage
pytest --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html
```

**Specific test file:**

```bash
pytest tests/test_conversation.py
```

**Specific test function:**

```bash
pytest tests/test_conversation.py::test_conversation_flow
```

**With verbose output:**

```bash
pytest -v
```

**With print statements:**

```bash
pytest -s
```

### Writing Tests

**Unit Test Example:**

```python
# tests/test_audio.py
import pytest
from app.audio_pipeline import audio_pipeline, AudioFormat

@pytest.mark.asyncio
async def test_format_detection():
    """Test audio format detection"""

    # WAV file
    wav_bytes = b'RIFF....WAVE'
    format = audio_pipeline._detect_format(wav_bytes)
    assert format == AudioFormat.WAV

    # MP3 file
    mp3_bytes = b'\xff\xfb....'
    format = audio_pipeline._detect_format(mp3_bytes)
    assert format == AudioFormat.MP3
```

**Integration Test Example:**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_conversation_endpoint():
    """Test conversation API"""
    response = client.post("/api/conversation", json={
        "text": "Hello, I'm learning about testing"
    })

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "agent_text" in data
    assert len(data["agent_text"]) > 0
```

**Mock Example:**

```python
# tests/test_conversation.py
import pytest
from unittest.mock import AsyncMock, patch
from app.conversation_handler import conversation_handler

@pytest.mark.asyncio
async def test_api_error_handling():
    """Test graceful error handling"""

    with patch.object(
        conversation_handler.client.messages,
        'create',
        side_effect=Exception("API Error")
    ):
        response = await conversation_handler.generate_response(
            "Hello",
            context=[]
        )

        # Should return fallback message
        assert "trouble" in response.lower()
```

### Test Coverage Goals

- **Overall:** 80%+
- **Critical paths:** 95%+
- **New code:** 100%

**Check coverage:**

```bash
pytest --cov=app --cov-report=term-missing
```

---

## Debugging

### Local Debugging

**Using print() (temporary):**

```python
# Quick debugging (remember to remove!)
print(f"DEBUG: user_text = {user_text}")
print(f"DEBUG: context = {context}")
```

**Using logging (preferred):**

```python
import logging
logger = logging.getLogger(__name__)

# Debug messages
logger.debug(f"Processing user input: {user_text}")
logger.info(f"Generated response in {duration}ms")
logger.warning(f"Low confidence: {confidence}")
logger.error(f"API call failed: {error}")
```

**Set log level:**

```bash
# In .env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### VS Code Debugging

**launch.json:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
      ],
      "jinja": true,
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

**Breakpoints:**
- Click left margin in VS Code
- Execution pauses at breakpoint
- Inspect variables, step through code

### PyCharm Debugging

1. Edit Configurations → Add Python
2. Script path: `venv/bin/uvicorn`
3. Parameters: `app.main:app --reload`
4. Set breakpoints and run debugger

### API Debugging

**Using curl:**

```bash
# Test health
curl http://localhost:8000/

# Test conversation
curl -X POST http://localhost:8000/api/conversation \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}' | jq

# Test search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' | jq
```

**Using HTTPie (prettier):**

```bash
# Install
pip install httpie

# Test endpoints
http GET localhost:8000/
http POST localhost:8000/api/conversation text="test"
http POST localhost:8000/api/search query="test"
```

**Using Postman:**
- Import collection from API_DOCUMENTATION.md
- Set environment variables
- Run requests with pretty formatting

### WebSocket Debugging

**Using wscat:**

```bash
# Install
npm install -g wscat

# Connect
wscat -c ws://localhost:8000/ws/test-session

# Send messages
> {"type": "text", "text": "Hello"}
< {"type": "response", "user_text": "Hello", ...}

# End
> {"type": "end"}
```

**Using browser console:**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/test');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    type: 'text',
    text: 'Hello from console'
  }));
};

ws.onmessage = (e) => {
  console.log('Received:', JSON.parse(e.data));
};
```

### Database Debugging

**Inspect SQLite:**

```bash
# Open database
sqlite3 learning_captures.db

# List tables
.tables

# View schema
.schema captures

# Query data
SELECT * FROM captures LIMIT 5;

# Search
SELECT * FROM captures_fts WHERE captures_fts MATCH 'learning';

# Stats
SELECT COUNT(*) FROM captures;
SELECT COUNT(DISTINCT session_id) FROM captures;
```

**Inspect Redis:**

```bash
# Connect
redis-cli

# List keys
KEYS session:*

# Get value
GET session:abc-123:context

# View all keys
SCAN 0 MATCH session:* COUNT 100

# Monitor in real-time
MONITOR
```

---

## Code Style Guide

### Python Style (PEP 8 + Project Conventions)

**Formatting:**

```python
# Use Black formatter
pip install black
black app/

# Or format on save in VS Code
# .vscode/settings.json:
{
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

**Import Order:**

```python
# 1. Standard library
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

# 2. Third-party packages
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic

# 3. Local imports
from app.config import settings
from app.database import db
```

**Naming Conventions:**

```python
# Classes: PascalCase
class ConversationHandler:
    pass

# Functions/methods: snake_case
def generate_response(user_text: str) -> str:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Private: _leading_underscore
def _internal_helper():
    pass

# Variables: snake_case
user_input = "Hello"
session_id = "abc-123"
```

**Type Hints:**

```python
# Always use type hints
def process_audio(
    audio_bytes: bytes,
    format: AudioFormat = AudioFormat.WAV
) -> str:
    """Process audio and return transcription"""
    pass

# Complex types
from typing import List, Dict, Optional, Union

def get_context(
    session_id: str
) -> Optional[List[Dict[str, str]]]:
    pass
```

**Docstrings:**

```python
def generate_response(
    user_text: str,
    context: List[Dict],
    session_metadata: Optional[Dict] = None
) -> str:
    """
    Generate AI response using Claude.

    Args:
        user_text: User's input message
        context: Previous conversation exchanges
        session_metadata: Optional session information

    Returns:
        Claude's generated response text

    Raises:
        anthropic.RateLimitError: If API rate limit exceeded
        anthropic.APIError: If API call fails

    Example:
        >>> response = await generate_response(
        ...     "Hello",
        ...     context=[]
        ... )
        >>> print(response)
        "Hi there! What are you learning today?"
    """
    pass
```

**Comments:**

```python
# PATTERN: Cache-aside with TTL
# WHY: Fast context retrieval without database overhead
async def get_conversation_context(session_id: str) -> List[Dict]:
    # Query Redis first
    cached = await redis.get(f"session:{session_id}:context")

    if cached:
        return json.loads(cached)  # Cache hit

    # Cache miss - fetch from database
    return await db.get_session_history(session_id)
```

**Error Handling:**

```python
# Specific exceptions
try:
    response = await claude_api.create(...)
except anthropic.RateLimitError:
    return "I need a moment to catch up."
except anthropic.APIError as e:
    logger.error(f"Claude API error: {e}")
    return "I'm having trouble connecting."
except Exception as e:
    logger.exception("Unexpected error")
    raise
```

### File Organization

**Max file length:** 500 lines (refactor if longer)

**Function length:** < 50 lines (split if longer)

**Single Responsibility Principle:**

```python
# ✅ Good: Single purpose
async def transcribe_audio(audio_bytes: bytes) -> str:
    """Only handles transcription"""
    pass

# ❌ Bad: Multiple responsibilities
async def process_audio_and_generate_response(audio_bytes: bytes) -> str:
    """Does too much!"""
    text = transcribe(audio_bytes)
    response = generate_response(text)
    save_to_db(text, response)
    return response
```

---

## Contributing Guidelines

### Before Contributing

1. **Read documentation:**
   - README.md
   - ARCHITECTURE_V1.md
   - This guide

2. **Set up development environment:**
   - Follow Local Setup
   - Run tests successfully
   - Verify app works

3. **Check existing issues:**
   - Avoid duplicate work
   - Comment if you plan to work on it

### Creating Issues

**Bug Report:**

```markdown
**Bug Description:**
Search returns empty results even when captures exist

**Steps to Reproduce:**
1. Create conversation with text "machine learning"
2. Save to database
3. Search for "machine learning"
4. Observe: 0 results returned

**Expected Behavior:**
Should return 1 result

**Actual Behavior:**
Returns empty array

**Environment:**
- OS: macOS 13.5
- Python: 3.11.5
- Database size: 100 captures

**Logs:**
```
[paste relevant logs]
```

**Additional Context:**
Works correctly with fresh database
```

**Feature Request:**

```markdown
**Feature Description:**
Export conversation as PDF

**Use Case:**
Users want to save conversations for offline review

**Proposed Solution:**
Add /api/session/{id}/export endpoint that generates PDF

**Alternative Solutions:**
- Export as Markdown
- Export as JSON

**Additional Context:**
Should include metadata (date, session ID)
Similar to ChatGPT export feature
```

### Pull Request Process

1. **Create branch:**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes:**
   - Follow code style guide
   - Add tests
   - Update documentation

3. **Commit:**
   ```bash
   git commit -m "feat: add PDF export

   - Implement export endpoint
   - Add reportlab dependency
   - Create PDF template
   - Add tests

   Closes #42"
   ```

4. **Push:**
   ```bash
   git push origin feature/your-feature
   ```

5. **Create PR:**
   - Clear title and description
   - Reference related issues
   - Add screenshots if UI changes
   - Request review

6. **Address feedback:**
   - Respond to comments
   - Make requested changes
   - Push updates

7. **Merge:**
   - Wait for approval
   - Squash and merge
   - Delete branch

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers learn
- Credit others' contributions

---

## Common Issues

### Issue: ModuleNotFoundError

**Problem:**
```
ModuleNotFoundError: No module named 'anthropic'
```

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Issue: Redis Connection Error

**Problem:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution:**
```bash
# Check Redis is running
redis-cli ping

# If not running, start it
docker run -d -p 6379:6379 redis:7-alpine

# Or install locally
brew install redis
brew services start redis
```

---

### Issue: API Key Not Found

**Problem:**
```
ValueError: ANTHROPIC_API_KEY not set
```

**Solution:**
```bash
# Check .env file exists
ls -la .env

# If not, create from example
cp .env.example .env

# Edit with your keys
nano .env

# Verify loaded
python -c "from app.config import settings; print(settings.anthropic_api_key)"
```

---

### Issue: Port Already in Use

**Problem:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
PORT=8001 python -m app.main
```

---

### Issue: Database Locked

**Problem:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Close all connections
pkill -f "python -m app.main"

# Delete WAL files
rm learning_captures.db-wal
rm learning_captures.db-shm

# Restart app
python -m app.main
```

---

## Useful Commands

### Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run on different port
uvicorn app.main:app --port 8001

# Run with debug logs
LOG_LEVEL=DEBUG uvicorn app.main:app --reload

# Format code
black app/ tests/

# Check types
mypy app/

# Lint code
flake8 app/

# Sort imports
isort app/ tests/
```

### Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=app

# Specific test
pytest tests/test_conversation.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Only failed tests
pytest --lf
```

### Database

```bash
# Open database
sqlite3 learning_captures.db

# Backup database
cp learning_captures.db backup_$(date +%Y%m%d).db

# Clear database
rm learning_captures.db

# Inspect schema
sqlite3 learning_captures.db ".schema"

# Export data
sqlite3 learning_captures.db ".dump" > backup.sql

# Import data
sqlite3 learning_captures.db < backup.sql
```

### Docker

```bash
# Build image
docker build -t voice-agent .

# Run container
docker run -p 8000:8000 --env-file .env voice-agent

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Rebuild
docker-compose up --build
```

### Git

```bash
# Create branch
git checkout -b feature/xyz

# Commit with message
git commit -m "feat: add feature"

# Amend last commit
git commit --amend

# Push branch
git push origin feature/xyz

# Pull latest
git pull origin main

# Rebase on main
git rebase main

# Interactive rebase
git rebase -i HEAD~3

# Squash commits
git reset --soft HEAD~3
git commit -m "feat: combined changes"
```

---

## Next Steps

- Deploy to production: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- Understand API: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- Plan v2.0: [MIGRATION_PLAN.md](./MIGRATION_PLAN.md)

---

**Need Help?**
- Create an issue on GitHub
- Check documentation in /docs
- Review existing code for examples

Happy coding! 🚀
