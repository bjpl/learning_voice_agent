"""
Test Configuration and Fixtures
PATTERN: Centralized test setup with reusable fixtures
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import json
from datetime import datetime
from typing import Dict, List

# Test Configuration
pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Mock Environment Variables
@pytest.fixture(scope="session", autouse=True)
def mock_env():
    """Mock environment variables for testing"""
    with patch.dict('os.environ', {
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'REDIS_URL': 'redis://localhost:6379',
        'DATABASE_URL': 'sqlite:///:memory:',
        'CLAUDE_MODEL': 'claude-3-haiku-20240307',
        'WHISPER_MODEL': 'whisper-1',
        'CORS_ORIGINS': '["*"]'
    }):
        yield

# Database Fixtures
@pytest.fixture
async def test_db():
    """In-memory database for testing"""
    from app.database import Database

    db = Database(":memory:")
    await db.initialize()
    yield db
    # Cleanup happens automatically with in-memory DB

@pytest.fixture
async def db_with_data(test_db):
    """Database pre-populated with test data"""
    # Add test exchanges
    exchanges = [
        ("session1", "Tell me about Python", "Python is a programming language. What would you like to know?"),
        ("session1", "Its syntax", "Python uses indentation for blocks. What interests you most?"),
        ("session2", "Learning about AI", "AI is fascinating! What aspect interests you?"),
    ]

    for session_id, user_text, agent_text in exchanges:
        await test_db.save_exchange(session_id, user_text, agent_text)

    yield test_db

# State Manager Fixtures
@pytest.fixture
async def mock_redis():
    """Mock Redis client for state manager"""
    mock = AsyncMock()

    # In-memory storage for testing
    storage = {}

    async def get(key):
        return storage.get(key)

    async def setex(key, ttl, value):
        storage[key] = value
        return True

    async def delete(key):
        if key in storage:
            del storage[key]
        return True

    async def scan_iter(match=None):
        for key in storage.keys():
            if match is None or key.startswith(match.replace("*", "")):
                yield key

    mock.get = get
    mock.setex = setex
    mock.delete = delete
    mock.scan_iter = scan_iter
    mock.close = AsyncMock()

    return mock

@pytest.fixture
async def test_state_manager(mock_redis):
    """State manager with mocked Redis"""
    from app.state_manager import StateManager

    manager = StateManager()
    manager.redis_client = mock_redis
    return manager

# Conversation Handler Fixtures
@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for conversation handler"""
    mock_client = AsyncMock()

    # Mock message response
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="That's interesting! Tell me more.")]

    mock_client.messages.create = AsyncMock(return_value=mock_message)

    return mock_client

@pytest.fixture
async def test_conversation_handler(mock_anthropic_client):
    """Conversation handler with mocked Claude API"""
    from app.conversation_handler import ConversationHandler

    handler = ConversationHandler()
    handler.client = mock_anthropic_client
    return handler

# Audio Pipeline Fixtures
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for audio pipeline"""
    mock_client = AsyncMock()

    # Mock transcription response
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value="This is transcribed text"
    )

    return mock_client

@pytest.fixture
async def test_audio_pipeline(mock_openai_client):
    """Audio pipeline with mocked Whisper API"""
    from app.audio_pipeline import AudioPipeline, WhisperStrategy

    pipeline = AudioPipeline()
    strategy = WhisperStrategy()
    strategy.client = mock_openai_client
    pipeline.transcription_strategy = strategy

    return pipeline

@pytest.fixture
def sample_audio_wav():
    """Sample WAV audio bytes for testing"""
    # Minimal valid WAV file header
    return b'RIFF' + b'\x00' * 4 + b'WAVE' + b'fmt ' + b'\x00' * 100

@pytest.fixture
def sample_audio_mp3():
    """Sample MP3 audio bytes for testing"""
    # MP3 frame sync
    return b'\xff\xfb' + b'\x00' * 100

@pytest.fixture
def sample_audio_base64():
    """Sample base64 encoded audio"""
    import base64
    wav_data = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'fmt ' + b'\x00' * 100
    return base64.b64encode(wav_data).decode()

# FastAPI Test Client Fixtures
@pytest.fixture
async def test_app():
    """FastAPI test application"""
    from fastapi.testclient import TestClient
    from app.main import app

    # Override lifespan to skip Redis initialization in tests
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def test_lifespan(app):
        from app.database import db
        await db.initialize()
        yield

    app.router.lifespan_context = test_lifespan

    return app

@pytest.fixture
def client(test_app):
    """HTTP test client"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)

# Test Data Fixtures
@pytest.fixture
def sample_context():
    """Sample conversation context"""
    return [
        {
            "timestamp": "2024-01-15T10:00:00",
            "user": "I'm learning about databases",
            "agent": "That's great! What interests you about databases?"
        },
        {
            "timestamp": "2024-01-15T10:00:30",
            "user": "SQL and NoSQL differences",
            "agent": "Good question! What have you learned so far?"
        }
    ]

@pytest.fixture
def sample_exchanges():
    """Sample conversation exchanges for testing"""
    return [
        {"user": "Hello", "agent": "Hi there! What would you like to learn about?"},
        {"user": "Python", "agent": "Python is great! What aspect interests you?"},
        {"user": "Functions", "agent": "Functions are fundamental. Tell me more."}
    ]

# Mock HTTP Responses
@pytest.fixture
def mock_twilio_request():
    """Mock Twilio webhook request data"""
    return {
        "CallSid": "CA1234567890",
        "From": "+15551234567",
        "To": "+15559876543",
        "CallStatus": "in-progress",
        "SpeechResult": "I want to learn about AI"
    }

# Async Testing Helpers
@pytest.fixture
def assert_called_once():
    """Helper to assert async mock called once"""
    def _assert(mock, *args, **kwargs):
        assert mock.call_count == 1
        if args or kwargs:
            mock.assert_called_once_with(*args, **kwargs)
    return _assert

# Timing Fixtures for Performance Testing
@pytest.fixture
def timing():
    """Measure execution time"""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.end_time is None:
                return time.time() - self.start_time
            return self.end_time - self.start_time

    return Timer()

# Cleanup
@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after each test"""
    yield
    # Add any cleanup logic here
