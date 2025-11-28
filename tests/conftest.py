"""
Pytest Configuration and Fixtures
Provides shared fixtures for all test modules
"""
import pytest
import asyncio
import os
import sys
import time
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import external libraries, fall back to mocks if not available
try:
    import neo4j
except ImportError:
    from tests.mocks import neo4j
    sys.modules['neo4j'] = neo4j
    sys.modules['neo4j.exceptions'] = neo4j.exceptions

try:
    import chromadb
except ImportError:
    from tests.mocks import chromadb
    sys.modules['chromadb'] = chromadb
    sys.modules['chromadb.config'] = chromadb.config

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    from tests.mocks import twilio
    sys.modules['twilio'] = twilio
    sys.modules['twilio.rest'] = twilio.rest
    sys.modules['twilio.twiml'] = twilio.twiml
    sys.modules['twilio.twiml.voice_response'] = twilio.twiml.voice_response
    sys.modules['twilio.request_validator'] = twilio.request_validator

# Set test environment variables before importing app modules
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_learning_captures.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

# Mock structlog before importing app modules
class MockLogger:
    """Mock structlog logger that accepts kwargs"""
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method

# Patch the logger module before app imports
mock_logger = MockLogger()
sys.modules['app.logger'] = MagicMock()
sys.modules['app.logger'].db_logger = mock_logger
sys.modules['app.logger'].state_logger = mock_logger
sys.modules['app.logger'].audio_logger = mock_logger
sys.modules['app.logger'].api_logger = mock_logger
sys.modules['app.logger'].conversation_logger = mock_logger

# Also mock resilience module that might not exist
def mock_circuit_breaker(*args, **kwargs):
    """Mock circuit breaker decorator that accepts any args/kwargs"""
    def decorator(func):
        return func
    return decorator

def mock_retry(*args, **kwargs):
    """Mock retry decorator that accepts any args/kwargs"""
    def decorator(func):
        return func
    return decorator

def mock_timeout(*args, **kwargs):
    """Mock timeout decorator that accepts any args/kwargs"""
    def decorator(func):
        return func
    return decorator

class CircuitBreakerOpen(Exception):
    """Mock circuit breaker exception"""
    pass

sys.modules['app.resilience'] = MagicMock()
sys.modules['app.resilience'].with_circuit_breaker = mock_circuit_breaker
sys.modules['app.resilience'].with_retry = mock_retry
sys.modules['app.resilience'].with_timeout = mock_timeout
sys.modules['app.resilience'].CircuitBreakerOpen = CircuitBreakerOpen

# Mock advanced_prompts module
mock_prompt_engine = MagicMock()
mock_prompt_engine.build_system_prompt = MagicMock(return_value="Test system prompt")
mock_prompt_engine.build_user_message = MagicMock(return_value="Test user message")
mock_prompt_engine.select_strategy_for_context = MagicMock(return_value="basic")
mock_prompt_engine.extract_response = MagicMock(return_value=("Test response", None))
sys.modules['app.advanced_prompts'] = MagicMock()
sys.modules['app.advanced_prompts'].prompt_engine = mock_prompt_engine
sys.modules['app.advanced_prompts'].PromptStrategy = MagicMock()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for conversation tests"""
    mock_client = AsyncMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="This is a test response. What would you like to explore?")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for audio transcription tests"""
    mock_client = AsyncMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value="Test transcription text")
    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for state management tests"""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.setex = AsyncMock(return_value=True)
    mock_client.delete = AsyncMock(return_value=1)
    mock_client.scan_iter = AsyncMock(return_value=iter([]))
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def sample_context() -> list:
    """Sample conversation context for testing"""
    return [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "user": "I'm learning about machine learning",
            "agent": "That's exciting! What aspect interests you most?"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "user": "Neural networks seem complex",
            "agent": "They can be! Let's break it down. What do you already know about them?"
        }
    ]


@pytest.fixture
def sample_session_metadata() -> Dict[str, Any]:
    """Sample session metadata for testing"""
    return {
        "created_at": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat(),
        "exchange_count": 2,
        "source": "test"
    }


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Sample WAV audio bytes for testing (minimal valid WAV header)"""
    # Minimal WAV file header
    return (
        b'RIFF' +  # ChunkID
        b'\x24\x00\x00\x00' +  # ChunkSize (36 bytes)
        b'WAVE' +  # Format
        b'fmt ' +  # Subchunk1ID
        b'\x10\x00\x00\x00' +  # Subchunk1Size (16 for PCM)
        b'\x01\x00' +  # AudioFormat (1 = PCM)
        b'\x01\x00' +  # NumChannels (1 = mono)
        b'\x44\xac\x00\x00' +  # SampleRate (44100)
        b'\x88\x58\x01\x00' +  # ByteRate
        b'\x02\x00' +  # BlockAlign
        b'\x10\x00' +  # BitsPerSample (16)
        b'data' +  # Subchunk2ID
        b'\x00\x00\x00\x00'  # Subchunk2Size (0 bytes of data)
    )


@pytest.fixture
def sample_mp3_bytes() -> bytes:
    """Sample MP3 audio bytes for testing (ID3 header)"""
    return b'ID3\x04\x00\x00\x00\x00\x00\x00'


@pytest.fixture
def sample_ogg_bytes() -> bytes:
    """Sample OGG audio bytes for testing"""
    return b'OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00'


@pytest.fixture
def sample_base64_audio() -> str:
    """Base64 encoded audio for testing"""
    import base64
    wav_header = (
        b'RIFF' +
        b'\x24\x00\x00\x00' +
        b'WAVE' +
        b'fmt ' +
        b'\x10\x00\x00\x00' +
        b'\x01\x00' +
        b'\x01\x00' +
        b'\x44\xac\x00\x00' +
        b'\x88\x58\x01\x00' +
        b'\x02\x00' +
        b'\x10\x00' +
        b'data' +
        b'\x00\x00\x00\x00'
    )
    return base64.b64encode(wav_header).decode('utf-8')


@pytest.fixture
async def test_database(tmp_path):
    """Create a test database instance"""
    # Import Database class directly without the module-level singleton
    import aiosqlite
    import json

    class TestDatabase:
        """Test database that doesn't use structlog"""
        def __init__(self, db_path: str):
            self.db_path = db_path
            self._initialized = False

        async def initialize(self):
            if self._initialized:
                return
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS captures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_text TEXT NOT NULL,
                        agent_text TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                await db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts
                    USING fts5(
                        session_id UNINDEXED,
                        user_text,
                        agent_text,
                        content=captures,
                        content_rowid=id
                    )
                """)
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS captures_ai
                    AFTER INSERT ON captures BEGIN
                        INSERT INTO captures_fts(rowid, session_id, user_text, agent_text)
                        VALUES (new.id, new.session_id, new.user_text, new.agent_text);
                    END
                """)
                await db.commit()
            self._initialized = True

        async def get_connection(self):
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def _conn():
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    yield db
            return _conn()

        async def save_exchange(self, session_id, user_text, agent_text, metadata=None):
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "INSERT INTO captures (session_id, user_text, agent_text, metadata) VALUES (?, ?, ?, ?)",
                    (session_id, user_text, agent_text, json.dumps(metadata or {}))
                )
                await db.commit()
                return cursor.lastrowid

        async def get_session_history(self, session_id, limit=5):
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM captures WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in reversed(rows)]

        async def search_captures(self, query, limit=20):
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """SELECT c.*, snippet(captures_fts, 1, '<mark>', '</mark>', '...', 32) as user_snippet
                       FROM captures c JOIN captures_fts ON c.id = captures_fts.rowid
                       WHERE captures_fts MATCH ? LIMIT ?""",
                    (query, limit)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

        async def get_stats(self):
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """SELECT COUNT(*) as total_captures,
                              COUNT(DISTINCT session_id) as unique_sessions,
                              MAX(timestamp) as last_capture FROM captures"""
                )
                return dict(await cursor.fetchone())

    db_path = str(tmp_path / "test_captures.db")
    db = TestDatabase(db_path)
    await db.initialize()
    yield db


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    mock = MagicMock()
    mock.anthropic_api_key = "test-key"
    mock.openai_api_key = "test-openai-key"
    mock.claude_model = "claude-3-haiku-20240307"
    mock.claude_max_tokens = 150
    mock.claude_temperature = 0.7
    mock.whisper_model = "whisper-1"
    mock.max_audio_duration = 60
    mock.redis_url = "redis://localhost:6379"
    mock.redis_ttl = 1800
    mock.session_timeout = 180
    mock.max_context_exchanges = 5
    mock.host = "0.0.0.0"
    mock.port = 8000
    mock.cors_origins = ["*"]
    mock.database_url = "sqlite:///./test.db"
    mock.twilio_account_sid = None
    mock.twilio_auth_token = None
    mock.twilio_phone_number = None
    return mock


@pytest.fixture
def twilio_voice_form_data() -> Dict[str, str]:
    """Sample Twilio voice webhook form data"""
    return {
        "CallSid": "CA1234567890abcdef",
        "From": "+15551234567",
        "To": "+15559876543",
        "CallStatus": "ringing"
    }


@pytest.fixture
def twilio_speech_form_data() -> Dict[str, str]:
    """Sample Twilio speech result form data"""
    return {
        "CallSid": "CA1234567890abcdef",
        "SpeechResult": "I'm learning about artificial intelligence",
        "Confidence": "0.95"
    }


# Async test support
pytest_plugins = ['pytest_asyncio']


# FastAPI TestClient fixture
@pytest.fixture
def client():
    """Create FastAPI TestClient for integration tests."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


# Timing fixture for performance tests
class TimingContext:
    """Context manager for timing code execution."""
    def __init__(self):
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


@pytest.fixture
def timing():
    """Provide timing context for performance tests."""
    return TimingContext()


# Alias for test_db (used by some tests)
@pytest.fixture
async def test_db(test_database):
    """Alias for test_database fixture."""
    return test_database


# ============================================================================
# Neo4j Fixtures for Knowledge Graph Tests
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver with sessions"""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()

    # Mock session
    session = AsyncMock()

    # Mock concept result
    concept_result = AsyncMock()
    concept_result.single = AsyncMock(return_value={
        'name': 'test_concept',
        'description': 'Test description',
        'frequency': 1
    })

    # Mock related concepts result
    related_result = AsyncMock()
    related_result.values = AsyncMock(return_value=[
        ['concept1', 'desc1', 5, ['RELATES_TO'], [0.8], 1],
        ['concept2', 'desc2', 3, ['BUILDS_ON'], [0.7], 2]
    ])

    # Mock stats result
    stats_result = AsyncMock()
    stats_result.single = AsyncMock(return_value={
        'concept_count': 10,
        'relationship_count': 15,
        'entity_count': 5,
        'session_count': 3,
        'topic_count': 2
    })

    session.run = AsyncMock(return_value=concept_result)

    # Mock session context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()

    driver.session = MagicMock(return_value=session)

    return driver, session, concept_result, related_result, stats_result


@pytest.fixture
async def mock_knowledge_graph_store(mock_neo4j_driver):
    """Mock KnowledgeGraphStore instance"""
    from app.knowledge_graph.graph_store import KnowledgeGraphStore
    from app.knowledge_graph.config import KnowledgeGraphConfig

    driver, session, _, _, _ = mock_neo4j_driver

    config = KnowledgeGraphConfig()
    store = KnowledgeGraphStore(config)

    # Set up mocks
    store.driver = driver
    store._initialized = True

    # Mock session context manager
    store.session = lambda: session

    return store


@pytest.fixture
def sample_concept_data():
    """Sample concept data for testing"""
    return {
        'name': 'machine learning',
        'description': 'A field of AI',
        'metadata': {'category': 'AI'},
        'topic': 'artificial intelligence'
    }


@pytest.fixture
def sample_relationship_data():
    """Sample relationship data for testing"""
    return {
        'from_concept': 'neural networks',
        'to_concept': 'deep learning',
        'relationship_type': 'RELATES_TO',
        'strength': 0.9
    }


@pytest.fixture
def sample_session_data():
    """Sample session data for testing"""
    return {
        'session_id': 'test_session_123',
        'concepts': ['machine learning', 'neural networks'],
        'entities': [('TensorFlow', 'PRODUCT'), ('Python', 'LANGUAGE')],
        'metadata': {'user_id': 'test_user'}
    }


# ============================================================================
# ChromaDB Fixtures for Vector Store Tests
# ============================================================================

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    try:
        from tests.mocks.chromadb import PersistentClient, Collection
    except:
        import chromadb
        PersistentClient = chromadb.PersistentClient
        Collection = chromadb.Collection

    client = MagicMock(spec=PersistentClient)

    # Mock collection
    collection = MagicMock(spec=Collection)
    collection.name = "conversations"
    collection.count = MagicMock(return_value=100)
    collection.add = MagicMock()
    collection.query = MagicMock(return_value={
        'ids': [['id1', 'id2', 'id3']],
        'distances': [[0.1, 0.2, 0.3]],
        'documents': [['doc1', 'doc2', 'doc3']],
        'metadatas': [[{'session_id': 'test'}, {}, {}]]
    })
    collection.delete = MagicMock()

    client.get_or_create_collection = MagicMock(return_value=collection)
    client.list_collections = MagicMock(return_value=[{'name': 'conversations'}])
    client.delete_collection = MagicMock()

    return client


@pytest.fixture
async def mock_vector_store(mock_chroma_client):
    """Mock VectorStore instance"""
    from app.vector.vector_store import VectorStore

    store = VectorStore()
    store.client = mock_chroma_client
    store._initialized = True

    # Mock collections
    collection = mock_chroma_client.get_or_create_collection("conversations")
    store.collections = {'conversations': collection}

    return store


@pytest.fixture
def mock_embedding_generator():
    """Mock EmbeddingGenerator"""
    import numpy as np

    generator = AsyncMock()
    generator.initialize = AsyncMock()
    generator.generate_embedding = AsyncMock(
        return_value=np.random.rand(384).astype('float32')
    )
    generator.generate_batch = AsyncMock(
        return_value=[np.random.rand(384).astype('float32') for _ in range(3)]
    )
    generator.close = AsyncMock()

    return generator


@pytest.fixture
def mock_analysis_agent():
    """Mock AnalysisAgent for concept extraction"""
    agent = AsyncMock()

    async def mock_process(message):
        return MagicMock(
            content={
                "concepts": ["machine learning", "neural networks"],
                "entities": [
                    {"text": "Python", "label": "PRODUCT"},
                    {"text": "TensorFlow", "label": "PRODUCT"}
                ],
                "keywords": [
                    {"keyword": "algorithm", "frequency": 3}
                ],
                "topics": [
                    {"topic": "technology", "confidence": 0.9}
                ]
            }
        )

    agent.process = mock_process
    return agent


# ============================================================================
# Agent Fixtures for tests/agents/
# ============================================================================

@pytest.fixture
def sample_agent_message():
    """Sample AgentMessage for testing"""
    from app.agents.base import AgentMessage, MessageType
    return AgentMessage(
        sender="user",
        recipient="test_agent",
        message_type=MessageType.REQUEST,
        content={"text": "Hello, test agent"}
    )


@pytest.fixture
def sample_conversation_context():
    """Sample conversation context for testing"""
    return [
        {"user": "What is machine learning?", "agent": "Machine learning is..."},
        {"user": "Can you explain more?", "agent": "Sure, let me elaborate..."}
    ]


@pytest.fixture
async def conversation_agent():
    """ConversationAgent instance for testing"""
    from app.agents.conversation_agent import ConversationAgent
    agent = ConversationAgent()
    # Mock the client to avoid actual API calls
    agent.client = AsyncMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Test response from agent")]
    agent.client.messages.create = AsyncMock(return_value=mock_message)
    yield agent
    await agent.cleanup()


@pytest.fixture
async def orchestrator(mock_anthropic_client):
    """AgentOrchestrator instance for testing"""
    from app.agents.orchestrator import AgentOrchestrator
    from app.agents.conversation_agent import ConversationAgent

    orchestrator = AgentOrchestrator()
    await orchestrator.start()

    # Register conversation agent with mocked client
    agent = ConversationAgent()
    agent.client = mock_anthropic_client
    await orchestrator.register_agent(agent, capabilities=["conversation"])

    yield orchestrator

    await orchestrator.stop()


@pytest.fixture
async def test_agent_db(test_database):
    """Alias for test_database for agent tests"""
    return test_database


@pytest.fixture
def timing():
    """Timing context manager for performance tests"""
    class TimingContext:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = 0

        def __enter__(self):
            import time
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            import time
            self.end_time = time.perf_counter()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000

    return TimingContext()
