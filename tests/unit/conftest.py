"""
Unit Test Configuration and Fixtures
Provides sync fixtures optimized for unit testing
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def research_agent():
    """Create a research agent instance (sync fixture)"""
    from app.agents.research_agent import ResearchAgent
    agent = ResearchAgent(agent_id="research-agent-1")
    agent._initialized = True
    return agent


@pytest.fixture
def test_db(tmp_path):
    """Create a test database instance (sync fixture)"""
    import aiosqlite
    import json

    class TestDatabase:
        """Test database that doesn't use structlog"""
        def __init__(self, db_path: str):
            self.db_path = db_path
            self._initialized = False
            self._loop = None

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
                    CREATE INDEX IF NOT EXISTS idx_session_timestamp
                    ON captures(session_id, timestamp)
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

        def get_connection(self):
            from contextlib import asynccontextmanager
            db_path = self.db_path
            @asynccontextmanager
            async def _conn():
                async with aiosqlite.connect(db_path) as db:
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

    # Initialize synchronously using new event loop
    loop = asyncio.new_event_loop()
    loop.run_until_complete(db.initialize())
    loop.close()

    return db


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing"""
    from unittest.mock import patch
    with patch("httpx.AsyncClient") as mock:
        yield mock


# ============================================================================
# Audio Pipeline Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_wav():
    """Sample WAV audio bytes for testing (minimal valid WAV header)"""
    # Minimal WAV file header with some audio data
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
def sample_audio_mp3():
    """Sample MP3 audio bytes for testing (ID3 header)"""
    return b'ID3\x04\x00\x00\x00\x00\x00\x00'


@pytest.fixture
def sample_audio_base64(sample_audio_wav):
    """Base64 encoded audio for testing"""
    import base64
    return base64.b64encode(sample_audio_wav).decode('utf-8')


@pytest.fixture
def test_audio_pipeline():
    """AudioPipeline instance with mocked transcription for testing"""
    from app.audio_pipeline import AudioPipeline, WhisperStrategy

    pipeline = AudioPipeline()

    # Mock the transcription strategy
    mock_strategy = MagicMock(spec=WhisperStrategy)
    mock_strategy.client = MagicMock()
    mock_strategy.client.audio = MagicMock()
    mock_strategy.client.audio.transcriptions = MagicMock()
    mock_strategy.client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="This is transcribed text.")
    )

    # Mock the transcribe method to return async
    async def mock_transcribe(audio_data):
        return "This is transcribed text."

    mock_strategy.transcribe = AsyncMock(side_effect=mock_transcribe)

    pipeline.transcription_strategy = mock_strategy

    return pipeline


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for audio transcription tests"""
    mock_client = MagicMock()
    mock_client.audio = MagicMock()
    mock_client.audio.transcriptions = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "This is transcribed text"
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
    return mock_client


# ============================================================================
# Timing Fixture
# ============================================================================

@pytest.fixture
def timing():
    """Timing context manager for performance tests"""
    import time

    class TimingContext:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self._elapsed = 0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self._elapsed = self.end_time - self.start_time

        @property
        def elapsed(self):
            """Elapsed time in seconds"""
            return self._elapsed

        @property
        def elapsed_ms(self):
            """Elapsed time in milliseconds"""
            return self._elapsed * 1000

    return TimingContext()


# ============================================================================
# Conversation Handler Fixtures
# ============================================================================

@pytest.fixture
def test_conversation_handler(mock_anthropic_client):
    """ConversationHandler instance with mocked Anthropic client for testing"""
    from app.conversation_handler import ConversationHandler

    handler = ConversationHandler()
    handler.client = mock_anthropic_client

    return handler


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for conversation tests"""
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="This is a test response. What would you like to explore?")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    return mock_client


# ============================================================================
# Database with Data Fixtures
# ============================================================================

@pytest.fixture
def db_with_data(test_db):
    """Test database pre-populated with sample data"""
    # Initialize synchronously
    loop = asyncio.new_event_loop()

    async def populate():
        await test_db.save_exchange(
            session_id="session1",
            user_text="Tell me about Python",
            agent_text="Python is a programming language..."
        )
        await test_db.save_exchange(
            session_id="session1",
            user_text="Its syntax",
            agent_text="Python has clean syntax..."
        )
        await test_db.save_exchange(
            session_id="session2",
            user_text="What is AI?",
            agent_text="AI is artificial intelligence..."
        )

    loop.run_until_complete(populate())
    loop.close()

    return test_db
