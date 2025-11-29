"""
Test Configuration and Fixtures for Multi-Agent System

Provides fixtures for testing agents, orchestrator, and agent communication.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
from datetime import datetime
import uuid

# ============================================================================
# Mock Environment
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def mock_agent_env():
    """Mock environment variables for agent testing"""
    with patch.dict('os.environ', {
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'REDIS_URL': 'redis://localhost:6379/1',
        'DATABASE_URL': 'sqlite:///:memory:',
        'CLAUDE_MODEL': 'claude-3-5-sonnet-20241022',
        'WHISPER_MODEL': 'whisper-1',
        'ENABLE_V2_AGENTS': 'true',
        'LOG_LEVEL': 'WARNING'
    }):
        yield

# ============================================================================
# AgentMessage Fixtures
# ============================================================================

@pytest.fixture
def sample_agent_message():
    """Sample AgentMessage for testing"""
    from app.agents.base import AgentMessage, MessageType

    return AgentMessage(
        message_id=f"msg-{uuid.uuid4()}",
        sender="user",
        recipient="conversation_agent",
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "Hello, I'm learning about AI",
            "session_id": "test-session-123"
        },
        metadata={"source": "test"},
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def conversation_request():
    """Conversation request message"""
    from app.agents.base import AgentMessage, MessageType

    return lambda text: AgentMessage(
        sender="user",
        recipient="conversation_agent",
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": text}
    )

@pytest.fixture
def analysis_request():
    """Analysis request message"""
    from app.agents.base import AgentMessage, MessageType

    return lambda text: AgentMessage(
        sender="user",
        recipient="analysis_agent",
        message_type=MessageType.ANALYSIS_REQUEST,
        content={"text": text, "analysis_types": ["concepts", "entities"]}
    )

# ============================================================================
# Mock API Clients
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for Claude API"""
    mock = AsyncMock()

    # Create mock content block with type attribute (required for tool loop)
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "This is a helpful response about AI."

    # Create mock message response
    mock_message = MagicMock()
    mock_message.content = [mock_content_block]
    mock_message.usage = MagicMock(input_tokens=50, output_tokens=30)
    mock_message.model = "claude-3-5-sonnet-20241022"
    mock_message.stop_reason = "end_turn"

    mock.messages.create = AsyncMock(return_value=mock_message)

    return mock

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock = AsyncMock()

    # Mock Whisper transcription
    mock.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="This is transcribed text")
    )

    # Mock GPT-4V vision
    mock_vision_response = MagicMock()
    mock_vision_response.choices = [
        MagicMock(message=MagicMock(content="This image shows a neural network diagram."))
    ]
    mock.chat.completions.create = AsyncMock(return_value=mock_vision_response)

    return mock

# ============================================================================
# Mock Search APIs
# ============================================================================

@pytest.fixture
def mock_web_search():
    """Mock web search function - returns dict with results"""
    async def search(query: str, max_results: int = 5) -> Dict[str, Any]:
        return {
            "source": "web",
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1}: {query}",
                    "url": f"https://example.com/result-{i+1}",
                    "content": f"This is a search result about {query}. It contains relevant information.",
                    "snippet": f"This is a search result about {query}. It contains relevant information.",
                    "source": "web",
                    "relevance_score": 0.95 - (i * 0.1),
                    "published_date": "2024-11-20"
                }
                for i in range(min(max_results, 3))
            ]
        }

    return search

@pytest.fixture
def mock_arxiv_search():
    """Mock ArXiv paper search - returns dict with results"""
    async def search(query: str, max_results: int = 5) -> Dict[str, Any]:
        return {
            "source": "arxiv",
            "query": query,
            "results": [
                {
                    "title": f"Research Paper {i+1}: {query}",
                    "url": f"https://arxiv.org/abs/2024.{10000+i}",
                    "snippet": f"Abstract: This paper explores {query} using novel approaches...",
                    "content": f"Abstract: This paper explores {query} using novel approaches...",
                    "source": "arxiv",
                    "authors": ["Smith, J.", "Johnson, A.", "Williams, K."],
                    "published_date": "2024-11-15",
                    "relevance_score": 0.9 - (i * 0.1)
                }
                for i in range(min(max_results, 2))
            ]
        }

    return search

@pytest.fixture
def mock_wikipedia_search():
    """Mock Wikipedia lookup - returns dict with results"""
    async def search(query: str, max_results: int = 5) -> Dict[str, Any]:
        return {
            "source": "wikipedia",
            "query": query,
            "results": [
                {
                    "title": query.title(),
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "snippet": f"Wikipedia article about {query}. Comprehensive information...",
                    "extract": f"This is the full text of the Wikipedia article about {query}.",
                    "source": "wikipedia",
                    "categories": ["Technology", "Computer Science"],
                    "last_updated": "2024-11-15"
                }
            ]
        }

    return search

# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def conversation_agent(mock_anthropic_client):
    """ConversationAgent with mocked Claude API"""
    from app.agents.conversation_agent import ConversationAgent

    agent = ConversationAgent()
    agent.client = mock_anthropic_client
    # Skip initialization - will be called in tests if needed
    agent._initialized = True

    return agent

@pytest.fixture
def analysis_agent():
    """AnalysisAgent for testing"""
    from app.agents.analysis_agent import AnalysisAgent

    agent = AnalysisAgent()
    agent._initialized = True

    return agent

@pytest.fixture
def research_agent(mock_web_search, mock_arxiv_search, mock_wikipedia_search):
    """ResearchAgent with mocked search APIs"""
    from app.agents.research_agent import ResearchAgent

    agent = ResearchAgent()

    # Mock the internal methods that the agent actually uses
    agent._web_search = mock_web_search
    agent._duckduckgo_search = mock_web_search  # Also mock the fallback
    agent._tavily_search = mock_web_search
    agent._arxiv_search = mock_arxiv_search
    agent._wikipedia_search = mock_wikipedia_search
    agent._initialized = True

    return agent

@pytest.fixture
def synthesis_agent(mock_anthropic_client):
    """SynthesisAgent with mocked Claude API"""
    from app.agents.synthesis_agent import SynthesisAgent

    agent = SynthesisAgent()
    agent.client = mock_anthropic_client
    agent._initialized = True

    return agent

@pytest.fixture
def vision_agent(mock_openai_client):
    """VisionAgent with mocked GPT-4V"""
    from app.agents.vision_agent import VisionAgent

    agent = VisionAgent()
    agent.client = mock_openai_client
    agent._initialized = True

    return agent

# ============================================================================
# Orchestrator Fixtures
# ============================================================================

@pytest.fixture
def orchestrator(
    conversation_agent,
    analysis_agent,
    research_agent,
    synthesis_agent
):
    """AgentOrchestrator with all agents"""
    from app.agents.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator()

    # Inject mocked agents
    orch.agents = {
        "conversation": conversation_agent,
        "analysis": analysis_agent,
        "research": research_agent,
        "synthesis": synthesis_agent
    }

    orch._initialized = True

    return orch

# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_conversation_context():
    """Sample conversation context"""
    return [
        {
            "timestamp": "2024-11-21T10:00:00",
            "user": "I'm learning about machine learning",
            "agent": "That's great! What aspect of ML interests you?"
        },
        {
            "timestamp": "2024-11-21T10:00:30",
            "user": "Neural networks and deep learning",
            "agent": "Excellent choice! Would you like to start with the basics?"
        },
        {
            "timestamp": "2024-11-21T10:01:00",
            "user": "Yes, please explain backpropagation",
            "agent": "Backpropagation is the algorithm used to train neural networks..."
        }
    ]

@pytest.fixture
def sample_conversation_history():
    """Extended conversation history for synthesis"""
    return [
        {"user": "I'm learning Python", "agent": "Great! What interests you about Python?"},
        {"user": "Web development", "agent": "Python is excellent for web dev!"},
        {"user": "Should I use Django or FastAPI?", "agent": "What's your use case?"},
        {"user": "REST API for mobile app", "agent": "FastAPI is perfect for that!"},
        {"user": "How do I handle authentication?", "agent": "JWT tokens are common."},
        {"user": "Can you explain JWT?", "agent": "JWT stands for JSON Web Token..."},
        {"user": "What about database?", "agent": "PostgreSQL or MongoDB work well."},
        {"user": "I'll use PostgreSQL", "agent": "Good choice! SQLAlchemy is helpful."},
        {"user": "How do I deploy?", "agent": "Railway, Heroku, or AWS are options."},
        {"user": "Railway sounds easy", "agent": "Yes, Railway is very beginner-friendly!"}
    ]

@pytest.fixture
def sample_analysis_result():
    """Sample output from AnalysisAgent"""
    return {
        "concepts": ["machine learning", "neural networks", "deep learning", "backpropagation"],
        "entities": {
            "TECH": ["Python", "TensorFlow"],
            "FIELD": ["AI", "computer science"]
        },
        "sentiment": {
            "score": 0.85,
            "label": "enthusiastic"
        },
        "topics": ["artificial intelligence", "programming"],
        "relationships": [
            {"source": "neural networks", "relation": "is_part_of", "target": "machine learning"},
            {"source": "backpropagation", "relation": "used_in", "target": "neural networks"}
        ]
    }

@pytest.fixture
def sample_research_results():
    """Sample output from ResearchAgent"""
    return {
        "query": "transformer architecture",
        "results": [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "snippet": "We propose a new simple network architecture, the Transformer...",
                "source": "arxiv",
                "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
                "relevance_score": 0.98
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "url": "https://arxiv.org/abs/1810.04805",
                "snippet": "We introduce BERT, Bidirectional Encoder Representations...",
                "source": "arxiv",
                "authors": ["Devlin, J.", "Chang, M."],
                "relevance_score": 0.95
            }
        ],
        "sources_used": ["arxiv", "web"]
    }

# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def test_agent_db():
    """In-memory database for agent testing"""
    from app.database import Database

    db = Database(":memory:")
    db._initialized = True

    return db

@pytest.fixture
def db_with_agent_data(test_agent_db):
    """Database pre-populated with agent test data"""
    # Data will be added in tests if needed
    return test_agent_db

# ============================================================================
# Performance Testing Fixtures
# ============================================================================

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

        @property
        def elapsed_ms(self):
            return self.elapsed * 1000

    return Timer()

# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def create_message():
    """Helper to create AgentMessages"""
    from app.agents.base import AgentMessage, MessageType

    def _create(
        sender: str = "user",
        recipient: str = "agent",
        message_type: MessageType = MessageType.CONVERSATION_REQUEST,
        content: Dict[str, Any] = None
    ) -> AgentMessage:
        return AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content or {},
            timestamp=datetime.utcnow()
        )

    return _create

@pytest.fixture
def assert_agent_response():
    """Helper to assert AgentMessage response"""
    def _assert(
        response,
        expected_type=None,
        has_content_keys=None,
        min_length=None
    ):
        from app.agents.base import AgentMessage

        assert isinstance(response, AgentMessage), "Response must be AgentMessage"

        if expected_type:
            assert response.message_type == expected_type

        if has_content_keys:
            for key in has_content_keys:
                assert key in response.content, f"Missing content key: {key}"

        if min_length and "text" in response.content:
            assert len(response.content["text"]) >= min_length

    return _assert

# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add any global cleanup logic here
