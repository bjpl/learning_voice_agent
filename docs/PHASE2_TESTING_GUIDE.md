# Phase 2: Multi-Agent Testing Guide

**Version:** 2.0.0-alpha
**Last Updated:** 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Testing Strategy](#testing-strategy)
3. [Test Environment Setup](#test-environment-setup)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [End-to-End Testing](#end-to-end-testing)
7. [Performance Testing](#performance-testing)
8. [Mocking External Services](#mocking-external-services)
9. [Test Coverage](#test-coverage)
10. [CI/CD Integration](#cicd-integration)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### Testing Philosophy

Phase 2 multi-agent testing follows the **test pyramid** approach:

```
      /\
     /  \      E2E (5%)
    /────\     - Full user journeys
   /      \    - Multi-agent workflows
  /────────\   Integration (20%)
 /          \  - Agent communication
/────────────\ - Tool integration
               Unit Tests (75%)
               - Individual agents
               - Message handling
               - Business logic
```

### Coverage Goals

- **Overall:** 80%+ code coverage
- **Critical paths:** 95%+ coverage
- **Agent core logic:** 100% coverage
- **Integration points:** 90%+ coverage

### Test Tools

- **pytest:** Test runner and framework
- **pytest-asyncio:** Async test support
- **pytest-mock:** Mocking utilities
- **pytest-cov:** Coverage reporting
- **faker:** Test data generation
- **locust:** Load testing
- **hypothesis:** Property-based testing

---

## Testing Strategy

### Test Levels

#### 1. Unit Tests

Test individual agent methods in isolation.

**Focus:**
- Agent initialization
- Message processing logic
- Tool invocation
- Error handling
- Helper methods

**Example:**
```python
async def test_conversation_agent_basic_response(conversation_agent):
    """Test basic response generation"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "Hello"}
    )

    response = await conversation_agent.process(message)

    assert response.message_type == MessageType.CONVERSATION_RESPONSE
    assert "text" in response.content
    assert len(response.content["text"]) > 0
```

#### 2. Integration Tests

Test agent interactions and communication.

**Focus:**
- Agent-to-agent messaging
- Orchestrator routing
- Tool integration
- Database interactions
- API integrations

**Example:**
```python
async def test_multi_agent_collaboration(orchestrator):
    """Test conversation and analysis agents working together"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "I'm learning about neural networks"}
    )

    response = await orchestrator.process(message)

    # Should involve both agents
    assert "conversation_agent" in response.metadata["agents_used"]
    assert "analysis_agent" in response.metadata["agents_used"]
```

#### 3. End-to-End Tests

Test complete user workflows.

**Focus:**
- Full conversation flows
- Multi-turn interactions
- Research-augmented responses
- Insight generation
- Error recovery

**Example:**
```python
async def test_research_augmented_conversation_flow(client):
    """Test complete research + conversation workflow"""
    # Step 1: User asks research question
    response1 = await client.post("/v2/conversation", json={
        "text": "What are the latest developments in transformers?",
        "enable_research": True
    })

    assert response1.status_code == 200
    data = response1.json()

    # Should include research
    assert "research" in data["metadata"]
    assert len(data["metadata"]["research"]["results"]) > 0

    # Response should reference research
    assert "recent" in data["agent_text"].lower() or "latest" in data["agent_text"].lower()
```

---

## Test Environment Setup

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Install coverage tools
pip install pytest-cov coverage

# Optional: Install load testing
pip install locust
```

### Configuration

```python
# tests/test_config.py

import pytest
from unittest.mock import patch

@pytest.fixture(scope="session", autouse=True)
def test_env():
    """Configure test environment"""
    with patch.dict('os.environ', {
        'ANTHROPIC_API_KEY': 'test-key',
        'OPENAI_API_KEY': 'test-key',
        'REDIS_URL': 'redis://localhost:6379/1',  # Test database
        'DATABASE_URL': 'sqlite:///:memory:',
        'CLAUDE_MODEL': 'claude-3-haiku-20240307',
        'ENABLE_V2_AGENTS': 'true',
        'LOG_LEVEL': 'WARNING'  # Reduce noise in tests
    }):
        yield
```

### Test Database

```python
# tests/agents/conftest.py

@pytest.fixture
async def test_db():
    """In-memory database for testing"""
    from app.database import Database

    db = Database(":memory:")
    await db.initialize()

    yield db

    # Cleanup happens automatically with in-memory DB
```

---

## Unit Testing

### BaseAgent Tests

```python
# tests/agents/test_base_agent.py

import pytest
from app.agents.base import BaseAgent, AgentMessage, MessageType

class TestAgent(BaseAgent):
    """Concrete agent for testing BaseAgent"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        return self._create_response(
            content={"result": "test"},
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=message
        )

@pytest.fixture
def test_agent():
    return TestAgent("test_agent")

def test_agent_initialization(test_agent):
    """Test agent initialization"""
    assert test_agent.agent_id == "test_agent"
    assert test_agent.state == {}

async def test_agent_process_creates_response(test_agent):
    """Test response creation"""
    message = AgentMessage(
        sender="user",
        recipient="test_agent",
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"test": "data"}
    )

    response = await test_agent.process(message)

    assert response.sender == "test_agent"
    assert response.recipient == "user"
    assert response.correlation_id == message.id

def test_create_response_helper(test_agent):
    """Test _create_response helper method"""
    original = AgentMessage(
        id="original-123",
        sender="user",
        recipient="test_agent",
        message_type=MessageType.CONVERSATION_REQUEST,
        content={}
    )

    response = test_agent._create_response(
        content={"result": "success"},
        message_type=MessageType.CONVERSATION_RESPONSE,
        original_message=original
    )

    assert response.sender == "test_agent"
    assert response.recipient == "user"
    assert response.correlation_id == "original-123"
    assert response.content == {"result": "success"}
```

### ConversationAgent Tests

```python
# tests/agents/test_conversation_agent.py

import pytest
from app.agents.conversation_agent import ConversationAgent
from app.agents.base import AgentMessage, MessageType

@pytest.fixture
async def conversation_agent(mock_anthropic_client):
    """ConversationAgent with mocked Claude API"""
    agent = ConversationAgent()
    agent.client = mock_anthropic_client
    await agent.initialize()
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test_basic_conversation(conversation_agent):
    """Test basic conversation response"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "Hello, how are you?"}
    )

    response = await conversation_agent.process(message)

    assert response.message_type == MessageType.CONVERSATION_RESPONSE
    assert "text" in response.content
    assert len(response.content["text"]) > 0
    assert "model" in response.content

@pytest.mark.asyncio
async def test_conversation_with_context(conversation_agent):
    """Test conversation with previous exchanges"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "What did I just mention?",
            "context": [
                {"user": "I'm learning Python", "agent": "That's great!"}
            ]
        }
    )

    response = await conversation_agent.process(message)

    # Response should reference context
    assert response.content["text"] is not None

@pytest.mark.asyncio
async def test_conversation_with_tool_use(conversation_agent):
    """Test tool invocation"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "What's 123 * 456?",
            "enable_tools": True
        }
    )

    response = await conversation_agent.process(message)

    # Should use calculator tool
    assert "56088" in response.content["text"] or "tools_used" in response.content

@pytest.mark.asyncio
async def test_conversation_error_handling(conversation_agent, mock_anthropic_client):
    """Test error handling when API fails"""
    # Make API fail
    mock_anthropic_client.messages.create.side_effect = Exception("API Error")

    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "Hello"}
    )

    response = await conversation_agent.process(message)

    # Should return error message
    assert response.message_type == MessageType.AGENT_ERROR
    assert "error" in response.content
```

### AnalysisAgent Tests

```python
# tests/agents/test_analysis_agent.py

import pytest
from app.agents.analysis_agent import AnalysisAgent
from app.agents.base import AgentMessage, MessageType

@pytest.fixture
async def analysis_agent():
    agent = AnalysisAgent()
    await agent.initialize()
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test_concept_extraction(analysis_agent):
    """Test concept extraction"""
    message = AgentMessage(
        message_type=MessageType.ANALYSIS_REQUEST,
        content={
            "text": "I'm learning PyTorch for deep learning and neural networks",
            "analysis_types": ["concepts"]
        }
    )

    response = await analysis_agent.process(message)

    assert "concepts" in response.content
    concepts = response.content["concepts"]

    # Should extract key concepts
    assert any("pytorch" in c.lower() for c in concepts)
    assert any("neural" in c.lower() or "network" in c.lower() for c in concepts)

@pytest.mark.asyncio
async def test_entity_recognition(analysis_agent):
    """Test named entity recognition"""
    message = AgentMessage(
        message_type=MessageType.ANALYSIS_REQUEST,
        content={
            "text": "I'm using TensorFlow and PyTorch from Google and Meta",
            "analysis_types": ["entities"]
        }
    )

    response = await analysis_agent.process(message)

    assert "entities" in response.content
    entities = response.content["entities"]

    # Should recognize tech and org entities
    assert "TECH" in entities or "ORG" in entities

@pytest.mark.asyncio
async def test_sentiment_analysis(analysis_agent):
    """Test sentiment analysis"""
    positive_message = AgentMessage(
        message_type=MessageType.ANALYSIS_REQUEST,
        content={
            "text": "I'm so excited about learning AI! It's amazing!",
            "analysis_types": ["sentiment"]
        }
    )

    response = await analysis_agent.process(positive_message)

    assert "sentiment" in response.content
    sentiment = response.content["sentiment"]

    assert sentiment["score"] > 0.5  # Positive sentiment
    assert sentiment["label"] in ["positive", "enthusiastic"]
```

### ResearchAgent Tests

```python
# tests/agents/test_research_agent.py

import pytest
from app.agents.research_agent import ResearchAgent
from app.agents.base import AgentMessage, MessageType

@pytest.fixture
async def research_agent(mock_web_search, mock_arxiv_search):
    """ResearchAgent with mocked search APIs"""
    agent = ResearchAgent()
    agent.web_search = mock_web_search
    agent.arxiv_search = mock_arxiv_search
    await agent.initialize()
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test_web_search(research_agent):
    """Test web search functionality"""
    message = AgentMessage(
        message_type=MessageType.RESEARCH_REQUEST,
        content={
            "query": "transformer architecture",
            "sources": ["web"],
            "max_results": 5
        }
    )

    response = await research_agent.process(message)

    assert "results" in response.content
    assert len(response.content["results"]) <= 5
    assert all("title" in r for r in response.content["results"])
    assert all("url" in r for r in response.content["results"])

@pytest.mark.asyncio
async def test_arxiv_search(research_agent):
    """Test ArXiv paper search"""
    message = AgentMessage(
        message_type=MessageType.RESEARCH_REQUEST,
        content={
            "query": "attention mechanisms",
            "sources": ["arxiv"],
            "max_results": 3
        }
    )

    response = await research_agent.process(message)

    assert "results" in response.content
    results = response.content["results"]

    # ArXiv results should have paper metadata
    assert all(r["source"] == "arxiv" for r in results)
    assert any("authors" in r for r in results)

@pytest.mark.asyncio
async def test_multi_source_search(research_agent):
    """Test searching multiple sources"""
    message = AgentMessage(
        message_type=MessageType.RESEARCH_REQUEST,
        content={
            "query": "neural networks",
            "sources": ["web", "arxiv"],
            "max_results": 5
        }
    )

    response = await research_agent.process(message)

    sources = {r["source"] for r in response.content["results"]}
    assert len(sources) > 1  # Should have results from multiple sources
```

### SynthesisAgent Tests

```python
# tests/agents/test_synthesis_agent.py

import pytest
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.base import AgentMessage, MessageType

@pytest.fixture
async def synthesis_agent():
    agent = SynthesisAgent()
    await agent.initialize()
    yield agent
    await agent.cleanup()

@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing"""
    return [
        {"user": "I'm learning Python", "agent": "Great! What interests you?"},
        {"user": "Functions and classes", "agent": "Good fundamentals!"},
        {"user": "Building a web app", "agent": "Exciting project!"},
        {"user": "Using FastAPI", "agent": "Excellent choice!"},
        {"user": "Adding authentication", "agent": "Important security feature!"}
    ]

@pytest.mark.asyncio
async def test_session_summary(synthesis_agent, sample_conversation_history):
    """Test session summarization"""
    message = AgentMessage(
        message_type=MessageType.SYNTHESIS_REQUEST,
        content={
            "synthesis_type": "summary",
            "conversations": sample_conversation_history
        }
    )

    response = await synthesis_agent.process(message)

    assert "summary" in response.content
    summary = response.content["summary"]

    # Summary should mention key topics
    assert any(kw in summary.lower() for kw in ["python", "fastapi", "web"])

@pytest.mark.asyncio
async def test_pattern_detection(synthesis_agent, sample_conversation_history):
    """Test learning pattern detection"""
    message = AgentMessage(
        message_type=MessageType.SYNTHESIS_REQUEST,
        content={
            "synthesis_type": "insights",
            "conversations": sample_conversation_history
        }
    )

    response = await synthesis_agent.process(message)

    assert "insights" in response.content
    insights = response.content["insights"]

    assert len(insights) > 0
    assert all("text" in i for i in insights)
    assert all("confidence" in i for i in insights)

@pytest.mark.asyncio
async def test_recommendations(synthesis_agent, sample_conversation_history):
    """Test recommendation generation"""
    message = AgentMessage(
        message_type=MessageType.SYNTHESIS_REQUEST,
        content={
            "synthesis_type": "recommendations",
            "conversations": sample_conversation_history
        }
    )

    response = await synthesis_agent.process(message)

    assert "recommendations" in response.content
    recommendations = response.content["recommendations"]

    assert len(recommendations) > 0
    assert all("text" in r for r in recommendations)
    assert all("type" in r for r in recommendations)
```

---

## Integration Testing

### Orchestrator Tests

```python
# tests/agents/test_orchestrator.py

import pytest
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base import AgentMessage, MessageType

@pytest.fixture
async def orchestrator():
    """Orchestrator with all agents"""
    orch = AgentOrchestrator()
    await orch.initialize()
    yield orch
    await orch.cleanup()

@pytest.mark.asyncio
async def test_simple_routing(orchestrator):
    """Test routing to single agent"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "Hello"}
    )

    response = await orchestrator.process(message)

    assert response.message_type == MessageType.CONVERSATION_RESPONSE
    assert "agents_used" in response.metadata
    assert "conversation_agent" in response.metadata["agents_used"]

@pytest.mark.asyncio
async def test_parallel_execution(orchestrator):
    """Test parallel agent execution"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "I'm learning about neural networks",
            "enable_analysis": True
        }
    )

    import time
    start = time.time()
    response = await orchestrator.process(message)
    duration = time.time() - start

    # Should use multiple agents
    agents_used = response.metadata["agents_used"]
    assert len(agents_used) > 1

    # Parallel execution should be faster than sequential
    # (Mock agents should process in < 1 second total)
    assert duration < 2.0

@pytest.mark.asyncio
async def test_research_pipeline(orchestrator):
    """Test research → synthesis → conversation pipeline"""
    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "What are the latest developments in transformers?",
            "enable_research": True
        }
    )

    response = await orchestrator.process(message)

    # Should use research, synthesis, and conversation agents
    agents_used = response.metadata["agents_used"]
    assert "research_agent" in agents_used
    assert "conversation_agent" in agents_used
```

### Agent Communication Tests

```python
# tests/agents/test_integration.py

import pytest
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base import AgentMessage, MessageType

@pytest.mark.asyncio
async def test_end_to_end_conversation_flow(orchestrator, test_db):
    """Test complete conversation flow with persistence"""
    session_id = "test-session-123"

    # First message
    msg1 = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "I'm learning about Python",
            "session_id": session_id
        }
    )
    response1 = await orchestrator.process(msg1)

    # Save to database
    await test_db.save_exchange(
        session_id,
        msg1.content["text"],
        response1.content["text"]
    )

    # Second message (should have context)
    msg2 = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": "Tell me more about functions",
            "session_id": session_id
        }
    )
    response2 = await orchestrator.process(msg2)

    # Should reference previous context
    assert response2.content["text"] is not None

    # Check database
    history = await test_db.get_session_history(session_id)
    assert len(history) == 2

@pytest.mark.asyncio
async def test_error_recovery(orchestrator, mock_anthropic_client):
    """Test error handling and recovery"""
    # Make first agent fail
    mock_anthropic_client.messages.create.side_effect = Exception("API Error")

    message = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "Hello"}
    )

    response = await orchestrator.process(message)

    # Should return error response
    assert response.message_type == MessageType.AGENT_ERROR

    # Reset mock
    mock_anthropic_client.messages.create.side_effect = None

    # Should work now
    response2 = await orchestrator.process(message)
    assert response2.message_type == MessageType.CONVERSATION_RESPONSE
```

---

## End-to-End Testing

### API Endpoint Tests

```python
# tests/e2e/test_api_endpoints.py

import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_v2_conversation_endpoint(client):
    """Test v2 conversation endpoint"""
    response = await client.post("/v2/conversation", json={
        "text": "Hello, I'm learning AI",
        "session_id": "test-session"
    })

    assert response.status_code == 200
    data = response.json()

    assert "session_id" in data
    assert "user_text" in data
    assert "agent_text" in data
    assert "metadata" in data

@pytest.mark.asyncio
async def test_conversation_with_research(client):
    """Test research-enabled conversation"""
    response = await client.post("/v2/conversation", json={
        "text": "What are the latest AI developments?",
        "enable_research": True
    })

    assert response.status_code == 200
    data = response.json()

    # Should include research metadata
    assert "research" in data["metadata"]
    assert len(data["metadata"]["research"]["results"]) > 0

@pytest.mark.asyncio
async def test_multi_turn_conversation(client):
    """Test multi-turn conversation with context"""
    session_id = "multi-turn-session"

    # Turn 1
    response1 = await client.post("/v2/conversation", json={
        "text": "I'm learning Python",
        "session_id": session_id
    })
    assert response1.status_code == 200

    # Turn 2
    response2 = await client.post("/v2/conversation", json={
        "text": "Can you help with functions?",
        "session_id": session_id
    })
    assert response2.status_code == 200

    # Should maintain context
    data2 = response2.json()
    assert "python" in data2["agent_text"].lower() or "function" in data2["agent_text"].lower()
```

---

## Performance Testing

### Load Tests

```python
# tests/performance/test_load.py

import pytest
import asyncio
import time

@pytest.mark.asyncio
@pytest.mark.performance
async def test_concurrent_requests(orchestrator):
    """Test handling concurrent requests"""
    num_requests = 50

    messages = [
        AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": f"Message {i}"}
        )
        for i in range(num_requests)
    ]

    start = time.time()
    responses = await asyncio.gather(
        *[orchestrator.process(msg) for msg in messages]
    )
    duration = time.time() - start

    # All should succeed
    assert len(responses) == num_requests
    assert all(r.message_type == MessageType.CONVERSATION_RESPONSE for r in responses)

    # Should handle in reasonable time (< 10 seconds for 50 requests)
    assert duration < 10.0

    print(f"\nProcessed {num_requests} requests in {duration:.2f}s")
    print(f"Average: {duration/num_requests*1000:.0f}ms per request")

@pytest.mark.performance
def test_response_time_targets():
    """Test response time meets targets"""
    # Target: P99 < 2000ms
    response_times = []

    for i in range(100):
        # Simulate request
        start = time.time()
        # ... actual test logic ...
        duration = (time.time() - start) * 1000  # Convert to ms
        response_times.append(duration)

    # Calculate percentiles
    response_times.sort()
    p50 = response_times[49]
    p95 = response_times[94]
    p99 = response_times[98]

    print(f"\nResponse times:")
    print(f"  P50: {p50:.0f}ms")
    print(f"  P95: {p95:.0f}ms")
    print(f"  P99: {p99:.0f}ms")

    # Assert targets
    assert p50 < 1000, "P50 should be < 1000ms"
    assert p99 < 2000, "P99 should be < 2000ms"
```

### Locust Load Testing

```python
# tests/performance/locustfile.py

from locust import HttpUser, task, between
import random

class ConversationUser(HttpUser):
    """Simulated user for load testing"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Initialize session"""
        self.session_id = f"session-{random.randint(1000, 9999)}"

    @task(3)
    def simple_conversation(self):
        """Simple conversation (most common)"""
        self.client.post("/v2/conversation", json={
            "text": "Tell me about machine learning",
            "session_id": self.session_id
        })

    @task(1)
    def research_conversation(self):
        """Research-enabled conversation"""
        self.client.post("/v2/conversation", json={
            "text": "What are the latest AI developments?",
            "enable_research": True,
            "session_id": self.session_id
        })

    @task(1)
    def search(self):
        """Search past conversations"""
        self.client.post("/api/search", json={
            "query": "neural networks",
            "limit": 10
        })
```

Run with: `locust -f tests/performance/locustfile.py`

---

## Mocking External Services

### Mock Anthropic API

```python
# tests/mocks.py

from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    mock = AsyncMock()

    # Mock message response
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="This is a mocked response.")]
    mock_message.usage = MagicMock(input_tokens=50, output_tokens=30)

    mock.messages.create = AsyncMock(return_value=mock_message)

    return mock
```

### Mock Web Search

```python
@pytest.fixture
def mock_web_search():
    """Mock web search"""
    async def search(query: str, max_results: int = 5):
        return [
            {
                "title": f"Result {i+1} for {query}",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is result {i+1} about {query}",
                "source": "web",
                "relevance_score": 0.9 - (i * 0.1)
            }
            for i in range(min(max_results, 3))
        ]

    return search
```

### Mock ArXiv Search

```python
@pytest.fixture
def mock_arxiv_search():
    """Mock ArXiv search"""
    async def search(query: str, max_results: int = 5):
        return [
            {
                "title": f"Academic Paper {i+1}: {query}",
                "url": f"https://arxiv.org/abs/2024.{i+1:05d}",
                "snippet": f"Abstract for paper {i+1}...",
                "source": "arxiv",
                "authors": ["Author A", "Author B"],
                "published_date": "2024-11-01"
            }
            for i in range(min(max_results, 2))
        ]

    return search
```

---

## Test Coverage

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=app/agents --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html

# Coverage summary
pytest --cov=app/agents --cov-report=term-missing
```

### Coverage Configuration

```ini
# .coveragerc

[run]
source = app/agents
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

### Coverage Targets

```python
# tests/test_coverage.py

import pytest
import coverage

def test_coverage_meets_targets():
    """Verify coverage meets targets"""
    cov = coverage.Coverage()
    cov.load()

    total = cov.report()

    assert total >= 80, f"Overall coverage {total}% is below 80%"

    # Check critical modules
    agent_coverage = cov.analysis('app/agents/base.py')
    assert agent_coverage[0] >= 95, "BaseAgent coverage below 95%"
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test-agents.yml

name: Agent Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run unit tests
        run: pytest tests/agents/test_*.py -v

      - name: Run integration tests
        run: pytest tests/integration/ -v

      - name: Run coverage
        run: |
          pytest --cov=app/agents --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: agents

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=80
```

---

## Troubleshooting

### Common Issues

#### Tests Hanging

**Symptom:** Tests never complete

**Solutions:**
```python
# Add timeout to async tests
@pytest.mark.asyncio
@pytest.mark.timeout(30)  # 30 second timeout
async def test_something():
    ...
```

#### Flaky Tests

**Symptom:** Tests fail intermittently

**Solutions:**
```python
# Use pytest-rerunfailures
@pytest.mark.flaky(reruns=3)
async def test_flaky():
    ...

# Or add explicit retries
for attempt in range(3):
    try:
        result = await flaky_operation()
        break
    except Exception:
        if attempt == 2:
            raise
        await asyncio.sleep(1)
```

#### Mock Not Working

**Symptom:** Real API called instead of mock

**Solutions:**
```python
# Ensure mock is patched before agent initialization
@pytest.fixture
async def agent_with_mock(mock_client):
    agent = MyAgent()
    agent.client = mock_client  # Inject mock
    await agent.initialize()
    return agent

# Or use patch
@patch('app.agents.my_agent.AnthropicClient')
async def test_something(mock_client_class):
    mock_client_class.return_value = mock_client
    ...
```

#### Memory Leaks in Tests

**Symptom:** Memory usage grows during test run

**Solutions:**
```python
# Add cleanup fixtures
@pytest.fixture
async def agent():
    a = MyAgent()
    await a.initialize()
    yield a
    await a.cleanup()  # Always cleanup

# Clear caches between tests
@pytest.fixture(autouse=True)
async def clear_caches():
    yield
    # Clear any global caches
    await cache.clear_all()
```

---

## Best Practices

1. **Isolate tests:** Each test should be independent
2. **Use fixtures:** Reuse setup code with pytest fixtures
3. **Mock external calls:** Don't hit real APIs in tests
4. **Test error paths:** Not just happy paths
5. **Measure coverage:** Aim for 80%+ overall
6. **Performance benchmarks:** Track regression
7. **Readable assertions:** Use descriptive messages
8. **Clean up resources:** Always cleanup in fixtures
9. **Use parametrize:** Test multiple inputs easily
10. **Document tests:** Explain what's being tested

---

## Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/agents/test_conversation_agent.py

# Specific test
pytest tests/agents/test_conversation_agent.py::test_basic_conversation

# With coverage
pytest --cov=app/agents --cov-report=html

# Verbose output
pytest -v

# Show print statements
pytest -s

# Only failed tests
pytest --lf

# Performance tests
pytest -m performance

# Parallel execution
pytest -n 4  # 4 workers
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Next Review:** After Phase 2 implementation
