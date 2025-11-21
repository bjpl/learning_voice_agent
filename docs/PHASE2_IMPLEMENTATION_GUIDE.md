# Phase 2: Multi-Agent System Implementation Guide

**Version:** 2.0.0-alpha
**Status:** Implementation Phase
**Duration:** Weeks 3-4
**Last Updated:** 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Agent Types](#agent-types)
4. [Communication Protocol](#communication-protocol)
5. [Orchestration Patterns](#orchestration-patterns)
6. [Integration with v1.0](#integration-with-v10)
7. [Migration Guide](#migration-guide)
8. [Code Examples](#code-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Goals

Phase 2 transforms the Learning Voice Agent from a single-handler system to a **multi-agent orchestration platform** with:

- **5 specialized agents** working in coordination
- **LangGraph-based orchestration** for intelligent routing
- **Parallel execution** for improved performance
- **Tool integration** for enhanced capabilities
- **Backward compatibility** with v1.0 APIs

### Key Benefits

- **30% faster responses** through parallel processing
- **50% better context understanding** with specialized agents
- **Tool-augmented intelligence** (web search, calculations, etc.)
- **Scalable architecture** for future agent additions
- **Zero downtime migration** from v1.0

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLIENT LAYER                               │
│  Web Browser | Mobile PWA | Phone Call | REST API                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   API GATEWAY (FastAPI)                          │
│  • /v2/conversation  • /v2/search  • /v2/insights               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 ORCHESTRATION LAYER                              │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  AgentOrchestrator (LangGraph Coordinator)             │     │
│  │  • Route requests to appropriate agents                │     │
│  │  • Manage agent lifecycle and state                    │     │
│  │  • Coordinate parallel/sequential execution            │     │
│  │  • Aggregate agent responses                           │     │
│  └────────────────────────────────────────────────────────┘     │
└───────────────┬──────────────┬──────────────┬────────────────────┘
                │              │              │
    ┌───────────▼───┐  ┌──────▼──────┐  ┌────▼─────────┐
    │ Conversation  │  │  Analysis   │  │  Research    │
    │    Agent      │  │   Agent     │  │   Agent      │
    │ (Claude 3.5)  │  │ (Concepts)  │  │  (Tools)     │
    └───────┬───────┘  └──────┬──────┘  └────┬─────────┘
            │                 │               │
    ┌───────▼────────────┐   │   ┌───────────▼──────────┐
    │  Synthesis Agent   │◄──┘   │   Vision Agent       │
    │  (Insights)        │       │   (GPT-4V)           │
    └────────────────────┘       └──────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     SHARED SERVICES                              │
│  • Memory Manager (Redis + Vector DB)                           │
│  • Tool Registry (Web Search, Calculator, Arxiv, etc.)          │
│  • Event Bus (Agent Communication)                              │
│  • Metrics Collector (Performance Tracking)                     │
└──────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. BaseAgent (Abstract Base Class)

All agents inherit from `BaseAgent` which provides:

- Lifecycle management (initialize, process, cleanup)
- Message handling protocol
- State management
- Error handling and resilience
- Metrics collection

#### 2. AgentOrchestrator

Coordinates agent execution with:

- **Routing logic**: Determines which agents to invoke
- **Execution strategies**: Parallel, sequential, or conditional
- **State management**: Maintains conversation context
- **Response aggregation**: Combines multiple agent outputs

#### 3. AgentMessage Protocol

Standardized message format for inter-agent communication:

```python
{
    "id": "msg-uuid",
    "sender": "conversation_agent",
    "recipient": "analysis_agent",
    "message_type": "request|response|event",
    "content": {...},
    "metadata": {...},
    "timestamp": "2025-11-21T10:00:00Z"
}
```

---

## Agent Types

### 1. ConversationAgent

**Purpose:** Primary conversational interface with users

**Model:** Claude 3.5 Sonnet (upgrade from Haiku)

**Responsibilities:**
- Generate natural language responses
- Ask clarifying questions
- Maintain conversation flow
- Handle context and memory

**Tools:**
- Calculator for math
- Date/time utilities
- Basic formatting

**Example Usage:**
```python
agent = ConversationAgent()
response = await agent.process(AgentMessage(
    content={"text": "Tell me about neural networks"},
    message_type="conversation_request"
))
# Response: Natural language explanation with follow-up question
```

### 2. AnalysisAgent

**Purpose:** Extract concepts, entities, and patterns from conversations

**Capabilities:**
- Named entity recognition
- Topic extraction
- Sentiment analysis
- Relationship mapping

**Output:**
```python
{
    "concepts": ["neural networks", "machine learning", "AI"],
    "entities": {"TECH": ["PyTorch", "TensorFlow"]},
    "sentiment": {"score": 0.8, "label": "enthusiastic"},
    "relationships": [
        {"source": "neural networks", "relation": "is_part_of", "target": "machine learning"}
    ]
}
```

### 3. ResearchAgent

**Purpose:** Retrieve external knowledge and perform searches

**Tools:**
- Web search (DuckDuckGo, Brave)
- ArXiv paper search
- Wikipedia lookup
- Documentation search

**Example:**
```python
agent = ResearchAgent()
result = await agent.process(AgentMessage(
    content={"query": "latest transformer architectures", "sources": ["arxiv", "web"]},
    message_type="research_request"
))
# Returns: Curated research results with citations
```

### 4. SynthesisAgent

**Purpose:** Generate insights, summaries, and connections

**Capabilities:**
- Session summarization
- Pattern detection across conversations
- Insight generation
- Learning recommendations

**Example Output:**
```python
{
    "insights": [
        "You've discussed ML fundamentals 5 times this week",
        "Interest shifting from theory to implementation"
    ],
    "patterns": ["evening learning sessions", "prefers practical examples"],
    "recommendations": ["Review gradient descent", "Try building a simple NN"]
}
```

### 5. VisionAgent

**Purpose:** Analyze images and documents (Phase 4, prepared in Phase 2)

**Model:** GPT-4V

**Capabilities:**
- Image understanding
- Document OCR and analysis
- Diagram interpretation
- Chart/graph extraction

---

## Communication Protocol

### AgentMessage Structure

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class MessageType(str, Enum):
    # Requests
    CONVERSATION_REQUEST = "conversation_request"
    ANALYSIS_REQUEST = "analysis_request"
    RESEARCH_REQUEST = "research_request"
    SYNTHESIS_REQUEST = "synthesis_request"

    # Responses
    CONVERSATION_RESPONSE = "conversation_response"
    ANALYSIS_RESPONSE = "analysis_response"
    RESEARCH_RESPONSE = "research_response"
    SYNTHESIS_RESPONSE = "synthesis_response"

    # Events
    AGENT_READY = "agent_ready"
    AGENT_ERROR = "agent_error"
    CONTEXT_UPDATE = "context_update"

class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg-{uuid.uuid4()}")
    sender: str = Field(..., description="Agent ID or 'user'")
    recipient: str = Field(..., description="Target agent ID or 'orchestrator'")
    message_type: MessageType
    content: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For request-response tracking
```

### Message Flow Examples

#### Single Agent Request
```
User → Orchestrator → ConversationAgent → Orchestrator → User
```

#### Multi-Agent Parallel Request
```
User → Orchestrator ┬→ ConversationAgent ┐
                    └→ AnalysisAgent     ┴→ Orchestrator → User
```

#### Sequential Agent Chain
```
User → Orchestrator → ResearchAgent → SynthesisAgent → ConversationAgent → Orchestrator → User
```

---

## Orchestration Patterns

### 1. Simple Routing (Default)

For basic conversation:

```python
async def route_simple(message: AgentMessage) -> AgentMessage:
    """Route to ConversationAgent only"""
    agent = self.agents["conversation"]
    response = await agent.process(message)
    return response
```

### 2. Parallel Execution

For multi-faceted queries:

```python
async def route_parallel(message: AgentMessage) -> AgentMessage:
    """Execute multiple agents in parallel"""
    tasks = [
        self.agents["conversation"].process(message),
        self.agents["analysis"].process(message),
    ]

    results = await asyncio.gather(*tasks)

    # Aggregate responses
    return self._aggregate_responses(results)
```

### 3. Sequential Pipeline

For research-heavy queries:

```python
async def route_pipeline(message: AgentMessage) -> AgentMessage:
    """Execute agents in sequence"""
    # Step 1: Research
    research_result = await self.agents["research"].process(message)

    # Step 2: Synthesize findings
    synthesis_msg = AgentMessage(
        content=research_result.content,
        message_type=MessageType.SYNTHESIS_REQUEST
    )
    synthesis_result = await self.agents["synthesis"].process(synthesis_msg)

    # Step 3: Generate response
    conversation_msg = AgentMessage(
        content={
            "context": synthesis_result.content,
            "original_query": message.content
        },
        message_type=MessageType.CONVERSATION_REQUEST
    )
    return await self.agents["conversation"].process(conversation_msg)
```

### 4. Conditional Routing

Based on message analysis:

```python
async def route_conditional(message: AgentMessage) -> AgentMessage:
    """Route based on message content"""
    text = message.content.get("text", "")

    # Check for research keywords
    if any(kw in text.lower() for kw in ["research", "papers", "latest", "arxiv"]):
        return await self.route_pipeline(message)

    # Check for analysis keywords
    elif any(kw in text.lower() for kw in ["analyze", "patterns", "insights"]):
        return await self.route_parallel(message)

    # Default to simple conversation
    else:
        return await self.route_simple(message)
```

---

## Integration with v1.0

### Backward Compatibility

Phase 2 maintains full backward compatibility:

```python
# v1.0 endpoint (unchanged)
@app.post("/api/conversation")
async def conversation_v1(request: ConversationRequest):
    # Uses legacy ConversationHandler
    return await legacy_handler.process(request)

# v2.0 endpoint (new multi-agent)
@app.post("/v2/conversation")
async def conversation_v2(request: ConversationRequest):
    # Uses AgentOrchestrator
    message = AgentMessage(
        content={"text": request.text},
        message_type=MessageType.CONVERSATION_REQUEST
    )
    response = await orchestrator.process(message)
    return response
```

### Feature Flags

Control rollout with feature flags:

```python
class Settings:
    enable_v2_agents: bool = env.bool("ENABLE_V2_AGENTS", default=False)
    v2_traffic_percentage: int = env.int("V2_TRAFFIC_PERCENT", default=0)

    # Agent-specific flags
    enable_research_agent: bool = env.bool("ENABLE_RESEARCH_AGENT", default=True)
    enable_synthesis_agent: bool = env.bool("ENABLE_SYNTHESIS_AGENT", default=False)

# In endpoint
@app.post("/api/conversation")
async def conversation(request: ConversationRequest):
    # Gradual rollout logic
    if settings.enable_v2_agents and random.randint(1, 100) <= settings.v2_traffic_percentage:
        return await conversation_v2(request)
    else:
        return await conversation_v1(request)
```

### Shared Resources

Both v1 and v2 share:

- **Database:** Same SQLite + FTS5 (v2 adds metadata)
- **Redis:** Same session management
- **Audio Pipeline:** Same Whisper integration
- **Metrics:** Unified metrics collection

---

## Migration Guide

### Step 1: Install Dependencies

```bash
# Add to requirements.txt
langgraph>=0.0.30
langchain>=0.1.0
langchain-anthropic>=0.1.0
anthropic>=0.18.1  # Already installed
```

### Step 2: Create Agent Base Classes

```python
# app/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, agent_id: str, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.state = {}

    async def initialize(self) -> None:
        """Initialize agent resources"""
        pass

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message and return response"""
        pass

    async def cleanup(self) -> None:
        """Clean up agent resources"""
        pass

    def _create_response(
        self,
        content: Dict[str, Any],
        message_type: MessageType,
        original_message: AgentMessage
    ) -> AgentMessage:
        """Helper to create response messages"""
        return AgentMessage(
            sender=self.agent_id,
            recipient=original_message.sender,
            message_type=message_type,
            content=content,
            correlation_id=original_message.id
        )
```

### Step 3: Implement ConversationAgent

```python
# app/agents/conversation_agent.py
from app.agents.base import BaseAgent, AgentMessage, MessageType
import anthropic

class ConversationAgent(BaseAgent):
    """Handles conversational interactions"""

    def __init__(self):
        super().__init__("conversation_agent")
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Upgrade from Haiku

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Generate conversational response"""
        user_text = message.content.get("text", "")
        context = message.content.get("context", [])

        # Format context for Claude
        messages = self._format_context(context)
        messages.append({"role": "user", "content": user_text})

        # Call Claude API
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self._get_system_prompt(),
            messages=messages
        )

        agent_text = response.content[0].text

        return self._create_response(
            content={"text": agent_text, "model": self.model},
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=message
        )
```

### Step 4: Create Orchestrator

```python
# app/agents/orchestrator.py
from app.agents.base import BaseAgent, AgentMessage
from app.agents.conversation_agent import ConversationAgent
from app.agents.analysis_agent import AnalysisAgent
# ... import other agents

class AgentOrchestrator:
    """Coordinates multiple agents"""

    def __init__(self):
        self.agents = {
            "conversation": ConversationAgent(),
            "analysis": AnalysisAgent(),
            # Add more agents as implemented
        }

    async def initialize(self):
        """Initialize all agents"""
        for agent in self.agents.values():
            await agent.initialize()

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Route message to appropriate agent(s)"""
        # Determine routing strategy
        strategy = self._select_strategy(message)

        # Execute strategy
        if strategy == "simple":
            return await self._route_simple(message)
        elif strategy == "parallel":
            return await self._route_parallel(message)
        elif strategy == "pipeline":
            return await self._route_pipeline(message)
```

### Step 5: Update API Endpoints

```python
# app/main.py
from app.agents.orchestrator import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator()

@app.on_event("startup")
async def startup():
    await orchestrator.initialize()

@app.post("/v2/conversation")
async def conversation_v2(request: ConversationRequest):
    """Multi-agent conversation endpoint"""
    message = AgentMessage(
        sender="user",
        recipient="orchestrator",
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": request.text,
            "session_id": request.session_id
        }
    )

    response = await orchestrator.process(message)

    return ConversationResponse(
        session_id=request.session_id,
        user_text=request.text,
        agent_text=response.content.get("text"),
        metadata=response.metadata
    )
```

---

## Code Examples

### Example 1: Basic Multi-Agent Conversation

```python
import asyncio
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base import AgentMessage, MessageType

async def main():
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()

    # Create message
    message = AgentMessage(
        sender="user",
        recipient="orchestrator",
        message_type=MessageType.CONVERSATION_REQUEST,
        content={"text": "Explain transformers in machine learning"}
    )

    # Process
    response = await orchestrator.process(message)

    print(f"User: {message.content['text']}")
    print(f"Agent: {response.content['text']}")
    print(f"Agents used: {response.metadata.get('agents_used')}")

asyncio.run(main())
```

### Example 2: Research-Augmented Response

```python
async def research_augmented_conversation(query: str):
    """Demonstrate research agent integration"""

    # Step 1: Research
    research_msg = AgentMessage(
        message_type=MessageType.RESEARCH_REQUEST,
        content={"query": query, "sources": ["arxiv", "web"]}
    )
    research_results = await orchestrator.agents["research"].process(research_msg)

    # Step 2: Conversation with research context
    conversation_msg = AgentMessage(
        message_type=MessageType.CONVERSATION_REQUEST,
        content={
            "text": query,
            "research_context": research_results.content.get("results")
        }
    )
    response = await orchestrator.agents["conversation"].process(conversation_msg)

    return response
```

### Example 3: Pattern Analysis

```python
async def analyze_learning_patterns(session_id: str):
    """Analyze user's learning patterns"""

    # Get session history
    history = await db.get_session_history(session_id, limit=50)

    # Request analysis
    analysis_msg = AgentMessage(
        message_type=MessageType.ANALYSIS_REQUEST,
        content={
            "conversations": history,
            "analysis_type": "patterns"
        }
    )

    analysis = await orchestrator.agents["analysis"].process(analysis_msg)

    # Generate insights
    synthesis_msg = AgentMessage(
        message_type=MessageType.SYNTHESIS_REQUEST,
        content={
            "analysis": analysis.content,
            "session_id": session_id
        }
    )

    insights = await orchestrator.agents["synthesis"].process(synthesis_msg)

    return insights.content
```

---

## Best Practices

### 1. Agent Design

- **Single Responsibility:** Each agent has one clear purpose
- **Stateless Execution:** Agents should be stateless; state lives in orchestrator
- **Idempotency:** Same input should produce same output
- **Error Handling:** Graceful degradation with fallback responses

### 2. Message Handling

- **Always validate messages:** Use Pydantic models
- **Include correlation IDs:** For request-response tracking
- **Add metadata:** Execution time, model used, tokens, etc.
- **Log all messages:** For debugging and analysis

### 3. Performance

- **Use async/await:** All I/O operations must be async
- **Parallel when possible:** Run independent agents in parallel
- **Cache aggressively:** Cache LLM responses, embeddings, etc.
- **Set timeouts:** Prevent hanging requests

### 4. Observability

- **Instrument everything:** Add metrics for all agent calls
- **Structured logging:** Use JSON logs for easy parsing
- **Trace requests:** OpenTelemetry for distributed tracing
- **Monitor costs:** Track API usage and costs per agent

---

## Troubleshooting

### Agent Not Responding

**Symptom:** Agent process() hangs indefinitely

**Solutions:**
1. Check timeout settings: `asyncio.wait_for(agent.process(msg), timeout=30)`
2. Verify API credentials are set
3. Check network connectivity
4. Review agent logs for errors

### Message Routing Errors

**Symptom:** Messages not reaching intended agent

**Solutions:**
1. Validate message_type matches agent expectations
2. Check recipient field is correct agent_id
3. Verify agent is initialized: `await orchestrator.initialize()`
4. Review orchestrator routing logic

### Performance Degradation

**Symptom:** Responses slower than v1.0

**Solutions:**
1. Enable parallel execution for independent agents
2. Add Redis caching for repeated queries
3. Use smaller models for simple tasks (Haiku for analysis)
4. Profile agent execution times

### Memory Leaks

**Symptom:** Memory usage grows over time

**Solutions:**
1. Call `agent.cleanup()` on shutdown
2. Clear conversation context periodically
3. Limit context window size
4. Use connection pooling for HTTP clients

---

## Next Steps

After implementing Phase 2:

1. **Phase 3:** Add vector memory and semantic search
2. **Phase 4:** Integrate vision agent for multi-modal
3. **Phase 5:** Implement real-time learning
4. **Load Testing:** Verify performance targets
5. **Production Rollout:** Gradual traffic migration

---

## Resources

- [Agent API Reference](./AGENT_API_REFERENCE.md)
- [Testing Guide](./PHASE2_TESTING_GUIDE.md)
- [Architecture v1.0](./ARCHITECTURE_V1.md)
- [Migration Plan](./MIGRATION_PLAN.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Next Review:** After Phase 2 completion
