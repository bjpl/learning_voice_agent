# Agent API Reference

**Version:** 2.0.0-alpha
**Last Updated:** 2025-11-21

---

## Table of Contents

1. [BaseAgent](#baseagent)
2. [ConversationAgent](#conversationagent)
3. [AnalysisAgent](#analysisagent)
4. [ResearchAgent](#researchagent)
5. [SynthesisAgent](#synthesisagent)
6. [VisionAgent](#visionagent)
7. [AgentOrchestrator](#agentorchestrator)
8. [AgentMessage](#agentmessage)
9. [Tool Integration](#tool-integration)
10. [Error Handling](#error-handling)

---

## BaseAgent

Abstract base class that all agents inherit from.

### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.

    Provides lifecycle management, message handling, and common utilities.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier for this agent
            config: Optional configuration dictionary
        """
```

### Methods

#### `async def initialize() -> None`

Initialize agent resources (API clients, models, etc.).

**Example:**
```python
agent = ConversationAgent()
await agent.initialize()
```

#### `async def process(message: AgentMessage) -> AgentMessage` (abstract)

Process incoming message and return response. Must be implemented by subclasses.

**Args:**
- `message` (AgentMessage): Incoming message to process

**Returns:**
- `AgentMessage`: Response message

**Raises:**
- `AgentError`: If processing fails

**Example:**
```python
request = AgentMessage(
    message_type=MessageType.CONVERSATION_REQUEST,
    content={"text": "Hello"}
)
response = await agent.process(request)
```

#### `async def cleanup() -> None`

Clean up agent resources (close connections, clear cache, etc.).

**Example:**
```python
await agent.cleanup()
```

#### `_create_response(content: Dict, message_type: MessageType, original: AgentMessage) -> AgentMessage`

Helper method to create response messages with proper metadata.

**Args:**
- `content`: Response content dictionary
- `message_type`: Type of response message
- `original`: Original request message

**Returns:**
- `AgentMessage`: Formatted response

---

## ConversationAgent

Handles natural language conversations with users.

### Class Definition

```python
from app.agents.base import BaseAgent, AgentMessage, MessageType
import anthropic

class ConversationAgent(BaseAgent):
    """
    Primary conversational agent using Claude 3.5 Sonnet.

    Handles:
    - Natural language understanding and generation
    - Context-aware responses
    - Clarifying questions
    - Conversation flow management
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize ConversationAgent.

        Args:
            model: Claude model to use (default: claude-3-5-sonnet)
        """
```

### Configuration

```python
config = {
    "model": "claude-3-5-sonnet-20241022",  # or "claude-3-haiku-20240307"
    "max_tokens": 300,
    "temperature": 0.7,
    "system_prompt": "...",  # Custom system prompt
    "tools": ["calculator", "datetime"],  # Available tools
}

agent = ConversationAgent(config=config)
```

### Methods

#### `async def process(message: AgentMessage) -> AgentMessage`

Generate conversational response.

**Message Content Schema:**
```python
{
    "text": str,                    # Required: User's message
    "context": List[Dict],          # Optional: Previous exchanges
    "session_id": str,              # Optional: Session identifier
    "enable_tools": bool,           # Optional: Enable tool use (default: True)
    "metadata": Dict[str, Any]      # Optional: Additional context
}
```

**Response Content Schema:**
```python
{
    "text": str,                    # Agent's response
    "model": str,                   # Model used
    "tokens_used": int,             # Token count
    "tools_used": List[str],        # Tools invoked (if any)
    "confidence": float,            # Response confidence (0-1)
    "metadata": Dict[str, Any]      # Additional metadata
}
```

**Example:**
```python
message = AgentMessage(
    message_type=MessageType.CONVERSATION_REQUEST,
    content={
        "text": "Explain gradient descent",
        "context": [
            {"user": "I'm learning ML", "agent": "That's great! What interests you?"}
        ],
        "enable_tools": True
    }
)

response = await conversation_agent.process(message)

print(response.content["text"])
# "Gradient descent is an optimization algorithm..."
print(response.content["tokens_used"])
# 245
```

#### `_format_context(exchanges: List[Dict]) -> str`

Format conversation history for Claude.

**Args:**
- `exchanges`: List of {"user": "...", "agent": "..."} dicts

**Returns:**
- Formatted context string

#### `_should_use_tool(user_text: str) -> List[str]`

Determine which tools to enable for this query.

**Args:**
- `user_text`: User's message

**Returns:**
- List of tool names to enable

---

## AnalysisAgent

Extracts concepts, entities, and patterns from conversations.

### Class Definition

```python
class AnalysisAgent(BaseAgent):
    """
    Analyzes conversations to extract structured information.

    Capabilities:
    - Named entity recognition
    - Concept extraction
    - Topic modeling
    - Sentiment analysis
    - Relationship mapping
    """
```

### Configuration

```python
config = {
    "extract_entities": True,       # Enable NER
    "extract_concepts": True,        # Enable concept extraction
    "analyze_sentiment": True,       # Enable sentiment analysis
    "min_confidence": 0.7,          # Minimum confidence threshold
    "max_concepts": 10              # Max concepts to extract
}
```

### Methods

#### `async def process(message: AgentMessage) -> AgentMessage`

Analyze text and extract structured information.

**Message Content Schema:**
```python
{
    "text": str,                    # Required: Text to analyze
    "analysis_types": List[str],    # Optional: ["entities", "concepts", "sentiment"]
    "context": List[Dict],          # Optional: Previous exchanges for context
}
```

**Response Content Schema:**
```python
{
    "concepts": List[str],          # Extracted concepts
    "entities": Dict[str, List],    # {"PERSON": ["Alice"], "ORG": ["OpenAI"]}
    "sentiment": {
        "score": float,             # -1 to 1
        "label": str                # "positive|negative|neutral"
    },
    "topics": List[str],            # Identified topics
    "relationships": List[Dict],    # [{"source": "A", "relation": "is_a", "target": "B"}]
    "metadata": {
        "confidence": float,
        "processing_time_ms": int
    }
}
```

**Example:**
```python
message = AgentMessage(
    message_type=MessageType.ANALYSIS_REQUEST,
    content={
        "text": "I'm learning PyTorch to build neural networks for computer vision",
        "analysis_types": ["entities", "concepts", "sentiment"]
    }
)

response = await analysis_agent.process(message)

print(response.content["concepts"])
# ["PyTorch", "neural networks", "computer vision", "deep learning"]

print(response.content["entities"])
# {"TECH": ["PyTorch"], "FIELD": ["computer vision"]}

print(response.content["sentiment"])
# {"score": 0.8, "label": "enthusiastic"}
```

#### `extract_concepts(text: str) -> List[str]`

Extract key concepts from text.

#### `extract_entities(text: str) -> Dict[str, List[str]]`

Perform named entity recognition.

#### `analyze_sentiment(text: str) -> Dict[str, Any]`

Analyze sentiment of text.

---

## ResearchAgent

Retrieves external knowledge using various search tools.

### Class Definition

```python
class ResearchAgent(BaseAgent):
    """
    Performs research using external knowledge sources.

    Tools:
    - Web search (DuckDuckGo, Brave)
    - ArXiv paper search
    - Wikipedia lookup
    - Documentation search
    """
```

### Configuration

```python
config = {
    "default_sources": ["web", "arxiv"],  # Default sources to search
    "max_results": 5,                      # Max results per source
    "timeout_seconds": 10,                 # Search timeout
    "cache_ttl": 3600,                     # Cache results for 1 hour
}
```

### Methods

#### `async def process(message: AgentMessage) -> AgentMessage`

Perform research and return results.

**Message Content Schema:**
```python
{
    "query": str,                   # Required: Search query
    "sources": List[str],           # Optional: ["web", "arxiv", "wikipedia"]
    "max_results": int,             # Optional: Max results (default: 5)
    "filters": Dict[str, Any],      # Optional: Source-specific filters
}
```

**Response Content Schema:**
```python
{
    "query": str,                   # Original query
    "results": List[Dict],          # Search results
    "sources_used": List[str],      # Sources that were searched
    "metadata": {
        "total_results": int,
        "search_time_ms": int,
        "cached": bool
    }
}

# Result format:
{
    "title": str,
    "url": str,
    "snippet": str,
    "source": str,                  # "web", "arxiv", etc.
    "relevance_score": float,       # 0-1
    "published_date": Optional[str],
    "authors": Optional[List[str]]   # For papers
}
```

**Example:**
```python
message = AgentMessage(
    message_type=MessageType.RESEARCH_REQUEST,
    content={
        "query": "attention mechanism in transformers",
        "sources": ["arxiv", "web"],
        "max_results": 3
    }
)

response = await research_agent.process(message)

for result in response.content["results"]:
    print(f"{result['title']}: {result['url']}")
    print(f"  {result['snippet']}\n")

# Output:
# Attention Is All You Need: https://arxiv.org/abs/1706.03762
#   We propose a new simple network architecture, the Transformer...
```

#### `search_web(query: str, max_results: int) -> List[Dict]`

Search the web using DuckDuckGo.

#### `search_arxiv(query: str, max_results: int) -> List[Dict]`

Search academic papers on ArXiv.

#### `search_wikipedia(query: str) -> Dict`

Look up Wikipedia article.

---

## SynthesisAgent

Generates insights, summaries, and recommendations.

### Class Definition

```python
class SynthesisAgent(BaseAgent):
    """
    Synthesizes information to generate insights and recommendations.

    Capabilities:
    - Session summarization
    - Pattern detection
    - Learning recommendations
    - Insight generation
    """
```

### Configuration

```python
config = {
    "summarization_model": "claude-3-haiku-20240307",  # Faster for summaries
    "min_session_length": 5,        # Min exchanges for patterns
    "insight_threshold": 0.7,       # Min confidence for insights
}
```

### Methods

#### `async def process(message: AgentMessage) -> AgentMessage`

Generate synthesis and insights.

**Message Content Schema:**
```python
{
    "synthesis_type": str,          # "summary", "insights", "recommendations"
    "conversations": List[Dict],    # Conversation history
    "analysis": Optional[Dict],     # Output from AnalysisAgent
    "timeframe": Optional[str],     # "session", "day", "week"
}
```

**Response Content Schema:**
```python
{
    "synthesis_type": str,
    "summary": Optional[str],       # If type="summary"
    "insights": List[Dict],         # If type="insights"
    "recommendations": List[Dict],  # If type="recommendations"
    "patterns": List[Dict],         # Detected patterns
    "metadata": {
        "conversations_analyzed": int,
        "confidence": float
    }
}

# Insight format:
{
    "text": str,                    # Human-readable insight
    "category": str,                # "learning_style", "topic_interest", etc.
    "confidence": float,
    "evidence": List[str]           # Supporting evidence
}

# Recommendation format:
{
    "text": str,                    # Human-readable recommendation
    "type": str,                    # "review", "practice", "explore"
    "priority": int,                # 1-5
    "related_topics": List[str]
}
```

**Example:**
```python
message = AgentMessage(
    message_type=MessageType.SYNTHESIS_REQUEST,
    content={
        "synthesis_type": "insights",
        "conversations": session_history,  # Last 50 exchanges
        "timeframe": "week"
    }
)

response = await synthesis_agent.process(message)

for insight in response.content["insights"]:
    print(f"[{insight['category']}] {insight['text']}")
    print(f"  Confidence: {insight['confidence']:.0%}\n")

# Output:
# [learning_style] You prefer hands-on examples over theoretical explanations
#   Confidence: 87%
#
# [topic_interest] Focus shifting from ML basics to advanced architectures
#   Confidence: 92%
```

#### `summarize_session(conversations: List[Dict]) -> str`

Generate session summary.

#### `detect_patterns(conversations: List[Dict]) -> List[Dict]`

Detect learning patterns.

#### `generate_recommendations(insights: List[Dict]) -> List[Dict]`

Generate personalized recommendations.

---

## VisionAgent

Analyzes images and documents (Phase 4 feature, API prepared).

### Class Definition

```python
class VisionAgent(BaseAgent):
    """
    Analyzes images and documents using GPT-4V.

    Capabilities:
    - Image understanding and description
    - Document OCR and analysis
    - Diagram interpretation
    - Chart/graph data extraction
    """
```

### Methods

#### `async def process(message: AgentMessage) -> AgentMessage`

Analyze image or document.

**Message Content Schema:**
```python
{
    "image_url": Optional[str],     # URL to image
    "image_base64": Optional[str],  # Base64 encoded image
    "analysis_type": str,           # "describe", "ocr", "extract_data"
    "prompt": Optional[str]         # Custom analysis prompt
}
```

**Response Content Schema:**
```python
{
    "description": str,             # Image description
    "text": Optional[str],          # Extracted text (if OCR)
    "structured_data": Optional[Dict],  # Extracted data (if chart/table)
    "objects": List[str],           # Detected objects
    "metadata": {
        "model": str,
        "confidence": float
    }
}
```

---

## AgentOrchestrator

Coordinates multiple agents and manages execution flow.

### Class Definition

```python
class AgentOrchestrator:
    """
    Orchestrates multiple agents to handle complex queries.

    Responsibilities:
    - Route messages to appropriate agents
    - Manage agent lifecycle
    - Execute parallel/sequential workflows
    - Aggregate responses
    """
```

### Methods

#### `async def initialize() -> None`

Initialize all registered agents.

**Example:**
```python
orchestrator = AgentOrchestrator()
await orchestrator.initialize()
```

#### `async def process(message: AgentMessage) -> AgentMessage`

Process message using appropriate agent(s).

**Args:**
- `message`: Incoming message

**Returns:**
- `AgentMessage`: Aggregated response

**Example:**
```python
message = AgentMessage(
    sender="user",
    recipient="orchestrator",
    message_type=MessageType.CONVERSATION_REQUEST,
    content={"text": "Explain transformers with recent research"}
)

response = await orchestrator.process(message)

# Orchestrator automatically:
# 1. Routes to ResearchAgent for papers
# 2. Routes to ConversationAgent with research context
# 3. Aggregates and returns response
```

#### `register_agent(agent: BaseAgent) -> None`

Register a new agent with the orchestrator.

**Example:**
```python
custom_agent = CustomAgent("my_agent")
orchestrator.register_agent(custom_agent)
```

#### `async def cleanup() -> None`

Clean up all agents.

---

## AgentMessage

Standard message format for inter-agent communication.

### Class Definition

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class MessageType(str, Enum):
    """Message types for agent communication"""
    CONVERSATION_REQUEST = "conversation_request"
    CONVERSATION_RESPONSE = "conversation_response"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    RESEARCH_REQUEST = "research_request"
    RESEARCH_RESPONSE = "research_response"
    SYNTHESIS_REQUEST = "synthesis_request"
    SYNTHESIS_RESPONSE = "synthesis_response"
    VISION_REQUEST = "vision_request"
    VISION_RESPONSE = "vision_response"
    AGENT_ERROR = "agent_error"

class AgentMessage(BaseModel):
    """Standard message format for agent communication"""

    id: str = Field(default_factory=lambda: f"msg-{uuid.uuid4()}")
    sender: str = Field(..., description="Agent ID or 'user'")
    recipient: str = Field(..., description="Target agent ID or 'orchestrator'")
    message_type: MessageType
    content: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
```

### Fields

- **id**: Unique message identifier (auto-generated)
- **sender**: Agent ID that sent the message (or "user")
- **recipient**: Target agent ID (or "orchestrator")
- **message_type**: Type of message (request/response/error)
- **content**: Message payload (schema varies by type)
- **metadata**: Additional context (execution time, model used, etc.)
- **timestamp**: Message creation time (UTC)
- **correlation_id**: Links responses to requests (optional)

### Example

```python
message = AgentMessage(
    sender="user",
    recipient="conversation_agent",
    message_type=MessageType.CONVERSATION_REQUEST,
    content={
        "text": "Hello world",
        "session_id": "sess-123"
    },
    metadata={
        "source": "web_ui",
        "user_agent": "Mozilla/5.0..."
    }
)
```

---

## Tool Integration

### Available Tools

#### Calculator
```python
# Used automatically by ConversationAgent
"What is 25 * 87 + 143?"
# Agent uses calculator tool, returns: "2,318"
```

#### DateTime
```python
# Used automatically by ConversationAgent
"What day was it 45 days ago?"
# Agent uses datetime tool, returns: "October 7, 2025"
```

#### Web Search
```python
# Used by ResearchAgent
await research_agent.search_web("latest AI developments")
```

#### ArXiv Search
```python
# Used by ResearchAgent
await research_agent.search_arxiv("attention mechanisms")
```

### Adding Custom Tools

```python
from app.agents.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Does something custom"

    async def execute(self, **kwargs) -> Dict[str, Any]:
        # Implementation
        return {"result": "..."}

# Register with agent
conversation_agent.register_tool(CustomTool())
```

---

## Error Handling

### AgentError

Base exception for all agent errors.

```python
from app.agents.base import AgentError

class AgentError(Exception):
    """Base exception for agent errors"""

    def __init__(
        self,
        message: str,
        agent_id: str,
        recoverable: bool = False,
        metadata: Optional[Dict] = None
    ):
        self.message = message
        self.agent_id = agent_id
        self.recoverable = recoverable
        self.metadata = metadata or {}
```

### Common Errors

#### APIError
```python
try:
    response = await agent.process(message)
except APIError as e:
    print(f"API call failed: {e.message}")
    # Fallback logic
```

#### TimeoutError
```python
try:
    response = await asyncio.wait_for(
        agent.process(message),
        timeout=30.0
    )
except asyncio.TimeoutError:
    print("Agent timed out")
    # Use default response
```

#### ValidationError
```python
try:
    message = AgentMessage(**invalid_data)
except ValidationError as e:
    print(f"Invalid message: {e}")
    # Fix and retry
```

### Error Response Format

When an agent encounters an error:

```python
AgentMessage(
    sender=agent.agent_id,
    recipient=original_message.sender,
    message_type=MessageType.AGENT_ERROR,
    content={
        "error_type": "APIError",
        "error_message": "API rate limit exceeded",
        "recoverable": True,
        "retry_after": 60
    },
    metadata={
        "original_message_id": original_message.id,
        "timestamp": datetime.utcnow()
    }
)
```

---

## Code Examples

### Complete Workflow Example

```python
import asyncio
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base import AgentMessage, MessageType

async def main():
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()

    try:
        # Create conversation request
        message = AgentMessage(
            sender="user",
            recipient="orchestrator",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Explain transformers with recent research",
                "enable_research": True,
                "session_id": "demo-session"
            }
        )

        # Process message
        response = await orchestrator.process(message)

        # Display results
        print(f"Response: {response.content['text']}")
        print(f"\nMetadata:")
        print(f"  Agents used: {response.metadata.get('agents_used')}")
        print(f"  Processing time: {response.metadata.get('processing_time_ms')}ms")
        print(f"  Tokens used: {response.metadata.get('tokens_used')}")

    finally:
        # Cleanup
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Versioning

All agents follow semantic versioning:

- **Major**: Breaking changes to agent interface
- **Minor**: New capabilities, backward compatible
- **Patch**: Bug fixes

Current versions:
- BaseAgent: 2.0.0
- ConversationAgent: 2.0.0
- AnalysisAgent: 2.0.0
- ResearchAgent: 2.0.0
- SynthesisAgent: 2.0.0
- VisionAgent: 2.0.0-alpha

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Next Review:** After Phase 2 implementation
