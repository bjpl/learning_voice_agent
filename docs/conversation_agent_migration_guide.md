# ConversationAgent Migration Guide

## Overview

This guide helps you migrate from the existing `conversation_handler.py` (Claude Haiku) to the new `ConversationAgent` (Claude 3.5 Sonnet) with tool calling support.

## Key Improvements

### Model Upgrade
- **Old**: Claude 3 Haiku (`claude-3-haiku-20240307`)
- **New**: Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)
- **Benefits**: Superior reasoning, better context understanding, more nuanced responses

### Tool Calling
The new agent supports 4 built-in tools:
1. **search_knowledge**: Search past conversations
2. **calculate**: Mathematical operations
3. **get_datetime**: Current time/date information
4. **memory_store**: Store and retrieve user facts

### Streaming Support
- Optional streaming responses for better UX
- Progressive content delivery
- Improved perceived latency

### Enhanced Intelligence
- Intent classification
- Entity extraction
- Context-aware responses
- Multi-turn dialogue management

## Migration Paths

### Path 1: Gradual Migration (Recommended)

Run both agents side-by-side with a feature flag:

```python
# app/main.py
from app.conversation_handler import conversation_handler  # v1
from app.agents.conversation_agent import conversation_agent_v2  # v2
from app.config import settings

# Feature flag
USE_V2_AGENT = os.getenv("USE_CONVERSATION_V2", "false").lower() == "true"

@app.post("/conversation")
async def handle_conversation(request: ConversationRequest):
    if USE_V2_AGENT:
        # Use new ConversationAgent
        message = AgentMessage(
            role=MessageRole.USER,
            content=request.text,
            message_type=MessageType.TEXT,
            context={
                "session_id": request.session_id,
                "conversation_history": request.history,
            }
        )
        response = await conversation_agent_v2.process(message)
        return {"response": response.content}
    else:
        # Use old conversation_handler
        response = await conversation_handler.generate_response(
            user_text=request.text,
            context=request.history,
            session_metadata={"session_id": request.session_id}
        )
        return {"response": response}
```

### Path 2: Direct Migration

Replace conversation_handler entirely:

```python
# Before (v1)
from app.conversation_handler import conversation_handler

response = await conversation_handler.generate_response(
    user_text="What is 2 + 2?",
    context=[{"user": "Hello", "agent": "Hi!"}],
    session_metadata={"session_id": "123"}
)

# After (v2)
from app.agents.conversation_agent import conversation_agent_v2
from app.agents.base import AgentMessage, MessageRole, MessageType

message = AgentMessage(
    role=MessageRole.USER,
    content="What is 2 + 2?",
    message_type=MessageType.TEXT,
    context={
        "session_id": "123",
        "conversation_history": [{"user": "Hello", "agent": "Hi!"}],
    }
)

response_message = await conversation_agent_v2.process(message)
response = response_message.content
```

## API Compatibility Layer

Create a compatibility wrapper for seamless migration:

```python
# app/agents/compatibility.py
from typing import List, Dict, Optional
from app.agents.conversation_agent import conversation_agent_v2
from app.agents.base import AgentMessage, MessageRole, MessageType

class ConversationHandlerV2:
    """
    Backward-compatible wrapper for ConversationAgent
    Provides same interface as old conversation_handler
    """

    def __init__(self):
        self.agent = conversation_agent_v2

    async def generate_response(
        self,
        user_text: str,
        context: List[Dict],
        session_metadata: Optional[Dict] = None
    ) -> str:
        """
        Drop-in replacement for old generate_response method
        """
        message = AgentMessage(
            role=MessageRole.USER,
            content=user_text,
            message_type=MessageType.TEXT,
            context={
                "conversation_history": context,
                **(session_metadata or {})
            }
        )

        response_message = await self.agent.process(message)
        return response_message.content

    def detect_intent(self, text: str) -> str:
        """Backward compatible intent detection"""
        return self.agent.detect_intent(text)

    def create_summary(self, exchanges: List[Dict]) -> str:
        """Simple summary for backward compatibility"""
        # Can be enhanced with agent's capabilities
        if not exchanges:
            return "We didn't get to explore any topics today."
        return f"We explored {len(exchanges)} topics today."

# Usage
conversation_handler = ConversationHandlerV2()
```

## Configuration Changes

### Environment Variables

Add to `.env`:

```bash
# ConversationAgent v2 Configuration
USE_CONVERSATION_V2=true
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_MAX_TOKENS=1024  # Increased from 150
ENABLE_TOOLS=true
ENABLE_STREAMING=false
```

### Update config.py

```python
# app/config.py
class Settings(BaseSettings):
    # ... existing settings ...

    # ConversationAgent v2
    use_conversation_v2: bool = Field(False, env="USE_CONVERSATION_V2")
    enable_tools: bool = Field(True, env="ENABLE_TOOLS")
    enable_streaming: bool = Field(False, env="ENABLE_STREAMING")

    # Claude Configuration (updated)
    claude_model: str = Field(
        "claude-3-5-sonnet-20241022",  # Changed from Haiku
        env="CLAUDE_MODEL"
    )
    claude_max_tokens: int = Field(
        1024,  # Increased from 150
        env="CLAUDE_MAX_TOKENS"
    )
```

## Testing Strategy

### Unit Tests

```bash
# Run ConversationAgent tests
pytest tests/test_conversation_agent.py -v

# Run integration tests
pytest tests/test_conversation_agent.py::TestToolCalling -v
```

### A/B Testing

Compare responses between v1 and v2:

```python
# tests/test_migration.py
import pytest
from app.conversation_handler import conversation_handler
from app.agents.conversation_agent import conversation_agent_v2
from app.agents.base import AgentMessage, MessageRole, MessageType

@pytest.mark.asyncio
async def test_response_comparison():
    """Compare responses from v1 and v2"""
    user_text = "What is machine learning?"
    context = [{"user": "Hello", "agent": "Hi! How can I help?"}]

    # v1 response
    v1_response = await conversation_handler.generate_response(
        user_text=user_text,
        context=context
    )

    # v2 response
    message = AgentMessage(
        role=MessageRole.USER,
        content=user_text,
        context={"conversation_history": context}
    )
    v2_response_message = await conversation_agent_v2.process(message)
    v2_response = v2_response_message.content

    # Both should respond (content check)
    assert len(v1_response) > 0
    assert len(v2_response) > 0

    print(f"\nv1: {v1_response}")
    print(f"\nv2: {v2_response}")
```

## Tool Usage Examples

### Calculator Tool

```python
message = AgentMessage(
    role=MessageRole.USER,
    content="What is the square root of 144?",
    message_type=MessageType.TEXT,
)

response = await conversation_agent_v2.process(message)
# Agent will use calculator tool and respond: "The square root of 144 is 12."
```

### Memory Tool

```python
# Store user preference
message = AgentMessage(
    role=MessageRole.USER,
    content="My name is Alice and I love Python programming",
    context={"memory_store": {}}
)

response = await conversation_agent_v2.process(message)
# Agent will store "user_name: Alice" and "favorite_language: Python"

# Later retrieval
message2 = AgentMessage(
    role=MessageRole.USER,
    content="What's my name again?",
    context={"memory_store": {"user_name": {"value": "Alice", "timestamp": "..."}}}
)

response2 = await conversation_agent_v2.process(message2)
# Agent will retrieve and respond: "Your name is Alice!"
```

### Search Tool

```python
message = AgentMessage(
    role=MessageRole.USER,
    content="What did we discuss about machine learning?",
    context={
        "conversation_history": [
            {"user": "I'm learning about neural networks", "agent": "Great!"},
            {"user": "I find backpropagation confusing", "agent": "Let me help..."},
        ]
    }
)

response = await conversation_agent_v2.process(message)
# Agent will search history and provide contextual response
```

## Monitoring & Metrics

### Track Agent Performance

```python
# Get agent metrics
metrics = conversation_agent_v2.get_metrics()

print(f"Total messages: {metrics['total_messages']}")
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Errors: {metrics['total_errors']}")
print(f"Avg processing time: {metrics['average_processing_time_ms']}ms")
```

### Compare v1 vs v2 Performance

```python
# app/monitoring.py
from prometheus_client import Histogram, Counter

conversation_latency = Histogram(
    'conversation_latency_seconds',
    'Conversation processing latency',
    ['version']  # v1 or v2
)

conversation_tokens = Counter(
    'conversation_tokens_total',
    'Total tokens used',
    ['version', 'model']
)

# Track both versions
@conversation_latency.labels(version='v2').time()
async def process_v2(message):
    response = await conversation_agent_v2.process(message)
    conversation_tokens.labels(
        version='v2',
        model='sonnet'
    ).inc(response.metadata.tokens_used)
    return response
```

## Rollback Plan

If issues arise, rollback is simple:

```bash
# 1. Set environment variable
export USE_CONVERSATION_V2=false

# 2. Restart application
docker-compose restart app

# 3. Verify v1 is active
curl -X POST http://localhost:8000/health
# Should show: "conversation_version": "v1"
```

## Cost Considerations

### Token Usage

- **Haiku**: ~$0.25 per million input tokens, ~$1.25 per million output tokens
- **Sonnet 3.5**: ~$3.00 per million input tokens, ~$15.00 per million output tokens

Sonnet is ~12x more expensive but provides significantly better quality.

### Cost Optimization

1. **Reduce max_tokens**: Set to 512 instead of 1024 for shorter conversations
2. **Context management**: Use conversation summarization for long sessions
3. **Selective upgrade**: Use Sonnet for complex queries, Haiku for simple ones
4. **Caching**: Cache tool results to avoid redundant API calls

```python
# Hybrid approach
def get_agent_for_query(query: str):
    """Route to appropriate model based on complexity"""
    complexity = assess_query_complexity(query)

    if complexity > 0.7:  # Complex query
        return conversation_agent_v2  # Sonnet
    else:  # Simple query
        return conversation_handler  # Haiku
```

## Common Issues & Solutions

### Issue 1: Import Errors

**Problem**: `ImportError: cannot import name 'MessageRole'`

**Solution**: Update imports:
```python
from app.agents.base import MessageRole, AgentMessage, MessageType
```

### Issue 2: Tool Not Found

**Problem**: Tool execution fails with "Tool not found"

**Solution**: Ensure tools are registered:
```python
from app.agents.tools import tool_registry
print(tool_registry.get_all_schemas())  # Verify tools loaded
```

### Issue 3: Context Format Mismatch

**Problem**: Context not passed correctly

**Solution**: Use correct format:
```python
# Correct
context = {
    "session_id": "123",
    "conversation_history": [
        {"user": "...", "agent": "..."}
    ],
    "memory_store": {}
}

# Incorrect
context = [{"user": "...", "agent": "..."}]  # Missing wrapper
```

## Next Steps

1. **Phase 1**: Enable v2 in development environment
2. **Phase 2**: A/B test with 10% of production traffic
3. **Phase 3**: Gradually increase to 50% traffic
4. **Phase 4**: Full migration after validation
5. **Phase 5**: Remove v1 code after 2 weeks of stable v2

## Support

For issues or questions:
- GitHub Issues: [project-repo]/issues
- Documentation: `/docs/conversation_agent.md`
- Tests: `/tests/test_conversation_agent.py`

## Conclusion

The new ConversationAgent provides significant improvements in intelligence, tool use, and extensibility while maintaining backward compatibility through the provided wrapper. Follow the gradual migration path for a safe transition.
