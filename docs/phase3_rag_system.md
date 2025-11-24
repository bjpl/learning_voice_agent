# Phase 3: RAG (Retrieval-Augmented Generation) System

**Status:** ✅ Complete
**Version:** 1.0.0
**Date:** 2025-11-21

## Overview

The RAG system enhances AI responses by retrieving relevant conversation history and injecting it as context for Claude 3.5 Sonnet. This enables contextually-aware, consistent responses that reference past conversations.

## Architecture

### Three-Stage Pipeline

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Query     │─────▶│   RAGRetriever   │─────▶│  Retrieved  │
│             │      │ (Hybrid Search)  │      │  Documents  │
└─────────────┘      └──────────────────┘      └──────┬──────┘
                                                       │
                                                       ▼
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Response   │◀─────│   RAGGenerator   │◀─────│   Context   │
│ + Citations │      │  (Claude 3.5)    │      │   Builder   │
└─────────────┘      └──────────────────┘      └─────────────┘
```

### Components

1. **RAGRetriever** (`app/rag/retriever.py`)
   - Hybrid search (vector + keyword)
   - Relevance filtering (threshold: 0.7)
   - Recency weighting
   - Deduplication
   - Top-k selection (default: 5)

2. **ContextBuilder** (`app/rag/context_builder.py`)
   - Token-aware assembly
   - Multiple format options
   - Automatic summarization
   - Context window management (max: 4000 tokens)

3. **RAGGenerator** (`app/rag/generator.py`)
   - Claude 3.5 Sonnet integration
   - RAG-optimized prompts
   - Citation extraction
   - Graceful fallback

## Installation

### Dependencies

```bash
# Install tiktoken for token counting
pip install tiktoken>=0.5.2

# Already installed from Phase 2
pip install anthropic>=0.18.1
pip install langchain-anthropic>=0.1.0
```

## Configuration

### Environment Variables

```bash
# RAG Configuration (all optional, defaults shown)

# Embedding
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSION=1536

# Retrieval
RAG_RETRIEVAL_TOP_K=5
RAG_RELEVANCE_THRESHOLD=0.7
RAG_USE_HYBRID_SEARCH=true
RAG_HYBRID_ALPHA=0.7

# Context Building
RAG_MAX_CONTEXT_TOKENS=4000
RAG_ENABLE_CONTEXT_SUMMARIZATION=true
RAG_PRIORITIZE_RECENT=true
RAG_RECENCY_DECAY_DAYS=7

# Generation
RAG_GENERATION_MODEL=claude-3-5-sonnet-20241022
RAG_GENERATION_MAX_TOKENS=1500
RAG_GENERATION_TEMPERATURE=0.7
RAG_ENABLE_CITATIONS=true

# Fallback
RAG_ENABLE_FALLBACK=true
RAG_FALLBACK_TO_BASIC_MODE=true
```

### Performance Profiles

```python
from app.rag import update_rag_config, get_performance_profile

# Fast mode (lower quality, faster)
update_rag_config(**get_performance_profile("fast"))

# Balanced mode (default)
update_rag_config(**get_performance_profile("balanced"))

# Quality mode (higher quality, slower)
update_rag_config(**get_performance_profile("quality"))
```

## Usage

### Basic Example

```python
from app.rag import RAGRetriever, ContextBuilder, RAGGenerator
from app.database import Database
from app.search.hybrid_search import create_hybrid_search_engine
from anthropic import AsyncAnthropic

# Initialize components
database = Database()
await database.initialize()

hybrid_search = create_hybrid_search_engine(database)
anthropic_client = AsyncAnthropic(api_key="your-api-key")

retriever = RAGRetriever(database, hybrid_search)
context_builder = ContextBuilder()
generator = RAGGenerator(anthropic_client)

# Execute RAG pipeline
query = "What did we discuss about machine learning?"

# 1. Retrieve relevant documents
retrieval_response = await retriever.retrieve(query, session_id="session-123")
print(f"Retrieved {retrieval_response.total_retrieved} documents")

# 2. Build context
context = await context_builder.build_context(
    retrieval_results=retrieval_response.results,
    query=query
)
print(f"Context uses {context.total_tokens} tokens")

# 3. Generate response
generation_response = await generator.generate(
    query=query,
    context=context
)
print(f"Response: {generation_response.response_text}")
print(f"Citations: {len(generation_response.citations)}")
```

### Advanced Example: Session-Scoped Search

```python
# Search only within current session
retrieval_response = await retriever.retrieve(
    query="What was my last question?",
    session_id="current-session-id",
    top_k=3
)
```

### Advanced Example: Custom Formatting

```python
from app.rag.context_builder import ContextFormat

# Compact format (saves tokens)
context = await context_builder.build_context(
    retrieval_results=retrieval_response.results,
    query=query,
    format_type=ContextFormat.COMPACT
)

# Conversation format (natural dialog)
context = await context_builder.build_context(
    retrieval_results=retrieval_response.results,
    query=query,
    format_type=ContextFormat.CONVERSATION
)
```

### Advanced Example: Batch Generation

```python
queries = [
    "What are my project goals?",
    "What technologies are we using?",
    "What are the next steps?"
]

# Retrieve context for each query
contexts = []
for query in queries:
    retrieval = await retriever.retrieve(query)
    context = await context_builder.build_context(retrieval.results, query)
    contexts.append(context)

# Generate responses concurrently
responses = await generator.generate_batch(queries, contexts)

for query, response in zip(queries, responses):
    print(f"Q: {query}")
    print(f"A: {response.response_text}\n")
```

## API Reference

### RAGRetriever

```python
retriever = RAGRetriever(database, hybrid_search_engine, config=None)

# Main retrieval method
response = await retriever.retrieve(
    query: str,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
    strategy: Optional[RetrievalStrategy] = None
) -> RetrievalResponse

# Find similar conversations
similar = await retriever.retrieve_similar_conversations(
    conversation_id: int,
    top_k: int = 5
) -> RetrievalResponse
```

**RetrievalResponse:**
- `query`: Original query
- `results`: List of RetrievalResult objects
- `total_retrieved`: Number of results
- `strategy_used`: Search strategy used
- `execution_time_ms`: Time taken

### ContextBuilder

```python
builder = ContextBuilder(config=None)

# Build context from results
context = await builder.build_context(
    retrieval_results: List[RetrievalResult],
    query: str,
    format_type: ContextFormat = ContextFormat.STRUCTURED,
    max_tokens: Optional[int] = None
) -> BuiltContext
```

**BuiltContext:**
- `formatted_context`: Formatted context string
- `documents`: List of ContextDocument objects
- `total_tokens`: Token count
- `is_truncated`: Whether documents were truncated
- `is_summarized`: Whether context was summarized

### RAGGenerator

```python
generator = RAGGenerator(anthropic_client, config=None)

# Generate single response
response = await generator.generate(
    query: str,
    context: Optional[BuiltContext] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> GenerationResponse

# Batch generation
responses = await generator.generate_batch(
    queries: List[str],
    contexts: Optional[List[BuiltContext]] = None
) -> List[GenerationResponse]
```

**GenerationResponse:**
- `response_text`: Generated response
- `citations`: List of Citation objects
- `tokens_used`: Output tokens used
- `context_used`: Whether context was used
- `mode`: Generation mode (rag/basic/fallback)

## Performance

### Benchmarks

Based on testing with conversation history of 1,000 captures:

| Metric | Value |
|--------|-------|
| Retrieval Time | 50-150ms |
| Context Building | 10-30ms |
| Generation Time | 800-2000ms |
| **Total E2E** | **~1-2 seconds** |

### Token Usage

| Component | Tokens |
|-----------|--------|
| Retrieved Context | 500-4000 |
| User Query | 10-100 |
| System Prompt | ~100 |
| Generation Output | 100-1500 |
| **Total** | **~700-5700** |

### Optimization Tips

1. **Reduce top_k** for faster retrieval:
   ```python
   retrieval = await retriever.retrieve(query, top_k=3)  # Instead of 5
   ```

2. **Use compact format** to save tokens:
   ```python
   context = await builder.build_context(results, query, ContextFormat.COMPACT)
   ```

3. **Lower max_tokens** for faster generation:
   ```python
   response = await generator.generate(query, context, max_tokens=1000)
   ```

4. **Use performance profiles**:
   ```python
   from app.rag import update_rag_config, get_performance_profile
   update_rag_config(**get_performance_profile("fast"))
   ```

## Fallback Behavior

The RAG system gracefully degrades when components fail:

### Retrieval Failure
- Returns empty RetrievalResponse
- Generator continues with basic mode (no context)

### Context Building Failure
- Returns empty BuiltContext
- Generator continues with basic mode

### Generation Failure
- Returns fallback response with error message
- Logs error for debugging

### Example: Handle Failures

```python
try:
    retrieval = await retriever.retrieve(query)

    if retrieval.total_retrieved == 0:
        print("No relevant context found, using basic generation")
        context = None
    else:
        context = await context_builder.build_context(retrieval.results, query)

    response = await generator.generate(query, context)

except Exception as e:
    print(f"RAG pipeline failed: {e}")
    # Fallback to basic generation
    response = await generator.generate(query, context=None)
```

## Testing

### Unit Tests

```bash
# Run RAG tests
pytest tests/test_rag_retriever.py
pytest tests/test_rag_context_builder.py
pytest tests/test_rag_generator.py
```

### Integration Test

```python
# See examples/phase3_rag_integration.py
python examples/phase3_rag_integration.py
```

## Monitoring

### Logging

The RAG system logs detailed events:

```python
# Retrieval events
"rag_retrieval_started"
"rag_retrieval_complete"
"rag_retrieval_failed"

# Context building events
"context_building_started"
"context_building_complete"
"context_summarization_triggered"

# Generation events
"rag_generation_started"
"rag_generation_complete"
"rag_generation_failed"
```

### Metrics to Monitor

1. **Retrieval Quality**
   - Average relevance score
   - % of queries with 0 results
   - Average documents retrieved

2. **Context Management**
   - Average context tokens
   - % of truncated contexts
   - % of summarized contexts

3. **Generation Quality**
   - Average response length
   - % with citations
   - Token usage

4. **Performance**
   - E2E latency (p50, p95, p99)
   - Individual component latency
   - Error rate

## Troubleshooting

### Issue: No results retrieved

**Cause:** Query doesn't match any conversation history

**Solution:**
- Lower relevance threshold: `RAG_RELEVANCE_THRESHOLD=0.6`
- Increase top_k: `RAG_RETRIEVAL_TOP_K=10`
- Check if embeddings exist in database

### Issue: Context too large

**Cause:** Retrieved documents exceed token limit

**Solution:**
- Enable summarization: `RAG_ENABLE_CONTEXT_SUMMARIZATION=true`
- Reduce max_tokens: `RAG_MAX_CONTEXT_TOKENS=2000`
- Reduce top_k: `RAG_RETRIEVAL_TOP_K=3`

### Issue: Poor response quality

**Cause:** Irrelevant context or poor retrieval

**Solution:**
- Increase relevance threshold: `RAG_RELEVANCE_THRESHOLD=0.75`
- Enable hybrid search: `RAG_USE_HYBRID_SEARCH=true`
- Use quality profile: `get_performance_profile("quality")`

### Issue: Slow generation

**Cause:** Large context or high max_tokens

**Solution:**
- Use fast profile: `get_performance_profile("fast")`
- Reduce context tokens: `RAG_MAX_CONTEXT_TOKENS=2000`
- Reduce generation tokens: `RAG_GENERATION_MAX_TOKENS=1000`

## Next Steps (Phase 4)

Potential enhancements:

1. **Streaming Responses**
   - Stream Claude responses as they generate
   - Update citations in real-time

2. **Advanced Citations**
   - Use Claude to generate structured citations
   - Extract relevant snippets automatically

3. **Context Caching**
   - Cache built contexts for common queries
   - Reduce context building latency

4. **Multi-turn RAG**
   - Maintain conversation state across turns
   - Update context dynamically

5. **Evaluation Framework**
   - Measure retrieval precision/recall
   - Evaluate response quality
   - A/B testing infrastructure

## References

- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [RAG Paper: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Hybrid Search Best Practices](https://www.pinecone.io/learn/hybrid-search-intro/)

## License

MIT License - See LICENSE file for details

---

**Phase 3 Complete** ✅
Next: Phase 4 - Advanced Features and Optimization
