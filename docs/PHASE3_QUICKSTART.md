# Phase 3 Hybrid Search - Quick Start Guide

## Prerequisites

```bash
# Install Phase 3 dependencies
pip install numpy chromadb==0.4.22 sentence-transformers==2.3.1
```

## Setup

### 1. Configure Environment (Optional)

For semantic search capabilities, add to `.env`:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

**Note**: The system works without OpenAI API key using keyword-only search. Semantic search is optional but recommended.

### 2. Start the Application

```bash
# From project root
python -m uvicorn app.main:app --reload
```

The hybrid search will initialize automatically on startup.

## Using Hybrid Search

### 1. Basic Search (Adaptive Strategy)

```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explain machine learning",
    "limit": 10
  }'
```

The system will automatically choose the best strategy based on your query.

### 2. Semantic Search (Conceptual)

Best for understanding concepts:
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does distributed consensus work?",
    "strategy": "semantic",
    "limit": 10
  }'
```

### 3. Keyword Search (Exact Matches)

Best for specific terms:
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI authentication",
    "strategy": "keyword",
    "limit": 10
  }'
```

### 4. Hybrid Search (Balanced)

Best combination of both:
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database performance optimization",
    "strategy": "hybrid",
    "limit": 10
  }'
```

## Response Format

```json
{
  "query": "machine learning",
  "strategy": "hybrid",
  "results": [
    {
      "id": 1,
      "session_id": "session-abc",
      "timestamp": "2025-11-21T10:00:00",
      "user_text": "What is machine learning?",
      "agent_text": "Machine learning is...",
      "score": 0.95,
      "rank": 1,
      "source": "hybrid",
      "vector_score": 0.87,
      "keyword_score": 0.92
    }
  ],
  "total_count": 10,
  "query_analysis": {
    "intent": "conceptual",
    "keywords": ["machine", "learning"],
    "suggested_strategy": "hybrid"
  },
  "execution_time_ms": 45.2,
  "vector_results_count": 8,
  "keyword_results_count": 6
}
```

## Running the Demo

```bash
# Set API key (optional but recommended)
export OPENAI_API_KEY=sk-...

# Run demo
python examples/hybrid_search_demo.py
```

The demo will:
1. Create sample conversation data
2. Generate embeddings (if API key available)
3. Demonstrate all search strategies
4. Show query analysis
5. Display performance statistics

## Running Tests

```bash
# All hybrid search tests
pytest tests/integration/test_hybrid_search.py -v

# Specific test
pytest tests/integration/test_hybrid_search.py::test_hybrid_search_strategy -v

# With coverage
pytest tests/integration/test_hybrid_search.py --cov=app/search --cov-report=html
```

## Search Strategy Decision Tree

```
Query Analysis
    │
    ├─ Exact phrase ("...") ──────────► KEYWORD
    │
    ├─ Very short (1-2 words) ────────► KEYWORD
    │
    ├─ Factual intent ────────────────► KEYWORD
    │   (when, where, who, date)
    │
    ├─ Conceptual intent ─────────────► SEMANTIC
    │   (explain, how does, why)
    │
    ├─ Long query (>5 words) ─────────► SEMANTIC
    │
    └─ Default ───────────────────────► HYBRID
```

## Performance Tips

### 1. Choose the Right Strategy

- **Semantic**: "explain how neural networks learn"
- **Keyword**: "FastAPI" or "database indexing"
- **Hybrid**: "database performance optimization tips"
- **Adaptive**: Let the system decide (recommended)

### 2. Optimize Result Limits

```python
# Fast queries (< 20ms)
limit = 5

# Balanced (20-50ms)
limit = 10

# Comprehensive (50-120ms)
limit = 30
```

### 3. Cache Frequently Used Queries

Embeddings are automatically cached for repeated queries.

## Monitoring

### Check Search Health

```bash
curl http://localhost:8000/api/health
```

Look for:
```json
{
  "dependencies": {
    "database": {"status": "healthy"},
    "claude_api": {"status": "configured"},
    "whisper_api": {"status": "configured"}
  }
}
```

### View Metrics

```bash
# Prometheus format
curl http://localhost:8000/metrics

# JSON format
curl http://localhost:8000/api/metrics
```

## Troubleshooting

### Problem: "Hybrid search is not available" (503 error)

**Cause**: Search engine failed to initialize

**Solution**:
1. Check logs for initialization errors
2. Verify database exists and is accessible
3. Restart application

### Problem: Slow semantic search

**Cause**: OpenAI API latency

**Solutions**:
1. Use `strategy="keyword"` for faster results
2. Reduce result limit
3. Enable result caching

### Problem: No results returned

**Causes**:
1. Query too specific
2. No matching content in database
3. Threshold too high

**Solutions**:
1. Try broader query
2. Use hybrid or keyword strategy
3. Lower similarity threshold in config

## Advanced Configuration

Create custom config:

```python
from app.search import HybridSearchConfig, create_hybrid_search_engine
from app.database import db

# Custom configuration
config = HybridSearchConfig(
    vector_weight=0.7,           # Prefer semantic results
    keyword_weight=0.3,
    max_results_per_search=30,   # More candidates
    final_result_limit=15,
    rrf_k=80,                    # Adjust RRF sensitivity
    vector_similarity_threshold=0.75  # Stricter threshold
)

# Create engine with custom config
engine = create_hybrid_search_engine(db, config)
```

## API Reference

### Endpoint
`POST /api/search/hybrid`

### Request Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Search query (1-500 chars) |
| strategy | string | No | "adaptive" | Search strategy |
| limit | integer | No | 10 | Max results (1-50) |

### Strategies
- `semantic` - Pure vector search
- `keyword` - Pure FTS5 search
- `hybrid` - Combined with RRF
- `adaptive` - Auto-select (default)

### Rate Limits
- 30 requests per minute per IP
- Use `/api/search` for higher limits (keyword-only)

## Examples

### Python Client

```python
import httpx

async def search_hybrid(query: str, strategy: str = "adaptive"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/search/hybrid",
            json={
                "query": query,
                "strategy": strategy,
                "limit": 10
            }
        )
        return response.json()

# Usage
results = await search_hybrid("machine learning concepts")
print(f"Found {results['total_count']} results in {results['execution_time_ms']}ms")
```

### JavaScript/TypeScript Client

```typescript
async function searchHybrid(query: string, strategy: string = "adaptive") {
  const response = await fetch("http://localhost:8000/api/search/hybrid", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      strategy,
      limit: 10
    })
  });

  return await response.json();
}

// Usage
const results = await searchHybrid("database optimization");
console.log(`Found ${results.total_count} results`);
```

## Next Steps

1. ✅ **Read Full Documentation**: See `docs/phase3_hybrid_search.md`
2. ✅ **Run Demo**: `python examples/hybrid_search_demo.py`
3. ✅ **Run Tests**: `pytest tests/integration/test_hybrid_search.py -v`
4. ✅ **Integrate**: Add hybrid search to your application
5. ✅ **Monitor**: Track performance and accuracy
6. ✅ **Optimize**: Tune weights and thresholds for your use case

## Support

- Full Documentation: `/docs/phase3_hybrid_search.md`
- Implementation Summary: `/docs/phase3_implementation_summary.md`
- Test Examples: `/tests/integration/test_hybrid_search.py`
- Demo Script: `/examples/hybrid_search_demo.py`

---

**Happy Searching!** 🔍✨
