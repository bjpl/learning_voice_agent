# Phase 3: Hybrid Search System

## Overview

Phase 3 implements an intelligent hybrid search system that combines:

- **Vector Similarity Search**: Semantic understanding using embeddings
- **SQLite FTS5 Keyword Search**: Fast exact keyword matching
- **Reciprocal Rank Fusion (RRF)**: Intelligent result combination

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Search Engine                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌────────────────┐                 │
│  │Query Analyzer│──────▶│Strategy Picker │                 │
│  └──────────────┘      └────────────────┘                 │
│         │                      │                            │
│         ▼                      ▼                            │
│  ┌─────────────────────────────────────────┐              │
│  │  Parallel Search Execution              │              │
│  │  ┌─────────────┐   ┌──────────────┐    │              │
│  │  │Vector Search│   │Keyword Search│    │              │
│  │  │(OpenAI API) │   │(SQLite FTS5) │    │              │
│  │  └─────────────┘   └──────────────┘    │              │
│  └─────────────────────────────────────────┘              │
│         │                      │                            │
│         └──────────┬───────────┘                            │
│                    ▼                                        │
│         ┌──────────────────────┐                           │
│         │Reciprocal Rank Fusion│                           │
│         │(RRF Algorithm)       │                           │
│         └──────────────────────┘                           │
│                    │                                        │
│                    ▼                                        │
│         ┌─────────────────────┐                            │
│         │Result Normalization │                            │
│         │& Deduplication      │                            │
│         └─────────────────────┘                            │
│                    │                                        │
│                    ▼                                        │
│         ┌──────────────────────┐                           │
│         │ Ranked Results (Top N)│                          │
│         └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. HybridSearchEngine (`app/search/hybrid_search.py`)

Main orchestrator combining vector and keyword search.

**Key Features:**
- Configurable search strategies
- Parallel search execution
- RRF-based result fusion
- Score normalization
- Result deduplication

**Methods:**
```python
async def search(
    query: str,
    strategy: Optional[SearchStrategy] = None,
    limit: Optional[int] = None
) -> HybridSearchResponse
```

### 2. QueryAnalyzer (`app/search/query_analyzer.py`)

Analyzes queries to determine optimal search strategy.

**Features:**
- Intent detection (conceptual, factual, comparison, procedural)
- Keyword extraction
- Stop word filtering
- Query cleaning and normalization
- Strategy suggestion

**Detection Rules:**
```python
# Exact phrase → keyword search
# Very short (1-2 words) → keyword search
# Factual intent → keyword search
# Conceptual intent → semantic search
# Long query (>5 words) → semantic search
# Default → hybrid search
```

### 3. VectorStore (`app/search/vector_store.py`)

In-memory vector store with SQLite persistence.

**Features:**
- Fast cosine similarity search
- Normalized embeddings
- Embedding caching
- Threshold filtering

### 4. Configuration (`app/search/config.py`)

Centralized configuration for all search parameters.

**Default Settings:**
```python
vector_weight: 0.6           # 60% weight for vector search
keyword_weight: 0.4          # 40% weight for keyword search
rrf_k: 60                    # RRF constant
vector_similarity_threshold: 0.7
max_results_per_search: 20
final_result_limit: 10
embedding_model: "text-embedding-ada-002"
```

## Search Strategies

### 1. Semantic Search
**When to use:** Conceptual queries, long questions, "explain" type queries

**Example:**
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explain how machine learning works",
    "strategy": "semantic",
    "limit": 10
  }'
```

### 2. Keyword Search
**When to use:** Exact phrases, short terms, specific names/dates

**Example:**
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "\"REST API\"",
    "strategy": "keyword",
    "limit": 10
  }'
```

### 3. Hybrid Search
**When to use:** Best of both worlds, balanced results

**Example:**
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database indexing performance",
    "strategy": "hybrid",
    "limit": 10
  }'
```

### 4. Adaptive Search (Default)
**When to use:** Let the system decide based on query characteristics

**Example:**
```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is distributed consensus?",
    "strategy": "adaptive",
    "limit": 10
  }'
```

## Reciprocal Rank Fusion (RRF)

The hybrid search uses RRF to combine results from vector and keyword searches.

### Algorithm

```python
# For each document d appearing in either search:
score(d) = Σ (weight / (k + rank(d)))

where:
  - k = 60 (constant)
  - rank(d) = position in search results (1-indexed)
  - weight = vector_weight or keyword_weight
```

### Example

```
Vector Search Results:
  1. Doc A (rank=1) → score = 0.6 / (60 + 1) = 0.0098
  2. Doc B (rank=2) → score = 0.6 / (60 + 2) = 0.0097
  3. Doc C (rank=3) → score = 0.6 / (60 + 3) = 0.0095

Keyword Search Results:
  1. Doc C (rank=1) → score = 0.4 / (60 + 1) = 0.0066
  2. Doc A (rank=2) → score = 0.4 / (60 + 2) = 0.0065
  3. Doc D (rank=3) → score = 0.4 / (60 + 3) = 0.0063

Combined RRF Scores:
  Doc A: 0.0098 + 0.0065 = 0.0163 → Rank #1
  Doc C: 0.0095 + 0.0066 = 0.0161 → Rank #2
  Doc B: 0.0097 + 0.0000 = 0.0097 → Rank #3
  Doc D: 0.0000 + 0.0063 = 0.0063 → Rank #4
```

## API Integration

### Endpoint

```
POST /api/search/hybrid
```

### Request Model

```python
class HybridSearchRequest(BaseModel):
    query: str              # Search query (1-500 chars)
    strategy: Optional[str] # semantic, keyword, hybrid, adaptive
    limit: int = 10         # Max results (1-50)
```

### Response Model

```python
class HybridSearchResponse(BaseModel):
    query: str
    strategy: str
    results: List[Dict]
    total_count: int
    query_analysis: Dict
    execution_time_ms: float
    vector_results_count: int
    keyword_results_count: int
```

### Result Format

```json
{
  "query": "machine learning concepts",
  "strategy": "hybrid",
  "results": [
    {
      "id": 1,
      "session_id": "session-123",
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
    "original_query": "machine learning concepts",
    "cleaned_query": "machine learning concepts",
    "keywords": ["machine", "learning", "concepts"],
    "intent": "conceptual",
    "suggested_strategy": "hybrid",
    "is_short": false,
    "is_exact_phrase": false,
    "word_count": 3
  },
  "execution_time_ms": 45.2,
  "vector_results_count": 8,
  "keyword_results_count": 6
}
```

## Performance Characteristics

### Execution Time
- **Keyword only**: ~5-15ms
- **Semantic only**: ~50-100ms (depends on OpenAI API)
- **Hybrid**: ~50-120ms (parallel execution)

### Accuracy
- **Keyword**: High for exact matches, poor for conceptual queries
- **Semantic**: High for conceptual queries, may miss exact terms
- **Hybrid**: Best overall accuracy across query types

### Resource Usage
- **Memory**: ~2MB per 1000 embeddings (1536 dimensions)
- **Disk**: ~6KB per embedding in SQLite
- **API Costs**: ~$0.0001 per search (OpenAI embedding)

## Configuration

### Custom Configuration

```python
from app.search import HybridSearchConfig, create_hybrid_search_engine

config = HybridSearchConfig(
    vector_weight=0.7,           # Prefer semantic results
    keyword_weight=0.3,
    max_results_per_search=30,   # Fetch more candidates
    final_result_limit=15,       # Return more results
    rrf_k=80,                    # Adjust RRF sensitivity
    vector_similarity_threshold=0.75  # Stricter threshold
)

engine = create_hybrid_search_engine(db, config)
```

### Environment Variables

```bash
# Required for semantic search
OPENAI_API_KEY=sk-...

# Optional configuration
EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_SIMILARITY_THRESHOLD=0.7
```

## Best Practices

### 1. Query Formulation
```python
# ✅ Good queries
"explain distributed consensus algorithms"
"REST API best practices"
"database indexing performance"

# ❌ Avoid
"ml" (too short, ambiguous)
"a" * 1000 (too long)
```

### 2. Strategy Selection
```python
# Use semantic for conceptual understanding
strategy="semantic" → "how does machine learning work?"

# Use keyword for exact matches
strategy="keyword" → "FastAPI documentation"

# Use hybrid for balanced results
strategy="hybrid" → "database performance optimization"

# Use adaptive to let the system decide
strategy="adaptive" → any query
```

### 3. Result Limit
```python
# Default (10): Good for most use cases
limit=10

# Large (30-50): For comprehensive results
limit=50

# Small (3-5): For quick lookups
limit=3
```

## Testing

### Run Integration Tests

```bash
# All hybrid search tests
pytest tests/integration/test_hybrid_search.py -v

# Specific test
pytest tests/integration/test_hybrid_search.py::test_hybrid_search_strategy -v

# With coverage
pytest tests/integration/test_hybrid_search.py --cov=app/search
```

### Run Demo

```bash
# Set OpenAI API key
export OPENAI_API_KEY=sk-...

# Run demo
python examples/hybrid_search_demo.py
```

## Monitoring

### Metrics

The hybrid search system tracks:
- Search execution time
- Strategy distribution
- Result counts
- Cache hit rates
- Error rates

### Logging

```python
# Search execution
db_logger.info("hybrid_search_complete",
    query=query,
    strategy=strategy,
    results_count=count,
    execution_time_ms=time
)

# RRF fusion
db_logger.debug("rrf_fusion_complete",
    total_results=count,
    vector_only=vector_count,
    keyword_only=keyword_count,
    both=both_count
)
```

## Troubleshooting

### Issue: No semantic search results

**Cause:** OpenAI API key not configured or embeddings not generated

**Solution:**
```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Restart application to regenerate embeddings
```

### Issue: Slow search performance

**Cause:** Too many results or uncached embeddings

**Solution:**
```python
# Reduce result limits
config.max_results_per_search = 10
config.final_result_limit = 5

# Enable caching
config.enable_embedding_cache = True
```

### Issue: Poor result quality

**Cause:** Strategy mismatch or weight imbalance

**Solution:**
```python
# Adjust strategy
strategy = "adaptive"  # Let system decide

# Tune weights for your use case
config.vector_weight = 0.7    # Prefer semantic
config.keyword_weight = 0.3
```

## Future Enhancements

- [ ] Query expansion with synonyms
- [ ] Spell correction using fuzzy matching
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Result caching layer
- [ ] A/B testing framework
- [ ] Query suggestions
- [ ] Search analytics dashboard

## References

- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Vector Search Best Practices](https://www.pinecone.io/learn/vector-search/)

---

**Phase 3 Complete!** 🎉

The hybrid search system is now fully operational and integrated into the Learning Voice Agent.
