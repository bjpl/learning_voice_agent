# Phase 3 Testing Guide

**Version:** 1.0.0
**Date:** 2025-01-21
**Coverage Target:** 80%+ for all Phase 3 components

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Structure](#test-structure)
3. [Mock Setup](#mock-setup)
4. [Vector Store Testing](#vector-store-testing)
5. [Embedding Testing](#embedding-testing)
6. [Hybrid Search Testing](#hybrid-search-testing)
7. [Knowledge Graph Testing](#knowledge-graph-testing)
8. [Integration Testing](#integration-testing)
9. [Performance Benchmarks](#performance-benchmarks)
10. [CI/CD Integration](#cicd-integration)

---

## Testing Strategy

### Test Pyramid

```
          ┌─────────────┐
          │   E2E Tests │  (10%)
          │   5 tests   │
          └─────────────┘
        ┌─────────────────┐
        │Integration Tests│  (20%)
        │    25 tests     │
        └─────────────────┘
    ┌───────────────────────┐
    │     Unit Tests        │  (70%)
    │     120+ tests        │
    └───────────────────────┘
```

### Test Categories

1. **Unit Tests** (120+ tests):
   - Vector store operations
   - Embedding generation
   - Search algorithms
   - Query analysis
   - Knowledge graph CRUD

2. **Integration Tests** (25+ tests):
   - Hybrid search end-to-end
   - RAG pipeline
   - Cross-component workflows

3. **Performance Tests** (5+ tests):
   - Search latency
   - Batch embedding throughput
   - Graph query performance

---

## Test Structure

### Directory Layout

```
tests/
├── vector/
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures for vector tests
│   ├── test_vector_store.py           # 25+ tests
│   └── test_embeddings.py             # 20+ tests
├── search/
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures for search tests
│   ├── test_hybrid_search.py          # 30+ tests
│   └── test_query_analyzer.py         # 15+ tests
├── knowledge_graph/
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures for graph tests
│   └── test_graph_store.py            # 30+ tests
├── integration/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_phase3_integration.py     # 25+ tests
└── performance/
    ├── __init__.py
    └── test_benchmarks.py             # 5+ tests
```

### Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_<component>_<scenario>()`
- Fixtures: `<resource>_fixture()`
- Mock objects: `mock_<component>()`

---

## Mock Setup

### ChromaDB Mock

**File:** `tests/vector/conftest.py`

```python
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import numpy as np

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    client = MagicMock()

    # Mock collection
    collection = MagicMock()
    collection.count.return_value = 100
    collection.add = MagicMock()
    collection.query = MagicMock(return_value={
        'ids': [['id1', 'id2']],
        'documents': [['doc1', 'doc2']],
        'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
        'distances': [[0.1, 0.2]]
    })

    client.get_or_create_collection.return_value = collection
    client.list_collections.return_value = [collection]

    return client

@pytest.fixture
def mock_vector_store(mock_chroma_client, monkeypatch):
    """Mock VectorStore instance"""
    from app.vector.vector_store import VectorStore

    store = VectorStore()
    monkeypatch.setattr(store, 'client', mock_chroma_client)
    store._initialized = True
    store.collections = {'conversations': mock_chroma_client.get_or_create_collection()}

    return store
```

### Embedding Generator Mock

```python
@pytest.fixture
def mock_embedding_generator():
    """Mock EmbeddingGenerator"""
    from app.vector.embeddings import EmbeddingGenerator

    generator = MagicMock(spec=EmbeddingGenerator)
    generator._initialized = True
    generator.generate_embedding = AsyncMock(
        return_value=np.random.rand(384).astype(np.float32)
    )
    generator.generate_batch = AsyncMock(
        side_effect=lambda texts, **kwargs: [
            np.random.rand(384).astype(np.float32) for _ in texts
        ]
    )

    return generator
```

### Neo4j Mock

**File:** `tests/knowledge_graph/conftest.py`

```python
@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j AsyncDriver"""
    driver = AsyncMock()

    # Mock session
    session = AsyncMock()

    # Mock query results
    result = AsyncMock()
    result.single = AsyncMock(return_value={
        'name': 'test_concept',
        'frequency': 5
    })
    result.values = AsyncMock(return_value=[
        ['related_concept', 'description', 3, ['RELATES_TO'], [0.8], 1]
    ])

    session.run = AsyncMock(return_value=result)

    # Context manager for session
    @asynccontextmanager
    async def get_session():
        yield session

    driver.session = MagicMock(return_value=get_session())
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()

    return driver
```

---

## Vector Store Testing

### Test Coverage Areas

1. **Initialization** (3 tests)
   - Successful initialization
   - Retry on failure
   - Directory creation

2. **Adding Embeddings** (5 tests)
   - Single embedding
   - Batch embeddings
   - With metadata
   - Pre-computed embeddings
   - Duplicate handling

3. **Searching** (8 tests)
   - Text query
   - Embedding query
   - Metadata filtering
   - Similarity threshold
   - Result limiting
   - Empty results
   - Invalid collection

4. **Collection Management** (5 tests)
   - Create collection
   - Delete collection
   - List collections
   - Get statistics
   - Invalid operations

5. **Error Handling** (4 tests)
   - Connection failures
   - Invalid inputs
   - Retry exhaustion
   - Graceful degradation

### Example Tests

**File:** `tests/vector/test_vector_store.py`

```python
import pytest
import numpy as np
from app.vector.vector_store import VectorStore

@pytest.mark.asyncio
async def test_add_embedding_success(mock_vector_store, mock_embedding_generator):
    """Test successful embedding addition"""
    # Arrange
    store = mock_vector_store
    text = "Machine learning is fascinating"
    metadata = {"session_id": "test_session"}

    # Act
    doc_id = await store.add_embedding(
        collection_name="conversations",
        text=text,
        metadata=metadata
    )

    # Assert
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    # Verify ChromaDB was called correctly
    collection = store.collections['conversations']
    collection.add.assert_called_once()
    call_args = collection.add.call_args
    assert len(call_args.kwargs['ids']) == 1
    assert len(call_args.kwargs['embeddings']) == 1
    assert call_args.kwargs['documents'][0] == text


@pytest.mark.asyncio
async def test_add_batch_embeddings(mock_vector_store):
    """Test batch embedding addition"""
    # Arrange
    texts = ["text1", "text2", "text3"]
    metadatas = [{"id": i} for i in range(3)]

    # Act
    doc_ids = await mock_vector_store.add_batch(
        collection_name="conversations",
        texts=texts,
        metadatas=metadatas
    )

    # Assert
    assert len(doc_ids) == 3
    assert all(isinstance(doc_id, str) for doc_id in doc_ids)


@pytest.mark.asyncio
async def test_search_similar_by_text(mock_vector_store):
    """Test similarity search with text query"""
    # Arrange
    query = "What is deep learning?"

    # Act
    results = await mock_vector_store.search_similar(
        collection_name="conversations",
        query_text=query,
        n_results=5
    )

    # Assert
    assert isinstance(results, list)
    assert len(results) <= 5

    for result in results:
        assert 'id' in result
        assert 'document' in result
        assert 'similarity' in result
        assert 0.0 <= result['similarity'] <= 1.0


@pytest.mark.asyncio
async def test_search_with_metadata_filter(mock_vector_store):
    """Test search with metadata filtering"""
    # Arrange
    query = "neural networks"
    metadata_filter = {"session_id": "specific_session"}

    # Act
    results = await mock_vector_store.search_similar(
        collection_name="conversations",
        query_text=query,
        metadata_filter=metadata_filter
    )

    # Assert
    collection = mock_vector_store.collections['conversations']
    collection.query.assert_called_once()
    call_args = collection.query.call_args
    assert call_args.kwargs['where'] == metadata_filter


@pytest.mark.asyncio
async def test_delete_embedding(mock_vector_store):
    """Test embedding deletion"""
    # Arrange
    doc_id = "test_doc_id"

    # Act
    result = await mock_vector_store.delete_embedding(
        collection_name="conversations",
        document_id=doc_id
    )

    # Assert
    assert result is True
    collection = mock_vector_store.collections['conversations']
    collection.delete.assert_called_once_with(ids=[doc_id])


@pytest.mark.asyncio
async def test_collection_not_found_error(mock_vector_store):
    """Test error when collection doesn't exist"""
    # Act & Assert
    with pytest.raises(ValueError, match="Collection .* not found"):
        await mock_vector_store.add_embedding(
            collection_name="nonexistent",
            text="test"
        )
```

---

## Embedding Testing

### Test Coverage Areas

1. **Model Loading** (3 tests)
   - Successful loading
   - Retry on download failure
   - GPU/CPU device selection

2. **Single Embedding** (5 tests)
   - Basic generation
   - Cache hit
   - Cache miss
   - Long text truncation
   - Error handling

3. **Batch Embedding** (5 tests)
   - Small batch
   - Large batch
   - Mixed cache hits/misses
   - Progress bar
   - Empty input

4. **Cache Management** (4 tests)
   - LRU eviction
   - TTL expiration
   - Clear cache
   - Cache statistics

5. **Performance** (3 tests)
   - Latency measurement
   - Throughput benchmark
   - Memory usage

### Example Tests

**File:** `tests/vector/test_embeddings.py`

```python
import pytest
import numpy as np
from app.vector.embeddings import EmbeddingGenerator, EmbeddingCache

@pytest.mark.asyncio
async def test_generate_single_embedding():
    """Test single embedding generation"""
    # Arrange
    generator = EmbeddingGenerator()
    await generator.initialize()

    # Act
    embedding = await generator.generate_embedding("test text")

    # Assert
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


@pytest.mark.asyncio
async def test_embedding_cache_hit(mock_embedding_generator):
    """Test cache hit behavior"""
    # Arrange
    text = "cached text"
    expected_embedding = np.random.rand(384).astype(np.float32)

    cache = EmbeddingCache(max_size=10)
    cache.set(text, expected_embedding)

    mock_embedding_generator.cache = cache

    # Act
    result = cache.get(text)

    # Assert
    assert result is not None
    np.testing.assert_array_equal(result, expected_embedding)


@pytest.mark.asyncio
async def test_batch_embedding_generation():
    """Test batch embedding generation"""
    # Arrange
    generator = EmbeddingGenerator()
    await generator.initialize()
    texts = ["text1", "text2", "text3"]

    # Act
    embeddings = await generator.generate_batch(texts)

    # Assert
    assert len(embeddings) == 3
    assert all(emb.shape == (384,) for emb in embeddings)


def test_cache_lru_eviction():
    """Test LRU cache eviction"""
    # Arrange
    cache = EmbeddingCache(max_size=2)
    emb1 = np.array([1.0] * 384, dtype=np.float32)
    emb2 = np.array([2.0] * 384, dtype=np.float32)
    emb3 = np.array([3.0] * 384, dtype=np.float32)

    # Act
    cache.set("text1", emb1)
    cache.set("text2", emb2)
    cache.set("text3", emb3)  # Should evict text1

    # Assert
    assert cache.get("text1") is None  # Evicted
    assert cache.get("text2") is not None
    assert cache.get("text3") is not None
```

---

## Hybrid Search Testing

### Test Coverage Areas

1. **Query Analysis** (8 tests)
   - Intent detection (conceptual, factual, comparison, procedural)
   - Keyword extraction
   - Stop word removal
   - Exact phrase detection
   - Strategy suggestion

2. **Vector Search** (7 tests)
   - Successful search
   - Empty results
   - Threshold filtering
   - Cache behavior
   - Error handling

3. **Keyword Search** (5 tests)
   - FTS5 integration
   - Snippet generation
   - Ranking
   - Special characters

4. **RRF Fusion** (7 tests)
   - Score combination
   - Deduplication
   - Weight adjustment
   - Edge cases (one empty result)

5. **End-to-End** (3 tests)
   - Adaptive strategy
   - Hybrid strategy
   - Performance benchmarks

### Example Tests

**File:** `tests/search/test_hybrid_search.py`

```python
import pytest
from app.search.hybrid_search import HybridSearchEngine
from app.search.config import SearchStrategy

@pytest.mark.asyncio
async def test_query_intent_detection():
    """Test query intent detection"""
    from app.search.query_analyzer import query_analyzer

    # Conceptual query
    analysis = await query_analyzer.analyze("What is machine learning?")
    assert analysis.intent == "conceptual"
    assert analysis.suggested_strategy == SearchStrategy.SEMANTIC

    # Factual query
    analysis = await query_analyzer.analyze("When was Python created?")
    assert analysis.intent == "factual"
    assert analysis.suggested_strategy == SearchStrategy.KEYWORD


@pytest.mark.asyncio
async def test_hybrid_search_execution(mock_database, mock_openai_client):
    """Test complete hybrid search execution"""
    # Arrange
    from app.search.hybrid_search import create_hybrid_search_engine

    engine = create_hybrid_search_engine(mock_database)
    engine.set_embedding_client(mock_openai_client)

    # Act
    response = await engine.search(
        query="machine learning",
        strategy=SearchStrategy.HYBRID,
        limit=10
    )

    # Assert
    assert response.query == "machine learning"
    assert response.strategy == "hybrid"
    assert response.total_count <= 10
    assert response.execution_time_ms > 0
    assert isinstance(response.results, list)


@pytest.mark.asyncio
async def test_rrf_fusion_combines_results():
    """Test Reciprocal Rank Fusion combining results"""
    from app.search.hybrid_search import HybridSearchEngine
    from app.search.config import HybridSearchConfig

    # Arrange
    engine = HybridSearchEngine(
        database=mock_database,
        vector_store=mock_vector_store,
        query_analyzer=mock_query_analyzer,
        config=HybridSearchConfig(vector_weight=0.6, keyword_weight=0.4)
    )

    vector_results = [
        SearchResult(id=1, rank=1, score=0.9, source='vector'),
        SearchResult(id=2, rank=2, score=0.8, source='vector')
    ]

    keyword_results = [
        SearchResult(id=2, rank=1, score=0.95, source='keyword'),
        SearchResult(id=3, rank=2, score=0.85, source='keyword')
    ]

    # Act
    combined = engine._reciprocal_rank_fusion(vector_results, keyword_results)

    # Assert
    assert len(combined) == 3  # Deduplicated
    assert combined[0].id == 2  # Appeared in both, should rank highest
    assert all(r.source == 'hybrid' for r in combined)


@pytest.mark.asyncio
async def test_search_with_empty_vector_results():
    """Test hybrid search when vector search returns empty"""
    # Arrange - mock vector search to return empty
    engine = create_hybrid_search_engine(mock_database)
    engine._vector_search = AsyncMock(return_value=[])
    engine._keyword_search = AsyncMock(return_value=[
        SearchResult(id=1, rank=1, score=0.9, source='keyword')
    ])

    # Act
    response = await engine.search("test query")

    # Assert
    assert len(response.results) == 1
    assert response.vector_results_count == 0
    assert response.keyword_results_count == 1
```

---

## Knowledge Graph Testing

### Test Coverage Areas

1. **Concept Management** (6 tests)
   - Add concept
   - Update frequency
   - Get concept
   - Delete concept
   - Topic linking

2. **Relationship Management** (6 tests)
   - Create relationship
   - Update strength
   - Different relationship types
   - Bidirectional relationships

3. **Querying** (8 tests)
   - Get related concepts
   - Depth limiting
   - Strength filtering
   - Most discussed concepts
   - Session tracking

4. **Schema Management** (4 tests)
   - Index creation
   - Constraint enforcement
   - Migration handling

5. **Performance** (6 tests)
   - Query latency
   - Batch operations
   - Large graph traversal

### Example Tests

**File:** `tests/knowledge_graph/test_graph_store.py`

```python
import pytest
from app.knowledge_graph.graph_store import KnowledgeGraphStore

@pytest.mark.asyncio
async def test_add_concept(mock_neo4j_driver):
    """Test adding a concept"""
    # Arrange
    graph = KnowledgeGraphStore(mock_config)
    graph.driver = mock_neo4j_driver
    graph._initialized = True

    # Act
    concept_id = await graph.add_concept(
        name="machine learning",
        description="AI subset",
        metadata={"category": "tech"}
    )

    # Assert
    assert concept_id == "machine learning"
    # Verify Neo4j query was called
    mock_neo4j_driver.session.assert_called()


@pytest.mark.asyncio
async def test_get_related_concepts(mock_neo4j_driver):
    """Test getting related concepts"""
    # Arrange
    graph = KnowledgeGraphStore(mock_config)
    graph.driver = mock_neo4j_driver
    graph._initialized = True

    # Act
    related = await graph.get_related_concepts(
        concept="machine learning",
        max_depth=2,
        min_strength=0.5
    )

    # Assert
    assert isinstance(related, list)
    assert len(related) > 0
    assert all('name' in r for r in related)
    assert all('distance' in r for r in related)
```

---

## Integration Testing

### RAG Pipeline Testing

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_rag_pipeline_end_to_end():
    """Test complete RAG pipeline"""
    # 1. Add documents to vector store
    await vector_store.add_batch(...)

    # 2. Execute search
    response = await hybrid_search.search(...)

    # 3. Verify results contain relevant context
    assert len(response.results) > 0

    # 4. Build context (mock)
    context = build_context(response.results)

    # 5. Generate response with Claude (mock)
    rag_response = await generate_rag_response(query, context)

    assert rag_response is not None
```

---

## Performance Benchmarks

```python
@pytest.mark.benchmark
def test_vector_search_latency(benchmark):
    """Benchmark vector search latency"""
    result = benchmark(
        lambda: asyncio.run(vector_store.search_similar(...))
    )

    # Assert < 100ms
    assert result < 0.1


@pytest.mark.benchmark
def test_embedding_throughput(benchmark):
    """Benchmark embedding generation throughput"""
    texts = ["text"] * 100

    result = benchmark(
        lambda: asyncio.run(embedding_generator.generate_batch(texts))
    )

    # Assert > 100 texts/second
    throughput = 100 / result
    assert throughput > 100
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Phase 3 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio

      - name: Run tests with coverage
        run: |
          pytest tests/vector tests/search tests/knowledge_graph \
            --cov=app.vector --cov=app.search --cov=app.knowledge_graph \
            --cov-report=xml --cov-report=html

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

### Running Tests Locally

```bash
# All Phase 3 tests
pytest tests/vector tests/search tests/knowledge_graph -v

# With coverage
pytest tests/vector tests/search tests/knowledge_graph \
  --cov=app.vector --cov=app.search --cov=app.knowledge_graph \
  --cov-report=html

# View coverage report
open htmlcov/index.html

# Run specific test file
pytest tests/vector/test_vector_store.py -v

# Run single test
pytest tests/vector/test_vector_store.py::test_add_embedding_success -v

# Run with benchmarks
pytest --benchmark-only
```

---

## Coverage Report Example

```
Name                                Stmts   Miss  Cover
-------------------------------------------------------
app/vector/vector_store.py            245     18    93%
app/vector/embeddings.py              167     12    93%
app/vector/config.py                   84      5    94%
app/search/hybrid_search.py           234     32    86%
app/search/query_analyzer.py          123     15    88%
app/search/config.py                   67      3    96%
app/knowledge_graph/graph_store.py    312     45    86%
app/knowledge_graph/config.py          45      2    96%
-------------------------------------------------------
TOTAL                                1277    132    90%
```

**Target:** 80%+ coverage ✅

---

**Next:** [Phase 3 Usage Examples](PHASE3_USAGE_EXAMPLES.md)
