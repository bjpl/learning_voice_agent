# Phase 3 Vector API Reference

**Version:** 1.0.0
**Date:** 2025-01-21

Complete API documentation for Phase 3 vector memory components.

## Table of Contents

1. [VectorStore API](#vectorstore-api)
2. [EmbeddingGenerator API](#embeddinggenerator-api)
3. [HybridSearchEngine API](#hybridsearchengine-api)
4. [QueryAnalyzer API](#queryanalyzer-api)
5. [KnowledgeGraphStore API](#knowledgegraphstore-api)
6. [Configuration APIs](#configuration-apis)
7. [Error Handling](#error-handling)

---

## VectorStore API

**Module:** `app.vector.vector_store`

### Class: `VectorStore`

Persistent vector database using ChromaDB.

#### Constructor

```python
VectorStore(config: Optional[VectorConfig] = None)
```

**Parameters:**
- `config` (VectorConfig, optional): Vector configuration. Uses default if None.

**Example:**
```python
from app.vector.vector_store import VectorStore, vector_store

# Use singleton
store = vector_store

# Or create custom instance
from app.vector.config import VectorConfig
custom_config = VectorConfig(similarity_threshold=0.8)
store = VectorStore(custom_config)
```

---

#### `initialize()`

Initialize vector store and load collections.

```python
async def initialize() -> None
```

**Returns:** None

**Raises:**
- `RuntimeError`: If ChromaDB initialization fails
- `ConnectionError`: If persist directory inaccessible

**Example:**
```python
await vector_store.initialize()
```

---

#### `add_embedding()`

Add single embedding to collection.

```python
async def add_embedding(
    collection_name: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    embedding: Optional[np.ndarray] = None
) -> str
```

**Parameters:**
- `collection_name` (str): Target collection name
- `text` (str): Text content to embed and store
- `metadata` (dict, optional): Additional metadata to attach
- `document_id` (str, optional): Document ID (auto-generated if None)
- `embedding` (np.ndarray, optional): Pre-computed embedding (generated if None)

**Returns:** Document ID (str)

**Raises:**
- `ValueError`: If collection not found
- `RuntimeError`: If embedding generation fails

**Example:**
```python
doc_id = await vector_store.add_embedding(
    collection_name="conversations",
    text="Machine learning is a subset of AI",
    metadata={
        "session_id": "sess_123",
        "timestamp": "2025-01-21T10:30:00Z",
        "speaker": "user"
    }
)
# Returns: "a1b2c3d4-5678-90ab-cdef-1234567890ab"
```

---

#### `add_batch()`

Add multiple embeddings in batch (efficient for bulk operations).

```python
async def add_batch(
    collection_name: str,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    document_ids: Optional[List[str]] = None
) -> List[str]
```

**Parameters:**
- `collection_name` (str): Target collection name
- `texts` (List[str]): List of texts to embed
- `metadatas` (List[dict], optional): List of metadata dicts
- `document_ids` (List[str], optional): List of document IDs

**Returns:** List of document IDs

**Example:**
```python
texts = [
    "What is deep learning?",
    "Explain neural networks",
    "Machine learning basics"
]

metadatas = [
    {"session_id": "sess_1", "speaker": "user"},
    {"session_id": "sess_1", "speaker": "user"},
    {"session_id": "sess_2", "speaker": "user"}
]

doc_ids = await vector_store.add_batch(
    collection_name="conversations",
    texts=texts,
    metadatas=metadatas
)
# Returns: ["id1", "id2", "id3"]
```

---

#### `search_similar()`

Search for similar embeddings using semantic similarity.

```python
async def search_similar(
    collection_name: str,
    query_text: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    n_results: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_distances: bool = True
) -> List[Dict[str, Any]]
```

**Parameters:**
- `collection_name` (str): Collection to search
- `query_text` (str, optional): Query text (will be embedded)
- `query_embedding` (np.ndarray, optional): Pre-computed query embedding
- `n_results` (int, optional): Number of results (uses config default if None)
- `metadata_filter` (dict, optional): Filter results by metadata
- `include_distances` (bool): Include similarity distances in results

**Returns:** List of matching documents

**Result Structure:**
```python
[
    {
        "id": "doc_id",
        "document": "Full text content",
        "metadata": {"session_id": "...", ...},
        "similarity": 0.92,  # 0-1 scale (higher = more similar)
        "distance": 0.08     # Raw distance (lower = more similar)
    },
    ...
]
```

**Example:**
```python
# Search by text
results = await vector_store.search_similar(
    collection_name="conversations",
    query_text="How does backpropagation work?",
    n_results=5,
    metadata_filter={"session_id": "sess_123"}
)

for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Text: {result['document']}")
    print(f"Metadata: {result['metadata']}")
```

---

#### `delete_embedding()`

Delete embedding from collection.

```python
async def delete_embedding(
    collection_name: str,
    document_id: str
) -> bool
```

**Parameters:**
- `collection_name` (str): Collection name
- `document_id` (str): Document ID to delete

**Returns:** True if deleted, False if not found

**Example:**
```python
deleted = await vector_store.delete_embedding(
    collection_name="conversations",
    document_id="old_doc_id"
)
```

---

#### `get_collection_stats()`

Get statistics for a collection.

```python
async def get_collection_stats(
    collection_name: str
) -> Dict[str, Any]
```

**Returns:**
```python
{
    "name": "conversations",
    "count": 1234,
    "metadata": {
        "hnsw:space": "cosine",
        ...
    }
}
```

---

#### `create_collection()`

Create a new collection.

```python
async def create_collection(
    name: str,
    metadata_schema: Optional[Dict[str, str]] = None,
    distance_metric: str = "cosine"
) -> None
```

**Parameters:**
- `name` (str): Collection name
- `metadata_schema` (dict, optional): Metadata schema definition
- `distance_metric` (str): Distance metric ("cosine", "l2", or "ip")

**Example:**
```python
await vector_store.create_collection(
    name="knowledge_base",
    metadata_schema={
        "source": "str",
        "category": "str",
        "confidence": "float"
    },
    distance_metric="cosine"
)
```

---

## EmbeddingGenerator API

**Module:** `app.vector.embeddings`

### Class: `EmbeddingGenerator`

Singleton class for generating embeddings using Sentence Transformers.

#### Constructor

```python
EmbeddingGenerator(config: Optional[VectorConfig] = None)
```

Automatically returns singleton instance.

---

#### `generate_embedding()`

Generate embedding for single text.

```python
async def generate_embedding(
    text: str,
    use_cache: bool = True
) -> np.ndarray
```

**Parameters:**
- `text` (str): Input text to embed
- `use_cache` (bool): Whether to use cached results

**Returns:** Embedding vector (np.ndarray of shape `(384,)`)

**Example:**
```python
from app.vector.embeddings import embedding_generator

await embedding_generator.initialize()

embedding = await embedding_generator.generate_embedding(
    "Machine learning is fascinating"
)
# Returns: array([0.123, -0.456, 0.789, ...])  # 384 dimensions
```

---

#### `generate_batch()`

Generate embeddings for multiple texts efficiently.

```python
async def generate_batch(
    texts: List[str],
    batch_size: Optional[int] = None,
    show_progress: bool = False
) -> List[np.ndarray]
```

**Parameters:**
- `texts` (List[str]): List of texts to embed
- `batch_size` (int, optional): Override default batch size
- `show_progress` (bool): Show progress bar for large batches

**Returns:** List of embedding vectors

**Example:**
```python
texts = ["text1", "text2", "text3"]
embeddings = await embedding_generator.generate_batch(texts)
# Returns: [array(...), array(...), array(...)]
```

---

#### `get_model_info()`

Get information about loaded model.

```python
def get_model_info() -> Dict[str, Any]
```

**Returns:**
```python
{
    "status": "initialized",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "max_sequence_length": 256,
    "device": "cpu",
    "cache": {
        "size": 245,
        "max_size": 1000,
        "ttl_seconds": 3600
    }
}
```

---

#### `clear_cache()`

Clear embedding cache.

```python
def clear_cache() -> None
```

---

## HybridSearchEngine API

**Module:** `app.search.hybrid_search`

### Class: `HybridSearchEngine`

Combines vector and keyword search using Reciprocal Rank Fusion.

#### `search()`

Execute hybrid search.

```python
async def search(
    query: str,
    strategy: Optional[SearchStrategy] = None,
    limit: Optional[int] = None
) -> HybridSearchResponse
```

**Parameters:**
- `query` (str): Search query string
- `strategy` (SearchStrategy, optional): Search strategy (None = adaptive)
- `limit` (int, optional): Maximum results to return

**Returns:** `HybridSearchResponse`

**Response Structure:**
```python
{
    "query": "machine learning",
    "strategy": "hybrid",
    "results": [
        {
            "id": 123,
            "session_id": "sess_abc",
            "timestamp": "2025-01-21T10:30:00Z",
            "user_text": "What is machine learning?",
            "agent_text": "Machine learning is...",
            "score": 0.95,
            "rank": 1,
            "source": "hybrid",
            "vector_score": 0.92,
            "keyword_score": 0.87,
            "user_snippet": "What is <mark>machine</mark> <mark>learning</mark>?",
            "agent_snippet": "<mark>Machine</mark> <mark>learning</mark> is..."
        },
        ...
    ],
    "total_count": 10,
    "query_analysis": {
        "intent": "conceptual",
        "suggested_strategy": "semantic",
        "keywords": ["machine", "learning"]
    },
    "execution_time_ms": 156.23,
    "vector_results_count": 8,
    "keyword_results_count": 7
}
```

**Example:**
```python
from app.search.hybrid_search import create_hybrid_search_engine
from app.search.config import SearchStrategy
from app.database import database

engine = create_hybrid_search_engine(database)
engine.set_embedding_client(openai_client)

# Adaptive search (auto-select strategy)
response = await engine.search("machine learning")

# Force hybrid search
response = await engine.search(
    query="neural networks",
    strategy=SearchStrategy.HYBRID,
    limit=10
)

# Semantic-only search
response = await engine.search(
    query="explain backpropagation",
    strategy=SearchStrategy.SEMANTIC
)

# Keyword-only search
response = await engine.search(
    query="\"exact phrase match\"",
    strategy=SearchStrategy.KEYWORD
)
```

---

## QueryAnalyzer API

**Module:** `app.search.query_analyzer`

### Class: `QueryAnalyzer`

Analyzes queries to optimize search strategy.

#### `analyze()`

Analyze query and suggest search strategy.

```python
async def analyze(query: str) -> QueryAnalysis
```

**Parameters:**
- `query` (str): User's search query

**Returns:** `QueryAnalysis`

**QueryAnalysis Structure:**
```python
{
    "original_query": "What is machine learning?",
    "cleaned_query": "what is machine learning",
    "keywords": ["machine", "learning"],
    "intent": "conceptual",
    "suggested_strategy": SearchStrategy.SEMANTIC,
    "is_short": False,
    "is_exact_phrase": False,
    "word_count": 4
}
```

**Example:**
```python
from app.search.query_analyzer import query_analyzer

analysis = await query_analyzer.analyze("How does neural network work?")

print(analysis.intent)              # "conceptual"
print(analysis.suggested_strategy)   # SearchStrategy.SEMANTIC
print(analysis.keywords)            # ["neural", "network", "work"]
```

---

## KnowledgeGraphStore API

**Module:** `app.knowledge_graph.graph_store`

### Class: `KnowledgeGraphStore`

Neo4j-backed knowledge graph for concept relationships.

#### `add_concept()`

Add or update concept node.

```python
async def add_concept(
    name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None
) -> str
```

**Parameters:**
- `name` (str): Concept name (unique identifier)
- `description` (str, optional): Optional description
- `metadata` (dict, optional): Additional properties
- `topic` (str, optional): Optional parent topic

**Returns:** Concept name (ID)

**Example:**
```python
from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.knowledge_graph.config import KnowledgeGraphConfig

config = KnowledgeGraphConfig()
graph = KnowledgeGraphStore(config)
await graph.initialize()

concept_id = await graph.add_concept(
    name="neural networks",
    description="Artificial neural networks for machine learning",
    metadata={"category": "technology", "difficulty": "intermediate"},
    topic="machine learning"
)
```

---

#### `add_relationship()`

Create or update relationship between concepts.

```python
async def add_relationship(
    from_concept: str,
    to_concept: str,
    relationship_type: str = "RELATES_TO",
    strength: float = 1.0,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `from_concept` (str): Source concept name
- `to_concept` (str): Target concept name
- `relationship_type` (str): Type of relationship
- `strength` (float): Relationship strength (0.0-1.0)
- `context` (str, optional): Context where relationship observed
- `metadata` (dict, optional): Additional properties

**Relationship Types:**
- `RELATES_TO`: General relationship
- `BUILDS_ON`: Prerequisite relationship
- `MENTIONED_IN`: Mentioned in session
- `INSTANCE_OF`: Entity is instance of concept
- `CONTAINS`: Topic contains concept

**Example:**
```python
await graph.add_relationship(
    from_concept="neural networks",
    to_concept="machine learning",
    relationship_type="BUILDS_ON",
    strength=0.9,
    context="Prerequisites for deep learning"
)
```

---

#### `get_related_concepts()`

Get concepts related to given concept.

```python
async def get_related_concepts(
    concept: str,
    max_depth: int = 2,
    min_strength: float = 0.3,
    limit: int = 20
) -> List[Dict[str, Any]]
```

**Parameters:**
- `concept` (str): Starting concept name
- `max_depth` (int): Maximum relationship depth
- `min_strength` (float): Minimum relationship strength
- `limit` (int): Maximum results

**Returns:** List of related concepts

**Result Structure:**
```python
[
    {
        "name": "deep learning",
        "description": "Neural networks with multiple layers",
        "frequency": 15,
        "relationship_types": ["BUILDS_ON", "RELATES_TO"],
        "strengths": [0.9, 0.7],
        "distance": 1
    },
    ...
]
```

**Example:**
```python
related = await graph.get_related_concepts(
    concept="machine learning",
    max_depth=2,
    min_strength=0.5,
    limit=10
)

for concept in related:
    print(f"{concept['name']} (freq: {concept['frequency']}, dist: {concept['distance']})")
```

---

#### `get_most_discussed_concepts()`

Get most frequently discussed concepts.

```python
async def get_most_discussed_concepts(
    limit: int = 10
) -> List[Dict[str, Any]]
```

**Returns:**
```python
[
    {
        "name": "machine learning",
        "description": "...",
        "frequency": 42,
        "last_seen": datetime(2025, 1, 21, 10, 30)
    },
    ...
]
```

---

#### `get_graph_stats()`

Get graph statistics.

```python
async def get_graph_stats() -> Dict[str, Any]
```

**Returns:**
```python
{
    "concepts": 245,
    "relationships": 812,
    "entities": 67,
    "sessions": 153,
    "topics": 12
}
```

---

## Configuration APIs

### VectorConfig

```python
from app.vector.config import VectorConfig, vector_config

# Access global config
config = vector_config

# Get collection config
conv_config = config.get_collection_config("conversations")

# Add new collection
from app.vector.config import CollectionConfig
new_collection = CollectionConfig(
    name="custom",
    distance_metric="cosine",
    metadata_schema={"field": "str"}
)
config.add_collection(new_collection)

# Export config as dict
config_dict = config.to_dict()
```

### RAGConfig

```python
from app.rag.config import rag_config, update_rag_config

# Access config
print(rag_config.retrieval_top_k)  # 5

# Update config
update_rag_config(
    retrieval_top_k=10,
    relevance_threshold=0.8
)

# Use performance profile
from app.rag.config import get_performance_profile
fast_settings = get_performance_profile("fast")
update_rag_config(**fast_settings)
```

---

## Error Handling

### Common Exceptions

#### VectorStore Errors

```python
try:
    await vector_store.add_embedding(...)
except ValueError as e:
    # Collection not found
    print(f"Collection error: {e}")
except RuntimeError as e:
    # Embedding generation failed
    print(f"Embedding error: {e}")
```

#### Search Errors

```python
try:
    response = await engine.search(query)
except Exception as e:
    # Graceful degradation - returns empty results
    print(f"Search failed: {e}")
    # response.results will be []
```

#### Knowledge Graph Errors

```python
from neo4j.exceptions import ServiceUnavailable

try:
    await graph.initialize()
except ServiceUnavailable:
    print("Neo4j not available")
except Exception as e:
    print(f"Graph initialization failed: {e}")
```

### Retry Behavior

All APIs use `@with_retry` decorator for automatic retry on transient failures:

- **Max attempts**: 3
- **Initial wait**: 0.5s
- **Exponential backoff**: 2x per retry

```python
# Automatically retried up to 3 times
await vector_store.add_embedding(...)
```

---

## Code Examples

### Complete Vector Search Example

```python
from app.vector.vector_store import vector_store
from app.vector.embeddings import embedding_generator

# Initialize
await vector_store.initialize()
await embedding_generator.initialize()

# Add documents
doc_ids = await vector_store.add_batch(
    collection_name="conversations",
    texts=[
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing analyzes text"
    ],
    metadatas=[
        {"session_id": "s1", "speaker": "agent"},
        {"session_id": "s1", "speaker": "agent"},
        {"session_id": "s2", "speaker": "agent"}
    ]
)

# Search
results = await vector_store.search_similar(
    collection_name="conversations",
    query_text="What is deep learning?",
    n_results=5
)

for result in results:
    print(f"[{result['similarity']:.3f}] {result['document']}")
```

### Complete Hybrid Search Example

```python
from app.search.hybrid_search import create_hybrid_search_engine
from app.database import database

engine = create_hybrid_search_engine(database)
engine.set_embedding_client(openai_client)

response = await engine.search("machine learning concepts")

print(f"Found {response.total_count} results in {response.execution_time_ms}ms")
print(f"Strategy: {response.strategy}")

for result in response.results[:3]:
    print(f"\n[Rank {result['rank']}] Score: {result['score']:.3f}")
    print(f"Source: {result['source']}")
    print(f"Text: {result['user_text']}")
```

---

**Next:** [Phase 3 Testing Guide](PHASE3_TESTING_GUIDE.md)
