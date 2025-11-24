# Vector Database Module

High-performance semantic search and embedding storage for the learning_voice_agent v2.0 system.

## Quick Start

```python
from app.vector import VectorStore

# Initialize
store = VectorStore()
await store.initialize()

# Add embedding
await store.add_embedding(
    collection_name="conversations",
    text="Your text here",
    metadata={"session_id": "123"}
)

# Search
results = await store.search_similar(
    collection_name="conversations",
    query_text="Your query",
    n_results=5
)
```

## Components

### VectorStore
ChromaDB-based persistent vector storage with async support.

**Key Methods:**
- `add_embedding(collection, text, metadata)`: Add single embedding
- `add_batch(collection, texts, metadatas)`: Batch insertion
- `search_similar(collection, query, n_results)`: Semantic search
- `get_collection_stats(collection)`: Collection metadata

### EmbeddingGenerator
Sentence-transformers based embedding generation with caching.

**Key Methods:**
- `generate_embedding(text)`: Single text embedding
- `generate_batch(texts)`: Batch generation
- `get_model_info()`: Model information
- `clear_cache()`: Cache management

### VectorConfig
Centralized configuration for all vector components.

**Pre-configured Collections:**
- `conversations`: Conversation embeddings
- `knowledge`: Knowledge base entries
- `user_memories`: User-specific memories

## Features

✅ **Persistent Storage**: Data survives restarts
✅ **Async Operations**: Non-blocking I/O
✅ **Batch Processing**: Efficient bulk operations
✅ **Smart Caching**: LRU cache with TTL
✅ **Retry Logic**: Automatic failure recovery
✅ **Metadata Filtering**: Precise queries
✅ **Multi-Collection**: Organize by use case

## Configuration

```python
from app.vector.config import VectorConfig, EmbeddingModelConfig

config = VectorConfig(
    persist_directory=Path("./data/chromadb"),
    embedding_model=EmbeddingModelConfig(
        model_name="all-MiniLM-L6-v2",
        device="cpu",  # or "cuda"
        batch_size=32,
        cache_size=1000
    )
)
```

## Technical Specs

- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Backend**: ChromaDB 0.4.22
- **Distance**: Cosine similarity
- **Cache**: 1000 items, 1-hour TTL
- **Storage**: Persistent disk-based

## Testing

```bash
pytest tests/test_vector_integration.py -v
```

## Dependencies

```bash
pip install chromadb==0.4.22 sentence-transformers==2.3.1 torch==2.1.2
```

## Performance

- **Embedding**: ~10ms per text (cached: <1ms)
- **Batch**: 32 texts in ~50ms
- **Search**: <100ms for 10k vectors
- **Memory**: ~200MB model + 4KB per vector

## Examples

### Conversation Search

```python
# Add conversation history
await store.add_batch(
    collection_name="conversations",
    texts=["Hello", "How are you?", "I'm fine"],
    metadatas=[
        {"session_id": "abc", "speaker": "user"},
        {"session_id": "abc", "speaker": "agent"},
        {"session_id": "abc", "speaker": "user"}
    ]
)

# Find similar exchanges
results = await store.search_similar(
    collection_name="conversations",
    query_text="Hi there",
    n_results=5,
    metadata_filter={"session_id": "abc"}
)
```

### Knowledge Base

```python
# Create custom collection
await store.create_collection(
    name="knowledge",
    metadata_schema={"source": "str", "category": "str"}
)

# Add knowledge
await store.add_embedding(
    collection_name="knowledge",
    text="Vector databases enable semantic search",
    metadata={"source": "docs", "category": "database"}
)
```

## Architecture

```
VectorStore
├── ChromaDB Client (persistent)
├── EmbeddingGenerator (singleton)
│   ├── SentenceTransformer Model
│   └── EmbeddingCache (LRU + TTL)
└── VectorConfig (configuration)
```

## Best Practices

1. **Initialize Once**: Use singleton pattern
2. **Batch When Possible**: 10-100x faster than individual adds
3. **Filter Metadata**: Reduce search space
4. **Monitor Cache**: Track hit rates
5. **Clean Up**: Delete old collections
6. **Set Thresholds**: Filter low-similarity results

## Troubleshooting

**Issue**: Slow first query
**Solution**: Model loading is lazy; expected on first use

**Issue**: High memory usage
**Solution**: Reduce cache size or clear periodically

**Issue**: Search returns no results
**Solution**: Check similarity_threshold (default: 0.7)

**Issue**: Import errors
**Solution**: Install dependencies: `pip install -r requirements.txt`

## Logging

All operations log via structlog:

```python
db_logger.info("embedding_added", collection="conversations", document_id="abc123")
db_logger.error("search_failed", collection="conversations", error="...")
```

## License

Part of the learning_voice_agent v2.0 system.
