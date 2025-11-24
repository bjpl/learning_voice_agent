# Phase 3: Vector Memory & RAG Architecture

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Architecture Design - Ready for Review
**Author:** System Architect
**SPARC Phase:** Specification → Architecture

---

## Executive Summary

### Objectives

Phase 3 transforms the learning_voice_agent from keyword-only search to **semantic memory with retrieval-augmented generation (RAG)**, enabling:

- **Semantic understanding** through vector embeddings
- **Hybrid search** combining vector similarity + keyword matching
- **Knowledge graph** for concept relationships
- **RAG-powered responses** with source citations
- **Intelligent retrieval** integrated with Phase 2 multi-agent system

### Key Design Decisions & Justifications

| Decision Area | Recommendation | Justification |
|--------------|----------------|---------------|
| **Vector Database** | ChromaDB (embedded) | ✅ Zero external API costs<br>✅ Embedded in Python app<br>✅ Railway-friendly single instance<br>✅ 10M+ vectors supported<br>❌ No horizontal scaling (acceptable for single-user) |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 | ✅ Fast inference (~50ms)<br>✅ Small footprint (384 dims)<br>✅ Good quality (semantic similarity)<br>✅ Free/open-source<br>❌ Not SOTA (acceptable for v2.0) |
| **Hybrid Search** | 70% vector + 30% keyword (BM25) | ✅ Best of both worlds<br>✅ Handles exact matches + semantic<br>✅ Tunable weights<br>✅ Proven pattern (Algolia, Elastic) |
| **Knowledge Graph** | Neo4j Community (embedded) | ✅ Free embedded version<br>✅ Powerful graph queries<br>✅ Railway-compatible<br>✅ Export to Aura later if needed<br>❌ Single-instance only (fine for now) |
| **RAG Strategy** | Basic retrieval + simple re-ranking | ✅ Avoid overengineering<br>✅ Proven pattern<br>✅ Fast (<100ms retrieval)<br>✅ Can enhance later with HyDE, multi-query, etc. |

### Success Criteria

- ✅ Semantic search returns relevant results (>85% precision@5)
- ✅ Hybrid search outperforms keyword-only (measured A/B test)
- ✅ RAG responses include accurate citations
- ✅ Retrieval latency <100ms (P95)
- ✅ Knowledge graph visualizable in UI
- ✅ Zero data loss during migration from FTS5-only

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Vector Database Design](#vector-database-design)
3. [Embedding Pipeline](#embedding-pipeline)
4. [Hybrid Search System](#hybrid-search-system)
5. [Knowledge Graph Schema](#knowledge-graph-schema)
6. [RAG System Design](#rag-system-design)
7. [Integration with Phase 2 Agents](#integration-with-phase-2-agents)
8. [Performance Targets](#performance-targets)
9. [Migration Strategy](#migration-strategy)
10. [Implementation Plan](#implementation-plan)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                               │
│  Web (Next.js) │ Mobile (RN) │ Phone (Twilio) │ API (FastAPI)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: MULTI-AGENT ORCHESTRATOR                  │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │Conversation │  Analysis   │  Research   │  Synthesis  │     │
│  │   Agent     │   Agent     │   Agent     │   Agent     │     │
│  └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘     │
│         │             │             │             │            │
│         └─────────────┴─────────────┴─────────────┘            │
│                           │                                    │
│                    [PHASE 3 INTEGRATION]                       │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 3: VECTOR MEMORY LAYER                    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │             RAG Orchestrator                          │     │
│  │  • Query Processing                                   │     │
│  │  • Hybrid Search (Vector + Keyword)                   │     │
│  │  • Context Assembly                                   │     │
│  │  • Re-ranking & Filtering                             │     │
│  └────────────────────┬──────────────────────────────────┘     │
│                       │                                        │
│       ┌───────────────┼───────────────┬────────────────┐       │
│       ▼               ▼               ▼                ▼       │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ChromaDB │   │  Neo4j   │   │  Redis   │   │ SQLite   │    │
│  │         │   │          │   │          │   │  FTS5    │    │
│  │ Vector  │   │Knowledge │   │  Cache   │   │ Archive  │    │
│  │ Search  │   │  Graph   │   │ Embed.   │   │ Backup   │    │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘    │
└─────────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING PIPELINE                             │
│                                                                 │
│  Text Input → Preprocessing → Model Inference → Normalization  │
│     │              │                │                │          │
│  Clean/         Tokenize      all-MiniLM-L6-v2   L2 Norm       │
│  Chunk          (512 tokens)     (384 dims)     (unit vector)  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   PHASE 3 DATA FLOW                              │
└──────────────────────────────────────────────────────────────────┘

1. INGESTION FLOW (Conversation → Storage)
   ────────────────────────────────────────

   User Message
        │
        ▼
   ConversationAgent (Phase 2)
        │
        ├─→ AnalysisAgent → Concepts/Entities
        │
        ▼
   [PHASE 3 INGESTION]
        │
        ├─→ Embedding Pipeline
        │   ├─→ Text Chunking (512 tokens)
        │   ├─→ all-MiniLM-L6-v2 (50ms)
        │   └─→ Normalized vector (384 dims)
        │
        ├─→ ChromaDB Store
        │   ├─→ Vector index (HNSW)
        │   ├─→ Metadata {session, timestamp, concepts}
        │   └─→ Document text
        │
        ├─→ Neo4j Graph Store
        │   ├─→ Create/Update nodes (Capture, Concept, Entity)
        │   ├─→ Create relationships (MENTIONS, RELATED_TO)
        │   └─→ Update graph statistics
        │
        └─→ SQLite FTS5 (backup/archive)
            └─→ Full-text index (existing)


2. RETRIEVAL FLOW (Query → Results)
   ─────────────────────────────────

   User Query: "Tell me about neural networks"
        │
        ▼
   [PHASE 3 RAG ORCHESTRATOR]
        │
        ├─→ Query Processing
        │   ├─→ Intent detection (semantic search)
        │   ├─→ Query expansion (synonyms)
        │   └─→ Embedding generation (50ms)
        │
        ├─→ Parallel Retrieval (asyncio.gather)
        │   │
        │   ├─→ Vector Search (ChromaDB)
        │   │   ├─→ Similarity search (cosine)
        │   │   ├─→ Top 10 results
        │   │   └─→ Filter by threshold (>0.7)
        │   │
        │   ├─→ Keyword Search (SQLite FTS5)
        │   │   ├─→ BM25 ranking
        │   │   ├─→ Top 10 results
        │   │   └─→ Extract snippets
        │   │
        │   └─→ Graph Traversal (Neo4j)
        │       ├─→ Find related concepts
        │       ├─→ Traverse relationships
        │       └─→ Return connected captures
        │
        ├─→ Hybrid Fusion (30ms)
        │   ├─→ Reciprocal Rank Fusion (RRF)
        │   ├─→ Weight: 70% vector, 30% keyword
        │   └─→ De-duplicate by capture_id
        │
        ├─→ Re-ranking (20ms)
        │   ├─→ Recency boost (decay factor)
        │   ├─→ Concept relevance boost
        │   └─→ Final top-k selection (k=5)
        │
        └─→ Context Assembly
            ├─→ Format for LLM context
            ├─→ Add citations/metadata
            └─→ Return to ConversationAgent
                │
                ▼
   ConversationAgent generates RAG-powered response
        │
        ▼
   Response with citations: "Based on your previous discussions
   about neural networks [1][2], transformers use attention..."


3. KNOWLEDGE GRAPH UPDATE FLOW
   ────────────────────────────

   AnalysisAgent extracts: ["neural networks", "transformers", "attention"]
        │
        ▼
   [NEO4J GRAPH UPDATE]
        │
        ├─→ Create/Merge Concept Nodes
        │   MERGE (c1:Concept {name: "neural networks"})
        │   MERGE (c2:Concept {name: "transformers"})
        │   MERGE (c3:Concept {name: "attention"})
        │
        ├─→ Create Capture Node
        │   CREATE (cap:Capture {
        │     id: "cap_123",
        │     text: "...",
        │     timestamp: "2025-11-21T10:30:00Z",
        │     session_id: "sess_abc"
        │   })
        │
        ├─→ Create Relationships
        │   (cap)-[:MENTIONS {weight: 0.9}]->(c1)
        │   (cap)-[:MENTIONS {weight: 0.8}]->(c2)
        │   (cap)-[:MENTIONS {weight: 0.7}]->(c3)
        │   (c2)-[:USES]->(c3)  # transformers use attention
        │
        └─→ Update Statistics
            SET c1.mention_count = c1.mention_count + 1
            SET c1.last_mentioned = timestamp()
```

---

## Vector Database Design

### ChromaDB Configuration

**Why ChromaDB:**
1. **Embedded**: Runs in same Python process (no external service)
2. **Persistent**: Disk-backed storage on Railway
3. **Fast**: HNSW indexing for sub-100ms queries
4. **Metadata filtering**: Query by session, date, concepts
5. **Battle-tested**: Used by LangChain, LlamaIndex

**Architecture:**
```python
# Configuration
chroma_client = chromadb.PersistentClient(
    path="/app/data/chroma",  # Railway persistent volume
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=False,
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(
    name="learning_captures",
    metadata={
        "hnsw:space": "cosine",           # Cosine similarity
        "hnsw:construction_ef": 200,      # Build quality (higher = better)
        "hnsw:search_ef": 100,            # Search quality
        "hnsw:M": 16                      # Graph connectivity
    },
    embedding_function=None  # We generate embeddings separately
)
```

### Collection Schema

**Primary Collection: `learning_captures`**

```python
# Document structure in ChromaDB
{
    # Core fields
    "ids": ["cap_uuid_123", "cap_uuid_456", ...],

    # Vector embeddings (384 dimensions from all-MiniLM-L6-v2)
    "embeddings": [
        [0.023, -0.145, 0.567, ...],  # 384 floats, L2-normalized
        [0.112, -0.023, 0.334, ...],
        ...
    ],

    # Original text (for display/re-ranking)
    "documents": [
        "I'm learning about neural networks and how they process data...",
        "Transformers use attention mechanisms to focus on relevant parts...",
        ...
    ],

    # Rich metadata for filtering
    "metadatas": [
        {
            "session_id": "sess_abc123",
            "timestamp": "2025-11-21T10:30:00Z",
            "user_text": "Tell me about neural networks",
            "agent_text": "Neural networks are...",
            "concepts": ["neural networks", "deep learning"],  # From AnalysisAgent
            "entities": ["backpropagation", "gradient descent"],
            "capture_type": "conversation",  # conversation | insight | summary
            "agent_involved": ["conversation", "analysis"],
            "importance_score": 0.85,  # Calculated by SynthesisAgent
            "word_count": 127,
            "has_code": false,
            "language": "en"
        },
        ...
    ]
}
```

### Query Patterns

**1. Semantic Similarity Search**
```python
results = collection.query(
    query_embeddings=[query_vector],  # 384-dim vector
    n_results=10,
    where={
        "session_id": {"$eq": "sess_abc123"},  # Filter by session
        "timestamp": {"$gte": "2025-11-01T00:00:00Z"}  # Recent only
    },
    include=["embeddings", "documents", "metadatas", "distances"]
)

# Returns:
# {
#   'ids': [['cap_1', 'cap_2', ...]],
#   'distances': [[0.23, 0.35, ...]],  # Cosine distance (0=identical)
#   'documents': [['text1', 'text2', ...]],
#   'metadatas': [[{...}, {...}, ...]]
# }
```

**2. Concept-Based Retrieval**
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=20,
    where={
        "concepts": {"$in": ["neural networks", "deep learning"]}
    }
)
```

**3. Multi-Modal Filtering**
```python
# Find important conversations about specific topics from last week
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={
        "$and": [
            {"capture_type": {"$eq": "conversation"}},
            {"importance_score": {"$gte": 0.7}},
            {"timestamp": {"$gte": seven_days_ago}},
            {"concepts": {"$contains": "transformers"}}
        ]
    }
)
```

### Performance Optimizations

**HNSW Index Tuning:**
- **M=16**: 16 bidirectional links per node (balance speed/recall)
- **efConstruction=200**: High build quality (one-time cost)
- **efSearch=100**: Fast search with good recall (>95%)

**Expected Performance:**
- **Indexing**: ~5ms per document (384-dim vector)
- **Search**: <50ms for k=10 (10K documents)
- **Search**: <100ms for k=10 (100K documents)
- **Memory**: ~1.5KB per document (vector + metadata)

**Scaling Limits (Single Instance):**
- **10K documents**: ~15MB RAM, <50ms search
- **100K documents**: ~150MB RAM, <100ms search
- **1M documents**: ~1.5GB RAM, <200ms search ✅ Single-user target
- **10M+ documents**: Needs distributed ChromaDB (future)

---

## Embedding Pipeline

### Model Selection: sentence-transformers/all-MiniLM-L6-v2

**Specifications:**
- **Dimensions**: 384 (vs 768 for mpnet-base-v2)
- **Parameters**: 22.7M (vs 109M for mpnet)
- **Speed**: ~50ms per sentence (CPU) | ~10ms (GPU)
- **Quality**: Semantic Textual Similarity score 68.06% (vs 69.57% mpnet)
- **Context**: 512 tokens max

**Trade-off Analysis:**

| Metric | all-MiniLM-L6-v2 | all-mpnet-base-v2 | Winner |
|--------|------------------|-------------------|--------|
| Speed | 50ms | 150ms | ✅ MiniLM (3x faster) |
| Quality | 68.06% STS | 69.57% STS | mpnet (+1.5%) |
| Memory | 90MB model | 420MB model | ✅ MiniLM (4.6x smaller) |
| Dimensions | 384 | 768 | ✅ MiniLM (2x less storage) |
| Railway fit | ✅ Excellent | ⚠️ Workable | ✅ MiniLM |

**Recommendation**: Use all-MiniLM-L6-v2 for v2.0
- Quality difference negligible for single-user learning (<2%)
- Speed matters for real-time conversations
- Can swap models later without code changes

### Pipeline Architecture

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class EmbeddingPipeline:
    """
    SPARC: Architecture - Embedding generation pipeline
    PATTERN: Batch processing with caching
    WHY: Optimize for both speed and quality
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load model once at startup
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512  # Token limit

        # Cache for frequently accessed embeddings
        self.cache = {}  # {text_hash: embedding}
        self.cache_size = 1000  # LRU cache

    async def embed_text(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Input text (max 512 tokens)
            normalize: L2 normalize to unit vector

        Returns:
            384-dim numpy array
        """
        # Check cache
        text_hash = hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]

        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        # Cache result
        self._update_cache(text_hash, embedding)

        return embedding

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batched for efficiency)

        Args:
            texts: List of input texts
            batch_size: Batch size for GPU processing
            normalize: L2 normalize to unit vectors

        Returns:
            Array of shape (N, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return embeddings

    def chunk_text(
        self,
        text: str,
        max_tokens: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Split long text into overlapping chunks

        Args:
            text: Input text (potentially >512 tokens)
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks

        Returns:
            List of text chunks
        """
        # Simple word-based chunking (can enhance with NLTK later)
        words = text.split()
        chunks = []

        stride = max_tokens - overlap
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i:i + max_tokens])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def _update_cache(self, key: str, value: np.ndarray) -> None:
        """Update LRU cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
```

### Text Preprocessing

```python
class TextPreprocessor:
    """
    SPARC: Refinement - Text cleaning for better embeddings
    PATTERN: Pipeline of transformations
    WHY: Clean text → better semantic understanding
    """

    def __init__(self):
        self.stopwords = set(['the', 'a', 'an', 'in', 'on', ...])  # Optional

    def preprocess(self, text: str, preserve_meaning: bool = True) -> str:
        """
        Clean and normalize text for embedding

        Args:
            text: Raw input text
            preserve_meaning: Keep stopwords/punctuation for semantics

        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Lowercase (optional - model handles case)
        # text = text.lower()  # Don't lowercase - model is case-aware

        # Remove URLs (optional - depends on use case)
        # text = re.sub(r'http\S+', '', text)

        # Keep punctuation/stopwords for semantic meaning
        # Aggressive cleaning can hurt embedding quality!

        return text
```

### Integration with Ingestion Flow

```python
class VectorIngestionService:
    """
    SPARC: Completion - End-to-end ingestion pipeline
    PATTERN: Async pipeline with error handling
    WHY: Reliable storage with audit trail
    """

    def __init__(
        self,
        chroma_collection,
        embedding_pipeline: EmbeddingPipeline,
        preprocessor: TextPreprocessor
    ):
        self.chroma = chroma_collection
        self.embedder = embedding_pipeline
        self.preprocessor = preprocessor
        self.logger = get_logger("vector_ingestion")

    async def ingest_conversation(
        self,
        capture_id: str,
        user_text: str,
        agent_text: str,
        session_id: str,
        concepts: List[str],
        entities: List[str],
        metadata: dict
    ) -> bool:
        """
        Ingest a conversation turn into vector store

        Returns:
            True if successful, False otherwise
        """
        try:
            # Combine user + agent text for embedding
            combined_text = f"User: {user_text}\nAgent: {agent_text}"

            # Preprocess
            clean_text = self.preprocessor.preprocess(combined_text)

            # Generate embedding
            embedding = await self.embedder.embed_text(clean_text)

            # Prepare metadata
            full_metadata = {
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_text": user_text,
                "agent_text": agent_text,
                "concepts": concepts,
                "entities": entities,
                "capture_type": "conversation",
                "word_count": len(combined_text.split()),
                **metadata  # Additional metadata
            }

            # Store in ChromaDB
            self.chroma.add(
                ids=[capture_id],
                embeddings=[embedding.tolist()],
                documents=[combined_text],
                metadatas=[full_metadata]
            )

            self.logger.info(
                "conversation_ingested",
                capture_id=capture_id,
                session_id=session_id,
                concepts=concepts
            )

            return True

        except Exception as e:
            self.logger.error(
                "ingestion_failed",
                capture_id=capture_id,
                error=str(e),
                exc_info=True
            )
            return False
```

---

## Hybrid Search System

### Search Architecture

**Strategy**: Combine vector similarity (semantic) with BM25 keyword matching (lexical)

**Why Hybrid?**
- Vector search: Good for "neural networks" → "deep learning" (synonyms)
- Keyword search: Good for exact terms like "BERT", "GPT-4", names
- Together: Best recall and precision

### Reciprocal Rank Fusion (RRF)

**Algorithm:**
```python
class HybridSearchFusion:
    """
    SPARC: Architecture - Hybrid search with RRF fusion
    PATTERN: Reciprocal Rank Fusion (RRF)
    WHY: Proven better than simple score averaging

    RRF Formula:
        score(doc) = sum(1 / (k + rank_i))
        where rank_i is rank in result list i
              k = 60 (constant)
    """

    def __init__(
        self,
        chroma_collection,
        sqlite_fts_db,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60
    ):
        self.chroma = chroma_collection
        self.fts = sqlite_fts_db
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.logger = get_logger("hybrid_search")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector + keyword

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (session_id, date range, etc.)

        Returns:
            Ranked list of SearchResult objects
        """
        # 1. Parallel retrieval
        vector_task = self._vector_search(query, top_k=20, filters=filters)
        keyword_task = self._keyword_search(query, top_k=20, filters=filters)

        vector_results, keyword_results = await asyncio.gather(
            vector_task,
            keyword_task
        )

        # 2. Reciprocal Rank Fusion
        fused_scores = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results
        )

        # 3. Re-rank and filter
        final_results = self._rerank(fused_scores, query)

        # 4. Return top-k
        return final_results[:top_k]

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict]
    ) -> List[Tuple[str, float]]:
        """
        Semantic vector search via ChromaDB

        Returns:
            List of (doc_id, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = await self.embedder.embed_text(query)

        # Search ChromaDB
        results = self.chroma.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
            include=["distances", "metadatas", "documents"]
        )

        # Convert distances to similarity scores
        # ChromaDB returns cosine distance (0=identical, 2=opposite)
        # Convert to similarity: 1 - (distance / 2)
        doc_scores = []
        for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
            similarity = 1 - (distance / 2)  # [0, 1] range
            doc_scores.append((doc_id, similarity))

        return doc_scores

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict]
    ) -> List[Tuple[str, float]]:
        """
        BM25 keyword search via SQLite FTS5

        Returns:
            List of (doc_id, bm25_score) tuples
        """
        # Build FTS5 query
        # Convert "neural networks" → "neural OR networks"
        fts_query = " OR ".join(query.split())

        # Execute FTS5 search
        sql = """
            SELECT
                id,
                bm25(captures_fts) as score,
                snippet(captures_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
            FROM captures_fts
            WHERE captures_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """

        results = await self.fts.fetch_all(
            sql,
            [fts_query, top_k]
        )

        # Normalize BM25 scores to [0, 1]
        if results:
            max_score = max(r['score'] for r in results)
            doc_scores = [
                (r['id'], r['score'] / max_score)
                for r in results
            ]
        else:
            doc_scores = []

        return doc_scores

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Fuse results using Reciprocal Rank Fusion

        RRF: score(doc) = sum(weight_i / (k + rank_i))

        Args:
            vector_results: (doc_id, score) from vector search
            keyword_results: (doc_id, score) from keyword search

        Returns:
            {doc_id: fused_score}
        """
        fused = {}

        # Process vector results
        for rank, (doc_id, score) in enumerate(vector_results, start=1):
            rrf_score = self.vector_weight / (self.rrf_k + rank)
            fused[doc_id] = fused.get(doc_id, 0) + rrf_score

        # Process keyword results
        for rank, (doc_id, score) in enumerate(keyword_results, start=1):
            rrf_score = self.keyword_weight / (self.rrf_k + rank)
            fused[doc_id] = fused.get(doc_id, 0) + rrf_score

        return fused

    def _rerank(
        self,
        fused_scores: Dict[str, float],
        query: str
    ) -> List[SearchResult]:
        """
        Re-rank fused results with additional signals

        Signals:
        - Recency (time decay)
        - Concept overlap
        - Importance score
        """
        results = []

        for doc_id, base_score in fused_scores.items():
            # Fetch document metadata
            doc = self._get_document(doc_id)

            # Apply recency boost (exponential decay)
            age_days = (datetime.utcnow() - doc.timestamp).days
            recency_boost = np.exp(-age_days / 30)  # 30-day half-life

            # Apply importance boost
            importance_boost = doc.metadata.get('importance_score', 0.5)

            # Combined score
            final_score = (
                base_score * 0.6 +
                recency_boost * 0.2 +
                importance_boost * 0.2
            )

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=final_score,
                    text=doc.text,
                    metadata=doc.metadata,
                    snippet=self._generate_snippet(doc.text, query)
                )
            )

        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)

        return results
```

### Performance Benchmarks

**Expected Latency (P95):**
```
Hybrid Search (top_k=5, 10K documents):
├─ Vector search (ChromaDB):      40ms
├─ Keyword search (FTS5):          30ms
├─ RRF fusion:                     10ms
├─ Re-ranking:                     20ms
└─ TOTAL:                         100ms ✅

Hybrid Search (top_k=5, 100K documents):
├─ Vector search (ChromaDB):      80ms
├─ Keyword search (FTS5):          50ms
├─ RRF fusion:                     15ms
├─ Re-ranking:                     25ms
└─ TOTAL:                         170ms ✅
```

---

## Knowledge Graph Schema

### Neo4j Community Edition Setup

**Why Neo4j Community:**
- ✅ Free embedded version (perfect for Railway)
- ✅ Powerful Cypher query language
- ✅ Built-in graph algorithms
- ✅ Excellent Python driver (neo4j-python)
- ❌ Single-instance only (acceptable for v2.0)

**Configuration:**
```python
from neo4j import GraphDatabase

class Neo4jKnowledgeGraph:
    """
    SPARC: Architecture - Knowledge graph for concept relationships
    PATTERN: Property graph model
    WHY: Natural representation of learning relationships
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()
        self._create_indexes()

    def _create_constraints(self):
        """Create uniqueness constraints"""
        with self.driver.session() as session:
            # Unique IDs
            session.run("""
                CREATE CONSTRAINT capture_id IF NOT EXISTS
                FOR (c:Capture) REQUIRE c.id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT concept_name IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.name IS UNIQUE
            """)

    def _create_indexes(self):
        """Create indexes for fast lookups"""
        with self.driver.session() as session:
            session.run("""
                CREATE INDEX capture_timestamp IF NOT EXISTS
                FOR (c:Capture) ON (c.timestamp)
            """)

            session.run("""
                CREATE INDEX concept_category IF NOT EXISTS
                FOR (c:Concept) ON (c.category)
            """)
```

### Graph Schema

```
Node Types:
───────────

(Capture)                       - A conversation turn or insight
  ├─ id: String (UUID)          - Unique identifier
  ├─ text: String               - Full text content
  ├─ timestamp: DateTime        - When captured
  ├─ session_id: String         - Conversation session
  ├─ user_text: String          - User's input
  ├─ agent_text: String         - Agent's response
  ├─ capture_type: String       - conversation | insight | summary
  └─ importance: Float [0-1]    - Calculated importance

(Concept)                       - Abstract learning concept
  ├─ name: String               - Concept name (e.g., "neural networks")
  ├─ category: String           - Domain (ai/ml, science, etc.)
  ├─ definition: String         - Optional definition
  ├─ mention_count: Int         - How many times mentioned
  ├─ last_mentioned: DateTime   - Most recent mention
  └─ importance: Float [0-1]    - Calculated importance

(Entity)                        - Named entity (person, org, tool)
  ├─ name: String               - Entity name
  ├─ type: String               - PERSON | ORG | TOOL | PAPER
  ├─ context: String            - First mention context
  └─ mention_count: Int         - Frequency

(Session)                       - Conversation session
  ├─ id: String                 - Session identifier
  ├─ start_time: DateTime       - Session start
  ├─ end_time: DateTime         - Session end
  ├─ exchange_count: Int        - Number of turns
  └─ primary_topics: [String]   - Main topics discussed


Relationship Types:
───────────────────

(Capture)-[:MENTIONS {weight: Float}]->(Concept)
  • Weight: Importance of concept in capture (0-1)
  • Created by: AnalysisAgent

(Concept)-[:RELATED_TO {strength: Float, type: String}]->(Concept)
  • Strength: How strongly related (0-1)
  • Type: "prerequisite" | "uses" | "similar_to" | "part_of"
  • Example: (transformers)-[:USES {strength: 0.9}]->(attention)

(Capture)-[:PART_OF]->(Session)
  • Links captures to their session

(Capture)-[:FOLLOWS]->(Capture)
  • Sequential relationship in conversation

(Entity)-[:MENTIONED_IN]->(Capture)
  • Links entities to where they appear
```

### Example Graph Queries

**1. Find Related Concepts**
```cypher
// Find concepts related to "neural networks"
MATCH (c1:Concept {name: "neural networks"})-[r:RELATED_TO]-(c2:Concept)
RETURN c2.name, r.strength, r.type
ORDER BY r.strength DESC
LIMIT 10

// Result:
// "deep learning", 0.95, "similar_to"
// "backpropagation", 0.90, "uses"
// "gradient descent", 0.85, "uses"
```

**2. Find Captures by Concept Traversal**
```cypher
// Find all captures that mention "transformers" or related concepts
MATCH (query:Concept {name: "transformers"})
MATCH (query)-[:RELATED_TO]-(related:Concept)
MATCH (capture:Capture)-[:MENTIONS]->(related)
RETURN DISTINCT capture
ORDER BY capture.timestamp DESC
LIMIT 5
```

**3. Discover Learning Paths**
```cypher
// Find prerequisite chain for "transformers"
MATCH path = (start:Concept {name: "transformers"})<-[:RELATED_TO*1..3 {type: "prerequisite"}]-(prereq:Concept)
RETURN path
ORDER BY length(path) DESC
```

**4. Session Topic Analysis**
```cypher
// Find main topics in a session
MATCH (s:Session {id: "sess_abc123"})<-[:PART_OF]-(c:Capture)-[:MENTIONS]->(concept:Concept)
WITH concept, count(*) as mentions
ORDER BY mentions DESC
LIMIT 5
RETURN concept.name, mentions
```

### Integration with Vector Store

**Pattern**: Bi-directional enhancement
- **Vector → Graph**: Semantic search finds captures → Graph finds related concepts
- **Graph → Vector**: Graph traversal finds related concepts → Vector search finds similar captures

```python
async def enhanced_retrieval(
    self,
    query: str,
    use_graph: bool = True
) -> List[SearchResult]:
    """
    Enhanced retrieval using both vector and graph

    Flow:
    1. Vector search finds top-k relevant captures
    2. Graph traversal finds related concepts
    3. Vector search again with expanded query
    4. Combine and re-rank
    """
    # Initial vector search
    initial_results = await self.hybrid_search.search(query, top_k=5)

    if use_graph and initial_results:
        # Extract concepts from initial results
        concepts = set()
        for result in initial_results:
            concepts.update(result.metadata.get('concepts', []))

        # Graph traversal to find related concepts
        related_concepts = await self.graph.find_related_concepts(
            list(concepts),
            max_distance=2
        )

        # Expand query with related concepts
        expanded_query = f"{query} {' '.join(related_concepts)}"

        # Second vector search with expanded query
        expanded_results = await self.hybrid_search.search(
            expanded_query,
            top_k=10
        )

        # Merge and re-rank
        final_results = self._merge_results(
            initial_results,
            expanded_results
        )
    else:
        final_results = initial_results

    return final_results
```

---

## RAG System Design

### RAG Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG ORCHESTRATOR                          │
└──────────────────────────────────────────────────────────────┘

User Query: "How do transformers work?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. QUERY PROCESSING                                         │
│                                                             │
│  ├─→ Intent Detection                                      │
│  │   • Is this a factual question? ✓                       │
│  │   • Does it need retrieval? ✓                           │
│  │   • Is it conversational only? ✗                        │
│  │                                                          │
│  ├─→ Query Expansion                                       │
│  │   • Original: "How do transformers work?"               │
│  │   • Expanded: "transformers architecture attention      │
│  │               mechanisms neural networks NLP"           │
│  │                                                          │
│  └─→ Embedding Generation                                  │
│      • all-MiniLM-L6-v2 (50ms)                             │
│      • 384-dim vector                                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. RETRIEVAL PHASE                                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Hybrid Search (70% vector + 30% keyword)           │   │
│  │                                                     │   │
│  │  ┌─ Vector Search (ChromaDB)                       │   │
│  │  │  • Cosine similarity                            │   │
│  │  │  • Top 10 results                                │   │
│  │  │  • Threshold: >0.7 similarity                   │   │
│  │  │                                                  │   │
│  │  ├─ Keyword Search (FTS5)                          │   │
│  │  │  • BM25 ranking                                  │   │
│  │  │  • Top 10 results                                │   │
│  │  │  • Exact term matching                          │   │
│  │  │                                                  │   │
│  │  └─ Graph Traversal (Neo4j)                        │   │
│  │     • Find related concepts                        │   │
│  │     • Traverse RELATED_TO edges                    │   │
│  │     • Return connected captures                    │   │
│  │                                                     │   │
│  │  → RRF Fusion → Top 5 documents                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Retrieved Context:                                         │
│  [1] "Transformers use attention mechanisms..." (0.89)     │
│  [2] "The Attention is All You Need paper..." (0.87)       │
│  [3] "Self-attention allows the model to..." (0.85)        │
│  [4] "Multi-head attention is a key component..." (0.82)   │
│  [5] "Positional encoding is needed because..." (0.78)     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. CONTEXT ASSEMBLY                                         │
│                                                             │
│  ├─→ Re-ranking (optional)                                 │
│  │   • Diversity filtering (avoid redundancy)              │
│  │   • Recency boost                                       │
│  │   • Importance weighting                                │
│  │                                                          │
│  ├─→ Context Formatting                                    │
│  │   • Add source citations [1], [2], etc.                │
│  │   • Format as LLM context                               │
│  │   • Truncate to fit context window                     │
│  │                                                          │
│  └─→ Metadata Extraction                                   │
│      • Timestamps for each source                          │
│      • Concepts mentioned                                   │
│      • Session IDs (for follow-up)                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. GENERATION PHASE                                         │
│                                                             │
│  System Prompt:                                             │
│  """                                                        │
│  You are a learning companion. Answer the user's question   │
│  using ONLY the provided context. Cite sources with [1],    │
│  [2], etc. If the context doesn't contain the answer,       │
│  say so honestly.                                           │
│  """                                                        │
│                                                             │
│  Context:                                                   │
│  [1] "Transformers use attention mechanisms..."            │
│  [2] "The Attention is All You Need paper..."              │
│  ...                                                        │
│                                                             │
│  Query: "How do transformers work?"                         │
│                                                             │
│  ───────────────────────────────────────────────────────   │
│                                                             │
│  Claude 3.5 Sonnet generates:                               │
│  "Transformers are a neural network architecture that       │
│  uses attention mechanisms to process sequences [1].        │
│  Introduced in the 'Attention is All You Need' paper [2],  │
│  they rely on self-attention to focus on relevant parts     │
│  of the input [3]. Unlike RNNs, transformers don't          │
│  process sequentially, which allows for parallelization.    │
│  Would you like me to explain attention mechanisms in       │
│  more detail?"                                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING                                          │
│                                                             │
│  ├─→ Citation Formatting                                   │
│  │   • Expand [1] to include timestamp/session             │
│  │   • Add "View source" links                             │
│  │                                                          │
│  ├─→ Confidence Scoring                                    │
│  │   • How well did retrieved context match query?         │
│  │   • Flag low-confidence answers                         │
│  │                                                          │
│  └─→ Feedback Collection                                   │
│      • Was this answer helpful? (thumbs up/down)            │
│      • Track for model improvement                          │
└─────────────────────────────────────────────────────────────┘
```

### RAG Implementation

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class RAGContext:
    """Retrieved context for RAG generation"""
    documents: List[str]         # Retrieved document texts
    metadata: List[dict]          # Document metadata
    scores: List[float]           # Relevance scores
    citations: List[str]          # Citation markers [1], [2], etc.

class RAGOrchestrator:
    """
    SPARC: Completion - Full RAG system
    PATTERN: Retrieve → Re-rank → Generate
    WHY: Ground LLM responses in user's own knowledge
    """

    def __init__(
        self,
        hybrid_search: HybridSearchFusion,
        knowledge_graph: Neo4jKnowledgeGraph,
        llm_client,  # Claude API client
        max_context_tokens: int = 4000
    ):
        self.search = hybrid_search
        self.graph = knowledge_graph
        self.llm = llm_client
        self.max_context_tokens = max_context_tokens
        self.logger = get_logger("rag_orchestrator")

    async def answer_query(
        self,
        query: str,
        session_id: str,
        use_rag: bool = True,
        top_k: int = 5
    ) -> dict:
        """
        Answer query using RAG

        Args:
            query: User question
            session_id: Current session
            use_rag: Whether to use retrieval (False = direct LLM)
            top_k: Number of documents to retrieve

        Returns:
            {
                'answer': str,
                'sources': List[dict],
                'confidence': float,
                'used_rag': bool
            }
        """
        if not use_rag or not self._should_use_rag(query):
            # Direct LLM response without retrieval
            answer = await self.llm.generate(query, session_id)
            return {
                'answer': answer,
                'sources': [],
                'confidence': 0.7,
                'used_rag': False
            }

        # 1. Retrieve relevant context
        search_results = await self.search.search(
            query=query,
            top_k=top_k,
            filters={'session_id': session_id}  # Optional: limit to current session
        )

        if not search_results:
            # No relevant context found
            self.logger.warning(
                "no_relevant_context",
                query=query,
                session_id=session_id
            )
            return {
                'answer': "I don't have any relevant information about that in our conversation history.",
                'sources': [],
                'confidence': 0.0,
                'used_rag': True
            }

        # 2. Assemble context for LLM
        rag_context = self._assemble_context(search_results)

        # 3. Generate answer with citations
        answer = await self._generate_with_rag(
            query=query,
            context=rag_context,
            session_id=session_id
        )

        # 4. Format sources for response
        sources = self._format_sources(search_results)

        # 5. Calculate confidence
        confidence = self._calculate_confidence(search_results, answer)

        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'used_rag': True
        }

    def _should_use_rag(self, query: str) -> bool:
        """
        Determine if query needs retrieval

        Heuristics:
        - Factual questions: Yes
        - "Tell me about..." : Yes
        - "What did I say..." : Yes
        - Greetings: No
        - Short responses: No
        """
        # Simple heuristics (can enhance with ML classifier)
        query_lower = query.lower()

        # Retrieval indicators
        retrieval_keywords = [
            'what', 'how', 'why', 'when', 'where',
            'tell me about', 'explain', 'what did i',
            'remind me', 'find', 'search'
        ]

        # Non-retrieval indicators
        skip_keywords = [
            'hello', 'hi', 'hey', 'thanks', 'thank you',
            'ok', 'okay', 'yes', 'no'
        ]

        # Short queries probably don't need RAG
        if len(query.split()) < 3:
            return False

        # Check for skip keywords
        if any(keyword in query_lower for keyword in skip_keywords):
            return False

        # Check for retrieval keywords
        if any(keyword in query_lower for keyword in retrieval_keywords):
            return True

        # Default: use RAG for medium/long queries
        return len(query.split()) >= 5

    def _assemble_context(
        self,
        search_results: List[SearchResult]
    ) -> RAGContext:
        """
        Assemble retrieved documents into LLM context

        Format:
        ---
        [1] (2025-11-20 from session abc123)
        "User: Tell me about neural networks
        Agent: Neural networks are computational models..."

        [2] (2025-11-19 from session abc123)
        "User: How does backpropagation work?
        Agent: Backpropagation is the algorithm..."
        ---
        """
        documents = []
        metadata_list = []
        scores = []
        citations = []

        total_tokens = 0
        max_tokens = self.max_context_tokens

        for idx, result in enumerate(search_results, start=1):
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            doc_tokens = len(result.text) // 4

            if total_tokens + doc_tokens > max_tokens:
                # Truncate to fit
                available_tokens = max_tokens - total_tokens
                truncated_text = result.text[: available_tokens * 4]
                text = f"{truncated_text}..."
            else:
                text = result.text

            # Format with citation marker
            timestamp = result.metadata.get('timestamp', 'unknown')
            session = result.metadata.get('session_id', 'unknown')

            formatted = f"""[{idx}] ({timestamp} from session {session})
{text}"""

            documents.append(formatted)
            metadata_list.append(result.metadata)
            scores.append(result.score)
            citations.append(f"[{idx}]")

            total_tokens += doc_tokens

            if total_tokens >= max_tokens:
                break

        return RAGContext(
            documents=documents,
            metadata=metadata_list,
            scores=scores,
            citations=citations
        )

    async def _generate_with_rag(
        self,
        query: str,
        context: RAGContext,
        session_id: str
    ) -> str:
        """
        Generate answer using LLM with retrieved context
        """
        # Build system prompt
        system_prompt = """You are a learning companion helping the user recall and connect their thoughts.

Your role:
- Answer questions using ONLY the provided context from the user's previous conversations
- Cite sources using [1], [2], etc. when referencing context
- If the context doesn't fully answer the question, say so honestly
- Connect ideas across different conversation sessions when relevant
- Keep responses conversational and under 3 sentences unless more detail is needed

If the context is insufficient, suggest: "I don't have enough information about that in our conversations. Would you like to explore this topic together?"
"""

        # Build user prompt with context
        user_prompt = f"""Context from previous conversations:

{chr(10).join(context.documents)}

---

Question: {query}

Please answer using the context above. Cite sources with [1], [2], etc."""

        # Call LLM
        response = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            session_id=session_id,
            temperature=0.7
        )

        return response

    def _format_sources(
        self,
        search_results: List[SearchResult]
    ) -> List[dict]:
        """Format sources for API response"""
        sources = []

        for idx, result in enumerate(search_results, start=1):
            sources.append({
                'citation': f"[{idx}]",
                'text': result.text[:200] + "..." if len(result.text) > 200 else result.text,
                'timestamp': result.metadata.get('timestamp'),
                'session_id': result.metadata.get('session_id'),
                'concepts': result.metadata.get('concepts', []),
                'score': round(result.score, 3),
                'doc_id': result.doc_id
            })

        return sources

    def _calculate_confidence(
        self,
        search_results: List[SearchResult],
        answer: str
    ) -> float:
        """
        Calculate confidence in RAG answer

        Factors:
        - Top result similarity score
        - Number of high-quality results
        - Citation presence in answer
        """
        if not search_results:
            return 0.0

        # Top result score (0-1)
        top_score = search_results[0].score

        # How many results above threshold (0.7)?
        high_quality_count = sum(1 for r in search_results if r.score > 0.7)
        quality_factor = min(high_quality_count / 3, 1.0)  # Cap at 3 results

        # Are there citations in the answer?
        citation_present = any(f"[{i}]" in answer for i in range(1, 6))
        citation_factor = 1.0 if citation_present else 0.7

        # Combined confidence
        confidence = top_score * 0.5 + quality_factor * 0.3 + citation_factor * 0.2

        return round(confidence, 2)
```

### RAG Evaluation Metrics

**Retrieval Quality:**
- **Precision@5**: % of top-5 results that are relevant (target: >85%)
- **Recall@5**: % of relevant docs retrieved in top-5 (target: >70%)
- **MRR (Mean Reciprocal Rank)**: Average rank of first relevant result (target: >0.8)

**Generation Quality:**
- **Citation accuracy**: Do citations match retrieved docs? (target: 100%)
- **Groundedness**: Is answer supported by context? (human eval, target: >90%)
- **Helpfulness**: User feedback thumbs up/down (target: >80% positive)

**End-to-End Latency:**
- Retrieval: <100ms (P95)
- Generation: <800ms (P95)
- Total RAG: <900ms (P95)

---

## Integration with Phase 2 Agents

### ConversationAgent Integration

```python
# app/agents/conversation_agent.py (enhanced)

class ConversationAgent(BaseAgent):
    """
    PHASE 3: Enhanced with RAG capabilities
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_orchestrator = RAGOrchestrator(...)  # Inject RAG
        self.use_rag = True  # Feature flag

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process conversation with optional RAG
        """
        user_input = message.content.get('user_input')
        session_id = message.content.get('session_id')

        # Decide whether to use RAG
        use_rag = self._should_use_rag(user_input)

        if use_rag and self.use_rag:
            # RAG-powered response
            rag_result = await self.rag_orchestrator.answer_query(
                query=user_input,
                session_id=session_id,
                top_k=5
            )

            response_text = rag_result['answer']
            metadata = {
                'rag_used': True,
                'rag_confidence': rag_result['confidence'],
                'sources': rag_result['sources']
            }
        else:
            # Direct LLM response (existing behavior)
            response_text = await self._generate_response(user_input, session_id)
            metadata = {'rag_used': False}

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.CONVERSATION_COMPLETE,
            content={
                'agent_text': response_text,
                'metadata': metadata
            }
        )
```

### AnalysisAgent → Knowledge Graph

```python
# app/agents/analysis_agent.py (enhanced)

class AnalysisAgent(BaseAgent):
    """
    PHASE 3: Enhanced to populate knowledge graph
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_graph = Neo4jKnowledgeGraph(...)  # Inject graph

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Extract concepts AND update knowledge graph
        """
        user_input = message.content.get('user_input')
        session_id = message.content.get('session_id')

        # 1. Extract concepts (existing)
        analysis_result = await self._analyze(user_input)

        # 2. Update knowledge graph (NEW)
        await self._update_knowledge_graph(
            capture_id=message.message_id,
            text=user_input,
            concepts=analysis_result.concepts,
            entities=analysis_result.entities,
            session_id=session_id
        )

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.ANALYSIS_COMPLETE,
            content={
                'concepts': [c.to_dict() for c in analysis_result.concepts],
                'entities': [e.to_dict() for e in analysis_result.entities],
                'relationships': analysis_result.relationships
            }
        )

    async def _update_knowledge_graph(
        self,
        capture_id: str,
        text: str,
        concepts: List[Concept],
        entities: List[Entity],
        session_id: str
    ):
        """Update Neo4j knowledge graph"""
        async with self.knowledge_graph.driver.session() as session:
            # Create Capture node
            await session.run("""
                MERGE (cap:Capture {id: $id})
                SET cap.text = $text,
                    cap.timestamp = datetime(),
                    cap.session_id = $session_id
            """, id=capture_id, text=text, session_id=session_id)

            # Create Concept nodes and relationships
            for concept in concepts:
                await session.run("""
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.category = $category,
                                   c.mention_count = 1,
                                   c.last_mentioned = datetime()
                    ON MATCH SET c.mention_count = c.mention_count + 1,
                                  c.last_mentioned = datetime()

                    WITH c
                    MATCH (cap:Capture {id: $capture_id})
                    MERGE (cap)-[:MENTIONS {weight: $weight}]->(c)
                """,
                name=concept.name,
                category=concept.category,
                capture_id=capture_id,
                weight=concept.importance
                )
```

### ResearchAgent → Vector Ingestion

```python
# app/agents/research_agent.py (enhanced)

class ResearchAgent(BaseAgent):
    """
    PHASE 3: Enhanced to store research results in vector store
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_ingestion = VectorIngestionService(...)  # Inject

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Research query AND store findings
        """
        query = message.content.get('query')
        session_id = message.content.get('session_id')

        # 1. Execute research (existing)
        research_result = await self._research(query)

        # 2. Store valuable findings in vector store (NEW)
        for fact in research_result.facts:
            if fact.confidence > 0.8:  # Only high-quality facts
                await self.vector_ingestion.ingest_conversation(
                    capture_id=f"research_{fact.id}",
                    user_text=query,
                    agent_text=fact.statement,
                    session_id=session_id,
                    concepts=[fact.topic],
                    entities=[],
                    metadata={
                        'capture_type': 'research',
                        'source': fact.source_id,
                        'importance_score': fact.confidence
                    }
                )

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.RESEARCH_COMPLETE,
            content={
                'facts': [f.to_dict() for f in research_result.facts],
                'sources': [s.to_dict() for s in research_result.sources]
            }
        )
```

### Orchestrator Enhancements

```python
# app/agents/orchestrator.py (enhanced)

class AgentOrchestrator:
    """
    PHASE 3: Orchestrate with vector memory integration
    """

    async def process_request(
        self,
        user_input: str,
        session_id: str
    ) -> OrchestrationResult:
        """
        Enhanced orchestration with vector memory

        Flow:
        1. Check if RAG should be used
        2. Route to appropriate agents
        3. Ingest results into vector store
        4. Return response
        """
        # 1. Classify input (existing)
        classification = await self._classify_input(user_input)

        # 2. Route to agents (existing)
        selected_agents = self._route_to_agents(classification)

        # 3. Execute agents in parallel (existing)
        agent_results = await self._execute_agents(
            selected_agents,
            user_input,
            session_id
        )

        # 4. Ingest into vector store (NEW)
        await self._ingest_to_vector_store(
            user_input=user_input,
            agent_results=agent_results,
            session_id=session_id
        )

        # 5. Synthesize response (existing)
        final_response = await self._synthesize_results(agent_results)

        return final_response

    async def _ingest_to_vector_store(
        self,
        user_input: str,
        agent_results: dict,
        session_id: str
    ):
        """Store conversation turn in vector database"""
        conversation_result = agent_results.get('conversation')
        analysis_result = agent_results.get('analysis')

        if not conversation_result:
            return

        await self.vector_ingestion.ingest_conversation(
            capture_id=str(uuid.uuid4()),
            user_text=user_input,
            agent_text=conversation_result.get('agent_text'),
            session_id=session_id,
            concepts=analysis_result.get('concepts', []) if analysis_result else [],
            entities=analysis_result.get('entities', []) if analysis_result else [],
            metadata={
                'capture_type': 'conversation',
                'agents_involved': list(agent_results.keys()),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
```

---

## Performance Targets

### Latency Budgets (P95)

```
End-to-End Conversation with RAG:
├─ User input processing:           20ms
├─ Intent classification:            30ms
├─ Retrieval (hybrid search):       100ms
│  ├─ Vector search (ChromaDB):      50ms
│  ├─ Keyword search (FTS5):         30ms
│  └─ RRF fusion + re-rank:          20ms
├─ LLM generation (with context):   800ms
├─ Vector ingestion (async):         50ms
│  ├─ Embedding generation:          50ms
│  └─ ChromaDB insert:               10ms
├─ Graph update (async):             30ms
└─ TOTAL:                           1030ms ✅ (target: <1200ms)

Note: Vector ingestion and graph update run asynchronously
      after response is sent to user, so they don't block.
```

### Throughput Requirements

**Single Railway Instance:**
- **Concurrent users**: 10-20 (single-user focused)
- **Requests/second**: 5-10 sustained
- **Peak burst**: 20 req/s for 30 seconds

**Database Performance:**
- **ChromaDB**:
  - 10K docs: <50ms search
  - 100K docs: <100ms search ✅ Target
  - 1M docs: <200ms search (future scaling)

- **Neo4j**:
  - Concept traversal: <30ms (depth=2)
  - Graph stats queries: <50ms

- **SQLite FTS5**:
  - Keyword search: <30ms (100K rows)

### Resource Limits (Railway)

**Memory:**
- ChromaDB: ~200MB (100K docs)
- Neo4j: ~150MB (embedded)
- all-MiniLM-L6-v2 model: ~90MB
- Redis cache: ~50MB
- **Total**: ~500MB baseline + ~500MB working ≈ **1GB RAM** ✅

**Disk:**
- ChromaDB vectors: ~150MB (100K docs @ 1.5KB each)
- Neo4j graph: ~50MB (10K nodes/edges)
- SQLite archive: ~100MB
- **Total**: ~300MB ✅ (Railway: 1GB persistent storage)

**CPU:**
- Embedding generation: 50ms/doc (CPU)
- Vector search: <100ms (CPU-optimized HNSW)
- **Railway Shared CPU**: Adequate for single-user

---

## Migration Strategy

### Phase 1: Parallel Write (Week 1)

**Objective**: Write to both old (SQLite FTS5) and new (ChromaDB + Neo4j) systems

```python
# Dual-write pattern
async def save_conversation(user_text, agent_text, session_id):
    # 1. Legacy write (existing)
    await sqlite_db.save_capture(user_text, agent_text, session_id)

    # 2. New vector write (parallel)
    try:
        await vector_ingestion.ingest_conversation(
            capture_id=str(uuid.uuid4()),
            user_text=user_text,
            agent_text=agent_text,
            session_id=session_id,
            concepts=[],  # Populated by AnalysisAgent
            entities=[],
            metadata={'migrated': False}
        )
    except Exception as e:
        logger.error("vector_write_failed", error=str(e))
        # Don't fail the request - legacy write succeeded
```

**Validation**:
- ✅ All new conversations written to both systems
- ✅ No data loss
- ✅ Latency impact <50ms (async writes)

### Phase 2: Historical Data Migration (Week 1-2)

**Objective**: Backfill existing SQLite data into ChromaDB/Neo4j

```python
async def migrate_historical_data(batch_size=100):
    """
    Migrate existing conversations to vector store

    Strategy:
    1. Fetch batches from SQLite
    2. Generate embeddings (batched)
    3. Ingest to ChromaDB + Neo4j
    4. Mark as migrated
    """
    offset = 0
    total_migrated = 0

    while True:
        # Fetch batch
        batch = await sqlite_db.fetch_all("""
            SELECT id, user_text, agent_text, session_id, timestamp
            FROM captures
            WHERE migrated = 0
            ORDER BY timestamp ASC
            LIMIT ? OFFSET ?
        """, [batch_size, offset])

        if not batch:
            break

        # Generate embeddings in batch
        texts = [f"User: {r['user_text']}\nAgent: {r['agent_text']}"
                 for r in batch]
        embeddings = await embedding_pipeline.embed_batch(texts)

        # Ingest to ChromaDB
        for row, embedding in zip(batch, embeddings):
            await vector_ingestion.ingest_conversation(
                capture_id=row['id'],
                user_text=row['user_text'],
                agent_text=row['agent_text'],
                session_id=row['session_id'],
                concepts=[],  # No historical analysis (optional: run AnalysisAgent)
                entities=[],
                metadata={
                    'migrated': True,
                    'migrated_at': datetime.utcnow().isoformat(),
                    'timestamp': row['timestamp']
                }
            )

        # Mark as migrated
        ids = [r['id'] for r in batch]
        await sqlite_db.execute("""
            UPDATE captures
            SET migrated = 1
            WHERE id IN ({})
        """.format(','.join('?' * len(ids))), ids)

        total_migrated += len(batch)
        offset += batch_size

        logger.info(
            "migration_progress",
            total_migrated=total_migrated,
            batch_size=batch_size
        )

    logger.info("migration_complete", total_records=total_migrated)
```

**Timeline**:
- 10K historical captures @ 100/batch = 100 batches
- ~1 second per batch (embedding + ingestion)
- **Total: ~2-3 minutes** (run during deployment)

### Phase 3: Dual Read with Fallback (Week 2)

**Objective**: Read from vector store, fallback to SQLite if needed

```python
async def search_conversations(query: str, top_k=5):
    """
    Search with fallback strategy
    """
    try:
        # Try vector search first
        results = await hybrid_search.search(query, top_k=top_k)

        if results and results[0].score > 0.6:
            # Good vector results
            return results
        else:
            # Fall back to keyword search
            logger.warning("vector_search_low_quality", top_score=results[0].score if results else 0)
            return await sqlite_fts5_search(query, top_k=top_k)

    except Exception as e:
        # Fall back on error
        logger.error("vector_search_failed", error=str(e))
        return await sqlite_fts5_search(query, top_k=top_k)
```

### Phase 4: Feature Flag Rollout (Week 2-3)

**Objective**: Gradual rollout to production

```python
# config.py
class FeatureFlags:
    ENABLE_VECTOR_SEARCH = os.getenv("ENABLE_VECTOR_SEARCH", "false") == "true"
    ENABLE_RAG = os.getenv("ENABLE_RAG", "false") == "true"
    ENABLE_KNOWLEDGE_GRAPH = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "false") == "true"

    # Gradual rollout percentages
    VECTOR_SEARCH_PERCENTAGE = int(os.getenv("VECTOR_SEARCH_PCT", "0"))
    RAG_PERCENTAGE = int(os.getenv("RAG_PCT", "0"))
```

**Rollout Schedule**:
```
Week 2: Development Testing
├─ ENABLE_VECTOR_SEARCH = true (100%)
├─ ENABLE_RAG = false
└─ Monitor metrics, fix bugs

Week 3: Beta Testing
├─ ENABLE_VECTOR_SEARCH = true (100%)
├─ ENABLE_RAG = true (25%)  ← Gradual rollout
└─ Collect user feedback

Week 4: Full Production
├─ ENABLE_VECTOR_SEARCH = true (100%)
├─ ENABLE_RAG = true (100%)
├─ ENABLE_KNOWLEDGE_GRAPH = true
└─ Deprecate FTS5-only search
```

### Phase 5: Validation & Cleanup (Week 4)

**Validation Checklist**:
- ✅ All new conversations in vector store
- ✅ Historical data migrated (100%)
- ✅ RAG responses include citations
- ✅ Hybrid search outperforms keyword-only
- ✅ Latency targets met (<1200ms P95)
- ✅ Zero data loss confirmed

**Cleanup**:
- Remove dual-write code
- Archive old FTS5 search endpoints
- Update documentation

---

## Implementation Plan

### Week 1: Foundation

**Day 1-2: Setup**
- [ ] Install dependencies (sentence-transformers, chromadb, neo4j)
- [ ] Configure ChromaDB persistent storage
- [ ] Configure Neo4j embedded instance
- [ ] Create database schemas
- [ ] Set up feature flags

**Day 3-4: Embedding Pipeline**
- [ ] Implement `EmbeddingPipeline` class
- [ ] Implement `TextPreprocessor` class
- [ ] Implement `VectorIngestionService` class
- [ ] Write unit tests (>80% coverage)
- [ ] Benchmark embedding speed

**Day 5: Historical Migration**
- [ ] Implement migration script
- [ ] Test on sample data
- [ ] Run full historical migration
- [ ] Validate data integrity

### Week 2: Search & Retrieval

**Day 1-2: ChromaDB Integration**
- [ ] Implement ChromaDB collection setup
- [ ] Implement vector search queries
- [ ] Test metadata filtering
- [ ] Optimize HNSW parameters

**Day 3-4: Hybrid Search**
- [ ] Implement `HybridSearchFusion` class
- [ ] Implement RRF algorithm
- [ ] Implement re-ranking logic
- [ ] Write integration tests
- [ ] Benchmark search performance

**Day 5: Knowledge Graph**
- [ ] Implement `Neo4jKnowledgeGraph` class
- [ ] Create Cypher queries
- [ ] Test graph traversal
- [ ] Integrate with AnalysisAgent

### Week 3: RAG System

**Day 1-2: RAG Orchestrator**
- [ ] Implement `RAGOrchestrator` class
- [ ] Implement context assembly
- [ ] Implement citation formatting
- [ ] Test with various query types

**Day 3-4: Agent Integration**
- [ ] Enhance ConversationAgent with RAG
- [ ] Enhance AnalysisAgent with graph updates
- [ ] Enhance ResearchAgent with vector ingestion
- [ ] Update Orchestrator for dual-write

**Day 5: End-to-End Testing**
- [ ] Integration tests (all agents + RAG)
- [ ] Performance benchmarks
- [ ] User acceptance testing
- [ ] Fix bugs and optimize

### Week 4: Production Rollout

**Day 1-2: Gradual Rollout**
- [ ] Deploy with feature flags off
- [ ] Enable vector search (100%)
- [ ] Enable RAG (25% → 50% → 100%)
- [ ] Monitor metrics and logs

**Day 3-4: Validation**
- [ ] Run validation tests
- [ ] Collect user feedback
- [ ] Measure performance against targets
- [ ] Document learnings

**Day 5: Cleanup**
- [ ] Remove feature flags (or set to 100%)
- [ ] Archive old code
- [ ] Update documentation
- [ ] Write Phase 3 completion report

---

## Conclusion

This Phase 3 architecture transforms the learning_voice_agent from keyword search to **intelligent semantic memory with RAG**, enabling:

### Key Capabilities Unlocked

✅ **Semantic Understanding**: Vector embeddings capture meaning, not just keywords
✅ **Hybrid Search**: Best of both worlds (semantic + lexical)
✅ **Knowledge Graph**: Visualize and traverse concept relationships
✅ **RAG-Powered Responses**: Answers grounded in user's own knowledge with citations
✅ **Intelligent Retrieval**: Integration with Phase 2 multi-agent system

### Design Philosophy

This architecture follows the project's "avoid overengineering" principle:

1. **Start Simple**: ChromaDB embedded (not Pinecone cloud), basic re-ranking (not advanced HyDE)
2. **Optimize for Single-User**: Railway single-instance deployment, <1GB RAM
3. **Scale Later**: Can migrate to Pinecone, cloud Neo4j, advanced RAG later if needed
4. **Proven Patterns**: RRF fusion, HNSW indexing, basic RAG (battle-tested approaches)

### Next Steps

1. **Review & Approve** this architecture design
2. **Begin Week 1 Implementation** (setup + embedding pipeline)
3. **Parallel Work**: Frontend team can start designing search UI
4. **Success Metrics**: Track retrieval quality, RAG citation accuracy, user satisfaction

### Success Metrics (Reminder)

- ✅ Semantic search >85% precision@5
- ✅ RAG responses include accurate citations
- ✅ Retrieval latency <100ms (P95)
- ✅ End-to-end RAG <1200ms (P95)
- ✅ Zero data loss during migration

---

**Document Status**: Architecture Design Complete - Ready for Implementation
**Author**: System Architect
**Date**: 2025-11-21
**Version**: 1.0
**Next Review**: After Week 1 implementation
