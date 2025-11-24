# Phase 3: Neo4j Knowledge Graph - Implementation Guide

## Overview

The Knowledge Graph module captures and organizes concepts, entities, and their relationships from learning conversations using Neo4j. It provides intelligent querying capabilities for discovering learning paths, identifying knowledge gaps, and generating concept maps.

## Architecture

### Components

```
app/knowledge_graph/
├── __init__.py              # Module initialization and exports
├── config.py                # Neo4j configuration management
├── graph_store.py           # Neo4j CRUD operations (~660 lines)
├── concept_extractor.py     # NLP-based concept extraction (~580 lines)
└── query_engine.py          # Advanced graph queries (~520 lines)
```

### Graph Schema

#### Node Types

```cypher
# Concept: Core ideas and topics
(:Concept {
    name: String (UNIQUE),
    description: String,
    first_seen: DateTime,
    last_seen: DateTime,
    frequency: Integer,
    metadata: Map,
    created_at: DateTime,
    updated_at: DateTime
})

# Entity: Named entities from conversations
(:Entity {
    text: String,
    type: String,  # PERSON, ORG, GPE, PRODUCT, etc.
    first_seen: DateTime,
    last_seen: DateTime,
    metadata: Map
})

# Topic: High-level categories
(:Topic {
    name: String (UNIQUE),
    description: String,
    created_at: DateTime
})

# Session: Conversation sessions
(:Session {
    id: String (UNIQUE),
    timestamp: DateTime,
    exchange_count: Integer,
    metadata: Map,
    last_updated: DateTime
})
```

#### Relationship Types

```cypher
# Concept relationships
(Concept)-[:RELATES_TO {
    strength: Float,        # 0.0-1.0
    first_seen: DateTime,
    last_seen: DateTime,
    observation_count: Integer,
    context: String,
    metadata: Map
}]->(Concept)

(Concept)-[:BUILDS_ON {
    strength: Float,
    # ... same properties as RELATES_TO
}]->(Concept)

# Session relationships
(Concept)-[:MENTIONED_IN {
    timestamp: DateTime,
    count: Integer,
    last_mention: DateTime
}]->(Session)

(Entity)-[:MENTIONED_IN {
    timestamp: DateTime,
    count: Integer
}]->(Session)

# Entity-Concept relationship
(Entity)-[:INSTANCE_OF]->(Concept)

# Topic-Concept relationship
(Topic)-[:CONTAINS]->(Concept)
```

## Key Features

### 1. Concept Extraction (`ConceptExtractor`)

Extracts concepts and relationships from conversations using NLP:

```python
from app.knowledge_graph import concept_extractor

# Extract from single text
result = await concept_extractor.extract_from_text(
    text="I'm learning about neural networks and deep learning",
    session_id="optional_session_id"
)

# Extract from conversation
result = await concept_extractor.extract_from_conversation(
    session_id="session_123",
    exchanges=[
        {"user": "What is machine learning?", "agent": "..."},
        {"user": "Tell me about neural networks", "agent": "..."}
    ]
)

# Result structure
{
    "concepts": ["neural networks", "deep learning", ...],
    "entities": [("TensorFlow", "PRODUCT"), ...],
    "relationships": [
        {
            "from_concept": "neural networks",
            "to_concept": "deep learning",
            "type": "RELATES_TO",
            "strength": 0.85,
            "context": "...neural networks and deep learning..."
        }
    ],
    "topics": ["Artificial Intelligence"],
    "stats": {
        "concept_count": 5,
        "entity_count": 2,
        "relationship_count": 3
    }
}
```

#### Extraction Features

- **NLP-Powered**: Integrates with `AnalysisAgent` for entity recognition
- **Relationship Detection**: Uses co-occurrence, proximity, and linguistic cues
- **Strength Calculation**: Multi-signal strength (proximity, frequency, context)
- **Topic Classification**: Automatic categorization into topic hierarchy
- **Prerequisite Detection**: Identifies BUILDS_ON relationships using indicators

### 2. Graph Storage (`KnowledgeGraphStore`)

Manages Neo4j connection and CRUD operations:

```python
from app.knowledge_graph import graph_store

# Initialize (done automatically on app startup)
await graph_store.initialize()

# Add concept
concept_id = await graph_store.add_concept(
    name="machine learning",
    description="ML is a subset of AI...",
    metadata={"category": "AI"},
    topic="Artificial Intelligence"
)

# Add relationship
await graph_store.add_relationship(
    from_concept="neural networks",
    to_concept="deep learning",
    relationship_type="BUILDS_ON",
    strength=0.9,
    context="Neural networks are fundamental to deep learning"
)

# Get related concepts
related = await graph_store.get_related_concepts(
    concept="machine learning",
    max_depth=2,
    min_strength=0.5,
    limit=10
)

# Get statistics
stats = await graph_store.get_graph_stats()
# Returns: {"concepts": 50, "relationships": 75, "entities": 20, ...}
```

### 3. Query Engine (`GraphQueryEngine`)

Advanced querying capabilities:

```python
from app.knowledge_graph import query_engine

# Find related concepts with ranking
related = await query_engine.find_related_concepts(
    concept="machine learning",
    max_depth=2,
    min_strength=0.3,
    limit=10
)

# Get learning path
path = await query_engine.get_learning_path(
    from_concept="python basics",
    to_concept="deep learning",
    max_depth=5
)

# Result:
{
    "found": True,
    "path": ["python basics", "programming", "machine learning", "deep learning"],
    "steps": 3,
    "difficulty": "moderate",
    "estimated_time": "2-4 hours",
    "detailed_steps": [
        {
            "step": 1,
            "concept": "python basics",
            "description": "...",
            "is_start": True
        },
        ...
    ]
}

# Identify knowledge gaps
gaps = await query_engine.identify_knowledge_gaps(
    known_concepts=["python", "statistics"],
    target_concept="deep learning",
    max_depth=3
)

# Result:
{
    "target_concept": "deep learning",
    "known_concepts": ["python", "statistics"],
    "gaps": [
        {
            "concept": "machine learning",
            "priority_score": 0.85,
            "frequency": 10,
            "known_connections": 2
        },
        {
            "concept": "neural networks",
            "priority_score": 0.78,
            ...
        }
    ],
    "recommendations": [
        "Start by learning these concepts: machine learning, neural networks",
        "Easiest learning path starts from: python",
        "You're close to understanding: machine learning"
    ]
}

# Generate concept map
concept_map = await query_engine.get_concept_map(
    central_concept="machine learning",
    radius=2,
    max_concepts=15
)

# Result:
{
    "nodes": [
        {"id": "machine learning", "type": "central", "frequency": 10},
        {"id": "deep learning", "type": "related", "frequency": 5, "distance": 1},
        ...
    ],
    "edges": [
        {"from": "machine learning", "to": "deep learning", "type": "RELATES_TO", "strength": 0.8},
        ...
    ]
}

# Get trending concepts
trending = await query_engine.get_trending_concepts(
    time_window_hours=24,
    limit=10
)
```

## Configuration

### Environment Variables

```bash
# Neo4j Connection (Server Mode)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Embedded Mode (for Railway/local deployment)
NEO4J_EMBEDDED=true
NEO4J_DATA_PATH=./data/neo4j

# Connection Pool
NEO4J_MAX_POOL_SIZE=50
NEO4J_CONNECT_TIMEOUT=30.0
NEO4J_RETRY_TIME=30.0
NEO4J_QUERY_TIMEOUT=5.0

# Graph Algorithm Settings
GRAPH_MAX_DEPTH=3
GRAPH_MIN_STRENGTH=0.3

# Concept Extraction Settings
CONCEPT_MIN_FREQUENCY=2
CONCEPT_SIMILARITY_THRESHOLD=0.7
```

### Neo4j Setup

#### Option 1: Embedded Mode (Recommended for Railway)

```python
from app.knowledge_graph.config import KnowledgeGraphConfig

config = KnowledgeGraphConfig()
config.embedded = True
config.data_path = "./data/neo4j"
```

#### Option 2: Server Mode (Docker)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

```python
config = KnowledgeGraphConfig()
config.embedded = False
config.uri = "bolt://localhost:7687"
config.password = "your_password"
```

## Integration with Existing System

### 1. Conversation Handler Integration

```python
from app.conversation_handler import ConversationHandler
from app.knowledge_graph import concept_extractor

class ConversationHandler:
    async def process_message(self, session_id, user_message):
        # ... existing processing ...

        # Extract concepts from conversation
        exchanges = await self.get_session_exchanges(session_id)
        await concept_extractor.extract_from_conversation(
            session_id=session_id,
            exchanges=exchanges
        )
```

### 2. API Endpoints

Add new endpoints in `app/main.py`:

```python
from app.knowledge_graph import graph_store, query_engine

@app.get("/api/knowledge-graph/concepts/{concept}/related")
async def get_related_concepts(concept: str, max_depth: int = 2):
    """Get concepts related to the given concept"""
    related = await query_engine.find_related_concepts(
        concept=concept,
        max_depth=max_depth
    )
    return {"concept": concept, "related": related}

@app.get("/api/knowledge-graph/learning-path")
async def get_learning_path(from_concept: str, to_concept: str):
    """Get learning path between two concepts"""
    path = await query_engine.get_learning_path(
        from_concept=from_concept,
        to_concept=to_concept
    )
    return path

@app.get("/api/knowledge-graph/gaps")
async def identify_gaps(
    known_concepts: List[str] = Query(...),
    target_concept: str = Query(...)
):
    """Identify knowledge gaps"""
    gaps = await query_engine.identify_knowledge_gaps(
        known_concepts=known_concepts,
        target_concept=target_concept
    )
    return gaps

@app.get("/api/knowledge-graph/stats")
async def get_graph_stats():
    """Get graph statistics"""
    return await graph_store.get_graph_stats()

@app.get("/api/knowledge-graph/concept-map/{concept}")
async def get_concept_map(concept: str, radius: int = 2):
    """Get concept map for visualization"""
    return await query_engine.get_concept_map(
        central_concept=concept,
        radius=radius
    )
```

## Testing

```bash
# Run knowledge graph tests
pytest tests/test_knowledge_graph.py -v

# Run with coverage
pytest tests/test_knowledge_graph.py --cov=app/knowledge_graph --cov-report=html

# Run specific test class
pytest tests/test_knowledge_graph.py::TestConceptExtractor -v
```

## Performance Considerations

### Indexing

The system automatically creates indexes on:
- Concept names (unique constraint)
- Concept frequency
- Concept last_seen timestamp
- Entity types
- Session timestamps

### Query Optimization

- **Batch Operations**: Use `add_session` for bulk concept/entity linking
- **Depth Limiting**: Keep `max_depth` ≤ 3 for optimal performance
- **Strength Filtering**: Use `min_strength` to reduce result set
- **Caching**: Consider implementing query result caching for frequent patterns

### Scalability

- **Connection Pooling**: Configured for up to 50 concurrent connections
- **Async Operations**: All operations are async for non-blocking I/O
- **Transaction Management**: Automatic retry on transient errors
- **Memory Management**: Embedded mode configured with optimized heap sizes

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```python
   # Check Neo4j is running
   await graph_store.driver.verify_connectivity()
   ```

2. **Slow Queries**
   - Reduce `max_depth` parameter
   - Increase `min_strength` threshold
   - Check index creation status

3. **Memory Issues (Embedded Mode)**
   ```bash
   # Adjust heap sizes in config
   NEO4J_HEAP_INITIAL=256m
   NEO4J_HEAP_MAX=512m
   ```

## Future Enhancements

- [ ] Temporal analysis and concept evolution tracking
- [ ] Concept similarity using embeddings
- [ ] Automatic relationship type inference
- [ ] Multi-user knowledge graph isolation
- [ ] Graph visualization UI
- [ ] Export to knowledge graph formats (RDF, OWL)
- [ ] Integration with external knowledge bases (Wikipedia, DBpedia)

## References

- [Neo4j Python Driver Documentation](https://neo4j.com/docs/python-manual/current/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)

## License

Part of the Learning Voice Agent project - Phase 3 implementation.
