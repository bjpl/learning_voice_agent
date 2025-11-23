# Phase 3 Usage Examples

**Version:** 1.0.0
**Date:** 2025-01-21

End-to-end code examples for Phase 3 Vector Memory & RAG features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Vector Search Examples](#vector-search-examples)
3. [Hybrid Search Examples](#hybrid-search-examples)
4. [Knowledge Graph Examples](#knowledge-graph-examples)
5. [RAG Pipeline Examples](#rag-pipeline-examples)
6. [Advanced Use Cases](#advanced-use-cases)

---

## Quick Start

### Setup and Initialization

```python
import asyncio
from app.vector.vector_store import vector_store
from app.vector.embeddings import embedding_generator
from app.search.hybrid_search import create_hybrid_search_engine
from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.knowledge_graph.config import KnowledgeGraphConfig
from app.database import database

async def initialize_phase3():
    """Initialize all Phase 3 components"""

    # 1. Initialize vector store
    await vector_store.initialize()
    print("✅ Vector store initialized")

    # 2. Initialize embedding generator
    await embedding_generator.initialize()
    print("✅ Embedding generator initialized")

    # 3. Initialize knowledge graph
    kg_config = KnowledgeGraphConfig()
    graph = KnowledgeGraphStore(kg_config)
    await graph.initialize()
    print("✅ Knowledge graph initialized")

    # 4. Create hybrid search engine
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI()

    search_engine = create_hybrid_search_engine(database)
    search_engine.set_embedding_client(openai_client)
    print("✅ Hybrid search engine ready")

    return {
        'vector_store': vector_store,
        'embeddings': embedding_generator,
        'search': search_engine,
        'graph': graph
    }

# Run initialization
components = asyncio.run(initialize_phase3())
```

---

## Vector Search Examples

### Example 1: Add and Search Conversations

```python
async def example_conversation_storage():
    """Store and retrieve conversation history"""

    # Add conversation exchanges
    conversation = [
        {
            "user": "What is machine learning?",
            "agent": "Machine learning is a subset of AI that enables systems to learn from data."
        },
        {
            "user": "How does it differ from deep learning?",
            "agent": "Deep learning is a specialized form of ML using neural networks with multiple layers."
        },
        {
            "user": "What are common applications?",
            "agent": "Image recognition, NLP, recommendation systems, and autonomous vehicles."
        }
    ]

    # Store in vector database
    doc_ids = []
    for i, exchange in enumerate(conversation):
        # Combine user and agent text for embedding
        combined_text = f"User: {exchange['user']}\nAgent: {exchange['agent']}"

        doc_id = await vector_store.add_embedding(
            collection_name="conversations",
            text=combined_text,
            metadata={
                "session_id": "demo_session_1",
                "exchange_id": i,
                "timestamp": "2025-01-21T10:30:00Z",
                "user_text": exchange['user'],
                "agent_text": exchange['agent']
            }
        )
        doc_ids.append(doc_id)
        print(f"✅ Stored exchange {i}: {doc_id}")

    # Search for related conversations
    query = "neural networks and AI"
    results = await vector_store.search_similar(
        collection_name="conversations",
        query_text=query,
        n_results=3
    )

    print(f"\n🔍 Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.3f}")
        print(f"   User: {result['metadata']['user_text']}")
        print(f"   Agent: {result['metadata']['agent_text'][:100]}...")

# Run example
asyncio.run(example_conversation_storage())
```

**Output:**
```
✅ Stored exchange 0: a1b2c3d4-...
✅ Stored exchange 1: b2c3d4e5-...
✅ Stored exchange 2: c3d4e5f6-...

🔍 Search results for: 'neural networks and AI'

1. Similarity: 0.924
   User: How does it differ from deep learning?
   Agent: Deep learning is a specialized form of ML using neural networks with multiple layers.

2. Similarity: 0.887
   User: What is machine learning?
   Agent: Machine learning is a subset of AI that enables systems to learn from data.

3. Similarity: 0.812
   User: What are common applications?
   Agent: Image recognition, NLP, recommendation systems, and autonomous vehicles.
```

---

### Example 2: Batch Processing

```python
async def example_batch_import():
    """Import large dataset efficiently"""

    # Sample knowledge base articles
    articles = [
        "Supervised learning uses labeled training data to learn mappings from inputs to outputs.",
        "Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.",
        "Reinforcement learning trains agents through reward and punishment in interactive environments.",
        "Transfer learning adapts pre-trained models to new but related tasks with minimal data.",
        "Ensemble methods combine multiple models to improve prediction accuracy and robustness."
    ]

    # Add metadata
    metadatas = [
        {"category": "learning_types", "difficulty": "beginner", "source": "ml_guide"},
        {"category": "learning_types", "difficulty": "intermediate", "source": "ml_guide"},
        {"category": "learning_types", "difficulty": "advanced", "source": "ml_guide"},
        {"category": "techniques", "difficulty": "advanced", "source": "ml_guide"},
        {"category": "techniques", "difficulty": "intermediate", "source": "ml_guide"}
    ]

    # Batch add (much faster than individual adds)
    print("⏳ Adding batch...")
    import time
    start = time.time()

    doc_ids = await vector_store.add_batch(
        collection_name="conversations",
        texts=articles,
        metadatas=metadatas
    )

    elapsed = time.time() - start
    print(f"✅ Added {len(doc_ids)} documents in {elapsed:.2f}s")
    print(f"   Throughput: {len(doc_ids)/elapsed:.1f} docs/sec")

    # Search with metadata filter
    results = await vector_store.search_similar(
        collection_name="conversations",
        query_text="training with labeled data",
        n_results=3,
        metadata_filter={"category": "learning_types"}
    )

    print(f"\n🔍 Found {len(results)} results in 'learning_types' category")
    for result in results:
        print(f"   - {result['document'][:80]}...")

# Run example
asyncio.run(example_batch_import())
```

---

## Hybrid Search Examples

### Example 3: Adaptive Search Strategy

```python
async def example_adaptive_search():
    """Demonstrate adaptive search strategy selection"""

    from app.search.hybrid_search import create_hybrid_search_engine
    from app.search.config import SearchStrategy

    engine = create_hybrid_search_engine(database)
    engine.set_embedding_client(openai_client)

    # Different query types
    queries = [
        ("What is the concept of backpropagation?", "Conceptual - uses SEMANTIC"),
        ("When was Python created?", "Factual - uses KEYWORD"),
        ("How to implement neural networks?", "Procedural - uses SEMANTIC"),
        ('"machine learning" basics', "Exact phrase - uses KEYWORD")
    ]

    for query, description in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Type: {description}")
        print('='*60)

        # Let system choose strategy
        response = await engine.search(
            query=query,
            strategy=SearchStrategy.ADAPTIVE
        )

        print(f"\n📊 Results:")
        print(f"   Strategy selected: {response.strategy}")
        print(f"   Total results: {response.total_count}")
        print(f"   Vector results: {response.vector_results_count}")
        print(f"   Keyword results: {response.keyword_results_count}")
        print(f"   Execution time: {response.execution_time_ms:.2f}ms")

        # Show query analysis
        analysis = response.query_analysis
        print(f"\n🔍 Query Analysis:")
        print(f"   Intent: {analysis['intent']}")
        print(f"   Keywords: {analysis['keywords']}")
        print(f"   Word count: {analysis['word_count']}")

        # Top result
        if response.results:
            top = response.results[0]
            print(f"\n🏆 Top Result (score: {top['score']:.3f}):")
            print(f"   {top['user_text']}")

# Run example
asyncio.run(example_adaptive_search())
```

**Output:**
```
============================================================
Query: What is the concept of backpropagation?
Type: Conceptual - uses SEMANTIC
============================================================

📊 Results:
   Strategy selected: semantic
   Total results: 5
   Vector results: 5
   Keyword results: 0
   Execution time: 89.45ms

🔍 Query Analysis:
   Intent: conceptual
   Keywords: ['concept', 'backpropagation']
   Word count: 6

🏆 Top Result (score: 0.932):
   How does backpropagation work in neural networks?
```

---

### Example 4: Hybrid Search Comparison

```python
async def example_search_comparison():
    """Compare different search strategies"""

    query = "neural network training techniques"

    strategies = [
        SearchStrategy.SEMANTIC,
        SearchStrategy.KEYWORD,
        SearchStrategy.HYBRID
    ]

    print(f"Query: '{query}'\n")

    for strategy in strategies:
        response = await engine.search(
            query=query,
            strategy=strategy,
            limit=5
        )

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.value.upper()}")
        print('='*60)

        print(f"Execution time: {response.execution_time_ms:.2f}ms")
        print(f"Results found: {response.total_count}")

        print("\nTop 3 Results:")
        for i, result in enumerate(response.results[:3], 1):
            print(f"{i}. [Score: {result['score']:.3f}] {result['user_text'][:60]}...")
            if result.get('vector_score') and result.get('keyword_score'):
                print(f"   Vector: {result['vector_score']:.3f}, Keyword: {result['keyword_score']:.3f}")

# Run example
asyncio.run(example_search_comparison())
```

---

## Knowledge Graph Examples

### Example 5: Building a Concept Graph

```python
async def example_build_concept_graph():
    """Build knowledge graph from conversations"""

    from app.knowledge_graph.graph_store import KnowledgeGraphStore
    from app.knowledge_graph.config import KnowledgeGraphConfig

    config = KnowledgeGraphConfig()
    graph = KnowledgeGraphStore(config)
    await graph.initialize()

    # Add concepts from a learning conversation
    concepts = [
        ("machine learning", "Field of AI focused on learning from data", "artificial intelligence"),
        ("deep learning", "ML using multi-layer neural networks", "machine learning"),
        ("neural networks", "Computing systems inspired by biological neural networks", "deep learning"),
        ("backpropagation", "Algorithm for training neural networks", "neural networks"),
        ("gradient descent", "Optimization algorithm for minimizing loss", "machine learning")
    ]

    print("📚 Building knowledge graph...\n")

    for name, description, topic in concepts:
        await graph.add_concept(
            name=name,
            description=description,
            topic=topic
        )
        print(f"✅ Added concept: {name}")

    # Create relationships
    relationships = [
        ("deep learning", "machine learning", "BUILDS_ON", 0.95),
        ("neural networks", "deep learning", "BUILDS_ON", 0.90),
        ("backpropagation", "neural networks", "BUILDS_ON", 0.85),
        ("gradient descent", "backpropagation", "RELATES_TO", 0.80),
        ("deep learning", "neural networks", "CONTAINS", 0.92)
    ]

    print("\n🔗 Creating relationships...\n")

    for from_c, to_c, rel_type, strength in relationships:
        await graph.add_relationship(
            from_concept=from_c,
            to_concept=to_c,
            relationship_type=rel_type,
            strength=strength
        )
        print(f"✅ {from_c} --[{rel_type}:{strength}]--> {to_c}")

    # Query related concepts
    print("\n\n🔍 Exploring relationships...\n")

    concept = "neural networks"
    related = await graph.get_related_concepts(
        concept=concept,
        max_depth=2,
        min_strength=0.7,
        limit=10
    )

    print(f"Concepts related to '{concept}':")
    for r in related:
        print(f"\n  📌 {r['name']} (distance: {r['distance']}, freq: {r['frequency']})")
        print(f"     Relationships: {r['relationship_types']}")
        print(f"     Strengths: {[f'{s:.2f}' for s in r['strengths']]}")

    # Get graph statistics
    stats = await graph.get_graph_stats()
    print(f"\n\n📊 Graph Statistics:")
    print(f"   Concepts: {stats['concepts']}")
    print(f"   Relationships: {stats['relationships']}")
    print(f"   Topics: {stats['topics']}")

    await graph.close()

# Run example
asyncio.run(example_build_concept_graph())
```

**Output:**
```
📚 Building knowledge graph...

✅ Added concept: machine learning
✅ Added concept: deep learning
✅ Added concept: neural networks
✅ Added concept: backpropagation
✅ Added concept: gradient descent

🔗 Creating relationships...

✅ deep learning --[BUILDS_ON:0.95]--> machine learning
✅ neural networks --[BUILDS_ON:0.9]--> deep learning
✅ backpropagation --[BUILDS_ON:0.85]--> neural networks
✅ gradient descent --[RELATES_TO:0.8]--> backpropagation
✅ deep learning --[CONTAINS:0.92]--> neural networks

🔍 Exploring relationships...

Concepts related to 'neural networks':

  📌 deep learning (distance: 1, freq: 2)
     Relationships: ['BUILDS_ON']
     Strengths: ['0.90']

  📌 backpropagation (distance: 1, freq: 1)
     Relationships: ['BUILDS_ON']
     Strengths: ['0.85']

  📌 machine learning (distance: 2, freq: 3)
     Relationships: ['BUILDS_ON', 'BUILDS_ON']
     Strengths: ['0.90', '0.95']

📊 Graph Statistics:
   Concepts: 5
   Relationships: 5
   Topics: 3
```

---

### Example 6: Session Tracking

```python
async def example_session_tracking():
    """Track concepts discussed in conversation sessions"""

    # Simulate a learning session
    session_id = "learning_session_001"

    # Concepts discussed
    concepts_discussed = [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "neural networks"
    ]

    # Entities mentioned
    entities_mentioned = [
        ("TensorFlow", "PRODUCT"),
        ("PyTorch", "PRODUCT"),
        ("Andrew Ng", "PERSON"),
        ("Stanford University", "ORG")
    ]

    # Add session to graph
    await graph.add_session(
        session_id=session_id,
        concepts=concepts_discussed,
        entities=entities_mentioned,
        metadata={
            "exchange_count": 12,
            "duration": 450,  # seconds
            "user_satisfaction": 4.5
        }
    )

    print(f"✅ Session tracked: {session_id}")
    print(f"   Concepts: {len(concepts_discussed)}")
    print(f"   Entities: {len(entities_mentioned)}")

# Run example
asyncio.run(example_session_tracking())
```

---

## RAG Pipeline Examples

### Example 7: RAG-Enhanced Conversation

```python
async def example_rag_conversation():
    """Demonstrate RAG-enhanced conversation flow"""

    # 1. User asks a question
    user_question = "What are the key differences between supervised and unsupervised learning?"

    print(f"❓ User Question: {user_question}\n")

    # 2. Retrieve relevant context
    print("🔍 Retrieving relevant context...")

    search_response = await engine.search(
        query=user_question,
        strategy=SearchStrategy.HYBRID,
        limit=5
    )

    print(f"   Found {search_response.total_count} relevant conversations")
    print(f"   Search time: {search_response.execution_time_ms:.2f}ms\n")

    # 3. Build context from retrieved documents
    context_parts = []
    for i, result in enumerate(search_response.results[:3], 1):
        context_parts.append(
            f"[Context {i}] User: {result['user_text']}\n"
            f"Agent: {result['agent_text']}"
        )

    context = "\n\n".join(context_parts)

    print("📄 Retrieved Context:")
    print("="*60)
    print(context[:300] + "...")
    print("="*60 + "\n")

    # 4. Build RAG prompt
    rag_prompt = f"""Based on the following conversation history, please answer the user's question.

Conversation History:
{context}

User Question: {user_question}

Please provide a comprehensive answer that builds on the conversation history."""

    # 5. Generate response with Claude (mock)
    print("🤖 Generating RAG-enhanced response...\n")

    # In production, this would call Claude:
    # from anthropic import AsyncAnthropic
    # client = AsyncAnthropic()
    # response = await client.messages.create(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=1500,
    #     messages=[{"role": "user", "content": rag_prompt}]
    # )

    # Mock response for demo
    rag_response = """Based on our previous discussions:

**Supervised Learning:**
- Uses labeled training data
- The algorithm learns from examples with known outputs
- Examples: classification, regression
- Requires human annotation effort

**Unsupervised Learning:**
- Works with unlabeled data
- Finds hidden patterns and structures
- Examples: clustering, dimensionality reduction
- No need for labeled datasets

The key difference is that supervised learning needs labeled data to learn the input-output mapping, while unsupervised learning discovers patterns in data without predefined labels.

[References: Context 1, Context 2]"""

    print("💡 RAG-Enhanced Response:")
    print("="*60)
    print(rag_response)
    print("="*60)

    return {
        'question': user_question,
        'context_used': len(search_response.results),
        'response': rag_response
    }

# Run example
result = asyncio.run(example_rag_conversation())
```

---

## Advanced Use Cases

### Example 8: Multi-Modal Search

```python
async def example_multimodal_search():
    """Search across different content types"""

    # Add different content types with metadata
    contents = [
        {
            "text": "Neural networks consist of layers of interconnected nodes",
            "metadata": {"type": "definition", "media": "text"}
        },
        {
            "text": "Code example: model = Sequential([Dense(128), Dropout(0.2)])",
            "metadata": {"type": "code", "media": "text", "language": "python"}
        },
        {
            "text": "Diagram showing forward and backward propagation flow",
            "metadata": {"type": "diagram", "media": "image"}
        }
    ]

    # Add to vector store
    for content in contents:
        await vector_store.add_embedding(
            collection_name="conversations",
            text=content["text"],
            metadata=content["metadata"]
        )

    # Search for code examples
    code_results = await vector_store.search_similar(
        collection_name="conversations",
        query_text="show me code for neural network",
        metadata_filter={"type": "code"}
    )

    print("🔍 Code Examples Found:")
    for result in code_results:
        print(f"   {result['document']}")

# Run example
asyncio.run(example_multimodal_search())
```

---

### Example 9: Personalized Context

```python
async def example_personalized_context():
    """Retrieve context specific to user's learning history"""

    user_id = "user_123"

    # Search within user's sessions only
    results = await vector_store.search_similar(
        collection_name="conversations",
        query_text="explain gradient descent",
        metadata_filter={"user_id": user_id},
        n_results=10
    )

    # Prioritize recent conversations
    from datetime import datetime, timedelta

    def recency_score(timestamp_str):
        """Calculate recency score (exponential decay)"""
        timestamp = datetime.fromisoformat(timestamp_str)
        age_days = (datetime.now() - timestamp).days
        decay_rate = 0.1  # 10% decay per day
        return 1.0 * (0.9 ** age_days)

    # Re-rank by combining similarity and recency
    for result in results:
        similarity = result['similarity']
        recency = recency_score(result['metadata']['timestamp'])
        result['combined_score'] = 0.7 * similarity + 0.3 * recency

    # Sort by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)

    print(f"📊 Personalized results for {user_id}:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Combined Score: {result['combined_score']:.3f}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Text: {result['document'][:80]}...")

# Run example
asyncio.run(example_personalized_context())
```

---

### Example 10: Real-time Knowledge Graph Update

```python
async def example_realtime_graph_update():
    """Update knowledge graph in real-time during conversation"""

    async def process_conversation_exchange(user_text, agent_text, session_id):
        """Process a single conversation exchange"""

        # 1. Extract concepts (simplified - would use NLP in production)
        def extract_concepts(text):
            # Simple keyword extraction
            keywords = ["machine learning", "neural network", "gradient descent"]
            return [k for k in keywords if k in text.lower()]

        user_concepts = extract_concepts(user_text)
        agent_concepts = extract_concepts(agent_text)
        all_concepts = set(user_concepts + agent_concepts)

        # 2. Add concepts to graph
        for concept in all_concepts:
            await graph.add_concept(name=concept)

        # 3. Create relationships between co-occurring concepts
        concepts_list = list(all_concepts)
        for i, concept1 in enumerate(concepts_list):
            for concept2 in concepts_list[i+1:]:
                await graph.add_relationship(
                    from_concept=concept1,
                    to_concept=concept2,
                    relationship_type="RELATES_TO",
                    strength=0.5,
                    context=session_id
                )

        # 4. Link to session
        await graph.add_session(
            session_id=session_id,
            concepts=list(all_concepts),
            entities=[],
            metadata={"exchange_type": "learning"}
        )

        print(f"✅ Updated graph: {len(all_concepts)} concepts, session {session_id}")

    # Simulate conversation
    exchanges = [
        ("What is gradient descent?", "Gradient descent is an optimization algorithm...", "sess_1"),
        ("How is it used in neural networks?", "Neural networks use gradient descent to minimize loss...", "sess_1"),
    ]

    for user, agent, session in exchanges:
        await process_conversation_exchange(user, agent, session)

    # View updated graph
    stats = await graph.get_graph_stats()
    print(f"\n📊 Updated Graph: {stats['concepts']} concepts, {stats['relationships']} relationships")

# Run example
asyncio.run(example_realtime_graph_update())
```

---

## Complete Workflow Example

```python
async def complete_workflow_demo():
    """Complete end-to-end workflow demonstration"""

    print("="*60)
    print("PHASE 3: COMPLETE WORKFLOW DEMONSTRATION")
    print("="*60 + "\n")

    # 1. Initialize all components
    print("1️⃣  Initializing components...")
    components = await initialize_phase3()
    print()

    # 2. Add sample conversations
    print("2️⃣  Adding sample conversations...")
    await example_conversation_storage()
    print()

    # 3. Build knowledge graph
    print("3️⃣  Building knowledge graph...")
    await example_build_concept_graph()
    print()

    # 4. Execute hybrid search
    print("4️⃣  Executing hybrid search...")
    await example_adaptive_search()
    print()

    # 5. RAG-enhanced response
    print("5️⃣  Generating RAG-enhanced response...")
    await example_rag_conversation()
    print()

    print("="*60)
    print("✅ WORKFLOW COMPLETE")
    print("="*60)

# Run complete demo
asyncio.run(complete_workflow_demo())
```

---

**Next Steps:**
- See [Phase 3 Implementation Guide](PHASE3_IMPLEMENTATION_GUIDE.md) for setup
- See [Phase 3 API Reference](PHASE3_VECTOR_API_REFERENCE.md) for detailed API docs
- See [Phase 3 Testing Guide](PHASE3_TESTING_GUIDE.md) for testing strategies

---

**Questions?** Open an issue on GitHub or check the [documentation](../README.md).
