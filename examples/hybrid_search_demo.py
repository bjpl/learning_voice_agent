#!/usr/bin/env python3
"""
Hybrid Search Demo - Phase 3
Demonstrates the hybrid search capabilities combining vector similarity and FTS5
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import Database
from app.search import create_hybrid_search_engine, SearchStrategy
from app.search.vector_store import vector_store
import numpy as np


async def populate_sample_data(db: Database):
    """Populate database with sample learning captures"""
    print("📚 Populating sample data...")

    sample_conversations = [
        {
            "session": "ml-basics",
            "exchanges": [
                ("What is machine learning?",
                 "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make decisions."),
                ("Explain supervised learning",
                 "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model is trained on input-output pairs and learns to predict outputs for new inputs. Common examples include classification and regression tasks."),
                ("What are neural networks?",
                 "Neural networks are computational models inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns through training."),
            ]
        },
        {
            "session": "web-dev",
            "exchanges": [
                ("How do I build a REST API?",
                 "To build a REST API, you need to: 1) Choose a framework (like FastAPI, Flask, or Express), 2) Define your resources and endpoints, 3) Implement CRUD operations, 4) Add authentication and validation, 5) Document your API. Use HTTP methods (GET, POST, PUT, DELETE) appropriately."),
                ("What is FastAPI?",
                 "FastAPI is a modern, high-performance Python web framework for building APIs. It's based on standard Python type hints, provides automatic API documentation, has built-in validation, and is one of the fastest Python frameworks available."),
                ("Explain webhooks",
                 "Webhooks are user-defined HTTP callbacks triggered by specific events. Instead of polling for updates, webhooks push data to your application when events occur. They're commonly used for real-time integrations and event-driven architectures."),
            ]
        },
        {
            "session": "distributed-systems",
            "exchanges": [
                ("What are distributed systems?",
                 "Distributed systems are collections of independent computers that appear to users as a single coherent system. They work together to achieve common goals, sharing resources and coordinating activities across network boundaries."),
                ("Explain consensus algorithms",
                 "Consensus algorithms enable multiple nodes in a distributed system to agree on a single value or state, even in the presence of failures. Examples include Paxos, Raft, and Byzantine fault tolerance algorithms. They're crucial for maintaining consistency."),
                ("What is CAP theorem?",
                 "The CAP theorem states that a distributed system can only guarantee two of three properties simultaneously: Consistency (all nodes see the same data), Availability (system remains operational), and Partition tolerance (system works despite network failures)."),
            ]
        },
        {
            "session": "databases",
            "exchanges": [
                ("SQL vs NoSQL databases?",
                 "SQL databases are relational, use structured schemas, and support ACID transactions. They're great for complex queries and data integrity. NoSQL databases are schema-flexible, horizontally scalable, and optimized for specific data models (document, key-value, graph, columnar)."),
                ("What is database indexing?",
                 "Database indexing creates data structures that improve query performance by allowing faster data retrieval. Indexes work like book indexes - they help locate data without scanning entire tables. Trade-off: faster reads but slower writes and additional storage."),
                ("Explain database normalization",
                 "Normalization organizes database tables to reduce redundancy and improve data integrity. It involves dividing tables into smaller tables and defining relationships. Common forms are 1NF, 2NF, 3NF, and BCNF, each with increasing levels of normalization."),
            ]
        }
    ]

    for conversation in sample_conversations:
        session_id = conversation["session"]
        for user_text, agent_text in conversation["exchanges"]:
            await db.save_exchange(session_id, user_text, agent_text)

    print(f"✅ Added {sum(len(c['exchanges']) for c in sample_conversations)} sample exchanges")


async def generate_sample_embeddings(db: Database):
    """Generate embeddings for sample data (requires OpenAI API key)"""
    try:
        import os
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set - skipping embedding generation")
            print("   Semantic search will not be available in this demo")
            return False

        print("🧠 Generating embeddings...")
        client = AsyncOpenAI(api_key=api_key)

        # Get all captures
        async with db.get_connection() as conn:
            cursor = await conn.execute("SELECT id, user_text, agent_text FROM captures")
            captures = await cursor.fetchall()

        for capture in captures:
            capture_id = capture['id']
            combined_text = f"User: {capture['user_text']}\nAgent: {capture['agent_text']}"

            # Generate embedding
            response = await client.embeddings.create(
                input=combined_text,
                model="text-embedding-ada-002"
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Store embedding
            await vector_store.add_embedding(
                capture_id=capture_id,
                embedding=embedding,
                model="text-embedding-ada-002"
            )

        print(f"✅ Generated embeddings for {len(captures)} captures")
        return True

    except Exception as e:
        print(f"⚠️  Failed to generate embeddings: {e}")
        return False


async def demo_search_strategies(engine):
    """Demonstrate different search strategies"""

    test_queries = [
        ("machine learning basics", "Conceptual query about ML"),
        ("REST API", "Short technical term"),
        ("How do distributed systems handle failures?", "Long question"),
        ("\"consensus algorithms\"", "Exact phrase search"),
        ("database", "Generic term"),
    ]

    print("\n" + "="*80)
    print("🔍 HYBRID SEARCH DEMONSTRATION")
    print("="*80)

    for query, description in test_queries:
        print(f"\n📝 Query: '{query}'")
        print(f"   ({description})")
        print("-" * 80)

        # Test different strategies
        strategies = [
            SearchStrategy.ADAPTIVE,
            SearchStrategy.KEYWORD,
            SearchStrategy.SEMANTIC,
            SearchStrategy.HYBRID
        ]

        for strategy in strategies:
            try:
                response = await engine.search(
                    query=query,
                    strategy=strategy,
                    limit=3
                )

                print(f"\n  Strategy: {strategy.value.upper()}")
                print(f"  Execution time: {response.execution_time_ms:.2f}ms")
                print(f"  Results: {response.total_count}")

                if strategy == SearchStrategy.HYBRID:
                    print(f"  Vector results: {response.vector_results_count}")
                    print(f"  Keyword results: {response.keyword_results_count}")

                # Show top result
                if response.results:
                    result = response.results[0]
                    print(f"\n  Top Result (score: {result['score']:.3f}):")
                    print(f"    User: {result['user_text'][:80]}...")
                    print(f"    Agent: {result['agent_text'][:80]}...")

            except Exception as e:
                print(f"  ❌ Error with {strategy.value}: {e}")

        print()


async def demo_query_analysis(engine):
    """Demonstrate query analysis capabilities"""

    print("\n" + "="*80)
    print("🧪 QUERY ANALYSIS DEMONSTRATION")
    print("="*80)

    test_queries = [
        "What is machine learning?",
        "API implementation",
        "Explain the difference between SQL and NoSQL",
        "database indexing performance optimization techniques"
    ]

    for query in test_queries:
        response = await engine.search(query, strategy=SearchStrategy.ADAPTIVE, limit=1)
        analysis = response.query_analysis

        print(f"\n📝 Query: '{query}'")
        print(f"  Intent: {analysis.get('intent', 'unknown')}")
        print(f"  Keywords: {', '.join(analysis.get('keywords', []))}")
        print(f"  Suggested strategy: {analysis.get('suggested_strategy', 'unknown')}")
        print(f"  Word count: {analysis.get('word_count', 0)}")
        print(f"  Is short query: {analysis.get('is_short', False)}")


async def main():
    """Main demo function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 3: HYBRID SEARCH SYSTEM DEMO                       ║
║                                                                              ║
║  Combining Vector Similarity Search with SQLite FTS5 Keyword Search         ║
║  Using Reciprocal Rank Fusion (RRF) for optimal results                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize database
    db = Database(db_path="demo_hybrid_search.db")
    await db.initialize()

    # Initialize vector store
    await vector_store.initialize()

    # Populate sample data
    await populate_sample_data(db)

    # Generate embeddings (if OpenAI key available)
    embeddings_available = await generate_sample_embeddings(db)

    # Create hybrid search engine
    print("\n🔧 Initializing hybrid search engine...")
    engine = create_hybrid_search_engine(db)

    # Set up OpenAI client for semantic search
    if embeddings_available:
        import os
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        engine.set_embedding_client(client)
        print("✅ Semantic search enabled")
    else:
        print("⚠️  Semantic search disabled (no OpenAI API key)")

    # Run demos
    await demo_search_strategies(engine)
    await demo_query_analysis(engine)

    # Statistics
    print("\n" + "="*80)
    print("📊 STATISTICS")
    print("="*80)

    db_stats = await db.get_stats()
    vector_stats = await vector_store.get_stats()

    print(f"\n  Database:")
    print(f"    Total captures: {db_stats.get('total_captures', 0)}")
    print(f"    Unique sessions: {db_stats.get('unique_sessions', 0)}")

    print(f"\n  Vector Store:")
    print(f"    Total embeddings: {vector_stats.get('total_embeddings', 0)}")
    print(f"    Cache size: {vector_stats.get('cache_size', 0)}")
    print(f"    Embedding model: {vector_stats.get('model', 'N/A')}")

    print("\n" + "="*80)
    print("✅ Demo complete! Database saved to: demo_hybrid_search.db")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
