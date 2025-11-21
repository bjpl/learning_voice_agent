#!/usr/bin/env python3
"""
Phase 3: RAG System Integration Example

This example demonstrates the complete RAG pipeline:
1. Retrieve relevant conversation history
2. Build formatted context
3. Generate Claude response with citations

Run:
    python examples/phase3_rag_integration.py
"""

import asyncio
import os
from datetime import datetime
from anthropic import AsyncAnthropic

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Database
from app.search.hybrid_search import create_hybrid_search_engine
from app.search.vector_store import vector_store
from app.rag import (
    RAGRetriever,
    ContextBuilder,
    RAGGenerator,
    rag_config,
    get_performance_profile,
    update_rag_config
)
from app.rag.context_builder import ContextFormat
from app.logger import db_logger


async def setup_database():
    """Initialize database with sample conversation data"""
    print("=" * 60)
    print("Setting up database with sample conversations...")
    print("=" * 60)

    db = Database()
    await db.initialize()
    await vector_store.initialize()

    # Add sample conversations
    sample_conversations = [
        {
            "session_id": "session-001",
            "user_text": "What are the best practices for machine learning projects?",
            "agent_text": "Key best practices for ML projects include: 1) Start with a clear problem definition, 2) Ensure high-quality training data, 3) Use version control for code and models, 4) Implement proper validation and testing, 5) Monitor model performance in production.",
        },
        {
            "session_id": "session-001",
            "user_text": "How do I choose between different ML algorithms?",
            "agent_text": "Choosing an ML algorithm depends on: 1) Your data type and size, 2) Problem complexity, 3) Interpretability requirements, 4) Training time constraints, 5) Accuracy needs. For structured data, start with tree-based models like Random Forest or XGBoost. For images/text, use deep learning.",
        },
        {
            "session_id": "session-002",
            "user_text": "What is RAG in the context of LLMs?",
            "agent_text": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by retrieving relevant documents from a knowledge base and using them as context. This allows the model to access up-to-date information and reduce hallucinations.",
        },
        {
            "session_id": "session-002",
            "user_text": "How does hybrid search work?",
            "agent_text": "Hybrid search combines vector similarity search (semantic) with keyword search (lexical) using techniques like Reciprocal Rank Fusion. This leverages both semantic understanding and exact keyword matching for better retrieval quality.",
        },
        {
            "session_id": "session-003",
            "user_text": "What are embeddings in NLP?",
            "agent_text": "Embeddings are dense vector representations of text that capture semantic meaning. Words or sentences with similar meanings have similar embeddings. They enable machine learning models to understand and process natural language effectively.",
        },
    ]

    for conv in sample_conversations:
        capture_id = await db.save_capture(
            session_id=conv["session_id"],
            user_text=conv["user_text"],
            agent_text=conv["agent_text"]
        )
        print(f"  ✓ Added conversation {capture_id}: {conv['user_text'][:50]}...")

    print(f"\nAdded {len(sample_conversations)} sample conversations")
    print("=" * 60)
    print()

    return db


async def demonstrate_basic_rag():
    """Demonstrate basic RAG pipeline"""
    print("=" * 60)
    print("DEMO 1: Basic RAG Pipeline")
    print("=" * 60)

    # Setup
    db = await setup_database()
    hybrid_search = create_hybrid_search_engine(db)

    # Set OpenAI client for embeddings (if available)
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        hybrid_search.set_embedding_client(openai_client)
        print("✓ OpenAI client configured for embeddings")
    except Exception as e:
        print(f"⚠ OpenAI client not available: {e}")
        print("  Continuing with keyword-only search\n")

    # Initialize Anthropic client
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Initialize RAG components
    retriever = RAGRetriever(db, hybrid_search)
    context_builder = ContextBuilder()
    generator = RAGGenerator(anthropic_client)

    print()

    # Test query
    query = "What is RAG and how does it work?"

    print(f"Query: {query}\n")

    # Step 1: Retrieve
    print("Step 1: Retrieving relevant documents...")
    retrieval_response = await retriever.retrieve(query, top_k=3)

    print(f"  ✓ Retrieved {retrieval_response.total_retrieved} documents")
    print(f"  ✓ Strategy: {retrieval_response.strategy_used}")
    print(f"  ✓ Execution time: {retrieval_response.execution_time_ms:.2f}ms\n")

    if retrieval_response.results:
        print("  Top results:")
        for i, result in enumerate(retrieval_response.results[:3], 1):
            print(f"    {i}. Score: {result.score:.3f} | {result.user_text[:60]}...")
        print()

    # Step 2: Build context
    print("Step 2: Building context...")
    context = await context_builder.build_context(
        retrieval_results=retrieval_response.results,
        query=query,
        format_type=ContextFormat.STRUCTURED
    )

    print(f"  ✓ Context built with {context.document_count} documents")
    print(f"  ✓ Total tokens: {context.total_tokens}")
    print(f"  ✓ Truncated: {context.is_truncated}")
    print(f"  ✓ Execution time: {context.build_time_ms:.2f}ms\n")

    # Step 3: Generate
    print("Step 3: Generating response...")
    generation_response = await generator.generate(
        query=query,
        context=context
    )

    print(f"  ✓ Response generated")
    print(f"  ✓ Mode: {generation_response.mode}")
    print(f"  ✓ Tokens used: {generation_response.tokens_used}")
    print(f"  ✓ Citations: {len(generation_response.citations)}")
    print(f"  ✓ Execution time: {generation_response.generation_time_ms:.2f}ms\n")

    print("-" * 60)
    print("RESPONSE:")
    print("-" * 60)
    print(generation_response.response_text)
    print("-" * 60)

    if generation_response.citations:
        print("\nCITATIONS:")
        for i, citation in enumerate(generation_response.citations, 1):
            print(f"  [{i}] Document {citation.document_id} | Score: {citation.relevance_score:.3f}")
            print(f"      {citation.excerpt[:80]}...")

    print("\n" + "=" * 60)
    print()


async def demonstrate_session_scoped_rag():
    """Demonstrate session-scoped RAG"""
    print("=" * 60)
    print("DEMO 2: Session-Scoped RAG")
    print("=" * 60)

    db = await setup_database()
    hybrid_search = create_hybrid_search_engine(db)

    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        hybrid_search.set_embedding_client(openai_client)
    except:
        pass

    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Enable session-scoped search
    update_rag_config(session_scoped_search=True)

    retriever = RAGRetriever(db, hybrid_search)
    context_builder = ContextBuilder()
    generator = RAGGenerator(anthropic_client)

    print()

    query = "What did we discuss about machine learning?"
    session_id = "session-001"

    print(f"Query: {query}")
    print(f"Session: {session_id}\n")

    # Retrieve within session
    retrieval_response = await retriever.retrieve(
        query=query,
        session_id=session_id,
        top_k=5
    )

    print(f"✓ Retrieved {retrieval_response.total_retrieved} documents from session")
    print(f"✓ Session scoped: {retrieval_response.session_scoped}\n")

    if retrieval_response.results:
        print("Results:")
        for i, result in enumerate(retrieval_response.results, 1):
            print(f"  {i}. [{result.session_id}] {result.user_text[:50]}...")
        print()

    # Build context and generate
    context = await context_builder.build_context(retrieval_response.results, query)
    generation_response = await generator.generate(query, context)

    print("-" * 60)
    print("RESPONSE:")
    print("-" * 60)
    print(generation_response.response_text)
    print("-" * 60)
    print("\n" + "=" * 60)
    print()


async def demonstrate_performance_profiles():
    """Demonstrate performance profiles"""
    print("=" * 60)
    print("DEMO 3: Performance Profiles")
    print("=" * 60)

    profiles = ["fast", "balanced", "quality"]

    print("\nAvailable profiles:\n")

    for profile in profiles:
        config = get_performance_profile(profile)
        print(f"{profile.upper()} Profile:")
        print(f"  - Retrieval top-k: {config['retrieval_top_k']}")
        print(f"  - Relevance threshold: {config['relevance_threshold']}")
        print(f"  - Max context tokens: {config['max_context_tokens']}")
        print(f"  - Max generation tokens: {config['generation_max_tokens']}")
        print()

    print("Usage:")
    print("  from app.rag import update_rag_config, get_performance_profile")
    print("  update_rag_config(**get_performance_profile('fast'))")

    print("\n" + "=" * 60)
    print()


async def demonstrate_batch_generation():
    """Demonstrate batch generation"""
    print("=" * 60)
    print("DEMO 4: Batch Generation")
    print("=" * 60)

    db = await setup_database()
    hybrid_search = create_hybrid_search_engine(db)

    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        hybrid_search.set_embedding_client(openai_client)
    except:
        pass

    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    retriever = RAGRetriever(db, hybrid_search)
    context_builder = ContextBuilder()
    generator = RAGGenerator(anthropic_client)

    print()

    queries = [
        "What is RAG?",
        "What are ML best practices?",
        "What are embeddings?"
    ]

    print(f"Processing {len(queries)} queries concurrently...\n")

    # Retrieve contexts
    contexts = []
    for query in queries:
        retrieval = await retriever.retrieve(query, top_k=2)
        context = await context_builder.build_context(
            retrieval.results,
            query,
            format_type=ContextFormat.COMPACT
        )
        contexts.append(context)

    # Generate batch
    start_time = datetime.now()
    responses = await generator.generate_batch(queries, contexts)
    execution_time = (datetime.now() - start_time).total_seconds() * 1000

    print(f"✓ Generated {len(responses)} responses")
    print(f"✓ Total time: {execution_time:.2f}ms")
    print(f"✓ Average time per query: {execution_time / len(queries):.2f}ms\n")

    for i, (query, response) in enumerate(zip(queries, responses), 1):
        print(f"{i}. Q: {query}")
        print(f"   A: {response.response_text[:100]}...")
        print(f"   Tokens: {response.tokens_used}\n")

    print("=" * 60)
    print()


async def demonstrate_fallback():
    """Demonstrate fallback behavior"""
    print("=" * 60)
    print("DEMO 5: Fallback Behavior")
    print("=" * 60)

    db = await setup_database()
    hybrid_search = create_hybrid_search_engine(db)
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    retriever = RAGRetriever(db, hybrid_search)
    context_builder = ContextBuilder()
    generator = RAGGenerator(anthropic_client)

    print()

    # Query with no relevant results
    query = "What is quantum computing in detail?"

    print(f"Query: {query}")
    print("(This query has no relevant conversation history)\n")

    # Retrieve
    retrieval_response = await retriever.retrieve(query, top_k=5)

    print(f"✓ Retrieved {retrieval_response.total_retrieved} documents")

    if retrieval_response.total_retrieved == 0:
        print("  ⚠ No relevant context found")
        print("  → Falling back to basic generation mode\n")
        context = None
    else:
        context = await context_builder.build_context(retrieval_response.results, query)

    # Generate (will use basic mode if no context)
    generation_response = await generator.generate(query, context)

    print(f"✓ Response generated")
    print(f"✓ Mode: {generation_response.mode}")
    print(f"✓ Context used: {generation_response.context_used}\n")

    print("-" * 60)
    print("RESPONSE:")
    print("-" * 60)
    print(generation_response.response_text)
    print("-" * 60)

    print("\n" + "=" * 60)
    print()


async def main():
    """Run all demonstrations"""
    print("\n")
    print("=" * 60)
    print("PHASE 3: RAG SYSTEM INTEGRATION EXAMPLES")
    print("=" * 60)
    print()

    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Please set your API key: export ANTHROPIC_API_KEY='your-key'")
        return

    print("Configuration:")
    print(f"  - Max context tokens: {rag_config.max_context_tokens}")
    print(f"  - Retrieval top-k: {rag_config.retrieval_top_k}")
    print(f"  - Relevance threshold: {rag_config.relevance_threshold}")
    print(f"  - Generation model: {rag_config.generation_model}")
    print()

    # Run demonstrations
    try:
        await demonstrate_basic_rag()
        await demonstrate_session_scoped_rag()
        await demonstrate_performance_profiles()
        await demonstrate_batch_generation()
        await demonstrate_fallback()

        print("=" * 60)
        print("ALL DEMOS COMPLETE ✅")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
