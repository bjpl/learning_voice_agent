"""
RAG (Retrieval-Augmented Generation) System

PHASE 3: Production-Ready RAG Implementation

ARCHITECTURE:
- RAGRetriever: Hybrid search for relevant conversation history
- ContextBuilder: Assemble and format retrieved context
- RAGGenerator: Claude-powered generation with citations

PATTERN: Three-stage pipeline
WHY: Separation of concerns, testability, reusability
RESILIENCE: Graceful degradation at each stage

USAGE:
    from app.rag import RAGRetriever, ContextBuilder, RAGGenerator

    # Initialize components
    retriever = RAGRetriever(database, hybrid_search_engine)
    context_builder = ContextBuilder()
    generator = RAGGenerator(anthropic_client)

    # Execute RAG pipeline
    docs = await retriever.retrieve(query, session_id)
    context = await context_builder.build_context(docs, query)
    response = await generator.generate(query, context)
"""

from app.rag.config import (
    RAGConfig,
    rag_config,
    get_rag_config,
    update_rag_config,
    get_performance_profile
)

from app.rag.retriever import (
    RAGRetriever,
    RetrievalResult,
    RetrievalResponse
)

from app.rag.context_builder import (
    ContextBuilder,
    ContextDocument,
    BuiltContext
)

from app.rag.generator import (
    RAGGenerator,
    GenerationResponse,
    Citation
)

__all__ = [
    # Configuration
    "RAGConfig",
    "rag_config",
    "get_rag_config",
    "update_rag_config",
    "get_performance_profile",

    # Retrieval
    "RAGRetriever",
    "RetrievalResult",
    "RetrievalResponse",

    # Context Building
    "ContextBuilder",
    "ContextDocument",
    "BuiltContext",

    # Generation
    "RAGGenerator",
    "GenerationResponse",
    "Citation",
]

__version__ = "1.0.0"
