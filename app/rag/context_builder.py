"""
RAG Context Builder - Assemble Retrieved Context

SPECIFICATION:
- Assemble retrieved documents into coherent context
- Prioritize recent + relevant information
- Manage context window (token limits)
- Summarize if context too large
- Format for Claude consumption

ARCHITECTURE:
- Token-aware context assembly
- Automatic summarization for large contexts
- Structured formatting with metadata
- Deduplication and cleanup

PATTERN: Builder pattern with token management
WHY: Control context size for optimal generation
RESILIENCE: Graceful handling of oversized contexts
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from app.logger import db_logger
from app.rag.config import RAGConfig, rag_config
from app.rag.retriever import RetrievalResult


class ContextFormat(str, Enum):
    """Context formatting options"""
    STRUCTURED = "structured"  # Formatted with clear sections
    COMPACT = "compact"        # Minimal formatting
    CONVERSATION = "conversation"  # Dialog format


@dataclass
class ContextDocument:
    """
    Individual document in context

    CONCEPT: Enriched retrieval result with formatting
    WHY: Additional metadata for citation and display
    """
    id: int
    session_id: str
    timestamp: str
    user_text: str
    agent_text: str
    score: float
    rank: int
    age_days: float

    # Token counts
    tokens: int = 0

    # Formatting
    is_summarized: bool = False
    summary: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BuiltContext:
    """
    Complete assembled context

    CONCEPT: Token-managed context with metadata
    WHY: Structured context for generation
    """
    query: str
    documents: List[ContextDocument]
    formatted_context: str

    # Token statistics
    total_tokens: int
    max_tokens: int
    is_truncated: bool = False
    is_summarized: bool = False

    # Metadata
    document_count: int = 0
    format_type: str = "structured"
    build_time_ms: float = 0.0

    # Context breakdown
    context_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "documents": [d.to_dict() for d in self.documents],
            "formatted_context": self.formatted_context,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "is_truncated": self.is_truncated,
            "is_summarized": self.is_summarized,
            "document_count": self.document_count,
            "format_type": self.format_type,
            "build_time_ms": self.build_time_ms,
            "context_metadata": self.context_metadata,
        }


class ContextBuilder:
    """
    Build formatted context from retrieved documents

    ALGORITHM:
    1. Convert retrieval results to documents
    2. Count tokens for each document
    3. Fit documents within token budget
    4. Format context based on strategy
    5. Summarize if needed

    PATTERN: Token-aware assembly with summarization fallback
    WHY: Maximize context quality within token limits
    RESILIENCE: Automatic summarization prevents overflow
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize context builder

        Args:
            config: RAG configuration (uses default if None)
        """
        self.config = config or rag_config

        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (for GPT-4 and Claude)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                db_logger.warning(
                    "tiktoken_initialization_failed",
                    error=str(e),
                    fallback="character-based estimation"
                )
                self.tokenizer = None
        else:
            db_logger.warning(
                "tiktoken_not_available",
                fallback="character-based token estimation"
            )
            self.tokenizer = None

        db_logger.info(
            "context_builder_initialized",
            max_tokens=self.config.max_context_tokens,
            tokenizer=self.tokenizer is not None
        )

    async def build_context(
        self,
        retrieval_results: List[RetrievalResult],
        query: str,
        format_type: ContextFormat = ContextFormat.STRUCTURED,
        max_tokens: Optional[int] = None
    ) -> BuiltContext:
        """
        Build formatted context from retrieval results

        ALGORITHM:
        1. Convert results to documents
        2. Count tokens
        3. Fit within budget
        4. Format context
        5. Summarize if needed

        Args:
            retrieval_results: Retrieved documents
            query: Original query
            format_type: How to format the context
            max_tokens: Override max tokens from config

        Returns:
            BuiltContext with formatted text and metadata
        """
        start_time = datetime.now()
        max_tokens = max_tokens or self.config.max_context_tokens

        try:
            db_logger.info(
                "context_building_started",
                query=query[:100],
                document_count=len(retrieval_results),
                max_tokens=max_tokens,
                format_type=format_type.value
            )

            # Convert retrieval results to documents
            documents = self._create_documents(retrieval_results)

            # Fit documents within token budget
            fitted_docs, is_truncated = await self._fit_to_budget(
                documents=documents,
                max_tokens=max_tokens
            )

            # Format context
            formatted_context = self._format_context(
                documents=fitted_docs,
                query=query,
                format_type=format_type
            )

            # Count final tokens
            total_tokens = self._count_tokens(formatted_context)

            # Summarize if still over budget
            is_summarized = False
            if total_tokens > max_tokens and self.config.enable_context_summarization:
                formatted_context = await self._summarize_context(
                    formatted_context,
                    max_tokens
                )
                total_tokens = self._count_tokens(formatted_context)
                is_summarized = True

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            db_logger.info(
                "context_building_complete",
                query=query[:100],
                document_count=len(fitted_docs),
                total_tokens=total_tokens,
                is_truncated=is_truncated,
                is_summarized=is_summarized,
                execution_time_ms=round(execution_time, 2)
            )

            return BuiltContext(
                query=query,
                documents=fitted_docs,
                formatted_context=formatted_context,
                total_tokens=total_tokens,
                max_tokens=max_tokens,
                is_truncated=is_truncated,
                is_summarized=is_summarized,
                document_count=len(fitted_docs),
                format_type=format_type.value,
                build_time_ms=round(execution_time, 2),
                context_metadata={
                    "original_document_count": len(retrieval_results),
                    "fitted_document_count": len(fitted_docs),
                    "truncated_count": len(retrieval_results) - len(fitted_docs)
                }
            )

        except Exception as e:
            db_logger.error(
                "context_building_failed",
                query=query[:100],
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )

            # Fallback: return empty context
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return BuiltContext(
                query=query,
                documents=[],
                formatted_context="",
                total_tokens=0,
                max_tokens=max_tokens,
                is_truncated=False,
                is_summarized=False,
                document_count=0,
                format_type=format_type.value,
                build_time_ms=round(execution_time, 2),
                context_metadata={"error": str(e)}
            )

    def _create_documents(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[ContextDocument]:
        """Convert retrieval results to context documents"""
        documents = []

        for result in retrieval_results:
            # Count tokens for this document
            doc_text = f"{result.user_text}\n{result.agent_text}"
            tokens = self._count_tokens(doc_text)

            documents.append(ContextDocument(
                id=result.id,
                session_id=result.session_id,
                timestamp=result.timestamp,
                user_text=result.user_text,
                agent_text=result.agent_text,
                score=result.final_score if hasattr(result, 'final_score') else result.score,
                rank=result.rank,
                age_days=result.age_days if hasattr(result, 'age_days') else 0.0,
                tokens=tokens
            ))

        return documents

    async def _fit_to_budget(
        self,
        documents: List[ContextDocument],
        max_tokens: int
    ) -> Tuple[List[ContextDocument], bool]:
        """
        Fit documents within token budget

        ALGORITHM:
        1. Reserve tokens for formatting overhead (~200 tokens)
        2. Greedily select documents in rank order
        3. Stop when budget exceeded

        Returns:
            (fitted_documents, is_truncated)
        """
        # Reserve tokens for formatting
        overhead_tokens = 200
        available_tokens = max_tokens - overhead_tokens

        fitted = []
        used_tokens = 0
        is_truncated = False

        for doc in documents:
            if used_tokens + doc.tokens <= available_tokens:
                fitted.append(doc)
                used_tokens += doc.tokens
            else:
                is_truncated = True
                break

        db_logger.debug(
            "context_token_fitting",
            original_count=len(documents),
            fitted_count=len(fitted),
            used_tokens=used_tokens,
            available_tokens=available_tokens,
            is_truncated=is_truncated
        )

        return fitted, is_truncated

    def _format_context(
        self,
        documents: List[ContextDocument],
        query: str,
        format_type: ContextFormat
    ) -> str:
        """
        Format documents into context string

        PATTERN: Template-based formatting
        WHY: Consistent, readable context for Claude
        """
        if not documents:
            return ""

        if format_type == ContextFormat.STRUCTURED:
            return self._format_structured(documents, query)
        elif format_type == ContextFormat.COMPACT:
            return self._format_compact(documents)
        elif format_type == ContextFormat.CONVERSATION:
            return self._format_conversation(documents)
        else:
            return self._format_structured(documents, query)

    def _format_structured(
        self,
        documents: List[ContextDocument],
        query: str
    ) -> str:
        """
        Format context with clear structure and metadata

        PATTERN: Markdown-like formatting
        WHY: Easy for Claude to parse and reference
        """
        lines = [
            "# Relevant Conversation History",
            "",
            f"Found {len(documents)} relevant past conversations related to: \"{query}\"",
            "",
            "---",
            ""
        ]

        for i, doc in enumerate(documents, start=1):
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(doc.timestamp)
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = doc.timestamp

            # Age display
            if doc.age_days < 1:
                age_str = "today"
            elif doc.age_days < 2:
                age_str = "yesterday"
            else:
                age_str = f"{int(doc.age_days)} days ago"

            lines.extend([
                f"## Conversation {i} (Relevance: {doc.score:.2f})",
                f"**Time:** {time_str} ({age_str})",
                f"**Session:** {doc.session_id}",
                "",
                f"**User:** {doc.user_text}",
                "",
                f"**Assistant:** {doc.agent_text}",
                "",
                "---",
                ""
            ])

        return "\n".join(lines)

    def _format_compact(self, documents: List[ContextDocument]) -> str:
        """
        Format context compactly to save tokens

        PATTERN: Minimal formatting
        WHY: Maximize content within token budget
        """
        lines = []

        for i, doc in enumerate(documents, start=1):
            lines.append(f"[{i}] User: {doc.user_text}")
            lines.append(f"    Assistant: {doc.agent_text}")

        return "\n".join(lines)

    def _format_conversation(self, documents: List[ContextDocument]) -> str:
        """
        Format as natural conversation flow

        PATTERN: Dialog format
        WHY: Natural reading for Claude
        """
        lines = ["Previous conversations:\n"]

        for doc in documents:
            lines.append(f"User: {doc.user_text}")
            lines.append(f"Assistant: {doc.agent_text}")
            lines.append("")

        return "\n".join(lines)

    async def _summarize_context(
        self,
        context: str,
        max_tokens: int
    ) -> str:
        """
        Summarize context if too large

        PATTERN: Truncation with preservation of structure
        WHY: Simple fallback without LLM call
        RESILIENCE: Always returns valid context

        NOTE: For production, consider using Claude to summarize
        """
        # Simple truncation for now
        # In production, could use Claude API to intelligently summarize

        db_logger.warning(
            "context_summarization_triggered",
            original_tokens=self._count_tokens(context),
            max_tokens=max_tokens,
            method="truncation"
        )

        # Estimate characters to keep (rough approximation: 1 token ≈ 4 chars)
        max_chars = max_tokens * 4

        if len(context) <= max_chars:
            return context

        # Truncate with ellipsis
        truncated = context[:max_chars - 50]
        truncated += "\n\n[Context truncated due to length...]"

        return truncated

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        PATTERN: Tokenizer with character-based fallback
        WHY: Accurate token counting for budget management
        RESILIENCE: Estimation if tokenizer unavailable
        """
        if not text:
            return 0

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                db_logger.debug(
                    "token_counting_failed",
                    error=str(e),
                    fallback="character_estimation"
                )

        # Fallback: estimate tokens (rough approximation)
        # Average: 1 token ≈ 4 characters for English text
        return len(text) // 4
