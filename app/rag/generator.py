"""
RAG Generator - Claude-Powered Response Generation

SPECIFICATION:
- Generate responses using Claude 3.5 Sonnet
- Inject retrieved context into prompts
- Cite sources in responses
- Handle generation failures gracefully
- Support streaming responses (future)

ARCHITECTURE:
- Anthropic Claude integration
- RAG-optimized prompt engineering
- Citation extraction and linking
- Fallback to non-RAG generation

PATTERN: Template-based prompt construction with citations
WHY: Leverage Claude's strong instruction-following
RESILIENCE: Graceful degradation to basic generation
"""

import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

from anthropic import AsyncAnthropic
from anthropic.types import Message

from app.logger import db_logger
from app.rag.config import RAGConfig, rag_config
from app.rag.context_builder import BuiltContext


class GenerationMode(str, Enum):
    """Generation mode options"""
    RAG = "rag"               # Full RAG with context
    BASIC = "basic"           # No context injection
    FALLBACK = "fallback"     # Degraded mode


@dataclass
class Citation:
    """
    Source citation for generated content

    CONCEPT: Link response claims to source documents
    WHY: Transparency and verifiability
    """
    document_id: int
    session_id: str
    timestamp: str
    excerpt: str
    relevance_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class GenerationResponse:
    """
    Complete RAG generation response

    CONCEPT: Generated text with metadata and citations
    WHY: Structured output for debugging and evaluation
    """
    query: str
    response_text: str
    mode: str

    # Generation metadata
    model: str
    tokens_used: int
    generation_time_ms: float

    # RAG metadata
    citations: List[Citation] = field(default_factory=list)
    context_used: bool = False
    context_tokens: int = 0

    # Performance
    finish_reason: str = "stop"

    # Debugging
    generation_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "response_text": self.response_text,
            "mode": self.mode,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "generation_time_ms": self.generation_time_ms,
            "citations": [c.to_dict() for c in self.citations],
            "context_used": self.context_used,
            "context_tokens": self.context_tokens,
            "finish_reason": self.finish_reason,
            "generation_metadata": self.generation_metadata,
        }


class RAGGenerator:
    """
    Claude-powered RAG generation

    ALGORITHM:
    1. Build RAG prompt with context
    2. Call Claude API
    3. Extract citations from response
    4. Package response with metadata

    PATTERN: Template-based prompt construction
    WHY: Consistent, optimized prompts for Claude
    RESILIENCE: Fallback to basic generation if context unavailable
    """

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize RAG generator

        Args:
            anthropic_client: Async Anthropic client
            config: RAG configuration (uses default if None)
        """
        self.client = anthropic_client
        self.config = config or rag_config

        db_logger.info(
            "rag_generator_initialized",
            model=self.config.generation_model,
            max_tokens=self.config.generation_max_tokens,
            temperature=self.config.generation_temperature,
            citations_enabled=self.config.enable_citations
        )

    async def generate(
        self,
        query: str,
        context: Optional[BuiltContext] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GenerationResponse:
        """
        Generate response using RAG

        ALGORITHM:
        1. Determine generation mode
        2. Build prompt (with or without context)
        3. Call Claude API
        4. Extract citations if enabled
        5. Package response

        Args:
            query: User query
            context: Retrieved and formatted context (optional)
            system_prompt: Override system prompt (optional)
            temperature: Override temperature (optional)
            max_tokens: Override max tokens (optional)

        Returns:
            GenerationResponse with text and metadata
        """
        start_time = datetime.now()

        try:
            # Determine mode
            mode = GenerationMode.RAG if context and context.formatted_context else GenerationMode.BASIC

            db_logger.info(
                "rag_generation_started",
                query=query[:100],
                mode=mode.value,
                context_tokens=context.total_tokens if context else 0
            )

            # Build prompt
            messages, system = self._build_prompt(
                query=query,
                context=context,
                system_prompt=system_prompt,
                mode=mode
            )

            # Call Claude API
            response = await self._call_claude(
                messages=messages,
                system=system,
                temperature=temperature or self.config.generation_temperature,
                max_tokens=max_tokens or self.config.generation_max_tokens
            )

            # Extract response text
            response_text = response.content[0].text if response.content else ""

            # Extract citations if enabled
            citations = []
            if self.config.enable_citations and context and context.documents:
                citations = self._extract_citations(
                    response_text=response_text,
                    documents=context.documents
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            db_logger.info(
                "rag_generation_complete",
                query=query[:100],
                mode=mode.value,
                response_length=len(response_text),
                citations_count=len(citations),
                tokens_used=response.usage.output_tokens,
                execution_time_ms=round(execution_time, 2)
            )

            return GenerationResponse(
                query=query,
                response_text=response_text,
                mode=mode.value,
                model=response.model,
                tokens_used=response.usage.output_tokens,
                generation_time_ms=round(execution_time, 2),
                citations=citations,
                context_used=mode == GenerationMode.RAG,
                context_tokens=context.total_tokens if context else 0,
                finish_reason=response.stop_reason or "stop",
                generation_metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "context_document_count": len(context.documents) if context else 0,
                }
            )

        except Exception as e:
            db_logger.error(
                "rag_generation_failed",
                query=query[:100],
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )

            # Fallback: return error response
            if self.config.enable_fallback:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                return self._fallback_response(
                    query=query,
                    error=str(e),
                    execution_time_ms=execution_time
                )
            else:
                raise

    async def generate_batch(
        self,
        queries: List[str],
        contexts: Optional[List[BuiltContext]] = None
    ) -> List[GenerationResponse]:
        """
        Generate responses for multiple queries

        PATTERN: Concurrent API calls
        WHY: Parallelization reduces total latency

        Args:
            queries: List of user queries
            contexts: Optional list of contexts (one per query)

        Returns:
            List of GenerationResponse objects
        """
        if contexts is None:
            contexts = [None] * len(queries)

        if len(queries) != len(contexts):
            raise ValueError(
                f"Queries and contexts length mismatch: {len(queries)} vs {len(contexts)}"
            )

        db_logger.info(
            "batch_generation_started",
            batch_size=len(queries)
        )

        # Generate concurrently
        tasks = [
            self.generate(query=q, context=c)
            for q, c in zip(queries, contexts)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                db_logger.error(
                    "batch_generation_item_failed",
                    index=i,
                    query=queries[i][:100],
                    error=str(response)
                )
                # Add fallback response
                valid_responses.append(
                    self._fallback_response(
                        query=queries[i],
                        error=str(response),
                        execution_time_ms=0.0
                    )
                )
            else:
                valid_responses.append(response)

        db_logger.info(
            "batch_generation_complete",
            batch_size=len(queries),
            successful=sum(1 for r in valid_responses if r.mode != "fallback")
        )

        return valid_responses

    def _build_prompt(
        self,
        query: str,
        context: Optional[BuiltContext],
        system_prompt: Optional[str],
        mode: GenerationMode
    ) -> tuple[List[Dict[str, str]], str]:
        """
        Build RAG-optimized prompt

        PATTERN: Template-based construction with context injection
        WHY: Consistent, high-quality prompts
        """
        # Default system prompt
        if system_prompt is None:
            if mode == GenerationMode.RAG:
                system_prompt = self._rag_system_prompt()
            else:
                system_prompt = self._basic_system_prompt()

        # Build user message
        if mode == GenerationMode.RAG and context:
            user_message = self._rag_user_message(query, context)
        else:
            user_message = query

        messages = [
            {
                "role": "user",
                "content": user_message
            }
        ]

        return messages, system_prompt

    def _rag_system_prompt(self) -> str:
        """
        System prompt for RAG mode

        PATTERN: Instruction-following with citation guidance
        WHY: Direct Claude to use context and cite sources
        """
        citation_instructions = ""
        if self.config.enable_citations:
            citation_instructions = """
When referencing information from the conversation history, mention which conversation you're referring to (e.g., "As discussed in Conversation 1..." or "Based on our previous conversation about...").
"""

        return f"""You are a helpful AI assistant with access to conversation history.

Your task is to answer the user's question using the provided conversation history as context. Follow these guidelines:

1. Prioritize information from the conversation history when relevant
2. Be specific and reference past conversations when applicable
3. If the conversation history doesn't contain relevant information, use your general knowledge
4. Be concise and direct in your responses
5. Maintain consistency with past conversations
{citation_instructions}

Remember: The conversation history shows what has been discussed before. Use it to provide contextually aware and consistent responses."""

    def _basic_system_prompt(self) -> str:
        """System prompt for basic mode (no context)"""
        return """You are a helpful AI assistant. Answer the user's question clearly and concisely."""

    def _rag_user_message(self, query: str, context: BuiltContext) -> str:
        """
        Build user message with injected context

        PATTERN: Context-first, query-second
        WHY: Ensure Claude sees context before answering
        """
        return f"""{context.formatted_context}

---

Current Question: {query}

Please answer the question above, using the conversation history provided when relevant."""

    async def _call_claude(
        self,
        messages: List[Dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int
    ) -> Message:
        """
        Call Claude API

        PATTERN: Direct API call with error handling
        WHY: Centralized API interaction
        RESILIENCE: Timeout and retry handled by client
        """
        try:
            response = await self.client.messages.create(
                model=self.config.generation_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages
            )

            return response

        except Exception as e:
            db_logger.error(
                "claude_api_call_failed",
                model=self.config.generation_model,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    def _extract_citations(
        self,
        response_text: str,
        documents: List[Any]
    ) -> List[Citation]:
        """
        Extract citations from response

        PATTERN: Heuristic matching
        WHY: Simple, effective citation extraction
        RESILIENCE: Best-effort matching

        NOTE: In production, could use Claude to generate structured citations
        """
        citations = []

        # Simple heuristic: look for mentions of "Conversation N"
        import re
        pattern = r"[Cc]onversation (\d+)"
        matches = re.finditer(pattern, response_text)

        mentioned_indices = set()
        for match in matches:
            index = int(match.group(1)) - 1  # Convert to 0-indexed
            mentioned_indices.add(index)

        # Create citations for mentioned conversations
        for idx in mentioned_indices:
            if 0 <= idx < len(documents):
                doc = documents[idx]

                # Extract excerpt (first 200 chars)
                excerpt = doc.user_text[:200]
                if len(doc.user_text) > 200:
                    excerpt += "..."

                citations.append(Citation(
                    document_id=doc.id,
                    session_id=doc.session_id,
                    timestamp=doc.timestamp,
                    excerpt=excerpt,
                    relevance_score=doc.score
                ))

        # If no explicit mentions, cite all documents used
        if not citations and documents:
            for doc in documents[:3]:  # Cite top 3
                excerpt = doc.user_text[:200]
                if len(doc.user_text) > 200:
                    excerpt += "..."

                citations.append(Citation(
                    document_id=doc.id,
                    session_id=doc.session_id,
                    timestamp=doc.timestamp,
                    excerpt=excerpt,
                    relevance_score=doc.score
                ))

        db_logger.debug(
            "citations_extracted",
            citation_count=len(citations),
            document_count=len(documents)
        )

        return citations

    def _fallback_response(
        self,
        query: str,
        error: str,
        execution_time_ms: float
    ) -> GenerationResponse:
        """
        Create fallback response on error

        PATTERN: Graceful degradation
        WHY: Always return valid response
        """
        return GenerationResponse(
            query=query,
            response_text="I apologize, but I'm unable to generate a response at this time. Please try again.",
            mode=GenerationMode.FALLBACK.value,
            model=self.config.generation_model,
            tokens_used=0,
            generation_time_ms=execution_time_ms,
            citations=[],
            context_used=False,
            context_tokens=0,
            finish_reason="error",
            generation_metadata={
                "error": error,
                "fallback": True
            }
        )
