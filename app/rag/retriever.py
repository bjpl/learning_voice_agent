"""
RAG Retriever - Context Retrieval with Hybrid Search

SPECIFICATION:
- Retrieve top-k most relevant conversation history
- Use hybrid search (vector + keyword)
- Apply relevance filtering (minimum score threshold)
- Deduplicate results
- Session-scoped or global search

ARCHITECTURE:
- Wraps existing HybridSearchEngine
- Adds RAG-specific filtering and ranking
- Time-aware relevance scoring
- Graceful fallback behavior

PATTERN: Facade over hybrid search with RAG enhancements
WHY: Reuse existing search infrastructure, add RAG-specific logic
RESILIENCE: Fallback to keyword-only search if vector search fails
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum

from app.logger import db_logger
from app.database import Database
from app.search.hybrid_search import HybridSearchEngine, SearchResult
from app.search.config import SearchStrategy
from app.rag.config import RAGConfig, rag_config


class RetrievalStrategy(str, Enum):
    """Retrieval strategy options"""
    SEMANTIC = "semantic"  # Vector similarity only
    KEYWORD = "keyword"   # Keyword matching only
    HYBRID = "hybrid"     # Combined vector + keyword
    ADAPTIVE = "adaptive" # Auto-select based on query


@dataclass
class RetrievalResult:
    """
    Individual retrieved document

    CONCEPT: Enriched search result with RAG metadata
    WHY: Additional context for ranking and citation generation
    """
    id: int
    session_id: str
    timestamp: str
    user_text: str
    agent_text: str
    score: float
    rank: int
    source: str  # 'vector', 'keyword', or 'hybrid'

    # Optional metadata
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    snippet_user: Optional[str] = None
    snippet_agent: Optional[str] = None

    # RAG-specific enrichments
    recency_weight: float = 1.0
    final_score: float = 0.0  # Combined score with recency
    age_days: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RetrievalResponse:
    """
    Complete retrieval response with metadata

    CONCEPT: Structured retrieval results with debugging info
    WHY: Transparency for debugging and optimization
    """
    query: str
    results: List[RetrievalResult]
    total_retrieved: int
    strategy_used: str
    execution_time_ms: float

    # Filter statistics
    filtered_count: int = 0
    deduplication_count: int = 0

    # Session context
    session_id: Optional[str] = None
    session_scoped: bool = False

    # Performance metadata
    retrieval_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_retrieved": self.total_retrieved,
            "strategy_used": self.strategy_used,
            "execution_time_ms": self.execution_time_ms,
            "filtered_count": self.filtered_count,
            "deduplication_count": self.deduplication_count,
            "session_id": self.session_id,
            "session_scoped": self.session_scoped,
            "retrieval_metadata": self.retrieval_metadata,
        }


class RAGRetriever:
    """
    RAG-optimized retrieval using hybrid search

    ALGORITHM:
    1. Execute hybrid search (vector + keyword)
    2. Filter by relevance threshold
    3. Apply recency weighting
    4. Deduplicate similar results
    5. Rank by combined score
    6. Return top-k results

    PATTERN: Pipeline with configurable stages
    WHY: Flexibility to tune retrieval quality
    RESILIENCE: Graceful degradation if stages fail
    """

    def __init__(
        self,
        database: Database,
        hybrid_search_engine: HybridSearchEngine,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize RAG retriever

        Args:
            database: Database instance for session filtering
            hybrid_search_engine: Configured hybrid search engine
            config: RAG configuration (uses default if None)
        """
        self.db = database
        self.search_engine = hybrid_search_engine
        self.config = config or rag_config

        db_logger.info(
            "rag_retriever_initialized",
            top_k=self.config.retrieval_top_k,
            threshold=self.config.relevance_threshold,
            hybrid_alpha=self.config.hybrid_alpha
        )

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents for RAG

        ALGORITHM:
        1. Execute hybrid search
        2. Filter by relevance threshold
        3. Apply session filtering if requested
        4. Add recency weighting
        5. Deduplicate results
        6. Limit to top-k

        Args:
            query: User query to find relevant context for
            session_id: Optional session ID for scoped search
            top_k: Maximum results (overrides config)
            strategy: Search strategy (overrides config)

        Returns:
            RetrievalResponse with filtered and ranked results
        """
        start_time = datetime.now()

        try:
            # Determine search parameters
            top_k = top_k or self.config.retrieval_top_k

            # Map RAG strategy to search strategy
            search_strategy = self._map_strategy(strategy)

            # Determine if session-scoped
            session_scoped = self.config.session_scoped_search and session_id is not None

            db_logger.info(
                "rag_retrieval_started",
                query=query[:100],
                top_k=top_k,
                strategy=search_strategy.value if search_strategy else "adaptive",
                session_scoped=session_scoped
            )

            # Execute hybrid search with timeout
            try:
                search_response = await asyncio.wait_for(
                    self.search_engine.search(
                        query=query,
                        strategy=search_strategy,
                        limit=top_k * 3  # Retrieve more for filtering
                    ),
                    timeout=self.config.retrieval_timeout
                )
            except asyncio.TimeoutError:
                db_logger.warning(
                    "rag_retrieval_timeout",
                    query=query[:100],
                    timeout=self.config.retrieval_timeout
                )
                # Return empty results on timeout
                return self._empty_response(query, session_id, session_scoped)

            # Convert search results to retrieval results
            results = self._convert_search_results(search_response.results)

            # Apply RAG-specific filtering and ranking
            filtered_results = await self._filter_and_rank(
                results=results,
                query=query,
                session_id=session_id if session_scoped else None,
                top_k=top_k
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            db_logger.info(
                "rag_retrieval_complete",
                query=query[:100],
                retrieved_count=len(filtered_results),
                execution_time_ms=round(execution_time, 2)
            )

            return RetrievalResponse(
                query=query,
                results=filtered_results,
                total_retrieved=len(filtered_results),
                strategy_used=search_response.strategy,
                execution_time_ms=round(execution_time, 2),
                filtered_count=len(results) - len(filtered_results),
                session_id=session_id,
                session_scoped=session_scoped,
                retrieval_metadata={
                    "vector_results": search_response.vector_results_count,
                    "keyword_results": search_response.keyword_results_count,
                    "query_analysis": search_response.query_analysis
                }
            )

        except Exception as e:
            db_logger.error(
                "rag_retrieval_failed",
                query=query[:100],
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )

            # Fallback: return empty results
            if self.config.enable_fallback:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                return self._empty_response(
                    query=query,
                    session_id=session_id,
                    session_scoped=session_scoped,
                    execution_time_ms=execution_time
                )
            else:
                raise

    async def retrieve_similar_conversations(
        self,
        conversation_id: int,
        top_k: int = 5
    ) -> RetrievalResponse:
        """
        Find conversations similar to a given conversation

        PATTERN: Conversation-to-conversation similarity
        WHY: Useful for finding related discussions

        Args:
            conversation_id: ID of the conversation to find similar ones to
            top_k: Maximum similar conversations to return

        Returns:
            RetrievalResponse with similar conversations
        """
        try:
            # Fetch the conversation text
            async with self.db.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT user_text, agent_text
                    FROM captures
                    WHERE id = ?
                    """,
                    (conversation_id,)
                )
                row = await cursor.fetchone()

                if not row:
                    db_logger.warning(
                        "conversation_not_found",
                        conversation_id=conversation_id
                    )
                    return self._empty_response(
                        query=f"Conversation {conversation_id}",
                        session_id=None,
                        session_scoped=False
                    )

                # Use combined text as query
                query = f"{row['user_text']} {row['agent_text']}"

            # Retrieve similar conversations
            response = await self.retrieve(
                query=query,
                top_k=top_k + 1,  # +1 to account for self-match
                strategy=RetrievalStrategy.SEMANTIC
            )

            # Filter out the original conversation
            response.results = [
                r for r in response.results
                if r.id != conversation_id
            ][:top_k]

            response.total_retrieved = len(response.results)

            return response

        except Exception as e:
            db_logger.error(
                "similar_conversation_retrieval_failed",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True
            )
            raise

    def _map_strategy(
        self,
        rag_strategy: Optional[RetrievalStrategy]
    ) -> Optional[SearchStrategy]:
        """Map RAG strategy to search strategy"""
        if rag_strategy is None:
            return None

        mapping = {
            RetrievalStrategy.SEMANTIC: SearchStrategy.SEMANTIC,
            RetrievalStrategy.KEYWORD: SearchStrategy.KEYWORD,
            RetrievalStrategy.HYBRID: SearchStrategy.HYBRID,
            RetrievalStrategy.ADAPTIVE: SearchStrategy.ADAPTIVE,
        }

        return mapping.get(rag_strategy, SearchStrategy.ADAPTIVE)

    def _convert_search_results(
        self,
        search_results: List[Dict]
    ) -> List[RetrievalResult]:
        """Convert search results to retrieval results"""
        results = []

        for result in search_results:
            results.append(RetrievalResult(
                id=result['id'],
                session_id=result['session_id'],
                timestamp=result['timestamp'],
                user_text=result['user_text'],
                agent_text=result['agent_text'],
                score=result['score'],
                rank=result['rank'],
                source=result['source'],
                vector_score=result.get('vector_score'),
                keyword_score=result.get('keyword_score'),
                snippet_user=result.get('user_snippet'),
                snippet_agent=result.get('agent_snippet')
            ))

        return results

    async def _filter_and_rank(
        self,
        results: List[RetrievalResult],
        query: str,
        session_id: Optional[str],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Filter and rank results with RAG-specific logic

        ALGORITHM:
        1. Filter by relevance threshold
        2. Filter by session if scoped
        3. Filter by age if configured
        4. Add recency weighting
        5. Deduplicate similar results
        6. Rank by final score
        7. Limit to top-k
        """
        # Step 1: Filter by relevance threshold
        filtered = [
            r for r in results
            if r.score >= self.config.relevance_threshold
        ]

        db_logger.debug(
            "rag_relevance_filter",
            before=len(results),
            after=len(filtered),
            threshold=self.config.relevance_threshold
        )

        # Step 2: Session filtering
        if session_id:
            filtered = [r for r in filtered if r.session_id == session_id]
            db_logger.debug(
                "rag_session_filter",
                session_id=session_id,
                count=len(filtered)
            )

        # Step 3: Age filtering
        if self.config.max_document_age_days:
            filtered = self._filter_by_age(
                filtered,
                self.config.max_document_age_days
            )

        # Step 4: Add recency weighting
        if self.config.prioritize_recent:
            filtered = self._add_recency_weighting(filtered)
        else:
            # No recency weighting, final score = base score
            for result in filtered:
                result.final_score = result.score

        # Step 5: Deduplicate
        if self.config.deduplicate_context:
            dedup_before = len(filtered)
            filtered = self._deduplicate_results(filtered)
            dedup_count = dedup_before - len(filtered)

            db_logger.debug(
                "rag_deduplication",
                removed=dedup_count,
                remaining=len(filtered)
            )

        # Step 6: Sort by final score
        filtered.sort(key=lambda r: r.final_score, reverse=True)

        # Re-rank
        for rank, result in enumerate(filtered, start=1):
            result.rank = rank

        # Step 7: Limit to top-k
        return filtered[:top_k]

    def _filter_by_age(
        self,
        results: List[RetrievalResult],
        max_age_days: int
    ) -> List[RetrievalResult]:
        """Filter results by maximum age"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        filtered = []
        for result in results:
            try:
                timestamp = datetime.fromisoformat(result.timestamp)
                if timestamp >= cutoff_date:
                    filtered.append(result)
            except (ValueError, TypeError):
                # Keep result if timestamp parsing fails
                filtered.append(result)

        db_logger.debug(
            "rag_age_filter",
            max_age_days=max_age_days,
            before=len(results),
            after=len(filtered)
        )

        return filtered

    def _add_recency_weighting(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Add recency weighting to scores

        ALGORITHM: Exponential decay
        recency_weight = exp(-age_days / decay_days)
        final_score = base_score * (0.7 + 0.3 * recency_weight)

        WHY: Recent conversations are usually more relevant
        PATTERN: Exponential decay with configurable half-life
        """
        now = datetime.now()
        decay_days = self.config.recency_decay_days

        for result in results:
            try:
                timestamp = datetime.fromisoformat(result.timestamp)
                age_days = (now - timestamp).total_seconds() / 86400

                # Exponential decay
                import math
                recency_weight = math.exp(-age_days / decay_days)

                # Combine base score with recency (70% base, 30% recency)
                result.recency_weight = recency_weight
                result.age_days = age_days
                result.final_score = result.score * (0.7 + 0.3 * recency_weight)

            except (ValueError, TypeError) as e:
                # Fallback: use base score
                result.recency_weight = 1.0
                result.age_days = 0.0
                result.final_score = result.score

                db_logger.debug(
                    "recency_weight_calculation_failed",
                    result_id=result.id,
                    error=str(e)
                )

        return results

    def _deduplicate_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Remove duplicate or highly similar results

        PATTERN: Greedy selection with similarity threshold
        WHY: Avoid redundant context
        ALGORITHM:
        1. Sort by score (highest first)
        2. For each result, check similarity to selected results
        3. Keep if dissimilar to all selected results
        """
        if not results:
            return []

        # Sort by final score (highest first)
        sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

        selected = [sorted_results[0]]  # Always keep top result

        for result in sorted_results[1:]:
            # Check similarity to all selected results
            is_duplicate = False

            for selected_result in selected:
                similarity = self._calculate_text_similarity(
                    result.user_text + " " + result.agent_text,
                    selected_result.user_text + " " + selected_result.agent_text
                )

                if similarity >= self.config.deduplication_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(result)

        return selected

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity

        PATTERN: Jaccard similarity on word sets
        WHY: Fast and effective for deduplication
        RESILIENCE: Simple fallback, no dependencies
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _empty_response(
        self,
        query: str,
        session_id: Optional[str],
        session_scoped: bool,
        execution_time_ms: float = 0.0
    ) -> RetrievalResponse:
        """Create empty retrieval response"""
        return RetrievalResponse(
            query=query,
            results=[],
            total_retrieved=0,
            strategy_used="none",
            execution_time_ms=execution_time_ms,
            session_id=session_id,
            session_scoped=session_scoped
        )
