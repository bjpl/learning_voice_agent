"""
Hybrid Search Engine - Vector + FTS5
PATTERN: Reciprocal Rank Fusion for result combination
WHY: Leverage both semantic understanding and exact keyword matching
RESILIENCE: Graceful degradation if one search type fails
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

from app.logger import db_logger
from app.database import Database
from app.search.vector_store import VectorStore, vector_store
from app.search.query_analyzer import QueryAnalyzer, query_analyzer, QueryAnalysis
from app.search.config import (
    HybridSearchConfig,
    DEFAULT_SEARCH_CONFIG,
    SearchStrategy
)


@dataclass
class SearchResult:
    """Individual search result with combined score"""
    id: int
    session_id: str
    timestamp: str
    user_text: str
    agent_text: str
    score: float
    rank: int
    source: str  # 'vector', 'keyword', or 'hybrid'
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    user_snippet: Optional[str] = None
    agent_snippet: Optional[str] = None


@dataclass
class HybridSearchResponse:
    """Complete search response with metadata"""
    query: str
    strategy: str
    results: List[Dict]
    total_count: int
    query_analysis: Dict
    execution_time_ms: float
    vector_results_count: int = 0
    keyword_results_count: int = 0


class HybridSearchEngine:
    """
    Hybrid search engine combining vector similarity and FTS5 keyword search

    ALGORITHM:
    1. Analyze query → detect intent, extract keywords
    2. Execute vector search (semantic similarity)
    3. Execute FTS5 search (keyword matching)
    4. Combine results using Reciprocal Rank Fusion
    5. Normalize scores and deduplicate
    6. Return top N results
    """

    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        query_analyzer: QueryAnalyzer,
        config: HybridSearchConfig = DEFAULT_SEARCH_CONFIG
    ):
        self.db = database
        self.vector_store = vector_store
        self.query_analyzer = query_analyzer
        self.config = config
        self._embedding_client = None  # Will be set by initialize()

    def set_embedding_client(self, client):
        """Set OpenAI client for embeddings"""
        self._embedding_client = client

    async def search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        limit: Optional[int] = None
    ) -> HybridSearchResponse:
        """
        Execute hybrid search

        Args:
            query: Search query string
            strategy: Search strategy (None = adaptive)
            limit: Maximum results to return

        Returns:
            HybridSearchResponse with combined results
        """
        start_time = datetime.now()

        try:
            # Step 1: Analyze query
            analysis = await self.query_analyzer.analyze(query)

            # Determine strategy
            if strategy is None or strategy == SearchStrategy.ADAPTIVE:
                strategy = analysis.suggested_strategy

            # Set result limit
            if limit is None:
                limit = self.config.final_result_limit

            db_logger.info(
                "hybrid_search_started",
                query=query,
                strategy=strategy.value,
                limit=limit
            )

            # Step 2: Execute searches based on strategy
            vector_results = []
            keyword_results = []

            if strategy in [SearchStrategy.SEMANTIC, SearchStrategy.HYBRID]:
                vector_results = await self._vector_search(
                    query,
                    analysis,
                    self.config.max_results_per_search
                )

            if strategy in [SearchStrategy.KEYWORD, SearchStrategy.HYBRID]:
                keyword_results = await self._keyword_search(
                    query,
                    analysis,
                    self.config.max_results_per_search
                )

            # Step 3: Combine results
            if strategy == SearchStrategy.HYBRID:
                combined_results = self._reciprocal_rank_fusion(
                    vector_results,
                    keyword_results
                )
            elif strategy == SearchStrategy.SEMANTIC:
                combined_results = vector_results
            else:  # KEYWORD
                combined_results = keyword_results

            # Step 4: Limit results
            final_results = combined_results[:limit]

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            db_logger.info(
                "hybrid_search_complete",
                query=query,
                strategy=strategy.value,
                results_count=len(final_results),
                vector_count=len(vector_results),
                keyword_count=len(keyword_results),
                execution_time_ms=execution_time
            )

            # Build response
            return HybridSearchResponse(
                query=query,
                strategy=strategy.value,
                results=[asdict(r) for r in final_results],
                total_count=len(final_results),
                query_analysis=asdict(analysis),
                execution_time_ms=round(execution_time, 2),
                vector_results_count=len(vector_results),
                keyword_results_count=len(keyword_results)
            )

        except Exception as e:
            db_logger.error(
                "hybrid_search_failed",
                query=query,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            # Return empty results on error
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return HybridSearchResponse(
                query=query,
                strategy="error",
                results=[],
                total_count=0,
                query_analysis={},
                execution_time_ms=round(execution_time, 2),
                vector_results_count=0,
                keyword_results_count=0
            )

    async def _vector_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int
    ) -> List[SearchResult]:
        """
        Execute vector similarity search

        Returns:
            List of SearchResult objects with vector scores
        """
        try:
            if not self._embedding_client:
                db_logger.warning("vector_search_skipped_no_client")
                return []

            # Check cache first
            cached_embedding = self.vector_store.get_cached_embedding(query)

            if cached_embedding is not None:
                query_embedding = cached_embedding
                db_logger.debug("vector_search_cache_hit", query=query)
            else:
                # Generate embedding for query
                response = await self._embedding_client.embeddings.create(
                    input=query,
                    model=self.config.embedding_model
                )
                query_embedding = np.array(response.data[0].embedding, dtype=np.float32)

                # Cache the embedding
                self.vector_store.cache_embedding(query, query_embedding)

            # Search vector store
            similar_captures = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit,
                threshold=self.config.vector_similarity_threshold
            )

            if not similar_captures:
                return []

            # Fetch full capture details from database
            capture_ids = [cap_id for cap_id, _ in similar_captures]
            score_map = {cap_id: score for cap_id, score in similar_captures}

            results = []
            async with self.db.get_connection() as db:
                placeholders = ','.join('?' * len(capture_ids))
                cursor = await db.execute(
                    f"""
                    SELECT id, session_id, timestamp, user_text, agent_text
                    FROM captures
                    WHERE id IN ({placeholders})
                    """,
                    capture_ids
                )
                rows = await cursor.fetchall()

                for rank, row in enumerate(rows, start=1):
                    capture_id = row['id']
                    results.append(SearchResult(
                        id=capture_id,
                        session_id=row['session_id'],
                        timestamp=row['timestamp'],
                        user_text=row['user_text'],
                        agent_text=row['agent_text'],
                        score=score_map[capture_id],
                        rank=rank,
                        source='vector',
                        vector_score=score_map[capture_id]
                    ))

            db_logger.debug(
                "vector_search_complete",
                results_count=len(results),
                query=query
            )

            return results

        except Exception as e:
            db_logger.error(
                "vector_search_failed",
                query=query,
                error=str(e),
                exc_info=True
            )
            return []

    async def _keyword_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int
    ) -> List[SearchResult]:
        """
        Execute FTS5 keyword search

        Returns:
            List of SearchResult objects with keyword scores
        """
        try:
            # Use database FTS5 search
            fts_results = await self.db.search_captures(query, limit)

            if not fts_results:
                return []

            # Convert to SearchResult objects
            results = []
            for rank, row in enumerate(fts_results, start=1):
                # FTS5 rank is negative (lower is better), normalize to 0-1
                # We'll use rank position as score for now
                score = 1.0 / (rank + 1)  # Simple rank-based scoring

                results.append(SearchResult(
                    id=row['id'],
                    session_id=row['session_id'],
                    timestamp=row['timestamp'],
                    user_text=row['user_text'],
                    agent_text=row['agent_text'],
                    score=score,
                    rank=rank,
                    source='keyword',
                    keyword_score=score,
                    user_snippet=row.get('user_snippet'),
                    agent_snippet=row.get('agent_snippet')
                ))

            db_logger.debug(
                "keyword_search_complete",
                results_count=len(results),
                query=query
            )

            return results

        except Exception as e:
            db_logger.error(
                "keyword_search_failed",
                query=query,
                error=str(e),
                exc_info=True
            )
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine search results using Reciprocal Rank Fusion (RRF)

        RRF Formula: score(d) = Σ 1 / (k + rank(d))
        where k is a constant (typically 60)

        Args:
            vector_results: Results from vector search
            keyword_results: Results from FTS5 search

        Returns:
            Combined and re-ranked results
        """
        try:
            k = self.config.rrf_k
            combined_scores: Dict[int, Dict] = {}

            # Process vector results
            for result in vector_results:
                if result.id not in combined_scores:
                    combined_scores[result.id] = {
                        'result': result,
                        'rrf_score': 0.0,
                        'vector_contribution': 0.0,
                        'keyword_contribution': 0.0
                    }

                rrf_contribution = self.config.vector_weight / (k + result.rank)
                combined_scores[result.id]['rrf_score'] += rrf_contribution
                combined_scores[result.id]['vector_contribution'] = rrf_contribution

            # Process keyword results
            for result in keyword_results:
                if result.id not in combined_scores:
                    combined_scores[result.id] = {
                        'result': result,
                        'rrf_score': 0.0,
                        'vector_contribution': 0.0,
                        'keyword_contribution': 0.0
                    }

                rrf_contribution = self.config.keyword_weight / (k + result.rank)
                combined_scores[result.id]['rrf_score'] += rrf_contribution
                combined_scores[result.id]['keyword_contribution'] = rrf_contribution

                # Preserve snippets from keyword search
                if result.user_snippet:
                    combined_scores[result.id]['result'].user_snippet = result.user_snippet
                if result.agent_snippet:
                    combined_scores[result.id]['result'].agent_snippet = result.agent_snippet

            # Sort by combined RRF score
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['rrf_score'],
                reverse=True
            )

            # Build final result list with normalized scores
            if sorted_results:
                max_score = sorted_results[0]['rrf_score']
                min_score = sorted_results[-1]['rrf_score'] if len(sorted_results) > 1 else 0

                final_results = []
                for rank, item in enumerate(sorted_results, start=1):
                    result = item['result']

                    # Normalize score to 0-1 range
                    if max_score > min_score:
                        normalized_score = (item['rrf_score'] - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 1.0

                    # Update result metadata
                    result.score = normalized_score
                    result.rank = rank
                    result.source = 'hybrid'

                    final_results.append(result)

                db_logger.debug(
                    "rrf_fusion_complete",
                    total_results=len(final_results),
                    vector_only=sum(1 for r in final_results if r.vector_score and not r.keyword_score),
                    keyword_only=sum(1 for r in final_results if r.keyword_score and not r.vector_score),
                    both=sum(1 for r in final_results if r.vector_score and r.keyword_score)
                )

                return final_results

            return []

        except Exception as e:
            db_logger.error(
                "rrf_fusion_failed",
                error=str(e),
                exc_info=True
            )
            # Fallback: return vector results if available, else keyword
            return vector_results if vector_results else keyword_results


# Factory function to create configured search engine
def create_hybrid_search_engine(
    database: Database,
    config: Optional[HybridSearchConfig] = None
) -> HybridSearchEngine:
    """
    Create and configure a hybrid search engine

    Args:
        database: Database instance
        config: Optional custom configuration

    Returns:
        Configured HybridSearchEngine
    """
    if config is None:
        config = DEFAULT_SEARCH_CONFIG

    engine = HybridSearchEngine(
        database=database,
        vector_store=vector_store,
        query_analyzer=query_analyzer,
        config=config
    )

    return engine
