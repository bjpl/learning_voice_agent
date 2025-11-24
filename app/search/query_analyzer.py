"""
Query Analyzer for Hybrid Search
PATTERN: Query preprocessing and intent detection
WHY: Optimize search strategy based on query characteristics
"""
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import asyncio

from app.logger import db_logger
from app.search.config import (
    HybridSearchConfig,
    DEFAULT_SEARCH_CONFIG,
    SearchStrategy,
    INTENT_PATTERNS,
    STOP_WORDS
)


@dataclass
class QueryAnalysis:
    """Results of query analysis"""
    original_query: str
    cleaned_query: str
    keywords: List[str]
    intent: str
    suggested_strategy: SearchStrategy
    is_short: bool
    is_exact_phrase: bool
    word_count: int


class QueryAnalyzer:
    """
    Analyzes queries to optimize search strategy

    CONCEPT: Detects query intent and characteristics to determine
    the best search approach (semantic, keyword, or hybrid)
    """

    def __init__(self, config: HybridSearchConfig = DEFAULT_SEARCH_CONFIG):
        self.config = config

    async def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query and suggest search strategy

        Args:
            query: User's search query

        Returns:
            QueryAnalysis with extracted features and suggested strategy
        """
        try:
            # Clean and normalize query
            cleaned = self._clean_query(query)

            # Extract keywords
            keywords = self._extract_keywords(cleaned)

            # Detect intent
            intent = self._detect_intent(cleaned)

            # Check query characteristics
            is_short = len(keywords) <= 3
            is_exact_phrase = self._is_exact_phrase(query)
            word_count = len(cleaned.split())

            # Suggest strategy
            strategy = self._suggest_strategy(
                intent=intent,
                is_short=is_short,
                is_exact_phrase=is_exact_phrase,
                word_count=word_count
            )

            analysis = QueryAnalysis(
                original_query=query,
                cleaned_query=cleaned,
                keywords=keywords,
                intent=intent,
                suggested_strategy=strategy,
                is_short=is_short,
                is_exact_phrase=is_exact_phrase,
                word_count=word_count
            )

            db_logger.debug(
                "query_analyzed",
                query=query,
                intent=intent,
                strategy=strategy.value,
                keywords_count=len(keywords)
            )

            return analysis

        except Exception as e:
            db_logger.error(
                "query_analysis_failed",
                query=query,
                error=str(e),
                exc_info=True
            )
            # Return basic analysis
            return QueryAnalysis(
                original_query=query,
                cleaned_query=query,
                keywords=query.split(),
                intent="unknown",
                suggested_strategy=SearchStrategy.HYBRID,
                is_short=False,
                is_exact_phrase=False,
                word_count=len(query.split())
            )

    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize query text

        Steps:
        1. Trim whitespace
        2. Convert to lowercase
        3. Remove special characters (except quotes)
        4. Normalize spaces
        """
        # Trim and lowercase
        cleaned = query.strip().lower()

        # Validate length
        if len(cleaned) > self.config.max_query_length:
            cleaned = cleaned[:self.config.max_query_length]

        # Remove special characters but preserve quotes for exact phrases
        cleaned = re.sub(r'[^\w\s"\'-]', ' ', cleaned)

        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())

        return cleaned

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query

        Removes:
        - Stop words
        - Very short words (< 2 chars)
        - Duplicate words
        """
        # Split into words
        words = query.lower().split()

        # Filter out stop words and short words
        keywords = [
            word for word in words
            if word not in STOP_WORDS and len(word) >= 2
        ]

        # Remove duplicates while preserving order
        seen: Set[str] = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def _detect_intent(self, query: str) -> str:
        """
        Detect query intent based on patterns

        Intent types:
        - conceptual: Understanding concepts, explanations
        - factual: Specific facts, dates, names
        - comparison: Comparing items
        - procedural: How-to, steps
        - unknown: Cannot determine
        """
        query_lower = query.lower()

        # Check each intent pattern
        intent_scores = {}
        for intent_type, patterns in INTENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent_type] = score

        # Return intent with highest score
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        return "unknown"

    def _is_exact_phrase(self, query: str) -> bool:
        """Check if query is an exact phrase search (in quotes)"""
        return query.strip().startswith('"') and query.strip().endswith('"')

    def _suggest_strategy(
        self,
        intent: str,
        is_short: bool,
        is_exact_phrase: bool,
        word_count: int
    ) -> SearchStrategy:
        """
        Suggest optimal search strategy based on query characteristics

        Rules:
        1. Exact phrase → keyword search
        2. Very short (1-2 words) → keyword search
        3. Factual intent → keyword search
        4. Conceptual intent → semantic search
        5. Long query (>5 words) → semantic search
        6. Default → hybrid search
        """
        # Exact phrase always uses keyword search
        if is_exact_phrase:
            return SearchStrategy.KEYWORD

        # Very short queries work better with keyword
        if word_count <= 2:
            return SearchStrategy.KEYWORD

        # Intent-based selection
        if intent == "factual":
            return SearchStrategy.KEYWORD
        elif intent == "conceptual":
            return SearchStrategy.SEMANTIC

        # Long queries benefit from semantic understanding
        if word_count > 5:
            return SearchStrategy.SEMANTIC

        # Default to hybrid for balanced results
        return SearchStrategy.HYBRID

    def expand_query(self, query: str, synonyms: Dict[str, List[str]]) -> str:
        """
        Expand query with synonyms (optional feature)

        Args:
            query: Original query
            synonyms: Dict mapping words to their synonyms

        Returns:
            Expanded query with synonyms
        """
        if not self.config.enable_query_expansion:
            return query

        words = query.split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)
            if word in synonyms:
                expanded_words.extend(synonyms[word])

        return ' '.join(expanded_words)

    def suggest_corrections(self, query: str) -> List[str]:
        """
        Suggest spelling corrections (placeholder for future implementation)

        Could integrate with:
        - Levenshtein distance
        - Corpus-based suggestions
        - ML-based correction
        """
        if not self.config.enable_spell_correction:
            return []

        # Placeholder - could implement fuzzy matching here
        return []


# Global query analyzer instance
query_analyzer = QueryAnalyzer()
