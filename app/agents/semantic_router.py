"""
Semantic Agent Router - Intelligence-Based Agent Selection

SPECIFICATION:
- Route queries to optimal agent using semantic similarity
- Pre-compute agent capability embeddings for fast matching
- Support confidence scoring and multi-agent suggestions
- Integrate with existing vector store infrastructure

ARCHITECTURE:
- Lazy initialization of embedding model
- Cached agent embeddings for performance
- Cosine similarity for agent matching
- Fallback to keyword matching for low confidence

PATTERN: Strategy pattern with semantic similarity
WHY: More intelligent routing than keyword matching
REFINEMENT: Cache embeddings, async operations, proper error handling
"""
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, fall back gracefully
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class AgentCapability:
    """
    Agent capability definition with semantic embeddings.

    Attributes:
        name: Agent identifier (e.g., 'conversation', 'research')
        capabilities: List of capability keywords
        description: Detailed description of agent purpose
        embedding: Pre-computed embedding vector for semantic matching
        priority: Priority weight for tie-breaking (higher = preferred)
    """
    name: str
    capabilities: List[str]
    description: str
    embedding: Optional[np.ndarray] = None
    priority: float = 1.0

    def __post_init__(self):
        """Validate capability definition."""
        if not self.capabilities:
            raise ValueError(f"Agent {self.name} must have at least one capability")
        if not self.description:
            raise ValueError(f"Agent {self.name} must have a description")


class SemanticAgentRouter:
    """
    Routes queries to optimal agent using semantic similarity.

    PATTERN: Semantic matching with fallback strategies
    WHY: More intelligent than keyword matching, handles synonyms and intent

    Features:
    - Pre-computed agent embeddings for fast routing
    - Confidence scoring for routing decisions
    - Support for multi-agent recommendations
    - Graceful fallback when embeddings unavailable
    """

    def __init__(self, embedding_model_name: Optional[str] = None):
        """
        Initialize semantic router.

        Args:
            embedding_model_name: Name of sentence-transformers model to use
                                 (defaults to settings.embedding_model)
        """
        self._embedding_model_name = embedding_model_name or getattr(
            settings, 'embedding_model', 'all-MiniLM-L6-v2'
        )
        self._embedding_model: Optional[Any] = None
        self._agent_capabilities: Dict[str, AgentCapability] = {}
        self._initialized = False

        logger.info(
            "Semantic router created",
            embedding_model=self._embedding_model_name,
            embeddings_available=EMBEDDINGS_AVAILABLE
        )

    async def initialize(self) -> bool:
        """
        Initialize the router by loading embedding model and computing agent embeddings.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Load embedding model in thread pool (synchronous operation)
            if EMBEDDINGS_AVAILABLE:
                loop = asyncio.get_event_loop()
                self._embedding_model = await loop.run_in_executor(
                    None,
                    self._load_embedding_model
                )
            else:
                logger.warning(
                    "sentence-transformers not available. "
                    "Semantic routing will use fallback keyword matching."
                )
                self._embedding_model = None

            # Register default agents
            await self._register_default_agents()

            self._initialized = True
            logger.info("Semantic router initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize semantic router: {e}", exc_info=True)
            return False

    def _load_embedding_model(self) -> Optional[Any]:
        """
        Load sentence-transformers model (synchronous).

        Returns:
            Loaded model or None if unavailable
        """
        if not EMBEDDINGS_AVAILABLE:
            return None

        try:
            model = SentenceTransformer(self._embedding_model_name)
            logger.info(f"Loaded embedding model: {self._embedding_model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            return None

    async def _register_default_agents(self) -> None:
        """
        Register default agent capabilities with pre-computed embeddings.

        SPECIFICATION: Define capabilities for each agent type
        WHY: These definitions drive the routing logic
        """
        # ConversationAgent - General conversation and questions
        await self._register_agent(
            AgentCapability(
                name="conversation",
                capabilities=[
                    "chat", "talk", "discuss", "conversation", "dialogue",
                    "question", "ask", "tell", "explain", "help",
                    "greeting", "hello", "hi", "hey",
                    "general", "casual", "basic"
                ],
                description=(
                    "Handle casual conversation, general questions, greetings, "
                    "and basic information requests. Best for direct questions "
                    "and ongoing dialogue."
                ),
                priority=1.0  # Default priority
            )
        )

        # ResearchAgent - External information gathering
        await self._register_agent(
            AgentCapability(
                name="research",
                capabilities=[
                    "search", "find", "lookup", "research", "investigate",
                    "information", "data", "facts", "learn", "discover",
                    "external", "web", "online", "internet",
                    "current", "latest", "recent", "news"
                ],
                description=(
                    "Research and find external information from the web, "
                    "current events, latest news, and factual data. "
                    "Use when query requires up-to-date information or "
                    "external knowledge lookup."
                ),
                priority=1.2  # Higher priority for clear research intent
            )
        )

        # AnalysisAgent - Deep analysis and explanations
        await self._register_agent(
            AgentCapability(
                name="analysis",
                capabilities=[
                    "analyze", "analysis", "examine", "investigate", "study",
                    "explain", "detail", "deep", "comprehensive", "thorough",
                    "breakdown", "understand", "interpret", "evaluate",
                    "compare", "contrast", "why", "how", "reasoning"
                ],
                description=(
                    "Provide deep analysis, detailed explanations, and "
                    "comprehensive breakdowns of complex topics. "
                    "Best for analytical questions requiring reasoning "
                    "and in-depth understanding."
                ),
                priority=1.1  # Slightly higher for analysis requests
            )
        )

        # SynthesisAgent - Information synthesis and summarization
        await self._register_agent(
            AgentCapability(
                name="synthesis",
                capabilities=[
                    "summarize", "summary", "combine", "integrate", "merge",
                    "synthesize", "consolidate", "overview", "brief",
                    "main points", "key takeaways", "conclusion",
                    "wrap up", "recap", "distill"
                ],
                description=(
                    "Synthesize and summarize information from multiple sources, "
                    "combine ideas, and provide overviews. "
                    "Use when user wants consolidated information or summaries."
                ),
                priority=1.0
            )
        )

        logger.info(
            f"Registered {len(self._agent_capabilities)} agent capabilities"
        )

    async def _register_agent(self, capability: AgentCapability) -> None:
        """
        Register an agent capability with pre-computed embedding.

        Args:
            capability: Agent capability definition
        """
        # Create combined text for embedding
        capability_text = (
            f"{capability.description} "
            f"Capabilities: {', '.join(capability.capabilities)}"
        )

        # Generate embedding
        if self._embedding_model:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._generate_embedding,
                capability_text
            )
            capability.embedding = embedding
        else:
            # No embedding available, will use keyword fallback
            capability.embedding = None

        self._agent_capabilities[capability.name] = capability
        logger.debug(f"Registered agent: {capability.name}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text (synchronous).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self._embedding_model is None:
            return np.array([])

        embedding = self._embedding_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        if len(a) == 0 or len(b) == 0:
            return 0.0

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def route(self, query: str) -> Tuple[str, float]:
        """
        Route query to best agent based on semantic similarity.

        Args:
            query: User query text

        Returns:
            Tuple of (agent_name, confidence_score)

        Example:
            >>> router = SemanticAgentRouter()
            >>> await router.initialize()
            >>> agent, confidence = await router.route("What's the latest news?")
            >>> print(f"Route to {agent} with confidence {confidence}")
            Route to research with confidence 0.85
        """
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        if self._embedding_model:
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                self._generate_embedding,
                query
            )

            # Calculate similarities with all agents
            similarities = []
            for name, capability in self._agent_capabilities.items():
                if capability.embedding is not None:
                    similarity = self._cosine_similarity(
                        query_embedding,
                        capability.embedding
                    )
                    # Apply priority weighting
                    weighted_score = similarity * capability.priority
                    similarities.append((name, weighted_score, similarity))

            if similarities:
                # Sort by weighted score
                similarities.sort(key=lambda x: x[1], reverse=True)
                best_agent, weighted_score, raw_similarity = similarities[0]

                logger.debug(
                    f"Semantic routing: {best_agent} "
                    f"(similarity: {raw_similarity:.3f}, "
                    f"weighted: {weighted_score:.3f})"
                )

                return best_agent, raw_similarity

        # Fallback to keyword matching if embeddings unavailable or failed
        return self._keyword_fallback_route(query)

    def _keyword_fallback_route(self, query: str) -> Tuple[str, float]:
        """
        Fallback routing using simple keyword matching.

        Args:
            query: User query text

        Returns:
            Tuple of (agent_name, confidence_score)
        """
        query_lower = query.lower()
        best_match = "conversation"
        best_score = 0.3  # Default low confidence

        for name, capability in self._agent_capabilities.items():
            # Count matching keywords
            matches = sum(
                1 for keyword in capability.capabilities
                if keyword in query_lower
            )

            if matches > 0:
                # Simple scoring: matches / total_keywords
                score = min(matches / len(capability.capabilities), 1.0)
                # Apply priority
                score *= capability.priority

                if score > best_score:
                    best_match = name
                    best_score = score

        logger.debug(
            f"Keyword fallback routing: {best_match} (score: {best_score:.3f})"
        )

        return best_match, best_score / 2.0  # Reduce confidence for fallback

    async def get_top_agents(
        self,
        query: str,
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top N candidate agents with similarity scores.

        Args:
            query: User query text
            n: Number of top agents to return (default: 3)

        Returns:
            List of (agent_name, confidence_score) tuples, sorted by score

        Example:
            >>> router = SemanticAgentRouter()
            >>> await router.initialize()
            >>> candidates = await router.get_top_agents(
            ...     "Analyze the latest research on AI", n=2
            ... )
            >>> for agent, score in candidates:
            ...     print(f"{agent}: {score:.3f}")
            research: 0.850
            analysis: 0.720
        """
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        if self._embedding_model:
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                self._generate_embedding,
                query
            )

            # Calculate similarities
            similarities = []
            for name, capability in self._agent_capabilities.items():
                if capability.embedding is not None:
                    similarity = self._cosine_similarity(
                        query_embedding,
                        capability.embedding
                    )
                    similarities.append((name, similarity))

            # Sort and return top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n]

        # Fallback: return single best match
        best_agent, score = self._keyword_fallback_route(query)
        return [(best_agent, score)]

    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with agent information or None if not found
        """
        capability = self._agent_capabilities.get(agent_name)
        if not capability:
            return None

        return {
            "name": capability.name,
            "capabilities": capability.capabilities,
            "description": capability.description,
            "priority": capability.priority,
            "has_embedding": capability.embedding is not None
        }

    def list_agents(self) -> List[str]:
        """
        List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self._agent_capabilities.keys())


# Global singleton instance
_semantic_router: Optional[SemanticAgentRouter] = None


async def get_semantic_router() -> SemanticAgentRouter:
    """
    Get or create global semantic router instance.

    Returns:
        Initialized SemanticAgentRouter instance
    """
    global _semantic_router

    if _semantic_router is None:
        _semantic_router = SemanticAgentRouter()
        await _semantic_router.initialize()

    return _semantic_router
