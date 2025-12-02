"""
Unit Tests for Semantic Agent Router

PATTERN: Comprehensive test coverage with mocking
WHY: Ensure semantic routing works correctly with and without embeddings
COVERAGE: Initialization, routing, fallback, multi-agent selection
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from app.agents.semantic_router import (
    SemanticAgentRouter,
    AgentCapability,
    get_semantic_router,
    EMBEDDINGS_AVAILABLE
)


@pytest.fixture
def router():
    """Create a semantic router instance for testing."""
    return SemanticAgentRouter()


@pytest.fixture
def router_with_mock_model(router):
    """Create router with mocked embedding model."""
    # Mock the embedding model
    mock_model = Mock()
    mock_model.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    router._embedding_model = mock_model
    return router


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_valid_capability(self):
        """Test creating a valid agent capability."""
        cap = AgentCapability(
            name="test",
            capabilities=["test1", "test2"],
            description="Test agent"
        )

        assert cap.name == "test"
        assert cap.capabilities == ["test1", "test2"]
        assert cap.description == "Test agent"
        assert cap.embedding is None
        assert cap.priority == 1.0

    def test_capability_requires_capabilities_list(self):
        """Test that capability requires non-empty capabilities list."""
        with pytest.raises(ValueError, match="must have at least one capability"):
            AgentCapability(
                name="test",
                capabilities=[],
                description="Test agent"
            )

    def test_capability_requires_description(self):
        """Test that capability requires description."""
        with pytest.raises(ValueError, match="must have a description"):
            AgentCapability(
                name="test",
                capabilities=["test"],
                description=""
            )


class TestSemanticAgentRouter:
    """Tests for SemanticAgentRouter class."""

    @pytest.mark.asyncio
    async def test_initialization(self, router):
        """Test router initialization."""
        assert router._initialized is False
        assert router._embedding_model is None
        assert len(router._agent_capabilities) == 0

        # Initialize
        success = await router.initialize()

        # Should succeed (even without embeddings, uses fallback)
        assert success is True
        assert router._initialized is True

        # Should have registered default agents
        assert len(router._agent_capabilities) == 4
        assert "conversation" in router._agent_capabilities
        assert "research" in router._agent_capabilities
        assert "analysis" in router._agent_capabilities
        assert "synthesis" in router._agent_capabilities

    @pytest.mark.asyncio
    async def test_double_initialization(self, router):
        """Test that double initialization is safe."""
        await router.initialize()
        assert router._initialized is True

        # Initialize again
        result = await router.initialize()
        assert result is True
        assert router._initialized is True

    @pytest.mark.asyncio
    async def test_route_conversation_query(self, router):
        """Test routing a general conversation query."""
        await router.initialize()

        query = "Hello, how are you today?"
        agent, confidence = await router.route(query)

        assert agent in ["conversation", "research", "analysis", "synthesis"]
        assert 0.0 <= confidence <= 1.0

        # Should prefer conversation for greeting
        # (exact agent depends on embedding model availability)
        if EMBEDDINGS_AVAILABLE:
            assert agent == "conversation"

    @pytest.mark.asyncio
    async def test_route_research_query(self, router):
        """Test routing a research-oriented query."""
        await router.initialize()

        query = "Search for the latest information about quantum computing"
        agent, confidence = await router.route(query)

        assert agent in ["conversation", "research", "analysis", "synthesis"]
        assert 0.0 <= confidence <= 1.0

        # Should prefer research for search queries
        if EMBEDDINGS_AVAILABLE:
            assert agent == "research"

    @pytest.mark.asyncio
    async def test_route_analysis_query(self, router):
        """Test routing an analysis query."""
        await router.initialize()

        query = "Can you analyze and explain the differences between machine learning approaches?"
        agent, confidence = await router.route(query)

        assert agent in ["conversation", "research", "analysis", "synthesis"]
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_route_synthesis_query(self, router):
        """Test routing a synthesis query."""
        await router.initialize()

        query = "Please summarize the main points from our previous discussions"
        agent, confidence = await router.route(query)

        assert agent in ["conversation", "research", "analysis", "synthesis"]
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_get_top_agents(self, router):
        """Test getting top N agent candidates."""
        await router.initialize()

        query = "Research and analyze recent AI developments"
        candidates = await router.get_top_agents(query, n=2)

        assert len(candidates) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in candidates)

        # Check agents and scores
        for agent, score in candidates:
            assert agent in ["conversation", "research", "analysis", "synthesis"]
            assert 0.0 <= score <= 1.0

        # Should be sorted by score (descending)
        if len(candidates) == 2:
            assert candidates[0][1] >= candidates[1][1]

    @pytest.mark.asyncio
    async def test_keyword_fallback_route(self, router):
        """Test keyword fallback routing when embeddings unavailable."""
        await router.initialize()

        # Test with clear research keywords
        query = "search for information about Python programming"
        agent, confidence = router._keyword_fallback_route(query)

        # Should detect research intent from "search" keyword
        assert agent in router._agent_capabilities.keys()
        assert 0.0 < confidence <= 1.0

    @pytest.mark.asyncio
    async def test_keyword_fallback_default(self, router):
        """Test keyword fallback returns conversation for unclear queries."""
        await router.initialize()

        # Ambiguous query with no clear keywords
        query = "xyz abc 123"
        agent, confidence = router._keyword_fallback_route(query)

        # Should default to conversation
        assert agent == "conversation"
        assert confidence < 0.5  # Low confidence

    @pytest.mark.asyncio
    async def test_cosine_similarity(self, router):
        """Test cosine similarity calculation."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        vec_c = np.array([0.0, 1.0, 0.0])

        # Identical vectors should have similarity 1.0
        sim_same = router._cosine_similarity(vec_a, vec_b)
        assert abs(sim_same - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0.0
        sim_ortho = router._cosine_similarity(vec_a, vec_c)
        assert abs(sim_ortho - 0.0) < 0.001

    @pytest.mark.asyncio
    async def test_cosine_similarity_edge_cases(self, router):
        """Test cosine similarity edge cases."""
        # Empty vectors
        empty = np.array([])
        vec = np.array([1.0, 2.0])
        assert router._cosine_similarity(empty, vec) == 0.0

        # Zero vectors
        zero = np.array([0.0, 0.0])
        assert router._cosine_similarity(zero, vec) == 0.0

    def test_get_agent_info(self, router):
        """Test getting agent information."""
        # Add a test capability
        router._agent_capabilities["test_agent"] = AgentCapability(
            name="test_agent",
            capabilities=["test1", "test2"],
            description="Test agent for testing",
            priority=1.5
        )

        # Get info
        info = router.get_agent_info("test_agent")

        assert info is not None
        assert info["name"] == "test_agent"
        assert info["capabilities"] == ["test1", "test2"]
        assert info["description"] == "Test agent for testing"
        assert info["priority"] == 1.5
        assert "has_embedding" in info

    def test_get_agent_info_not_found(self, router):
        """Test getting info for non-existent agent."""
        info = router.get_agent_info("nonexistent")
        assert info is None

    def test_list_agents(self, router):
        """Test listing registered agents."""
        # Add test capabilities
        router._agent_capabilities["agent1"] = AgentCapability(
            name="agent1",
            capabilities=["cap1"],
            description="Agent 1"
        )
        router._agent_capabilities["agent2"] = AgentCapability(
            name="agent2",
            capabilities=["cap2"],
            description="Agent 2"
        )

        agents = router.list_agents()

        assert "agent1" in agents
        assert "agent2" in agents
        assert len(agents) == 2


class TestSemanticRouterWithMocks:
    """Tests for semantic router with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_route_with_mocked_embeddings(self, router_with_mock_model):
        """Test routing with mocked embedding model."""
        router = router_with_mock_model

        # Register a test agent
        await router._register_agent(
            AgentCapability(
                name="test_agent",
                capabilities=["test"],
                description="Test agent"
            )
        )
        router._initialized = True

        # Route a query
        agent, confidence = await router.route("test query")

        # Should call the mocked model
        assert router._embedding_model.encode.called
        assert agent == "test_agent"
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_generate_embedding_sync(self, router_with_mock_model):
        """Test synchronous embedding generation."""
        router = router_with_mock_model

        embedding = router._generate_embedding("test text")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3
        assert router._embedding_model.encode.called


class TestGlobalRouter:
    """Tests for global router singleton."""

    @pytest.mark.asyncio
    async def test_get_semantic_router_singleton(self):
        """Test that get_semantic_router returns singleton instance."""
        router1 = await get_semantic_router()
        router2 = await get_semantic_router()

        # Should be same instance
        assert router1 is router2
        assert router1._initialized is True


class TestRouterIntegration:
    """Integration tests for semantic router."""

    @pytest.mark.asyncio
    async def test_full_routing_workflow(self):
        """Test complete routing workflow from initialization to routing."""
        # Create and initialize router
        router = SemanticAgentRouter()
        success = await router.initialize()
        assert success is True

        # Test various queries
        test_cases = [
            ("Hello there!", "conversation"),
            ("Search for latest AI news", "research"),
            ("Analyze this data please", "analysis"),
            ("Summarize our discussion", "synthesis"),
        ]

        for query, expected_category in test_cases:
            agent, confidence = await router.route(query)

            # Agent should be one of the registered agents
            assert agent in router.list_agents()
            assert 0.0 <= confidence <= 1.0

            # For clear queries, confidence should be reasonable
            # (exact matching depends on embedding model)
            if EMBEDDINGS_AVAILABLE:
                assert confidence > 0.3

    @pytest.mark.asyncio
    async def test_routing_without_embeddings(self):
        """Test that routing works even without sentence-transformers."""
        router = SemanticAgentRouter()

        # Force no embedding model
        router._embedding_model = None
        await router.initialize()

        # Should still route using keyword fallback
        agent, confidence = await router.route("search for information")

        assert agent in router.list_agents()
        assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_router_concurrency():
    """Test that router handles concurrent requests safely."""
    router = SemanticAgentRouter()
    await router.initialize()

    # Route multiple queries concurrently
    queries = [
        "Hello there",
        "Search for AI news",
        "Analyze this data",
        "Summarize the results"
    ]

    # Execute all routes concurrently
    results = await asyncio.gather(
        *[router.route(q) for q in queries]
    )

    # All should succeed
    assert len(results) == len(queries)
    for agent, confidence in results:
        assert agent in router.list_agents()
        assert 0.0 <= confidence <= 1.0
