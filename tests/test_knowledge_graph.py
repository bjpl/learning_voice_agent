"""
Tests for Knowledge Graph Module

PATTERN: Comprehensive test coverage with mocks
WHY: Ensure reliability, prevent regressions
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from app.knowledge_graph.config import KnowledgeGraphConfig
from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.knowledge_graph.concept_extractor import ConceptExtractor
from app.knowledge_graph.query_engine import GraphQueryEngine


# Fixtures

@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = KnowledgeGraphConfig()
    config.embedded = True
    config.data_path = "/tmp/neo4j_test"
    return config


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver"""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()

    # Mock session
    session = AsyncMock()
    session.run = AsyncMock()

    driver.session = MagicMock(return_value=session)

    return driver


@pytest.fixture
def graph_store(mock_config):
    """Create graph store instance"""
    return KnowledgeGraphStore(mock_config)


@pytest.fixture
def mock_analysis_agent():
    """Mock AnalysisAgent"""
    agent = AsyncMock()

    # Mock analysis result
    async def mock_process(message):
        return Mock(
            content={
                "concepts": ["machine learning", "neural networks", "deep learning"],
                "entities": [
                    {"text": "Python", "label": "PRODUCT"},
                    {"text": "TensorFlow", "label": "PRODUCT"}
                ],
                "keywords": [
                    {"keyword": "algorithm", "frequency": 3},
                    {"keyword": "training", "frequency": 2}
                ],
                "topics": [
                    {"topic": "technology", "confidence": 0.9}
                ],
                "sentiment": {"polarity": 0.7, "label": "positive"}
            }
        )

    agent.process = mock_process
    return agent


@pytest.fixture
def concept_extractor(graph_store, mock_analysis_agent):
    """Create concept extractor instance"""
    return ConceptExtractor(graph_store, mock_analysis_agent)


@pytest.fixture
def query_engine(graph_store):
    """Create query engine instance"""
    return GraphQueryEngine(graph_store)


# KnowledgeGraphStore Tests

class TestKnowledgeGraphStore:
    """Tests for KnowledgeGraphStore"""

    @pytest.mark.asyncio
    async def test_initialization(self, graph_store, mock_config):
        """Test graph store initialization"""
        with patch('app.knowledge_graph.graph_store.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_db.driver.return_value = mock_driver

            # Mock session for schema creation
            mock_session = AsyncMock()
            mock_session.run = AsyncMock()
            mock_driver.session.return_value.__aenter__.return_value = mock_session

            await graph_store.initialize()

            assert graph_store._initialized is True
            mock_db.driver.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_concept(self, graph_store):
        """Test adding concept to graph"""
        # Mock driver and session
        graph_store._initialized = True
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single.return_value = {"name": "machine learning", "frequency": 1}
        mock_session.run.return_value = mock_result

        with patch.object(graph_store, 'session') as mock_ctx:
            mock_ctx.return_value.__aenter__.return_value = mock_session

            concept_id = await graph_store.add_concept(
                name="machine learning",
                description="ML is awesome",
                metadata={"category": "AI"}
            )

            assert concept_id == "machine learning"
            mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_relationship(self, graph_store):
        """Test adding relationship between concepts"""
        graph_store._initialized = True
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        with patch.object(graph_store, 'session') as mock_ctx:
            mock_ctx.return_value.__aenter__.return_value = mock_session

            await graph_store.add_relationship(
                from_concept="neural networks",
                to_concept="deep learning",
                relationship_type="RELATES_TO",
                strength=0.9
            )

            mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_related_concepts(self, graph_store):
        """Test getting related concepts"""
        graph_store._initialized = True
        mock_session = AsyncMock()

        # Mock query result
        mock_result = AsyncMock()
        mock_result.values.return_value = [
            ["deep learning", "Advanced ML", 5, ["RELATES_TO"], [0.8], 1],
            ["supervised learning", "ML technique", 3, ["BUILDS_ON"], [0.7], 2]
        ]
        mock_session.run.return_value = mock_result

        with patch.object(graph_store, 'session') as mock_ctx:
            mock_ctx.return_value.__aenter__.return_value = mock_session

            related = await graph_store.get_related_concepts(
                concept="machine learning",
                max_depth=2,
                min_strength=0.5
            )

            assert len(related) == 2
            assert related[0]["name"] == "deep learning"
            assert related[0]["frequency"] == 5

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, graph_store):
        """Test getting graph statistics"""
        graph_store._initialized = True
        mock_session = AsyncMock()

        mock_result = AsyncMock()
        mock_result.single.return_value = {
            "concept_count": 10,
            "relationship_count": 15,
            "entity_count": 5,
            "session_count": 3,
            "topic_count": 2
        }
        mock_session.run.return_value = mock_result

        with patch.object(graph_store, 'session') as mock_ctx:
            mock_ctx.return_value.__aenter__.return_value = mock_session

            stats = await graph_store.get_graph_stats()

            assert stats["concepts"] == 10
            assert stats["relationships"] == 15
            assert stats["entities"] == 5


# ConceptExtractor Tests

class TestConceptExtractor:
    """Tests for ConceptExtractor"""

    @pytest.mark.asyncio
    async def test_extract_from_text(self, concept_extractor):
        """Test extracting concepts from text"""
        text = "I'm learning about machine learning and neural networks in Python"

        # Mock graph store methods
        concept_extractor.graph_store.add_concept = AsyncMock()
        concept_extractor.graph_store.add_entity = AsyncMock()
        concept_extractor.graph_store.add_relationship = AsyncMock()

        result = await concept_extractor.extract_from_text(text)

        assert "concepts" in result
        assert "entities" in result
        assert "relationships" in result
        assert "stats" in result
        assert len(result["concepts"]) > 0

    @pytest.mark.asyncio
    async def test_extract_concepts_from_analysis(self, concept_extractor):
        """Test extracting concepts from analysis results"""
        analysis = {
            "concepts": ["machine learning", "deep learning", "ai"],
            "keywords": [
                {"keyword": "algorithm", "frequency": 3},
                {"keyword": "model", "frequency": 2}
            ]
        }

        concepts = concept_extractor._extract_concepts_from_analysis(analysis)

        assert "machine learning" in concepts
        assert "deep learning" in concepts
        assert len(concepts) >= 3

    @pytest.mark.asyncio
    async def test_normalize_concept(self, concept_extractor):
        """Test concept normalization"""
        assert concept_extractor._normalize_concept("Machine Learning") == "machine learning"
        assert concept_extractor._normalize_concept("  The Algorithm  ") == "algorithm"
        assert concept_extractor._normalize_concept("A Deep  Learning") == "deep learning"

    @pytest.mark.asyncio
    async def test_classify_topics(self, concept_extractor):
        """Test topic classification"""
        text = "I'm learning machine learning and deep learning with neural networks"
        concepts = ["machine learning", "deep learning", "neural networks"]

        topics = concept_extractor._classify_topics(text, concepts)

        assert len(topics) > 0
        assert "Artificial Intelligence" in topics or "Computer Science" in topics

    @pytest.mark.asyncio
    async def test_detect_relationships(self, concept_extractor):
        """Test relationship detection"""
        concepts = ["machine learning", "neural networks", "deep learning"]
        text = "Machine learning includes neural networks and deep learning techniques"
        analysis = {"concepts": concepts}

        relationships = await concept_extractor._detect_relationships(
            concepts, text, analysis
        )

        assert len(relationships) > 0
        assert all("from_concept" in r for r in relationships)
        assert all("to_concept" in r for r in relationships)
        assert all("strength" in r for r in relationships)

    @pytest.mark.asyncio
    async def test_calculate_relationship_strength(self, concept_extractor):
        """Test relationship strength calculation"""
        strength = concept_extractor._calculate_relationship_strength(
            "machine learning",
            "deep learning",
            "machine learning and deep learning",
            [0],
            [28],
            28
        )

        assert 0.0 <= strength <= 1.0

    @pytest.mark.asyncio
    async def test_extract_from_conversation(self, concept_extractor):
        """Test extracting from conversation"""
        exchanges = [
            {"user": "What is machine learning?", "agent": "ML is..."},
            {"user": "Tell me about neural networks", "agent": "Neural networks are..."}
        ]

        # Mock graph store methods
        concept_extractor.graph_store.add_concept = AsyncMock()
        concept_extractor.graph_store.add_entity = AsyncMock()
        concept_extractor.graph_store.add_relationship = AsyncMock()
        concept_extractor.graph_store.add_session = AsyncMock()

        result = await concept_extractor.extract_from_conversation(
            session_id="test_session",
            exchanges=exchanges
        )

        assert result["session_id"] == "test_session"
        assert "concepts" in result
        assert "stats" in result
        assert result["stats"]["exchange_count"] == 2


# GraphQueryEngine Tests

class TestGraphQueryEngine:
    """Tests for GraphQueryEngine"""

    @pytest.mark.asyncio
    async def test_find_related_concepts(self, query_engine):
        """Test finding related concepts"""
        # Mock graph store
        query_engine.graph_store.get_related_concepts = AsyncMock(
            return_value=[
                {
                    "name": "deep learning",
                    "description": "Advanced ML",
                    "frequency": 5,
                    "relationship_types": ["RELATES_TO"],
                    "strengths": [0.8],
                    "distance": 1
                },
                {
                    "name": "supervised learning",
                    "description": "ML with labels",
                    "frequency": 3,
                    "relationship_types": ["BUILDS_ON"],
                    "strengths": [0.7],
                    "distance": 2
                }
            ]
        )

        related = await query_engine.find_related_concepts(
            concept="machine learning",
            max_depth=2
        )

        assert len(related) == 2
        assert "relevance_score" in related[0]
        # Should be sorted by relevance
        assert related[0]["relevance_score"] >= related[1]["relevance_score"]

    @pytest.mark.asyncio
    async def test_rank_concepts(self, query_engine):
        """Test concept ranking"""
        concepts = [
            {
                "name": "concept1",
                "distance": 2,
                "frequency": 5,
                "strengths": [0.7, 0.8]
            },
            {
                "name": "concept2",
                "distance": 1,
                "frequency": 3,
                "strengths": [0.9]
            }
        ]

        ranked = query_engine._rank_concepts(concepts, "origin")

        assert len(ranked) == 2
        assert all("relevance_score" in c for c in ranked)
        # concept2 should rank higher (closer distance)
        assert ranked[0]["name"] == "concept2"

    @pytest.mark.asyncio
    async def test_get_learning_path(self, query_engine):
        """Test learning path discovery"""
        # Mock session and query result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.values.return_value = [
            [
                ["python", "programming", "machine learning"],  # path
                ["BUILDS_ON", "RELATES_TO"],  # rel_types
                [0.8, 0.7],  # strengths
                2,  # steps
                2.5  # total_cost
            ]
        ]
        mock_session.run.return_value = mock_result

        # Mock get_concept for path details
        async def mock_get_concept(name):
            return {
                "name": name,
                "description": f"Description of {name}",
                "frequency": 5
            }

        query_engine.graph_store.get_concept = mock_get_concept

        with patch.object(query_engine.graph_store, 'session') as mock_ctx:
            mock_ctx.return_value.__aenter__.return_value = mock_session

            path = await query_engine.get_learning_path(
                from_concept="python",
                to_concept="machine learning"
            )

            assert path["found"] is True
            assert path["from_concept"] == "python"
            assert path["to_concept"] == "machine learning"
            assert "path" in path
            assert "difficulty" in path
            assert "estimated_time" in path

    @pytest.mark.asyncio
    async def test_identify_knowledge_gaps(self, query_engine):
        """Test knowledge gap identification"""
        # Mock get_learning_path
        async def mock_learning_path(from_concept, to_concept, max_depth=3):
            return {
                "found": True,
                "path": ["python", "programming", "algorithms", "machine learning"],
                "steps": 3,
                "difficulty": "moderate"
            }

        query_engine.get_learning_path = mock_learning_path

        # Mock get_concept
        async def mock_get_concept(name):
            return {
                "name": name,
                "description": f"Description of {name}",
                "frequency": 3
            }

        # Mock get_related_concepts
        async def mock_get_related(concept, max_depth, limit):
            return [
                {"name": "python", "distance": 1},
                {"name": "machine learning", "distance": 1}
            ]

        query_engine.graph_store.get_concept = mock_get_concept
        query_engine.graph_store.get_related_concepts = mock_get_related

        gaps = await query_engine.identify_knowledge_gaps(
            known_concepts=["python"],
            target_concept="machine learning",
            max_depth=3
        )

        assert "gaps" in gaps
        assert "recommendations" in gaps
        assert gaps["target_concept"] == "machine learning"

    @pytest.mark.asyncio
    async def test_get_concept_map(self, query_engine):
        """Test concept map generation"""
        # Mock find_related_concepts
        async def mock_find_related(concept, max_depth, limit):
            return [
                {
                    "name": "deep learning",
                    "frequency": 5,
                    "distance": 1,
                    "relationship_types": ["RELATES_TO"],
                    "strengths": [0.8]
                },
                {
                    "name": "supervised learning",
                    "frequency": 3,
                    "distance": 2,
                    "relationship_types": ["BUILDS_ON"],
                    "strengths": [0.7]
                }
            ]

        query_engine.find_related_concepts = mock_find_related

        concept_map = await query_engine.get_concept_map(
            central_concept="machine learning",
            radius=2
        )

        assert "nodes" in concept_map
        assert "edges" in concept_map
        assert concept_map["central_concept"] == "machine learning"
        assert len(concept_map["nodes"]) >= 3  # Central + 2 related
        assert len(concept_map["edges"]) >= 2

    @pytest.mark.asyncio
    async def test_estimate_difficulty(self, query_engine):
        """Test difficulty estimation"""
        assert query_engine._estimate_difficulty(1, [0.8]) == "easy"
        assert query_engine._estimate_difficulty(2, [0.6]) == "moderate"
        assert query_engine._estimate_difficulty(3, [0.5]) == "challenging"
        assert query_engine._estimate_difficulty(5, [0.4]) == "advanced"

    @pytest.mark.asyncio
    async def test_estimate_learning_time(self, query_engine):
        """Test learning time estimation"""
        assert "minutes" in query_engine._estimate_learning_time(1)
        assert "hours" in query_engine._estimate_learning_time(3)
        assert "hours" in query_engine._estimate_learning_time(5)


# Integration Tests

class TestKnowledgeGraphIntegration:
    """Integration tests for knowledge graph"""

    @pytest.mark.asyncio
    async def test_full_extraction_and_query_flow(
        self,
        concept_extractor,
        query_engine
    ):
        """Test full flow: extract concepts -> store -> query"""
        # Mock all graph store operations
        concept_extractor.graph_store.add_concept = AsyncMock()
        concept_extractor.graph_store.add_entity = AsyncMock()
        concept_extractor.graph_store.add_relationship = AsyncMock()

        # Extract from text
        text = "Machine learning and deep learning are related to artificial intelligence"

        extraction_result = await concept_extractor.extract_from_text(text)

        assert extraction_result["stats"]["concept_count"] > 0

        # Mock query result
        query_engine.graph_store.get_related_concepts = AsyncMock(
            return_value=[
                {
                    "name": "artificial intelligence",
                    "frequency": 1,
                    "relationship_types": ["RELATES_TO"],
                    "strengths": [0.7],
                    "distance": 1
                }
            ]
        )

        # Query related concepts
        related = await query_engine.find_related_concepts(
            concept="machine learning",
            max_depth=2
        )

        assert len(related) > 0

    @pytest.mark.asyncio
    async def test_conversation_to_learning_path(
        self,
        concept_extractor,
        query_engine
    ):
        """Test conversation extraction followed by learning path"""
        # Mock graph operations
        concept_extractor.graph_store.add_concept = AsyncMock()
        concept_extractor.graph_store.add_entity = AsyncMock()
        concept_extractor.graph_store.add_relationship = AsyncMock()
        concept_extractor.graph_store.add_session = AsyncMock()

        # Extract from conversation
        exchanges = [
            {"user": "I know Python basics", "agent": "Great!"},
            {"user": "How do I learn machine learning?", "agent": "Start with..."}
        ]

        result = await concept_extractor.extract_from_conversation(
            session_id="test_session",
            exchanges=exchanges
        )

        assert result["stats"]["concept_count"] > 0
