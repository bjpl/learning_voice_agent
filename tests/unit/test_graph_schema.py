"""
Unit Tests for Graph Schema Definitions
PATTERN: Type and data structure validation
WHY: Ensure graph schema components work correctly

Test Coverage:
- Node type enumerations
- Relationship type enumerations
- Data model conversions
- Cypher query builders
- Helper functions
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from app.vector.schema import (
    NodeType,
    RelationshipType,
    ConversationNode,
    ConceptNode,
    TopicNode,
    EntityNode,
    Relationship,
    CypherQueryBuilder,
    create_conversation_node_cypher,
    create_concept_node_cypher,
    create_relationship_cypher
)


# ===== Enum Tests =====

def test_node_type_enum():
    """Test NodeType enum values"""
    assert NodeType.CONVERSATION == "Conversation"
    assert NodeType.CONCEPT == "Concept"
    assert NodeType.TOPIC == "Topic"
    assert NodeType.ENTITY == "Entity"
    assert NodeType.SESSION == "Session"


def test_relationship_type_enum():
    """Test RelationshipType enum values"""
    assert RelationshipType.ABOUT == "ABOUT"
    assert RelationshipType.RELATES_TO == "RELATES_TO"
    assert RelationshipType.FOLLOWS == "FOLLOWS"
    assert RelationshipType.MENTIONED_IN == "MENTIONED_IN"
    assert RelationshipType.INSTANCE_OF == "INSTANCE_OF"
    assert RelationshipType.CONTAINS == "CONTAINS"
    assert RelationshipType.BUILDS_ON == "BUILDS_ON"


# ===== ConversationNode Tests =====

def test_conversation_node_creation():
    """Test ConversationNode creation"""
    timestamp = datetime.utcnow()
    node = ConversationNode(
        id="conv_123",
        session_id="session_abc",
        timestamp=timestamp,
        user_text="What is AI?",
        agent_text="AI is artificial intelligence...",
        metadata={"source": "voice"}
    )

    assert node.id == "conv_123"
    assert node.session_id == "session_abc"
    assert node.timestamp == timestamp
    assert node.user_text == "What is AI?"
    assert node.metadata["source"] == "voice"


def test_conversation_node_with_embedding():
    """Test ConversationNode with embedding vector"""
    embedding = [0.1, 0.2, 0.3] * 128  # 384-dim
    node = ConversationNode(
        id="conv_123",
        session_id="session_abc",
        timestamp=datetime.utcnow(),
        embedding=embedding
    )

    assert node.embedding == embedding
    assert len(node.embedding) == 384


def test_conversation_node_to_cypher_props():
    """Test conversion to Cypher properties"""
    timestamp = datetime.utcnow()
    node = ConversationNode(
        id="conv_123",
        session_id="session_abc",
        timestamp=timestamp,
        user_text="A" * 600,  # Long text
        agent_text="B" * 600,
        metadata={"key": "value"}
    )

    props = node.to_cypher_props()

    assert props["id"] == "conv_123"
    assert props["session_id"] == "session_abc"
    assert props["timestamp"] == timestamp.isoformat()
    assert len(props["user_text"]) <= 500  # Should be truncated
    assert len(props["agent_text"]) <= 500
    assert props["key"] == "value"
    assert "embedding" not in props  # Embeddings excluded


# ===== ConceptNode Tests =====

def test_concept_node_creation():
    """Test ConceptNode creation"""
    node = ConceptNode(
        name="machine learning",
        description="ML is a subset of AI",
        category="AI",
        frequency=10
    )

    assert node.name == "machine learning"
    assert node.description == "ML is a subset of AI"
    assert node.category == "AI"
    assert node.frequency == 10


def test_concept_node_minimal():
    """Test ConceptNode with minimal fields"""
    node = ConceptNode(name="python")

    assert node.name == "python"
    assert node.description is None
    assert node.frequency == 0


def test_concept_node_to_cypher_props():
    """Test ConceptNode Cypher property conversion"""
    first_seen = datetime.utcnow()
    last_seen = datetime.utcnow()

    node = ConceptNode(
        name="deep learning",
        description="Multi-layer neural networks",
        category="AI",
        frequency=5,
        first_seen=first_seen,
        last_seen=last_seen,
        metadata={"level": "advanced"}
    )

    props = node.to_cypher_props()

    assert props["name"] == "deep learning"
    assert props["description"] == "Multi-layer neural networks"
    assert props["category"] == "AI"
    assert props["frequency"] == 5
    assert props["first_seen"] == first_seen.isoformat()
    assert props["last_seen"] == last_seen.isoformat()
    assert props["level"] == "advanced"


# ===== TopicNode Tests =====

def test_topic_node_creation():
    """Test TopicNode creation"""
    node = TopicNode(
        name="Artificial Intelligence",
        description="AI and ML topics",
        parent_topic="Computer Science"
    )

    assert node.name == "Artificial Intelligence"
    assert node.description == "AI and ML topics"
    assert node.parent_topic == "Computer Science"


def test_topic_node_to_cypher_props():
    """Test TopicNode Cypher properties"""
    node = TopicNode(
        name="Machine Learning",
        description="ML subtopics",
        metadata={"priority": "high"}
    )

    props = node.to_cypher_props()

    assert props["name"] == "Machine Learning"
    assert props["description"] == "ML subtopics"
    assert props["priority"] == "high"


# ===== EntityNode Tests =====

def test_entity_node_creation():
    """Test EntityNode creation"""
    node = EntityNode(
        text="Albert Einstein",
        entity_type="PERSON",
        metadata={"field": "physics"}
    )

    assert node.text == "Albert Einstein"
    assert node.entity_type == "PERSON"
    assert node.metadata["field"] == "physics"


def test_entity_node_to_cypher_props():
    """Test EntityNode Cypher properties"""
    node = EntityNode(
        text="OpenAI",
        entity_type="ORG",
        metadata={"founded": "2015"}
    )

    props = node.to_cypher_props()

    assert props["text"] == "OpenAI"
    assert props["type"] == "ORG"
    assert props["founded"] == "2015"


# ===== Relationship Tests =====

def test_relationship_creation():
    """Test Relationship creation"""
    timestamp = datetime.utcnow()
    rel = Relationship(
        from_node="conv_123",
        to_node="concept_ml",
        rel_type=RelationshipType.ABOUT,
        strength=0.9,
        context="User asked about machine learning",
        timestamp=timestamp
    )

    assert rel.from_node == "conv_123"
    assert rel.to_node == "concept_ml"
    assert rel.rel_type == RelationshipType.ABOUT
    assert rel.strength == 0.9
    assert rel.context == "User asked about machine learning"


def test_relationship_to_cypher_props():
    """Test Relationship Cypher properties"""
    timestamp = datetime.utcnow()
    rel = Relationship(
        from_node="a",
        to_node="b",
        rel_type=RelationshipType.RELATES_TO,
        strength=0.75,
        context="Related concepts",
        timestamp=timestamp,
        metadata={"confidence": 0.8}
    )

    props = rel.to_cypher_props()

    assert props["strength"] == 0.75
    assert props["context"] == "Related concepts"
    assert props["timestamp"] == timestamp.isoformat()
    assert props["confidence"] == 0.8


# ===== CypherQueryBuilder Tests =====

def test_cypher_builder_basic():
    """Test basic Cypher query building"""
    builder = CypherQueryBuilder()
    query, params = builder \
        .match_node(NodeType.CONCEPT, {"name": "test"}) \
        .return_fields(["c.name"]) \
        .build()

    assert "MATCH" in query
    assert "Concept" in query
    assert "RETURN" in query
    assert "param_0" in params
    assert params["param_0"] == "test"


def test_cypher_builder_with_relationship():
    """Test query builder with relationships"""
    builder = CypherQueryBuilder()
    query, params = builder \
        .match_node(NodeType.CONVERSATION, {"id": "conv_123"}, alias="conv") \
        .match_relationship(RelationshipType.ABOUT, direction="->") \
        .match_node(NodeType.CONCEPT, alias="c") \
        .return_fields(["conv.id", "c.name"]) \
        .build()

    assert "MATCH (conv:Conversation" in query
    assert "[:ABOUT]->" in query
    assert "(c:Concept)" in query
    assert "RETURN" in query


def test_cypher_builder_with_where():
    """Test query builder with WHERE clause"""
    builder = CypherQueryBuilder()
    query, params = builder \
        .match_node(NodeType.CONCEPT) \
        .where("c.frequency > 5") \
        .return_fields(["c.name"]) \
        .build()

    assert "WHERE c.frequency > 5" in query


def test_cypher_builder_with_limit_order():
    """Test query builder with LIMIT and ORDER BY"""
    builder = CypherQueryBuilder()
    query, params = builder \
        .match_node(NodeType.CONCEPT) \
        .return_fields(["c.name", "c.frequency"]) \
        .order_by("c.frequency", descending=True) \
        .limit(10) \
        .build()

    assert "ORDER BY c.frequency DESC" in query
    assert "LIMIT 10" in query


def test_cypher_builder_error_relationship_without_node():
    """Test that relationship requires a prior node"""
    builder = CypherQueryBuilder()

    with pytest.raises(ValueError, match="Must call match_node"):
        builder.match_relationship(RelationshipType.ABOUT)


# ===== Helper Function Tests =====

def test_create_conversation_node_cypher():
    """Test conversation node Cypher generation"""
    timestamp = datetime.utcnow()
    node = ConversationNode(
        id="conv_123",
        session_id="session_abc",
        timestamp=timestamp,
        user_text="Test user text",
        agent_text="Test agent text"
    )

    query, params = create_conversation_node_cypher(node)

    assert "MERGE (c:Conversation {id: $id})" in query
    assert "ON CREATE SET" in query
    assert "ON MATCH SET" in query
    assert params["id"] == "conv_123"
    assert params["session_id"] == "session_abc"
    assert params["user_text"] == "Test user text"


def test_create_concept_node_cypher():
    """Test concept node Cypher generation"""
    node = ConceptNode(
        name="machine learning",
        description="ML description",
        category="AI"
    )

    query, params = create_concept_node_cypher(node)

    assert "MERGE (c:Concept {name: $name})" in query
    assert "c.frequency = 1" in query
    assert "c.frequency = c.frequency + 1" in query
    assert params["name"] == "machine learning"
    assert params["description"] == "ML description"


def test_create_relationship_cypher():
    """Test relationship Cypher generation"""
    rel = Relationship(
        from_node="node_a",
        to_node="node_b",
        rel_type=RelationshipType.RELATES_TO,
        strength=0.8,
        context="Test context"
    )

    query, params = create_relationship_cypher(rel)

    assert "MATCH (a {id: $from_node})" in query
    assert "MATCH (b {id: $to_node})" in query
    assert f"MERGE (a)-[r:{RelationshipType.RELATES_TO.value}]->(b)" in query
    assert "r.strength = r.strength" in query  # Incremental update
    assert params["from_node"] == "node_a"
    assert params["strength"] == 0.8


# ===== Edge Cases =====

def test_empty_metadata():
    """Test nodes with empty metadata"""
    node = ConceptNode(name="test", metadata={})
    props = node.to_cypher_props()

    assert "name" in props
    assert props["name"] == "test"


def test_none_optional_fields():
    """Test handling of None optional fields"""
    node = ConceptNode(
        name="test",
        description=None,
        category=None
    )
    props = node.to_cypher_props()

    assert "description" not in props
    assert "category" not in props
    assert props["name"] == "test"


def test_cypher_builder_multiple_parameters():
    """Test builder with multiple parameters"""
    builder = CypherQueryBuilder()
    query, params = builder \
        .match_node(NodeType.CONCEPT, {"name": "ml", "category": "AI"}) \
        .return_fields(["c.name"]) \
        .build()

    assert len(params) == 2
    assert "ml" in params.values()
    assert "AI" in params.values()


# ===== Integration Tests =====

def test_full_graph_pattern():
    """Test building a complete graph pattern"""
    # Create nodes
    conv_node = ConversationNode(
        id="conv_1",
        session_id="session_1",
        timestamp=datetime.utcnow()
    )

    concept_node = ConceptNode(
        name="python",
        description="Programming language"
    )

    # Create relationship
    rel = Relationship(
        from_node="conv_1",
        to_node="python",
        rel_type=RelationshipType.ABOUT,
        strength=0.9
    )

    # Generate Cypher
    conv_query, conv_params = create_conversation_node_cypher(conv_node)
    concept_query, concept_params = create_concept_node_cypher(concept_node)
    rel_query, rel_params = create_relationship_cypher(rel)

    # Assert all queries generated
    assert "MERGE" in conv_query
    assert "MERGE" in concept_query
    assert "MERGE" in rel_query
    assert all(isinstance(p, dict) for p in [conv_params, concept_params, rel_params])
