"""
Graph Schema Definitions - RuVector Knowledge Graph Structure
PATTERN: Schema-first design with type definitions
WHY: Consistent graph structure across vector and knowledge graph operations

SPECIFICATION:
- Define node types for conversations, concepts, topics
- Define relationship types for semantic connections
- Provide Pydantic models for validation
- Support embedding vectors in graph nodes

ARCHITECTURE:
[Conversation Node] --ABOUT--> [Concept Node] --RELATES_TO--> [Concept Node]
       |                           |
       |                           +---> [Topic Node] (CONTAINS)
       |
       +--FOLLOWS--> [Conversation Node]

REFINEMENT:
- Use dataclasses for lightweight models
- Support optional embeddings for hybrid operations
- Enable temporal tracking via timestamps
- Provide helper functions for Cypher generation
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ===== Node Type Constants =====

class NodeType(str, Enum):
    """
    Graph node types

    CONCEPT: Enumeration for type safety
    WHY: Prevent typos, enable IDE autocomplete
    """
    CONVERSATION = "Conversation"
    CONCEPT = "Concept"
    TOPIC = "Topic"
    ENTITY = "Entity"
    SESSION = "Session"


# ===== Relationship Type Constants =====

class RelationshipType(str, Enum):
    """
    Graph relationship types

    CONCEPT: Semantic relationship categorization
    WHY: Enable graph traversal with meaningful edges
    """
    ABOUT = "ABOUT"  # Conversation is about a concept
    RELATES_TO = "RELATES_TO"  # Concept relates to another concept
    FOLLOWS = "FOLLOWS"  # Conversation follows another chronologically
    MENTIONED_IN = "MENTIONED_IN"  # Concept/entity mentioned in session
    INSTANCE_OF = "INSTANCE_OF"  # Entity is instance of concept
    CONTAINS = "CONTAINS"  # Topic contains concepts
    BUILDS_ON = "BUILDS_ON"  # Concept builds on another (prerequisite)


# ===== Node Data Models =====

@dataclass
class ConversationNode:
    """
    Conversation node with optional embedding

    PATTERN: Immutable dataclass with validation
    WHY: Type-safe graph operations

    Example:
        node = ConversationNode(
            id="conv_123",
            embedding=[0.1, 0.2, ...],
            session_id="session_abc",
            timestamp=datetime.utcnow()
        )
    """
    id: str
    session_id: str
    timestamp: datetime
    embedding: Optional[List[float]] = None
    user_text: Optional[str] = None
    agent_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_props(self) -> Dict[str, Any]:
        """
        Convert to Cypher properties (excludes embedding)

        CONCEPT: Separate embeddings from graph properties
        WHY: Embeddings handled by vector layer, metadata in graph layer
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_text": self.user_text[:500] if self.user_text else None,
            "agent_text": self.agent_text[:500] if self.agent_text else None,
            **self.metadata
        }


@dataclass
class ConceptNode:
    """
    Concept node with optional embedding

    PATTERN: Semantic concept representation
    WHY: Bridge between semantic search and knowledge graph

    Concepts can have embeddings for:
    - Semantic similarity queries
    - Hybrid vector + graph search
    - Concept clustering
    """
    name: str
    embedding: Optional[List[float]] = None
    description: Optional[str] = None
    category: Optional[str] = None
    frequency: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_props(self) -> Dict[str, Any]:
        """Convert to Cypher properties"""
        props = {
            "name": self.name,
            "frequency": self.frequency,
        }

        if self.description:
            props["description"] = self.description

        if self.category:
            props["category"] = self.category

        if self.first_seen:
            props["first_seen"] = self.first_seen.isoformat()

        if self.last_seen:
            props["last_seen"] = self.last_seen.isoformat()

        props.update(self.metadata)
        return props


@dataclass
class TopicNode:
    """
    Topic node for hierarchical organization

    PATTERN: Hierarchical categorization
    WHY: Enable topic-based filtering and navigation
    """
    name: str
    description: Optional[str] = None
    parent_topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_props(self) -> Dict[str, Any]:
        """Convert to Cypher properties"""
        props = {"name": self.name}

        if self.description:
            props["description"] = self.description

        if self.parent_topic:
            props["parent_topic"] = self.parent_topic

        props.update(self.metadata)
        return props


@dataclass
class EntityNode:
    """
    Entity node (person, organization, location, etc.)

    PATTERN: Named entity representation
    WHY: Track specific entities mentioned in conversations
    """
    text: str
    entity_type: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_props(self) -> Dict[str, Any]:
        """Convert to Cypher properties"""
        return {
            "text": self.text,
            "type": self.entity_type,
            **self.metadata
        }


# ===== Relationship Data Models =====

@dataclass
class Relationship:
    """
    Graph relationship with metadata

    PATTERN: Rich relationship with properties
    WHY: Store relationship strength, context, temporal data
    """
    from_node: str
    to_node: str
    rel_type: RelationshipType
    strength: float = 1.0
    context: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cypher_props(self) -> Dict[str, Any]:
        """Convert to Cypher relationship properties"""
        props = {
            "strength": self.strength,
        }

        if self.context:
            props["context"] = self.context

        if self.timestamp:
            props["timestamp"] = self.timestamp.isoformat()

        props.update(self.metadata)
        return props


# ===== Cypher Query Builders =====

class CypherQueryBuilder:
    """
    Helper class for building Cypher queries

    PATTERN: Fluent query builder
    WHY: Type-safe, composable query construction

    Example:
        query = CypherQueryBuilder() \
            .match_node(NodeType.CONVERSATION, {"id": "conv_123"}) \
            .match_relationship(RelationshipType.ABOUT) \
            .match_node(NodeType.CONCEPT) \
            .return_fields(["concept.name", "concept.frequency"]) \
            .build()
    """

    def __init__(self):
        self._clauses: List[str] = []
        self._parameters: Dict[str, Any] = {}
        self._param_counter = 0

    def match_node(
        self,
        node_type: NodeType,
        properties: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None
    ) -> 'CypherQueryBuilder':
        """
        Add MATCH clause for node

        Args:
            node_type: Type of node
            properties: Node properties to match
            alias: Variable alias (auto-generated if None)
        """
        alias = alias or node_type.value.lower()

        if properties:
            prop_clauses = []
            for key, value in properties.items():
                param_name = f"param_{self._param_counter}"
                self._param_counter += 1
                prop_clauses.append(f"{key}: ${param_name}")
                self._parameters[param_name] = value

            props_str = "{" + ", ".join(prop_clauses) + "}"
            self._clauses.append(f"MATCH ({alias}:{node_type.value} {props_str})")
        else:
            self._clauses.append(f"MATCH ({alias}:{node_type.value})")

        return self

    def match_relationship(
        self,
        rel_type: RelationshipType,
        direction: str = "->",
        alias: Optional[str] = None
    ) -> 'CypherQueryBuilder':
        """
        Add relationship pattern to last MATCH clause

        Args:
            rel_type: Relationship type
            direction: Arrow direction (-> or <-)
            alias: Relationship variable alias
        """
        if not self._clauses:
            raise ValueError("Must call match_node before match_relationship")

        rel_alias = f"[{alias}:{rel_type.value}]" if alias else f"[:{rel_type.value}]"

        if direction == "->":
            self._clauses[-1] += f"-{rel_alias}->"
        elif direction == "<-":
            self._clauses[-1] += f"<-{rel_alias}-"
        else:
            self._clauses[-1] += f"-{rel_alias}-"

        return self

    def where(self, condition: str) -> 'CypherQueryBuilder':
        """Add WHERE clause"""
        self._clauses.append(f"WHERE {condition}")
        return self

    def return_fields(self, fields: List[str]) -> 'CypherQueryBuilder':
        """Add RETURN clause"""
        self._clauses.append(f"RETURN {', '.join(fields)}")
        return self

    def limit(self, count: int) -> 'CypherQueryBuilder':
        """Add LIMIT clause"""
        self._clauses.append(f"LIMIT {count}")
        return self

    def order_by(self, field: str, descending: bool = False) -> 'CypherQueryBuilder':
        """Add ORDER BY clause"""
        direction = "DESC" if descending else "ASC"
        self._clauses.append(f"ORDER BY {field} {direction}")
        return self

    def build(self) -> tuple[str, Dict[str, Any]]:
        """
        Build final query and parameters

        Returns:
            Tuple of (query_string, parameters_dict)
        """
        query = "\n".join(self._clauses)
        return query, self._parameters


# ===== Helper Functions =====

def create_conversation_node_cypher(node: ConversationNode) -> tuple[str, Dict[str, Any]]:
    """
    Generate Cypher query to create/merge conversation node

    PATTERN: MERGE for idempotent writes
    WHY: Safe to call multiple times, updates on conflict
    """
    query = """
    MERGE (c:Conversation {id: $id})
    ON CREATE SET
        c.session_id = $session_id,
        c.timestamp = datetime($timestamp),
        c.user_text = $user_text,
        c.agent_text = $agent_text,
        c.created_at = datetime()
    ON MATCH SET
        c.updated_at = datetime()
    SET c += $metadata
    RETURN c
    """

    props = node.to_cypher_props()
    params = {
        "id": node.id,
        "session_id": node.session_id,
        "timestamp": node.timestamp.isoformat(),
        "user_text": node.user_text,
        "agent_text": node.agent_text,
        "metadata": node.metadata
    }

    return query, params


def create_concept_node_cypher(node: ConceptNode) -> tuple[str, Dict[str, Any]]:
    """Generate Cypher query to create/merge concept node"""
    query = """
    MERGE (c:Concept {name: $name})
    ON CREATE SET
        c.frequency = 1,
        c.first_seen = datetime(),
        c.created_at = datetime()
    ON MATCH SET
        c.frequency = c.frequency + 1,
        c.last_seen = datetime(),
        c.updated_at = datetime()
    SET
        c.description = COALESCE($description, c.description),
        c.category = COALESCE($category, c.category),
        c += $metadata
    RETURN c
    """

    params = {
        "name": node.name,
        "description": node.description,
        "category": node.category,
        "metadata": node.metadata
    }

    return query, params


def create_relationship_cypher(rel: Relationship) -> tuple[str, Dict[str, Any]]:
    """Generate Cypher query to create/merge relationship"""
    query = f"""
    MATCH (a {{id: $from_node}})
    MATCH (b {{id: $to_node}})
    MERGE (a)-[r:{rel.rel_type.value}]->(b)
    ON CREATE SET
        r.strength = $strength,
        r.first_seen = datetime(),
        r.observation_count = 1,
        r.created_at = datetime()
    ON MATCH SET
        r.strength = r.strength + ($strength * 0.1),
        r.last_seen = datetime(),
        r.observation_count = r.observation_count + 1,
        r.updated_at = datetime()
    SET
        r.context = $context,
        r += $metadata
    RETURN r
    """

    params = {
        "from_node": rel.from_node,
        "to_node": rel.to_node,
        "strength": rel.strength,
        "context": rel.context,
        "metadata": rel.metadata
    }

    return query, params
