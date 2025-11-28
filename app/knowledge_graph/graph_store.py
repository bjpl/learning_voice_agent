"""
Knowledge Graph Store - Neo4j Integration

SPECIFICATION:
- Input: Concepts, entities, relationships
- Output: Graph nodes and relationships
- Constraints: < 200ms write, < 500ms read
- Intelligence: Relationship strength, temporal tracking

PSEUDOCODE:
1. Connect to Neo4j (embedded or server)
2. Create/merge nodes with properties
3. Establish relationships with metadata
4. Query graph using Cypher
5. Maintain indexes for performance

ARCHITECTURE:
- Async Neo4j driver for non-blocking I/O
- Connection pooling for efficiency
- Transaction management for consistency
- Automatic index creation

REFINEMENT:
- Batch writes for multiple concepts
- Incremental strength updates
- Temporal relationship tracking
- Automatic schema management
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, TransientError

from app.knowledge_graph.config import KnowledgeGraphConfig
from app.logger import get_logger
from app.resilience import with_retry

logger = get_logger(__name__)


class KnowledgeGraphStore:
    """
    Neo4j-backed knowledge graph store

    PATTERN: Repository pattern with async operations
    WHY: Separation of concerns, testability, performance

    Node Types:
    - Concept: Core ideas and topics
    - Entity: Named entities (people, places, organizations)
    - Topic: High-level categories
    - Session: Conversation sessions

    Relationship Types:
    - RELATES_TO: Concept-to-concept relationships
    - MENTIONED_IN: Concept/entity mentioned in session
    - INSTANCE_OF: Entity is instance of concept
    - CONTAINS: Topic contains concepts
    - BUILDS_ON: Concept builds on another (prerequisite)

    Example:
        store = KnowledgeGraphStore(config)
        await store.initialize()

        # Add concept
        concept_id = await store.add_concept(
            name="neural networks",
            description="Artificial neural networks for ML",
            metadata={"category": "technology"}
        )

        # Create relationship
        await store.add_relationship(
            "neural networks", "machine learning",
            relationship_type="BUILDS_ON",
            strength=0.9
        )

        # Query
        related = await store.get_related_concepts("neural networks", max_depth=2)
    """

    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        self._initialized = False
        self.logger = logger

    @with_retry(max_attempts=3, min_wait=1.0)
    async def initialize(self):
        """
        Initialize Neo4j connection and schema

        PATTERN: Lazy initialization with retry
        WHY: Resilient startup, graceful degradation
        """
        if self._initialized:
            return

        try:
            self.logger.info(
                "knowledge_graph_initialization_started",
                uri=self.config.uri,
                embedded=self.config.embedded
            )

            # Create driver
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password) if self.config.password else None,
                **self.config.connection_config
            )

            # Verify connectivity
            await self.driver.verify_connectivity()

            # Create schema (indexes and constraints)
            await self._create_schema()

            self._initialized = True
            self.logger.info("knowledge_graph_initialized", status="ready")

        except ServiceUnavailable as e:
            self.logger.error(
                "neo4j_connection_failed",
                error=str(e),
                uri=self.config.uri
            )
            raise
        except Exception as e:
            self.logger.error(
                "knowledge_graph_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def close(self):
        """Close Neo4j driver"""
        if self.driver:
            await self.driver.close()
            self._initialized = False
            self.logger.info("knowledge_graph_closed")

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        """Context manager for Neo4j sessions"""
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call initialize() first.")

        async with self.driver.session(database=self.config.database) as session:
            yield session

    async def _create_schema(self):
        """
        Create indexes and constraints for performance

        PATTERN: Schema-first design
        WHY: Query performance, data integrity
        """
        schema_queries = [
            # Concept constraints and indexes
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX concept_frequency IF NOT EXISTS FOR (c:Concept) ON (c.frequency)",
            "CREATE INDEX concept_last_seen IF NOT EXISTS FOR (c:Concept) ON (c.last_seen)",

            # Entity constraints and indexes
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)",

            # Topic constraint
            "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",

            # Session constraint
            "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
            "CREATE INDEX session_timestamp IF NOT EXISTS FOR (s:Session) ON (s.timestamp)",
        ]

        async with self.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    # Some constraints may already exist
                    self.logger.debug("schema_creation_warning", query=query, error=str(e))

        self.logger.info("knowledge_graph_schema_created")

    @with_retry(max_attempts=2, min_wait=0.5)
    async def add_concept(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None
    ) -> str:
        """
        Add or update a concept node

        Args:
            name: Concept name (unique identifier)
            description: Optional description
            metadata: Additional properties
            topic: Optional parent topic

        Returns:
            Concept name (ID)

        PATTERN: MERGE for idempotent writes
        WHY: Upsert semantics, handles duplicates gracefully
        """
        try:
            async with self.session() as session:
                result = await session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET
                        c.first_seen = datetime(),
                        c.frequency = 1,
                        c.created_at = datetime()
                    ON MATCH SET
                        c.last_seen = datetime(),
                        c.frequency = c.frequency + 1,
                        c.updated_at = datetime()
                    SET
                        c.description = COALESCE($description, c.description),
                        c.metadata = $metadata
                    RETURN c.name as name, c.frequency as frequency
                    """,
                    name=name,
                    description=description,
                    metadata=metadata or {}
                )

                record = await result.single()

                # Link to topic if provided
                if topic and record:
                    await self._link_to_topic(name, topic, session)

                self.logger.info(
                    "concept_added",
                    concept=name,
                    frequency=record["frequency"] if record else 0,
                    has_topic=bool(topic)
                )

                return name

        except Exception as e:
            self.logger.error(
                "add_concept_failed",
                concept=name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _link_to_topic(self, concept_name: str, topic_name: str, session: AsyncSession):
        """Link concept to topic"""
        await session.run(
            """
            MERGE (t:Topic {name: $topic_name})
            ON CREATE SET t.created_at = datetime()
            WITH t
            MATCH (c:Concept {name: $concept_name})
            MERGE (t)-[:CONTAINS]->(c)
            """,
            concept_name=concept_name,
            topic_name=topic_name
        )

    @with_retry(max_attempts=2, min_wait=0.5)
    async def add_entity(
        self,
        text: str,
        entity_type: str,
        concept: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add entity node and optionally link to concept

        Args:
            text: Entity text (e.g., "Albert Einstein")
            entity_type: Entity type (PERSON, ORG, GPE, etc.)
            concept: Optional concept this is an instance of
            metadata: Additional properties

        Returns:
            Entity ID
        """
        try:
            async with self.session() as session:
                result = await session.run(
                    """
                    MERGE (e:Entity {text: $text, type: $type})
                    ON CREATE SET
                        e.first_seen = datetime(),
                        e.created_at = datetime()
                    ON MATCH SET
                        e.last_seen = datetime(),
                        e.updated_at = datetime()
                    SET e.metadata = $metadata
                    RETURN elementId(e) as id
                    """,
                    text=text,
                    type=entity_type,
                    metadata=metadata or {}
                )

                record = await result.single()
                entity_id = record["id"]

                # Link to concept if provided
                if concept:
                    await session.run(
                        """
                        MATCH (e:Entity {text: $text, type: $type})
                        MERGE (c:Concept {name: $concept})
                        ON CREATE SET c.created_at = datetime(), c.frequency = 1
                        MERGE (e)-[:INSTANCE_OF]->(c)
                        """,
                        text=text,
                        type=entity_type,
                        concept=concept
                    )

                self.logger.info(
                    "entity_added",
                    text=text,
                    type=entity_type,
                    concept=concept
                )

                return entity_id

        except Exception as e:
            self.logger.error(
                "add_entity_failed",
                text=text,
                error=str(e)
            )
            raise

    @with_retry(max_attempts=2, min_wait=0.5)
    async def add_relationship(
        self,
        from_concept: str,
        to_concept: str,
        relationship_type: str = "RELATES_TO",
        strength: float = 1.0,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create or update relationship between concepts

        Args:
            from_concept: Source concept name
            to_concept: Target concept name
            relationship_type: Type of relationship (RELATES_TO, BUILDS_ON, etc.)
            strength: Relationship strength (0.0-1.0)
            context: Optional context where relationship observed
            metadata: Additional properties

        PATTERN: Incremental strength updates
        WHY: Relationship strength grows with repeated observations
        """
        try:
            async with self.session() as session:
                await session.run(
                    f"""
                    MERGE (c1:Concept {{name: $from_concept}})
                    ON CREATE SET c1.created_at = datetime(), c1.frequency = 1
                    MERGE (c2:Concept {{name: $to_concept}})
                    ON CREATE SET c2.created_at = datetime(), c2.frequency = 1
                    MERGE (c1)-[r:{relationship_type}]->(c2)
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
                        r.metadata = $metadata
                    """,
                    from_concept=from_concept,
                    to_concept=to_concept,
                    strength=strength,
                    context=context,
                    metadata=metadata or {}
                )

                self.logger.info(
                    "relationship_added",
                    from_concept=from_concept,
                    to_concept=to_concept,
                    type=relationship_type,
                    strength=strength
                )

        except Exception as e:
            self.logger.error(
                "add_relationship_failed",
                from_concept=from_concept,
                to_concept=to_concept,
                error=str(e)
            )
            raise

    async def add_session(
        self,
        session_id: str,
        concepts: List[str],
        entities: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add conversation session and link to concepts/entities

        Args:
            session_id: Session identifier
            concepts: List of concept names mentioned
            entities: List of (text, type) tuples
            metadata: Session metadata (exchange_count, duration, etc.)
        """
        try:
            async with self.session() as session:
                # Create session node
                await session.run(
                    """
                    MERGE (s:Session {id: $session_id})
                    ON CREATE SET
                        s.timestamp = datetime(),
                        s.exchange_count = 0
                    SET
                        s.exchange_count = s.exchange_count + 1,
                        s.metadata = $metadata,
                        s.last_updated = datetime()
                    """,
                    session_id=session_id,
                    metadata=metadata or {}
                )

                # Link concepts
                for concept in concepts:
                    await session.run(
                        """
                        MATCH (s:Session {id: $session_id})
                        MERGE (c:Concept {name: $concept})
                        ON CREATE SET c.created_at = datetime(), c.frequency = 1
                        MERGE (c)-[r:MENTIONED_IN]->(s)
                        ON CREATE SET r.timestamp = datetime(), r.count = 1
                        ON MATCH SET r.count = r.count + 1, r.last_mention = datetime()
                        """,
                        session_id=session_id,
                        concept=concept
                    )

                # Link entities
                for text, entity_type in entities:
                    await session.run(
                        """
                        MATCH (s:Session {id: $session_id})
                        MERGE (e:Entity {text: $text, type: $type})
                        ON CREATE SET e.created_at = datetime()
                        MERGE (e)-[r:MENTIONED_IN]->(s)
                        ON CREATE SET r.timestamp = datetime(), r.count = 1
                        ON MATCH SET r.count = r.count + 1
                        """,
                        session_id=session_id,
                        text=text,
                        type=entity_type
                    )

                self.logger.info(
                    "session_added",
                    session_id=session_id,
                    concepts_count=len(concepts),
                    entities_count=len(entities)
                )

        except Exception as e:
            self.logger.error(
                "add_session_failed",
                session_id=session_id,
                error=str(e)
            )
            raise

    async def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept by name"""
        try:
            async with self.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Concept {name: $name})
                    RETURN c.name as name, c.description as description,
                           c.frequency as frequency, c.first_seen as first_seen,
                           c.last_seen as last_seen, c.metadata as metadata
                    """,
                    name=name
                )

                record = await result.single()
                if record:
                    return dict(record)
                return None

        except Exception as e:
            self.logger.error("get_concept_failed", concept=name, error=str(e))
            return None

    async def get_related_concepts(
        self,
        concept: str,
        max_depth: int = 2,
        min_strength: float = 0.3,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get concepts related to given concept

        Args:
            concept: Starting concept name
            max_depth: Maximum relationship depth
            min_strength: Minimum relationship strength
            limit: Maximum results

        Returns:
            List of {name, relationship_type, strength, distance}
        """
        try:
            async with self.session() as session:
                result = await session.run(
                    """
                    MATCH path = (c:Concept {name: $name})-[r*1..$max_depth]-(related:Concept)
                    WHERE ALL(rel IN relationships(path) WHERE rel.strength >= $min_strength)
                    WITH related,
                         [rel IN relationships(path) | type(rel)] as rel_types,
                         [rel IN relationships(path) | rel.strength] as strengths,
                         length(path) as distance
                    RETURN DISTINCT
                        related.name as name,
                        related.description as description,
                        related.frequency as frequency,
                        rel_types,
                        strengths,
                        distance
                    ORDER BY distance ASC, related.frequency DESC
                    LIMIT $limit
                    """,
                    name=concept,
                    max_depth=max_depth,
                    min_strength=min_strength,
                    limit=limit
                )

                records = await result.values()
                return [
                    {
                        "name": record[0],
                        "description": record[1],
                        "frequency": record[2],
                        "relationship_types": record[3],
                        "strengths": record[4],
                        "distance": record[5]
                    }
                    for record in records
                ]

        except Exception as e:
            self.logger.error(
                "get_related_concepts_failed",
                concept=concept,
                error=str(e)
            )
            return []

    async def get_most_discussed_concepts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently discussed concepts"""
        try:
            async with self.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Concept)
                    RETURN c.name as name, c.description as description,
                           c.frequency as frequency, c.last_seen as last_seen
                    ORDER BY c.frequency DESC
                    LIMIT $limit
                    """,
                    limit=limit
                )

                records = await result.values()
                return [
                    {
                        "name": record[0],
                        "description": record[1],
                        "frequency": record[2],
                        "last_seen": record[3]
                    }
                    for record in records
                ]

        except Exception as e:
            self.logger.error("get_most_discussed_failed", error=str(e))
            return []

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        try:
            async with self.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Concept)
                    OPTIONAL MATCH (c)-[r]->()
                    WITH count(DISTINCT c) as concept_count, count(r) as relationship_count
                    MATCH (e:Entity)
                    WITH concept_count, relationship_count, count(e) as entity_count
                    MATCH (s:Session)
                    WITH concept_count, relationship_count, entity_count, count(s) as session_count
                    MATCH (t:Topic)
                    RETURN concept_count, relationship_count, entity_count, session_count, count(t) as topic_count
                    """
                )

                record = await result.single()
                if record:
                    return {
                        "concepts": record["concept_count"],
                        "relationships": record["relationship_count"],
                        "entities": record["entity_count"],
                        "sessions": record["session_count"],
                        "topics": record["topic_count"]
                    }

                return {
                    "concepts": 0,
                    "relationships": 0,
                    "entities": 0,
                    "sessions": 0,
                    "topics": 0
                }

        except Exception as e:
            self.logger.error("get_graph_stats_failed", error=str(e))
            return {}
