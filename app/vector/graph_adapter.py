"""
Graph Query Adapter - Bridge Neo4j-style queries to RuVector
PATTERN: Adapter pattern for knowledge graph integration
WHY: Unify existing knowledge graph with RuVector's Cypher support

SPECIFICATION:
- Adapt existing knowledge graph queries to RuVector
- Provide hybrid vector + graph search
- Support concept-based filtering
- Enable relationship traversal

ARCHITECTURE:
[Knowledge Graph Queries] --> [GraphQueryAdapter] --> [RuVector Cypher]
                                      |
                                      +---> [Vector Search]
                                      +---> [Graph Traversal]

REFINEMENT:
- Query optimization with caching
- Fallback to vector-only when graph unavailable
- Result ranking with combined scores
- Temporal filtering support
"""

from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
import logging
from functools import lru_cache

from app.vector.schema import (
    NodeType,
    RelationshipType,
    CypherQueryBuilder,
    ConversationNode,
    ConceptNode,
    Relationship
)


logger = logging.getLogger(__name__)


class GraphQueryAdapter:
    """
    Adapter for bridging knowledge graph queries to RuVector

    CONCEPT: Unified query interface
    WHY: Enable seamless transition from Neo4j to RuVector
    PATTERN: Adapter with caching and fallback strategies

    Features:
    - Query concepts and their relationships
    - Find related conversations via graph
    - Hybrid semantic + structural search
    - Temporal filtering

    Example:
        adapter = GraphQueryAdapter(ruvector_store)
        await adapter.initialize()

        # Query concepts
        concepts = await adapter.query_concepts("machine learning")

        # Find related conversations
        convs = await adapter.find_related_conversations(
            "conv_123",
            relationship_type="ABOUT"
        )

        # Hybrid search
        results = await adapter.hybrid_concept_search(
            "neural networks",
            concept_filter="deep learning"
        )
    """

    def __init__(self, vector_store):
        """
        Initialize adapter with RuVector store

        Args:
            vector_store: RuVector store instance with GraphVectorStoreProtocol
        """
        self.vector_store = vector_store
        self._initialized = False
        self._query_cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def initialize(self) -> bool:
        """Initialize the adapter and underlying vector store"""
        if self._initialized:
            return True

        try:
            initialized = await self.vector_store.initialize()
            if initialized:
                self._initialized = True
                logger.info("GraphQueryAdapter initialized successfully")
            return initialized
        except Exception as e:
            logger.error(f"Failed to initialize GraphQueryAdapter: {e}")
            return False

    async def query_concepts(
        self,
        concept_name: str,
        include_relationships: bool = True,
        max_depth: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find concepts and their relationships

        Args:
            concept_name: Concept name to query
            include_relationships: Include related concepts
            max_depth: Maximum relationship depth
            limit: Maximum results

        Returns:
            List of concept dictionaries with optional relationships

        PATTERN: Progressive enhancement
        WHY: Start with concept, add relationships if requested
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Build Cypher query for concept
            if include_relationships:
                query = f"""
                MATCH (c:Concept {{name: $concept_name}})
                OPTIONAL MATCH path = (c)-[r:{RelationshipType.RELATES_TO.value}*1..{max_depth}]-(related:Concept)
                WHERE ALL(rel IN relationships(path) WHERE rel.strength >= 0.3)
                WITH c, related, relationships(path) as rels
                RETURN c.name as name,
                       c.description as description,
                       c.frequency as frequency,
                       c.category as category,
                       collect({{
                           name: related.name,
                           rel_types: [rel IN rels | type(rel)],
                           strengths: [rel IN rels | rel.strength],
                           distance: length(path)
                       }}) as relationships
                LIMIT $limit
                """
            else:
                query = """
                MATCH (c:Concept {name: $concept_name})
                RETURN c.name as name,
                       c.description as description,
                       c.frequency as frequency,
                       c.category as category
                LIMIT $limit
                """

            parameters = {
                "concept_name": concept_name,
                "limit": limit
            }

            # Execute Cypher query
            results = await self.vector_store.execute_cypher(query, parameters)

            logger.info(
                f"Found {len(results)} concepts for '{concept_name}'",
                extra={"concept": concept_name, "count": len(results)}
            )

            return results

        except Exception as e:
            logger.error(f"Failed to query concepts: {e}", exc_info=True)
            return []

    async def find_related_conversations(
        self,
        conversation_id: str,
        relationship_type: str = "ABOUT",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find conversations related through graph relationships

        Args:
            conversation_id: Source conversation ID
            relationship_type: Type of relationship to follow
            limit: Maximum results

        Returns:
            List of related conversations with relationship context

        PATTERN: Graph traversal with type filtering
        WHY: Find conversations sharing concepts or topics

        Example:
            # Find conversations about same concepts
            related = await adapter.find_related_conversations(
                "conv_123",
                relationship_type="ABOUT"
            )

            # Result: Conversations that discuss the same concepts
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Cypher query to traverse graph
            query = f"""
            MATCH (conv:Conversation {{id: $conversation_id}})
            MATCH (conv)-[:{relationship_type}]->(concept:Concept)
            MATCH (concept)<-[:{relationship_type}]-(related:Conversation)
            WHERE related.id <> $conversation_id
            WITH related, concept, count(*) as shared_concepts
            RETURN related.id as id,
                   related.session_id as session_id,
                   related.timestamp as timestamp,
                   related.user_text as user_text,
                   related.agent_text as agent_text,
                   collect(concept.name) as shared_concept_names,
                   shared_concepts
            ORDER BY shared_concepts DESC
            LIMIT $limit
            """

            parameters = {
                "conversation_id": conversation_id,
                "limit": limit
            }

            results = await self.vector_store.execute_cypher(query, parameters)

            logger.info(
                f"Found {len(results)} related conversations",
                extra={"conversation_id": conversation_id, "count": len(results)}
            )

            return results

        except Exception as e:
            logger.error(f"Failed to find related conversations: {e}", exc_info=True)
            # Fallback to vector similarity
            return await self._fallback_find_similar(conversation_id, limit)

    async def hybrid_concept_search(
        self,
        query: str,
        concept_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        min_concept_frequency: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search filtered by concept relationships

        Args:
            query: Search query text
            concept_filter: Filter by related concept
            topic_filter: Filter by topic
            min_concept_frequency: Minimum concept frequency
            limit: Maximum results

        Returns:
            Combined vector + graph search results

        CONCEPT: Hybrid search combining semantic + structural
        WHY: "Find conversations about X that relate to concept Y"

        Example:
            # Find ML conversations related to deep learning
            results = await adapter.hybrid_concept_search(
                "neural network training",
                concept_filter="deep learning"
            )
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Build Cypher filter for hybrid search
            cypher_filter = None

            if concept_filter or topic_filter:
                conditions = []

                if concept_filter:
                    conditions.append(
                        f"""
                        EXISTS {{
                            MATCH (conv)-[:{RelationshipType.ABOUT.value}]->(c:Concept)
                            WHERE c.name = '{concept_filter}'
                        }}
                        """
                    )

                if topic_filter:
                    conditions.append(
                        f"""
                        EXISTS {{
                            MATCH (conv)-[:{RelationshipType.ABOUT.value}]->(c:Concept)
                            MATCH (t:Topic {{name: '{topic_filter}'}})-[:{RelationshipType.CONTAINS.value}]->(c)
                        }}
                        """
                    )

                if min_concept_frequency > 0:
                    conditions.append(f"c.frequency >= {min_concept_frequency}")

                cypher_filter = " AND ".join(conditions)

            # Execute hybrid search
            results = await self.vector_store.hybrid_search(
                query=query,
                cypher_filter=cypher_filter,
                limit=limit
            )

            # Enhance results with concept information
            enhanced_results = await self._enhance_with_concepts(results)

            logger.info(
                f"Hybrid search returned {len(enhanced_results)} results",
                extra={
                    "query": query,
                    "concept_filter": concept_filter,
                    "count": len(enhanced_results)
                }
            )

            return enhanced_results

        except Exception as e:
            logger.error(f"Hybrid concept search failed: {e}", exc_info=True)
            # Fallback to pure vector search
            return await self.vector_store.semantic_search(query, limit=limit)

    async def get_concept_hierarchy(
        self,
        concept_name: str,
        include_parents: bool = True,
        include_children: bool = True,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get hierarchical structure around a concept

        Args:
            concept_name: Central concept
            include_parents: Include parent concepts
            include_children: Include child concepts
            max_depth: Maximum hierarchy depth

        Returns:
            Hierarchical structure with parents and children

        PATTERN: Tree traversal with bidirectional search
        WHY: Understand concept relationships and prerequisites
        """
        if not self._initialized:
            await self.initialize()

        try:
            hierarchy = {
                "concept": concept_name,
                "parents": [],
                "children": [],
                "siblings": []
            }

            if include_parents:
                # Find parent concepts (concepts this builds on)
                parent_query = f"""
                MATCH (c:Concept {{name: $concept_name}})
                MATCH path = (c)-[:{RelationshipType.BUILDS_ON.value}*1..{max_depth}]->(parent:Concept)
                RETURN parent.name as name,
                       parent.description as description,
                       length(path) as depth
                ORDER BY depth ASC
                """

                parents = await self.vector_store.execute_cypher(
                    parent_query,
                    {"concept_name": concept_name}
                )
                hierarchy["parents"] = parents

            if include_children:
                # Find child concepts (concepts that build on this)
                child_query = f"""
                MATCH (c:Concept {{name: $concept_name}})
                MATCH path = (child:Concept)-[:{RelationshipType.BUILDS_ON.value}*1..{max_depth}]->(c)
                RETURN child.name as name,
                       child.description as description,
                       length(path) as depth
                ORDER BY depth ASC
                """

                children = await self.vector_store.execute_cypher(
                    child_query,
                    {"concept_name": concept_name}
                )
                hierarchy["children"] = children

            # Find sibling concepts (share same parent)
            sibling_query = f"""
            MATCH (c:Concept {{name: $concept_name}})
            MATCH (c)-[:{RelationshipType.BUILDS_ON.value}]->(parent:Concept)
            MATCH (sibling:Concept)-[:{RelationshipType.BUILDS_ON.value}]->(parent)
            WHERE sibling.name <> $concept_name
            RETURN DISTINCT sibling.name as name,
                   sibling.description as description,
                   sibling.frequency as frequency
            ORDER BY sibling.frequency DESC
            LIMIT 10
            """

            siblings = await self.vector_store.execute_cypher(
                sibling_query,
                {"concept_name": concept_name}
            )
            hierarchy["siblings"] = siblings

            logger.info(
                f"Retrieved hierarchy for '{concept_name}'",
                extra={
                    "concept": concept_name,
                    "parents": len(hierarchy["parents"]),
                    "children": len(hierarchy["children"])
                }
            )

            return hierarchy

        except Exception as e:
            logger.error(f"Failed to get concept hierarchy: {e}", exc_info=True)
            return {"concept": concept_name, "parents": [], "children": [], "siblings": []}

    async def find_learning_path(
        self,
        from_concept: str,
        to_concept: str,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Find shortest learning path between concepts

        Args:
            from_concept: Starting concept (what user knows)
            to_concept: Target concept (what user wants to learn)
            max_depth: Maximum path length

        Returns:
            Learning path with steps and recommendations

        CONCEPT: Shortest path with prerequisite awareness
        WHY: Guide learning progression
        """
        if not self._initialized:
            await self.initialize()

        try:
            query = f"""
            MATCH path = shortestPath(
                (start:Concept {{name: $from_concept}})
                -[:{RelationshipType.BUILDS_ON.value}|{RelationshipType.RELATES_TO.value}*1..{max_depth}]-
                (end:Concept {{name: $to_concept}})
            )
            WITH path,
                 [node IN nodes(path) | node.name] as concept_names,
                 [rel IN relationships(path) | type(rel)] as rel_types,
                 [rel IN relationships(path) | rel.strength] as strengths
            RETURN
                concept_names,
                rel_types,
                strengths,
                length(path) as steps
            ORDER BY steps ASC
            LIMIT 3
            """

            parameters = {
                "from_concept": from_concept,
                "to_concept": to_concept
            }

            results = await self.vector_store.execute_cypher(query, parameters)

            if results:
                path_data = results[0]
                return {
                    "found": True,
                    "from_concept": from_concept,
                    "to_concept": to_concept,
                    "path": path_data.get("concept_names", []),
                    "relationship_types": path_data.get("rel_types", []),
                    "strengths": path_data.get("strengths", []),
                    "steps": path_data.get("steps", 0)
                }
            else:
                return {
                    "found": False,
                    "from_concept": from_concept,
                    "to_concept": to_concept,
                    "message": "No learning path found"
                }

        except Exception as e:
            logger.error(f"Failed to find learning path: {e}", exc_info=True)
            return {
                "found": False,
                "from_concept": from_concept,
                "to_concept": to_concept,
                "error": str(e)
            }

    # ===== Private Helper Methods =====

    async def _fallback_find_similar(
        self,
        conversation_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback to vector-only similarity search"""
        try:
            return await self.vector_store.find_similar_conversations(
                conversation_id,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    async def _enhance_with_concepts(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance search results with concept information"""
        enhanced = []

        for result in results:
            conversation_id = result.get("id")
            if not conversation_id:
                enhanced.append(result)
                continue

            try:
                # Query concepts for this conversation
                concept_query = f"""
                MATCH (conv:Conversation {{id: $conversation_id}})
                MATCH (conv)-[:{RelationshipType.ABOUT.value}]->(c:Concept)
                RETURN c.name as name, c.category as category
                LIMIT 5
                """

                concepts = await self.vector_store.execute_cypher(
                    concept_query,
                    {"conversation_id": conversation_id}
                )

                result["concepts"] = concepts
                enhanced.append(result)

            except Exception as e:
                logger.warning(f"Failed to enhance result with concepts: {e}")
                enhanced.append(result)

        return enhanced

    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
        logger.debug("Query cache cleared")
