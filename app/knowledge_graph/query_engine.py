"""
Graph Query Engine - Advanced knowledge graph queries

SPECIFICATION:
- Input: Query parameters (concept, depth, filters)
- Output: Related concepts, learning paths, knowledge gaps
- Constraints: < 500ms query time
- Intelligence: Path finding, gap detection, recommendations

PSEUDOCODE:
1. Build Cypher queries dynamically
2. Execute against Neo4j graph
3. Process and rank results
4. Generate learning paths using graph algorithms
5. Detect knowledge gaps by analyzing connections

ARCHITECTURE:
- Cypher query builder for flexibility
- Graph algorithms (shortest path, centrality)
- Result ranking and filtering
- Learning path optimization

REFINEMENT:
- Query caching for common patterns
- Multi-criteria ranking (frequency, recency, strength)
- Concept map generation
- Knowledge gap recommendations
"""

from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.logger import get_logger

logger = get_logger(__name__)


class GraphQueryEngine:
    """
    Advanced query engine for knowledge graph

    PATTERN: Query builder with result processing
    WHY: Flexible querying, optimized performance

    Features:
    - Find related concepts with ranking
    - Discover learning paths
    - Identify knowledge gaps
    - Generate concept maps
    - Temporal analysis (trending topics)

    Example:
        engine = GraphQueryEngine(graph_store)

        # Find related concepts
        related = await engine.find_related_concepts(
            "machine learning",
            max_depth=2,
            limit=10
        )

        # Get learning path
        path = await engine.get_learning_path(
            from_concept="programming basics",
            to_concept="machine learning"
        )

        # Identify knowledge gaps
        gaps = await engine.identify_knowledge_gaps(
            known_concepts=["python", "statistics"],
            target_concept="deep learning"
        )
    """

    def __init__(self, graph_store: KnowledgeGraphStore):
        self.graph_store = graph_store
        self.logger = logger

        # Query settings
        self.default_max_depth = 3
        self.default_limit = 20
        self.min_relationship_strength = 0.3

    async def find_related_concepts(
        self,
        concept: str,
        max_depth: int = 2,
        min_strength: float = None,
        limit: int = None,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find concepts related to given concept

        Args:
            concept: Starting concept name
            max_depth: Maximum relationship depth (1-3)
            min_strength: Minimum relationship strength filter
            limit: Maximum results
            relationship_types: Filter by relationship types

        Returns:
            List of related concepts with metadata and ranking

        PATTERN: Graph traversal with filtering
        WHY: Discover related knowledge efficiently
        """
        min_strength = min_strength or self.min_relationship_strength
        limit = limit or self.default_limit

        try:
            self.logger.info(
                "finding_related_concepts",
                concept=concept,
                max_depth=max_depth,
                min_strength=min_strength
            )

            # Get related concepts from graph store
            related = await self.graph_store.get_related_concepts(
                concept=concept,
                max_depth=max_depth,
                min_strength=min_strength,
                limit=limit * 2  # Get more for filtering
            )

            # Rank and filter results
            ranked_results = self._rank_concepts(related, concept)

            # Filter by relationship types if specified
            if relationship_types:
                ranked_results = [
                    r for r in ranked_results
                    if any(rt in r.get("relationship_types", []) for rt in relationship_types)
                ]

            # Apply limit
            ranked_results = ranked_results[:limit]

            self.logger.info(
                "related_concepts_found",
                concept=concept,
                count=len(ranked_results)
            )

            return ranked_results

        except Exception as e:
            self.logger.error(
                "find_related_concepts_failed",
                concept=concept,
                error=str(e),
                exc_info=True
            )
            return []

    def _rank_concepts(
        self,
        concepts: List[Dict[str, Any]],
        origin_concept: str
    ) -> List[Dict[str, Any]]:
        """
        Rank concepts by relevance using multiple factors

        Factors:
        - Distance (closer = higher rank)
        - Frequency (more discussed = higher rank)
        - Relationship strength
        - Recency (recently seen = higher rank)
        """
        ranked = []

        for concept in concepts:
            score = 0.0

            # Distance factor (inverse, normalized)
            distance = concept.get("distance", 1)
            distance_score = 1.0 / distance
            score += distance_score * 0.4

            # Frequency factor
            frequency = concept.get("frequency", 1)
            frequency_score = min(frequency / 10.0, 1.0)
            score += frequency_score * 0.3

            # Relationship strength factor
            strengths = concept.get("strengths", [0.5])
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.5
            score += avg_strength * 0.3

            concept["relevance_score"] = round(score, 3)
            ranked.append(concept)

        # Sort by score descending
        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)

        return ranked

    async def get_learning_path(
        self,
        from_concept: str,
        to_concept: str,
        max_depth: int = None
    ) -> Dict[str, Any]:
        """
        Find learning path between two concepts

        Uses shortest path algorithm with relationship strength weighting

        Args:
            from_concept: Starting concept (what user knows)
            to_concept: Target concept (what user wants to learn)
            max_depth: Maximum path length

        Returns:
            Dictionary with path, steps, and recommendations

        PATTERN: Shortest path with weighted edges
        WHY: Optimal learning progression
        """
        max_depth = max_depth or self.default_max_depth

        try:
            self.logger.info(
                "finding_learning_path",
                from_concept=from_concept,
                to_concept=to_concept,
                max_depth=max_depth
            )

            async with self.graph_store.session() as session:
                # Find shortest path weighted by inverse strength (higher strength = lower cost)
                result = await session.run(
                    """
                    MATCH path = shortestPath(
                        (start:Concept {name: $from_concept})-[r*1..$max_depth]-(end:Concept {name: $to_concept})
                    )
                    WHERE ALL(rel IN relationships(path) WHERE rel.strength >= $min_strength)
                    WITH path,
                         [rel IN relationships(path) | rel.strength] as strengths,
                         [node IN nodes(path) | node.name] as concept_names,
                         [rel IN relationships(path) | type(rel)] as rel_types,
                         reduce(cost = 0, rel IN relationships(path) | cost + (1.0 / rel.strength)) as total_cost
                    RETURN
                        concept_names,
                        rel_types,
                        strengths,
                        length(path) as steps,
                        total_cost
                    ORDER BY total_cost ASC
                    LIMIT 3
                    """,
                    from_concept=from_concept,
                    to_concept=to_concept,
                    max_depth=max_depth,
                    min_strength=self.min_relationship_strength
                )

                records = await result.values()

                if not records:
                    return {
                        "found": False,
                        "from_concept": from_concept,
                        "to_concept": to_concept,
                        "message": "No learning path found"
                    }

                # Get best path (lowest cost)
                best_path = records[0]

                path_result = {
                    "found": True,
                    "from_concept": from_concept,
                    "to_concept": to_concept,
                    "path": best_path[0],  # concept_names
                    "relationship_types": best_path[1],  # rel_types
                    "strengths": best_path[2],
                    "steps": best_path[3],
                    "total_cost": round(best_path[4], 2),
                    "difficulty": self._estimate_difficulty(best_path[3], best_path[2]),
                    "estimated_time": self._estimate_learning_time(best_path[3]),
                    "alternative_paths": len(records) - 1
                }

                # Get additional context for each step
                path_result["detailed_steps"] = await self._get_path_details(
                    path_result["path"]
                )

                self.logger.info(
                    "learning_path_found",
                    from_concept=from_concept,
                    to_concept=to_concept,
                    steps=path_result["steps"]
                )

                return path_result

        except Exception as e:
            self.logger.error(
                "get_learning_path_failed",
                from_concept=from_concept,
                to_concept=to_concept,
                error=str(e),
                exc_info=True
            )
            return {
                "found": False,
                "from_concept": from_concept,
                "to_concept": to_concept,
                "error": str(e)
            }

    def _estimate_difficulty(self, steps: int, strengths: List[float]) -> str:
        """Estimate learning difficulty"""
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.5

        if steps <= 1 and avg_strength >= 0.7:
            return "easy"
        elif steps <= 2 and avg_strength >= 0.5:
            return "moderate"
        elif steps <= 3:
            return "challenging"
        else:
            return "advanced"

    def _estimate_learning_time(self, steps: int) -> str:
        """Estimate learning time in human-readable format"""
        # Rough estimate: 30 minutes per step
        minutes = steps * 30

        if minutes < 60:
            return f"{minutes} minutes"
        elif minutes < 120:
            return "1-2 hours"
        elif minutes < 240:
            return "2-4 hours"
        else:
            hours = minutes // 60
            return f"{hours}+ hours"

    async def _get_path_details(self, path: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information for each concept in path"""
        details = []

        for i, concept_name in enumerate(path):
            concept = await self.graph_store.get_concept(concept_name)

            if concept:
                details.append({
                    "step": i + 1,
                    "concept": concept_name,
                    "description": concept.get("description"),
                    "frequency": concept.get("frequency", 0),
                    "is_start": i == 0,
                    "is_end": i == len(path) - 1
                })

        return details

    async def identify_knowledge_gaps(
        self,
        known_concepts: List[str],
        target_concept: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Identify knowledge gaps between what user knows and target concept

        Args:
            known_concepts: List of concepts user already knows
            target_concept: Concept user wants to learn
            max_depth: Maximum search depth

        Returns:
            Dictionary with missing concepts, recommendations, and priority

        PATTERN: Graph analysis for gap detection
        WHY: Personalized learning recommendations
        """
        try:
            self.logger.info(
                "identifying_knowledge_gaps",
                known_count=len(known_concepts),
                target=target_concept
            )

            # Find all concepts on paths from known concepts to target
            all_path_concepts = set()
            gap_details = []

            for known in known_concepts:
                path_result = await self.get_learning_path(
                    from_concept=known,
                    to_concept=target_concept,
                    max_depth=max_depth
                )

                if path_result.get("found"):
                    path = path_result.get("path", [])
                    all_path_concepts.update(path)

                    gap_details.append({
                        "from_concept": known,
                        "path": path,
                        "steps": path_result.get("steps", 0),
                        "difficulty": path_result.get("difficulty", "unknown")
                    })

            # Identify gaps (concepts in paths but not in known_concepts)
            known_set = set(known_concepts)
            gaps = all_path_concepts - known_set - {target_concept}

            # Rank gaps by importance
            ranked_gaps = await self._rank_knowledge_gaps(
                gaps=list(gaps),
                known_concepts=known_concepts,
                target_concept=target_concept
            )

            result = {
                "target_concept": target_concept,
                "known_concepts": known_concepts,
                "gaps": ranked_gaps,
                "gap_count": len(ranked_gaps),
                "paths_analyzed": len(gap_details),
                "recommendations": self._generate_gap_recommendations(
                    ranked_gaps, gap_details
                ),
                "analyzed_at": datetime.utcnow().isoformat()
            }

            self.logger.info(
                "knowledge_gaps_identified",
                target=target_concept,
                gap_count=len(ranked_gaps)
            )

            return result

        except Exception as e:
            self.logger.error(
                "identify_knowledge_gaps_failed",
                target=target_concept,
                error=str(e),
                exc_info=True
            )
            return {
                "target_concept": target_concept,
                "known_concepts": known_concepts,
                "gaps": [],
                "error": str(e)
            }

    async def _rank_knowledge_gaps(
        self,
        gaps: List[str],
        known_concepts: List[str],
        target_concept: str
    ) -> List[Dict[str, Any]]:
        """Rank knowledge gaps by priority"""
        ranked_gaps = []

        for gap in gaps:
            # Get concept details
            concept = await self.graph_store.get_concept(gap)

            if not concept:
                continue

            # Calculate priority score
            priority_score = 0.0

            # Factor 1: Frequency (more discussed = more important)
            frequency = concept.get("frequency", 1)
            frequency_score = min(frequency / 10.0, 1.0)
            priority_score += frequency_score * 0.4

            # Factor 2: Connections to known concepts
            related = await self.graph_store.get_related_concepts(
                concept=gap,
                max_depth=1,
                limit=100
            )
            known_connections = sum(
                1 for r in related
                if r.get("name") in known_concepts
            )
            connection_score = min(known_connections / 5.0, 1.0)
            priority_score += connection_score * 0.3

            # Factor 3: Distance to target
            target_connections = sum(
                1 for r in related
                if r.get("name") == target_concept
            )
            target_score = min(target_connections, 1.0)
            priority_score += target_score * 0.3

            ranked_gaps.append({
                "concept": gap,
                "description": concept.get("description"),
                "frequency": frequency,
                "priority_score": round(priority_score, 3),
                "known_connections": known_connections,
                "target_connections": target_connections
            })

        # Sort by priority descending
        ranked_gaps.sort(key=lambda x: x["priority_score"], reverse=True)

        return ranked_gaps

    def _generate_gap_recommendations(
        self,
        gaps: List[Dict[str, Any]],
        path_details: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate learning recommendations based on gaps"""
        recommendations = []

        if not gaps:
            recommendations.append("You have all prerequisite knowledge!")
            return recommendations

        # Recommend top priority gaps
        top_gaps = gaps[:3]

        if len(top_gaps) == 1:
            recommendations.append(
                f"Focus on learning: {top_gaps[0]['concept']}"
            )
        else:
            gap_list = ", ".join([g["concept"] for g in top_gaps])
            recommendations.append(
                f"Start by learning these concepts: {gap_list}"
            )

        # Recommend easiest path
        if path_details:
            easiest_path = min(path_details, key=lambda x: x.get("steps", 99))
            recommendations.append(
                f"Easiest learning path starts from: {easiest_path['from_concept']}"
            )

        # Recommend based on existing knowledge
        high_connection_gaps = [
            g for g in gaps
            if g.get("known_connections", 0) >= 2
        ]

        if high_connection_gaps:
            recommendations.append(
                f"You're close to understanding: {high_connection_gaps[0]['concept']}"
            )

        return recommendations

    async def get_concept_map(
        self,
        central_concept: str,
        radius: int = 2,
        max_concepts: int = 15
    ) -> Dict[str, Any]:
        """
        Generate concept map around central concept

        Args:
            central_concept: Center of the map
            radius: How far to expand (1-3)
            max_concepts: Maximum concepts to include

        Returns:
            Dictionary with nodes and edges for visualization
        """
        try:
            self.logger.info(
                "generating_concept_map",
                central_concept=central_concept,
                radius=radius
            )

            # Get related concepts
            related = await self.find_related_concepts(
                concept=central_concept,
                max_depth=radius,
                limit=max_concepts
            )

            # Build nodes
            nodes = [
                {
                    "id": central_concept,
                    "label": central_concept,
                    "type": "central",
                    "frequency": 0  # Will be updated
                }
            ]

            for concept in related:
                nodes.append({
                    "id": concept["name"],
                    "label": concept["name"],
                    "type": "related",
                    "frequency": concept.get("frequency", 0),
                    "distance": concept.get("distance", 1)
                })

            # Build edges
            edges = []

            for concept in related:
                rel_types = concept.get("relationship_types", [])
                strengths = concept.get("strengths", [])

                for rel_type, strength in zip(rel_types, strengths):
                    edges.append({
                        "from": central_concept,
                        "to": concept["name"],
                        "type": rel_type,
                        "strength": strength,
                        "weight": strength
                    })

            concept_map = {
                "central_concept": central_concept,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": datetime.utcnow().isoformat()
            }

            self.logger.info(
                "concept_map_generated",
                central_concept=central_concept,
                nodes=len(nodes),
                edges=len(edges)
            )

            return concept_map

        except Exception as e:
            self.logger.error(
                "get_concept_map_failed",
                central_concept=central_concept,
                error=str(e),
                exc_info=True
            )
            return {
                "central_concept": central_concept,
                "nodes": [],
                "edges": [],
                "error": str(e)
            }

    async def get_trending_concepts(
        self,
        time_window_hours: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending concepts within time window

        Args:
            time_window_hours: Time window in hours
            limit: Maximum results

        Returns:
            List of trending concepts with metrics
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

            async with self.graph_store.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Concept)
                    WHERE c.last_seen >= datetime($cutoff_time)
                    RETURN c.name as name,
                           c.description as description,
                           c.frequency as frequency,
                           c.last_seen as last_seen
                    ORDER BY c.frequency DESC, c.last_seen DESC
                    LIMIT $limit
                    """,
                    cutoff_time=cutoff_time.isoformat(),
                    limit=limit
                )

                records = await result.values()

                trending = [
                    {
                        "concept": record[0],
                        "description": record[1],
                        "frequency": record[2],
                        "last_seen": record[3],
                        "trend": "rising"  # Could be calculated with historical data
                    }
                    for record in records
                ]

                self.logger.info(
                    "trending_concepts_retrieved",
                    count=len(trending),
                    time_window=time_window_hours
                )

                return trending

        except Exception as e:
            self.logger.error(
                "get_trending_concepts_failed",
                error=str(e),
                exc_info=True
            )
            return []
