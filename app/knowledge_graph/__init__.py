"""
Knowledge Graph Module - Neo4j-powered concept relationships

SPECIFICATION:
- Input: Conversation text and exchanges
- Output: Graph of concepts, entities, and relationships
- Constraints: Real-time updates, efficient queries
- Intelligence: Relationship strength, learning paths

ARCHITECTURE:
- KnowledgeGraphStore: Neo4j connection and CRUD
- ConceptExtractor: NLP-powered concept extraction
- GraphQueryEngine: Cypher query builder and executor

INTEGRATION:
- AnalysisAgent provides entity/concept extraction
- Database provides conversation history
- API endpoints for graph visualization

Example:
    from app.knowledge_graph import graph_store, concept_extractor, query_engine

    # Extract and store concepts
    concepts = await concept_extractor.extract_from_text("Learning about neural networks")
    await graph_store.add_concept("neural networks", "AI model architecture")

    # Query relationships
    related = await query_engine.find_related_concepts("neural networks", max_depth=2)
    path = await query_engine.get_learning_path("basics", "advanced topic")
"""

from app.knowledge_graph.config import KnowledgeGraphConfig
from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.knowledge_graph.concept_extractor import ConceptExtractor
from app.knowledge_graph.query_engine import GraphQueryEngine

# Global instances (initialized on app startup)
config = KnowledgeGraphConfig()
graph_store = KnowledgeGraphStore(config)
concept_extractor = ConceptExtractor(graph_store)
query_engine = GraphQueryEngine(graph_store)

__all__ = [
    "KnowledgeGraphConfig",
    "KnowledgeGraphStore",
    "ConceptExtractor",
    "GraphQueryEngine",
    "config",
    "graph_store",
    "concept_extractor",
    "query_engine",
]
