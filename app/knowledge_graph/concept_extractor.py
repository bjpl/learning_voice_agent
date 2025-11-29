"""
Concept Extractor - Extract and build knowledge graph from conversations

SPECIFICATION:
- Input: Conversation text, exchanges, analysis results
- Output: Concepts, entities, relationships in Neo4j graph
- Constraints: < 500ms extraction, high accuracy
- Intelligence: Co-occurrence detection, relationship strength

PSEUDOCODE:
1. Receive conversation text/exchanges
2. Use AnalysisAgent for entity/concept extraction
3. Detect relationships using co-occurrence patterns
4. Calculate relationship strength using multiple signals
5. Store in graph with temporal tracking
6. Build incrementally as conversations occur

ARCHITECTURE:
- Integration with AnalysisAgent for NLP
- Async graph updates via KnowledgeGraphStore
- Batch operations for efficiency
- Relationship detection using proximity and frequency

REFINEMENT:
- Multi-signal relationship strength (co-occurrence, proximity, frequency)
- Topic hierarchy detection
- Prerequisite relationship inference
- Temporal concept tracking
- Deduplication and merging
"""

from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import asyncio
import re

from app.agents.analysis_agent import AnalysisAgent, AgentMessage, MessageType
from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.logger import get_logger

logger = get_logger(__name__)


class ConceptExtractor:
    """
    Extract concepts and relationships from conversations for knowledge graph

    PATTERN: Pipeline architecture with multi-stage processing
    WHY: Separation of concerns, reusable components

    Features:
    - NLP-powered concept extraction via AnalysisAgent
    - Relationship detection using multiple signals
    - Incremental graph building
    - Automatic topic classification
    - Prerequisite detection for learning paths

    Example:
        extractor = ConceptExtractor(graph_store)

        # Extract from single text
        result = await extractor.extract_from_text(
            "I'm learning about neural networks and deep learning"
        )

        # Extract from conversation
        result = await extractor.extract_from_conversation(
            session_id="session_123",
            exchanges=[
                {"user": "What is machine learning?", "agent": "..."},
                {"user": "Tell me about neural networks", "agent": "..."}
            ]
        )

        # Get extraction stats
        stats = await extractor.get_extraction_stats()
    """

    def __init__(self, graph_store: KnowledgeGraphStore, analysis_agent: Optional[AnalysisAgent] = None):
        self.graph_store = graph_store
        self.analysis_agent = analysis_agent or AnalysisAgent()
        self.logger = logger

        # Relationship detection settings
        self.proximity_threshold = 100  # chars
        self.min_co_occurrence = 2  # minimum co-occurrences for relationship
        self.base_relationship_strength = 0.5

        # Topic keywords for classification
        self.topic_hierarchy = self._build_topic_hierarchy()

        # Prerequisite indicators (words suggesting one concept builds on another)
        self.prerequisite_indicators = {
            'requires', 'needs', 'builds on', 'based on', 'extends',
            'after', 'before', 'first', 'then', 'next', 'advanced',
            'beginner', 'intermediate', 'fundamental', 'basic'
        }

    def _build_topic_hierarchy(self) -> Dict[str, List[str]]:
        """Build topic hierarchy for classification"""
        return {
            "Computer Science": [
                "algorithms", "data structures", "programming", "software engineering",
                "databases", "operating systems", "networks", "security"
            ],
            "Artificial Intelligence": [
                "machine learning", "deep learning", "neural networks", "nlp",
                "computer vision", "reinforcement learning", "ai", "models"
            ],
            "Mathematics": [
                "linear algebra", "calculus", "statistics", "probability",
                "optimization", "discrete math", "geometry", "algebra"
            ],
            "Data Science": [
                "data analysis", "visualization", "statistics", "modeling",
                "analytics", "big data", "data mining", "pandas"
            ],
            "Web Development": [
                "html", "css", "javascript", "react", "vue", "angular",
                "frontend", "backend", "api", "rest", "graphql"
            ]
        }

    async def extract_from_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract concepts and relationships from single text

        Args:
            text: Input text to analyze
            session_id: Optional session identifier
            metadata: Additional metadata

        Returns:
            Dictionary with extracted concepts, entities, relationships, and stats

        PATTERN: Pipeline processing
        WHY: Clear stages, easy to test and debug
        """
        if not text or not text.strip():
            return self._empty_extraction_result()

        try:
            self.logger.info("extraction_started", text_length=len(text), has_session=bool(session_id))

            # Stage 1: NLP Analysis using AnalysisAgent
            analysis = await self._analyze_text(text)

            # Stage 2: Extract concepts and entities
            concepts = self._extract_concepts_from_analysis(analysis)
            entities = self._extract_entities_from_analysis(analysis)

            # Stage 3: Classify topics
            topics = self._classify_topics(text, concepts)

            # Stage 4: Detect relationships between concepts
            relationships = await self._detect_relationships(concepts, text, analysis)

            # Stage 5: Store in graph
            await self._store_in_graph(
                concepts=concepts,
                entities=entities,
                relationships=relationships,
                topics=topics,
                session_id=session_id,
                metadata=metadata
            )

            result = {
                "concepts": concepts,
                "entities": entities,
                "relationships": relationships,
                "topics": topics,
                "stats": {
                    "concept_count": len(concepts),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "topic_count": len(topics)
                },
                "extracted_at": datetime.utcnow().isoformat()
            }

            self.logger.info(
                "extraction_completed",
                concepts=len(concepts),
                entities=len(entities),
                relationships=len(relationships)
            )

            return result

        except Exception as e:
            self.logger.error(
                "extraction_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return self._empty_extraction_result()

    async def extract_from_conversation(
        self,
        session_id: str,
        exchanges: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract concepts from full conversation

        Args:
            session_id: Session identifier
            exchanges: List of conversation exchanges [{user: str, agent: str}]
            metadata: Session metadata

        Returns:
            Aggregated extraction results

        PATTERN: Incremental aggregation
        WHY: Handle conversations of any length, track evolution
        """
        if not exchanges:
            return self._empty_extraction_result()

        try:
            self.logger.info(
                "conversation_extraction_started",
                session_id=session_id,
                exchange_count=len(exchanges)
            )

            # Combine all user messages for analysis
            all_user_text = " ".join([ex.get("user", "") for ex in exchanges])

            # Analyze conversation using AnalysisAgent
            analysis_message = AgentMessage(
                sender="concept_extractor",
                recipient="analysis_agent",
                message_type=MessageType.ANALYSIS_REQUEST,
                content={
                    "action": "analyze_conversation",
                    "exchanges": exchanges
                }
            )

            analysis_result = await self.analysis_agent.process(analysis_message)
            analysis = analysis_result.content

            # Extract concepts and entities
            concepts = self._extract_concepts_from_analysis(analysis)
            entities = self._extract_entities_from_analysis(analysis)

            # Classify topics
            topics = self._classify_topics(all_user_text, concepts)

            # Detect relationships with conversation context
            relationships = await self._detect_relationships(concepts, all_user_text, analysis)

            # Store in graph with session linkage
            await self._store_conversation_in_graph(
                session_id=session_id,
                concepts=concepts,
                entities=entities,
                relationships=relationships,
                topics=topics,
                metadata={
                    **(metadata or {}),
                    "exchange_count": len(exchanges),
                    "total_text_length": len(all_user_text)
                }
            )

            result = {
                "session_id": session_id,
                "concepts": concepts,
                "entities": entities,
                "relationships": relationships,
                "topics": topics,
                "stats": {
                    "concept_count": len(concepts),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "topic_count": len(topics),
                    "exchange_count": len(exchanges)
                },
                "extracted_at": datetime.utcnow().isoformat()
            }

            self.logger.info(
                "conversation_extraction_completed",
                session_id=session_id,
                concepts=len(concepts),
                relationships=len(relationships)
            )

            return result

        except Exception as e:
            self.logger.error(
                "conversation_extraction_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return self._empty_extraction_result()

    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Use AnalysisAgent for NLP analysis"""
        message = AgentMessage(
            sender="concept_extractor",
            recipient="analysis_agent",
            message_type=MessageType.ANALYSIS_REQUEST,
            content={
                "action": "analyze_text",
                "text": text
            }
        )

        result = await self.analysis_agent.process(message)
        return result.content

    def _extract_concepts_from_analysis(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Extract unique concepts from analysis results

        PATTERN: Deduplication with normalization
        WHY: Avoid duplicate concepts with different casing/forms
        """
        concepts = set()

        # Get concepts from analysis
        for concept in analysis.get("concepts", []):
            normalized = self._normalize_concept(concept)
            if normalized and len(normalized) > 2:  # Filter very short concepts
                concepts.add(normalized)

        # Get keywords as additional concepts
        for keyword_data in analysis.get("keywords", []):
            keyword = keyword_data.get("keyword", "")
            normalized = self._normalize_concept(keyword)
            if normalized and len(normalized) > 3:
                concepts.add(normalized)

        return sorted(list(concepts))

    def _extract_entities_from_analysis(self, analysis: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract entities as (text, type) tuples"""
        entities = []

        for entity in analysis.get("entities", []):
            text = entity.get("text", "")
            label = entity.get("label", "UNKNOWN")

            if text and len(text) > 1:
                entities.append((text, label))

        return entities

    def _normalize_concept(self, concept: str) -> str:
        """Normalize concept for consistency"""
        # Lowercase and strip
        normalized = concept.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove common prefixes/suffixes
        for prefix in ['the ', 'a ', 'an ']:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        return normalized

    def _classify_topics(self, text: str, concepts: List[str]) -> List[str]:
        """
        Classify text into topic categories

        Returns:
            List of topic names
        """
        text_lower = text.lower()
        topics = set()

        # Check each topic's keywords
        for topic, keywords in self.topic_hierarchy.items():
            # Check if any keywords or concepts match
            matches = sum(1 for kw in keywords if kw in text_lower)
            concept_matches = sum(1 for concept in concepts if any(kw in concept for kw in keywords))

            if matches + concept_matches >= 2:  # Minimum 2 matches
                topics.add(topic)

        return sorted(list(topics))

    async def _detect_relationships(
        self,
        concepts: List[str],
        text: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships between concepts using multiple signals

        Signals:
        1. Co-occurrence proximity (within N characters)
        2. Co-occurrence frequency
        3. Prerequisite indicators
        4. Topic similarity

        Returns:
            List of {from_concept, to_concept, type, strength, context}
        """
        if len(concepts) < 2:
            return []

        relationships = []
        text_lower = text.lower()

        # Detect co-occurrence relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Find all positions of both concepts
                positions1 = [m.start() for m in re.finditer(re.escape(concept1), text_lower)]
                positions2 = [m.start() for m in re.finditer(re.escape(concept2), text_lower)]

                if not positions1 or not positions2:
                    continue

                # Check proximity
                min_distance = min(
                    abs(p1 - p2)
                    for p1 in positions1
                    for p2 in positions2
                )

                if min_distance <= self.proximity_threshold:
                    # Calculate relationship strength
                    strength = self._calculate_relationship_strength(
                        concept1, concept2, text_lower, positions1, positions2, min_distance
                    )

                    # Determine relationship type
                    rel_type = self._determine_relationship_type(
                        concept1, concept2, text_lower
                    )

                    # Extract context
                    context = self._extract_relationship_context(
                        concept1, concept2, text, positions1[0], positions2[0]
                    )

                    relationships.append({
                        "from_concept": concept1,
                        "to_concept": concept2,
                        "type": rel_type,
                        "strength": round(strength, 2),
                        "context": context,
                        "distance": min_distance
                    })

        return relationships

    def _calculate_relationship_strength(
        self,
        concept1: str,
        concept2: str,
        text: str,
        positions1: List[int],
        positions2: List[int],
        min_distance: int
    ) -> float:
        """
        Calculate relationship strength using multiple factors

        Factors:
        - Proximity (closer = stronger)
        - Co-occurrence frequency (more = stronger)
        - Relative positions in text
        """
        # Base strength
        strength = self.base_relationship_strength

        # Proximity factor (inverse of distance, normalized)
        proximity_factor = 1.0 - (min_distance / self.proximity_threshold)
        strength += proximity_factor * 0.3

        # Frequency factor
        co_occurrence_count = len(positions1) * len(positions2)
        frequency_factor = min(co_occurrence_count / 10.0, 0.2)  # Cap at 0.2
        strength += frequency_factor

        # Normalize to 0-1 range
        return min(max(strength, 0.0), 1.0)

    def _determine_relationship_type(
        self,
        concept1: str,
        concept2: str,
        text: str
    ) -> str:
        """
        Determine relationship type using linguistic cues

        Types:
        - BUILDS_ON: Prerequisite relationship
        - RELATES_TO: General co-occurrence
        """
        # Extract text between concepts
        idx1 = text.find(concept1)
        idx2 = text.find(concept2)

        if idx1 < 0 or idx2 < 0:
            return "RELATES_TO"

        # Get text between concepts
        start = min(idx1, idx2)
        end = max(idx1, idx2)
        between_text = text[start:end].lower()

        # Check for prerequisite indicators
        for indicator in self.prerequisite_indicators:
            if indicator in between_text:
                # Determine direction based on position
                if idx1 < idx2:
                    return "BUILDS_ON"  # concept1 builds on concept2
                else:
                    return "BUILDS_ON"  # concept2 builds on concept1

        return "RELATES_TO"

    def _extract_relationship_context(
        self,
        concept1: str,
        concept2: str,
        text: str,
        pos1: int,
        pos2: int,
        context_window: int = 50
    ) -> str:
        """Extract surrounding context for relationship"""
        start = max(0, min(pos1, pos2) - context_window)
        end = min(len(text), max(pos1, pos2) + max(len(concept1), len(concept2)) + context_window)

        context = text[start:end].strip()

        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    async def _store_in_graph(
        self,
        concepts: List[str],
        entities: List[Tuple[str, str]],
        relationships: List[Dict[str, Any]],
        topics: List[str],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store extraction results in graph"""
        try:
            # Store concepts
            for concept in concepts:
                topic = topics[0] if topics else None
                await self.graph_store.add_concept(
                    name=concept,
                    topic=topic,
                    metadata=metadata
                )

            # Store entities
            for text, entity_type in entities:
                # Try to link entity to related concept
                concept = self._find_related_concept(text, concepts)
                await self.graph_store.add_entity(
                    text=text,
                    entity_type=entity_type,
                    concept=concept
                )

            # Store relationships
            for rel in relationships:
                await self.graph_store.add_relationship(
                    from_concept=rel["from_concept"],
                    to_concept=rel["to_concept"],
                    relationship_type=rel["type"],
                    strength=rel["strength"],
                    context=rel.get("context")
                )

            self.logger.info(
                "graph_storage_completed",
                concepts=len(concepts),
                entities=len(entities),
                relationships=len(relationships)
            )

        except Exception as e:
            self.logger.error("graph_storage_failed", error=str(e), exc_info=True)
            raise

    async def _store_conversation_in_graph(
        self,
        session_id: str,
        concepts: List[str],
        entities: List[Tuple[str, str]],
        relationships: List[Dict[str, Any]],
        topics: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store conversation with session linkage"""
        try:
            # Store concepts and entities first
            await self._store_in_graph(
                concepts=concepts,
                entities=entities,
                relationships=relationships,
                topics=topics,
                metadata=metadata
            )

            # Link to session
            await self.graph_store.add_session(
                session_id=session_id,
                concepts=concepts,
                entities=entities,
                metadata=metadata
            )

            self.logger.info(
                "conversation_stored",
                session_id=session_id,
                concepts=len(concepts),
                relationships=len(relationships)
            )

        except Exception as e:
            self.logger.error(
                "conversation_storage_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            raise

    def _find_related_concept(self, entity_text: str, concepts: List[str]) -> Optional[str]:
        """Find concept related to entity"""
        entity_lower = entity_text.lower()

        # Direct match
        if entity_lower in concepts:
            return entity_lower

        # Partial match
        for concept in concepts:
            if concept in entity_lower or entity_lower in concept:
                return concept

        return None

    async def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics from graph"""
        return await self.graph_store.get_graph_stats()

    def _empty_extraction_result(self) -> Dict[str, Any]:
        """Return empty extraction result"""
        return {
            "concepts": [],
            "entities": [],
            "relationships": [],
            "topics": [],
            "stats": {
                "concept_count": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "topic_count": 0
            },
            "extracted_at": datetime.utcnow().isoformat()
        }
