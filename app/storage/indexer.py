"""
Multi-Modal Indexer

SPECIFICATION:
- Index files in vector store with embeddings
- Index metadata in full-text search
- Link files to knowledge graph concepts
- Support for images, documents, and audio

ARCHITECTURE:
- Integration with ChromaDB vector store
- Integration with Neo4j knowledge graph
- Async indexing operations
- Batch processing support
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime

from app.storage.config import storage_config
from app.storage.metadata_store import metadata_store
from app.vector.embeddings import embedding_generator
from app.knowledge_graph.graph_store import KnowledgeGraphStore
from app.knowledge_graph.config import kg_config
from app.storage.chroma_db import chroma_storage
from app.logger import get_logger
from app.resilience import with_retry

logger = get_logger(__name__)


class MultiModalIndexer:
    """
    Multi-modal file indexer

    PATTERN: Facade pattern for multiple indexing backends
    WHY: Unified interface for vector, text, and graph indexing

    Features:
    - Vector embeddings for semantic search
    - Full-text indexing for metadata search
    - Knowledge graph linking for relationships
    - Batch operations for efficiency

    Example:
        indexer = MultiModalIndexer()
        await indexer.initialize()

        # Index image with vision analysis
        await indexer.index_image(
            file_id="img_123",
            vision_analysis={
                "objects": ["cat", "tree", "sky"],
                "description": "A cat sitting under a tree",
                "labels": [{"name": "cat", "confidence": 0.98}]
            },
            session_id="session_456"
        )

        # Index document with extracted text
        await indexer.index_document(
            file_id="doc_123",
            extracted_text="Machine learning is a subset of AI...",
            session_id="session_456"
        )

        # Search similar files
        results = await indexer.search_similar_files(
            query="cats in nature",
            file_type="image",
            n_results=5
        )
    """

    def __init__(
        self,
        enable_vector: bool = True,
        enable_knowledge_graph: bool = True
    ):
        self.config = storage_config
        self.metadata_store = metadata_store
        self.enable_vector = enable_vector and self.config.enable_vector_indexing
        self.enable_kg = enable_knowledge_graph and self.config.enable_knowledge_graph

        self.embedding_generator = embedding_generator if self.enable_vector else None
        self.chroma_storage = chroma_storage if self.enable_vector else None
        self.kg_store: Optional[KnowledgeGraphStore] = None

        self._initialized = False

    async def initialize(self):
        """Initialize indexer and all backends"""
        if self._initialized:
            return

        try:
            logger.info(
                "indexer_initialization_started",
                vector_enabled=self.enable_vector,
                kg_enabled=self.enable_kg
            )

            # Initialize metadata store
            await self.metadata_store.initialize()

            # Initialize vector embeddings
            if self.enable_vector:
                await self.embedding_generator.initialize()
                logger.info("vector_indexing_enabled")

            # Initialize knowledge graph
            if self.enable_kg:
                self.kg_store = KnowledgeGraphStore(kg_config)
                await self.kg_store.initialize()
                logger.info("knowledge_graph_indexing_enabled")

            self._initialized = True
            logger.info("indexer_initialized")

        except Exception as e:
            logger.error(
                "indexer_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    @with_retry(max_attempts=2, min_wait=0.5)
    async def index_image(
        self,
        file_id: str,
        vision_analysis: Dict[str, Any],
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Index image file with vision analysis

        PATTERN: Multi-backend indexing with error isolation
        WHY: Resilient to individual backend failures

        Args:
            file_id: File identifier
            vision_analysis: Results from vision API (objects, labels, description)
            session_id: Session identifier
            metadata: Additional metadata
        """
        try:
            logger.info("indexing_image_started", file_id=file_id)

            # Save analysis to metadata store
            await self.metadata_store.save_analysis(
                file_id=file_id,
                analysis_type="vision",
                result=vision_analysis
            )

            # Extract text for indexing
            description = vision_analysis.get("description", "")
            objects = vision_analysis.get("objects", [])
            labels = vision_analysis.get("labels", [])

            # Combine into searchable text
            label_names = [label.get("name", "") for label in labels if isinstance(label, dict)]
            indexable_text = f"{description} {' '.join(objects)} {' '.join(label_names)}"

            # Vector indexing
            if self.enable_vector and indexable_text.strip():
                await self._index_to_vector_store(
                    file_id=file_id,
                    text=indexable_text,
                    file_type="image",
                    session_id=session_id,
                    metadata={
                        **(metadata or {}),
                        "analysis_type": "vision",
                        "objects": objects[:5],  # Top 5 objects
                        "has_description": bool(description)
                    }
                )

            # Knowledge graph linking
            if self.enable_kg:
                # Extract concepts from objects and labels
                concepts = list(set(objects + label_names))[:10]  # Top 10 unique concepts
                await self._link_to_knowledge_graph(
                    file_id=file_id,
                    concepts=concepts,
                    session_id=session_id
                )

            logger.info(
                "image_indexed",
                file_id=file_id,
                concepts_count=len(objects),
                has_description=bool(description)
            )

        except Exception as e:
            logger.error(
                "index_image_failed",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            raise

    @with_retry(max_attempts=2, min_wait=0.5)
    async def index_document(
        self,
        file_id: str,
        extracted_text: str,
        session_id: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        analysis_type: str = "extraction"
    ):
        """
        Index document file with extracted text

        Args:
            file_id: File identifier
            extracted_text: Text extracted from document
            session_id: Session identifier
            document_metadata: Metadata (pages, author, title, etc.)
            analysis_type: Type of extraction (extraction, ocr)
        """
        try:
            logger.info(
                "indexing_document_started",
                file_id=file_id,
                text_length=len(extracted_text)
            )

            # Save analysis to metadata store
            analysis_result = {
                "text": extracted_text,
                "text_length": len(extracted_text),
                **(document_metadata or {})
            }
            await self.metadata_store.save_analysis(
                file_id=file_id,
                analysis_type=analysis_type,
                result=analysis_result
            )

            # Vector indexing (chunk long documents)
            if self.enable_vector and extracted_text.strip():
                chunks = self._chunk_text(extracted_text, chunk_size=1000, overlap=100)

                for i, chunk in enumerate(chunks):
                    await self._index_to_vector_store(
                        file_id=f"{file_id}_chunk_{i}",
                        text=chunk,
                        file_type="document",
                        session_id=session_id,
                        metadata={
                            "parent_file_id": file_id,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "analysis_type": analysis_type,
                            **(document_metadata or {})
                        }
                    )

            # Knowledge graph linking (extract key concepts)
            if self.enable_kg:
                # Simple keyword extraction (in production, use NER or keyword extraction)
                concepts = await self._extract_concepts_from_text(extracted_text)
                await self._link_to_knowledge_graph(
                    file_id=file_id,
                    concepts=concepts,
                    session_id=session_id
                )

            logger.info(
                "document_indexed",
                file_id=file_id,
                chunks_count=len(chunks) if self.enable_vector else 0
            )

        except Exception as e:
            logger.error(
                "index_document_failed",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            raise

    @with_retry(max_attempts=2, min_wait=0.5)
    async def index_audio(
        self,
        file_id: str,
        transcription: str,
        session_id: str,
        audio_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Index audio file with transcription

        Args:
            file_id: File identifier
            transcription: Audio transcription text
            session_id: Session identifier
            audio_metadata: Metadata (duration, speaker, etc.)
        """
        try:
            logger.info("indexing_audio_started", file_id=file_id)

            # Save transcription as analysis
            analysis_result = {
                "transcription": transcription,
                "text_length": len(transcription),
                **(audio_metadata or {})
            }
            await self.metadata_store.save_analysis(
                file_id=file_id,
                analysis_type="transcription",
                result=analysis_result
            )

            # Index transcription text (similar to document)
            if self.enable_vector and transcription.strip():
                await self._index_to_vector_store(
                    file_id=file_id,
                    text=transcription,
                    file_type="audio",
                    session_id=session_id,
                    metadata={
                        "analysis_type": "transcription",
                        **(audio_metadata or {})
                    }
                )

            # Knowledge graph linking
            if self.enable_kg:
                concepts = await self._extract_concepts_from_text(transcription)
                await self._link_to_knowledge_graph(
                    file_id=file_id,
                    concepts=concepts,
                    session_id=session_id
                )

            logger.info("audio_indexed", file_id=file_id)

        except Exception as e:
            logger.error("index_audio_failed", file_id=file_id, error=str(e))
            raise

    async def search_similar_files(
        self,
        query: str,
        file_type: Optional[str] = None,
        session_id: Optional[str] = None,
        n_results: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar files using vector similarity

        Args:
            query: Search query
            file_type: Filter by file type
            session_id: Filter by session
            n_results: Number of results
            min_score: Minimum similarity score

        Returns:
            List of similar files with scores
        """
        if not self.enable_vector:
            logger.warning("vector_search_disabled")
            return []

        try:
            # Build metadata filter
            where_filter = {}
            if file_type:
                where_filter["file_type"] = file_type
            if session_id:
                where_filter["session_id"] = session_id

            # Search vector store
            results = self.chroma_storage.search(
                query=query,
                n_results=n_results,
                min_score=min_score
            )

            # Enrich results with full metadata
            enriched_results = []
            for result in results:
                file_id = result["metadata"].get("parent_file_id") or result["metadata"].get("file_id")
                if file_id:
                    file_metadata = await self.metadata_store.get_file_by_id(file_id)
                    if file_metadata:
                        enriched_results.append({
                            **result,
                            "file_metadata": file_metadata
                        })

            logger.info(
                "similar_files_searched",
                query_length=len(query),
                results_count=len(enriched_results),
                file_type=file_type
            )

            return enriched_results

        except Exception as e:
            logger.error("search_similar_files_failed", query=query[:100], error=str(e))
            return []

    async def update_index(self, file_id: str):
        """
        Re-index file after analysis update

        Args:
            file_id: File identifier
        """
        try:
            # Get file metadata
            file_metadata = await self.metadata_store.get_file_by_id(file_id)
            if not file_metadata:
                logger.warning("update_index_file_not_found", file_id=file_id)
                return

            # Get latest analysis
            analyses = await self.metadata_store.get_file_analysis(file_id)
            if not analyses:
                logger.warning("update_index_no_analysis", file_id=file_id)
                return

            # Re-index based on file type and analysis
            file_type = file_metadata["file_type"]
            session_id = file_metadata["session_id"]

            for analysis in analyses:
                analysis_type = analysis["analysis_type"]
                result = analysis["analysis_result"]

                if analysis_type == "vision":
                    await self.index_image(file_id, result, session_id)
                elif analysis_type in ["extraction", "ocr"]:
                    text = result.get("text", "")
                    await self.index_document(file_id, text, session_id, analysis_type=analysis_type)
                elif analysis_type == "transcription":
                    transcription = result.get("transcription", "")
                    await self.index_audio(file_id, transcription, session_id)

            logger.info("index_updated", file_id=file_id)

        except Exception as e:
            logger.error("update_index_failed", file_id=file_id, error=str(e))
            raise

    # Private helper methods

    async def _index_to_vector_store(
        self,
        file_id: str,
        text: str,
        file_type: str,
        session_id: str,
        metadata: Dict[str, Any]
    ):
        """Index text to vector store"""
        try:
            # Add file context to text
            indexed_text = f"[{file_type}] {text}"

            # Add to ChromaDB
            self.chroma_storage.add_conversation(
                session_id=session_id,
                user_text=indexed_text,
                agent_text="",  # No agent response for files
                metadata={
                    **metadata,
                    "file_id": file_id,
                    "file_type": file_type,
                    "type": "file",
                    "indexed_at": datetime.utcnow().isoformat()
                }
            )

            logger.debug("vector_indexed", file_id=file_id, text_length=len(text))

        except Exception as e:
            logger.error("vector_indexing_failed", file_id=file_id, error=str(e))
            # Don't raise - allow other indexing to continue

    async def _link_to_knowledge_graph(
        self,
        file_id: str,
        concepts: List[str],
        session_id: str
    ):
        """Link file to knowledge graph concepts"""
        if not self.kg_store:
            return

        try:
            for concept in concepts[:10]:  # Limit to top 10 concepts
                # Add concept to graph
                await self.kg_store.add_concept(
                    name=concept.lower(),
                    description=f"Concept from file {file_id}",
                    metadata={"source": "file_indexing"}
                )

                # Link file to concept in metadata store
                await self.metadata_store.link_file_to_concept(
                    file_id=file_id,
                    concept_name=concept.lower(),
                    link_type="extracted",
                    confidence=0.8
                )

            logger.debug("knowledge_graph_linked", file_id=file_id, concepts_count=len(concepts))

        except Exception as e:
            logger.error("knowledge_graph_linking_failed", file_id=file_id, error=str(e))
            # Don't raise - allow other indexing to continue

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks

        PATTERN: Sliding window chunking
        WHY: Maintain context across chunks for better search
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.7:  # At least 70% of chunk
                    end = start + last_period + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    async def _extract_concepts_from_text(self, text: str, max_concepts: int = 10) -> List[str]:
        """
        Extract key concepts from text

        PATTERN: Simple frequency-based extraction
        WHY: Lightweight, no external dependencies
        NOTE: In production, use NER or keyword extraction models
        """
        # Simple implementation: extract capitalized words and common nouns
        # This is a placeholder - in production use NER or RAKE
        import re

        # Extract capitalized words (likely proper nouns)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Count frequencies
        from collections import Counter
        word_counts = Counter(words)

        # Get top concepts
        concepts = [word for word, count in word_counts.most_common(max_concepts)]

        return concepts


# Global indexer instance
multimodal_indexer = MultiModalIndexer()
