"""
MultiModal Indexer - Vector Indexing for Images and Documents

SPECIFICATION:
- Generate embeddings for image analyses and document chunks
- Store in vector database (ChromaDB)
- Support multi-modal queries
- Track indexing status
- Batch indexing operations

ARCHITECTURE:
- Integration with existing vector store
- OpenAI embeddings for text
- Combined image analysis + text embeddings
- Async batch operations

PATTERN: Adapter pattern for vector store integration
WHY: Unified indexing interface for multiple content types
RESILIENCE: Batch processing, error handling, retry logic
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from openai import AsyncOpenAI
from app.logger import api_logger
from app.config import settings
from app.search.vector_store import vector_store


class MultiModalIndexer:
    """
    Vector indexing for multimodal content

    PATTERN: Service class for vector indexing
    WHY: Centralized embedding and indexing logic
    """

    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize multimodal indexer

        Args:
            embedding_model: OpenAI embedding model
        """
        self.embedding_model = embedding_model
        self.openai_client = None

        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        api_logger.info(
            "multimodal_indexer_initialized",
            embedding_model=embedding_model,
            openai_configured=self.openai_client is not None
        )

    async def index_image(
        self,
        file_id: str,
        analysis: Dict,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Index image analysis in vector store

        ALGORITHM:
        1. Extract analysis text
        2. Generate embedding
        3. Store in vector store with metadata
        4. Return success status

        Args:
            file_id: Unique file ID
            analysis: Vision analysis results
            session_id: Optional session ID

        Returns:
            True if indexed successfully
        """
        if not self.openai_client:
            api_logger.warning(
                "openai_not_configured",
                message="Cannot generate embeddings without OpenAI API key"
            )
            return False

        if not analysis.get('success'):
            api_logger.warning(
                "analysis_failed",
                file_id=file_id,
                message="Cannot index failed analysis"
            )
            return False

        try:
            # Extract analysis text
            analysis_text = analysis.get('analysis', '')
            if not analysis_text:
                return False

            # Prepare text for embedding
            index_text = f"Image Analysis: {analysis_text}"

            # Generate embedding
            api_logger.debug(
                "generating_image_embedding",
                file_id=file_id,
                text_length=len(index_text)
            )

            response = await self.openai_client.embeddings.create(
                input=index_text,
                model=self.embedding_model
            )

            import numpy as np
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Store in vector store
            await vector_store.add_embedding(
                capture_id=file_id,  # Use file_id as capture_id
                embedding=embedding,
                model=self.embedding_model
            )

            api_logger.info(
                "image_indexed",
                file_id=file_id,
                embedding_dimension=len(embedding),
                session_id=session_id
            )

            return True

        except Exception as e:
            api_logger.error(
                "image_indexing_error",
                file_id=file_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return False

    async def index_document(
        self,
        file_id: str,
        chunks: List[Dict],
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Index document chunks in vector store

        ALGORITHM:
        1. For each chunk:
           a. Generate embedding
           b. Store in vector store
        2. Track indexing progress
        3. Return statistics

        Args:
            file_id: Unique file ID
            chunks: List of text chunks
            session_id: Optional session ID

        Returns:
            Indexing statistics
        """
        if not self.openai_client:
            api_logger.warning("openai_not_configured")
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }

        try:
            indexed_count = 0
            failed_count = 0

            api_logger.info(
                "document_indexing_started",
                file_id=file_id,
                chunk_count=len(chunks)
            )

            # Index chunks in batches
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Process batch
                results = await asyncio.gather(
                    *[self._index_chunk(file_id, chunk) for chunk in batch],
                    return_exceptions=True
                )

                # Count successes and failures
                for result in results:
                    if isinstance(result, Exception):
                        failed_count += 1
                    elif result:
                        indexed_count += 1
                    else:
                        failed_count += 1

            api_logger.info(
                "document_indexing_complete",
                file_id=file_id,
                indexed_count=indexed_count,
                failed_count=failed_count,
                total_chunks=len(chunks)
            )

            return {
                "success": True,
                "file_id": file_id,
                "total_chunks": len(chunks),
                "indexed_count": indexed_count,
                "failed_count": failed_count,
                "session_id": session_id
            }

        except Exception as e:
            api_logger.error(
                "document_indexing_error",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            return {
                "success": False,
                "error": str(e)
            }

    async def _index_chunk(self, file_id: str, chunk: Dict) -> bool:
        """
        Index individual document chunk

        Args:
            file_id: File ID
            chunk: Chunk dictionary

        Returns:
            True if successful
        """
        try:
            chunk_text = chunk.get('text', '')
            if not chunk_text:
                return False

            # Generate embedding
            response = await self.openai_client.embeddings.create(
                input=chunk_text,
                model=self.embedding_model
            )

            import numpy as np
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Create unique ID for chunk
            chunk_id = f"{file_id}_chunk_{chunk['chunk_id']}"

            # Store in vector store
            await vector_store.add_embedding(
                capture_id=chunk_id,
                embedding=embedding,
                model=self.embedding_model
            )

            return True

        except Exception as e:
            api_logger.error(
                "chunk_indexing_error",
                file_id=file_id,
                chunk_id=chunk.get('chunk_id'),
                error=str(e)
            )
            return False

    async def index_multimodal_conversation(
        self,
        conversation_id: str,
        text: str,
        image_analyses: Optional[List[Dict]] = None,
        document_chunks: Optional[List[Dict]] = None
    ) -> bool:
        """
        Index multimodal conversation with combined context

        ALGORITHM:
        1. Combine text, image analyses, and document content
        2. Generate unified embedding
        3. Store with multimodal metadata

        Args:
            conversation_id: Conversation ID
            text: Conversation text
            image_analyses: Optional list of image analyses
            document_chunks: Optional list of document chunks

        Returns:
            True if indexed successfully
        """
        if not self.openai_client:
            return False

        try:
            # Build combined text for embedding
            combined_parts = [text]

            # Add image analyses
            if image_analyses:
                for analysis in image_analyses:
                    if analysis.get('success'):
                        combined_parts.append(f"[Image: {analysis['analysis']}]")

            # Add document excerpts
            if document_chunks:
                for chunk in document_chunks[:3]:  # Limit to first 3 chunks
                    combined_parts.append(f"[Document: {chunk.get('text', '')[:200]}]")

            combined_text = "\n\n".join(combined_parts)

            # Generate embedding
            response = await self.openai_client.embeddings.create(
                input=combined_text,
                model=self.embedding_model
            )

            import numpy as np
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Store in vector store
            await vector_store.add_embedding(
                capture_id=conversation_id,
                embedding=embedding,
                model=self.embedding_model
            )

            api_logger.info(
                "multimodal_conversation_indexed",
                conversation_id=conversation_id,
                has_images=bool(image_analyses),
                has_documents=bool(document_chunks)
            )

            return True

        except Exception as e:
            api_logger.error(
                "multimodal_conversation_indexing_error",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True
            )
            return False

    async def search_multimodal(
        self,
        query: str,
        top_k: int = 10,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Search multimodal content

        Args:
            query: Search query
            top_k: Number of results
            filter_type: Optional filter ('image', 'document', 'conversation')

        Returns:
            List of search results
        """
        if not self.openai_client:
            api_logger.warning("openai_not_configured")
            return []

        try:
            # Generate query embedding
            response = await self.openai_client.embeddings.create(
                input=query,
                model=self.embedding_model
            )

            import numpy as np
            query_embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Search vector store
            results = await vector_store.search_similar(
                query_embedding=query_embedding,
                limit=top_k
            )

            # Filter by type if requested
            if filter_type:
                results = [
                    r for r in results
                    if filter_type in r.get('capture_id', '')
                ]

            api_logger.info(
                "multimodal_search_complete",
                query=query[:100],
                results_count=len(results),
                filter_type=filter_type
            )

            return results

        except Exception as e:
            api_logger.error(
                "multimodal_search_error",
                query=query[:100],
                error=str(e),
                exc_info=True
            )
            return []


# Singleton instance
multimodal_indexer = MultiModalIndexer()
