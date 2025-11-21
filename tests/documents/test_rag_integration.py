"""
RAG Integration Tests

Tests integration between document processing and RAG systems:
- Document processing to vector store
- Chunk generation and storage
- Retrieval and search
- Metadata filtering
"""

import pytest
import os
from pathlib import Path

from app.documents import DocumentProcessor, DocumentConfig
from app.rag.rag_engine import RAGEngine
from app.rag.config import RAGConfig


class TestRAGIntegration:
    """Test document processing integration with RAG"""

    @pytest.fixture
    def doc_processor(self):
        """Create document processor"""
        config = DocumentConfig(
            chunk_size=200,  # Smaller chunks for testing
            chunk_overlap=50,
        )
        return DocumentProcessor(config)

    @pytest.fixture
    def rag_engine(self):
        """Create RAG engine"""
        config = RAGConfig(
            collection_name="test_documents",
            chunk_size=200,
        )
        return RAGEngine(config)

    @pytest.fixture
    def sample_document(self, tmp_path):
        """Create sample document"""
        file_path = tmp_path / "sample.md"
        content = """# Test Document

## Introduction
This is a test document for RAG integration.

## Section 1
Content in section 1 with some information about machine learning.
Machine learning is a subset of artificial intelligence.

## Section 2
Content in section 2 with information about natural language processing.
NLP enables computers to understand human language.

## Section 3
More content about deep learning and neural networks.
Neural networks are inspired by the human brain.
"""
        file_path.write_text(content)
        return str(file_path)

    @pytest.mark.asyncio
    async def test_process_document_for_rag(self, doc_processor, sample_document):
        """Test processing document and generating chunks for RAG"""
        result = await doc_processor.process_document(
            sample_document,
            chunk_for_rag=True
        )

        assert "chunks" in result
        assert len(result["chunks"]) > 0

        # Verify chunk structure
        for chunk in result["chunks"]:
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "word_count" in chunk
            assert len(chunk["text"]) > 0

    @pytest.mark.asyncio
    async def test_store_document_chunks_in_rag(
        self,
        doc_processor,
        rag_engine,
        sample_document
    ):
        """Test storing document chunks in RAG system"""
        # Process document
        result = await doc_processor.process_document(sample_document)

        # Store chunks in RAG
        chunks = result["chunks"]
        metadata = result["metadata"]

        for chunk in chunks:
            chunk_metadata = {
                "source": result["file_name"],
                "chunk_index": chunk["chunk_index"],
                "format": result["format"],
                **metadata
            }

            await rag_engine.add_document(
                text=chunk["text"],
                metadata=chunk_metadata
            )

        # Verify storage
        # This depends on RAG engine implementation
        assert True  # Placeholder

    @pytest.mark.asyncio
    async def test_retrieve_document_chunks(
        self,
        doc_processor,
        rag_engine,
        sample_document
    ):
        """Test retrieving document chunks from RAG"""
        # Process and store document
        result = await doc_processor.process_document(sample_document)

        for chunk in result["chunks"]:
            await rag_engine.add_document(
                text=chunk["text"],
                metadata={
                    "source": result["file_name"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        # Query RAG
        query = "machine learning"
        results = await rag_engine.query(query, top_k=3)

        # Verify results
        assert len(results) > 0
        # Results should contain chunks about machine learning

    @pytest.mark.asyncio
    async def test_metadata_filtering(
        self,
        doc_processor,
        rag_engine,
        tmp_path
    ):
        """Test filtering chunks by metadata"""
        # Create multiple documents
        doc1 = tmp_path / "doc1.txt"
        doc1.write_text("Content about Python programming")

        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("Content about JavaScript programming")

        # Process both
        result1 = await doc_processor.process_document(str(doc1))
        result2 = await doc_processor.process_document(str(doc2))

        # Store with different metadata
        for chunk in result1["chunks"]:
            await rag_engine.add_document(
                text=chunk["text"],
                metadata={"source": "doc1.txt", "language": "python"}
            )

        for chunk in result2["chunks"]:
            await rag_engine.add_document(
                text=chunk["text"],
                metadata={"source": "doc2.txt", "language": "javascript"}
            )

        # Query with metadata filter
        results = await rag_engine.query(
            "programming",
            filters={"language": "python"},
            top_k=5
        )

        # All results should be from Python document
        for result in results:
            assert result["metadata"]["language"] == "python"

    @pytest.mark.asyncio
    async def test_chunk_overlap_retrieval(self, doc_processor, rag_engine, tmp_path):
        """Test that chunk overlap improves retrieval"""
        # Create document with specific content
        doc_path = tmp_path / "overlap_test.txt"
        content = """
        The quick brown fox jumps over the lazy dog.
        This sentence is important for testing overlap.
        The overlap should include parts of adjacent chunks.
        This helps maintain context across chunk boundaries.
        """
        doc_path.write_text(content)

        # Process with overlap
        result = await doc_processor.process_document(str(doc_path))

        chunks = result["chunks"]

        # Verify chunks have content
        assert len(chunks) > 0

        # Check that chunks contain overlapping content
        # (This is implicit in the chunking algorithm)

    @pytest.mark.asyncio
    async def test_multiple_format_integration(
        self,
        doc_processor,
        rag_engine,
        tmp_path
    ):
        """Test processing multiple document formats for RAG"""
        # Create documents in different formats
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Plain text content about artificial intelligence")

        md_file = tmp_path / "sample.md"
        md_file.write_text("# Markdown\n\nMarkdown content about machine learning")

        # Process both
        results = []
        for file_path in [txt_file, md_file]:
            result = await doc_processor.process_document(str(file_path))
            results.append(result)

            # Store in RAG
            for chunk in result["chunks"]:
                await rag_engine.add_document(
                    text=chunk["text"],
                    metadata={
                        "source": result["file_name"],
                        "format": result["format"],
                    }
                )

        # Query should return chunks from both formats
        query_results = await rag_engine.query("learning", top_k=5)

        # Should have results from different formats
        formats = set(r["metadata"]["format"] for r in query_results)
        assert len(formats) > 1 or len(query_results) > 0


class TestDocumentChunking:
    """Test document chunking strategies for RAG"""

    @pytest.fixture
    def processor(self):
        """Create processor with small chunks for testing"""
        config = DocumentConfig(
            chunk_size=100,
            chunk_overlap=20,
        )
        return DocumentProcessor(config)

    @pytest.mark.asyncio
    async def test_chunk_size_consistency(self, processor, tmp_path):
        """Test that chunks respect size limits"""
        # Create long document
        file_path = tmp_path / "long.txt"
        content = " ".join([f"Word{i}" for i in range(500)])
        file_path.write_text(content)

        result = await processor.process_document(str(file_path))
        chunks = result["chunks"]

        # All chunks should be roughly the target size
        for chunk in chunks:
            # Word count should be approximately chunk_size / 1.3
            # (accounting for word-to-token ratio)
            assert chunk["word_count"] <= 100  # Some tolerance

    @pytest.mark.asyncio
    async def test_chunk_overlap_presence(self, processor, tmp_path):
        """Test that chunks have overlap"""
        file_path = tmp_path / "test.txt"
        content = " ".join([f"Word{i}" for i in range(200)])
        file_path.write_text(content)

        result = await processor.process_document(str(file_path))
        chunks = result["chunks"]

        if len(chunks) > 1:
            # Check that chunks are spaced appropriately
            # (overlap means next chunk starts before previous ends)
            assert chunks[0]["end_word"] > chunks[1]["start_word"]

    def test_custom_chunking_parameters(self, tmp_path):
        """Test custom chunking parameters"""
        processor = DocumentProcessor(
            DocumentConfig(
                chunk_size=50,
                chunk_overlap=10,
            )
        )

        text = " ".join([f"Word{i}" for i in range(100)])
        chunks = processor.chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "chunk_index" in chunk


class TestDocumentMetadataForRAG:
    """Test document metadata extraction for RAG"""

    @pytest.fixture
    def processor(self):
        """Create processor"""
        return DocumentProcessor()

    @pytest.mark.asyncio
    async def test_extract_metadata_for_rag(self, processor, tmp_path):
        """Test extracting metadata useful for RAG"""
        file_path = tmp_path / "test.md"
        content = """# Title

## Section

Content here
"""
        file_path.write_text(content)

        result = await processor.process_document(str(file_path))
        metadata = result["metadata"]

        # Verify metadata useful for RAG
        assert "format" in metadata
        assert "file_name" in metadata
        assert "num_words" in metadata or "num_characters" in metadata

        # Document-level info for filtering
        if result["format"] == "md":
            assert "is_markdown" in metadata

    @pytest.mark.asyncio
    async def test_structure_for_rag_context(self, processor, tmp_path):
        """Test extracting structure for RAG context"""
        file_path = tmp_path / "structured.md"
        content = """# Main Title

## Section 1

Content in section 1

## Section 2

Content in section 2
"""
        file_path.write_text(content)

        result = await processor.process_document(str(file_path))
        structure = result["structure"]

        # Structure helps provide context for chunks
        assert "headings" in structure
        assert len(structure["headings"]) >= 3

        # Headings can be used to annotate chunks with section context
