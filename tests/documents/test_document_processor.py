"""
Tests for DocumentProcessor

Tests the main document processing factory class including:
- Format detection
- Document processing pipeline
- Text extraction
- Metadata extraction
- Structure extraction
- Text chunking

NOTE: Requires PyMuPDF package. Tests are skipped if not available.
"""

import pytest
import os
import tempfile
from pathlib import Path

# Check if PyMuPDF is available
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")

from app.documents import DocumentProcessor, DocumentConfig
from app.documents.document_processor import (
    DocumentProcessorError,
    UnsupportedFormatError
)


class TestDocumentProcessor:
    """Test DocumentProcessor class"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        config = DocumentConfig(
            chunk_size=100,
            chunk_overlap=20,
            max_file_size=1024 * 1024,  # 1MB for tests
        )
        return DocumentProcessor(config)

    @pytest.fixture
    def sample_text_file(self, tmp_path):
        """Create sample text file"""
        file_path = tmp_path / "sample.txt"
        content = "This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some content."
        file_path.write_text(content)
        return str(file_path)

    @pytest.fixture
    def sample_markdown_file(self, tmp_path):
        """Create sample markdown file"""
        file_path = tmp_path / "sample.md"
        content = """# Main Title

## Section 1

This is some content.

- Item 1
- Item 2
- Item 3

## Section 2

```python
def hello():
    print("world")
```

[Link](https://example.com)
"""
        file_path.write_text(content)
        return str(file_path)

    def test_init(self, processor):
        """Test processor initialization"""
        assert processor is not None
        assert processor.config is not None
        assert len(processor.parsers) == 4

    def test_format_detection(self, processor):
        """Test format detection"""
        assert processor._detect_format("test.pdf") == "pdf"
        assert processor._detect_format("test.docx") == "docx"
        assert processor._detect_format("test.txt") == "txt"
        assert processor._detect_format("test.md") == "md"
        assert processor._detect_format("test.markdown") == "md"

    def test_is_format_supported(self, processor):
        """Test format support checking"""
        assert processor.is_format_supported("test.pdf") is True
        assert processor.is_format_supported("test.docx") is True
        assert processor.is_format_supported("test.txt") is True
        assert processor.is_format_supported("test.md") is True
        assert processor.is_format_supported("test.xyz") is False

    def test_get_supported_formats(self, processor):
        """Test getting supported formats"""
        formats = processor.get_supported_formats()
        assert isinstance(formats, list)
        assert "pdf" in formats
        assert "docx" in formats
        assert "txt" in formats
        assert "md" in formats

    @pytest.mark.asyncio
    async def test_process_text_file(self, processor, sample_text_file):
        """Test processing text file"""
        result = await processor.process_document(sample_text_file)

        assert result is not None
        assert "text" in result
        assert "metadata" in result
        assert "chunks" in result
        assert result["format"] == "txt"
        assert len(result["text"]) > 0
        assert len(result["chunks"]) > 0

    @pytest.mark.asyncio
    async def test_process_markdown_file(self, processor, sample_markdown_file):
        """Test processing markdown file"""
        result = await processor.process_document(sample_markdown_file)

        assert result is not None
        assert result["format"] == "md"
        assert "text" in result
        assert "structure" in result
        assert len(result["text"]) > 0

        # Check structure extraction
        structure = result["structure"]
        assert "headings" in structure
        assert len(structure["headings"]) > 0

    @pytest.mark.asyncio
    async def test_extract_text_only(self, processor, sample_text_file):
        """Test extracting only text"""
        text = await processor.extract_text(sample_text_file)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "test document" in text.lower()

    @pytest.mark.asyncio
    async def test_extract_metadata_only(self, processor, sample_text_file):
        """Test extracting only metadata"""
        metadata = await processor.extract_metadata(sample_text_file)

        assert isinstance(metadata, dict)
        assert "format" in metadata
        assert "file_name" in metadata
        assert metadata["format"] == "txt"

    @pytest.mark.asyncio
    async def test_extract_structure_only(self, processor, sample_markdown_file):
        """Test extracting only structure"""
        structure = await processor.extract_structure(sample_markdown_file)

        assert isinstance(structure, dict)
        assert "headings" in structure

    def test_chunk_text(self, processor):
        """Test text chunking"""
        text = " ".join([f"Word{i}" for i in range(200)])

        chunks = processor.chunk_text(text, chunk_size=50, overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, processor):
        """Test processing nonexistent file"""
        with pytest.raises(FileNotFoundError):
            await processor.process_document("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_process_unsupported_format(self, processor, tmp_path):
        """Test processing unsupported format"""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content")

        with pytest.raises(UnsupportedFormatError):
            await processor.process_document(str(file_path))

    @pytest.mark.asyncio
    async def test_process_file_too_large(self, tmp_path):
        """Test processing file exceeding size limit"""
        # Create processor with tiny file size limit
        config = DocumentConfig(max_file_size=100)  # 100 bytes
        processor = DocumentProcessor(config)

        # Create large file
        file_path = tmp_path / "large.txt"
        file_path.write_text("x" * 1000)  # 1000 bytes

        with pytest.raises(DocumentProcessorError):
            await processor.process_document(str(file_path))

    @pytest.mark.asyncio
    async def test_process_without_chunking(self, processor, sample_text_file):
        """Test processing without RAG chunking"""
        result = await processor.process_document(
            sample_text_file,
            chunk_for_rag=False
        )

        assert result is not None
        assert "chunks" in result
        assert len(result["chunks"]) == 0

    @pytest.mark.asyncio
    async def test_processing_metrics(self, processor, sample_text_file):
        """Test that processing metrics are included"""
        result = await processor.process_document(sample_text_file)

        assert "processed_at" in result
        assert "processing_time_seconds" in result
        assert "num_chunks" in result
        assert "text_length" in result
        assert result["processing_time_seconds"] >= 0


class TestDocumentProcessorIntegration:
    """Integration tests for document processor"""

    @pytest.fixture
    def processor(self):
        """Create processor with default config"""
        return DocumentProcessor()

    @pytest.mark.asyncio
    async def test_multiple_document_processing(self, processor, tmp_path):
        """Test processing multiple documents"""
        # Create multiple files
        files = []

        for i in range(3):
            file_path = tmp_path / f"doc{i}.txt"
            file_path.write_text(f"Document {i} content")
            files.append(str(file_path))

        # Process all files
        results = []
        for file_path in files:
            result = await processor.process_document(file_path)
            results.append(result)

        assert len(results) == 3
        assert all(r["format"] == "txt" for r in results)

    @pytest.mark.asyncio
    async def test_markdown_features_extraction(self, processor, tmp_path):
        """Test extracting various Markdown features"""
        file_path = tmp_path / "features.md"
        content = """# Title

## Heading 2

### Heading 3

Regular paragraph.

- List item 1
- List item 2

1. Numbered 1
2. Numbered 2

```python
code here
```

[Link text](https://example.com)

> Blockquote
"""
        file_path.write_text(content)

        result = await processor.process_document(str(file_path))
        structure = result["structure"]

        # Verify headings extracted
        assert len(structure["headings"]) >= 3

        # Verify code blocks extracted
        if "code_blocks" in structure:
            assert len(structure["code_blocks"]) >= 1

        # Verify links extracted
        if "links" in structure:
            assert len(structure["links"]) >= 1
