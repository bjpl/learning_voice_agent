"""
Document Processor Module

Main factory class for processing various document formats.

This module provides a unified interface for processing:
- PDF files
- DOCX files
- Plain text files
- Markdown files

Features:
- Automatic format detection
- Text extraction
- Metadata extraction
- Structure extraction
- Text chunking for RAG
- Parallel processing support

Usage:
    processor = DocumentProcessor()
    result = await processor.process_document("/path/to/file.pdf")

    # Access results
    text = result["text"]
    chunks = result["chunks"]
    metadata = result["metadata"]
"""

from typing import Dict, List, Optional, Any
import os
import asyncio
from pathlib import Path
from datetime import datetime

from app.documents.config import DocumentConfig
from app.documents.pdf_parser import PDFParser, PDFParserError
from app.documents.docx_parser import DOCXParser, DOCXParserError
from app.documents.text_parser import TextParser, TextParserError
from app.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessorError(Exception):
    """Base exception for document processor errors"""
    pass


class UnsupportedFormatError(DocumentProcessorError):
    """Raised when document format is not supported"""
    pass


class DocumentProcessor:
    """
    Factory class for processing various document formats

    Automatically detects document format and routes to appropriate parser.
    Provides unified interface for text extraction, metadata extraction,
    and text chunking.
    """

    def __init__(self, config: Optional[DocumentConfig] = None):
        """
        Initialize document processor

        Args:
            config: Document processing configuration
        """
        self.config = config or DocumentConfig()
        self.logger = logger

        # Initialize parsers
        self.parsers = {
            "pdf": PDFParser(self.config),
            "docx": DOCXParser(self.config),
            "txt": TextParser(self.config),
            "md": TextParser(self.config),
        }

    async def process_document(
        self,
        file_path: str,
        password: Optional[str] = None,
        chunk_for_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Process document and extract all content

        This is the main entry point for document processing.
        It extracts text, metadata, structure, and optionally chunks
        the text for RAG systems.

        Args:
            file_path: Path to document file
            password: Optional password for encrypted PDFs
            chunk_for_rag: Whether to chunk text for RAG (default True)

        Returns:
            Dictionary containing:
                - text: Full extracted text
                - metadata: Document metadata
                - structure: Document structure (headings, TOC, etc.)
                - chunks: Text chunks for RAG (if chunk_for_rag=True)
                - format: Document format
                - processed_at: Processing timestamp

        Raises:
            UnsupportedFormatError: If format is not supported
            DocumentProcessorError: If processing fails
        """
        start_time = datetime.now()
        self.logger.info(f"Processing document: {file_path}")

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Validate file size
        if not self.config.validate_file_size(file_path):
            max_mb = self.config.max_file_size / (1024 * 1024)
            raise DocumentProcessorError(
                f"File size exceeds maximum of {max_mb:.1f}MB"
            )

        # Detect format
        doc_format = self._detect_format(file_path)

        if not self.config.is_format_supported(doc_format):
            raise UnsupportedFormatError(
                f"Format '{doc_format}' is not supported. "
                f"Supported formats: {self.config.supported_formats}"
            )

        # Get appropriate parser
        parser = self.parsers.get(doc_format)

        if not parser:
            raise UnsupportedFormatError(f"No parser available for format: {doc_format}")

        try:
            # Extract content in parallel
            extraction_tasks = [
                self._extract_text_safe(parser, file_path, password),
                self._extract_metadata_safe(parser, file_path, password),
                self._extract_structure_safe(parser, file_path, password),
            ]

            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

            text = results[0] if not isinstance(results[0], Exception) else ""
            metadata = results[1] if not isinstance(results[1], Exception) else {}
            structure = results[2] if not isinstance(results[2], Exception) else {}

            # Handle extraction errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(
                        f"Extraction {i} failed: {str(result)}"
                    )

            if not text:
                raise DocumentProcessorError("Failed to extract text from document")

            # Chunk text for RAG if requested
            chunks = []
            if chunk_for_rag:
                chunks = self._chunk_text(
                    text,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "text": text,
                "metadata": metadata,
                "structure": structure,
                "chunks": chunks,
                "format": doc_format,
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "processed_at": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "num_chunks": len(chunks),
                "text_length": len(text),
            }

            self.logger.info(
                f"Successfully processed {doc_format} document in {processing_time:.2f}s. "
                f"Extracted {len(text)} characters in {len(chunks)} chunks."
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise DocumentProcessorError(f"Failed to process document: {str(e)}")

    async def extract_text(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> str:
        """
        Extract only text from document

        Args:
            file_path: Path to document file
            password: Optional password for encrypted PDFs

        Returns:
            Extracted text
        """
        doc_format = self._detect_format(file_path)
        parser = self._get_parser(doc_format)

        return await self._extract_text_safe(parser, file_path, password)

    async def extract_metadata(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract only metadata from document

        Args:
            file_path: Path to document file
            password: Optional password for encrypted PDFs

        Returns:
            Document metadata
        """
        doc_format = self._detect_format(file_path)
        parser = self._get_parser(doc_format)

        return await self._extract_metadata_safe(parser, file_path, password)

    async def extract_structure(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract only structure from document

        Args:
            file_path: Path to document file
            password: Optional password for encrypted PDFs

        Returns:
            Document structure
        """
        doc_format = self._detect_format(file_path)
        parser = self._get_parser(doc_format)

        return await self._extract_structure_safe(parser, file_path, password)

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text for RAG systems

        Args:
            text: Text to chunk
            chunk_size: Chunk size in tokens (uses config default if None)
            overlap: Overlap size in tokens (uses config default if None)

        Returns:
            List of chunk dicts with text and metadata
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        return self._chunk_text(text, chunk_size, overlap)

    def _detect_format(self, file_path: str) -> str:
        """
        Detect document format from file extension

        Args:
            file_path: Path to file

        Returns:
            Format string (pdf, docx, txt, md)
        """
        extension = os.path.splitext(file_path)[1].lower().lstrip('.')

        # Map extensions to formats
        format_map = {
            "pdf": "pdf",
            "docx": "docx",
            "doc": "docx",  # Treat old .doc as docx
            "txt": "txt",
            "md": "md",
            "markdown": "md",
        }

        return format_map.get(extension, extension)

    def _get_parser(self, doc_format: str):
        """
        Get parser for document format

        Args:
            doc_format: Document format

        Returns:
            Parser instance

        Raises:
            UnsupportedFormatError: If format not supported
        """
        parser = self.parsers.get(doc_format)

        if not parser:
            raise UnsupportedFormatError(
                f"Format '{doc_format}' not supported. "
                f"Supported: {list(self.parsers.keys())}"
            )

        return parser

    async def _extract_text_safe(
        self,
        parser,
        file_path: str,
        password: Optional[str] = None
    ) -> str:
        """
        Safely extract text with error handling

        Args:
            parser: Parser instance
            file_path: Path to file
            password: Optional password

        Returns:
            Extracted text or empty string on error
        """
        try:
            # PDF parser accepts password, others don't
            if isinstance(parser, PDFParser):
                return await parser.extract_text(file_path, password)
            else:
                return await parser.extract_text(file_path)
        except Exception as e:
            self.logger.error(f"Text extraction failed: {str(e)}")
            return ""

    async def _extract_metadata_safe(
        self,
        parser,
        file_path: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Safely extract metadata with error handling

        Args:
            parser: Parser instance
            file_path: Path to file
            password: Optional password

        Returns:
            Metadata dict or empty dict on error
        """
        try:
            if isinstance(parser, PDFParser):
                return await parser.extract_metadata(file_path, password)
            else:
                return await parser.extract_metadata(file_path)
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}")
            return {}

    async def _extract_structure_safe(
        self,
        parser,
        file_path: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Safely extract structure with error handling

        Args:
            parser: Parser instance
            file_path: Path to file
            password: Optional password

        Returns:
            Structure dict or empty dict on error
        """
        try:
            if isinstance(parser, PDFParser):
                return await parser.extract_structure(file_path, password)
            else:
                return await parser.extract_structure(file_path)
        except Exception as e:
            self.logger.error(f"Structure extraction failed: {str(e)}")
            return {}

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Chunk text with overlap for RAG systems

        Uses approximate token-based chunking (words as proxy for tokens).
        For production use, consider using tiktoken for accurate token counting.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens

        Returns:
            List of chunk dicts with text, index, and metadata
        """
        # Simple word-based chunking (approximate tokens)
        words = text.split()

        # Adjust for word-to-token ratio (roughly 1.3 tokens per word)
        words_per_chunk = int(chunk_size / 1.3)
        words_overlap = int(overlap / 1.3)

        chunks = []
        start = 0

        while start < len(words):
            end = start + words_per_chunk

            # Get chunk words
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk_text,
                "start_word": start,
                "end_word": end,
                "word_count": len(chunk_words),
                "char_count": len(chunk_text),
            })

            # Move to next chunk with overlap
            start = end - words_overlap

            # Avoid infinite loop if overlap is too large
            if start <= chunks[-1]["start_word"] and len(chunks) > 1:
                break

        self.logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks

    def is_format_supported(self, file_path: str) -> bool:
        """
        Check if file format is supported

        Args:
            file_path: Path to file

        Returns:
            True if format is supported
        """
        doc_format = self._detect_format(file_path)
        return self.config.is_format_supported(doc_format)

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported formats

        Returns:
            List of supported format strings
        """
        return sorted(list(self.config.supported_formats))
