"""
Document Processing Module

This module provides comprehensive document processing capabilities for:
- PDF files (using PyMuPDF/fitz)
- DOCX files (using python-docx)
- TXT files (plain text)
- Markdown files (with structure extraction)

Main Components:
- DocumentProcessor: Factory class for processing any document format
- PDFParser: Specialized PDF processing
- DOCXParser: Specialized DOCX processing
- TextParser: Text and Markdown processing

Usage:
    from app.documents import DocumentProcessor

    processor = DocumentProcessor()
    result = await processor.process_document("/path/to/document.pdf")

    # Access extracted content
    text = result["text"]
    metadata = result["metadata"]
    chunks = result["chunks"]  # Pre-chunked for RAG
"""

from app.documents.document_processor import DocumentProcessor
from app.documents.pdf_parser import PDFParser
from app.documents.docx_parser import DOCXParser
from app.documents.text_parser import TextParser
from app.documents.config import DocumentConfig

__all__ = [
    "DocumentProcessor",
    "PDFParser",
    "DOCXParser",
    "TextParser",
    "DocumentConfig",
]

__version__ = "1.0.0"
