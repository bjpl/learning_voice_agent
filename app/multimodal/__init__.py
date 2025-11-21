"""
Multi-Modal Processing Module

SPECIFICATION:
- Image analysis with Claude Vision
- Document processing (PDF, DOCX, TXT, MD)
- File management and storage
- Metadata persistence
- Vector indexing for multimodal content

ARCHITECTURE:
- file_manager: Upload and storage handling
- vision_analyzer: Image analysis with Claude Vision API
- document_processor: Text extraction from documents
- metadata_store: File metadata persistence
- multimodal_indexer: Vector embeddings for images/documents

PATTERN: Modular service layer with clear separation of concerns
WHY: Each component has single responsibility for maintainability
"""

from app.multimodal.file_manager import FileManager, file_manager
from app.multimodal.vision_analyzer import VisionAnalyzer, vision_analyzer
from app.multimodal.document_processor import DocumentProcessor, document_processor
from app.multimodal.metadata_store import MetadataStore, metadata_store
from app.multimodal.multimodal_indexer import MultiModalIndexer, multimodal_indexer

__all__ = [
    'FileManager',
    'file_manager',
    'VisionAnalyzer',
    'vision_analyzer',
    'DocumentProcessor',
    'document_processor',
    'MetadataStore',
    'metadata_store',
    'MultiModalIndexer',
    'multimodal_indexer',
]
