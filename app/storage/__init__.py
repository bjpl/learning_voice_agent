"""
Storage Module

Provides storage and retrieval capabilities:
- Vector storage (ChromaDB)
- Multi-modal file storage
- Metadata management
- File indexing
"""
from app.storage.chroma_db import ChromaDBStorage
from app.storage.file_manager import FileManager, file_manager
from app.storage.metadata_store import MetadataStore, metadata_store
from app.storage.indexer import MultiModalIndexer, multimodal_indexer
from app.storage.config import StorageConfig, storage_config

__all__ = [
    "ChromaDBStorage",
    "FileManager",
    "file_manager",
    "MetadataStore",
    "metadata_store",
    "MultiModalIndexer",
    "multimodal_indexer",
    "StorageConfig",
    "storage_config",
]
