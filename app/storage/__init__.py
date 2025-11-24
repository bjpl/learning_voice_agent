"""
Storage Module

Provides storage and retrieval capabilities:
- Base store abstraction for SQLite persistence
- Vector storage (ChromaDB)
- Multi-modal file storage
- Metadata management
- File indexing
"""
from app.storage.base_store import (
    BaseStore,
    SQLiteStore,
    StoreError,
    ConnectionError,
    TransactionError,
    ValidationError,
)
from app.storage.chroma_db import ChromaDBStorage
from app.storage.file_manager import FileManager, file_manager
from app.storage.metadata_store import MetadataStore, metadata_store
from app.storage.indexer import MultiModalIndexer, multimodal_indexer
from app.storage.config import StorageConfig, storage_config

__all__ = [
    # Base abstractions
    "BaseStore",
    "SQLiteStore",
    "StoreError",
    "ConnectionError",
    "TransactionError",
    "ValidationError",
    # Storage implementations
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
