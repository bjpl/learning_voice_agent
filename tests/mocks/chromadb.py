"""
Mock implementation of chromadb library for testing

Provides minimal mock classes needed for tests to import successfully.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import numpy as np


# Mock config module
class config:
    """Mock config module"""

    class Settings:
        """Mock Settings class"""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class Collection:
    """Mock ChromaDB Collection"""

    def __init__(self, name: str):
        self.name = name
        self._data = []
        self._count = 100  # Mock default count

    def add(self, ids: List[str], embeddings: List, documents: List[str],
            metadatas: Optional[List[Dict]] = None):
        """Mock add method"""
        for i, doc_id in enumerate(ids):
            self._data.append({
                'id': doc_id,
                'embedding': embeddings[i] if embeddings else None,
                'document': documents[i],
                'metadata': metadatas[i] if metadatas else {}
            })

    def query(self, query_embeddings: List = None, query_texts: List[str] = None,
              n_results: int = 10, where: Optional[Dict] = None):
        """Mock query method"""
        # Return mock results
        return {
            'ids': [['id1', 'id2', 'id3']],
            'distances': [[0.1, 0.2, 0.3]],
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[{'session_id': 'test'}, {'session_id': 'test'}, {'session_id': 'test'}]]
        }

    def delete(self, ids: List[str]):
        """Mock delete method"""
        self._data = [d for d in self._data if d['id'] not in ids]

    def count(self):
        """Mock count method"""
        return self._count

    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None):
        """Mock get method"""
        return {
            'ids': [],
            'documents': [],
            'metadatas': []
        }


class Client:
    """Mock ChromaDB Client"""

    def __init__(self):
        self._collections = {}

    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Mock get_or_create_collection"""
        if name not in self._collections:
            self._collections[name] = Collection(name)
        return self._collections[name]

    def get_collection(self, name: str):
        """Mock get_collection"""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def delete_collection(self, name: str):
        """Mock delete_collection"""
        if name in self._collections:
            del self._collections[name]

    def list_collections(self):
        """Mock list_collections"""
        return [{'name': name} for name in self._collections.keys()]


class PersistentClient(Client):
    """Mock PersistentClient"""

    def __init__(self, path: str = None):
        super().__init__()
        self.path = path


class EphemeralClient(Client):
    """Mock EphemeralClient"""
    pass


class Settings:
    """Mock Settings class"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Mock embedding functions
class EmbeddingFunction:
    """Mock embedding function base class"""

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings"""
        return [np.random.rand(384).tolist() for _ in texts]


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Mock sentence transformer embedding function"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
