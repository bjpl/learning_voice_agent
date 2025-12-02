"""
Mock RuVector module for testing
PATTERN: Mock implementation matching RuVector API
WHY: Enable testing without RuVector dependency installed
"""
from typing import List, Dict, Optional, Any
from unittest.mock import MagicMock
import numpy as np


class MockVectorDB:
    """
    Mock implementation of RuVector VectorDB.

    Provides in-memory vector storage for testing purposes.
    Matches the RuVector API for seamless test integration.
    """

    def __init__(self, dim: int = 384, persist_path: str = None):
        self.dim = dim
        self.persist_path = persist_path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._gnn_enabled = False
        self._compression_tiers = []
        self._training_signals: List[Dict] = []

    def insert(self, id: str, embedding: List[float], metadata: Dict = None) -> None:
        """Insert a vector with metadata."""
        self._data[id] = {
            "id": id,
            "embedding": np.array(embedding),
            "metadata": metadata or {}
        }

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Dict = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self._data:
            return []

        query = np.array(query_embedding)
        results = []

        for id, item in self._data.items():
            # Apply filter if provided
            if filter:
                metadata = item.get("metadata", {})
                match = all(metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue

            # Calculate cosine similarity
            embedding = item["embedding"]
            similarity = np.dot(query, embedding) / (
                np.linalg.norm(query) * np.linalg.norm(embedding) + 1e-8
            )

            results.append({
                "id": id,
                "similarity": float(similarity),
                "score": float(similarity),
                "metadata": item.get("metadata", {})
            })

        # Sort by similarity and return top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID."""
        if id in self._data:
            item = self._data[id]
            return {
                "id": id,
                "embedding": item["embedding"].tolist(),
                "metadata": item.get("metadata", {})
            }
        return None

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        if id in self._data:
            del self._data[id]
            return True
        return False

    def count(self) -> int:
        """Return number of vectors stored."""
        return len(self._data)

    def configure_gnn(
        self,
        enabled: bool = True,
        attention_heads: int = 8,
        learning_rate: float = 0.001
    ) -> None:
        """Configure GNN settings."""
        self._gnn_enabled = enabled
        self._attention_heads = attention_heads
        self._learning_rate = learning_rate

    def configure_compression(self, tiers: List[Dict]) -> None:
        """Configure compression tiers."""
        self._compression_tiers = tiers

    def train_positive(self, id: str, weight: float = 1.0) -> None:
        """Record positive training signal."""
        self._training_signals.append({
            "type": "positive",
            "id": id,
            "weight": weight
        })

    def train_negative(self, id: str, weight: float = 1.0) -> None:
        """Record negative training signal."""
        self._training_signals.append({
            "type": "negative",
            "id": id,
            "weight": weight
        })

    def execute(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query (mock implementation)."""
        # Basic mock - just return empty for now
        return []

    def hybrid_search(
        self,
        embedding: List[float],
        cypher: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """Hybrid vector + graph search (mock)."""
        # Fall back to vector search for mock
        return self.search(embedding, k=limit)

    def close(self) -> None:
        """Close the database."""
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        positive = sum(1 for s in self._training_signals if s["type"] == "positive")
        negative = sum(1 for s in self._training_signals if s["type"] == "negative")
        return {
            "positive_count": positive,
            "negative_count": negative,
            "total_signals": len(self._training_signals)
        }


class MockAgentRouter:
    """Mock implementation of RuVector AgentRouter."""

    def __init__(self):
        self._agents: Dict[str, Dict] = {}

    def add_agent(
        self,
        name: str,
        embedding: List[float],
        capabilities: List[str] = None
    ) -> None:
        """Register an agent."""
        self._agents[name] = {
            "embedding": np.array(embedding),
            "capabilities": capabilities or []
        }

    def route(self, query_embedding: List[float]) -> str:
        """Route query to best agent."""
        if not self._agents:
            return None

        query = np.array(query_embedding)
        best_agent = None
        best_similarity = -1

        for name, agent in self._agents.items():
            similarity = np.dot(query, agent["embedding"]) / (
                np.linalg.norm(query) * np.linalg.norm(agent["embedding"]) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_agent = name

        return best_agent


# Module-level exports matching ruvector package
VectorDB = MockVectorDB
AgentRouter = MockAgentRouter


def create_mock_ruvector_module():
    """Create a mock module that can be used with patch.dict('sys.modules')."""
    mock_module = MagicMock()
    mock_module.VectorDB = MockVectorDB
    mock_module.AgentRouter = MockAgentRouter
    return mock_module
