"""
Vector Store Factory - Backend Selection and A/B Testing
PATTERN: Factory pattern with feature flag support
WHY: Enable seamless switching and A/B testing between vector backends

SPECIFICATION:
- Support multiple backends: chromadb, ruvector, auto
- Enable A/B testing for gradual rollout
- Provide dual-write mode for migration safety
- Maintain single interface for consumers
"""
import random
import logging
from typing import Optional, Union
from enum import Enum

from app.config import settings
from app.vector.protocol import VectorStoreProtocol, LearningVectorStoreProtocol

logger = logging.getLogger(__name__)


class VectorBackend(str, Enum):
    """Supported vector store backends."""
    CHROMADB = "chromadb"
    RUVECTOR = "ruvector"
    AUTO = "auto"


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    CONCEPT: Dependency injection for vector storage
    WHY: Allow runtime backend selection and A/B testing
    PATTERN: Factory with singleton caching

    Usage:
        # Auto-select best available backend
        store = VectorStoreFactory.create()

        # Force specific backend
        store = VectorStoreFactory.create(backend="ruvector")

        # A/B testing (uses config percentage)
        store = VectorStoreFactory.create_with_ab_test(session_id="user123")
    """

    _instances: dict = {}

    @classmethod
    def create(
        cls,
        backend: Union[str, VectorBackend] = "auto",
        force_new: bool = False
    ) -> VectorStoreProtocol:
        """
        Create or retrieve a vector store instance.

        Args:
            backend: Backend type ("chromadb", "ruvector", "auto")
            force_new: Create new instance even if cached

        Returns:
            VectorStoreProtocol implementation

        Raises:
            ValueError: If backend is invalid or unavailable
        """
        backend_str = backend.value if isinstance(backend, VectorBackend) else backend

        # Check cache unless force_new
        if not force_new and backend_str in cls._instances:
            return cls._instances[backend_str]

        # Resolve "auto" to best available backend
        if backend_str == "auto":
            backend_str = cls._resolve_auto_backend()

        # Create the appropriate store
        store = cls._create_backend(backend_str)

        # Cache the instance
        cls._instances[backend_str] = store

        logger.info(f"Created vector store: {backend_str}")
        return store

    @classmethod
    def _resolve_auto_backend(cls) -> str:
        """
        Determine best available backend.

        PATTERN: Prefer RuVector if available, fallback to ChromaDB
        WHY: RuVector offers superior features (learning, speed)
        """
        # Check if RuVector is available
        try:
            from app.vector.ruvector_store import RUVECTOR_AVAILABLE
            if RUVECTOR_AVAILABLE:
                logger.debug("Auto-selected: ruvector (available)")
                return "ruvector"
        except ImportError:
            pass

        # Check if ChromaDB is available
        try:
            from app.vector_store import CHROMADB_AVAILABLE
            if CHROMADB_AVAILABLE:
                logger.debug("Auto-selected: chromadb (ruvector unavailable)")
                return "chromadb"
        except ImportError:
            pass

        # Neither available - return ruvector and let it gracefully degrade
        logger.warning("No vector backend available. Using ruvector with degraded mode.")
        return "ruvector"

    @classmethod
    def _create_backend(cls, backend: str) -> VectorStoreProtocol:
        """Create a specific backend instance."""
        if backend == "ruvector":
            from app.vector.ruvector_store import RuVectorStore
            return RuVectorStore()

        elif backend == "chromadb":
            from app.vector_store import VectorStore
            return VectorStore()

        else:
            raise ValueError(f"Unknown vector backend: {backend}")

    @classmethod
    def create_with_ab_test(
        cls,
        session_id: Optional[str] = None
    ) -> VectorStoreProtocol:
        """
        Create vector store with A/B testing.

        CONCEPT: Gradual rollout of RuVector
        WHY: Validate performance before full migration
        PATTERN: Deterministic assignment based on session ID

        Args:
            session_id: Session ID for deterministic assignment

        Returns:
            VectorStoreProtocol (either ChromaDB or RuVector based on test)
        """
        if not settings.vector_ab_test_enabled:
            return cls.create(backend=settings.vector_backend)

        # Deterministic assignment based on session ID
        if session_id:
            # Use hash for consistent assignment
            hash_value = hash(session_id) % 100
            use_ruvector = hash_value < settings.vector_ab_test_ruvector_percentage
        else:
            # Random assignment
            use_ruvector = random.randint(0, 99) < settings.vector_ab_test_ruvector_percentage

        backend = "ruvector" if use_ruvector else "chromadb"
        logger.debug(f"A/B test assigned: {backend} (session={session_id})")

        return cls.create(backend=backend)

    @classmethod
    def get_learning_store(cls) -> Optional[LearningVectorStoreProtocol]:
        """
        Get a vector store with learning capabilities.

        Returns:
            LearningVectorStoreProtocol if available, None otherwise
        """
        store = cls.create(backend="ruvector")

        if isinstance(store, LearningVectorStoreProtocol):
            return store

        logger.warning("Learning vector store not available")
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached instances (for testing)."""
        cls._instances.clear()


class DualWriteVectorStore:
    """
    Dual-write vector store for safe migration.

    CONCEPT: Write to both backends during migration
    WHY: Enable validation and safe rollback
    PATTERN: Decorator/proxy for dual operations

    Usage:
        dual_store = DualWriteVectorStore(
            primary=ruvector_store,
            secondary=chromadb_store
        )
        await dual_store.add_conversation(...)  # Writes to both
    """

    def __init__(
        self,
        primary: VectorStoreProtocol,
        secondary: VectorStoreProtocol,
        read_from: str = "primary"
    ):
        """
        Initialize dual-write store.

        Args:
            primary: Primary backend (usually RuVector)
            secondary: Secondary backend (usually ChromaDB)
            read_from: Which backend to read from ("primary" or "secondary")
        """
        self.primary = primary
        self.secondary = secondary
        self.read_from = read_from
        self._write_errors: list = []

    async def initialize(self) -> bool:
        """Initialize both backends."""
        primary_ok = await self.primary.initialize()
        secondary_ok = await self.secondary.initialize()
        return primary_ok and secondary_ok

    async def add_conversation(
        self,
        conversation_id: str,
        user_text: str,
        agent_text: str,
        session_id: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """Add conversation to both backends."""
        primary_ok = await self.primary.add_conversation(
            conversation_id, user_text, agent_text, session_id, metadata
        )

        secondary_ok = await self.secondary.add_conversation(
            conversation_id, user_text, agent_text, session_id, metadata
        )

        if not secondary_ok:
            self._write_errors.append({
                "id": conversation_id,
                "backend": "secondary",
                "error": "write failed"
            })

        return primary_ok  # Primary success is what matters

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        session_filter: Optional[str] = None
    ) -> list:
        """Search from configured read backend."""
        store = self.primary if self.read_from == "primary" else self.secondary
        return await store.semantic_search(query, limit, similarity_threshold, session_filter)

    async def find_similar_conversations(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> list:
        """Find similar from configured read backend."""
        store = self.primary if self.read_from == "primary" else self.secondary
        return await store.find_similar_conversations(conversation_id, limit)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete from both backends."""
        primary_ok = await self.primary.delete_conversation(conversation_id)
        secondary_ok = await self.secondary.delete_conversation(conversation_id)
        return primary_ok and secondary_ok

    async def get_stats(self) -> dict:
        """Get stats from both backends."""
        primary_stats = await self.primary.get_stats()
        secondary_stats = await self.secondary.get_stats()

        return {
            "mode": "dual_write",
            "read_from": self.read_from,
            "primary": primary_stats,
            "secondary": secondary_stats,
            "write_errors": len(self._write_errors)
        }

    async def close(self) -> None:
        """Close both backends."""
        await self.primary.close()
        await self.secondary.close()

    async def validate_consistency(self, sample_size: int = 100) -> dict:
        """
        Validate data consistency between backends.

        CONCEPT: Migration validation
        WHY: Ensure both backends have matching data before cutover
        """
        primary_stats = await self.primary.get_stats()
        secondary_stats = await self.secondary.get_stats()

        return {
            "primary_count": primary_stats.get("count", 0),
            "secondary_count": secondary_stats.get("count", 0),
            "write_errors": self._write_errors[-sample_size:],
            "consistent": primary_stats.get("count", 0) == secondary_stats.get("count", 0)
        }
