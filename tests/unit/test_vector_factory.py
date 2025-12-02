"""
Unit Tests for VectorStoreFactory
PATTERN: TDD for factory and A/B testing logic
WHY: Ensure correct backend selection and migration support
"""
import pytest
from unittest.mock import patch, MagicMock


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    def test_create_chromadb_backend(self):
        """Test creating ChromaDB backend explicitly."""
        with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
            mock_store = MagicMock()
            mock_create.return_value = mock_store

            from app.vector.factory import VectorStoreFactory
            VectorStoreFactory.clear_cache()

            store = VectorStoreFactory.create(backend="chromadb")

            mock_create.assert_called_with("chromadb")
            assert store == mock_store

    def test_create_ruvector_backend(self):
        """Test creating RuVector backend explicitly."""
        with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
            mock_store = MagicMock()
            mock_create.return_value = mock_store

            from app.vector.factory import VectorStoreFactory
            VectorStoreFactory.clear_cache()

            store = VectorStoreFactory.create(backend="ruvector")

            mock_create.assert_called_with("ruvector")
            assert store == mock_store

    def test_auto_selection_prefers_ruvector(self):
        """Test auto selection prefers RuVector when available."""
        with patch('app.vector.ruvector_store.RUVECTOR_AVAILABLE', True):
            with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
                mock_store = MagicMock()
                mock_create.return_value = mock_store

                from app.vector.factory import VectorStoreFactory
                VectorStoreFactory.clear_cache()

                store = VectorStoreFactory.create(backend="auto")

                mock_create.assert_called_with("ruvector")

    def test_auto_fallback_to_chromadb(self):
        """Test auto selection falls back to ChromaDB when RuVector unavailable."""
        with patch('app.vector.ruvector_store.RUVECTOR_AVAILABLE', False):
            with patch('app.vector_store.CHROMADB_AVAILABLE', True):
                with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
                    mock_store = MagicMock()
                    mock_create.return_value = mock_store

                    from app.vector.factory import VectorStoreFactory
                    VectorStoreFactory.clear_cache()

                    store = VectorStoreFactory.create(backend="auto")

                    mock_create.assert_called_with("chromadb")

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        from app.vector.factory import VectorStoreFactory
        VectorStoreFactory.clear_cache()

        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory._create_backend("invalid_backend")

        assert "Unknown vector backend" in str(exc_info.value)

    def test_cache_returns_same_instance(self):
        """Test that factory caches instances."""
        with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
            mock_store = MagicMock()
            mock_create.return_value = mock_store

            from app.vector.factory import VectorStoreFactory
            VectorStoreFactory.clear_cache()

            store1 = VectorStoreFactory.create(backend="ruvector")
            store2 = VectorStoreFactory.create(backend="ruvector")

            assert store1 is store2
            assert mock_create.call_count == 1

    def test_force_new_bypasses_cache(self):
        """Test force_new creates new instance."""
        with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
            mock_store1 = MagicMock()
            mock_store2 = MagicMock()
            mock_create.side_effect = [mock_store1, mock_store2]

            from app.vector.factory import VectorStoreFactory
            VectorStoreFactory.clear_cache()

            store1 = VectorStoreFactory.create(backend="ruvector")
            store2 = VectorStoreFactory.create(backend="ruvector", force_new=True)

            assert store1 is not store2
            assert mock_create.call_count == 2


class TestVectorStoreFactoryABTesting:
    """Tests for A/B testing functionality."""

    def test_ab_test_deterministic_assignment(self):
        """Test A/B test gives consistent assignment for same session."""
        with patch('app.config.settings') as mock_settings:
            mock_settings.vector_ab_test_enabled = True
            mock_settings.vector_ab_test_ruvector_percentage = 50
            mock_settings.vector_backend = "auto"

            with patch('app.vector.factory.VectorStoreFactory.create') as mock_create:
                mock_store = MagicMock()
                mock_create.return_value = mock_store

                from app.vector.factory import VectorStoreFactory

                # Same session should get same assignment
                session_id = "test_session_123"

                store1 = VectorStoreFactory.create_with_ab_test(session_id=session_id)
                store2 = VectorStoreFactory.create_with_ab_test(session_id=session_id)

                # Both calls should use same backend
                calls = mock_create.call_args_list
                assert calls[0] == calls[1]

    def test_ab_test_disabled_uses_config_backend(self):
        """Test A/B test disabled uses configured backend."""
        with patch('app.vector.factory.settings') as mock_settings:
            mock_settings.vector_ab_test_enabled = False
            mock_settings.vector_backend = "chromadb"

            with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
                mock_store = MagicMock()
                mock_create.return_value = mock_store

                from app.vector.factory import VectorStoreFactory
                VectorStoreFactory.clear_cache()

                VectorStoreFactory.create_with_ab_test(session_id="any_session")

                mock_create.assert_called_with("chromadb")

    def test_ab_test_percentage_zero_always_chromadb(self):
        """Test 0% RuVector means always ChromaDB."""
        with patch('app.vector.factory.settings') as mock_settings:
            mock_settings.vector_ab_test_enabled = True
            mock_settings.vector_ab_test_ruvector_percentage = 0

            with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
                mock_store = MagicMock()
                mock_create.return_value = mock_store

                from app.vector.factory import VectorStoreFactory
                VectorStoreFactory.clear_cache()

                # Test multiple sessions - each gets new instance due to cache clear
                backends_used = []
                for i in range(10):
                    VectorStoreFactory.clear_cache()  # Clear before each call
                    VectorStoreFactory.create_with_ab_test(session_id=f"session_{i}")
                    # Get the backend from the last call
                    if mock_create.call_args_list:
                        backends_used.append(mock_create.call_args_list[-1][0][0])

                # All should be chromadb (0% means no sessions get ruvector)
                for backend in backends_used:
                    assert backend == 'chromadb', f"Expected chromadb but got {backend}"

    def test_ab_test_percentage_hundred_always_ruvector(self):
        """Test 100% RuVector means always RuVector."""
        with patch('app.vector.factory.settings') as mock_settings:
            mock_settings.vector_ab_test_enabled = True
            mock_settings.vector_ab_test_ruvector_percentage = 100

            with patch('app.vector.factory.VectorStoreFactory._create_backend') as mock_create:
                mock_store = MagicMock()
                mock_create.return_value = mock_store

                from app.vector.factory import VectorStoreFactory
                VectorStoreFactory.clear_cache()

                # Test multiple sessions - each gets new instance due to cache clear
                backends_used = []
                for i in range(10):
                    VectorStoreFactory.clear_cache()  # Clear before each call
                    VectorStoreFactory.create_with_ab_test(session_id=f"session_{i}")
                    # Get the backend from the last call
                    if mock_create.call_args_list:
                        backends_used.append(mock_create.call_args_list[-1][0][0])

                # All should be ruvector (100% means all sessions get ruvector)
                for backend in backends_used:
                    assert backend == 'ruvector', f"Expected ruvector but got {backend}"


class TestDualWriteVectorStore:
    """Tests for DualWriteVectorStore migration helper."""

    @pytest.mark.asyncio
    async def test_dual_write_initializes_both(self):
        """Test dual write initializes both backends."""
        from app.vector.factory import DualWriteVectorStore

        primary = MagicMock()
        primary.initialize = MagicMock(return_value=True)
        secondary = MagicMock()
        secondary.initialize = MagicMock(return_value=True)

        # Make initialize awaitable
        async def async_init_primary():
            return True
        async def async_init_secondary():
            return True

        primary.initialize = async_init_primary
        secondary.initialize = async_init_secondary

        dual_store = DualWriteVectorStore(primary=primary, secondary=secondary)
        result = await dual_store.initialize()

        assert result is True

    @pytest.mark.asyncio
    async def test_dual_write_adds_to_both(self):
        """Test dual write adds to both backends."""
        from app.vector.factory import DualWriteVectorStore

        primary = MagicMock()
        secondary = MagicMock()

        async def async_add(*args, **kwargs):
            return True

        primary.add_conversation = async_add
        secondary.add_conversation = async_add

        dual_store = DualWriteVectorStore(primary=primary, secondary=secondary)

        result = await dual_store.add_conversation(
            "conv_1", "user", "agent", "session"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_dual_write_reads_from_configured(self):
        """Test dual write reads from configured backend."""
        from app.vector.factory import DualWriteVectorStore

        primary = MagicMock()
        secondary = MagicMock()

        async def primary_search(*args, **kwargs):
            return [{"id": "from_primary"}]
        async def secondary_search(*args, **kwargs):
            return [{"id": "from_secondary"}]

        primary.semantic_search = primary_search
        secondary.semantic_search = secondary_search

        # Test reading from primary
        dual_store = DualWriteVectorStore(
            primary=primary,
            secondary=secondary,
            read_from="primary"
        )

        results = await dual_store.semantic_search("query")
        assert results[0]["id"] == "from_primary"

        # Test reading from secondary
        dual_store.read_from = "secondary"
        results = await dual_store.semantic_search("query")
        assert results[0]["id"] == "from_secondary"

    @pytest.mark.asyncio
    async def test_dual_write_deletes_from_both(self):
        """Test dual write deletes from both backends."""
        from app.vector.factory import DualWriteVectorStore

        primary = MagicMock()
        secondary = MagicMock()

        async def async_delete(*args):
            return True

        primary.delete_conversation = async_delete
        secondary.delete_conversation = async_delete

        dual_store = DualWriteVectorStore(primary=primary, secondary=secondary)

        result = await dual_store.delete_conversation("conv_1")

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_consistency(self):
        """Test consistency validation between backends."""
        from app.vector.factory import DualWriteVectorStore

        primary = MagicMock()
        secondary = MagicMock()

        async def primary_stats():
            return {"count": 100}
        async def secondary_stats():
            return {"count": 100}

        primary.get_stats = primary_stats
        secondary.get_stats = secondary_stats

        dual_store = DualWriteVectorStore(primary=primary, secondary=secondary)

        validation = await dual_store.validate_consistency()

        assert validation["primary_count"] == 100
        assert validation["secondary_count"] == 100
        assert validation["consistent"] is True


class TestGetVectorStore:
    """Tests for get_vector_store convenience function."""

    def test_get_vector_store_default(self):
        """Test default behavior."""
        with patch('app.vector.VectorStoreFactory.create') as mock_create:
            with patch('app.config.settings') as mock_settings:
                mock_settings.vector_ab_test_enabled = False

                mock_store = MagicMock()
                mock_create.return_value = mock_store

                from app.vector import get_vector_store

                store = get_vector_store()

                mock_create.assert_called_with(backend="auto")

    def test_get_vector_store_with_ab_test(self):
        """Test with A/B testing enabled."""
        with patch('app.vector.VectorStoreFactory.create_with_ab_test') as mock_ab:
            with patch('app.config.settings') as mock_settings:
                mock_settings.vector_ab_test_enabled = True

                mock_store = MagicMock()
                mock_ab.return_value = mock_store

                from app.vector import get_vector_store

                store = get_vector_store(session_id="test_session")

                mock_ab.assert_called_with(session_id="test_session")
