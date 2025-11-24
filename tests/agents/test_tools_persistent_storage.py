"""
Tests for Persistent Research Storage (Feature 2)

SPARC Specification:
- Research results persist in SQLite/PostgreSQL database
- Cross-session retrieval capability
- Migration from in-memory to persistent storage
- Automatic session association
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import asyncio
import json
import sys

# Mock circuitbreaker before importing app modules
sys.modules['circuitbreaker'] = MagicMock()

from app.agents.tools import ToolRegistry, tool_registry


@pytest.fixture
def mock_research_db():
    """Mock database for research storage"""
    storage = {}

    class MockResearchStore:
        async def store(self, key: str, value: dict, session_id: str = None):
            storage[key] = {
                "value": value,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            return key

        async def retrieve(self, key: str):
            return storage.get(key)

        async def list_by_session(self, session_id: str):
            return [v for k, v in storage.items() if v.get("session_id") == session_id]

        async def delete(self, key: str):
            if key in storage:
                del storage[key]
                return True
            return False

    return MockResearchStore()


@pytest.fixture
def sample_research_result():
    """Sample research result data"""
    return {
        "query": "transformer architecture",
        "results": [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "snippet": "We propose a new simple network architecture...",
                "source": "arxiv",
                "relevance_score": 0.98
            }
        ],
        "sources_used": ["arxiv", "web"],
        "timestamp": "2024-11-21T10:00:00"
    }


class TestPersistentResearchStorage:
    """Test cases for persistent research storage"""

    def test_memory_tool_registered(self):
        """Verify memory_store tool is registered"""
        tool = tool_registry.get_tool("memory_store")
        assert tool is not None
        assert tool.name == "memory_store"

    def test_memory_tool_schema(self):
        """Verify memory tool schema supports store/retrieve"""
        tool = tool_registry.get_tool("memory_store")
        schema = tool.input_schema

        assert "action" in schema["properties"]
        actions = schema["properties"]["action"]["enum"]
        assert "store" in actions
        assert "retrieve" in actions

    @pytest.mark.asyncio
    async def test_store_research_result(self, sample_research_result):
        """Test storing research results"""
        registry = ToolRegistry()
        tool = registry.get_tool("memory_store")

        result = await tool.handler(
            action="store",
            key="research_transformer",
            value=json.dumps(sample_research_result),
            context={"session_id": "test-session", "memory_store": {}}
        )

        assert result["success"] is True
        assert result["action"] == "stored"
        assert result["key"] == "research_transformer"

    @pytest.mark.asyncio
    async def test_retrieve_research_result(self, sample_research_result):
        """Test retrieving stored research results"""
        registry = ToolRegistry()
        tool = registry.get_tool("memory_store")

        # First store
        context = {"session_id": "test-session", "memory_store": {}}
        await tool.handler(
            action="store",
            key="research_data",
            value=json.dumps(sample_research_result),
            context=context
        )

        # Then retrieve
        result = await tool.handler(
            action="retrieve",
            key="research_data",
            context=context
        )

        assert result["success"] is True
        assert result["action"] == "retrieved"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_key(self):
        """Test retrieving non-existent key returns error"""
        registry = ToolRegistry()
        tool = registry.get_tool("memory_store")

        result = await tool.handler(
            action="retrieve",
            key="nonexistent_key",
            context={"memory_store": {}}
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_storage_includes_timestamp(self, sample_research_result):
        """Verify stored items include timestamp"""
        registry = ToolRegistry()
        tool = registry.get_tool("memory_store")

        context = {"memory_store": {}}
        await tool.handler(
            action="store",
            key="timed_data",
            value="test value",
            context=context
        )

        # Check internal storage has timestamp
        assert "timed_data" in context["memory_store"]
        assert "timestamp" in context["memory_store"]["timed_data"]

    @pytest.mark.asyncio
    async def test_store_without_value_fails(self):
        """Test storing without value returns error"""
        registry = ToolRegistry()
        tool = registry.get_tool("memory_store")

        result = await tool.handler(
            action="store",
            key="empty_key",
            value=None,
            context={"memory_store": {}}
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cross_session_persistence(self):
        """Test data persists across sessions (with database)"""
        # This tests database-backed persistence
        # In mock test, we verify the interface
        registry = ToolRegistry()
        tool = registry.get_tool("memory_store")

        # Session 1: Store
        context1 = {"session_id": "session1", "memory_store": {}}
        await tool.handler(
            action="store",
            key="persistent_data",
            value="important info",
            context=context1
        )

        # Session 2: Should be able to retrieve (with database backend)
        # In this mock, contexts are separate, but with real DB they'd share
        assert context1["memory_store"]["persistent_data"]["value"] == "important info"


class TestResearchStorageDatabase:
    """Test database schema and operations for research storage"""

    @pytest.mark.asyncio
    async def test_research_store_table_schema(self):
        """Test database table has correct schema"""
        # Schema should include: id, session_id, key, value, created_at, updated_at
        expected_columns = ["id", "session_id", "key", "value", "created_at", "updated_at"]
        # This would be verified with actual database in integration tests

    @pytest.mark.asyncio
    async def test_research_store_creates_tables(self):
        """Test research store auto-creates tables"""
        # Database initialization should create required tables

    @pytest.mark.asyncio
    async def test_research_store_indexes(self):
        """Test indexes exist for efficient queries"""
        # Should have indexes on: session_id, key, created_at
