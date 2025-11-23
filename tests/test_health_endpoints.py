"""
Tests for Health Check Endpoints
PATTERN: Integration tests for API endpoints
WHY: Validate health monitoring works correctly

Week 2 Production Hardening Tests
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoints:
    """Tests for health check API endpoints."""

    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        mock = AsyncMock()
        mock.get_stats = AsyncMock(return_value={
            "total_captures": 100,
            "unique_sessions": 25,
            "last_capture": "2024-01-14T12:00:00"
        })
        mock.initialize = AsyncMock()
        return mock

    @pytest.fixture
    def mock_state_manager(self):
        """Mock state manager."""
        mock = AsyncMock()
        mock.get_active_sessions = AsyncMock(return_value=["session1", "session2"])
        mock.initialize = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def mock_resilient_redis(self):
        """Mock resilient Redis client."""
        mock = MagicMock()
        mock.is_connected = True
        mock.circuit_state = "closed"
        mock.ping = AsyncMock(return_value=True)
        mock.info = AsyncMock(return_value={
            "connected_clients": 5,
            "used_memory_human": "10MB"
        })
        mock.circuit_breaker = MagicMock()
        mock.circuit_breaker.failures = 0
        mock.circuit_breaker.failure_threshold = 5
        return mock

    @pytest.mark.asyncio
    async def test_simple_health_check(self, mock_db, mock_state_manager):
        """Test /health endpoint returns healthy status."""
        with patch("app.main.db", mock_db), \
             patch("app.main.state_manager", mock_state_manager):

            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_detailed_health_check_healthy(
        self, mock_db, mock_state_manager, mock_resilient_redis
    ):
        """Test /health/detailed returns component status when healthy."""
        with patch("app.main.db", mock_db), \
             patch("app.main.state_manager", mock_state_manager), \
             patch("app.main.resilient_redis", mock_resilient_redis):

            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]

    @pytest.mark.asyncio
    async def test_detailed_health_check_degraded(
        self, mock_db, mock_state_manager
    ):
        """Test /health/detailed returns degraded when component fails."""
        # Make database fail
        mock_db.get_stats.side_effect = Exception("Database connection lost")

        with patch("app.main.db", mock_db), \
             patch("app.main.state_manager", mock_state_manager), \
             patch("app.main.resilient_redis", None):

            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["database"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_redis_health_endpoint(self, mock_resilient_redis):
        """Test /health/redis endpoint."""
        with patch("app.main.resilient_redis", mock_resilient_redis):

            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/health/redis")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["connected"] is True
        assert "circuit_breaker" in data

    @pytest.mark.asyncio
    async def test_redis_health_when_unavailable(self):
        """Test /health/redis when resilient redis is not configured."""
        with patch("app.main.resilient_redis", None):

            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/health/redis")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_root_endpoint_shows_health_endpoints(
        self, mock_db, mock_state_manager
    ):
        """Test root endpoint includes health endpoint info."""
        with patch("app.main.db", mock_db), \
             patch("app.main.state_manager", mock_state_manager):

            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "health" in data["endpoints"]
        assert "health_detailed" in data["endpoints"]


class TestHealthEndpointIntegration:
    """Integration tests that verify endpoint behavior."""

    @pytest.mark.asyncio
    async def test_health_check_timestamp_format(self):
        """Test health check returns valid ISO timestamp."""
        from app.main import app
        from datetime import datetime

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/health")

        data = response.json()
        # Should not raise if timestamp is valid ISO format
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
