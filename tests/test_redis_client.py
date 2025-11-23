"""
Tests for Resilient Redis Client
PATTERN: Unit tests with mocked Redis
WHY: Validate circuit breaker and retry logic without real Redis

Week 2 Production Hardening Tests
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from redis.exceptions import ConnectionError, TimeoutError, RedisError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.redis_client import (
    ResilientRedisClient,
    RedisConnectionConfig,
    CircuitBreaker,
    with_redis_fallback
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker pattern implementation."""

    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_max_calls=2
        )

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, circuit_breaker):
        """Circuit should start in CLOSED state."""
        assert circuit_breaker.state == CircuitBreaker.CLOSED
        assert await circuit_breaker.can_execute()

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self, circuit_breaker):
        """Circuit should OPEN after reaching failure threshold."""
        # Record failures up to threshold
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreaker.OPEN
        assert not await circuit_breaker.can_execute()

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, circuit_breaker):
        """Success should reset failure counter."""
        await circuit_breaker.record_failure()
        await circuit_breaker.record_failure()
        assert circuit_breaker.failures == 2

        await circuit_breaker.record_success()
        assert circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_recovery_timeout(self, circuit_breaker):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreaker.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should transition to HALF_OPEN on next check
        assert await circuit_breaker.can_execute()
        assert circuit_breaker.state == CircuitBreaker.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_successful_half_open(self, circuit_breaker):
        """Circuit should CLOSE after successful call in HALF_OPEN state."""
        # Open and wait for half-open
        for _ in range(3):
            await circuit_breaker.record_failure()
        await asyncio.sleep(1.1)
        await circuit_breaker.can_execute()

        assert circuit_breaker.state == CircuitBreaker.HALF_OPEN

        # Record success
        await circuit_breaker.record_success()
        assert circuit_breaker.state == CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_half_open_failure(self, circuit_breaker):
        """Circuit should reopen if failure occurs in HALF_OPEN state."""
        # Open and wait for half-open
        for _ in range(3):
            await circuit_breaker.record_failure()
        await asyncio.sleep(1.1)
        await circuit_breaker.can_execute()

        assert circuit_breaker.state == CircuitBreaker.HALF_OPEN

        # Record failure
        await circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreaker.OPEN


class TestRedisConnectionConfig:
    """Tests for Redis connection configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RedisConnectionConfig()
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RedisConnectionConfig(
            url="redis://custom:6379",
            max_retries=10,
            base_delay=1.0
        )
        assert config.url == "redis://custom:6379"
        assert config.max_retries == 10
        assert config.base_delay == 1.0


class TestResilientRedisClient:
    """Tests for ResilientRedisClient with retry logic."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value="test_value")
        mock.set = AsyncMock(return_value=True)
        mock.setex = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.info = AsyncMock(return_value={"connected_clients": 5})
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def client(self):
        """Create a ResilientRedisClient with fast retry config."""
        config = RedisConnectionConfig(
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1
        )
        return ResilientRedisClient(config)

    @pytest.mark.asyncio
    async def test_get_with_fallback(self, client, mock_redis):
        """Test get operation returns fallback on failure."""
        client.client = mock_redis
        client._connected = True

        # Normal operation
        result = await client.get("test_key")
        assert result == "test_value"

        # Test fallback on error
        mock_redis.get.side_effect = ConnectionError("Connection lost")
        result = await client.get("test_key", fallback="default")
        assert result == "default"

    @pytest.mark.asyncio
    async def test_set_operation(self, client, mock_redis):
        """Test set operation with retry."""
        client.client = mock_redis
        client._connected = True

        result = await client.set("key", "value", ex=300)
        assert result is True

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, client, mock_redis):
        """Test that operations retry on connection error."""
        client.client = mock_redis
        client._connected = True

        # First two calls fail, third succeeds
        mock_redis.get.side_effect = [
            ConnectionError("Failed 1"),
            ConnectionError("Failed 2"),
            "success"
        ]

        result = await client.get("key")
        assert result == "success"
        assert mock_redis.get.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, client, mock_redis):
        """Test circuit breaker opens after repeated failures."""
        client.client = mock_redis
        client._connected = True

        # All calls fail
        mock_redis.get.side_effect = ConnectionError("Always fails")

        # Make enough calls to open circuit
        for _ in range(5):
            await client.get("key", fallback=None)

        # Circuit should be open or half-open now
        assert client.circuit_state in [CircuitBreaker.OPEN, CircuitBreaker.HALF_OPEN]

    @pytest.mark.asyncio
    async def test_ping_operation(self, client, mock_redis):
        """Test ping for health check."""
        client.client = mock_redis
        client._connected = True

        result = await client.ping()
        assert result is True

        # Test ping failure
        mock_redis.ping.side_effect = ConnectionError("No connection")
        result = await client.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_close_gracefully(self, client, mock_redis):
        """Test graceful connection close."""
        client.client = mock_redis
        client._connected = True

        await client.close()

        assert client._connected is False
        assert client.client is None

    @pytest.mark.asyncio
    async def test_returns_fallback_when_not_connected(self, client):
        """Test operations return fallback when not connected."""
        client._connected = False
        client.client = None

        assert await client.get("key", fallback="default") == "default"
        assert await client.set("key", "value") is False
        assert await client.delete("key") == 0


class TestWithRedisFallbackDecorator:
    """Tests for the with_redis_fallback decorator."""

    @pytest.mark.asyncio
    async def test_decorator_returns_fallback_on_error(self):
        """Test decorator catches Redis errors and returns fallback."""

        @with_redis_fallback(fallback_value=[])
        async def get_items():
            raise ConnectionError("Redis down")

        result = await get_items()
        assert result == []

    @pytest.mark.asyncio
    async def test_decorator_returns_result_on_success(self):
        """Test decorator returns actual result on success."""

        @with_redis_fallback(fallback_value=[])
        async def get_items():
            return ["item1", "item2"]

        result = await get_items()
        assert result == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_decorator_handles_timeout_error(self):
        """Test decorator handles TimeoutError."""

        @with_redis_fallback(fallback_value="timeout_fallback")
        async def slow_operation():
            raise TimeoutError("Operation timed out")

        result = await slow_operation()
        assert result == "timeout_fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
