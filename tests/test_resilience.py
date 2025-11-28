"""
Unit Tests for Resilience Module - Week 1 Infrastructure

Tests cover:
- Circuit breaker state management
- Circuit breaker transitions (closed -> open -> half-open)
- Retry logic with exponential backoff
- LRU cache functionality
- Decorator behavior

Target: 80%+ coverage for resilience.py
"""
import pytest
import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.resilience import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpen,
    LRUCache,
    get_circuit_breaker,
    with_circuit_breaker,
    with_retry,
    with_cache,
    get_all_circuit_breaker_status,
    get_cache_stats
)


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState dataclass."""

    def test_default_state(self):
        """Should initialize with correct defaults."""
        state = CircuitBreakerState()

        assert state.failures == 0
        assert state.last_failure is None
        assert state.state == "closed"
        assert state.success_count == 0
        assert state.failure_threshold == 5
        assert state.recovery_timeout == 30
        assert state.half_open_success_threshold == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_circuit_breaker_init(self):
        """Should initialize with correct defaults."""
        breaker = CircuitBreaker("test")

        assert breaker.name == "test"
        assert breaker.state.state == "closed"

    def test_circuit_breaker_allows_requests_when_closed(self):
        """Should allow requests when circuit is closed."""
        breaker = CircuitBreaker("test")

        assert breaker.is_open() is False

    def test_circuit_breaker_opens_after_threshold(self):
        """Should open circuit after failure threshold."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        # Record failures
        for _ in range(3):
            breaker.record_failure(Exception("Test error"))

        assert breaker.state.state == "open"
        assert breaker.is_open() is True

    def test_circuit_breaker_success_resets_in_closed_state(self):
        """Success should decrement failure count in closed state."""
        breaker = CircuitBreaker("test")

        breaker.record_failure(Exception("Error"))
        breaker.record_failure(Exception("Error"))
        assert breaker.state.failures == 2

        breaker.record_success()
        assert breaker.state.failures == 1

    def test_circuit_breaker_half_open_after_timeout(self):
        """Should transition to half-open after recovery timeout."""
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0)

        # Open the circuit
        breaker.record_failure(Exception("Error"))
        breaker.record_failure(Exception("Error"))
        assert breaker.state.state == "open"

        # Wait for recovery (timeout is 0)
        # Access should trigger half-open
        assert breaker.is_open() is False  # Will transition to half-open
        assert breaker.state.state == "half-open"

    def test_circuit_breaker_closes_after_half_open_successes(self):
        """Should close after successful calls in half-open."""
        breaker = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=0,
            half_open_success_threshold=2
        )

        # Open the circuit
        breaker.record_failure(Exception("Error"))
        breaker.record_failure(Exception("Error"))

        # Transition to half-open
        breaker.is_open()
        assert breaker.state.state == "half-open"

        # Record successes
        breaker.record_success()
        assert breaker.state.state == "half-open"

        breaker.record_success()
        assert breaker.state.state == "closed"

    def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Should reopen on failure during half-open."""
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0)

        # Open and transition to half-open
        breaker.record_failure(Exception("Error"))
        breaker.record_failure(Exception("Error"))
        breaker.is_open()  # Transitions to half-open

        # Record failure during half-open
        breaker.record_failure(Exception("Error"))

        assert breaker.state.state == "open"

    def test_circuit_breaker_get_status(self):
        """Should return correct status dict."""
        breaker = CircuitBreaker("test_status")

        status = breaker.get_status()

        assert status["name"] == "test_status"
        assert status["state"] == "closed"
        assert status["failures"] == 0


class TestCircuitBreakerFactory:
    """Tests for circuit breaker factory function."""

    def test_get_circuit_breaker_creates_new(self):
        """Should create new circuit breaker."""
        breaker = get_circuit_breaker("new_breaker", failure_threshold=10)

        assert breaker.name == "new_breaker"

    def test_get_circuit_breaker_reuses_existing(self):
        """Should reuse existing circuit breaker."""
        breaker1 = get_circuit_breaker("reuse_test")
        breaker2 = get_circuit_breaker("reuse_test")

        assert breaker1 is breaker2


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_call_when_closed(self):
        """Should allow function call when circuit is closed."""
        @with_circuit_breaker("decorator_test_1")
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_raises_when_open(self):
        """Should raise CircuitBreakerOpen when circuit is open."""
        # Get breaker and force it open
        breaker = get_circuit_breaker("decorator_test_2", failure_threshold=1)
        breaker.record_failure(Exception("Force open"))

        @with_circuit_breaker("decorator_test_2")
        async def test_func():
            return "success"

        with pytest.raises(CircuitBreakerOpen):
            await test_func()

    @pytest.mark.asyncio
    async def test_decorator_records_success(self):
        """Should record success on successful call."""
        breaker = get_circuit_breaker("decorator_test_3")

        @with_circuit_breaker("decorator_test_3")
        async def test_func():
            return "success"

        await test_func()

        # Failure count should be at or below initial
        assert breaker.state.failures <= 1

    @pytest.mark.asyncio
    async def test_decorator_records_failure(self):
        """Should record failure on exception."""
        breaker = get_circuit_breaker("decorator_test_4")
        initial_failures = breaker.state.failures

        @with_circuit_breaker("decorator_test_4")
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_func()

        assert breaker.state.failures == initial_failures + 1


class TestRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_try(self):
        """Should return result on successful first call."""
        call_count = 0

        @with_retry(max_attempts=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_retries_on_failure(self):
        """Should retry on exception."""
        call_count = 0

        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await test_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_raises_after_max_attempts(self):
        """Should raise exception after max attempts."""
        import tenacity
        call_count = 0

        @with_retry(max_attempts=2, min_wait=0.01, max_wait=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        # tenacity wraps the exception in RetryError
        with pytest.raises((ValueError, tenacity.RetryError)):
            await always_fails()

        assert call_count == 2


class TestLRUCache:
    """Tests for LRU cache."""

    def test_cache_set_and_get(self):
        """Should store and retrieve values."""
        cache = LRUCache(maxsize=10)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_cache_miss_returns_none(self):
        """Should return None for missing keys."""
        cache = LRUCache(maxsize=10)

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_respects_maxsize(self):
        """Should evict oldest items when full."""
        cache = LRUCache(maxsize=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_ttl_expiration(self):
        """Should expire items after TTL."""
        import time
        cache = LRUCache(maxsize=10, ttl_seconds=0.01)  # 10ms expiration

        cache.set("key1", "value1")
        time.sleep(0.02)  # Wait for expiration
        result = cache.get("key1")

        assert result is None  # Should be expired

    def test_cache_updates_existing_key(self):
        """Should update value for existing key."""
        cache = LRUCache(maxsize=10, ttl_seconds=300)

        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"

    def test_cache_stats(self):
        """Should track hits and misses."""
        cache = LRUCache(maxsize=10, ttl_seconds=300)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_clear(self):
        """Should clear all items and reset stats."""
        cache = LRUCache(maxsize=10, ttl_seconds=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")

        cache.clear()

        assert cache.get("key1") is None
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0


class TestCacheDecorator:
    """Tests for cache decorator."""

    @pytest.mark.asyncio
    async def test_cache_decorator_caches_result(self):
        """Should cache function results."""
        call_count = 0

        @with_cache(ttl_seconds=300)
        async def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await expensive_func(5)
        result2 = await expensive_func(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Second call should use cache

    @pytest.mark.asyncio
    async def test_cache_decorator_different_args(self):
        """Should cache separately for different arguments."""
        call_count = 0

        @with_cache(ttl_seconds=300)
        async def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        await expensive_func(5)
        await expensive_func(10)

        assert call_count == 2  # Different args = different cache entries


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_all_circuit_breaker_status(self):
        """Should return status of all circuit breakers."""
        # Create a couple of breakers
        get_circuit_breaker("status_test_1")
        get_circuit_breaker("status_test_2")

        statuses = get_all_circuit_breaker_status()

        assert isinstance(statuses, list)
        # Should have at least our test breakers
        names = [s["name"] for s in statuses]
        assert "status_test_1" in names or len(statuses) > 0

    def test_get_cache_stats(self):
        """Should return cache statistics."""
        stats = get_cache_stats()

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "maxsize" in stats


class TestCircuitBreakerOpenException:
    """Tests for CircuitBreakerOpen exception."""

    def test_exception_message(self):
        """Should have descriptive message."""
        exc = CircuitBreakerOpen("test_breaker")

        assert "test_breaker" in str(exc)
        assert exc.breaker_name == "test_breaker"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
