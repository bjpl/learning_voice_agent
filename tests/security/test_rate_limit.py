"""
Rate Limiting Tests - Plan A Security

Tests for:
- Rate limit enforcement
- Different limits by endpoint category
- Redis vs local fallback
- Rate limit headers
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from starlette.testclient import TestClient

from app.security.rate_limit import (
    RateLimiter,
    RateLimitConfig,
    get_rate_limiter,
    rate_limit,
)


@pytest.fixture
def limiter():
    """Fresh RateLimiter instance for testing."""
    return RateLimiter(redis_client=None)  # Use local store


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.url.path = "/api/test"
    request.client.host = "127.0.0.1"
    request.headers = {}
    return request


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_default_limits_exist(self):
        """Default limits should be defined."""
        limiter = RateLimiter()

        assert "auth" in limiter.DEFAULT_LIMITS
        assert "api" in limiter.DEFAULT_LIMITS
        assert "health" in limiter.DEFAULT_LIMITS

    def test_custom_limit_configuration(self, limiter):
        """Custom limits should override defaults."""
        limiter.configure("/api/custom", requests=50, window=30)

        config = limiter._custom_limits.get("/api/custom")
        assert config is not None
        assert config.requests == 50
        assert config.window == 30


class TestLocalRateLimiting:
    """Test in-memory rate limiting."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self, limiter, mock_request):
        """Requests under limit should be allowed."""
        for i in range(5):
            result = await limiter.check_rate_limit(mock_request)
            assert result["remaining"] >= 0

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self, limiter, mock_request):
        """Requests over limit should be blocked."""
        # Configure low limit for testing
        config = RateLimitConfig(requests=3, window=60)

        # Make requests up to limit
        for i in range(3):
            await limiter.check_rate_limit(mock_request, custom_limit=config)

        # Next request should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit(mock_request, custom_limit=config)

        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_window(self, limiter, mock_request):
        """Rate limit should reset after window expires."""
        # Configure short window
        config = RateLimitConfig(requests=2, window=1)

        # Use up limit
        await limiter.check_rate_limit(mock_request, custom_limit=config)
        await limiter.check_rate_limit(mock_request, custom_limit=config)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        result = await limiter.check_rate_limit(mock_request, custom_limit=config)
        assert result["remaining"] >= 0

    @pytest.mark.asyncio
    async def test_different_ips_have_separate_limits(self, limiter):
        """Different IPs should have separate rate limits."""
        config = RateLimitConfig(requests=2, window=60)

        request1 = MagicMock()
        request1.url.path = "/api/test"
        request1.client.host = "192.168.1.1"
        request1.headers = {}

        request2 = MagicMock()
        request2.url.path = "/api/test"
        request2.client.host = "192.168.1.2"
        request2.headers = {}

        # Use up limit for IP1
        await limiter.check_rate_limit(request1, custom_limit=config)
        await limiter.check_rate_limit(request1, custom_limit=config)

        # IP1 should be blocked
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(request1, custom_limit=config)

        # IP2 should still be allowed
        result = await limiter.check_rate_limit(request2, custom_limit=config)
        assert result["remaining"] >= 0


class TestRateLimitCategories:
    """Test rate limits by endpoint category."""

    @pytest.mark.asyncio
    async def test_auth_endpoints_have_stricter_limit(self, limiter):
        """Auth endpoints should have lower limit."""
        auth_request = MagicMock()
        auth_request.url.path = "/api/auth/login"
        auth_request.client.host = "127.0.0.1"
        auth_request.headers = {}

        api_request = MagicMock()
        api_request.url.path = "/api/data"
        api_request.client.host = "127.0.0.2"
        api_request.headers = {}

        auth_result = await limiter.check_rate_limit(auth_request)
        api_result = await limiter.check_rate_limit(api_request)

        # Auth limit should be lower than API limit
        assert auth_result["limit"] < api_result["limit"]

    @pytest.mark.asyncio
    async def test_health_endpoints_have_higher_limit(self, limiter):
        """Health endpoints should have higher limit."""
        health_request = MagicMock()
        health_request.url.path = "/health"
        health_request.client.host = "127.0.0.1"
        health_request.headers = {}

        api_request = MagicMock()
        api_request.url.path = "/api/data"
        api_request.client.host = "127.0.0.2"
        api_request.headers = {}

        health_result = await limiter.check_rate_limit(health_request)
        api_result = await limiter.check_rate_limit(api_request)

        # Health limit should be higher than API limit
        assert health_result["limit"] > api_result["limit"]


class TestRateLimitHeaders:
    """Test rate limit response headers."""

    @pytest.mark.asyncio
    async def test_rate_limit_info_returned(self, limiter, mock_request):
        """Rate limit info should be returned."""
        result = await limiter.check_rate_limit(mock_request)

        assert "limit" in result
        assert "remaining" in result
        assert "reset" in result

    @pytest.mark.asyncio
    async def test_remaining_decreases(self, limiter, mock_request):
        """Remaining count should decrease with each request."""
        config = RateLimitConfig(requests=10, window=60)

        result1 = await limiter.check_rate_limit(mock_request, custom_limit=config)
        result2 = await limiter.check_rate_limit(mock_request, custom_limit=config)

        assert result2["remaining"] < result1["remaining"]


class TestXForwardedFor:
    """Test handling of X-Forwarded-For header."""

    @pytest.mark.asyncio
    async def test_uses_forwarded_ip(self, limiter):
        """Should use X-Forwarded-For IP when present."""
        config = RateLimitConfig(requests=2, window=60)

        request1 = MagicMock()
        request1.url.path = "/api/test"
        request1.client.host = "127.0.0.1"
        request1.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}

        request2 = MagicMock()
        request2.url.path = "/api/test"
        request2.client.host = "127.0.0.1"  # Same client host
        request2.headers = {"X-Forwarded-For": "10.0.0.2, 192.168.1.1"}  # Different forwarded

        # Use up limit for first forwarded IP
        await limiter.check_rate_limit(request1, custom_limit=config)
        await limiter.check_rate_limit(request1, custom_limit=config)

        # First forwarded IP should be blocked
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(request1, custom_limit=config)

        # Second forwarded IP should be allowed
        result = await limiter.check_rate_limit(request2, custom_limit=config)
        assert result["remaining"] >= 0


class TestCleanup:
    """Test cleanup of expired rate limit entries."""

    def test_cleanup_removes_expired(self, limiter):
        """Cleanup should remove expired entries."""
        # Add some entries
        limiter._local_store["test:key1"].count = 5
        limiter._local_store["test:key1"].window_start = time.time() - 200

        limiter._local_store["test:key2"].count = 5
        limiter._local_store["test:key2"].window_start = time.time()

        # Cleanup
        removed = limiter.cleanup_expired()

        assert removed == 1
        assert "test:key1" not in limiter._local_store
        assert "test:key2" in limiter._local_store
