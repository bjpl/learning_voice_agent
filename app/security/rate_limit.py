"""
Rate Limiting Implementation - Plan A Security

SPARC Implementation:
- Specification: Protect endpoints from abuse
- Architecture: Redis-backed with in-memory fallback
- Features: Configurable limits, distributed support
"""

import time
import logging
from typing import Optional, Dict, Callable
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint."""
    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    key_func: Optional[Callable[[Request], str]] = None  # Custom key function


@dataclass
class RateLimitInfo:
    """Rate limit state information."""
    count: int = 0
    window_start: float = 0.0


class RateLimiter:
    """
    Rate limiter with Redis support and in-memory fallback.

    PATTERN: Sliding window counter algorithm
    WHY: Balance between accuracy and memory efficiency
    """

    # Default rate limits by endpoint category
    DEFAULT_LIMITS = {
        "auth": RateLimitConfig(requests=10, window=60),  # 10 req/min for auth
        "api": RateLimitConfig(requests=100, window=60),  # 100 req/min for API
        "health": RateLimitConfig(requests=1000, window=60),  # 1000 req/min for health
        "websocket": RateLimitConfig(requests=30, window=60),  # 30 req/min for WS
        "admin": RateLimitConfig(requests=50, window=60),  # 50 req/min for admin
    }

    def __init__(self, redis_client=None):
        """
        Initialize rate limiter.

        Args:
            redis_client: Optional Redis client for distributed limiting
        """
        self._redis = redis_client
        self._local_store: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._custom_limits: Dict[str, RateLimitConfig] = {}

    def configure(self, endpoint: str, requests: int, window: int) -> None:
        """Configure custom rate limit for an endpoint."""
        self._custom_limits[endpoint] = RateLimitConfig(requests=requests, window=window)

    def _get_key(self, request: Request, endpoint_category: str) -> str:
        """Generate rate limit key from request."""
        # Get client IP (handle proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"rate_limit:{endpoint_category}:{client_ip}"

    def _get_limit_config(self, path: str) -> RateLimitConfig:
        """Get rate limit configuration for a path."""
        # Check custom limits first
        if path in self._custom_limits:
            return self._custom_limits[path]

        # Determine category from path
        if path.startswith("/api/auth"):
            return self.DEFAULT_LIMITS["auth"]
        elif path.startswith("/health"):
            return self.DEFAULT_LIMITS["health"]
        elif path.startswith("/admin"):
            return self.DEFAULT_LIMITS["admin"]
        elif path.startswith("/ws"):
            return self.DEFAULT_LIMITS["websocket"]
        else:
            return self.DEFAULT_LIMITS["api"]

    async def check_rate_limit(
        self,
        request: Request,
        custom_limit: Optional[RateLimitConfig] = None,
    ) -> Dict[str, int]:
        """
        Check if request is within rate limit.

        Args:
            request: FastAPI request object
            custom_limit: Optional custom limit configuration

        Returns:
            Dict with remaining requests and reset time

        Raises:
            HTTPException: If rate limit exceeded
        """
        path = request.url.path
        config = custom_limit or self._get_limit_config(path)

        # Determine category for key
        if path.startswith("/api/auth"):
            category = "auth"
        elif path.startswith("/health"):
            category = "health"
        elif path.startswith("/admin"):
            category = "admin"
        elif path.startswith("/ws"):
            category = "websocket"
        else:
            category = "api"

        key = self._get_key(request, category)

        # Try Redis first, fall back to local
        if self._redis:
            return await self._check_redis(key, config)
        else:
            return self._check_local(key, config)

    async def _check_redis(self, key: str, config: RateLimitConfig) -> Dict[str, int]:
        """Check rate limit using Redis."""
        try:
            now = time.time()
            window_start = now - config.window

            # Use Redis pipeline for atomic operations
            pipe = self._redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current entries
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Set expiry
            pipe.expire(key, config.window)

            results = await pipe.execute()
            current_count = results[1]

            if current_count >= config.requests:
                # Get TTL for retry-after header
                ttl = await self._redis.ttl(key)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(ttl),
                        "X-RateLimit-Limit": str(config.requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + ttl)),
                    },
                )

            remaining = config.requests - current_count - 1
            reset_time = int(now + config.window)

            return {
                "limit": config.requests,
                "remaining": max(0, remaining),
                "reset": reset_time,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}, falling back to local")
            return self._check_local(key, config)

    def _check_local(self, key: str, config: RateLimitConfig) -> Dict[str, int]:
        """Check rate limit using local memory."""
        now = time.time()
        info = self._local_store[key]

        # Reset window if expired
        if now - info.window_start >= config.window:
            info.count = 0
            info.window_start = now

        # Check limit
        if info.count >= config.requests:
            retry_after = int(config.window - (now - info.window_start))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(config.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(info.window_start + config.window)),
                },
            )

        # Increment counter
        info.count += 1

        remaining = config.requests - info.count
        reset_time = int(info.window_start + config.window)

        return {
            "limit": config.requests,
            "remaining": max(0, remaining),
            "reset": reset_time,
        }

    def cleanup_expired(self) -> int:
        """Clean up expired local rate limit entries."""
        now = time.time()
        expired_keys = []

        for key, info in self._local_store.items():
            # Get config for this key
            category = key.split(":")[1] if ":" in key else "api"
            config = self.DEFAULT_LIMITS.get(category, self.DEFAULT_LIMITS["api"])

            if now - info.window_start >= config.window * 2:
                expired_keys.append(key)

        for key in expired_keys:
            del self._local_store[key]

        return len(expired_keys)


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        # Try to get Redis client
        redis_client = None
        try:
            from app.redis_client import resilient_redis
            if resilient_redis and resilient_redis.is_connected:
                redis_client = resilient_redis._client
        except ImportError:
            pass

        _rate_limiter = RateLimiter(redis_client=redis_client)

    return _rate_limiter


def rate_limit(
    requests: Optional[int] = None,
    window: Optional[int] = None,
):
    """
    Decorator for rate limiting endpoints.

    Args:
        requests: Max requests per window (default: category default)
        window: Time window in seconds (default: 60)

    Usage:
        @app.get("/api/resource")
        @rate_limit(requests=50, window=60)
        async def get_resource():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            limiter = get_rate_limiter()

            custom_config = None
            if requests is not None and window is not None:
                custom_config = RateLimitConfig(requests=requests, window=window)

            # Check rate limit
            rate_info = await limiter.check_rate_limit(request, custom_config)

            # Execute function
            response = await func(request, *args, **kwargs)

            # Add rate limit headers to response if it's a Response object
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

            return response

        return wrapper
    return decorator


class RateLimitMiddleware:
    """
    Middleware for global rate limiting.

    Use this for blanket rate limiting across all endpoints.
    For endpoint-specific limits, use the @rate_limit decorator.
    """

    def __init__(self, app, limiter: Optional[RateLimiter] = None):
        self.app = app
        self.limiter = limiter or get_rate_limiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Create a minimal request object for rate limiting
        from starlette.requests import Request
        request = Request(scope, receive)

        try:
            rate_info = await self.limiter.check_rate_limit(request)

            # Store rate info in scope for later use
            scope["rate_limit_info"] = rate_info

        except HTTPException as e:
            # Send rate limit exceeded response
            from starlette.responses import JSONResponse
            response = JSONResponse(
                content={"detail": e.detail},
                status_code=e.status_code,
                headers=e.headers or {},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
