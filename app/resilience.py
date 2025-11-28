"""
API Resilience Layer - Week 1 Infrastructure
PATTERN: Circuit breaker + retry with exponential backoff
WHY: Graceful degradation and cost control for external API calls
"""
from typing import TypeVar, Callable, Any, Optional
from functools import wraps, lru_cache
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
        RetryError
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

from app.logger import get_logger

logger = get_logger("voice_agent.resilience")

T = TypeVar('T')


@dataclass
class CircuitBreakerState:
    """
    Track circuit breaker state for an API endpoint.

    PATTERN: Circuit breaker pattern
    WHY: Prevent cascade failures and reduce costs when API is down
    """
    failures: int = 0
    last_failure: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    success_count: int = 0

    # Configuration
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    half_open_success_threshold: int = 2


class CircuitBreaker:
    """
    Circuit breaker implementation for API calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit tripped, requests fail fast
    - HALF-OPEN: Testing if service recovered

    PATTERN: State machine for failure handling
    WHY: Protects both client and server during outages
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_success_threshold: int = 2
    ):
        self.name = name
        self.state = CircuitBreakerState(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_success_threshold=half_open_success_threshold
        )
        self._lock = threading.Lock()

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        with self._lock:
            if self.state.state == "closed":
                return True

            if self.state.state == "open":
                # Check if recovery timeout has passed
                if self.state.last_failure:
                    elapsed = (datetime.utcnow() - self.state.last_failure).total_seconds()
                    if elapsed >= self.state.recovery_timeout:
                        self.state.state = "half-open"
                        self.state.success_count = 0
                        logger.info(
                            "circuit_half_open",
                            breaker=self.name,
                            elapsed_seconds=elapsed
                        )
                        return True
                return False

            # half-open: allow limited requests
            return True

    def record_success(self) -> None:
        """Record successful API call."""
        with self._lock:
            if self.state.state == "half-open":
                self.state.success_count += 1
                if self.state.success_count >= self.state.half_open_success_threshold:
                    self.state.state = "closed"
                    self.state.failures = 0
                    logger.info("circuit_closed", breaker=self.name)
            elif self.state.state == "closed":
                # Reset failure count on success
                self.state.failures = max(0, self.state.failures - 1)

    def record_failure(self, error: Exception) -> None:
        """Record failed API call."""
        with self._lock:
            self.state.failures += 1
            self.state.last_failure = datetime.utcnow()

            if self.state.state == "half-open":
                # Any failure in half-open returns to open
                self.state.state = "open"
                logger.warning(
                    "circuit_reopened",
                    breaker=self.name,
                    error=str(error)
                )
            elif self.state.failures >= self.state.failure_threshold:
                self.state.state = "open"
                logger.error(
                    "circuit_opened",
                    breaker=self.name,
                    failures=self.state.failures,
                    error=str(error)
                )

    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return not self._should_allow_request()

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.state,
            "failures": self.state.failures,
            "last_failure": self.state.last_failure.isoformat() if self.state.last_failure else None
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker '{breaker_name}' is open")


# Global circuit breakers for different APIs
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 30
) -> CircuitBreaker:
    """Get or create a circuit breaker for an API."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    return _circuit_breakers[name]


def with_circuit_breaker(
    breaker_name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: int = 30
):
    """
    Decorator to wrap async function with circuit breaker.

    Usage:
        @with_circuit_breaker("claude_api")
        async def call_claude(prompt: str) -> str:
            ...

        # Or with custom thresholds:
        @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Use function name as breaker name if default
        actual_breaker_name = breaker_name if breaker_name != "default" else func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            breaker = get_circuit_breaker(
                actual_breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )

            if breaker.is_open():
                raise CircuitBreakerOpen(actual_breaker_name)

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                raise

        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.

    PATTERN: Exponential backoff
    WHY: Handle transient failures without overwhelming the API

    Args:
        max_attempts: Maximum retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        retry_exceptions: Tuple of exceptions to retry on

    Usage:
        @with_retry(max_attempts=3, retry_exceptions=(APIError,))
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if TENACITY_AVAILABLE:
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(retry_exceptions),
                before_sleep=before_sleep_log(logger, "WARNING")
            )
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                return await func(*args, **kwargs)
            return wrapper
        else:
            # Fallback implementation without tenacity
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                last_exception = None
                wait_time = min_wait

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except retry_exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning(
                                "retry_attempt",
                                attempt=attempt + 1,
                                max_attempts=max_attempts,
                                wait_time=wait_time,
                                error=str(e)
                            )
                            await asyncio.sleep(wait_time)
                            wait_time = min(wait_time * 2, max_wait)

                raise last_exception
            return wrapper
    return decorator


def with_timeout(timeout_seconds: float = 30.0):
    """
    Decorator for async function timeout.

    PATTERN: Timeout protection
    WHY: Prevent hanging operations from blocking the system

    Args:
        timeout_seconds: Maximum time to wait before raising TimeoutError

    Usage:
        @with_timeout(30)
        async def long_running_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "operation_timeout",
                    function=func.__name__,
                    timeout_seconds=timeout_seconds
                )
                raise TimeoutError(f"Operation {func.__name__} timed out after {timeout_seconds}s")
        return wrapper
    return decorator


class LRUCache:
    """
    Thread-safe LRU cache for API responses.

    PATTERN: Cache-aside with TTL
    WHY: Reduce API costs by caching repeated requests
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from function arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if (datetime.utcnow() - timestamp).total_seconds() > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    # Remove oldest item
                    self._cache.popitem(last=False)

            self._cache[key] = (value, datetime.utcnow())

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%"
            }


# Global caches for different purposes
_response_cache = LRUCache(maxsize=100, ttl_seconds=300)


def with_cache(ttl_seconds: int = 300, cache_key_fn: Optional[Callable] = None):
    """
    Decorator to cache async function results.

    Args:
        ttl_seconds: Time-to-live for cached values
        cache_key_fn: Optional function to generate cache key from args

    Usage:
        @with_cache(ttl_seconds=60)
        async def expensive_api_call(query: str) -> str:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = LRUCache(maxsize=100, ttl_seconds=ttl_seconds)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if cache_key_fn:
                key = cache_key_fn(*args, **kwargs)
            else:
                key = cache._make_key(*args, **kwargs)

            # Check cache
            cached = cache.get(key)
            if cached is not None:
                logger.debug("cache_hit", function=func.__name__, key=key[:8])
                return cached

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result)
            logger.debug("cache_miss", function=func.__name__, key=key[:8])
            return result

        # Attach cache stats method
        wrapper.cache_stats = cache.stats
        wrapper.clear_cache = cache.clear

        return wrapper
    return decorator


def get_all_circuit_breaker_status() -> list[dict]:
    """Get status of all circuit breakers."""
    return [breaker.get_status() for breaker in _circuit_breakers.values()]


def get_cache_stats() -> dict:
    """Get global cache statistics."""
    return _response_cache.stats()
