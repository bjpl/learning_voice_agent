"""
Redis Client with Failover Handling
PATTERN: Circuit breaker with exponential backoff retry
WHY: Production resilience against Redis connection failures
"""
import asyncio
import logging
from typing import Optional, Any, Callable
from functools import wraps
import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisConnectionConfig:
    """Redis connection configuration with retry parameters."""
    def __init__(
        self,
        url: str = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        connection_timeout: float = 5.0,
        socket_timeout: float = 5.0,
        health_check_interval: int = 30,
    ):
        self.url = url or settings.redis_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.connection_timeout = connection_timeout
        self.socket_timeout = socket_timeout
        self.health_check_interval = health_check_interval


class CircuitBreaker:
    """
    Circuit breaker pattern for Redis operations.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = self.CLOSED
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if operation can proceed based on circuit state."""
        async with self._lock:
            if self.state == self.CLOSED:
                return True

            if self.state == self.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = asyncio.get_event_loop().time() - self.last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self.state = self.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        return True
                return False

            if self.state == self.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self):
        """Record successful operation."""
        async with self._lock:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                logger.info("Circuit breaker CLOSED after successful recovery")
            self.failures = 0

    async def record_failure(self):
        """Record failed operation."""
        async with self._lock:
            self.failures += 1
            self.last_failure_time = asyncio.get_event_loop().time()

            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                logger.warning("Circuit breaker OPEN after half-open failure")
            elif self.failures >= self.failure_threshold:
                self.state = self.OPEN
                logger.warning(f"Circuit breaker OPEN after {self.failures} failures")


class ResilientRedisClient:
    """
    Redis client with automatic failover and retry handling.

    Features:
    - Exponential backoff retry
    - Circuit breaker pattern
    - Connection health monitoring
    - Graceful degradation
    """

    def __init__(self, config: RedisConnectionConfig = None):
        self.config = config or RedisConnectionConfig()
        self.client: Optional[redis.Redis] = None
        self.circuit_breaker = CircuitBreaker()
        self._connected = False
        self._health_check_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        Establish Redis connection with retry logic.

        Returns:
            bool: True if connected, False if all retries failed
        """
        for attempt in range(self.config.max_retries):
            try:
                self.client = await redis.from_url(
                    self.config.url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.connection_timeout,
                    max_connections=50,
                    health_check_interval=self.config.health_check_interval,
                )

                # Verify connection with ping
                await self.client.ping()
                self._connected = True
                logger.info(f"Redis connected successfully on attempt {attempt + 1}")

                # Start health check task
                self._start_health_check()

                return True

            except (ConnectionError, TimeoutError, RedisError) as e:
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                logger.warning(
                    f"Redis connection attempt {attempt + 1}/{self.config.max_retries} "
                    f"failed: {e}. Retrying in {delay:.2f}s"
                )

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)

        logger.error("Failed to connect to Redis after all retries")
        self._connected = False
        return False

    def _start_health_check(self):
        """Start background health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self):
        """Periodic health check for Redis connection."""
        while self._connected:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                if self.client:
                    await self.client.ping()
                    await self.circuit_breaker.record_success()
            except (ConnectionError, TimeoutError, RedisError) as e:
                logger.warning(f"Redis health check failed: {e}")
                await self.circuit_breaker.record_failure()
            except asyncio.CancelledError:
                break

    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        fallback: Any = None,
        **kwargs
    ) -> Any:
        """
        Execute Redis operation with retry and circuit breaker.

        Args:
            operation: Async function to execute
            *args: Arguments for operation
            fallback: Value to return on failure
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation or fallback value
        """
        if not await self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker OPEN - returning fallback")
            return fallback

        for attempt in range(self.config.max_retries):
            try:
                result = await operation(*args, **kwargs)
                await self.circuit_breaker.record_success()
                return result

            except (ConnectionError, TimeoutError) as e:
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )

                logger.warning(
                    f"Redis operation failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    await self.circuit_breaker.record_failure()

            except RedisError as e:
                logger.error(f"Redis operation error: {e}")
                await self.circuit_breaker.record_failure()
                break

        return fallback

    async def get(self, key: str, fallback: Any = None) -> Any:
        """Get value with retry."""
        if not self.client:
            return fallback
        return await self.execute_with_retry(self.client.get, key, fallback=fallback)

    async def set(self, key: str, value: str, ex: int = None) -> bool:
        """Set value with retry."""
        if not self.client:
            return False
        result = await self.execute_with_retry(
            self.client.set, key, value, ex=ex, fallback=False
        )
        return result is not False

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set value with expiration."""
        if not self.client:
            return False
        result = await self.execute_with_retry(
            self.client.setex, key, seconds, value, fallback=False
        )
        return result is not False

    async def delete(self, *keys: str) -> int:
        """Delete keys with retry."""
        if not self.client:
            return 0
        return await self.execute_with_retry(
            self.client.delete, *keys, fallback=0
        )

    async def scan_iter(self, match: str = "*", count: int = 100):
        """Scan keys with pattern matching."""
        if not self.client:
            return
        try:
            async for key in self.client.scan_iter(match=match, count=count):
                yield key
        except (ConnectionError, TimeoutError, RedisError) as e:
            logger.warning(f"Redis scan_iter failed: {e}")
            return

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        if not self.client:
            return False
        try:
            await self.client.ping()
            return True
        except (ConnectionError, TimeoutError, RedisError):
            return False

    async def info(self) -> dict:
        """Get Redis server info."""
        if not self.client:
            return {}
        try:
            return await self.client.info()
        except (ConnectionError, TimeoutError, RedisError):
            return {}

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.client is not None

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state."""
        return self.circuit_breaker.state

    async def close(self):
        """Close Redis connection gracefully."""
        self._connected = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self.client:
            await self.client.close()
            self.client = None

        logger.info("Redis connection closed")


# Global resilient Redis client
resilient_redis = ResilientRedisClient()


def with_redis_fallback(fallback_value: Any = None):
    """
    Decorator for graceful Redis failure handling.

    Usage:
        @with_redis_fallback(fallback_value=[])
        async def get_items():
            return await redis.get("items")
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (ConnectionError, TimeoutError, RedisError) as e:
                logger.warning(f"Redis operation failed in {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator
