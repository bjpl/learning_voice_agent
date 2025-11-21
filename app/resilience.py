"""
Resilience Engineering Utilities - SPARC Implementation

SPECIFICATION:
- Circuit breaker pattern for external service failures
- Exponential backoff retry logic with configurable attempts
- Timeout handling for all external calls
- Fallback response handlers for graceful degradation
- Health check utilities for monitoring service status

PSEUDOCODE:
1. Circuit breaker: Track failures, open circuit after threshold
2. Retry decorator: Wrap functions with exponential backoff
3. Timeout wrapper: Add timeout to async operations
4. Fallback handler: Return default responses on failures
5. Health checker: Test service availability

ARCHITECTURE:
- Decorator pattern for retry and timeout
- State pattern for circuit breaker
- Strategy pattern for fallback handlers

CODE:
"""
import asyncio
import time
from typing import Callable, Optional, Any, Dict, TypeVar, Generic
from functools import wraps
from enum import Enum
from dataclasses import dataclass
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from circuitbreaker import circuit, CircuitBreakerError

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers"""
    failure_threshold: int = 3  # Failures before opening
    recovery_timeout: int = 60  # Seconds before retry
    expected_exception: type = Exception


class ResilientCircuitBreaker:
    """
    PATTERN: Circuit breaker with state management
    WHY: Prevent cascading failures and allow recovery
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _on_success(self):
        """Reset circuit on success"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Record failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def with_retry(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: int = 2,
    exceptions: tuple = (Exception,)
):
    """
    PATTERN: Retry decorator with exponential backoff
    WHY: Transient failures should be retried before giving up

    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time between retries
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to retry on
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=initial_wait,
            max=max_wait,
            exp_base=exponential_base
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG)
    )


def with_timeout(seconds: float):
    """
    PATTERN: Timeout decorator for async functions
    WHY: Prevent indefinite hangs on slow external services

    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {seconds}s in {func.__name__}")
                raise TimeoutError(f"{func.__name__} exceeded timeout of {seconds}s")
        return wrapper
    return decorator


def with_circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
):
    """
    PATTERN: Circuit breaker decorator
    WHY: Prevent repeated calls to failing services

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        expected_exception: Exception type to track
    """
    def decorator(func: Callable):
        # Use circuitbreaker library's built-in decorator
        return circuit(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=func.__name__
        )(func)
    return decorator


class FallbackHandler(Generic[T]):
    """
    PATTERN: Fallback handler for graceful degradation
    WHY: System should continue functioning even when services fail
    """

    def __init__(self, fallback_value: T, log_failures: bool = True):
        self.fallback_value = fallback_value
        self.log_failures = log_failures

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function with fallback on failure"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if self.log_failures:
                logger.error(f"Function {func.__name__} failed: {e}. Using fallback.")
            return self.fallback_value


class HealthCheck:
    """
    PATTERN: Health check utilities for service monitoring
    WHY: Know which services are healthy before making calls
    """

    @staticmethod
    async def check_redis(redis_client) -> Dict[str, Any]:
        """Check Redis connection health"""
        try:
            start_time = time.time()
            await redis_client.ping()
            latency = (time.time() - start_time) * 1000  # Convert to ms

            info = await redis_client.info()

            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @staticmethod
    async def check_database(db_path: str) -> Dict[str, Any]:
        """Check SQLite database health"""
        try:
            import aiosqlite
            start_time = time.time()

            async with aiosqlite.connect(db_path) as db:
                await db.execute("SELECT 1")

            latency = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "path": db_path
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @staticmethod
    async def check_api_service(
        client,
        service_name: str,
        test_call: Callable
    ) -> Dict[str, Any]:
        """Check external API service health"""
        try:
            start_time = time.time()
            await test_call()
            latency = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "service": service_name,
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": service_name,
                "error": str(e)
            }


class RateLimiter:
    """
    PATTERN: Token bucket rate limiter
    WHY: Protect against abuse and manage API quotas
    """

    def __init__(self, max_calls: int, time_window: int):
        """
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, list] = {}

    async def acquire(self, key: str) -> bool:
        """
        Check if call is allowed under rate limit

        Args:
            key: Identifier for rate limiting (e.g., IP address, session ID)

        Returns:
            True if call is allowed, False if rate limited
        """
        now = time.time()

        # Initialize or get call history
        if key not in self.calls:
            self.calls[key] = []

        # Remove old calls outside time window
        self.calls[key] = [
            call_time for call_time in self.calls[key]
            if now - call_time < self.time_window
        ]

        # Check if under limit
        if len(self.calls[key]) < self.max_calls:
            self.calls[key].append(now)
            return True

        return False

    def get_retry_after(self, key: str) -> Optional[int]:
        """Get seconds until rate limit resets"""
        if key not in self.calls or not self.calls[key]:
            return None

        oldest_call = min(self.calls[key])
        retry_after = self.time_window - (time.time() - oldest_call)

        return max(0, int(retry_after))


# Pre-configured resilience patterns for common use cases

# Claude API: Circuit breaker with 3 failures, 10s timeout
claude_resilient = lambda func: with_circuit_breaker(
    failure_threshold=3,
    recovery_timeout=60
)(with_timeout(10)(with_retry(max_attempts=3, initial_wait=1.0)(func)))

# Whisper API: Retry with longer timeout
whisper_resilient = lambda func: with_timeout(15)(
    with_retry(max_attempts=2, initial_wait=2.0)(func)
)

# Redis: Connection retry with short timeout
redis_resilient = lambda func: with_retry(
    max_attempts=3,
    initial_wait=0.5,
    max_wait=2.0
)(func)

# Database: Transaction retry
db_resilient = lambda func: with_retry(
    max_attempts=3,
    initial_wait=0.5
)(func)
