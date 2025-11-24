"""
Observability and Metrics Collection - SPARC Implementation

SPECIFICATION:
- Comprehensive Prometheus metrics for monitoring
- Zero-impact async metrics collection
- Health checks with dependency status
- Application insights and custom metrics

ARCHITECTURE:
- Singleton pattern for metrics registry
- Decorator pattern for automatic instrumentation
- Strategy pattern for different metric types
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
from functools import wraps
import time
from typing import Callable, Optional, Dict, Any
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Create custom registry to avoid conflicts
metrics_registry = CollectorRegistry()

# ============================================================================
# REQUEST METRICS
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests by endpoint and method',
    ['endpoint', 'method', 'status'],
    registry=metrics_registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['endpoint', 'method'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=metrics_registry
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    ['endpoint'],
    registry=metrics_registry
)

# ============================================================================
# ERROR METRICS
# ============================================================================

errors_total = Counter(
    'errors_total',
    'Total errors by type and component',
    ['error_type', 'component'],
    registry=metrics_registry
)

api_errors_total = Counter(
    'api_errors_total',
    'External API errors by provider',
    ['provider', 'error_type'],
    registry=metrics_registry
)

# ============================================================================
# EXTERNAL API METRICS
# ============================================================================

claude_api_duration_seconds = Histogram(
    'claude_api_duration_seconds',
    'Claude API call duration in seconds',
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=metrics_registry
)

claude_api_calls_total = Counter(
    'claude_api_calls_total',
    'Total Claude API calls',
    ['model', 'status'],
    registry=metrics_registry
)

claude_api_tokens_total = Counter(
    'claude_api_tokens_total',
    'Total tokens used in Claude API calls',
    ['model', 'type'],  # type: input/output
    registry=metrics_registry
)

whisper_api_duration_seconds = Histogram(
    'whisper_api_duration_seconds',
    'Whisper API call duration in seconds',
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 10.0),
    registry=metrics_registry
)

whisper_api_calls_total = Counter(
    'whisper_api_calls_total',
    'Total Whisper API calls',
    ['status'],
    registry=metrics_registry
)

# ============================================================================
# WEBSOCKET METRICS
# ============================================================================

websocket_connections_active = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections',
    registry=metrics_registry
)

websocket_messages_total = Counter(
    'websocket_messages_total',
    'Total WebSocket messages',
    ['direction', 'message_type'],  # direction: sent/received
    registry=metrics_registry
)

websocket_connection_duration_seconds = Histogram(
    'websocket_connection_duration_seconds',
    'WebSocket connection duration in seconds',
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
    registry=metrics_registry
)

# ============================================================================
# DATABASE METRICS
# ============================================================================

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=metrics_registry
)

database_queries_total = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'table', 'status'],
    registry=metrics_registry
)

database_connections_active = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=metrics_registry
)

# ============================================================================
# CACHE METRICS
# ============================================================================

cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],  # operation: get/set/delete, status: hit/miss/error
    registry=metrics_registry
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio (hits / total requests)',
    registry=metrics_registry
)

# ============================================================================
# SESSION METRICS
# ============================================================================

sessions_active = Gauge(
    'sessions_active',
    'Number of active user sessions',
    registry=metrics_registry
)

sessions_total = Counter(
    'sessions_total',
    'Total sessions created',
    registry=metrics_registry
)

session_duration_seconds = Histogram(
    'session_duration_seconds',
    'Session duration in seconds',
    buckets=(10, 30, 60, 120, 180, 300, 600, 1800, 3600),
    registry=metrics_registry
)

# ============================================================================
# CONVERSATION METRICS
# ============================================================================

conversation_exchanges_total = Counter(
    'conversation_exchanges_total',
    'Total conversation exchanges',
    ['intent'],
    registry=metrics_registry
)

conversation_quality_score = Histogram(
    'conversation_quality_score',
    'Conversation quality score (0-1)',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=metrics_registry
)

user_engagement_score = Gauge(
    'user_engagement_score',
    'Current user engagement score',
    ['session_id'],
    registry=metrics_registry
)

# ============================================================================
# AUDIO PROCESSING METRICS
# ============================================================================

audio_processing_duration_seconds = Histogram(
    'audio_processing_duration_seconds',
    'Audio processing duration in seconds',
    ['operation'],  # operation: transcribe/validate/decode
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0),
    registry=metrics_registry
)

audio_size_bytes = Histogram(
    'audio_size_bytes',
    'Audio file size in bytes',
    ['format'],
    buckets=(1024, 10240, 51200, 102400, 512000, 1048576, 5242880, 10485760),
    registry=metrics_registry
)

# ============================================================================
# COST TRACKING METRICS
# ============================================================================

api_cost_total = Counter(
    'api_cost_total',
    'Total estimated API costs in USD',
    ['provider'],
    registry=metrics_registry
)

# ============================================================================
# DECORATOR UTILITIES
# ============================================================================

def track_time(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to track execution time of functions

    Usage:
        @track_time(http_request_duration_seconds, {'endpoint': '/api/conversation', 'method': 'POST'})
        async def handler():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def count_errors(component: str):
    """
    Decorator to count errors by type

    Usage:
        @count_errors('conversation_handler')
        async def generate_response():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                errors_total.labels(error_type=error_type, component=component).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                errors_total.labels(error_type=error_type, component=component).inc()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@asynccontextmanager
async def track_in_progress(gauge: Gauge, labels: Optional[Dict[str, str]] = None):
    """
    Context manager to track in-progress requests

    Usage:
        async with track_in_progress(http_requests_in_progress, {'endpoint': '/api'}):
            await process_request()
    """
    if labels:
        gauge.labels(**labels).inc()
    else:
        gauge.inc()
    try:
        yield
    finally:
        if labels:
            gauge.labels(**labels).dec()
        else:
            gauge.dec()


class MetricsCollector:
    """
    Central metrics collector with convenience methods
    """

    def __init__(self):
        self.registry = metrics_registry
        self._cache_hits = 0
        self._cache_requests = 0

    def track_http_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration: float
    ):
        """Track HTTP request metrics"""
        http_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()

        http_request_duration_seconds.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)

    def track_claude_call(
        self,
        model: str,
        duration: float,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Track Claude API call metrics"""
        claude_api_calls_total.labels(model=model, status=status).inc()
        claude_api_duration_seconds.observe(duration)

        if input_tokens:
            claude_api_tokens_total.labels(model=model, type='input').inc(input_tokens)
        if output_tokens:
            claude_api_tokens_total.labels(model=model, type='output').inc(output_tokens)

        # Rough cost estimation (Claude Haiku pricing)
        input_cost = input_tokens * 0.00000025  # $0.25 per 1M input tokens
        output_cost = output_tokens * 0.00000125  # $1.25 per 1M output tokens
        api_cost_total.labels(provider='anthropic').inc(input_cost + output_cost)

    def track_whisper_call(self, duration: float, status: str, audio_duration: float = 0):
        """Track Whisper API call metrics"""
        whisper_api_calls_total.labels(status=status).inc()
        whisper_api_duration_seconds.observe(duration)

        # Rough cost estimation (Whisper pricing: $0.006 per minute)
        if audio_duration > 0:
            cost = (audio_duration / 60.0) * 0.006
            api_cost_total.labels(provider='openai').inc(cost)

    def track_database_query(
        self,
        operation: str,
        table: str,
        duration: float,
        status: str = 'success'
    ):
        """Track database query metrics"""
        database_queries_total.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()

        database_query_duration_seconds.labels(
            operation=operation,
            table=table
        ).observe(duration)

    def track_cache_operation(self, operation: str, hit: bool = False):
        """Track cache operation metrics"""
        status = 'hit' if hit else 'miss' if operation == 'get' else 'success'
        cache_operations_total.labels(operation=operation, status=status).inc()

        # Update hit ratio
        if operation == 'get':
            self._cache_requests += 1
            if hit:
                self._cache_hits += 1

            if self._cache_requests > 0:
                ratio = self._cache_hits / self._cache_requests
                cache_hit_ratio.set(ratio)

    def track_session(self, session_id: str, created: bool = False):
        """Track session metrics"""
        if created:
            sessions_total.inc()

        # This should be called periodically to update active sessions

    def track_conversation_exchange(self, intent: str, quality_score: float = None):
        """Track conversation metrics"""
        conversation_exchanges_total.labels(intent=intent).inc()

        if quality_score is not None:
            conversation_quality_score.observe(quality_score)

    def track_audio_processing(
        self,
        operation: str,
        duration: float,
        format: str = 'unknown',
        size_bytes: int = 0
    ):
        """Track audio processing metrics"""
        audio_processing_duration_seconds.labels(operation=operation).observe(duration)

        if size_bytes > 0:
            audio_size_bytes.labels(format=format).observe(size_bytes)

    def update_active_sessions(self, count: int):
        """Update active sessions gauge"""
        sessions_active.set(count)

    def update_active_websockets(self, count: int):
        """Update active WebSocket connections gauge"""
        websocket_connections_active.set(count)

    def update_database_connections(self, count: int):
        """Update active database connections gauge"""
        database_connections_active.set(count)

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary for JSON endpoints"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'application': {
                'name': 'learning_voice_agent',
                'version': '1.0.0'
            },
            'requests': {
                'total': self._get_counter_value(http_requests_total),
                'in_progress': self._get_gauge_value(http_requests_in_progress)
            },
            'sessions': {
                'active': self._get_gauge_value(sessions_active),
                'total': self._get_counter_value(sessions_total)
            },
            'websockets': {
                'active': self._get_gauge_value(websocket_connections_active)
            },
            'errors': {
                'total': self._get_counter_value(errors_total)
            },
            'external_apis': {
                'claude': {
                    'calls': self._get_counter_value(claude_api_calls_total)
                },
                'whisper': {
                    'calls': self._get_counter_value(whisper_api_calls_total)
                }
            },
            'cache': {
                'hit_ratio': self._get_gauge_value(cache_hit_ratio)
            },
            'costs': {
                'total_usd': self._get_counter_value(api_cost_total)
            }
        }

    def _get_counter_value(self, counter: Counter) -> float:
        """Get total value from counter (sum across all labels)"""
        try:
            samples = counter.collect()[0].samples
            return sum(sample.value for sample in samples)
        except (IndexError, AttributeError, TypeError) as e:
            # IndexError: No samples collected yet
            # AttributeError: Counter not properly initialized
            # TypeError: Unexpected sample structure
            return 0.0

    def _get_gauge_value(self, gauge: Gauge) -> float:
        """Get value from gauge (sum across all labels)"""
        try:
            samples = gauge.collect()[0].samples
            return sum(sample.value for sample in samples)
        except (IndexError, AttributeError, TypeError) as e:
            # IndexError: No samples collected yet
            # AttributeError: Gauge not properly initialized
            # TypeError: Unexpected sample structure
            return 0.0


# Global metrics collector instance
metrics_collector = MetricsCollector()
