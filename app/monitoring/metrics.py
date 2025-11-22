"""
Metrics Collection for Monitoring

SPECIFICATION:
- Request count and latency percentiles
- Active sessions count
- Error rates by type
- Memory usage tracking
- Conversation statistics

ARCHITECTURE:
- Rolling window statistics
- Thread-safe counters
- Prometheus format support
- JSON export for dashboards

PATTERN: Singleton metrics collector
WHY: Centralized metrics with consistent state
"""

import time
import threading
from collections import deque

# Optional psutil import for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from statistics import mean, quantiles

from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyPercentiles:
    """Latency statistics with percentiles"""
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class RequestMetrics:
    """Request-level metrics"""
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""
    error_type: str
    endpoint: Optional[str]
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationStats:
    """Conversation statistics"""
    total_sessions: int = 0
    active_sessions: int = 0
    total_exchanges: int = 0
    exchanges_per_hour: float = 0.0
    avg_session_duration_seconds: float = 0.0
    intents: Dict[str, int] = field(default_factory=dict)


class MonitoringMetrics:
    """
    Comprehensive metrics collection for monitoring

    PATTERN: Rolling window statistics with thread safety
    WHY: Accurate percentiles without unbounded memory

    Features:
    - Request latency tracking with percentiles
    - Error rate calculation by window
    - Active session monitoring
    - Memory and CPU usage
    - Conversation statistics

    Window sizes:
    - 1 minute: Real-time metrics
    - 5 minutes: Short-term trends
    - 1 hour: Hourly aggregates
    """

    def __init__(
        self,
        window_size_minutes: int = 60,
        max_samples: int = 10000,
    ):
        self._window_size = window_size_minutes * 60  # Convert to seconds
        self._max_samples = max_samples
        self._lock = threading.Lock()

        # Request metrics (rolling window)
        self._requests: deque = deque(maxlen=max_samples)
        self._errors: deque = deque(maxlen=max_samples)

        # Counters (lifetime)
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_conversations: int = 0
        self._total_exchanges: int = 0

        # Session tracking
        self._active_sessions: Dict[str, float] = {}  # session_id -> start_time
        self._session_durations: deque = deque(maxlen=1000)

        # Intent tracking
        self._intent_counts: Dict[str, int] = {}

        # Error tracking by type
        self._error_counts: Dict[str, int] = {}

        # Startup time
        self._startup_time: float = time.time()

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """
        Record an HTTP request metric

        Args:
            endpoint: Request endpoint path
            method: HTTP method
            status_code: Response status code
            latency_ms: Request latency in milliseconds
        """
        with self._lock:
            self._requests.append(RequestMetrics(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                latency_ms=latency_ms,
            ))
            self._total_requests += 1

            # Track errors (4xx and 5xx)
            if status_code >= 400:
                self._total_errors += 1

    def record_error(
        self,
        error_type: str,
        message: str,
        endpoint: Optional[str] = None,
    ) -> None:
        """
        Record an error occurrence

        Args:
            error_type: Type/class of error
            message: Error message
            endpoint: Associated endpoint (if any)
        """
        with self._lock:
            self._errors.append(ErrorMetrics(
                error_type=error_type,
                endpoint=endpoint,
                message=message,
            ))
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    def record_session_start(self, session_id: str) -> None:
        """Record a new session start"""
        with self._lock:
            self._active_sessions[session_id] = time.time()
            self._total_conversations += 1

    def record_session_end(self, session_id: str) -> None:
        """Record a session end"""
        with self._lock:
            if session_id in self._active_sessions:
                duration = time.time() - self._active_sessions[session_id]
                self._session_durations.append(duration)
                del self._active_sessions[session_id]

    def record_exchange(self, intent: Optional[str] = None) -> None:
        """Record a conversation exchange"""
        with self._lock:
            self._total_exchanges += 1
            if intent:
                self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1

    def update_active_sessions(self, count: int) -> None:
        """Update active session count from external source"""
        # This allows syncing with state_manager's session count
        pass  # Active sessions are tracked internally

    def get_request_latency_stats(
        self,
        window_seconds: Optional[int] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[LatencyPercentiles]:
        """
        Calculate latency percentiles for requests

        Args:
            window_seconds: Time window (default: all samples)
            endpoint: Filter by endpoint (default: all)

        Returns:
            LatencyPercentiles or None if no data
        """
        with self._lock:
            cutoff = time.time() - (window_seconds or self._window_size)

            # Filter requests
            latencies = [
                r.latency_ms
                for r in self._requests
                if r.timestamp >= cutoff
                and (endpoint is None or r.endpoint == endpoint)
            ]

            if not latencies:
                return None

            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            # Manual percentile calculation for small datasets
            def percentile(data: List[float], p: float) -> float:
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (k - f) * (data[c] - data[f]) if f != c else data[f]

            return LatencyPercentiles(
                count=n,
                mean_ms=mean(latencies),
                p50_ms=percentile(sorted_latencies, 50),
                p90_ms=percentile(sorted_latencies, 90),
                p95_ms=percentile(sorted_latencies, 95),
                p99_ms=percentile(sorted_latencies, 99),
                min_ms=min(latencies),
                max_ms=max(latencies),
            )

    def get_request_count(
        self,
        window_seconds: Optional[int] = None,
        status_code: Optional[int] = None,
    ) -> int:
        """
        Get request count within time window

        Args:
            window_seconds: Time window (default: all samples)
            status_code: Filter by status code (default: all)

        Returns:
            Request count
        """
        with self._lock:
            cutoff = time.time() - (window_seconds or self._window_size)

            return sum(
                1
                for r in self._requests
                if r.timestamp >= cutoff
                and (status_code is None or r.status_code == status_code)
            )

    def get_error_rate(self, window_seconds: int = 300) -> float:
        """
        Calculate error rate (4xx + 5xx / total) for time window

        Args:
            window_seconds: Time window for calculation

        Returns:
            Error rate as percentage (0-100)
        """
        with self._lock:
            cutoff = time.time() - window_seconds

            total = sum(1 for r in self._requests if r.timestamp >= cutoff)
            errors = sum(
                1
                for r in self._requests
                if r.timestamp >= cutoff and r.status_code >= 400
            )

            if total == 0:
                return 0.0

            return (errors / total) * 100

    def get_active_sessions_count(self) -> int:
        """Get current active session count"""
        with self._lock:
            # Clean up stale sessions (older than 30 minutes)
            cutoff = time.time() - 1800
            self._active_sessions = {
                k: v
                for k, v in self._active_sessions.items()
                if v >= cutoff
            }
            return len(self._active_sessions)

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics

        Returns:
            Memory usage dictionary with RSS, percent, etc.
        """
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not installed", "available": False}

        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
                "percent": round(process.memory_percent(), 2),
                "available": True,
            }
        except Exception as e:
            logger.warning("memory_usage_check_failed", error=str(e))
            return {"error": str(e), "available": False}

    def get_cpu_usage(self) -> Dict[str, Any]:
        """
        Get current CPU usage statistics

        Returns:
            CPU usage dictionary
        """
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not installed", "available": False}

        try:
            process = psutil.Process()

            return {
                "percent": round(process.cpu_percent(interval=0.1), 2),
                "num_threads": process.num_threads(),
                "available": True,
            }
        except Exception as e:
            logger.warning("cpu_usage_check_failed", error=str(e))
            return {"error": str(e), "available": False}

    def get_conversation_stats(self) -> ConversationStats:
        """
        Get comprehensive conversation statistics

        Returns:
            ConversationStats dataclass
        """
        with self._lock:
            # Calculate exchanges per hour
            uptime_hours = (time.time() - self._startup_time) / 3600
            exchanges_per_hour = (
                self._total_exchanges / uptime_hours if uptime_hours > 0 else 0
            )

            # Calculate average session duration
            avg_duration = (
                mean(self._session_durations)
                if self._session_durations
                else 0.0
            )

            return ConversationStats(
                total_sessions=self._total_conversations,
                active_sessions=len(self._active_sessions),
                total_exchanges=self._total_exchanges,
                exchanges_per_hour=round(exchanges_per_hour, 2),
                avg_session_duration_seconds=round(avg_duration, 2),
                intents=dict(self._intent_counts),
            )

    def get_error_breakdown(self) -> Dict[str, int]:
        """Get error counts by type"""
        with self._lock:
            return dict(self._error_counts)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary

        Returns:
            Dictionary with all metrics for JSON export
        """
        latency_stats = self.get_request_latency_stats(window_seconds=300)
        conversation_stats = self.get_conversation_stats()

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(time.time() - self._startup_time, 2),
            "requests": {
                "total": self._total_requests,
                "last_5_minutes": self.get_request_count(window_seconds=300),
                "last_hour": self.get_request_count(window_seconds=3600),
                "error_rate_percent": round(self.get_error_rate(300), 2),
            },
            "latency": latency_stats.to_dict() if latency_stats else None,
            "sessions": {
                "active": conversation_stats.active_sessions,
                "total": conversation_stats.total_sessions,
            },
            "conversations": {
                "total_exchanges": conversation_stats.total_exchanges,
                "exchanges_per_hour": conversation_stats.exchanges_per_hour,
                "avg_session_duration_seconds": conversation_stats.avg_session_duration_seconds,
                "intents": conversation_stats.intents,
            },
            "errors": {
                "total": self._total_errors,
                "by_type": self.get_error_breakdown(),
            },
            "resources": {
                "memory": self.get_memory_usage(),
                "cpu": self.get_cpu_usage(),
            },
        }

        return summary

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus text format

        Returns:
            Prometheus-compatible metrics string
        """
        lines = []
        latency_stats = self.get_request_latency_stats(window_seconds=300)
        conversation_stats = self.get_conversation_stats()
        memory = self.get_memory_usage()

        # Request metrics
        lines.append("# HELP lva_requests_total Total HTTP requests")
        lines.append("# TYPE lva_requests_total counter")
        lines.append(f"lva_requests_total {self._total_requests}")

        lines.append("# HELP lva_errors_total Total errors")
        lines.append("# TYPE lva_errors_total counter")
        lines.append(f"lva_errors_total {self._total_errors}")

        # Latency metrics
        if latency_stats:
            lines.append("# HELP lva_request_latency_ms Request latency in milliseconds")
            lines.append("# TYPE lva_request_latency_ms summary")
            lines.append(f'lva_request_latency_ms{{quantile="0.5"}} {latency_stats.p50_ms}')
            lines.append(f'lva_request_latency_ms{{quantile="0.9"}} {latency_stats.p90_ms}')
            lines.append(f'lva_request_latency_ms{{quantile="0.95"}} {latency_stats.p95_ms}')
            lines.append(f'lva_request_latency_ms{{quantile="0.99"}} {latency_stats.p99_ms}')
            lines.append(f"lva_request_latency_ms_sum {latency_stats.mean_ms * latency_stats.count}")
            lines.append(f"lva_request_latency_ms_count {latency_stats.count}")

        # Session metrics
        lines.append("# HELP lva_active_sessions Number of active sessions")
        lines.append("# TYPE lva_active_sessions gauge")
        lines.append(f"lva_active_sessions {conversation_stats.active_sessions}")

        lines.append("# HELP lva_total_sessions Total sessions created")
        lines.append("# TYPE lva_total_sessions counter")
        lines.append(f"lva_total_sessions {conversation_stats.total_sessions}")

        # Exchange metrics
        lines.append("# HELP lva_total_exchanges Total conversation exchanges")
        lines.append("# TYPE lva_total_exchanges counter")
        lines.append(f"lva_total_exchanges {conversation_stats.total_exchanges}")

        # Memory metrics
        if "rss_mb" in memory:
            lines.append("# HELP lva_memory_rss_mb Memory RSS in MB")
            lines.append("# TYPE lva_memory_rss_mb gauge")
            lines.append(f"lva_memory_rss_mb {memory['rss_mb']}")

        # Error rate
        lines.append("# HELP lva_error_rate_percent Error rate percentage")
        lines.append("# TYPE lva_error_rate_percent gauge")
        lines.append(f"lva_error_rate_percent {self.get_error_rate(300):.2f}")

        # Intent metrics
        lines.append("# HELP lva_intent_count Conversation intents count")
        lines.append("# TYPE lva_intent_count counter")
        for intent, count in conversation_stats.intents.items():
            lines.append(f'lva_intent_count{{intent="{intent}"}} {count}')

        return "\n".join(lines)


# Global monitoring metrics instance
monitoring_metrics = MonitoringMetrics()
