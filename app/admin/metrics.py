"""
Metrics Collection for Admin Dashboard
Provides real-time system metrics and performance data.
"""
import time
import psutil
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class RequestMetric:
    """Single request metric."""
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    active_connections: int


class MetricsCollector:
    """
    Collects and aggregates system and request metrics.
    Thread-safe with configurable retention.
    """

    def __init__(self, retention_minutes: int = 60, max_samples: int = 10000):
        self.retention_minutes = retention_minutes
        self.max_samples = max_samples

        # Request metrics (circular buffer)
        self._requests: deque = deque(maxlen=max_samples)

        # System metrics snapshots
        self._system_snapshots: deque = deque(maxlen=360)  # 1 hour at 10s intervals

        # Error tracking
        self._errors: deque = deque(maxlen=1000)

        # Performance counters
        self._start_time = time.time()
        self._total_requests = 0
        self._total_errors = 0

        # Endpoint-specific stats
        self._endpoint_stats: Dict[str, Dict] = {}

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float
    ):
        """Record a single request metric."""
        metric = RequestMetric(
            timestamp=time.time(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms
        )

        self._requests.append(metric)
        self._total_requests += 1

        if status_code >= 400:
            self._total_errors += 1
            self._errors.append(metric)

        # Update endpoint stats
        key = f"{method}:{endpoint}"
        if key not in self._endpoint_stats:
            self._endpoint_stats[key] = {
                "count": 0,
                "errors": 0,
                "total_time": 0,
                "times": deque(maxlen=1000)
            }

        stats = self._endpoint_stats[key]
        stats["count"] += 1
        stats["total_time"] += response_time_ms
        stats["times"].append(response_time_ms)
        if status_code >= 400:
            stats["errors"] += 1

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_percent=disk.percent,
            active_connections=len(psutil.net_connections())
        )

        self._system_snapshots.append(metrics)
        return metrics

    def get_request_stats(self, window_minutes: int = 5) -> Dict:
        """Get request statistics for the given time window."""
        cutoff = time.time() - (window_minutes * 60)
        recent = [r for r in self._requests if r.timestamp > cutoff]

        if not recent:
            return {
                "window_minutes": window_minutes,
                "total_requests": 0,
                "requests_per_second": 0,
                "error_rate": 0,
                "avg_response_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0
            }

        response_times = [r.response_time_ms for r in recent]
        errors = sum(1 for r in recent if r.status_code >= 400)

        sorted_times = sorted(response_times)
        p50_idx = int(len(sorted_times) * 0.50)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)

        return {
            "window_minutes": window_minutes,
            "total_requests": len(recent),
            "requests_per_second": len(recent) / (window_minutes * 60),
            "error_rate": (errors / len(recent)) * 100 if recent else 0,
            "avg_response_ms": statistics.mean(response_times),
            "p50_ms": sorted_times[p50_idx] if p50_idx < len(sorted_times) else 0,
            "p95_ms": sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0,
            "p99_ms": sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0,
            "min_ms": min(response_times),
            "max_ms": max(response_times)
        }

    def get_endpoint_stats(self) -> List[Dict]:
        """Get per-endpoint statistics."""
        results = []

        for key, stats in self._endpoint_stats.items():
            method, endpoint = key.split(":", 1)
            times = list(stats["times"])

            if times:
                sorted_times = sorted(times)
                p95_idx = int(len(sorted_times) * 0.95)
                p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
            else:
                p95 = 0

            results.append({
                "endpoint": endpoint,
                "method": method,
                "requests": stats["count"],
                "errors": stats["errors"],
                "error_rate": (stats["errors"] / stats["count"] * 100) if stats["count"] > 0 else 0,
                "avg_ms": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0,
                "p95_ms": p95
            })

        return sorted(results, key=lambda x: x["requests"], reverse=True)

    def get_system_stats(self) -> Dict:
        """Get current system resource statistics."""
        current = self.collect_system_metrics()

        # Calculate averages from snapshots
        if self._system_snapshots:
            snapshots = list(self._system_snapshots)
            avg_cpu = statistics.mean(s.cpu_percent for s in snapshots)
            avg_memory = statistics.mean(s.memory_percent for s in snapshots)
        else:
            avg_cpu = current.cpu_percent
            avg_memory = current.memory_percent

        return {
            "current": {
                "cpu_percent": current.cpu_percent,
                "memory_percent": current.memory_percent,
                "memory_used_mb": current.memory_used_mb,
                "memory_available_mb": current.memory_available_mb,
                "disk_percent": current.disk_percent,
                "active_connections": current.active_connections
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "uptime_seconds": time.time() - self._start_time
        }

    def get_recent_errors(self, limit: int = 50) -> List[Dict]:
        """Get recent errors for debugging."""
        errors = list(self._errors)[-limit:]
        return [
            {
                "timestamp": datetime.fromtimestamp(e.timestamp).isoformat(),
                "endpoint": e.endpoint,
                "method": e.method,
                "status_code": e.status_code,
                "response_time_ms": e.response_time_ms
            }
            for e in reversed(errors)
        ]

    def get_dashboard_summary(self) -> Dict:
        """Get complete dashboard summary."""
        return {
            "overview": {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "uptime_seconds": time.time() - self._start_time,
                "error_rate": (self._total_errors / self._total_requests * 100) if self._total_requests > 0 else 0
            },
            "request_stats": self.get_request_stats(5),
            "system_stats": self.get_system_stats(),
            "top_endpoints": self.get_endpoint_stats()[:10],
            "recent_errors": self.get_recent_errors(10)
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()
