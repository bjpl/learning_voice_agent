"""
Monitoring Module - Health Checks and Metrics

SPECIFICATION:
- Comprehensive health check endpoints for Kubernetes/Docker
- Prometheus-compatible metrics collection
- Application info and uptime tracking
- Dependency status monitoring

ARCHITECTURE:
- FastAPI router for endpoint registration
- Async health checks for non-blocking operations
- Singleton pattern for metrics aggregation

EXPORTS:
- Health check components
- Metrics collection utilities
- Monitoring router for FastAPI integration
"""

from app.monitoring.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    DetailedHealthResponse,
    health_checker,
)
from app.monitoring.metrics import (
    MonitoringMetrics,
    LatencyPercentiles,
    monitoring_metrics,
)
from app.monitoring.routes import monitoring_router

__all__ = [
    # Health check components
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "DetailedHealthResponse",
    "health_checker",
    # Metrics components
    "MonitoringMetrics",
    "LatencyPercentiles",
    "monitoring_metrics",
    # Router
    "monitoring_router",
]
