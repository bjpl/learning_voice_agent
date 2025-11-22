"""
Monitoring Routes - FastAPI Router

SPECIFICATION:
- Mount health endpoints on /health/*
- Metrics endpoint with Prometheus format option
- Info endpoint with version, uptime, environment

ARCHITECTURE:
- APIRouter for modular endpoint registration
- Dependency injection for health checker
- Response models for consistent API

PATTERN: Router composition pattern
WHY: Keep monitoring endpoints organized and reusable
"""

import time
import os
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Response, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.monitoring.health import health_checker, HealthStatus
from app.monitoring.metrics import monitoring_metrics
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


# Response Models
class BasicHealthResponse(BaseModel):
    """Basic health check response"""
    status: str
    timestamp: str
    service: str


class LivenessResponse(BaseModel):
    """Liveness probe response"""
    status: str
    timestamp: str


class InfoResponse(BaseModel):
    """Application info response"""
    service: str
    version: str
    environment: str
    uptime_seconds: float
    timestamp: str
    python_version: str
    host: str
    port: int


# Create router
monitoring_router = APIRouter(tags=["Monitoring"])


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@monitoring_router.get(
    "/health",
    response_model=BasicHealthResponse,
    summary="Basic Health Check",
    description="Simple health check that returns 200 if the service is running",
)
async def health_check() -> BasicHealthResponse:
    """
    Basic health check endpoint

    PATTERN: Simple ping for load balancers
    WHY: Fastest possible health verification

    Returns:
        BasicHealthResponse with status and timestamp
    """
    result = await health_checker.check_basic()
    return BasicHealthResponse(**result)


@monitoring_router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness Probe",
    description="Kubernetes liveness probe - returns 200 if process is alive",
)
async def liveness_probe() -> LivenessResponse:
    """
    Liveness probe for Kubernetes

    PATTERN: Kubernetes liveness probe
    WHY: Detect stuck or deadlocked processes

    Returns:
        LivenessResponse indicating process is alive
    """
    result = await health_checker.check_liveness()
    return LivenessResponse(**result)


@monitoring_router.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Kubernetes readiness probe - checks if service can accept traffic",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_probe() -> Response:
    """
    Readiness probe for Kubernetes

    PATTERN: Kubernetes readiness probe
    WHY: Only route traffic to ready instances

    Checks:
    - Database connectivity
    - Vector store availability
    - API key configuration

    Returns:
        JSON response with appropriate status code
    """
    result, status_code = await health_checker.check_readiness()
    return JSONResponse(content=result, status_code=status_code)


@monitoring_router.get(
    "/health/detailed",
    summary="Detailed Health Check",
    description="Comprehensive health status of all system components",
    responses={
        200: {"description": "All components healthy or degraded"},
        503: {"description": "One or more critical components unhealthy"},
    },
)
async def detailed_health_check() -> Response:
    """
    Detailed health check with all component statuses

    PATTERN: Full system health dashboard
    WHY: Debugging and monitoring dashboards need details

    Checks:
    - Database (SQLite)
    - Vector Store (ChromaDB)
    - API Configuration
    - Disk Space

    Returns:
        Detailed JSON response with component statuses
    """
    result, status_code = await health_checker.check_detailed()
    return JSONResponse(content=result, status_code=status_code)


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@monitoring_router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Metrics in Prometheus text format for scraping",
)
async def prometheus_metrics_endpoint(
    format: str = Query(
        "prometheus",
        description="Output format: 'prometheus' or 'json'",
        regex="^(prometheus|json)$",
    ),
) -> Response:
    """
    Metrics endpoint supporting Prometheus and JSON formats

    PATTERN: Multi-format metrics export
    WHY: Support both Prometheus scraping and dashboard APIs

    Args:
        format: Output format ('prometheus' or 'json')

    Returns:
        Metrics in requested format
    """
    if format == "json":
        return JSONResponse(content=monitoring_metrics.get_summary())

    # Prometheus format
    metrics_text = monitoring_metrics.get_prometheus_metrics()
    return Response(
        content=metrics_text,
        media_type="text/plain; charset=utf-8",
    )


@monitoring_router.get(
    "/metrics/json",
    summary="JSON Metrics",
    description="Metrics in JSON format for dashboards",
)
async def json_metrics_endpoint() -> Dict[str, Any]:
    """
    JSON metrics endpoint for dashboards

    PATTERN: Human-readable metrics export
    WHY: Easy integration with custom dashboards

    Returns:
        Comprehensive metrics dictionary
    """
    return monitoring_metrics.get_summary()


@monitoring_router.get(
    "/metrics/latency",
    summary="Latency Metrics",
    description="Request latency percentiles",
)
async def latency_metrics_endpoint(
    window_seconds: int = Query(
        300,
        description="Time window in seconds",
        ge=60,
        le=3600,
    ),
    endpoint: Optional[str] = Query(
        None,
        description="Filter by endpoint path",
    ),
) -> Dict[str, Any]:
    """
    Latency percentiles for requests

    Args:
        window_seconds: Time window for calculation
        endpoint: Optional endpoint filter

    Returns:
        Latency percentile statistics
    """
    stats = monitoring_metrics.get_request_latency_stats(
        window_seconds=window_seconds,
        endpoint=endpoint,
    )

    if stats is None:
        return {
            "message": "No requests in time window",
            "window_seconds": window_seconds,
            "endpoint": endpoint,
        }

    return {
        "window_seconds": window_seconds,
        "endpoint": endpoint or "all",
        "latency": stats.to_dict(),
    }


@monitoring_router.get(
    "/metrics/errors",
    summary="Error Metrics",
    description="Error rates and breakdown by type",
)
async def error_metrics_endpoint(
    window_seconds: int = Query(
        300,
        description="Time window in seconds",
        ge=60,
        le=3600,
    ),
) -> Dict[str, Any]:
    """
    Error metrics and breakdown

    Args:
        window_seconds: Time window for error rate calculation

    Returns:
        Error statistics and breakdown
    """
    return {
        "window_seconds": window_seconds,
        "error_rate_percent": round(
            monitoring_metrics.get_error_rate(window_seconds), 2
        ),
        "error_breakdown": monitoring_metrics.get_error_breakdown(),
        "total_errors": monitoring_metrics._total_errors,
    }


@monitoring_router.get(
    "/metrics/conversations",
    summary="Conversation Metrics",
    description="Conversation and session statistics",
)
async def conversation_metrics_endpoint() -> Dict[str, Any]:
    """
    Conversation statistics

    Returns:
        Conversation and session metrics
    """
    stats = monitoring_metrics.get_conversation_stats()
    return {
        "sessions": {
            "active": stats.active_sessions,
            "total": stats.total_sessions,
            "avg_duration_seconds": stats.avg_session_duration_seconds,
        },
        "exchanges": {
            "total": stats.total_exchanges,
            "per_hour": stats.exchanges_per_hour,
        },
        "intents": stats.intents,
    }


@monitoring_router.get(
    "/metrics/resources",
    summary="Resource Metrics",
    description="Memory and CPU usage statistics",
)
async def resource_metrics_endpoint() -> Dict[str, Any]:
    """
    System resource usage

    Returns:
        Memory and CPU statistics
    """
    return {
        "memory": monitoring_metrics.get_memory_usage(),
        "cpu": monitoring_metrics.get_cpu_usage(),
    }


# ============================================================================
# INFO ENDPOINT
# ============================================================================

@monitoring_router.get(
    "/info",
    response_model=InfoResponse,
    summary="Application Info",
    description="Application version, uptime, and environment information",
)
async def info_endpoint() -> InfoResponse:
    """
    Application information endpoint

    PATTERN: Service metadata exposure
    WHY: Debugging, version tracking, environment verification

    Returns:
        InfoResponse with service details
    """
    import sys

    return InfoResponse(
        service="learning-voice-agent",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        uptime_seconds=round(health_checker.get_uptime_seconds(), 2),
        timestamp=datetime.utcnow().isoformat(),
        python_version=sys.version.split()[0],
        host=settings.host,
        port=settings.port,
    )


# ============================================================================
# INITIALIZATION HELPER
# ============================================================================

def initialize_monitoring(startup_time: float) -> None:
    """
    Initialize monitoring with application startup time

    Should be called during FastAPI lifespan startup

    Args:
        startup_time: Application startup timestamp
    """
    health_checker.set_startup_time(startup_time)
    logger.info(
        "monitoring_initialized",
        startup_time=startup_time,
    )
