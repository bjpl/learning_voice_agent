"""
Middleware for request tracking, metrics, and CDN support.
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.admin.metrics import metrics_collector


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect request metrics for admin dashboard.
    Records timing, status codes, and endpoint data.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Record metric
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time_ms=response_time_ms
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"

        return response


class CDNHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add CDN-friendly cache headers.
    Optimizes static asset delivery through CDN.
    """

    # Static file extensions to cache
    STATIC_EXTENSIONS = {
        '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg',
        '.woff', '.woff2', '.ttf', '.eot', '.ico', '.webp'
    }

    # Cache durations (seconds)
    STATIC_MAX_AGE = 31536000  # 1 year for static assets
    HTML_MAX_AGE = 3600  # 1 hour for HTML
    API_MAX_AGE = 0  # No caching for API

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        path = request.url.path.lower()

        # Skip if already has cache-control
        if "cache-control" in response.headers:
            return response

        # Static assets - long cache with immutable
        if any(path.endswith(ext) for ext in self.STATIC_EXTENSIONS):
            response.headers["Cache-Control"] = f"public, max-age={self.STATIC_MAX_AGE}, immutable"
            response.headers["Vary"] = "Accept-Encoding"

        # HTML pages - short cache with revalidation
        elif path.endswith('.html') or path == '/':
            response.headers["Cache-Control"] = f"public, max-age={self.HTML_MAX_AGE}, must-revalidate"
            response.headers["Vary"] = "Accept-Encoding"

        # API endpoints - no cache
        elif path.startswith('/api/') or path.startswith('/admin/'):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"

        # Default - short cache
        else:
            response.headers["Cache-Control"] = "public, max-age=300"

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers for production deployment.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS for HTTPS deployments
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add unique request ID for tracing and debugging.
    Integrates with centralized logging for correlation ID tracking.

    PATTERN: Request tracing middleware with logger integration
    WHY: Track requests across distributed systems and async operations
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid
        from app.logger import set_correlation_id, clear_correlation_id

        # Check for existing correlation ID (from load balancer or upstream service)
        # Support both X-Request-ID and X-Correlation-ID headers
        correlation_id = (
            request.headers.get("X-Correlation-ID") or
            request.headers.get("X-Request-ID")
        )

        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set correlation ID in context for logging
        # This makes the correlation ID available to all loggers in this request
        set_correlation_id(correlation_id)

        # Store in request state for logging and access in route handlers
        request.state.request_id = correlation_id
        request.state.correlation_id = correlation_id

        try:
            response = await call_next(request)

            # Add correlation ID to response headers for client tracking
            response.headers["X-Request-ID"] = correlation_id
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        finally:
            # CRITICAL: Clear correlation ID to prevent leakage between requests
            # In async contexts, context vars can persist if not cleaned up
            clear_correlation_id()
