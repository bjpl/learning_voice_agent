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
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid

        # Check for existing request ID (from load balancer)
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in request state for logging
        request.state.request_id = request_id

        response = await call_next(request)

        # Add to response
        response.headers["X-Request-ID"] = request_id

        return response
