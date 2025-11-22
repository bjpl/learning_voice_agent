"""
Tests for Monitoring Module

SPECIFICATION:
- Test health check endpoints
- Test metrics collection
- Test route configuration
"""

import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

# Import monitoring components
from app.monitoring.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    health_checker,
)
from app.monitoring.metrics import (
    MonitoringMetrics,
    LatencyPercentiles,
    RequestMetrics,
    monitoring_metrics,
)


class TestHealthStatus:
    """Tests for HealthStatus enum"""

    def test_health_status_values(self):
        """Verify health status values"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass"""

    def test_component_health_creation(self):
        """Test creating ComponentHealth instance"""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            latency_ms=5.5,
            message="All good",
            details={"key": "value"},
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms == 5.5
        assert health.message == "All good"
        assert health.details == {"key": "value"}

    def test_component_health_to_dict(self):
        """Test converting ComponentHealth to dictionary"""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            latency_ms=10.123,
        )

        result = health.to_dict()

        assert result["name"] == "test"
        assert result["status"] == "healthy"
        assert result["latency_ms"] == 10.12
        assert "last_check" in result


class TestHealthChecker:
    """Tests for HealthChecker class"""

    def test_uptime_tracking(self):
        """Test uptime calculation"""
        checker = HealthChecker()
        checker.set_startup_time(time.time() - 100)

        uptime = checker.get_uptime_seconds()

        assert uptime >= 100
        assert uptime < 105

    @pytest.mark.asyncio
    async def test_basic_health_check(self):
        """Test basic health check returns healthy"""
        checker = HealthChecker()

        result = await checker.check_basic()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert result["service"] == "learning-voice-agent"

    @pytest.mark.asyncio
    async def test_liveness_check(self):
        """Test liveness probe returns alive"""
        checker = HealthChecker()

        result = await checker.check_liveness()

        assert result["status"] == "alive"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_api_configuration_check_no_keys(self):
        """Test API configuration check with no keys"""
        checker = HealthChecker()

        with patch("app.monitoring.health.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            mock_settings.openai_api_key = ""

            result = await checker.check_api_configuration()

            assert result.status == HealthStatus.UNHEALTHY
            assert "not configured" in result.message.lower()

    @pytest.mark.asyncio
    async def test_disk_space_check(self):
        """Test disk space check returns valid data"""
        checker = HealthChecker()

        result = await checker.check_disk_space()

        assert result.name == "disk_space"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
        ]
        assert "free_gb" in result.details or "error" in result.details


class TestMonitoringMetrics:
    """Tests for MonitoringMetrics class"""

    def test_record_request(self):
        """Test recording HTTP requests"""
        metrics = MonitoringMetrics()

        metrics.record_request(
            endpoint="/api/test",
            method="GET",
            status_code=200,
            latency_ms=50.0,
        )

        assert metrics._total_requests == 1
        assert len(metrics._requests) == 1

    def test_record_error(self):
        """Test recording errors"""
        metrics = MonitoringMetrics()

        metrics.record_error(
            error_type="ValueError",
            message="Test error",
            endpoint="/api/test",
        )

        assert "ValueError" in metrics._error_counts
        assert metrics._error_counts["ValueError"] == 1

    def test_session_tracking(self):
        """Test session start/end tracking"""
        metrics = MonitoringMetrics()

        metrics.record_session_start("session-123")
        assert "session-123" in metrics._active_sessions

        metrics.record_session_end("session-123")
        assert "session-123" not in metrics._active_sessions
        assert len(metrics._session_durations) == 1

    def test_exchange_tracking(self):
        """Test conversation exchange tracking"""
        metrics = MonitoringMetrics()

        metrics.record_exchange(intent="question")
        metrics.record_exchange(intent="question")
        metrics.record_exchange(intent="greeting")

        assert metrics._total_exchanges == 3
        assert metrics._intent_counts["question"] == 2
        assert metrics._intent_counts["greeting"] == 1

    def test_latency_percentiles_calculation(self):
        """Test latency percentile calculation"""
        metrics = MonitoringMetrics()

        # Record multiple requests with varying latencies
        for latency in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            metrics.record_request("/api/test", "GET", 200, latency)

        stats = metrics.get_request_latency_stats(window_seconds=300)

        assert stats is not None
        assert stats.count == 10
        assert stats.min_ms == 10
        assert stats.max_ms == 100
        assert 50 <= stats.p50_ms <= 60

    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        metrics = MonitoringMetrics()

        # Record 8 successful and 2 failed requests
        for _ in range(8):
            metrics.record_request("/api/test", "GET", 200, 50)
        for _ in range(2):
            metrics.record_request("/api/test", "GET", 500, 50)

        error_rate = metrics.get_error_rate(window_seconds=300)

        assert error_rate == 20.0  # 2/10 = 20%

    def test_get_summary(self):
        """Test summary generation"""
        metrics = MonitoringMetrics()

        # Record some data
        metrics.record_request("/api/test", "GET", 200, 50)
        metrics.record_session_start("test-session")
        metrics.record_exchange(intent="question")

        summary = metrics.get_summary()

        assert "timestamp" in summary
        assert "requests" in summary
        assert "sessions" in summary
        assert "conversations" in summary
        assert "errors" in summary
        assert "resources" in summary

    def test_prometheus_format_output(self):
        """Test Prometheus format output"""
        metrics = MonitoringMetrics()

        metrics.record_request("/api/test", "GET", 200, 50)

        prometheus_output = metrics.get_prometheus_metrics()

        assert "lva_requests_total" in prometheus_output
        assert "lva_errors_total" in prometheus_output
        assert "lva_active_sessions" in prometheus_output


class TestLatencyPercentiles:
    """Tests for LatencyPercentiles dataclass"""

    def test_latency_percentiles_to_dict(self):
        """Test LatencyPercentiles conversion to dict"""
        percentiles = LatencyPercentiles(
            count=100,
            mean_ms=50.5,
            p50_ms=45.0,
            p90_ms=80.0,
            p95_ms=90.0,
            p99_ms=98.0,
            min_ms=10.0,
            max_ms=100.0,
        )

        result = percentiles.to_dict()

        assert result["count"] == 100
        assert result["mean_ms"] == 50.5
        assert result["p50_ms"] == 45.0
        assert result["p99_ms"] == 98.0


class TestGlobalInstances:
    """Tests for global singleton instances"""

    def test_health_checker_singleton(self):
        """Verify health_checker is a singleton"""
        assert health_checker is not None
        assert isinstance(health_checker, HealthChecker)

    def test_monitoring_metrics_singleton(self):
        """Verify monitoring_metrics is a singleton"""
        assert monitoring_metrics is not None
        assert isinstance(monitoring_metrics, MonitoringMetrics)


class TestRequestMetrics:
    """Tests for RequestMetrics dataclass"""

    def test_request_metrics_creation(self):
        """Test RequestMetrics creation with defaults"""
        metric = RequestMetrics(
            endpoint="/api/test",
            method="POST",
            status_code=201,
            latency_ms=25.5,
        )

        assert metric.endpoint == "/api/test"
        assert metric.method == "POST"
        assert metric.status_code == 201
        assert metric.latency_ms == 25.5
        assert metric.timestamp > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
