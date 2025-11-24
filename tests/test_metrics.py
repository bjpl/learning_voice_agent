"""
Test Metrics Collection and Endpoints

This test verifies that the metrics system is working correctly.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.metrics import metrics_collector, http_requests_total


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


def test_metrics_endpoint_exists(client):
    """Test that /metrics endpoint returns Prometheus format"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    # Check for some expected metrics
    content = response.text
    assert "http_requests_total" in content
    assert "http_request_duration_seconds" in content


def test_metrics_json_endpoint(client):
    """Test that /api/metrics returns JSON format"""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = response.json()
    assert "timestamp" in data
    assert "application" in data
    assert "requests" in data
    assert "sessions" in data


def test_health_endpoint(client):
    """Test that /api/health returns health status"""
    response = client.get("/api/health")
    assert response.status_code in [200, 503]  # Could be degraded

    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "dependencies" in data
    assert "metrics" in data

    # Check dependencies
    assert "database" in data["dependencies"]
    assert "redis" in data["dependencies"]
    assert "claude_api" in data["dependencies"]
    assert "whisper_api" in data["dependencies"]


def test_metrics_collection():
    """Test that metrics collector works"""
    # Test HTTP request tracking
    metrics_collector.track_http_request(
        endpoint="/test",
        method="GET",
        status=200,
        duration=0.5
    )

    # Test Claude call tracking
    metrics_collector.track_claude_call(
        model="claude-3-haiku",
        duration=1.2,
        status="success",
        input_tokens=100,
        output_tokens=50
    )

    # Test Whisper call tracking
    metrics_collector.track_whisper_call(
        duration=0.8,
        status="success",
        audio_duration=5.0
    )

    # Test cache operation tracking
    metrics_collector.track_cache_operation("get", hit=True)
    metrics_collector.track_cache_operation("get", hit=False)

    # Test conversation exchange tracking
    metrics_collector.track_conversation_exchange("question", quality_score=0.85)

    # Verify metrics can be retrieved
    metrics_dict = metrics_collector.get_metrics_dict()
    assert "requests" in metrics_dict
    assert "external_apis" in metrics_dict
    assert "cache" in metrics_dict


def test_metrics_middleware(client):
    """Test that middleware tracks requests automatically"""
    # Make a request
    response = client.get("/")
    assert response.status_code == 200

    # Check that metrics were updated
    metrics_data = client.get("/metrics").text
    assert "http_requests_total" in metrics_data


def test_root_endpoint_includes_observability(client):
    """Test that root endpoint lists observability endpoints"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "endpoints" in data
    assert "metrics" in data["endpoints"]
    assert "health" in data["endpoints"]
    assert "metrics_json" in data["endpoints"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
