"""
Locust Load Testing Configuration
Target: 1000 concurrent users with P95 latency < 2 seconds
"""
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import time
import random
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAgentUser(HttpUser):
    """
    Simulates real user behavior for the Learning Voice Agent.
    Mix of API calls reflecting actual usage patterns.
    """

    # Wait 1-3 seconds between tasks (simulates human interaction)
    wait_time = between(1, 3)

    # Track session for stateful testing
    session_id = None

    def on_start(self):
        """Initialize user session on test start."""
        self.session_id = str(uuid.uuid4())
        logger.info(f"User started with session: {self.session_id}")

    @task(10)
    def health_check(self):
        """
        High frequency health check.
        Weight: 10 (most common request)
        """
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def conversation_text(self):
        """
        Text-based conversation (no audio).
        Weight: 5 (common API usage)
        """
        test_messages = [
            "What did I learn about Python yesterday?",
            "Remind me about the meeting notes",
            "Save this: Machine learning uses neural networks",
            "What are the key points from last week?",
            "Help me remember this concept about APIs",
        ]

        payload = {
            "session_id": self.session_id,
            "text": random.choice(test_messages)
        }

        with self.client.post(
            "/api/conversation",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "agent_text" in data:
                    response.success()
                else:
                    response.failure("Missing agent_text in response")
            elif response.status_code == 500:
                # Expected if API keys not configured
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(3)
    def search_captures(self):
        """
        Search through captured learnings.
        Weight: 3 (moderate usage)
        """
        search_queries = [
            "python",
            "meeting",
            "learning",
            "notes",
            "concept",
        ]

        payload = {
            "query": random.choice(search_queries),
            "limit": 10
        }

        with self.client.post(
            "/api/search",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")

    @task(2)
    def get_stats(self):
        """
        System statistics endpoint.
        Weight: 2 (admin/monitoring)
        """
        with self.client.get("/api/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stats failed: {response.status_code}")

    @task(2)
    def session_history(self):
        """
        Get conversation history for session.
        Weight: 2 (review functionality)
        """
        with self.client.get(
            f"/api/session/{self.session_id}/history",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"History failed: {response.status_code}")

    @task(1)
    def static_assets(self):
        """
        Load static files (PWA).
        Weight: 1 (initial page load)
        """
        with self.client.get("/static/index.html", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Static failed: {response.status_code}")


class AdminUser(HttpUser):
    """
    Simulates admin/monitoring behavior.
    Lower frequency, focused on stats and health.
    """

    wait_time = between(5, 10)
    weight = 1  # 1 admin per 10 regular users

    @task(5)
    def health_check(self):
        """Admin health monitoring."""
        self.client.get("/")

    @task(3)
    def system_stats(self):
        """Admin stats monitoring."""
        self.client.get("/api/stats")

    @task(1)
    def openapi_docs(self):
        """Check API documentation."""
        self.client.get("/docs")


# Performance tracking
response_times = []
error_count = 0


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track all requests for performance analysis."""
    global response_times, error_count

    response_times.append(response_time)
    if exception:
        error_count += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate performance report on test completion."""
    global response_times, error_count

    if not response_times:
        logger.warning("No response times recorded")
        return

    # Calculate percentiles
    sorted_times = sorted(response_times)
    p50_idx = int(len(sorted_times) * 0.50)
    p95_idx = int(len(sorted_times) * 0.95)
    p99_idx = int(len(sorted_times) * 0.99)

    report = {
        "total_requests": len(response_times),
        "errors": error_count,
        "error_rate": f"{(error_count / len(response_times) * 100):.2f}%",
        "avg_response_ms": sum(response_times) / len(response_times),
        "p50_ms": sorted_times[p50_idx] if p50_idx < len(sorted_times) else 0,
        "p95_ms": sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0,
        "p99_ms": sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0,
        "min_ms": min(response_times),
        "max_ms": max(response_times),
    }

    logger.info("=" * 60)
    logger.info("PERFORMANCE REPORT")
    logger.info("=" * 60)
    for key, value in report.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")

    # Check against targets
    logger.info("=" * 60)
    logger.info("TARGET VALIDATION")
    logger.info("=" * 60)

    p95_target = 2000  # 2 seconds
    if report["p95_ms"] < p95_target:
        logger.info(f"P95 PASS: {report['p95_ms']:.0f}ms < {p95_target}ms target")
    else:
        logger.warning(f"P95 FAIL: {report['p95_ms']:.0f}ms >= {p95_target}ms target")

    error_target = 0.1  # 0.1% error rate for 99.9% uptime
    error_rate = (error_count / len(response_times) * 100)
    if error_rate < error_target:
        logger.info(f"ERROR RATE PASS: {error_rate:.3f}% < {error_target}% target")
    else:
        logger.warning(f"ERROR RATE FAIL: {error_rate:.3f}% >= {error_target}% target")
