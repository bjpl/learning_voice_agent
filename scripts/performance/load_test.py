#!/usr/bin/env python3
"""
Performance Load Testing Script
PATTERN: Concurrent user simulation with metrics collection
WHY: Validate system under 100 concurrent user load

Usage:
    # Run default test (100 concurrent users)
    python scripts/performance/load_test.py

    # Custom configuration
    python scripts/performance/load_test.py --users 200 --duration 60 --ramp-up 10

    # Output results to file
    python scripts/performance/load_test.py --output results.json
"""
import asyncio
import argparse
import json
import logging
import sys
import os
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

import httpx

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("load_test")


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TestResults:
    """Aggregated test results."""
    test_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    concurrent_users: int

    # Response time metrics (ms)
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p90_response_time: float
    p95_response_time: float
    p99_response_time: float

    # Error breakdown
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    status_codes: Dict[int, int] = field(default_factory=dict)

    # Per-endpoint breakdown
    endpoint_metrics: Dict[str, Dict] = field(default_factory=dict)


class LoadTestClient:
    """
    HTTP client for load testing with metrics collection.

    Features:
    - Connection pooling
    - Automatic retries (configurable)
    - Response time tracking
    - Error categorization
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_connections: int = 100,
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.metrics: List[RequestMetrics] = []
        self._client: Optional[httpx.AsyncClient] = None
        self.max_connections = max_connections

    @asynccontextmanager
    async def session(self):
        """Context manager for HTTP client session."""
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_connections // 2
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=limits,
        ) as client:
            self._client = client
            yield self
            self._client = None

    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> RequestMetrics:
        """Execute a request and collect metrics."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with client.session():'")

        start_time = time.perf_counter()
        error = None
        status_code = 0
        success = False

        try:
            response = await self._client.request(method, endpoint, **kwargs)
            status_code = response.status_code
            success = 200 <= status_code < 400

        except httpx.TimeoutException as e:
            error = f"Timeout: {str(e)}"
        except httpx.ConnectError as e:
            error = f"Connection error: {str(e)}"
        except httpx.HTTPStatusError as e:
            error = f"HTTP error: {str(e)}"
            status_code = e.response.status_code if e.response else 0
        except Exception as e:
            error = f"Unknown error: {str(e)}"

        response_time_ms = (time.perf_counter() - start_time) * 1000

        metrics = RequestMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            success=success,
            error=error,
        )
        self.metrics.append(metrics)

        return metrics


class UserSimulator:
    """
    Simulates a single user's behavior.

    Scenarios:
    - Health check
    - Conversation flow
    - Search queries
    """

    def __init__(self, client: LoadTestClient, user_id: int):
        self.client = client
        self.user_id = user_id
        self.session_id = f"loadtest-user-{user_id}-{int(time.time())}"

    async def run_scenario(self, scenario: str = "mixed") -> List[RequestMetrics]:
        """Run a test scenario."""
        scenarios = {
            "health": self._health_scenario,
            "conversation": self._conversation_scenario,
            "search": self._search_scenario,
            "mixed": self._mixed_scenario,
        }

        handler = scenarios.get(scenario, self._mixed_scenario)
        return await handler()

    async def _health_scenario(self) -> List[RequestMetrics]:
        """Simple health check scenario."""
        metrics = []
        metrics.append(await self.client.request("GET", "/health"))
        return metrics

    async def _conversation_scenario(self) -> List[RequestMetrics]:
        """Simulate a conversation flow."""
        metrics = []

        # Start conversation
        metrics.append(await self.client.request(
            "POST",
            "/api/conversation",
            json={
                "session_id": self.session_id,
                "text": "Hello, I want to learn about machine learning"
            }
        ))

        # Follow-up question
        await asyncio.sleep(0.5)  # Simulate user thinking
        metrics.append(await self.client.request(
            "POST",
            "/api/conversation",
            json={
                "session_id": self.session_id,
                "text": "What are neural networks?"
            }
        ))

        return metrics

    async def _search_scenario(self) -> List[RequestMetrics]:
        """Simulate search queries."""
        metrics = []

        search_terms = ["machine learning", "neural", "conversation"]
        for term in search_terms:
            metrics.append(await self.client.request(
                "POST",
                "/api/search",
                json={"query": term, "limit": 10}
            ))
            await asyncio.sleep(0.2)

        return metrics

    async def _mixed_scenario(self) -> List[RequestMetrics]:
        """Mixed scenario combining different operations."""
        metrics = []

        # Health check
        metrics.append(await self.client.request("GET", "/health"))

        # Conversation
        metrics.append(await self.client.request(
            "POST",
            "/api/conversation",
            json={
                "session_id": self.session_id,
                "text": f"Test message from user {self.user_id}"
            }
        ))

        # Search
        metrics.append(await self.client.request(
            "POST",
            "/api/search",
            json={"query": "test", "limit": 5}
        ))

        return metrics


class LoadTestRunner:
    """
    Orchestrates load testing with configurable parameters.

    Features:
    - Gradual ramp-up
    - Configurable duration
    - Real-time progress reporting
    - Result aggregation
    """

    def __init__(
        self,
        base_url: str,
        concurrent_users: int = 100,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10,
        scenario: str = "mixed",
    ):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.duration = duration_seconds
        self.ramp_up = ramp_up_seconds
        self.scenario = scenario
        self.client = LoadTestClient(base_url, max_connections=concurrent_users + 10)

    async def run(self) -> TestResults:
        """Execute the load test."""
        logger.info(f"Starting load test: {self.concurrent_users} users, {self.duration}s duration")
        logger.info(f"Target: {self.base_url}")

        start_time = datetime.utcnow()

        async with self.client.session():
            # Ramp-up phase
            if self.ramp_up > 0:
                await self._ramp_up_phase()

            # Main test phase
            await self._main_test_phase()

        end_time = datetime.utcnow()

        return self._calculate_results(
            test_name=f"Load Test - {self.concurrent_users} users",
            start_time=start_time,
            end_time=end_time,
        )

    async def _ramp_up_phase(self):
        """Gradually increase load during ramp-up."""
        logger.info(f"Ramp-up phase: {self.ramp_up} seconds")

        users_per_step = max(1, self.concurrent_users // self.ramp_up)
        current_users = 0

        for second in range(self.ramp_up):
            current_users = min(current_users + users_per_step, self.concurrent_users)

            tasks = [
                self._run_user(i) for i in range(current_users)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(f"Ramp-up: {current_users}/{self.concurrent_users} users active")
            await asyncio.sleep(1)

    async def _main_test_phase(self):
        """Run the main test with full load."""
        logger.info(f"Main test phase: {self.duration} seconds with {self.concurrent_users} users")

        end_time = time.time() + self.duration
        iteration = 0

        while time.time() < end_time:
            iteration += 1

            tasks = [
                self._run_user(i) for i in range(self.concurrent_users)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            remaining = int(end_time - time.time())
            if iteration % 10 == 0:
                logger.info(
                    f"Progress: {len(self.client.metrics)} requests completed, "
                    f"{remaining}s remaining"
                )

            await asyncio.sleep(0.1)  # Small delay between iterations

    async def _run_user(self, user_id: int):
        """Run a single user simulation."""
        try:
            simulator = UserSimulator(self.client, user_id)
            await simulator.run_scenario(self.scenario)
        except Exception as e:
            logger.warning(f"User {user_id} scenario failed: {e}")

    def _calculate_results(
        self,
        test_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> TestResults:
        """Calculate aggregated test results."""
        metrics = self.client.metrics
        duration = (end_time - start_time).total_seconds()

        response_times = [m.response_time_ms for m in metrics]
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]

        # Calculate percentiles
        sorted_times = sorted(response_times)

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (data[c] - data[f]) * (k - f)

        # Error breakdown
        errors_by_type: Dict[str, int] = {}
        for m in failed:
            error_type = m.error.split(":")[0] if m.error else "Unknown"
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        # Status code breakdown
        status_codes: Dict[int, int] = {}
        for m in metrics:
            status_codes[m.status_code] = status_codes.get(m.status_code, 0) + 1

        # Per-endpoint metrics
        endpoint_metrics: Dict[str, Dict] = {}
        endpoints = set(m.endpoint for m in metrics)
        for endpoint in endpoints:
            ep_metrics = [m for m in metrics if m.endpoint == endpoint]
            ep_times = [m.response_time_ms for m in ep_metrics]
            endpoint_metrics[endpoint] = {
                "total_requests": len(ep_metrics),
                "successful": len([m for m in ep_metrics if m.success]),
                "avg_response_time": statistics.mean(ep_times) if ep_times else 0,
                "max_response_time": max(ep_times) if ep_times else 0,
            }

        return TestResults(
            test_name=test_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_requests=len(metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            requests_per_second=len(metrics) / duration if duration > 0 else 0,
            concurrent_users=self.concurrent_users,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p50_response_time=percentile(sorted_times, 50),
            p90_response_time=percentile(sorted_times, 90),
            p95_response_time=percentile(sorted_times, 95),
            p99_response_time=percentile(sorted_times, 99),
            errors_by_type=errors_by_type,
            status_codes=status_codes,
            endpoint_metrics=endpoint_metrics,
        )


def print_results(results: TestResults):
    """Print test results in a formatted way."""
    print("\n" + "=" * 60)
    print(f"LOAD TEST RESULTS: {results.test_name}")
    print("=" * 60)

    print(f"\nTest Duration: {results.duration_seconds:.2f}s")
    print(f"Concurrent Users: {results.concurrent_users}")

    print(f"\n--- Request Statistics ---")
    print(f"Total Requests:     {results.total_requests}")
    print(f"Successful:         {results.successful_requests}")
    print(f"Failed:             {results.failed_requests}")
    print(f"Success Rate:       {(results.successful_requests / results.total_requests * 100):.2f}%")
    print(f"Requests/Second:    {results.requests_per_second:.2f}")

    print(f"\n--- Response Times (ms) ---")
    print(f"Average:    {results.avg_response_time:.2f}")
    print(f"Min:        {results.min_response_time:.2f}")
    print(f"Max:        {results.max_response_time:.2f}")
    print(f"P50:        {results.p50_response_time:.2f}")
    print(f"P90:        {results.p90_response_time:.2f}")
    print(f"P95:        {results.p95_response_time:.2f}")
    print(f"P99:        {results.p99_response_time:.2f}")

    if results.errors_by_type:
        print(f"\n--- Errors by Type ---")
        for error_type, count in results.errors_by_type.items():
            print(f"  {error_type}: {count}")

    print(f"\n--- Status Codes ---")
    for code, count in sorted(results.status_codes.items()):
        print(f"  {code}: {count}")

    print(f"\n--- Endpoint Breakdown ---")
    for endpoint, metrics in results.endpoint_metrics.items():
        print(f"  {endpoint}:")
        print(f"    Requests: {metrics['total_requests']} (success: {metrics['successful']})")
        print(f"    Avg Time: {metrics['avg_response_time']:.2f}ms")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Load testing for Learning Voice Agent")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--users", type=int, default=100, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up time (seconds)")
    parser.add_argument("--scenario", default="mixed", choices=["health", "conversation", "search", "mixed"])
    parser.add_argument("--output", help="Output file for JSON results")

    args = parser.parse_args()

    runner = LoadTestRunner(
        base_url=args.url,
        concurrent_users=args.users,
        duration_seconds=args.duration,
        ramp_up_seconds=args.ramp_up,
        scenario=args.scenario,
    )

    results = await runner.run()
    print_results(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
