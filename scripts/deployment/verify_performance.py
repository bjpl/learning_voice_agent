#!/usr/bin/env python3
"""
Performance Verification Script
Validates response times and capacity for production deployment.

Usage:
    python scripts/deployment/verify_performance.py --target https://yourdomain.com
    python scripts/deployment/verify_performance.py --target https://yourdomain.com --samples 20
    python scripts/deployment/verify_performance.py --target https://yourdomain.com --concurrent 10

Performance Targets:
    - Average response time: < 300ms
    - P95 response time: < 500ms
    - P99 response time: < 1000ms
    - Success rate: > 99.5%
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class EndpointMetrics:
    """Metrics for a single endpoint."""
    name: str
    url: str
    samples: int
    success_count: int
    failure_count: int
    response_times_ms: List[float]

    @property
    def success_rate(self) -> float:
        if self.samples == 0:
            return 0.0
        return (self.success_count / self.samples) * 100

    @property
    def avg_response_ms(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return statistics.mean(self.response_times_ms)

    @property
    def p50_response_ms(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return statistics.median(self.response_times_ms)

    @property
    def p95_response_ms(self) -> float:
        if not self.response_times_ms:
            return 0.0
        sorted_times = sorted(self.response_times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]

    @property
    def p99_response_ms(self) -> float:
        if not self.response_times_ms:
            return 0.0
        sorted_times = sorted(self.response_times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]


@dataclass
class PerformanceReport:
    """Complete performance report."""
    target: str
    timestamp: str
    duration_seconds: float
    total_requests: int
    total_success: int
    total_failures: int
    endpoints: List[EndpointMetrics] = field(default_factory=list)

    # Targets
    target_avg_ms: float = 300.0
    target_p95_ms: float = 500.0
    target_p99_ms: float = 1000.0
    target_success_rate: float = 99.5

    @property
    def overall_success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_success / self.total_requests) * 100

    @property
    def overall_avg_ms(self) -> float:
        all_times = []
        for ep in self.endpoints:
            all_times.extend(ep.response_times_ms)
        if not all_times:
            return 0.0
        return statistics.mean(all_times)

    @property
    def overall_p95_ms(self) -> float:
        all_times = []
        for ep in self.endpoints:
            all_times.extend(ep.response_times_ms)
        if not all_times:
            return 0.0
        sorted_times = sorted(all_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]

    @property
    def all_targets_met(self) -> bool:
        return (
            self.overall_avg_ms <= self.target_avg_ms and
            self.overall_p95_ms <= self.target_p95_ms and
            self.overall_success_rate >= self.target_success_rate
        )


# =============================================================================
# Performance Verifier
# =============================================================================

class PerformanceVerifier:
    """
    Performance verification suite.

    Tests response times across multiple endpoints with configurable
    sample sizes and concurrency.
    """

    def __init__(self, target_url: str, samples: int = 10, concurrent: int = 5):
        self.target = target_url.rstrip('/')
        self.samples = samples
        self.concurrent = concurrent
        self.endpoints: Dict[str, EndpointMetrics] = {}

    async def run_verification(self) -> PerformanceReport:
        """Run performance verification."""
        if not AIOHTTP_AVAILABLE:
            print("Error: aiohttp not installed. Install with: pip install aiohttp")
            sys.exit(1)

        start_time = time.time()

        # Define endpoints to test
        test_endpoints = [
            ("Health", "/health", "GET", None),
            ("Root", "/", "GET", None),
            ("API Docs", "/docs", "GET", None),
            ("OpenAPI Schema", "/openapi.json", "GET", None),
        ]

        connector = aiohttp.TCPConnector(limit=self.concurrent * 2)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Test each endpoint
            for name, path, method, data in test_endpoints:
                await self._test_endpoint(session, name, path, method, data)

        end_time = time.time()

        # Build report
        total_requests = sum(ep.samples for ep in self.endpoints.values())
        total_success = sum(ep.success_count for ep in self.endpoints.values())
        total_failures = sum(ep.failure_count for ep in self.endpoints.values())

        return PerformanceReport(
            target=self.target,
            timestamp=datetime.utcnow().isoformat(),
            duration_seconds=round(end_time - start_time, 2),
            total_requests=total_requests,
            total_success=total_success,
            total_failures=total_failures,
            endpoints=list(self.endpoints.values())
        )

    async def _test_endpoint(self, session: aiohttp.ClientSession,
                             name: str, path: str, method: str,
                             data: Optional[dict] = None):
        """Test a single endpoint with multiple samples."""
        url = urljoin(self.target, path)
        metrics = EndpointMetrics(
            name=name,
            url=url,
            samples=0,
            success_count=0,
            failure_count=0,
            response_times_ms=[]
        )

        print(f"Testing {name} ({url})...")

        # Create tasks for concurrent requests
        tasks = []
        for _ in range(self.samples):
            tasks.append(self._single_request(session, url, method, data, metrics))

        # Run concurrently with semaphore
        semaphore = asyncio.Semaphore(self.concurrent)

        async def limited_request(task):
            async with semaphore:
                return await task

        await asyncio.gather(*[limited_request(t) for t in tasks])

        self.endpoints[name] = metrics

        # Print quick summary
        print(f"  -> {metrics.success_count}/{metrics.samples} successful, "
              f"avg: {metrics.avg_response_ms:.0f}ms, "
              f"p95: {metrics.p95_response_ms:.0f}ms")

    async def _single_request(self, session: aiohttp.ClientSession,
                              url: str, method: str,
                              data: Optional[dict],
                              metrics: EndpointMetrics):
        """Execute a single request and record metrics."""
        start = time.time()

        try:
            if method == "GET":
                async with session.get(url) as resp:
                    duration_ms = (time.time() - start) * 1000
                    metrics.samples += 1

                    if resp.status < 400:
                        metrics.success_count += 1
                        metrics.response_times_ms.append(duration_ms)
                    else:
                        metrics.failure_count += 1

            elif method == "POST":
                async with session.post(url, json=data) as resp:
                    duration_ms = (time.time() - start) * 1000
                    metrics.samples += 1

                    if resp.status < 400:
                        metrics.success_count += 1
                        metrics.response_times_ms.append(duration_ms)
                    else:
                        metrics.failure_count += 1

        except Exception as e:
            metrics.samples += 1
            metrics.failure_count += 1
            print(f"  Error: {str(e)[:50]}")


# =============================================================================
# Concurrent Load Test
# =============================================================================

async def run_concurrent_load_test(target: str, concurrent: int = 10,
                                   duration_seconds: int = 30) -> Dict:
    """
    Run a concurrent load test to measure throughput.

    Args:
        target: Target URL
        concurrent: Number of concurrent connections
        duration_seconds: Test duration

    Returns:
        Dict with throughput metrics
    """
    print(f"\n--- Concurrent Load Test ---")
    print(f"Concurrent connections: {concurrent}")
    print(f"Duration: {duration_seconds}s")

    url = urljoin(target, "/health")
    results = {
        "total_requests": 0,
        "success_count": 0,
        "failure_count": 0,
        "response_times": []
    }

    start_time = time.time()
    end_time = start_time + duration_seconds

    async def worker(session: aiohttp.ClientSession, results: dict):
        while time.time() < end_time:
            req_start = time.time()
            try:
                async with session.get(url) as resp:
                    duration = (time.time() - req_start) * 1000
                    results["total_requests"] += 1
                    if resp.status < 400:
                        results["success_count"] += 1
                        results["response_times"].append(duration)
                    else:
                        results["failure_count"] += 1
            except Exception:
                results["total_requests"] += 1
                results["failure_count"] += 1

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)

    connector = aiohttp.TCPConnector(limit=concurrent * 2)
    timeout = aiohttp.ClientTimeout(total=10)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [worker(session, results) for _ in range(concurrent)]
        await asyncio.gather(*workers)

    actual_duration = time.time() - start_time

    # Calculate throughput
    throughput = results["total_requests"] / actual_duration
    success_rate = (results["success_count"] / results["total_requests"] * 100) if results["total_requests"] > 0 else 0

    avg_response = statistics.mean(results["response_times"]) if results["response_times"] else 0
    p95_response = 0
    if results["response_times"]:
        sorted_times = sorted(results["response_times"])
        idx = int(len(sorted_times) * 0.95)
        p95_response = sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]

    print(f"\nResults:")
    print(f"  Total requests: {results['total_requests']}")
    print(f"  Throughput: {throughput:.1f} req/s")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg response: {avg_response:.0f}ms")
    print(f"  P95 response: {p95_response:.0f}ms")

    return {
        "total_requests": results["total_requests"],
        "throughput_rps": round(throughput, 2),
        "success_rate_pct": round(success_rate, 2),
        "avg_response_ms": round(avg_response, 2),
        "p95_response_ms": round(p95_response, 2)
    }


# =============================================================================
# Report Functions
# =============================================================================

def print_report(report: PerformanceReport):
    """Print performance report."""
    print("\n" + "=" * 70)
    print("PERFORMANCE VERIFICATION REPORT")
    print("=" * 70)

    print(f"\nTarget:     {report.target}")
    print(f"Timestamp:  {report.timestamp}")
    print(f"Duration:   {report.duration_seconds}s")

    print(f"\n--- Overall Metrics ---")
    print(f"Total Requests:  {report.total_requests}")
    print(f"Success Rate:    {report.overall_success_rate:.1f}% (target: >= {report.target_success_rate}%)")
    print(f"Avg Response:    {report.overall_avg_ms:.0f}ms (target: <= {report.target_avg_ms}ms)")
    print(f"P95 Response:    {report.overall_p95_ms:.0f}ms (target: <= {report.target_p95_ms}ms)")

    print(f"\n--- Endpoint Details ---")
    for ep in report.endpoints:
        status = "[PASS]" if ep.success_rate >= 99.5 and ep.p95_response_ms <= 500 else "[WARN]"
        print(f"\n{status} {ep.name}")
        print(f"       URL: {ep.url}")
        print(f"       Success: {ep.success_count}/{ep.samples} ({ep.success_rate:.1f}%)")
        print(f"       Avg: {ep.avg_response_ms:.0f}ms | P50: {ep.p50_response_ms:.0f}ms | "
              f"P95: {ep.p95_response_ms:.0f}ms | P99: {ep.p99_response_ms:.0f}ms")

    # Target validation
    print(f"\n--- Target Validation ---")

    targets = [
        ("Avg Response Time", report.overall_avg_ms, report.target_avg_ms, "<="),
        ("P95 Response Time", report.overall_p95_ms, report.target_p95_ms, "<="),
        ("Success Rate", report.overall_success_rate, report.target_success_rate, ">="),
    ]

    all_passed = True
    for name, actual, target, op in targets:
        if op == "<=":
            passed = actual <= target
        else:
            passed = actual >= target

        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False

        print(f"  {status} {name}: {actual:.1f} ({op} {target})")

    print("\n" + "=" * 70)
    if all_passed:
        print("PERFORMANCE VERIFICATION PASSED")
    else:
        print("PERFORMANCE VERIFICATION FAILED")
    print("=" * 70 + "\n")

    return all_passed


def save_report(report: PerformanceReport, output_path: str):
    """Save report to JSON."""
    report_dict = {
        "target": report.target,
        "timestamp": report.timestamp,
        "duration_seconds": report.duration_seconds,
        "total_requests": report.total_requests,
        "total_success": report.total_success,
        "total_failures": report.total_failures,
        "overall_success_rate": report.overall_success_rate,
        "overall_avg_ms": report.overall_avg_ms,
        "overall_p95_ms": report.overall_p95_ms,
        "targets_met": report.all_targets_met,
        "endpoints": [
            {
                "name": ep.name,
                "url": ep.url,
                "samples": ep.samples,
                "success_rate": ep.success_rate,
                "avg_ms": ep.avg_response_ms,
                "p50_ms": ep.p50_response_ms,
                "p95_ms": ep.p95_response_ms,
                "p99_ms": ep.p99_response_ms
            }
            for ep in report.endpoints
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"Report saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Performance verification for Learning Voice Agent"
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target URL (e.g., https://yourdomain.com)"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=10,
        help="Number of samples per endpoint (default: 10)"
    )
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)"
    )
    parser.add_argument(
        "--load-test",
        action="store_true",
        help="Run additional concurrent load test"
    )
    parser.add_argument(
        "--load-duration",
        type=int,
        default=30,
        help="Load test duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    print(f"\nStarting performance verification of {args.target}")
    print(f"Samples: {args.samples}, Concurrent: {args.concurrent}")
    print("=" * 70)

    verifier = PerformanceVerifier(
        target_url=args.target,
        samples=args.samples,
        concurrent=args.concurrent
    )

    report = await verifier.run_verification()

    passed = print_report(report)

    if args.load_test:
        load_results = await run_concurrent_load_test(
            args.target,
            concurrent=args.concurrent,
            duration_seconds=args.load_duration
        )

    if args.output:
        save_report(report, args.output)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
