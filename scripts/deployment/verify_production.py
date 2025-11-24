#!/usr/bin/env python3
"""
Production Verification Script
Comprehensive post-deployment verification for Learning Voice Agent.

Usage:
    python scripts/deployment/verify_production.py --target https://yourdomain.com
    python scripts/deployment/verify_production.py --target https://yourdomain.com --full
    python scripts/deployment/verify_production.py --target https://yourdomain.com --output report.json

Verification Categories:
    - Functional: Health, Auth, API endpoints, WebSocket
    - Security: Headers, CORS, Rate Limiting, SSL
    - Performance: Response times, load capacity
"""
import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import socket

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_production")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class VerificationResult:
    """Single verification check result."""
    name: str
    category: str
    passed: bool
    message: str
    duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None


@dataclass
class VerificationReport:
    """Complete verification report."""
    target: str
    timestamp: str
    duration_seconds: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[VerificationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100

    @property
    def overall_status(self) -> str:
        if self.failed_checks == 0:
            return "PASS"
        elif self.failed_checks <= 2:
            return "WARN"
        return "FAIL"


# =============================================================================
# Verification Class
# =============================================================================

class ProductionVerifier:
    """
    Comprehensive production verification suite.

    Performs:
    - Functional testing (health, auth, API)
    - Security validation (headers, CORS, rate limiting)
    - Performance verification (response times)
    """

    def __init__(self, target_url: str, test_email: Optional[str] = None,
                 test_password: Optional[str] = None):
        self.target = target_url.rstrip('/')
        self.test_email = test_email or f"test_{int(time.time())}@verification.local"
        self.test_password = test_password or "VerifyTest1234!"
        self.results: List[VerificationResult] = []
        self.auth_token: Optional[str] = None

    async def run_verification(self, full: bool = False) -> VerificationReport:
        """Run all verification checks."""
        start_time = time.time()

        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            sys.exit(1)

        async with aiohttp.ClientSession() as session:
            # Core functional tests
            await self._check_health(session)
            await self._check_root_endpoint(session)
            await self._check_docs_endpoint(session)

            # Auth flow
            await self._check_auth_register(session)
            await self._check_auth_login(session)
            await self._check_protected_endpoint(session)

            # API endpoints
            await self._check_conversation_api(session)

            # Security validation
            await self._check_security_headers(session)
            await self._check_cors_policy(session)
            await self._check_rate_limiting(session)

            # WebSocket (if available)
            if WEBSOCKETS_AVAILABLE:
                await self._check_websocket()

            if full:
                # Extended checks
                await self._check_ssl_configuration()
                await self._check_response_times(session)

        end_time = time.time()

        passed = len([r for r in self.results if r.passed])
        failed = len([r for r in self.results if not r.passed])

        return VerificationReport(
            target=self.target,
            timestamp=datetime.utcnow().isoformat(),
            duration_seconds=round(end_time - start_time, 2),
            total_checks=len(self.results),
            passed_checks=passed,
            failed_checks=failed,
            results=self.results
        )

    def _add_result(self, name: str, category: str, passed: bool,
                    message: str, duration_ms: float = 0.0,
                    details: Optional[Dict] = None):
        """Add a verification result."""
        self.results.append(VerificationResult(
            name=name,
            category=category,
            passed=passed,
            message=message,
            duration_ms=duration_ms,
            details=details
        ))

        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {name}: {message}")

    # =========================================================================
    # Functional Tests
    # =========================================================================

    async def _check_health(self, session: aiohttp.ClientSession):
        """Check health endpoint."""
        url = urljoin(self.target, "/health")
        start = time.time()

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                duration = (time.time() - start) * 1000

                if resp.status == 200:
                    self._add_result(
                        "Health Check", "functional", True,
                        f"Health endpoint returned 200 OK",
                        duration_ms=duration
                    )
                else:
                    self._add_result(
                        "Health Check", "functional", False,
                        f"Health endpoint returned {resp.status}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "Health Check", "functional", False,
                f"Health check failed: {str(e)}"
            )

    async def _check_root_endpoint(self, session: aiohttp.ClientSession):
        """Check root endpoint."""
        start = time.time()

        try:
            async with session.get(self.target, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                duration = (time.time() - start) * 1000

                if resp.status == 200:
                    self._add_result(
                        "Root Endpoint", "functional", True,
                        "Root endpoint accessible",
                        duration_ms=duration
                    )
                else:
                    self._add_result(
                        "Root Endpoint", "functional", False,
                        f"Root endpoint returned {resp.status}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "Root Endpoint", "functional", False,
                f"Root endpoint check failed: {str(e)}"
            )

    async def _check_docs_endpoint(self, session: aiohttp.ClientSession):
        """Check API docs endpoint."""
        url = urljoin(self.target, "/docs")
        start = time.time()

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                duration = (time.time() - start) * 1000

                if resp.status == 200:
                    self._add_result(
                        "API Documentation", "functional", True,
                        "API docs endpoint accessible",
                        duration_ms=duration
                    )
                else:
                    self._add_result(
                        "API Documentation", "functional", False,
                        f"API docs returned {resp.status}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "API Documentation", "functional", False,
                f"API docs check failed: {str(e)}"
            )

    async def _check_auth_register(self, session: aiohttp.ClientSession):
        """Test user registration."""
        url = urljoin(self.target, "/api/auth/register")
        start = time.time()

        try:
            payload = {
                "email": self.test_email,
                "password": self.test_password
            }

            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                duration = (time.time() - start) * 1000

                # 201 Created, 200 OK, or 409 Conflict (already exists) are acceptable
                if resp.status in [200, 201, 409]:
                    self._add_result(
                        "User Registration", "functional", True,
                        f"Registration endpoint working (status: {resp.status})",
                        duration_ms=duration
                    )
                else:
                    body = await resp.text()
                    self._add_result(
                        "User Registration", "functional", False,
                        f"Registration returned {resp.status}: {body[:100]}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "User Registration", "functional", False,
                f"Registration failed: {str(e)}"
            )

    async def _check_auth_login(self, session: aiohttp.ClientSession):
        """Test user login and get token."""
        url = urljoin(self.target, "/api/auth/login")
        start = time.time()

        try:
            # OAuth2 password flow uses form data
            payload = {
                "username": self.test_email,
                "password": self.test_password
            }

            async with session.post(
                url,
                data=payload,  # form data for OAuth2
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                duration = (time.time() - start) * 1000

                if resp.status == 200:
                    data = await resp.json()
                    self.auth_token = data.get("access_token")

                    if self.auth_token:
                        self._add_result(
                            "User Login", "functional", True,
                            "Login successful, token obtained",
                            duration_ms=duration,
                            details={"token_type": data.get("token_type", "bearer")}
                        )
                    else:
                        self._add_result(
                            "User Login", "functional", False,
                            "Login returned 200 but no token",
                            duration_ms=duration
                        )
                else:
                    body = await resp.text()
                    self._add_result(
                        "User Login", "functional", False,
                        f"Login returned {resp.status}: {body[:100]}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "User Login", "functional", False,
                f"Login failed: {str(e)}"
            )

    async def _check_protected_endpoint(self, session: aiohttp.ClientSession):
        """Test protected endpoint with auth token."""
        if not self.auth_token:
            self._add_result(
                "Protected Endpoint", "functional", False,
                "Skipped - no auth token available"
            )
            return

        url = urljoin(self.target, "/api/user/me")
        start = time.time()

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}

            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                duration = (time.time() - start) * 1000

                if resp.status == 200:
                    data = await resp.json()
                    self._add_result(
                        "Protected Endpoint", "functional", True,
                        "Protected endpoint accessible with token",
                        duration_ms=duration,
                        details={"email": data.get("email")}
                    )
                else:
                    self._add_result(
                        "Protected Endpoint", "functional", False,
                        f"Protected endpoint returned {resp.status}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "Protected Endpoint", "functional", False,
                f"Protected endpoint check failed: {str(e)}"
            )

    async def _check_conversation_api(self, session: aiohttp.ClientSession):
        """Test conversation API endpoint."""
        if not self.auth_token:
            self._add_result(
                "Conversation API", "functional", False,
                "Skipped - no auth token available"
            )
            return

        url = urljoin(self.target, "/api/conversation")
        start = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "message": "Hello, this is a verification test.",
                "session_id": f"verify-{int(time.time())}"
            }

            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                duration = (time.time() - start) * 1000

                if resp.status in [200, 201]:
                    self._add_result(
                        "Conversation API", "functional", True,
                        "Conversation API responding correctly",
                        duration_ms=duration
                    )
                else:
                    body = await resp.text()
                    self._add_result(
                        "Conversation API", "functional", False,
                        f"Conversation API returned {resp.status}: {body[:100]}",
                        duration_ms=duration
                    )
        except Exception as e:
            self._add_result(
                "Conversation API", "functional", False,
                f"Conversation API check failed: {str(e)}"
            )

    # =========================================================================
    # Security Validation
    # =========================================================================

    async def _check_security_headers(self, session: aiohttp.ClientSession):
        """Check security headers."""
        try:
            async with session.get(self.target, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                headers = resp.headers

                required_headers = {
                    "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                    "X-Content-Type-Options": ["nosniff"],
                    "Strict-Transport-Security": None,  # Just check existence
                }

                optional_headers = {
                    "Content-Security-Policy": None,
                    "X-XSS-Protection": None,
                    "Referrer-Policy": None,
                }

                found_headers = []
                missing_headers = []

                for header, valid_values in required_headers.items():
                    value = headers.get(header)
                    if value:
                        if valid_values is None or any(v in value for v in valid_values):
                            found_headers.append(header)
                        else:
                            missing_headers.append(f"{header} (invalid value)")
                    else:
                        missing_headers.append(header)

                # Check optional headers (just for info)
                optional_found = [h for h in optional_headers if headers.get(h)]

                if not missing_headers:
                    self._add_result(
                        "Security Headers", "security", True,
                        f"All required security headers present ({len(found_headers)} found)",
                        details={
                            "found": found_headers,
                            "optional_found": optional_found
                        }
                    )
                else:
                    self._add_result(
                        "Security Headers", "security", False,
                        f"Missing security headers: {', '.join(missing_headers)}",
                        details={
                            "found": found_headers,
                            "missing": missing_headers
                        }
                    )
        except Exception as e:
            self._add_result(
                "Security Headers", "security", False,
                f"Security headers check failed: {str(e)}"
            )

    async def _check_cors_policy(self, session: aiohttp.ClientSession):
        """Check CORS configuration."""
        url = urljoin(self.target, "/api/auth/login")

        try:
            # Test with unauthorized origin
            headers = {
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST"
            }

            async with session.options(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                acao = resp.headers.get("Access-Control-Allow-Origin", "")

                if acao == "*" or "malicious-site.com" in acao:
                    self._add_result(
                        "CORS Policy", "security", False,
                        "CORS allows unauthorized origins (wildcard or malicious)",
                        details={"access_control_allow_origin": acao}
                    )
                else:
                    self._add_result(
                        "CORS Policy", "security", True,
                        "CORS properly restricts unauthorized origins",
                        details={"access_control_allow_origin": acao or "not set"}
                    )
        except Exception as e:
            self._add_result(
                "CORS Policy", "security", False,
                f"CORS check failed: {str(e)}"
            )

    async def _check_rate_limiting(self, session: aiohttp.ClientSession):
        """Check rate limiting is active."""
        url = urljoin(self.target, "/health")
        rate_limited = False
        request_count = 0

        try:
            # Send rapid requests to trigger rate limiting
            for i in range(50):
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    request_count += 1
                    if resp.status == 429:
                        rate_limited = True
                        break
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.05)

            if rate_limited:
                self._add_result(
                    "Rate Limiting", "security", True,
                    f"Rate limiting active (triggered after {request_count} requests)",
                    details={"requests_before_limit": request_count}
                )
            else:
                self._add_result(
                    "Rate Limiting", "security", False,
                    f"Rate limiting not triggered after {request_count} requests",
                    details={"requests_sent": request_count}
                )
        except Exception as e:
            self._add_result(
                "Rate Limiting", "security", False,
                f"Rate limiting check failed: {str(e)}"
            )

    async def _check_websocket(self):
        """Check WebSocket connection."""
        ws_url = self.target.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = urljoin(ws_url, "/ws/verify-test")

        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            async with websockets.connect(
                ws_url,
                extra_headers=headers,
                close_timeout=5
            ) as ws:
                # Send a test message
                await ws.send(json.dumps({"type": "ping"}))

                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    self._add_result(
                        "WebSocket Connection", "functional", True,
                        "WebSocket connection successful",
                        details={"response_received": True}
                    )
                except asyncio.TimeoutError:
                    self._add_result(
                        "WebSocket Connection", "functional", True,
                        "WebSocket connected (no response, may be expected)",
                        details={"response_received": False}
                    )
        except Exception as e:
            self._add_result(
                "WebSocket Connection", "functional", False,
                f"WebSocket connection failed: {str(e)}"
            )

    # =========================================================================
    # Extended Checks
    # =========================================================================

    async def _check_ssl_configuration(self):
        """Check SSL/TLS configuration."""
        if not self.target.startswith("https://"):
            self._add_result(
                "SSL Configuration", "security", False,
                "Not using HTTPS"
            )
            return

        try:
            # Parse hostname from URL
            hostname = self.target.replace("https://", "").split("/")[0].split(":")[0]
            port = 443

            # Create SSL context
            context = ssl.create_default_context()

            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    protocol = ssock.version()
                    cipher = ssock.cipher()

                    # Check certificate validity
                    not_after = cert.get('notAfter', '')

                    self._add_result(
                        "SSL Configuration", "security", True,
                        f"SSL properly configured ({protocol})",
                        details={
                            "protocol": protocol,
                            "cipher": cipher[0] if cipher else None,
                            "cert_expires": not_after
                        }
                    )
        except ssl.SSLCertVerificationError as e:
            self._add_result(
                "SSL Configuration", "security", False,
                f"SSL certificate verification failed: {str(e)}"
            )
        except Exception as e:
            self._add_result(
                "SSL Configuration", "security", False,
                f"SSL check failed: {str(e)}"
            )

    async def _check_response_times(self, session: aiohttp.ClientSession):
        """Check average response times."""
        url = urljoin(self.target, "/health")
        times = []

        try:
            for _ in range(10):
                start = time.time()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        times.append((time.time() - start) * 1000)
                await asyncio.sleep(0.1)

            if times:
                avg_time = sum(times) / len(times)
                p95_time = sorted(times)[int(len(times) * 0.95)] if len(times) >= 10 else max(times)

                # Target: p95 < 500ms
                passed = p95_time < 500

                self._add_result(
                    "Response Times", "performance", passed,
                    f"Avg: {avg_time:.0f}ms, P95: {p95_time:.0f}ms (target: <500ms)",
                    duration_ms=avg_time,
                    details={
                        "average_ms": round(avg_time, 2),
                        "p95_ms": round(p95_time, 2),
                        "samples": len(times)
                    }
                )
            else:
                self._add_result(
                    "Response Times", "performance", False,
                    "Could not measure response times"
                )
        except Exception as e:
            self._add_result(
                "Response Times", "performance", False,
                f"Response time check failed: {str(e)}"
            )


# =============================================================================
# Report Functions
# =============================================================================

def print_report(report: VerificationReport):
    """Print verification report."""
    print("\n" + "=" * 70)
    print("PRODUCTION VERIFICATION REPORT")
    print("=" * 70)

    print(f"\nTarget:     {report.target}")
    print(f"Timestamp:  {report.timestamp}")
    print(f"Duration:   {report.duration_seconds}s")
    print(f"\nStatus:     {report.overall_status}")
    print(f"Pass Rate:  {report.pass_rate:.1f}%")
    print(f"Passed:     {report.passed_checks}/{report.total_checks}")
    print(f"Failed:     {report.failed_checks}/{report.total_checks}")

    # Group by category
    categories = {}
    for result in report.results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)

    for category, results in categories.items():
        print(f"\n--- {category.upper()} ---")
        for r in results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"  {status} {r.name}")
            print(f"         {r.message}")
            if r.duration_ms > 0:
                print(f"         ({r.duration_ms:.0f}ms)")

    # Summary
    print("\n" + "=" * 70)
    if report.overall_status == "PASS":
        print("ALL CHECKS PASSED - System ready for production")
    elif report.overall_status == "WARN":
        print("WARNINGS - Review failed checks before proceeding")
    else:
        print("VERIFICATION FAILED - Address issues before production deployment")
    print("=" * 70 + "\n")


def save_report(report: VerificationReport, output_path: str):
    """Save report to JSON file."""
    report_dict = asdict(report)
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Report saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Production verification for Learning Voice Agent"
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target URL (e.g., https://yourdomain.com)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full verification including SSL and performance"
    )
    parser.add_argument(
        "--email",
        help="Test email for auth verification"
    )
    parser.add_argument(
        "--password",
        help="Test password for auth verification"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    verifier = ProductionVerifier(
        target_url=args.target,
        test_email=args.email,
        test_password=args.password
    )

    print(f"\nStarting production verification of {args.target}")
    print("=" * 70)

    report = await verifier.run_verification(full=args.full)

    print_report(report)

    if args.output:
        save_report(report, args.output)

    # Exit with appropriate code
    if report.overall_status == "FAIL":
        sys.exit(1)
    elif report.overall_status == "WARN":
        sys.exit(0)  # Warnings don't fail CI
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
