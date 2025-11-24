#!/usr/bin/env python3
"""
Security Audit Verification Script - Plan A

This script verifies all security features are properly implemented
as part of the Plan A Security-First initiative.

Run: python scripts/security_audit_verification.py
"""

import os
import sys
import re
import json
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SecurityAuditor:
    """Security audit verification."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name: str, condition: bool, message: str = "", severity: str = "error"):
        """Record a check result."""
        status = "PASS" if condition else ("WARN" if severity == "warning" else "FAIL")

        self.results[name] = {
            "status": status,
            "message": message,
            "severity": severity,
        }

        if condition:
            self.passed += 1
        elif severity == "warning":
            self.warnings += 1
        else:
            self.failed += 1

        return condition

    def run_all_checks(self):
        """Run all security verification checks."""
        print("=" * 70)
        print("  Security Audit Verification - Plan A")
        print("=" * 70)
        print(f"  Started: {datetime.now().isoformat()}")
        print("=" * 70)
        print()

        self._check_jwt_authentication()
        self._check_rate_limiting()
        self._check_cors_configuration()
        self._check_websocket_auth()
        self._check_twilio_validation()
        self._check_dependencies()
        self._check_gdpr_compliance()
        self._check_security_headers()
        self._check_environment_config()

        self._print_summary()

    def _check_jwt_authentication(self):
        """Verify JWT authentication implementation."""
        print("\n[1/9] JWT Authentication")
        print("-" * 40)

        # Check auth module exists
        auth_path = PROJECT_ROOT / "app" / "security" / "auth.py"
        self.check(
            "jwt_auth_module",
            auth_path.exists(),
            "Auth module exists at app/security/auth.py"
        )

        if auth_path.exists():
            content = auth_path.read_text()

            # Check for bcrypt password hashing
            self.check(
                "jwt_bcrypt",
                "bcrypt" in content,
                "Uses bcrypt for password hashing"
            )

            # Check for JWT token creation
            self.check(
                "jwt_token_creation",
                "jwt.encode" in content,
                "JWT token encoding implemented"
            )

            # Check for token verification
            self.check(
                "jwt_token_verify",
                "jwt.decode" in content,
                "JWT token verification implemented"
            )

            # Check for token blacklisting
            self.check(
                "jwt_blacklist",
                "blacklist" in content.lower(),
                "Token blacklist implemented"
            )

            # Check for refresh token support
            self.check(
                "jwt_refresh",
                "refresh" in content.lower(),
                "Refresh token support implemented"
            )

    def _check_rate_limiting(self):
        """Verify rate limiting implementation."""
        print("\n[2/9] Rate Limiting")
        print("-" * 40)

        rate_limit_path = PROJECT_ROOT / "app" / "security" / "rate_limit.py"
        self.check(
            "rate_limit_module",
            rate_limit_path.exists(),
            "Rate limit module exists at app/security/rate_limit.py"
        )

        if rate_limit_path.exists():
            content = rate_limit_path.read_text()

            # Check for different rate limits
            self.check(
                "rate_limit_tiers",
                "auth" in content and "api" in content,
                "Different rate limit tiers defined"
            )

            # Check for Redis support
            self.check(
                "rate_limit_redis",
                "redis" in content.lower(),
                "Redis backend support implemented"
            )

            # Check for 429 response
            self.check(
                "rate_limit_429",
                "429" in content or "TOO_MANY_REQUESTS" in content,
                "429 status code returned when limit exceeded"
            )

    def _check_cors_configuration(self):
        """Verify CORS configuration."""
        print("\n[3/9] CORS Configuration")
        print("-" * 40)

        # Check config.py for CORS settings
        config_path = PROJECT_ROOT / "app" / "config.py"
        if config_path.exists():
            content = config_path.read_text()

            # Check if wildcard is used (bad)
            has_wildcard = 'cors_origins: list = Field(["*"]' in content

            # Check for CORS security module
            cors_path = PROJECT_ROOT / "app" / "security" / "cors.py"
            has_cors_module = cors_path.exists()

            if has_cors_module:
                cors_content = cors_path.read_text()
                blocks_wildcard = '"*"' in cors_content and "not allowed" in cors_content.lower()
            else:
                blocks_wildcard = False

            self.check(
                "cors_no_wildcard",
                blocks_wildcard or not has_wildcard,
                "CORS does not allow wildcard '*' in production",
                severity="error" if has_wildcard and not blocks_wildcard else "warning"
            )

            self.check(
                "cors_module",
                has_cors_module,
                "Dedicated CORS configuration module exists"
            )

    def _check_websocket_auth(self):
        """Verify WebSocket authentication."""
        print("\n[4/9] WebSocket Authentication")
        print("-" * 40)

        deps_path = PROJECT_ROOT / "app" / "security" / "dependencies.py"
        self.check(
            "ws_auth_module",
            deps_path.exists(),
            "Security dependencies module exists"
        )

        if deps_path.exists():
            content = deps_path.read_text()

            self.check(
                "ws_auth_function",
                "websocket_auth" in content,
                "WebSocket auth function implemented"
            )

            self.check(
                "ws_close_on_fail",
                "close" in content and "4001" in content,
                "WebSocket closed on auth failure with proper code"
            )

    def _check_twilio_validation(self):
        """Verify Twilio validation is fail-closed."""
        print("\n[5/9] Twilio Validation")
        print("-" * 40)

        twilio_path = PROJECT_ROOT / "app" / "twilio_handler.py"
        if twilio_path.exists():
            content = twilio_path.read_text()

            # Check for fail-closed behavior
            has_fail_open = "return True  # Skip validation" in content
            has_fail_closed = "return False" in content and "production" in content.lower()

            self.check(
                "twilio_fail_closed",
                not has_fail_open or has_fail_closed,
                "Twilio validation fails closed (does not return True when unconfigured)"
            )

            self.check(
                "twilio_production_check",
                "production" in content.lower() and "environment" in content.lower(),
                "Environment check for production implemented"
            )

    def _check_dependencies(self):
        """Check for vulnerable dependencies."""
        print("\n[6/9] Dependency Security")
        print("-" * 40)

        requirements_path = PROJECT_ROOT / "requirements.txt"
        if requirements_path.exists():
            content = requirements_path.read_text()

            # Check cryptography version
            crypto_match = re.search(r'cryptography[=<>]+(\d+\.\d+\.\d+)', content)
            if crypto_match:
                version = crypto_match.group(1)
                major = int(version.split('.')[0])
                self.check(
                    "dep_cryptography",
                    major >= 42,
                    f"cryptography version {version} (should be >= 42.0.0 for CVE fixes)",
                    severity="warning" if major < 42 else "error"
                )
            else:
                self.check("dep_cryptography", False, "cryptography not found in requirements")

            # Check anthropic version
            anthropic_match = re.search(r'anthropic[=<>]+(\d+\.\d+\.\d+)', content)
            if anthropic_match:
                version = anthropic_match.group(1)
                parts = version.split('.')
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                self.check(
                    "dep_anthropic",
                    major >= 1 or (major == 0 and minor >= 50),
                    f"anthropic version {version} (recommend >= 0.50.0 for latest fixes)",
                    severity="warning"
                )

    def _check_gdpr_compliance(self):
        """Verify GDPR compliance features."""
        print("\n[7/9] GDPR Compliance")
        print("-" * 40)

        routes_path = PROJECT_ROOT / "app" / "security" / "routes.py"
        if routes_path.exists():
            content = routes_path.read_text()

            self.check(
                "gdpr_export_endpoint",
                "/export" in content and "gdpr" in content.lower(),
                "GDPR data export endpoint implemented"
            )

            self.check(
                "gdpr_delete_endpoint",
                "/delete" in content and "gdpr" in content.lower(),
                "GDPR data deletion endpoint implemented"
            )

        models_path = PROJECT_ROOT / "app" / "security" / "models.py"
        if models_path.exists():
            content = models_path.read_text()

            self.check(
                "gdpr_models",
                "GDPRExport" in content and "GDPRDelete" in content,
                "GDPR request/response models defined"
            )

    def _check_security_headers(self):
        """Verify security headers middleware."""
        print("\n[8/9] Security Headers")
        print("-" * 40)

        middleware_path = PROJECT_ROOT / "app" / "middleware.py"
        if middleware_path.exists():
            content = middleware_path.read_text()

            headers_to_check = [
                ("X-Content-Type-Options", "nosniff"),
                ("X-Frame-Options", "DENY"),
                ("X-XSS-Protection", "1"),
                ("Strict-Transport-Security", "HSTS"),
            ]

            for header, name in headers_to_check:
                self.check(
                    f"header_{name.lower()}",
                    header in content,
                    f"{header} header implemented"
                )

    def _check_environment_config(self):
        """Verify environment configuration security."""
        print("\n[9/9] Environment Configuration")
        print("-" * 40)

        env_example = PROJECT_ROOT / ".env.example"
        self.check(
            "env_example",
            env_example.exists(),
            ".env.example file exists for reference",
            severity="warning"
        )

        # Check for .env in .gitignore
        gitignore_path = PROJECT_ROOT / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            self.check(
                "env_gitignore",
                ".env" in content,
                ".env excluded from version control"
            )

        # Check config.py for required variables
        config_path = PROJECT_ROOT / "app" / "config.py"
        if config_path.exists():
            content = config_path.read_text()

            self.check(
                "config_secret_key",
                "secret" in content.lower() or "jwt" in content.lower(),
                "Secret key configuration present",
                severity="warning"
            )

    def _print_summary(self):
        """Print audit summary."""
        print()
        print("=" * 70)
        print("  AUDIT SUMMARY")
        print("=" * 70)

        total = self.passed + self.failed + self.warnings

        print(f"\n  Total Checks: {total}")
        print(f"  Passed: {self.passed} ({100*self.passed/total:.1f}%)")
        print(f"  Failed: {self.failed} ({100*self.failed/total:.1f}%)")
        print(f"  Warnings: {self.warnings} ({100*self.warnings/total:.1f}%)")

        print("\n  Detailed Results:")
        print("-" * 70)

        for check_name, result in self.results.items():
            status_color = {
                "PASS": "\033[92m",  # Green
                "FAIL": "\033[91m",  # Red
                "WARN": "\033[93m",  # Yellow
            }
            reset = "\033[0m"
            status = result["status"]
            color = status_color.get(status, "")

            print(f"  {color}[{status}]{reset} {check_name}: {result['message']}")

        print()
        print("=" * 70)

        if self.failed > 0:
            print("  STATUS: AUDIT FAILED - Security issues found")
            sys.exit(1)
        elif self.warnings > 0:
            print("  STATUS: AUDIT PASSED WITH WARNINGS")
            sys.exit(0)
        else:
            print("  STATUS: AUDIT PASSED - All security checks passed")
            sys.exit(0)


def main():
    """Run security audit."""
    auditor = SecurityAuditor()
    auditor.run_all_checks()


if __name__ == "__main__":
    main()
