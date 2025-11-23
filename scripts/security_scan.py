#!/usr/bin/env python3
"""
Security Scanning Script
PATTERN: Automated security analysis with multiple tools
WHY: Identify vulnerabilities before production deployment

Tools:
- bandit: Python static security analysis
- safety: Dependency vulnerability checking
- Custom checks: Secrets detection, configuration validation

Usage:
    python scripts/security_scan.py
    python scripts/security_scan.py --output security_report.json
    python scripts/security_scan.py --fix  # Auto-fix where possible
"""
import asyncio
import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("security_scan")


@dataclass
class SecurityIssue:
    """Single security issue found."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    tool: str = "custom"


@dataclass
class SecurityReport:
    """Complete security scan report."""
    scan_time: str
    scan_duration_seconds: float
    project_path: str
    issues: List[SecurityIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    tools_run: List[str] = field(default_factory=list)

    # Compliance checks
    secrets_detected: bool = False
    vulnerable_dependencies: int = 0
    insecure_code_patterns: int = 0


class SecurityScanner:
    """
    Comprehensive security scanner.

    Checks:
    1. Static code analysis (bandit)
    2. Dependency vulnerabilities (safety)
    3. Hardcoded secrets detection
    4. Configuration security
    5. Input validation patterns
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.issues: List[SecurityIssue] = []
        self.tools_run: List[str] = []

    async def run_full_scan(self) -> SecurityReport:
        """Execute all security scans."""
        start_time = datetime.utcnow()
        logger.info(f"Starting security scan of {self.project_path}")

        # Run all scanners
        await self._run_bandit_scan()
        await self._run_safety_scan()
        await self._check_secrets()
        await self._check_config_security()
        await self._check_input_validation()
        await self._check_sql_injection()

        end_time = datetime.utcnow()

        # Build summary
        summary = {
            "CRITICAL": len([i for i in self.issues if i.severity == "CRITICAL"]),
            "HIGH": len([i for i in self.issues if i.severity == "HIGH"]),
            "MEDIUM": len([i for i in self.issues if i.severity == "MEDIUM"]),
            "LOW": len([i for i in self.issues if i.severity == "LOW"]),
            "INFO": len([i for i in self.issues if i.severity == "INFO"]),
        }

        return SecurityReport(
            scan_time=start_time.isoformat(),
            scan_duration_seconds=(end_time - start_time).total_seconds(),
            project_path=str(self.project_path),
            issues=self.issues,
            summary=summary,
            tools_run=self.tools_run,
            secrets_detected=any(i.category == "secrets" for i in self.issues),
            vulnerable_dependencies=len([i for i in self.issues if i.tool == "safety"]),
            insecure_code_patterns=len([i for i in self.issues if i.tool == "bandit"]),
        )

    async def _run_bandit_scan(self):
        """Run bandit static security analysis."""
        logger.info("Running bandit security scan...")
        self.tools_run.append("bandit")

        try:
            result = subprocess.run(
                ["bandit", "-r", str(self.project_path / "app"), "-f", "json", "-ll"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for issue in data.get("results", []):
                        self.issues.append(SecurityIssue(
                            severity=issue.get("issue_severity", "MEDIUM").upper(),
                            category="code_security",
                            description=issue.get("issue_text", ""),
                            file_path=issue.get("filename"),
                            line_number=issue.get("line_number"),
                            code_snippet=issue.get("code"),
                            recommendation=f"CWE: {issue.get('issue_cwe', {}).get('id', 'N/A')}",
                            tool="bandit"
                        ))
                except json.JSONDecodeError:
                    logger.warning("Could not parse bandit output")

        except FileNotFoundError:
            logger.warning("bandit not installed. Install with: pip install bandit")
            self.issues.append(SecurityIssue(
                severity="INFO",
                category="tooling",
                description="bandit not installed - static analysis skipped",
                recommendation="pip install bandit",
                tool="scanner"
            ))
        except subprocess.TimeoutExpired:
            logger.warning("bandit scan timed out")

    async def _run_safety_scan(self):
        """Run safety dependency vulnerability check."""
        logger.info("Running safety dependency scan...")
        self.tools_run.append("safety")

        requirements_file = self.project_path / "requirements.txt"
        if not requirements_file.exists():
            logger.warning("requirements.txt not found")
            return

        try:
            result = subprocess.run(
                ["safety", "check", "-r", str(requirements_file), "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.stdout:
                try:
                    # Safety outputs JSON with vulnerabilities
                    data = json.loads(result.stdout)

                    # Handle different safety output formats
                    vulns = []
                    if isinstance(data, dict):
                        vulns = data.get("vulnerabilities", [])
                    elif isinstance(data, list):
                        vulns = data

                    for vuln in vulns:
                        if isinstance(vuln, dict):
                            self.issues.append(SecurityIssue(
                                severity="HIGH",
                                category="dependency",
                                description=f"Vulnerable dependency: {vuln.get('package_name', 'unknown')}",
                                recommendation=f"Upgrade to version {vuln.get('analyzed_version', 'N/A')}+",
                                tool="safety"
                            ))
                        elif isinstance(vuln, (list, tuple)) and len(vuln) >= 4:
                            self.issues.append(SecurityIssue(
                                severity="HIGH",
                                category="dependency",
                                description=f"Vulnerable dependency: {vuln[0]} {vuln[2]}",
                                recommendation=vuln[3] if len(vuln) > 3 else "Check for updates",
                                tool="safety"
                            ))

                except json.JSONDecodeError:
                    # Safety might output non-JSON for no vulnerabilities
                    if "No known security vulnerabilities" in result.stdout:
                        logger.info("No dependency vulnerabilities found")
                    else:
                        logger.warning("Could not parse safety output")

        except FileNotFoundError:
            logger.warning("safety not installed. Install with: pip install safety")
            self.issues.append(SecurityIssue(
                severity="INFO",
                category="tooling",
                description="safety not installed - dependency scan skipped",
                recommendation="pip install safety",
                tool="scanner"
            ))
        except subprocess.TimeoutExpired:
            logger.warning("safety scan timed out")

    async def _check_secrets(self):
        """Check for hardcoded secrets."""
        logger.info("Scanning for hardcoded secrets...")
        self.tools_run.append("secrets_check")

        # Patterns that might indicate secrets
        secret_patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][^"\']{10,}["\']', "API Key"),
            (r'(?i)(secret|password|passwd|pwd)\s*[=:]\s*["\'][^"\']{8,}["\']', "Password/Secret"),
            (r'(?i)(token|auth[_-]?token)\s*[=:]\s*["\'][^"\']{10,}["\']', "Auth Token"),
            (r'sk-[a-zA-Z0-9]{48}', "OpenAI API Key"),
            (r'sk-ant-[a-zA-Z0-9-]{90,}', "Anthropic API Key"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
            (r'(?i)bearer\s+[a-zA-Z0-9._-]{20,}', "Bearer Token"),
        ]

        # Files to scan
        extensions = {".py", ".json", ".yaml", ".yml", ".env", ".ini", ".cfg"}
        exclude_dirs = {"venv", ".venv", "node_modules", "__pycache__", ".git"}

        for root, dirs, files in os.walk(self.project_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = Path(root) / file

                if file_path.suffix not in extensions and file != ".env.example":
                    continue

                try:
                    content = file_path.read_text(errors='ignore')

                    for pattern, secret_type in secret_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            # Skip if in example/template file
                            if ".example" in str(file_path) or "template" in str(file_path).lower():
                                continue

                            # Get line number
                            line_num = content[:match.start()].count('\n') + 1

                            self.issues.append(SecurityIssue(
                                severity="CRITICAL",
                                category="secrets",
                                description=f"Potential {secret_type} detected",
                                file_path=str(file_path.relative_to(self.project_path)),
                                line_number=line_num,
                                code_snippet=match.group()[:50] + "...",
                                recommendation="Use environment variables instead of hardcoded secrets",
                                tool="secrets_check"
                            ))

                except Exception as e:
                    logger.debug(f"Could not read {file_path}: {e}")

    async def _check_config_security(self):
        """Check configuration security."""
        logger.info("Checking configuration security...")
        self.tools_run.append("config_check")

        config_file = self.project_path / "app" / "config.py"
        if config_file.exists():
            content = config_file.read_text()

            # Check for debug mode
            if re.search(r'(?i)debug\s*[=:]\s*True', content):
                self.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="configuration",
                    description="Debug mode enabled in configuration",
                    file_path="app/config.py",
                    recommendation="Disable debug mode in production",
                    tool="config_check"
                ))

            # Check for wildcard CORS
            if re.search(r'cors_origins.*\["?\*"?\]', content):
                self.issues.append(SecurityIssue(
                    severity="MEDIUM",
                    category="configuration",
                    description="CORS allows all origins (*)",
                    file_path="app/config.py",
                    recommendation="Restrict CORS to specific trusted domains",
                    tool="config_check"
                ))

            # Check for default secrets
            if re.search(r'(?i)(default|example|test)["\']?\s*,?\s*env', content):
                self.issues.append(SecurityIssue(
                    severity="LOW",
                    category="configuration",
                    description="Default values used for sensitive settings",
                    file_path="app/config.py",
                    recommendation="Ensure all sensitive settings require explicit configuration",
                    tool="config_check"
                ))

    async def _check_input_validation(self):
        """Check for input validation issues."""
        logger.info("Checking input validation patterns...")
        self.tools_run.append("input_validation")

        python_files = list(self.project_path.glob("app/**/*.py"))

        for file_path in python_files:
            try:
                content = file_path.read_text()

                # Check for eval/exec usage
                if re.search(r'\b(eval|exec)\s*\(', content):
                    line_num = content[:re.search(r'\b(eval|exec)\s*\(', content).start()].count('\n') + 1
                    self.issues.append(SecurityIssue(
                        severity="CRITICAL",
                        category="code_security",
                        description="Use of eval() or exec() detected",
                        file_path=str(file_path.relative_to(self.project_path)),
                        line_number=line_num,
                        recommendation="Avoid eval/exec; use safer alternatives",
                        tool="input_validation"
                    ))

                # Check for shell=True in subprocess
                if re.search(r'subprocess\.[a-z]+\([^)]*shell\s*=\s*True', content):
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="code_security",
                        description="subprocess with shell=True detected",
                        file_path=str(file_path.relative_to(self.project_path)),
                        recommendation="Use subprocess with shell=False and pass arguments as list",
                        tool="input_validation"
                    ))

            except Exception as e:
                logger.debug(f"Could not analyze {file_path}: {e}")

    async def _check_sql_injection(self):
        """Check for SQL injection vulnerabilities."""
        logger.info("Checking for SQL injection patterns...")
        self.tools_run.append("sql_injection")

        python_files = list(self.project_path.glob("app/**/*.py"))

        # Patterns that might indicate SQL injection risk
        risky_patterns = [
            (r'execute\s*\(\s*["\'].*%s.*["\']\s*%', "String formatting in SQL"),
            (r'execute\s*\(\s*f["\']', "f-string in SQL query"),
            (r'execute\s*\([^)]*\+\s*[a-zA-Z_]', "String concatenation in SQL"),
        ]

        for file_path in python_files:
            try:
                content = file_path.read_text()

                for pattern, issue_type in risky_patterns:
                    if re.search(pattern, content):
                        self.issues.append(SecurityIssue(
                            severity="HIGH",
                            category="sql_injection",
                            description=f"Potential SQL injection: {issue_type}",
                            file_path=str(file_path.relative_to(self.project_path)),
                            recommendation="Use parameterized queries with ? placeholders",
                            tool="sql_injection"
                        ))

            except Exception as e:
                logger.debug(f"Could not analyze {file_path}: {e}")


def print_report(report: SecurityReport):
    """Print security report in formatted way."""
    print("\n" + "=" * 60)
    print("SECURITY SCAN REPORT")
    print("=" * 60)

    print(f"\nScan Time: {report.scan_time}")
    print(f"Duration: {report.scan_duration_seconds:.2f}s")
    print(f"Project: {report.project_path}")
    print(f"Tools Run: {', '.join(report.tools_run)}")

    print(f"\n--- Summary ---")
    print(f"CRITICAL: {report.summary.get('CRITICAL', 0)}")
    print(f"HIGH:     {report.summary.get('HIGH', 0)}")
    print(f"MEDIUM:   {report.summary.get('MEDIUM', 0)}")
    print(f"LOW:      {report.summary.get('LOW', 0)}")
    print(f"INFO:     {report.summary.get('INFO', 0)}")

    total_issues = sum(report.summary.values())
    print(f"\nTotal Issues: {total_issues}")

    if report.issues:
        print(f"\n--- Issues Details ---")
        for i, issue in enumerate(report.issues, 1):
            print(f"\n[{i}] {issue.severity}: {issue.description}")
            print(f"    Category: {issue.category}")
            if issue.file_path:
                print(f"    File: {issue.file_path}", end="")
                if issue.line_number:
                    print(f":{issue.line_number}", end="")
                print()
            if issue.recommendation:
                print(f"    Fix: {issue.recommendation}")

    # Risk assessment
    print(f"\n--- Risk Assessment ---")
    if report.summary.get('CRITICAL', 0) > 0:
        print("RISK LEVEL: CRITICAL - Immediate action required!")
    elif report.summary.get('HIGH', 0) > 0:
        print("RISK LEVEL: HIGH - Address before production deployment")
    elif report.summary.get('MEDIUM', 0) > 0:
        print("RISK LEVEL: MEDIUM - Review and address soon")
    else:
        print("RISK LEVEL: LOW - Good security posture")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Security scanning for Learning Voice Agent")
    parser.add_argument(
        "--path",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Project path to scan"
    )
    parser.add_argument("--output", help="Output file for JSON report")
    parser.add_argument("--fix", action="store_true", help="Attempt auto-fixes (not implemented)")

    args = parser.parse_args()

    scanner = SecurityScanner(args.path)
    report = await scanner.run_full_scan()

    print_report(report)

    if args.output:
        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)
        report_dict["issues"] = [asdict(i) for i in report.issues]

        with open(args.output, 'w') as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Report saved to {args.output}")

    # Exit with non-zero if critical/high issues found
    if report.summary.get('CRITICAL', 0) > 0 or report.summary.get('HIGH', 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
