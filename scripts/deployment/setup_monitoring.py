#!/usr/bin/env python3
"""
Monitoring Setup Script
Generates monitoring configuration files for various platforms.

Usage:
    python scripts/deployment/setup_monitoring.py --platform sentry
    python scripts/deployment/setup_monitoring.py --platform datadog
    python scripts/deployment/setup_monitoring.py --platform uptimerobot --url https://yourdomain.com
    python scripts/deployment/setup_monitoring.py --all --url https://yourdomain.com

Supported Platforms:
    - Sentry (error tracking)
    - DataDog (APM)
    - New Relic (performance)
    - UptimeRobot (uptime)
    - Prometheus/Grafana (metrics)
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


# =============================================================================
# Configuration Templates
# =============================================================================

def generate_sentry_config(dsn: str = "") -> Dict[str, Any]:
    """Generate Sentry configuration."""
    return {
        "dsn": dsn or "${SENTRY_DSN}",
        "environment": "${ENVIRONMENT:-production}",
        "release": "${APP_VERSION:-1.0.0}",
        "traces_sample_rate": 0.1,
        "profiles_sample_rate": 0.1,
        "send_default_pii": False,
        "attach_stacktrace": True,
        "integrations": {
            "fastapi": True,
            "sqlalchemy": True,
            "redis": True,
            "httpx": True
        },
        "ignore_errors": [
            "KeyboardInterrupt",
            "SystemExit",
            "asyncio.CancelledError"
        ],
        "before_send": "# Custom filtering logic",
        "notes": [
            "Set SENTRY_DSN environment variable",
            "Install: pip install sentry-sdk[fastapi]",
            "Import and initialize in app startup"
        ]
    }


def generate_datadog_config() -> Dict[str, Any]:
    """Generate DataDog APM configuration."""
    return {
        "dd_agent_config": {
            "api_key": "${DD_API_KEY}",
            "site": "datadoghq.com",
            "apm_enabled": True,
            "log_enabled": True,
            "env": "${ENVIRONMENT:-production}",
            "service": "learning-voice-agent",
            "version": "${APP_VERSION:-1.0.0}",
            "tags": [
                "team:backend",
                "project:learning-voice-agent"
            ]
        },
        "tracer_config": {
            "analytics_enabled": True,
            "runtime_metrics_enabled": True,
            "logs_injection": True
        },
        "notes": [
            "Set DD_API_KEY environment variable",
            "Install: pip install ddtrace",
            "Run with: ddtrace-run python -m app.main",
            "Or initialize in code with ddtrace.patch_all()"
        ]
    }


def generate_prometheus_config() -> str:
    """Generate Prometheus scrape configuration."""
    return """# Prometheus scrape configuration for Learning Voice Agent
# Add to your prometheus.yml

scrape_configs:
  - job_name: 'learning-voice-agent'
    scrape_interval: 15s
    scrape_timeout: 10s
    metrics_path: /metrics
    static_configs:
      - targets:
          - 'learning-voice-agent:8000'
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '([^:]+):\\d+'
        replacement: '${1}'
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'python_gc_.*'
        action: drop  # Optional: drop verbose GC metrics

# Alert rules
rule_files:
  - '/etc/prometheus/alerts/learning-voice-agent.yml'
"""


def generate_prometheus_alerts() -> str:
    """Generate Prometheus alerting rules."""
    return """# Prometheus alerting rules for Learning Voice Agent
# Save as /etc/prometheus/alerts/learning-voice-agent.yml

groups:
  - name: learning-voice-agent-alerts
    interval: 30s
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{job="learning-voice-agent",status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="learning-voice-agent"}[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"

      # High Response Time
      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{job="learning-voice-agent"}[5m])) by (le)
          ) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "P95 response time is {{ $value | humanizeDuration }} (threshold: 500ms)"

      # High CPU Usage
      - alert: HighCPUUsage
        expr: |
          process_cpu_seconds_total{job="learning-voice-agent"} > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value | humanizePercentage }}"

      # High Memory Usage
      - alert: HighMemoryUsage
        expr: |
          process_resident_memory_bytes{job="learning-voice-agent"}
          / node_memory_MemTotal_bytes > 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      # Service Down
      - alert: ServiceDown
        expr: up{job="learning-voice-agent"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Learning Voice Agent service has been down for more than 1 minute"

      # Too Many 429 Responses
      - alert: TooManyRateLimited
        expr: |
          sum(rate(http_requests_total{job="learning-voice-agent",status="429"}[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Too many rate-limited requests"
          description: "{{ $value }} requests/s are being rate limited"
"""


def generate_grafana_dashboard() -> Dict[str, Any]:
    """Generate Grafana dashboard JSON."""
    return {
        "dashboard": {
            "id": None,
            "uid": "learning-voice-agent",
            "title": "Learning Voice Agent",
            "tags": ["api", "python", "fastapi"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(rate(http_requests_total{job='learning-voice-agent'}[1m])) by (method)",
                            "legendFormat": "{{method}}"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "Error Rate",
                    "type": "graph",
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(rate(http_requests_total{job='learning-voice-agent',status=~'5..'}[1m]))",
                            "legendFormat": "5xx errors"
                        },
                        {
                            "expr": "sum(rate(http_requests_total{job='learning-voice-agent',status=~'4..'}[1m]))",
                            "legendFormat": "4xx errors"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Response Time P95",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job='learning-voice-agent'}[5m])) by (le))",
                            "legendFormat": "P95"
                        },
                        {
                            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job='learning-voice-agent'}[5m])) by (le))",
                            "legendFormat": "P50"
                        }
                    ]
                },
                {
                    "id": 4,
                    "title": "Active Connections",
                    "type": "stat",
                    "gridPos": {"x": 12, "y": 8, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "sum(http_connections_active{job='learning-voice-agent'})",
                            "legendFormat": "Active"
                        }
                    ]
                },
                {
                    "id": 5,
                    "title": "Memory Usage",
                    "type": "gauge",
                    "gridPos": {"x": 18, "y": 8, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "process_resident_memory_bytes{job='learning-voice-agent'} / 1024 / 1024",
                            "legendFormat": "MB"
                        }
                    ]
                }
            ]
        }
    }


def generate_uptimerobot_config(url: str) -> Dict[str, Any]:
    """Generate UptimeRobot monitor configuration."""
    base_url = url.rstrip('/')

    return {
        "monitors": [
            {
                "friendly_name": "Learning Voice Agent - Health",
                "url": f"{base_url}/health",
                "type": "http",
                "interval": 60,
                "timeout": 30,
                "http_method": "GET",
                "expected_status_codes": [200],
                "alert_contacts": ["# Add your alert contact IDs"]
            },
            {
                "friendly_name": "Learning Voice Agent - API",
                "url": f"{base_url}/api/health",
                "type": "http",
                "interval": 300,
                "timeout": 30,
                "http_method": "GET",
                "expected_status_codes": [200, 401],  # 401 if auth required
                "alert_contacts": ["# Add your alert contact IDs"]
            },
            {
                "friendly_name": "Learning Voice Agent - SSL",
                "url": base_url,
                "type": "ssl",
                "interval": 86400,  # Daily
                "ssl_expiry_reminder": 30,
                "alert_contacts": ["# Add your alert contact IDs"]
            }
        ],
        "alert_contacts": {
            "email": {
                "type": "email",
                "value": "# your-email@example.com"
            },
            "slack": {
                "type": "slack",
                "value": "# Slack webhook URL"
            }
        },
        "setup_instructions": [
            "1. Create account at https://uptimerobot.com",
            "2. Add monitors manually or use API",
            "3. API: https://api.uptimerobot.com/v2/newMonitor",
            "4. Set up alert contacts for notifications"
        ]
    }


def generate_alerting_config() -> Dict[str, Any]:
    """Generate alerting configuration."""
    return {
        "alert_rules": {
            "error_rate": {
                "threshold": 0.01,  # 1%
                "window": "5m",
                "severity": "critical",
                "description": "Error rate exceeds 1%"
            },
            "response_time_p95": {
                "threshold_ms": 1000,
                "window": "5m",
                "severity": "warning",
                "description": "P95 response time exceeds 1s"
            },
            "cpu_usage": {
                "threshold": 0.80,  # 80%
                "window": "10m",
                "severity": "warning",
                "description": "CPU usage exceeds 80%"
            },
            "memory_usage": {
                "threshold": 0.85,  # 85%
                "window": "10m",
                "severity": "warning",
                "description": "Memory usage exceeds 85%"
            },
            "disk_usage": {
                "threshold": 0.90,  # 90%
                "window": "30m",
                "severity": "critical",
                "description": "Disk usage exceeds 90%"
            },
            "service_down": {
                "threshold": 1,
                "window": "1m",
                "severity": "critical",
                "description": "Service is unreachable"
            },
            "high_rate_limiting": {
                "threshold": 10,  # requests/sec
                "window": "5m",
                "severity": "warning",
                "description": "Many requests being rate limited"
            }
        },
        "notification_channels": {
            "email": {
                "enabled": True,
                "recipients": ["# ops@example.com"]
            },
            "slack": {
                "enabled": True,
                "webhook": "# https://hooks.slack.com/services/..."
            },
            "pagerduty": {
                "enabled": False,
                "integration_key": "# PagerDuty integration key"
            }
        },
        "escalation_policy": {
            "warning": {
                "notify": ["slack"],
                "repeat_interval": "1h"
            },
            "critical": {
                "notify": ["email", "slack", "pagerduty"],
                "repeat_interval": "15m"
            }
        }
    }


# =============================================================================
# Setup Functions
# =============================================================================

def setup_sentry(output_dir: Path, dsn: str = ""):
    """Generate Sentry configuration files."""
    config = generate_sentry_config(dsn)

    # Save configuration
    config_file = output_dir / "sentry_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Generate initialization code
    init_code = '''# Sentry Initialization for Learning Voice Agent
# Add to app/main.py

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

def init_sentry():
    """Initialize Sentry error tracking."""
    import os

    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        print("Warning: SENTRY_DSN not set, error tracking disabled")
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("ENVIRONMENT", "production"),
        release=os.getenv("APP_VERSION", "1.0.0"),
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
        send_default_pii=False,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
        ],
        before_send=filter_events,
    )

def filter_events(event, hint):
    """Filter out non-actionable events."""
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        # Ignore expected errors
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            return None
    return event

# Call in app startup:
# init_sentry()
'''

    init_file = output_dir / "sentry_init.py"
    with open(init_file, 'w') as f:
        f.write(init_code)

    print(f"  - Created {config_file}")
    print(f"  - Created {init_file}")


def setup_prometheus(output_dir: Path):
    """Generate Prometheus configuration files."""
    # Scrape config
    scrape_file = output_dir / "prometheus_scrape.yml"
    with open(scrape_file, 'w') as f:
        f.write(generate_prometheus_config())

    # Alert rules
    alerts_file = output_dir / "prometheus_alerts.yml"
    with open(alerts_file, 'w') as f:
        f.write(generate_prometheus_alerts())

    print(f"  - Created {scrape_file}")
    print(f"  - Created {alerts_file}")


def setup_grafana(output_dir: Path):
    """Generate Grafana dashboard."""
    dashboard = generate_grafana_dashboard()

    dashboard_file = output_dir / "grafana_dashboard.json"
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard, f, indent=2)

    print(f"  - Created {dashboard_file}")


def setup_uptimerobot(output_dir: Path, url: str):
    """Generate UptimeRobot configuration."""
    config = generate_uptimerobot_config(url)

    config_file = output_dir / "uptimerobot_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  - Created {config_file}")


def setup_alerting(output_dir: Path):
    """Generate alerting configuration."""
    config = generate_alerting_config()

    config_file = output_dir / "alerting_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  - Created {config_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate monitoring configuration for Learning Voice Agent"
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["sentry", "datadog", "prometheus", "grafana", "uptimerobot", "alerting"],
        help="Specific platform to configure"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all configurations"
    )
    parser.add_argument(
        "--url", "-u",
        default="https://yourdomain.com",
        help="Target URL for uptime monitoring"
    )
    parser.add_argument(
        "--sentry-dsn",
        help="Sentry DSN (optional)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="config/monitoring",
        help="Output directory for configuration files"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("MONITORING SETUP")
    print(f"{'=' * 60}")
    print(f"Output Directory: {output_dir}")
    print(f"Target URL: {args.url}")
    print(f"{'=' * 60}\n")

    if args.all or args.platform == "sentry":
        print("Setting up Sentry...")
        setup_sentry(output_dir, args.sentry_dsn or "")

    if args.all or args.platform == "datadog":
        print("Setting up DataDog...")
        config = generate_datadog_config()
        config_file = output_dir / "datadog_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  - Created {config_file}")

    if args.all or args.platform == "prometheus":
        print("Setting up Prometheus...")
        setup_prometheus(output_dir)

    if args.all or args.platform == "grafana":
        print("Setting up Grafana...")
        setup_grafana(output_dir)

    if args.all or args.platform == "uptimerobot":
        print("Setting up UptimeRobot...")
        setup_uptimerobot(output_dir, args.url)

    if args.all or args.platform == "alerting":
        print("Setting up Alerting...")
        setup_alerting(output_dir)

    print(f"\n{'=' * 60}")
    print("SETUP COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nConfiguration files saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Review and customize the configuration files")
    print("2. Set required environment variables (API keys, DSNs)")
    print("3. Deploy configurations to your monitoring platforms")
    print("4. Test alerts to ensure notifications work")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
