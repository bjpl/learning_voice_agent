# Load Testing Guide

## Overview

This guide covers load testing for the Learning Voice Agent application using Locust.
The load testing suite validates performance under various conditions and ensures the
system meets production requirements.

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| P95 Response Time | < 500ms | User experience threshold for interactive applications |
| Success Rate | > 99.5% | Production reliability requirement |
| Error Rate | < 0.5% | Acceptable error tolerance |
| Concurrent Users | 1000 | Production capacity target |
| Memory Stability | No leaks over 10 min | Long-running reliability |
| Rate Limiting | Functional under load | Security compliance |

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install locust>=2.24.0

# Verify installation
locust --version
```

### Running Tests

```bash
# Quick smoke test (10 users, 1 minute)
./scripts/run_load_tests.sh --scenario smoke

# Full load test (100 users, 10 minutes)
./scripts/run_load_tests.sh --scenario load

# Stress test (1000 users)
./scripts/run_load_tests.sh --scenario stress

# With web UI (for debugging)
./scripts/run_load_tests.sh --web
```

## Test Scenarios

### 1. Smoke Test

Quick validation that the system is working correctly.

```bash
./scripts/run_load_tests.sh --scenario smoke
```

- **Users**: 10
- **Duration**: 1 minute
- **Spawn Rate**: 2/second
- **Purpose**: Validate basic functionality before more intensive tests

### 2. Load Test

Normal production capacity testing.

```bash
./scripts/run_load_tests.sh --scenario load
```

- **Users**: 100
- **Duration**: 10 minutes
- **Spawn Rate**: 10/second
- **Purpose**: Verify system handles expected daily load

### 3. Stress Test

Target maximum capacity testing.

```bash
./scripts/run_load_tests.sh --scenario stress
```

- **Users**: 1000
- **Duration**: 15 minutes
- **Spawn Rate**: 50/second
- **Purpose**: Verify system handles peak load

### 4. Spike Test

Sudden traffic surge testing.

```bash
./scripts/run_load_tests.sh --scenario spike
```

- **Users**: 500
- **Duration**: 5 minutes
- **Spawn Rate**: 500/second (instant)
- **Purpose**: Verify system recovers from sudden traffic spikes

### 5. Endurance Test

Long-running stability testing.

```bash
./scripts/run_load_tests.sh --scenario endurance
```

- **Users**: 200
- **Duration**: 30 minutes
- **Spawn Rate**: 10/second
- **Purpose**: Detect memory leaks and degradation over time

### 6. Authentication-Only Test

Focused testing of authentication endpoints.

```bash
./scripts/run_load_tests.sh --scenario auth-only
```

- **Users**: 50
- **Duration**: 5 minutes
- **Tags**: authenticated
- **Purpose**: Validate authentication under load

### 7. GDPR-Only Test

Focused testing of GDPR compliance endpoints.

```bash
./scripts/run_load_tests.sh --scenario gdpr-only
```

- **Users**: 20
- **Duration**: 5 minutes
- **Tags**: gdpr
- **Purpose**: Validate GDPR endpoints handle concurrent requests

### 8. Rate Limit Test

Verify rate limiting works correctly under load.

```bash
./scripts/run_load_tests.sh --scenario rate-limit
```

- **Users**: 100
- **Duration**: 2 minutes
- **Tags**: rate-limit
- **Purpose**: Confirm rate limiting activates and protects the system

## Endpoints Tested

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | User registration |
| `/api/auth/login/json` | POST | JSON login |
| `/api/auth/refresh` | POST | Token refresh |
| `/api/auth/logout` | POST | User logout |

### User Endpoints

| Endpoint | Method | Auth Required |
|----------|--------|---------------|
| `/api/user/me` | GET | Yes |
| `/api/user/me` | PATCH | Yes |

### GDPR Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gdpr/export` | POST | Request data export |
| `/api/gdpr/export/{id}` | GET | Check export status |
| `/api/gdpr/delete` | POST | Request account deletion |

### Core API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/conversation` | POST | Submit conversation |
| `/api/search` | POST | Search captures |
| `/api/stats` | GET | System statistics |
| `/api/session/{id}/history` | GET | Session history |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{session_id}` | Real-time conversation stream |

## Interpreting Results

### Success Criteria

A test run is considered successful when:

1. **P95 Response Time < 500ms**: 95% of requests complete within 500ms
2. **Success Rate > 99.5%**: Less than 0.5% of requests fail
3. **No Error Spikes**: Errors are distributed, not clustered
4. **Stable Memory**: No increasing memory trend over time

### Sample Healthy Results

```
--- Test Summary ---
Duration:           600.00s
Total Requests:     45000
Requests/Second:    75.00
Successful:         44875
Failed:             125
Success Rate:       99.72%

--- Response Times (ms) ---
Average:            45.23
Minimum:            8.12
Maximum:            892.45
P50 (Median):       32.00
P90:                98.00
P95:                145.00
P99:                312.00

TARGET VALIDATION
[PASS] P95 Response Time: 145ms < 500ms target
[PASS] Success Rate: 99.72% >= 99.5% target
[PASS] Error Rate: 0.28% <= 0.5% target
```

### Warning Signs

| Symptom | Possible Cause | Action |
|---------|----------------|--------|
| P95 > 500ms | Database bottleneck | Check query performance |
| Increasing latency over time | Memory leak | Profile application memory |
| High 5xx errors | Application errors | Check logs for exceptions |
| High 429 errors | Rate limiting triggered | Expected or adjust limits |
| Connection timeouts | Resource exhaustion | Scale horizontally |

## Output Files

Each test run generates:

| File | Description |
|------|-------------|
| `load_test_{scenario}_{timestamp}.html` | Interactive HTML report |
| `load_test_{scenario}_{timestamp}_stats.csv` | Request statistics |
| `load_test_{scenario}_{timestamp}_failures.csv` | Failure details |
| `load_test_{scenario}_{timestamp}_metrics.json` | Machine-readable metrics |

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Load Test

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install locust

      - name: Start application
        run: |
          uvicorn app.main:app --host 0.0.0.0 --port 8000 &
          sleep 10

      - name: Run load test
        run: |
          ./scripts/run_load_tests.sh \
            --scenario load \
            --host http://localhost:8000 \
            --output ./load-results

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: load-test-results
          path: ./load-results/
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All targets met |
| 1 | One or more targets missed |
| 2 | Test configuration error |

## Distributed Testing

For testing against production with higher load:

### Master Node

```bash
locust -f tests/performance/locustfile.py \
    --master \
    --host https://api.example.com
```

### Worker Nodes

```bash
locust -f tests/performance/locustfile.py \
    --worker \
    --master-host=MASTER_IP
```

## Customization

### Custom User Behavior

Edit `tests/performance/locustfile.py` to add custom test scenarios:

```python
class CustomUser(HttpUser):
    wait_time = between(1, 5)

    @task(10)
    def custom_endpoint(self):
        self.client.get("/api/custom")
```

### Adjusting Targets

Modify the `PERFORMANCE_TARGETS` in `locustfile.py`:

```python
PERFORMANCE_TARGETS = {
    "p95_response_ms": 500,        # Adjust as needed
    "success_rate_percent": 99.5,  # Adjust as needed
    "error_rate_percent": 0.5,     # Adjust as needed
}
```

## Troubleshooting

### High Latency

1. Check database query performance
2. Review Redis connection pool settings
3. Profile slow endpoints
4. Consider caching frequently accessed data

### High Error Rate

1. Check API key configuration
2. Review rate limit settings
3. Monitor CPU/memory usage
4. Verify database connection limits

### Connection Errors

1. Increase uvicorn worker count
2. Adjust connection timeouts
3. Scale horizontally with load balancer
4. Check network configuration

### Rate Limiting Issues

1. Verify rate limit configuration
2. Check Redis connectivity (if using distributed rate limiting)
3. Adjust rate limits for load testing (use test environment)

## Best Practices

1. **Always test in staging first** - Never run load tests against production without approval
2. **Start small** - Begin with smoke tests before stress tests
3. **Monitor resources** - Watch CPU, memory, database connections during tests
4. **Run during off-peak** - Schedule intensive tests during low-traffic periods
5. **Compare baselines** - Track performance over time to detect regressions
6. **Test realistically** - Use realistic user behavior patterns
7. **Clean up** - Test accounts created during load tests should be cleaned up

## Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [Performance Testing Best Practices](https://docs.locust.io/en/stable/writing-a-locustfile.html)
- [Distributed Testing](https://docs.locust.io/en/stable/running-distributed.html)
