# Load Testing Guide

## Overview

This directory contains load testing configuration for the Learning Voice Agent using Locust.

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| P95 Latency | < 2 seconds | User experience threshold |
| Error Rate | < 0.1% | 99.9% uptime requirement |
| Concurrent Users | 1000 | Production capacity target |
| Requests/sec | 500+ | Peak load handling |

## Quick Start

```bash
# Install Locust
pip install locust

# Run locally (web UI)
locust -f tests/load/locustfile.py --host http://localhost:8000

# Run headless (CI/CD)
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --users 1000 \
    --spawn-rate 50 \
    --run-time 5m \
    --html report.html
```

## Test Scenarios

### 1. Smoke Test (Quick validation)
```bash
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --users 10 \
    --spawn-rate 2 \
    --run-time 1m
```

### 2. Load Test (Normal capacity)
```bash
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m
```

### 3. Stress Test (Target capacity)
```bash
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --users 1000 \
    --spawn-rate 50 \
    --run-time 15m \
    --html stress_report.html
```

### 4. Spike Test (Sudden traffic)
```bash
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --users 500 \
    --spawn-rate 500 \
    --run-time 5m
```

### 5. Endurance Test (Long-running)
```bash
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --users 200 \
    --spawn-rate 10 \
    --run-time 1h
```

## Distributed Testing

For testing against production with higher load:

```bash
# Start master
locust -f tests/load/locustfile.py --master --host https://api.example.com

# Start workers (run on multiple machines)
locust -f tests/load/locustfile.py --worker --master-host=MASTER_IP
```

## Interpreting Results

### Key Metrics

- **RPS (Requests Per Second)**: Throughput capacity
- **P50/P95/P99**: Response time percentiles
- **Failure Rate**: Error percentage
- **Active Users**: Concurrent connections

### Healthy Results

```
Total Requests: 50000
Error Rate: 0.02%
P50: 45ms
P95: 180ms
P99: 450ms
Max: 1200ms
```

### Warning Signs

- P95 > 2000ms (latency target breach)
- Error Rate > 0.1% (uptime target breach)
- Increasing latency over time (memory leak)
- Sudden failure spike (resource exhaustion)

## CI/CD Integration

See `/.github/workflows/load-test.yml` for automated load testing in CI.

## Troubleshooting

### High Latency
1. Check database query performance
2. Review Redis connection pool
3. Analyze slow endpoints with profiling
4. Consider caching frequently accessed data

### High Error Rate
1. Check API key configuration
2. Review rate limits
3. Monitor memory/CPU usage
4. Check database connection limits

### Connection Errors
1. Increase uvicorn worker count
2. Adjust connection timeouts
3. Scale horizontally with auto-scaling
