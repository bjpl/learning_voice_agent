# Performance Report Template

## Test Information

| Field | Value |
|-------|-------|
| Date | YYYY-MM-DD |
| Environment | staging / production |
| Test Duration | X minutes |
| Peak Users | X concurrent |
| Tool Used | Locust 2.24.0 |

## Executive Summary

[ ] PASS - All targets met
[ ] PARTIAL - Some targets missed
[ ] FAIL - Critical targets missed

### Key Findings
- P95 Latency: XXX ms (target: < 2000ms)
- Error Rate: X.XX% (target: < 0.1%)
- Throughput: XXX req/sec
- Auto-scaling: X -> Y instances

## Target Validation

### P95 Latency < 2000ms

| Endpoint | P95 (ms) | Status |
|----------|----------|--------|
| GET / | | |
| POST /api/conversation | | |
| POST /api/search | | |
| GET /api/stats | | |

### Error Rate < 0.1%

| Metric | Value | Status |
|--------|-------|--------|
| Total Requests | | |
| Total Errors | | |
| Error Rate | | |

### Concurrent Users

| Phase | Target | Achieved | Status |
|-------|--------|----------|--------|
| Ramp-up | 100 | | |
| Sustained | 500 | | |
| Peak | 1000 | | |

## Detailed Metrics

### Response Time Distribution

```
P50 (median):  XXX ms
P75:           XXX ms
P90:           XXX ms
P95:           XXX ms
P99:           XXX ms
Max:           XXX ms
```

### Throughput Over Time

| Time (min) | Users | RPS | Avg (ms) | Errors |
|------------|-------|-----|----------|--------|
| 0-5 | | | | |
| 5-10 | | | | |
| 10-15 | | | | |
| 15-20 | | | | |

### Resource Utilization

| Resource | Avg | Peak | Status |
|----------|-----|------|--------|
| CPU | | | |
| Memory | | | |
| Connections | | | |

## Auto-Scaling Behavior

| Time | Event | Replicas | Trigger |
|------|-------|----------|---------|
| | Scale up | | CPU > 70% |
| | Scale up | | RPS > 100 |
| | Scale down | | Cooldown |

## Error Analysis

### Error Types

| Error | Count | % | Root Cause |
|-------|-------|---|------------|
| 5xx Server Error | | | |
| 4xx Client Error | | | |
| Timeout | | | |
| Connection Reset | | | |

### Error Timeline

[ Chart or description of when errors occurred ]

## Bottlenecks Identified

1. **Issue**:
   - Impact:
   - Recommendation:

2. **Issue**:
   - Impact:
   - Recommendation:

## Recommendations

### Immediate Actions
- [ ]
- [ ]

### Short-term Improvements
- [ ]
- [ ]

### Long-term Optimizations
- [ ]
- [ ]

## Test Commands Used

```bash
# Smoke test
locust -f tests/load/locustfile.py --host https://api.domain.com \
    --headless --users 50 --spawn-rate 10 --run-time 2m

# Full load test
locust -f tests/load/locustfile.py --host https://api.domain.com \
    --headless --users 1000 --spawn-rate 50 --run-time 15m \
    --html report.html --csv results
```

## Appendix

### A. Full Locust Report
[Attach HTML report]

### B. System Metrics Screenshots
[Attach monitoring screenshots]

### C. Configuration Used
```toml
# railway.toml settings
minReplicas = 1
maxReplicas = 10
cpuThresholdPercentage = 70
```

---

**Report Generated**: YYYY-MM-DD HH:MM UTC
**Report Author**:
**Reviewed By**:
