# Observability and Monitoring Guide

## Overview

The Learning Voice Agent includes comprehensive observability features using Prometheus metrics, health checks, and application insights. This guide covers all monitoring capabilities and how to use them effectively.

## Quick Start

### 1. Access Metrics Endpoints

```bash
# Prometheus metrics (for Prometheus scraping)
curl http://localhost:8000/metrics

# JSON metrics (for dashboards/debugging)
curl http://localhost:8000/api/metrics

# Health check (with dependency status)
curl http://localhost:8000/api/health
```

### 2. Setup Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'learning_voice_agent'
    scrape_interval: 10s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
```

### 3. Import Grafana Dashboard

1. Open Grafana
2. Go to Dashboards → Import
3. Upload `/monitoring/grafana_dashboard.json`
4. Select your Prometheus datasource
5. Click Import

## Metrics Categories

### HTTP Request Metrics

**Purpose**: Track API performance and usage patterns

- `http_requests_total` - Total requests by endpoint, method, and status
- `http_request_duration_seconds` - Request duration histogram
- `http_requests_in_progress` - Currently processing requests

**Usage**:
```promql
# Request rate per endpoint
rate(http_requests_total[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])
```

### External API Metrics

**Purpose**: Monitor Claude and Whisper API performance and costs

**Claude API**:
- `claude_api_calls_total` - Total calls by model and status
- `claude_api_duration_seconds` - Call duration histogram
- `claude_api_tokens_total` - Token usage by type (input/output)

**Whisper API**:
- `whisper_api_calls_total` - Total calls by status
- `whisper_api_duration_seconds` - Call duration histogram

**Usage**:
```promql
# Claude API success rate
rate(claude_api_calls_total{status="success"}[5m])
  / rate(claude_api_calls_total[5m])

# Average API response time
avg(rate(claude_api_duration_seconds_sum[5m])
  / rate(claude_api_duration_seconds_count[5m]))

# Token usage rate
rate(claude_api_tokens_total[1h]) * 3600
```

### WebSocket Metrics

**Purpose**: Monitor real-time connection health

- `websocket_connections_active` - Active connections
- `websocket_messages_total` - Messages sent/received
- `websocket_connection_duration_seconds` - Connection duration

**Usage**:
```promql
# Active WebSocket connections
websocket_connections_active

# Message rate
rate(websocket_messages_total[5m])

# Average connection duration
histogram_quantile(0.95, rate(websocket_connection_duration_seconds_bucket[5m]))
```

### Database Metrics

**Purpose**: Track database performance

- `database_queries_total` - Total queries by operation and table
- `database_query_duration_seconds` - Query duration histogram
- `database_connections_active` - Active connections

**Usage**:
```promql
# Query rate by operation
rate(database_queries_total[5m])

# Slow queries (p95 > 100ms)
histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[5m])) > 0.1

# Database load
sum(rate(database_queries_total[5m]))
```

### Cache Metrics

**Purpose**: Monitor Redis cache effectiveness

- `cache_operations_total` - Operations by type and status
- `cache_hit_ratio` - Current hit ratio

**Usage**:
```promql
# Cache hit ratio
cache_hit_ratio

# Cache operation rate
rate(cache_operations_total[5m])

# Miss rate
rate(cache_operations_total{status="miss"}[5m])
```

### Session Metrics

**Purpose**: Track user sessions and engagement

- `sessions_active` - Currently active sessions
- `sessions_total` - Total sessions created
- `session_duration_seconds` - Session duration histogram

**Usage**:
```promql
# Active sessions
sessions_active

# Session creation rate
rate(sessions_total[5m])

# Average session duration
histogram_quantile(0.50, rate(session_duration_seconds_bucket[5m]))
```

### Conversation Metrics

**Purpose**: Track conversation quality and engagement

- `conversation_exchanges_total` - Exchanges by intent
- `conversation_quality_score` - Quality score histogram
- `user_engagement_score` - Per-session engagement

**Usage**:
```promql
# Exchanges by intent
rate(conversation_exchanges_total[5m])

# Average quality score
histogram_quantile(0.50, rate(conversation_quality_score_bucket[5m]))

# Intent distribution
sum by (intent) (rate(conversation_exchanges_total[5m]))
```

### Cost Tracking

**Purpose**: Monitor API usage costs

- `api_cost_total` - Estimated costs by provider (USD)

**Usage**:
```promql
# Cost per hour
rate(api_cost_total[1h]) * 3600

# Daily cost estimate
increase(api_cost_total[24h])

# Cost breakdown by provider
sum by (provider) (rate(api_cost_total[1h]) * 3600)
```

### Error Metrics

**Purpose**: Track and categorize errors

- `errors_total` - Errors by type and component
- `api_errors_total` - External API errors by provider

**Usage**:
```promql
# Error rate
rate(errors_total[5m])

# Errors by component
sum by (component) (rate(errors_total[5m]))

# API error rate
rate(api_errors_total[5m])
```

## Health Check Endpoint

**Endpoint**: `GET /api/health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-21T12:00:00Z",
  "dependencies": {
    "database": {
      "status": "healthy",
      "latency_ms": 2.5
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 1.2
    },
    "claude_api": {
      "status": "healthy",
      "last_check": "2025-01-21T11:59:00Z"
    },
    "whisper_api": {
      "status": "healthy",
      "last_check": "2025-01-21T11:59:00Z"
    }
  },
  "metrics": {
    "active_sessions": 5,
    "active_websockets": 3,
    "uptime_seconds": 3600
  }
}
```

**Status Codes**:
- `200 OK` - All systems healthy
- `503 Service Unavailable` - One or more dependencies unhealthy

## Alerting Rules

### Critical Alerts

**High Error Rate**:
```yaml
- alert: HighErrorRate
  expr: rate(errors_total[5m]) > 0.1
  for: 5m
  annotations:
    summary: "Error rate exceeds 10%"
```

**API Failures**:
```yaml
- alert: ClaudeAPIFailures
  expr: rate(claude_api_calls_total{status="error"}[5m]) > 0.05
  for: 2m
  annotations:
    summary: "Claude API error rate > 5%"
```

**Low Cache Hit Ratio**:
```yaml
- alert: LowCacheHitRatio
  expr: cache_hit_ratio < 0.5
  for: 10m
  annotations:
    summary: "Cache hit ratio below 50%"
```

### Warning Alerts

**Slow Response Time**:
```yaml
- alert: SlowResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 5m
  annotations:
    summary: "95th percentile response time > 2s"
```

**High API Costs**:
```yaml
- alert: HighAPICosts
  expr: rate(api_cost_total[1h]) * 720 > 10  # $10/month
  for: 1h
  annotations:
    summary: "API costs trending above budget"
```

## Performance Optimization Tips

### 1. Monitor Response Times

```promql
# Identify slow endpoints
topk(5, histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])))
```

### 2. Track API Efficiency

```promql
# Claude API calls per request
rate(claude_api_calls_total[5m]) / rate(http_requests_total[5m])
```

### 3. Cache Effectiveness

```promql
# Cache operations per request
rate(cache_operations_total[5m]) / rate(http_requests_total[5m])
```

### 4. Database Load

```promql
# Queries per second
sum(rate(database_queries_total[5m]))
```

## Development vs Production

### Development Setup

```python
# In development, metrics are collected but not required
# Prometheus endpoint available at /metrics
# JSON metrics at /api/metrics for debugging
```

### Production Setup

1. **Deploy Prometheus**:
   ```bash
   docker run -p 9090:9090 -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```

2. **Deploy Grafana**:
   ```bash
   docker run -p 3000:3000 grafana/grafana
   ```

3. **Configure Alerting** (via Alertmanager or Grafana)

4. **Set up Log Aggregation** (optional, but recommended)

## Troubleshooting

### Metrics Not Appearing

1. Check endpoint: `curl http://localhost:8000/metrics`
2. Verify Prometheus scraping: Check Prometheus targets page
3. Check logs for errors during metrics collection

### High Memory Usage

Monitor metrics collection overhead:
```promql
# Metric collection latency should be < 10ms
rate(prometheus_target_scrape_duration_seconds_sum[5m])
```

### Missing Labels

Ensure all instrumentation includes proper labels:
```python
# Always include relevant context
metrics_collector.track_http_request(
    endpoint="/api/conversation",
    method="POST",
    status=200,
    duration=0.5
)
```

## Best Practices

1. **Use labels wisely** - Don't create too many unique label combinations
2. **Monitor costs** - Track API usage to avoid surprises
3. **Set up alerts** - Be proactive about issues
4. **Review regularly** - Check metrics weekly to identify trends
5. **Optimize based on data** - Use metrics to guide performance improvements

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)

## Support

For issues or questions about monitoring:
1. Check this documentation
2. Review Grafana dashboard for insights
3. Check application logs
4. Open an issue on GitHub
