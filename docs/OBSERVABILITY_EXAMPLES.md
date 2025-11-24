# Observability Examples

This document provides practical examples of using the observability features in the Learning Voice Agent.

## Basic Usage

### 1. Check Application Health

```bash
curl http://localhost:8000/api/health | jq
```

**Example Response:**
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
      "status": "configured",
      "last_check": "2025-01-21T11:59:00Z"
    },
    "whisper_api": {
      "status": "configured",
      "last_check": "2025-01-21T11:59:00Z"
    }
  },
  "metrics": {
    "active_sessions": 5,
    "uptime_seconds": 3600.5
  }
}
```

### 2. View JSON Metrics

```bash
curl http://localhost:8000/api/metrics | jq
```

**Example Response:**
```json
{
  "timestamp": "2025-01-21T12:00:00Z",
  "application": {
    "name": "learning_voice_agent",
    "version": "1.0.0"
  },
  "requests": {
    "total": 1523,
    "in_progress": 2
  },
  "sessions": {
    "active": 8,
    "total": 145
  },
  "websockets": {
    "active": 3
  },
  "errors": {
    "total": 5
  },
  "external_apis": {
    "claude": {
      "calls": 892
    },
    "whisper": {
      "calls": 673
    }
  },
  "cache": {
    "hit_ratio": 0.78
  },
  "costs": {
    "total_usd": 12.45
  }
}
```

### 3. Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

**Example Output:**
```prometheus
# HELP http_requests_total Total HTTP requests by endpoint and method
# TYPE http_requests_total counter
http_requests_total{endpoint="/api/conversation",method="POST",status="200"} 452.0
http_requests_total{endpoint="/api/search",method="POST",status="200"} 89.0
http_requests_total{endpoint="/ws/{session_id}",method="GET",status="200"} 34.0

# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{endpoint="/api/conversation",method="POST",le="0.1"} 234.0
http_request_duration_seconds_bucket{endpoint="/api/conversation",method="POST",le="0.5"} 421.0
http_request_duration_seconds_sum{endpoint="/api/conversation",method="POST"} 156.7
http_request_duration_seconds_count{endpoint="/api/conversation",method="POST"} 452.0

# HELP claude_api_calls_total Total Claude API calls
# TYPE claude_api_calls_total counter
claude_api_calls_total{model="claude-3-haiku-20240307",status="success"} 886.0
claude_api_calls_total{model="claude-3-haiku-20240307",status="error"} 6.0
```

## Monitoring Common Scenarios

### Monitor API Performance

```bash
# Watch metrics in real-time (updates every 2 seconds)
watch -n 2 'curl -s http://localhost:8000/api/metrics | jq ".external_apis"'
```

### Track Costs

```bash
# Get current cost estimate
curl -s http://localhost:8000/api/metrics | jq ".costs"

# Monitor cost trends with Prometheus
# Query: rate(api_cost_total[1h]) * 3600
```

### Monitor Active Sessions

```bash
# Get current sessions
curl -s http://localhost:8000/api/metrics | jq ".sessions"

# List session IDs
curl -s http://localhost:8000/api/stats | jq ".sessions.ids"
```

### Check Error Rates

```bash
# Get total errors
curl -s http://localhost:8000/api/metrics | jq ".errors"

# Detailed error breakdown from Prometheus
curl -s http://localhost:8000/metrics | grep errors_total
```

## PromQL Query Examples

### Request Rate

```promql
# Requests per second by endpoint
rate(http_requests_total[5m])

# Total requests across all endpoints
sum(rate(http_requests_total[5m]))

# Requests per minute
rate(http_requests_total[1m]) * 60
```

### Response Time Analysis

```promql
# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# 50th percentile (median)
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))

# Average response time
avg(rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m]))

# Max response time in last 5 minutes
max_over_time(http_request_duration_seconds_bucket[5m])
```

### Error Rate Monitoring

```promql
# Error rate (errors per second)
rate(errors_total[5m])

# Error percentage
(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) * 100

# Errors by component
sum by (component) (rate(errors_total[5m]))
```

### API Performance

```promql
# Claude API success rate
rate(claude_api_calls_total{status="success"}[5m]) / rate(claude_api_calls_total[5m])

# Average Claude API response time
avg(rate(claude_api_duration_seconds_sum[5m]) / rate(claude_api_duration_seconds_count[5m]))

# Whisper API call rate
rate(whisper_api_calls_total[5m])
```

### Cost Tracking

```promql
# Cost per hour
rate(api_cost_total[1h]) * 3600

# Projected daily cost
rate(api_cost_total[1h]) * 86400

# Cost breakdown by provider
sum by (provider) (rate(api_cost_total[1h]) * 3600)
```

### Session Metrics

```promql
# Active sessions
sessions_active

# Session creation rate
rate(sessions_total[5m])

# Average session duration
histogram_quantile(0.50, rate(session_duration_seconds_bucket[5m]))
```

### Cache Performance

```promql
# Cache hit ratio
cache_hit_ratio

# Cache operations per second
rate(cache_operations_total[5m])

# Cache misses per second
rate(cache_operations_total{status="miss"}[5m])
```

### WebSocket Monitoring

```promql
# Active WebSocket connections
websocket_connections_active

# WebSocket message rate
rate(websocket_messages_total[5m])

# Average connection duration
histogram_quantile(0.50, rate(websocket_connection_duration_seconds_bucket[5m]))
```

## Grafana Dashboard Setup

### 1. Add Prometheus Data Source

1. Go to Configuration → Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL to your Prometheus instance (e.g., `http://localhost:9090`)
5. Click "Save & Test"

### 2. Import Dashboard

1. Go to Dashboards → Import
2. Upload `/monitoring/grafana_dashboard.json`
3. Select your Prometheus data source
4. Click "Import"

### 3. Customize Panels

Add custom panels for your specific needs:

**Request Rate Panel:**
- Visualization: Graph
- Query: `rate(http_requests_total[5m])`
- Legend: `{{endpoint}} - {{method}}`

**Error Rate Panel:**
- Visualization: Graph
- Query: `rate(errors_total[5m])`
- Legend: `{{component}} - {{error_type}}`
- Alert: Trigger when error rate > 0.1

**Cost Tracking Panel:**
- Visualization: Stat
- Query: `increase(api_cost_total[24h])`
- Unit: Currency (USD)

## Alert Examples

### Prometheus Alerting Rules

```yaml
groups:
  - name: learning_voice_agent
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "Error rate exceeds 10%"
          description: "Error rate is {{ $value }} errors/sec"

      # API failures
      - alert: ClaudeAPIFailures
        expr: rate(claude_api_calls_total{status="error"}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Claude API error rate > 5%"

      # Slow responses
      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        annotations:
          summary: "95th percentile response time > 2s"

      # High costs
      - alert: HighAPICosts
        expr: rate(api_cost_total[1h]) * 720 > 10
        for: 1h
        annotations:
          summary: "API costs trending above $10/month"

      # Low cache hit ratio
      - alert: LowCacheHitRatio
        expr: cache_hit_ratio < 0.5
        for: 10m
        annotations:
          summary: "Cache hit ratio below 50%"

      # Database issues
      - alert: SlowDatabaseQueries
        expr: histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        annotations:
          summary: "Database queries taking > 100ms"
```

## Troubleshooting

### Metrics Not Updating

```bash
# Check if metrics endpoint is accessible
curl -I http://localhost:8000/metrics

# Verify Prometheus is scraping
# Go to http://localhost:9090/targets
```

### High Memory Usage

```bash
# Check number of unique metric combinations
curl -s http://localhost:8000/metrics | grep -c "^[a-z]"

# If too high, reduce label cardinality
```

### Missing Data Points

```bash
# Check application logs
tail -f logs/app.log | grep -i error

# Verify dependencies are healthy
curl http://localhost:8000/api/health
```

## Best Practices

1. **Set up alerts** - Don't wait for problems to find you
2. **Monitor trends** - Look for gradual degradation over time
3. **Track costs** - API usage can add up quickly
4. **Review metrics weekly** - Identify patterns and optimization opportunities
5. **Use labels sparingly** - Too many unique labels can cause performance issues
6. **Archive old data** - Keep Prometheus database size manageable
7. **Test alerts** - Regularly verify alerting is working

## Integration with Existing Tools

### Datadog

```python
# Add to app/metrics.py
from datadog import statsd

# Track metrics
statsd.increment('voice_agent.requests')
statsd.histogram('voice_agent.response_time', duration)
```

### New Relic

```python
# Add to app/main.py
import newrelic.agent

@app.middleware("http")
async def newrelic_middleware(request, call_next):
    with newrelic.agent.BackgroundTask(application, name='web-transaction'):
        return await call_next(request)
```

### CloudWatch

```python
# Add to app/metrics.py
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='VoiceAgent',
    MetricData=[{
        'MetricName': 'RequestCount',
        'Value': 1,
        'Unit': 'Count'
    }]
)
```
