# Deployment & Monitoring Guide

## Quick Start: Development Environment

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Verify Observability Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# JSON metrics
curl http://localhost:8000/api/metrics | jq

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Production Deployment with Monitoring

### Architecture

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Load Balancer  │
└─────────┬───────┘
          │
          ▼
┌──────────────────────────────┐
│  Learning Voice Agent (x3)   │
│  Port: 8000                  │
│  /metrics endpoint           │
└─────────┬────────────────────┘
          │
          ├──────────────┐
          ▼              ▼
    ┌─────────┐    ┌──────────┐
    │Database │    │  Redis   │
    │SQLite/PG│    │  Cache   │
    └─────────┘    └──────────┘
          │
          ▼
    ┌────────────────────────────┐
    │  Monitoring Stack          │
    ├────────────────────────────┤
    │ • Prometheus (metrics)     │
    │ • Grafana (visualization)  │
    │ • Alertmanager (alerts)    │
    └────────────────────────────┘
```

### Step 1: Docker Compose Setup

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  # Application
  voice-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/voice_agent
      - REDIS_URL=redis://redis:6379
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  # Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: voice_agent
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana_dashboard.json:/etc/grafana/provisioning/dashboards/voice_agent.json
    depends_on:
      - prometheus
    restart: unless-stopped

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  alertmanager_data:
```

### Step 2: Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Load alerting rules
rule_files:
  - /etc/prometheus/alerts.yml

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Scrape configurations
scrape_configs:
  - job_name: 'voice-agent'
    scrape_interval: 10s
    static_configs:
      - targets: ['voice-agent:8000']
    metrics_path: /metrics

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Step 3: Alert Rules

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: voice_agent_alerts
    interval: 30s
    rules:
      # Critical Alerts
      - alert: ServiceDown
        expr: up{job="voice-agent"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Voice Agent service is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: ClaudeAPIFailures
        expr: rate(claude_api_calls_total{status="error"}[5m]) / rate(claude_api_calls_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Claude API error rate > 5%"
          description: "{{ $value | humanizePercentage }} of Claude API calls are failing"

      # Warning Alerts
      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow response times detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 > 512
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB"

      - alert: LowCacheHitRatio
        expr: cache_hit_ratio < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit ratio"
          description: "Cache hit ratio is {{ $value | humanizePercentage }}"

      - alert: HighAPICosts
        expr: rate(api_cost_total[1h]) * 720 > 10
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "API costs trending high"
          description: "Projected monthly cost: ${{ $value }}"
```

### Step 4: Alertmanager Configuration

Create `monitoring/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourcompany.com'
  smtp_auth_username: 'alerts@yourcompany.com'
  smtp_auth_password: 'your-password'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true

    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@yourcompany.com'

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourcompany.com'
    # Optional: Slack integration
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@yourcompany.com'
```

### Step 5: Deploy

```bash
# Start all services
docker-compose -f docker-compose.monitoring.yml up -d

# Check services
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f voice-agent
```

### Step 6: Access Dashboards

- **Application**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

## Cloud Deployment (AWS Example)

### Using AWS ECS + CloudWatch

1. **Build and push Docker image:**

```bash
docker build -t voice-agent:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_REPO
docker tag voice-agent:latest YOUR_ECR_REPO/voice-agent:latest
docker push YOUR_ECR_REPO/voice-agent:latest
```

2. **Create ECS Task Definition with metrics:**

```json
{
  "family": "voice-agent",
  "containerDefinitions": [{
    "name": "voice-agent",
    "image": "YOUR_ECR_REPO/voice-agent:latest",
    "portMappings": [{
      "containerPort": 8000,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "DATABASE_URL", "value": "postgresql://..."},
      {"name": "REDIS_URL", "value": "redis://..."}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/health || exit 1"],
      "interval": 30,
      "timeout": 5,
      "retries": 3
    }
  }]
}
```

3. **Setup CloudWatch metrics:**

```python
# Add to app/metrics.py for CloudWatch integration
import boto3

cloudwatch = boto3.client('cloudwatch')

def push_to_cloudwatch():
    """Push key metrics to CloudWatch"""
    metrics_data = metrics_collector.get_metrics_dict()

    cloudwatch.put_metric_data(
        Namespace='VoiceAgent',
        MetricData=[
            {
                'MetricName': 'ActiveSessions',
                'Value': metrics_data['sessions']['active'],
                'Unit': 'Count'
            },
            {
                'MetricName': 'ErrorRate',
                'Value': metrics_data['errors']['total'],
                'Unit': 'Count'
            }
        ]
    )
```

## Kubernetes Deployment

### 1. Create Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-agent
  template:
    metadata:
      labels:
        app: voice-agent
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: voice-agent
        image: voice-agent:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: voice-agent-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 2. Create Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: voice-agent
  labels:
    app: voice-agent
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: voice-agent
```

### 3. Deploy

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl logs -f deployment/voice-agent
```

## Monitoring Best Practices

### 1. Set Up Alerts

- **Critical**: Service down, high error rate, API failures
- **Warning**: Slow responses, high costs, low cache hit ratio
- **Info**: Deployment events, scaling events

### 2. Regular Reviews

- Weekly: Review metrics trends
- Monthly: Analyze costs and optimization opportunities
- Quarterly: Update alert thresholds based on patterns

### 3. Dashboard Organization

Create multiple dashboards for different audiences:

- **Operations Dashboard**: System health, errors, performance
- **Business Dashboard**: Usage metrics, costs, user engagement
- **Development Dashboard**: API performance, database queries, cache hit ratios

### 4. Retention Policies

- **Prometheus**: 30 days of metrics
- **Logs**: 90 days in rotating files
- **Long-term storage**: Export to S3/Cloud Storage for analysis

## Troubleshooting

### Metrics Not Appearing

```bash
# Check if metrics endpoint works
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Check Prometheus logs
docker logs prometheus
```

### High Memory Usage

```bash
# Check Prometheus memory
docker stats prometheus

# Reduce retention if needed
--storage.tsdb.retention.time=7d
```

### Alert Fatigue

- Review and adjust alert thresholds
- Use proper grouping and routing
- Implement escalation policies
- Add context to alert messages

## Performance Optimization

### 1. Reduce Metric Cardinality

```python
# Bad: Too many unique labels
metrics.labels(session_id=session_id)

# Good: Use aggregated labels
metrics.labels(session_type="conversation")
```

### 2. Async Metrics Collection

All metrics are collected asynchronously with zero blocking.

### 3. Sampling for High-Volume Metrics

```python
# Sample 10% of requests for detailed tracking
if random.random() < 0.1:
    track_detailed_metrics()
```

## Security Considerations

1. **Protect metrics endpoint** - Use authentication in production
2. **Don't expose sensitive data** - No PII in metric labels
3. **Secure Grafana** - Strong passwords, HTTPS only
4. **Network isolation** - Keep monitoring stack in private network

## Cost Management

### Track API Costs

The metrics automatically track estimated costs for:
- Claude API calls (by tokens)
- Whisper API calls (by audio duration)

### Optimize Costs

```promql
# Cost per user session
rate(api_cost_total[1h]) / rate(sessions_total[1h])

# Most expensive endpoints
topk(5, sum by (endpoint) (api_cost_total))
```

## Next Steps

1. ✅ Deploy monitoring stack
2. ✅ Import Grafana dashboards
3. ✅ Configure alerts
4. ✅ Set up on-call rotation
5. ✅ Document runbooks
6. ✅ Train team on dashboards
7. ✅ Review and iterate
