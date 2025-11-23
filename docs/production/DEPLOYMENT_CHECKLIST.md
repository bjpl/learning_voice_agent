# Production Deployment Checklist

## Pre-Deployment Verification

### Code Quality
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code coverage above 80% (`pytest --cov=app`)
- [ ] No critical security vulnerabilities (Trivy scan)
- [ ] No hardcoded secrets in codebase
- [ ] Linting passes (`ruff check app/`)
- [ ] Type hints validated (`mypy app/`)

### Infrastructure
- [ ] Docker image builds successfully
- [ ] Docker image scanned for vulnerabilities
- [ ] Environment variables documented
- [ ] Secrets stored in Railway/CI secrets manager
- [ ] Database migrations applied
- [ ] Redis connection tested

### Performance
- [ ] Load test completed (1000 concurrent users)
- [ ] P95 latency < 2 seconds
- [ ] Error rate < 0.1%
- [ ] Memory usage stable under load
- [ ] No memory leaks detected

## Environment Variables

### Required
```bash
ANTHROPIC_API_KEY=sk-ant-...    # Claude API key
OPENAI_API_KEY=sk-...            # Whisper API key
REDIS_URL=redis://...            # Redis connection URL
DATABASE_URL=sqlite:///...       # Database connection
```

### Optional
```bash
TWILIO_ACCOUNT_SID=...           # For phone integration
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=...
ADMIN_API_KEY=...                # Admin dashboard access
```

### Production Settings
```bash
LOG_LEVEL=info
WORKERS=4
PORT=8000
CORS_ORIGINS=["https://yourdomain.com"]
```

## Railway Deployment Steps

### 1. Initial Setup
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Link to project
railway link
```

### 2. Configure Environment
```bash
# Set required variables
railway variables set ANTHROPIC_API_KEY=sk-ant-xxx
railway variables set OPENAI_API_KEY=sk-xxx
railway variables set LOG_LEVEL=info
railway variables set WORKERS=4
```

### 3. Deploy
```bash
# Deploy to Railway
railway up

# Check deployment status
railway status

# View logs
railway logs
```

### 4. Configure Auto-Scaling
Railway automatically handles scaling based on `railway.toml`:
- Min replicas: 1
- Max replicas: 10
- CPU threshold: 70%
- Memory threshold: 80%

### 5. Setup Custom Domain
```bash
# In Railway dashboard:
# 1. Go to Settings > Domains
# 2. Add custom domain
# 3. Configure DNS records
# 4. Enable HTTPS (automatic with Railway)
```

## CDN Configuration

### Cloudflare Setup
1. Add site to Cloudflare
2. Update nameservers
3. Configure caching rules:

```
# Page Rules
*.js, *.css, *.png, *.jpg - Cache Level: Cache Everything, Edge TTL: 1 month
/api/* - Cache Level: Bypass
/admin/* - Cache Level: Bypass
```

### Cache Headers (Built-in)
The application includes CDN-friendly headers via middleware:
- Static assets: 1 year cache, immutable
- HTML: 1 hour with revalidation
- API: No cache

## Monitoring Setup

### Health Endpoints
- `GET /` - Basic health check
- `GET /admin/api/health/detailed` - Component health
- `GET /admin/api/metrics` - Performance metrics

### External Monitoring (Recommended)
1. **Uptime Monitoring**: UptimeRobot, Pingdom
   - Monitor `https://yourdomain.com/`
   - Alert on 5xx errors
   - Check interval: 1 minute

2. **APM**: Datadog, New Relic
   - Application performance monitoring
   - Distributed tracing
   - Error tracking

3. **Logging**: Logtail, Papertrail
   - Centralized log aggregation
   - Search and alerting
   - Log retention

### Alert Thresholds
| Metric | Warning | Critical |
|--------|---------|----------|
| P95 Latency | > 1000ms | > 2000ms |
| Error Rate | > 0.5% | > 1% |
| CPU Usage | > 70% | > 90% |
| Memory Usage | > 70% | > 85% |
| Uptime | < 99.95% | < 99.9% |

## Post-Deployment Verification

### Smoke Tests
```bash
# Health check
curl https://yourdomain.com/

# API documentation
curl https://yourdomain.com/docs

# Admin dashboard (with key)
curl https://yourdomain.com/admin/dashboard?admin_key=xxx
```

### Load Test (Production Smoke)
```bash
# Light load test against production
locust -f tests/load/locustfile.py \
    --host https://yourdomain.com \
    --headless \
    --users 10 \
    --spawn-rate 2 \
    --run-time 2m
```

## Rollback Procedure

### Automatic Rollback (Railway)
Railway automatically rolls back on failed health checks.

### Manual Rollback
```bash
# List recent deployments
railway deployments

# Rollback to previous
railway rollback
```

### Database Rollback
```bash
# If using Alembic for migrations
alembic downgrade -1
```

## Security Checklist

### API Security
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] API keys not exposed in logs
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)

### Infrastructure Security
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] Secrets not in version control
- [ ] Minimal container permissions (non-root user)
- [ ] Network policies configured

### Monitoring Security
- [ ] Admin dashboard protected
- [ ] Audit logging enabled
- [ ] Anomaly detection configured

## Emergency Contacts

| Role | Contact | Responsibility |
|------|---------|----------------|
| DevOps Lead | - | Infrastructure issues |
| Backend Lead | - | Application issues |
| Security | - | Security incidents |

## Runbook Links

- [Incident Response](/docs/production/INCIDENT_RESPONSE.md)
- [Scaling Procedures](/docs/production/SCALING.md)
- [Database Operations](/docs/production/DATABASE_OPS.md)
