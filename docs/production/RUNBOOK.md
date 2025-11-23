# Operations Runbook

## System Overview

The Learning Voice Agent is a FastAPI application that provides:
- Voice conversation capture and AI responses
- Text-based conversation API
- Semantic search across captures
- Real-time WebSocket connections
- Admin monitoring dashboard

### Architecture
```
[Client] -> [CDN] -> [Railway Load Balancer] -> [FastAPI App (1-10 instances)]
                                                      |
                                    [Redis (Session State)]  [SQLite/PostgreSQL (Persistence)]
```

## Common Operations

### Viewing Logs

```bash
# Railway logs
railway logs --tail 100

# Filter by level
railway logs | grep ERROR

# Follow logs
railway logs -f
```

### Checking Health

```bash
# Basic health
curl https://api.yourdomain.com/

# Detailed health
curl https://api.yourdomain.com/admin/api/health/detailed \
  -H "X-Admin-Key: YOUR_ADMIN_KEY"

# System metrics
curl https://api.yourdomain.com/admin/api/metrics \
  -H "X-Admin-Key: YOUR_ADMIN_KEY"
```

### Scaling Manually

```bash
# Railway automatic scaling handles this, but for manual override:
# Go to Railway Dashboard > Service > Settings > Scaling

# Recommended configurations:
# - Light load: 1-2 instances
# - Normal: 2-4 instances
# - High traffic: 4-10 instances
```

### Database Operations

```bash
# Backup SQLite database
sqlite3 learning_captures.db ".backup backup_$(date +%Y%m%d).db"

# Check database stats
curl https://api.yourdomain.com/api/stats

# Search database
curl -X POST https://api.yourdomain.com/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 10}'
```

### Redis Operations

```bash
# Connect to Redis (if direct access available)
redis-cli -u $REDIS_URL

# Check memory usage
redis-cli -u $REDIS_URL INFO memory

# List active sessions
redis-cli -u $REDIS_URL KEYS "session:*"

# Flush expired sessions (careful!)
redis-cli -u $REDIS_URL --scan --pattern "session:*" | xargs redis-cli -u $REDIS_URL EXPIRE {} 1
```

## Troubleshooting

### High Latency

**Symptoms:**
- P95 > 2000ms
- Slow API responses
- User complaints about loading

**Diagnosis:**
```bash
# Check endpoint latency
curl https://api.yourdomain.com/admin/api/metrics/endpoints \
  -H "X-Admin-Key: YOUR_KEY"

# Check system resources
curl https://api.yourdomain.com/admin/api/metrics/system \
  -H "X-Admin-Key: YOUR_KEY"
```

**Resolution:**
1. Check if specific endpoint is slow (optimize query/logic)
2. Check Redis connection (network latency)
3. Check external API latency (Anthropic, OpenAI)
4. Scale up instances if CPU/memory high
5. Enable query caching if not already

### High Error Rate

**Symptoms:**
- Error rate > 0.1%
- 5xx responses in logs
- User complaints about failures

**Diagnosis:**
```bash
# Check recent errors
curl https://api.yourdomain.com/admin/api/metrics/errors?limit=50 \
  -H "X-Admin-Key: YOUR_KEY"

# Check logs for stack traces
railway logs | grep -A 10 "ERROR"
```

**Resolution:**
1. Check API key validity (Anthropic, OpenAI)
2. Check database connectivity
3. Check Redis connectivity
4. Review error patterns (specific endpoint?)
5. Check for resource exhaustion

### Memory Issues

**Symptoms:**
- Memory usage climbing over time
- OOM kills in logs
- Slow responses

**Diagnosis:**
```bash
# Check memory trend
curl https://api.yourdomain.com/admin/api/metrics/system \
  -H "X-Admin-Key: YOUR_KEY"
```

**Resolution:**
1. Restart affected instances
2. Check for memory leaks in recent changes
3. Review WebSocket connection cleanup
4. Check Redis memory usage
5. Scale horizontally to distribute load

### Connection Issues

**Symptoms:**
- Connection refused errors
- Timeouts
- Intermittent failures

**Diagnosis:**
```bash
# Check service status
railway status

# Check health endpoints
curl -v https://api.yourdomain.com/

# Check Redis
redis-cli -u $REDIS_URL PING
```

**Resolution:**
1. Check Railway service status
2. Verify DNS resolution
3. Check SSL certificate validity
4. Review network policies
5. Check for rate limiting

## Incident Response

### Severity Levels

| Level | Definition | Response Time | Example |
|-------|------------|---------------|---------|
| P1 - Critical | Service down | 15 minutes | Complete outage |
| P2 - High | Major feature broken | 1 hour | API not responding |
| P3 - Medium | Partial degradation | 4 hours | Slow responses |
| P4 - Low | Minor issue | 24 hours | Dashboard glitch |

### P1 Incident Procedure

1. **Acknowledge** (5 min)
   - Notify team
   - Start incident channel

2. **Assess** (10 min)
   - Check health endpoints
   - Review recent deployments
   - Check external dependencies

3. **Mitigate** (varies)
   - Rollback if deployment-related
   - Scale if resource-related
   - Failover if infrastructure-related

4. **Communicate**
   - Update status page
   - Notify affected users
   - Document timeline

5. **Resolve & Review**
   - Confirm resolution
   - Document root cause
   - Schedule post-mortem

### Rollback Procedure

```bash
# Quick rollback to previous deployment
railway rollback

# Verify health after rollback
curl https://api.yourdomain.com/admin/api/health/detailed \
  -H "X-Admin-Key: YOUR_KEY"
```

## Maintenance Windows

### Scheduled Maintenance
- Time: Sundays 02:00-04:00 UTC
- Notification: 48 hours advance
- Duration: 2 hours max

### Maintenance Checklist
- [ ] Notify users 48h in advance
- [ ] Enable maintenance mode
- [ ] Backup database
- [ ] Perform maintenance
- [ ] Verify functionality
- [ ] Disable maintenance mode
- [ ] Monitor for issues

## Contact Escalation

| Level | Contact | When |
|-------|---------|------|
| L1 | On-call engineer | First response |
| L2 | Team lead | P1/P2 incidents |
| L3 | Platform team | Infrastructure issues |

## Useful Commands Reference

```bash
# Deployment
railway up                    # Deploy
railway status                # Check status
railway rollback              # Rollback

# Logs
railway logs                  # View logs
railway logs -f               # Follow logs
railway logs --tail 100       # Last 100 lines

# Variables
railway variables             # List variables
railway variables set KEY=VAL # Set variable

# Database
sqlite3 db.db ".tables"       # List tables
sqlite3 db.db ".schema"       # Show schema

# Load Testing
locust -f tests/load/locustfile.py --host https://api.yourdomain.com

# Health Checks
curl https://api.yourdomain.com/
curl https://api.yourdomain.com/admin/api/health/detailed -H "X-Admin-Key: KEY"
```
