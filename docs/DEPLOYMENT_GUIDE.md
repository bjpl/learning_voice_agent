# Deployment Guide - Learning Voice Agent

**Version:** 1.0.0
**Last Updated:** 2025-11-21
**Target Platforms:** Railway, Docker, Cloud Providers

---

## Table of Contents

1. [Overview](#overview)
2. [Railway Deployment](#railway-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Environment Variables](#environment-variables)
5. [Monitoring Setup](#monitoring-setup)
6. [Backup Procedures](#backup-procedures)
7. [Scaling Considerations](#scaling-considerations)
8. [Cost Optimization](#cost-optimization)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Deployment Options

| Platform | Difficulty | Cost/month | Use Case |
|----------|-----------|-----------|----------|
| **Railway** | Easy | $5-20 | Recommended for production |
| **Docker** | Medium | $5-50 | Self-hosted, cloud VMs |
| **Fly.io** | Medium | $0-10 | Hobby projects |
| **DigitalOcean** | Medium | $12+ | Full control |
| **AWS/GCP** | Hard | $10-50 | Enterprise |

### Requirements

**All Deployments:**
- Python 3.11+
- Redis instance
- API keys (Anthropic, OpenAI)
- HTTPS/SSL (for production)

**Optional:**
- Twilio account (phone integration)
- Cloudflare account (CDN, tunnels)
- S3/R2 (backup storage)

---

## Railway Deployment

### Why Railway?

✅ **Pros:**
- One-click deployment
- Automatic HTTPS
- Built-in Redis support
- Auto-scaling
- GitHub integration
- Generous free tier

❌ **Cons:**
- US-only (currently)
- Limited customization
- Vendor lock-in

### Step-by-Step Deployment

#### 1. Install Railway CLI

```bash
# Install via npm
npm install -g @railway/cli

# Or via curl
curl -fsSL https://railway.app/install.sh | sh

# Verify installation
railway --version
```

#### 2. Login to Railway

```bash
# Login (opens browser)
railway login

# Verify
railway whoami
```

#### 3. Initialize Project

```bash
# Navigate to project
cd learning_voice_agent

# Initialize Railway project
railway init

# Follow prompts:
# - Create new project: Yes
# - Project name: learning-voice-agent
# - Environment: production
```

#### 4. Add Redis Service

```bash
# Add Redis plugin
railway add

# Select: Redis
# Confirm
```

This automatically:
- Creates Redis instance
- Sets `REDIS_URL` environment variable
- Configures networking

#### 5. Set Environment Variables

```bash
# Set API keys
railway variables set ANTHROPIC_API_KEY=sk-ant-api03-...
railway variables set OPENAI_API_KEY=sk-proj-...

# Optional: Twilio
railway variables set TWILIO_ACCOUNT_SID=ACxxxxxxxx
railway variables set TWILIO_AUTH_TOKEN=xxxxxxxx
railway variables set TWILIO_PHONE_NUMBER=+1234567890

# Application settings
railway variables set CLAUDE_MODEL=claude-3-haiku-20240307
railway variables set WHISPER_MODEL=whisper-1
railway variables set HOST=0.0.0.0
railway variables set PORT=8000

# Verify
railway variables
```

#### 6. Configure Deployment

Railway uses `railway.json` for configuration:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10,
    "healthcheckPath": "/",
    "healthcheckTimeout": 30
  }
}
```

#### 7. Deploy

```bash
# Deploy to Railway
railway up

# Monitor deployment
railway logs

# Once deployed, get URL
railway domain

# Example output:
# https://learning-voice-agent-production.up.railway.app
```

#### 8. Verify Deployment

```bash
# Health check
curl https://your-app.railway.app/

# Test conversation
curl -X POST https://your-app.railway.app/api/conversation \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am learning!"}'

# Check stats
curl https://your-app.railway.app/api/stats
```

#### 9. Set Up Custom Domain (Optional)

```bash
# Add custom domain
railway domain add your-domain.com

# Configure DNS:
# Add CNAME record:
# your-domain.com → your-app.railway.app

# Verify
railway domain list
```

#### 10. Enable Auto-Deploy

Link to GitHub for automatic deployments:

1. Go to Railway dashboard
2. Select your project
3. Settings → Connect to GitHub
4. Select repository
5. Configure branch (main)

Now every push to `main` triggers deployment!

---

### Railway Advanced Configuration

#### Environment-Specific Settings

```bash
# Create staging environment
railway environment create staging

# Switch to staging
railway environment use staging

# Deploy to staging
railway up

# Switch back to production
railway environment use production
```

#### Persistent Storage

```bash
# Create volume for SQLite
railway volume create

# Mount at /app/data
# Update Dockerfile to use /app/data/learning_captures.db
```

#### Scaling

```bash
# Scale up (via dashboard)
# Settings → Resources → Scale

# Or via CLI (coming soon)
railway scale --instances 3
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY static/ ./static/

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:////app/data/learning_captures.db
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # Optional: Cloudflare Tunnel
  cloudflared:
    image: cloudflare/cloudflared:latest
    command: tunnel --no-autoupdate run --token ${CLOUDFLARE_TOKEN}
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

### Deploy with Docker Compose

```bash
# Create .env file
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-proj-...
CLOUDFLARE_TOKEN=eyJh...  # Optional
EOF

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f app

# Check status
docker-compose ps

# Stop
docker-compose down
```

### DigitalOcean Deployment

```bash
# Create droplet (via web UI or CLI)
doctl compute droplet create voice-agent \
  --image ubuntu-22-04-x64 \
  --size s-1vcpu-1gb \
  --region nyc1 \
  --ssh-keys YOUR_SSH_KEY_ID

# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone repository
git clone https://github.com/your-repo/learning_voice_agent.git
cd learning_voice_agent

# Create .env
nano .env
# ... add API keys ...

# Run with Docker Compose
docker-compose up -d

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

---

## Environment Variables

### Required Variables

```bash
# AI APIs
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-proj-...

# Redis
REDIS_URL=redis://localhost:6379
# Railway auto-sets this ☝️
```

### Optional Variables

```bash
# Database
DATABASE_URL=sqlite:///./learning_captures.db

# Server
HOST=0.0.0.0
PORT=8000

# Twilio (optional)
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890

# Claude Configuration
CLAUDE_MODEL=claude-3-haiku-20240307
CLAUDE_MAX_TOKENS=150
CLAUDE_TEMPERATURE=0.7

# Audio
WHISPER_MODEL=whisper-1
MAX_AUDIO_DURATION=60

# Session
SESSION_TIMEOUT=180
MAX_CONTEXT_EXCHANGES=5
REDIS_TTL=1800

# CORS
CORS_ORIGINS=["*"]  # Restrict in production!

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Security Best Practices

**Never commit secrets:**

```bash
# .gitignore should include:
.env
.env.local
.env.production
*.db
*.log
```

**Use secrets management:**

```bash
# Railway
railway variables set API_KEY=... --secret

# Docker secrets
echo "sk-ant-..." | docker secret create anthropic_key -

# Environment file encryption
sops --encrypt .env > .env.enc
sops --decrypt .env.enc > .env
```

---

## Monitoring Setup

### Health Checks

**Endpoint:**

```bash
GET /

Response:
{
  "status": "healthy",
  "service": "Learning Voice Agent",
  "version": "1.0.0"
}
```

**Uptime Monitoring:**

```bash
# UptimeRobot (free)
# Add monitor:
# URL: https://your-app.railway.app/
# Interval: 5 minutes

# Healthchecks.io
curl https://hc-ping.com/YOUR_UUID

# Add to cron:
*/5 * * * * curl -fsS --retry 3 https://hc-ping.com/YOUR_UUID > /dev/null
```

### Application Monitoring

**Sentry (Error Tracking):**

```bash
# Install Sentry SDK
pip install sentry-sdk[fastapi]
```

```python
# app/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="https://...@sentry.io/...",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)
```

**Prometheus (Metrics):**

```bash
# Install prometheus-client
pip install prometheus-client
```

```python
# app/main.py
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
conversation_requests = Counter('conversation_requests_total', 'Total conversations')
conversation_duration = Histogram('conversation_duration_seconds', 'Conversation latency')

# Expose metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**Logs:**

```bash
# Railway logs
railway logs --tail 100

# Docker logs
docker-compose logs -f --tail 100 app

# Save logs
railway logs > logs_$(date +%Y%m%d).txt
```

### Alerts

**Railway Notifications:**
- Settings → Notifications
- Email, Slack, Discord
- Deployment status
- Resource usage

**UptimeRobot Alerts:**
- Email on downtime
- SMS (paid plan)
- Webhook integrations

---

## Backup Procedures

### SQLite Backup

**Manual Backup:**

```bash
# Local
cp learning_captures.db backups/backup_$(date +%Y%m%d).db

# Railway (via Railway CLI)
railway run sqlite3 learning_captures.db ".backup backup.db"
railway run cat backup.db > local_backup.db
```

**Automated Backup (Cron):**

```bash
# backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
sqlite3 learning_captures.db ".backup /backups/backup_$DATE.db"
find /backups -mtime +30 -delete  # Keep 30 days

# Cron (daily at 2 AM)
0 2 * * * /path/to/backup.sh
```

**Litestream (Continuous Backup):**

```bash
# Install Litestream
curl -fsSL https://github.com/benbjohnson/litestream/releases/download/v0.3.9/litestream-v0.3.9-linux-amd64.deb -o litestream.deb
sudo dpkg -i litestream.deb

# Configure
cat > litestream.yml <<EOF
dbs:
  - path: /app/learning_captures.db
    replicas:
      - url: s3://mybucket/learning_captures.db
        access-key-id: $AWS_ACCESS_KEY_ID
        secret-access-key: $AWS_SECRET_ACCESS_KEY
EOF

# Run
litestream replicate -config litestream.yml
```

**Cloudflare R2 Backup:**

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure R2
rclone config create r2 s3 \
  provider Cloudflare \
  access_key_id $R2_ACCESS_KEY \
  secret_access_key $R2_SECRET_KEY \
  endpoint $R2_ENDPOINT

# Backup
rclone copy learning_captures.db r2:backups/
```

### Redis Backup

**RDB Snapshot:**

```bash
# Trigger save
redis-cli BGSAVE

# Copy RDB file
cp /var/lib/redis/dump.rdb backups/redis_$(date +%Y%m%d).rdb
```

**AOF (Append-Only File):**

```bash
# Enable AOF
redis-cli CONFIG SET appendonly yes

# Backup AOF
cp /var/lib/redis/appendonly.aof backups/
```

### Restore Procedures

**SQLite Restore:**

```bash
# Stop application
railway down  # or docker-compose down

# Restore backup
cp backups/backup_20251121.db learning_captures.db

# Start application
railway up  # or docker-compose up -d

# Verify
curl https://your-app.railway.app/api/stats
```

**Redis Restore:**

```bash
# Stop Redis
docker-compose stop redis

# Restore RDB
cp backups/redis_20251121.rdb /var/lib/redis/dump.rdb

# Start Redis
docker-compose start redis
```

---

## Scaling Considerations

### Current Limitations (v1.0)

**Single Instance:**
- SQLite doesn't support horizontal scaling
- Redis is single instance
- No load balancing

**Bottlenecks:**
- AI API latency (Whisper + Claude)
- SQLite write concurrency
- Redis network latency

**Recommended Limits:**
- Concurrent users: < 50
- Total captures: < 1M
- Request rate: < 100 req/sec

### Scaling Strategies

**Vertical Scaling (Easy):**

```bash
# Railway: Increase resources
# Settings → Resources
# - CPU: 1 → 2 vCPUs
# - Memory: 512MB → 2GB

# Docker: Update docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
```

**Horizontal Scaling (v2.0 required):**

Prerequisites:
- Switch to PostgreSQL
- Stateless sessions (JWT)
- Redis Cluster
- Load balancer

```yaml
# docker-compose.yml (v2.0)
services:
  app:
    image: voice-agent:latest
    deploy:
      replicas: 3
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_CLUSTER_NODES=redis1:6379,redis2:6379,redis3:6379
```

**Database Scaling:**

v1.0: SQLite (single file)
→ v2.0: PostgreSQL with read replicas

```
                   ┌──────────┐
                   │  Primary │
                   │    PG    │
                   └────┬─────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Replica │    │ Replica │    │ Replica │
    │    1    │    │    2    │    │    3    │
    └─────────┘    └─────────┘    └─────────┘
      (reads)        (reads)        (reads)
```

**Caching Layer:**

```python
# Add Redis caching for frequent queries
@cache(ttl=300)  # 5 minutes
async def get_popular_searches():
    return await db.query("SELECT ... FROM captures")
```

---

## Cost Optimization

### Current Costs (v1.0)

**Per-Conversation Costs:**
- Whisper API: $0.0006/min (~$0.0005 per 5-sec clip)
- Claude Haiku: $0.00025 per request (150 tokens)
- **Total: ~$0.00075 per exchange**

**Monthly Costs (100 exchanges/day):**

| Service | Cost/month |
|---------|-----------|
| Claude API | $2.25 |
| Whisper API | $1.50 |
| Railway (Starter) | $5.00 |
| Redis (Railway plugin) | $0.00 |
| **Total** | **$8.75** |

**Heavy Usage (1000 exchanges/day):**

| Service | Cost/month |
|---------|-----------|
| Claude API | $22.50 |
| Whisper API | $15.00 |
| Railway (Pro) | $20.00 |
| Redis | $0.00 |
| **Total** | **$57.50** |

### Optimization Strategies

**1. Use Cheaper Models:**

```python
# Replace Haiku with even smaller model (when available)
CLAUDE_MODEL=claude-3-instant-20240307  # Hypothetical

# Or use GPT-3.5 (cheaper but different API)
```

**2. Reduce Token Usage:**

```python
# Shorter max_tokens
CLAUDE_MAX_TOKENS=100  # Down from 150

# Smaller context window
MAX_CONTEXT_EXCHANGES=3  # Down from 5
```

**3. Batch Requests:**

```python
# Process multiple inputs in one API call
batch = [input1, input2, input3]
results = await claude.batch_create(batch)
```

**4. Cache Responses:**

```python
# Cache common responses
@cache(key="response:{hash(user_text)}", ttl=3600)
async def generate_response(user_text):
    # ...
```

**5. Use Railway Free Tier:**

```
Free Tier Limits:
- $5 credit/month
- 500 hours execution
- Shared CPU/memory

Perfect for:
- Development
- Personal use
- Low traffic apps
```

**6. Self-Host:**

```bash
# Run on $6/month VPS (Hetzner, Vultr)
# - 1 vCPU
# - 2GB RAM
# - 40GB SSD

# Only pay for AI APIs
# Total: $5-10/month (AI only)
```

---

## Troubleshooting

### Deployment Fails

**Issue: Build fails**

```bash
# Check logs
railway logs

# Common causes:
# 1. Missing dependencies
# 2. Python version mismatch
# 3. Dockerfile errors

# Fix: Update requirements.txt
pip freeze > requirements.txt
git commit -am "fix: update dependencies"
railway up
```

**Issue: Health check fails**

```bash
# Check health endpoint
curl https://your-app.railway.app/

# If times out, check:
# 1. App is listening on 0.0.0.0:$PORT
# 2. Health check path is correct
# 3. Firewall allows traffic
```

**Issue: Environment variables not set**

```bash
# List variables
railway variables

# Set missing variables
railway variables set KEY=value

# Restart
railway restart
```

---

### Runtime Errors

**Issue: Redis connection fails**

```bash
# Check Redis is running
railway logs redis

# Verify REDIS_URL
railway variables | grep REDIS_URL

# Test connection
redis-cli -u $REDIS_URL ping
```

**Issue: API rate limits**

```python
# Add exponential backoff
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(min=1, max=60))
async def call_claude_api():
    # ...
```

**Issue: Database locked**

```bash
# Check connections
lsof learning_captures.db

# Kill processes
pkill -f "python -m app.main"

# Delete WAL files
rm *.db-wal *.db-shm

# Restart
railway restart
```

---

### Performance Issues

**Issue: Slow response times**

```bash
# Check latency breakdown
curl -w "@curl-format.txt" https://your-app.railway.app/api/conversation

# curl-format.txt:
time_namelookup:  %{time_namelookup}\n
time_connect:  %{time_connect}\n
time_total:  %{time_total}\n
```

**Issue: High memory usage**

```bash
# Check memory
railway metrics

# If high:
# 1. Reduce context window
# 2. Clear Redis periodically
# 3. Increase resources
```

**Issue: Database too large**

```bash
# Check size
du -h learning_captures.db

# Archive old data
sqlite3 learning_captures.db "
  DELETE FROM captures
  WHERE timestamp < datetime('now', '-90 days')
"

# Vacuum
sqlite3 learning_captures.db "VACUUM"
```

---

## Production Checklist

Before going live:

- [ ] All API keys configured
- [ ] HTTPS enabled (automatic on Railway)
- [ ] Custom domain configured
- [ ] Health checks working
- [ ] Monitoring set up (Sentry, UptimeRobot)
- [ ] Backup strategy configured
- [ ] Error tracking enabled
- [ ] Logs accessible
- [ ] CORS configured correctly
- [ ] Rate limiting enabled (v2.0)
- [ ] Database backed up
- [ ] Rollback plan documented
- [ ] Load testing completed
- [ ] Security audit performed
- [ ] Documentation updated

---

## Next Steps

- Monitor application: Check Railway dashboard regularly
- Set up alerts: Configure UptimeRobot, Sentry
- Plan v2.0: Review [MIGRATION_PLAN.md](./MIGRATION_PLAN.md)
- Optimize costs: Review usage monthly

---

**Need Help?**
- Railway Docs: https://docs.railway.app
- Discord: https://discord.gg/railway
- GitHub Issues: Create an issue

**Deployment successful!** 🎉
