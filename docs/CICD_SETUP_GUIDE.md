# CI/CD Setup Guide - Quick Start

## Overview

This guide will help you set up the complete CI/CD pipeline for the Learning Voice Agent project.

## Prerequisites

- GitHub repository with admin access
- Railway account (https://railway.app)
- Docker Hub account (https://hub.docker.com)
- API keys: Anthropic, OpenAI
- (Optional) Slack webhook for notifications
- (Optional) Codecov account for coverage tracking

## Step 1: Configure GitHub Secrets

Go to your repository Settings → Secrets and variables → Actions, then add:

### Railway Configuration
```bash
RAILWAY_TOKEN                    # Get from: railway login --browserless
RAILWAY_STAGING_PROJECT_ID       # Create staging project first
RAILWAY_STAGING_DOMAIN          # staging.yourapp.railway.app
RAILWAY_PRODUCTION_PROJECT_ID   # Create production project
RAILWAY_PRODUCTION_DOMAIN       # yourapp.railway.app
```

### API Keys
```bash
ANTHROPIC_API_KEY               # Production key
OPENAI_API_KEY                  # Production key
ANTHROPIC_API_KEY_STAGING       # Staging key (or same as prod for testing)
OPENAI_API_KEY_STAGING          # Staging key (or same as prod for testing)
```

### Docker Hub
```bash
DOCKER_USERNAME                 # Your Docker Hub username
DOCKER_PASSWORD                 # Access token from Docker Hub
```

### Optional Services
```bash
CODECOV_TOKEN                   # From codecov.io
SLACK_WEBHOOK_URL              # From Slack app configuration
```

## Step 2: Set Up Railway Projects

### Create Staging Environment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create staging project
railway init
railway up
railway variables set ANTHROPIC_API_KEY=your-key
railway variables set OPENAI_API_KEY=your-key
railway variables set REDIS_URL=redis://redis:6379

# Add Redis service
railway add redis

# Get project ID
railway status
# Copy the project ID to GitHub secrets as RAILWAY_STAGING_PROJECT_ID
```

### Create Production Environment
```bash
# Create production project
railway init
railway up
railway variables set ANTHROPIC_API_KEY=your-key
railway variables set OPENAI_API_KEY=your-key
railway variables set REDIS_URL=redis://redis:6379

# Add Redis service
railway add redis

# Get project ID
railway status
# Copy the project ID to GitHub secrets as RAILWAY_PRODUCTION_PROJECT_ID
```

## Step 3: Enable GitHub Actions

1. Go to repository Settings → Actions → General
2. Under "Actions permissions", select "Allow all actions and reusable workflows"
3. Under "Workflow permissions", select "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"

## Step 4: Install Pre-commit Hooks (Local Development)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

## Step 5: Test the Pipeline

### Test Workflow
```bash
# Trigger test workflow
git checkout -b test/ci-pipeline
touch test-file.txt
git add .
git commit -m "test: CI pipeline"
git push origin test/ci-pipeline

# Check workflow status
gh run list --workflow=test.yml
```

### Test Staging Deployment
```bash
# Push to main or develop
git checkout main
git merge test/ci-pipeline
git push origin main

# Monitor deployment
gh run list --workflow=deploy-staging.yml
railway logs --project staging
```

### Test Production Deployment
```bash
# Create a release
git tag v1.0.0
git push origin v1.0.0

# Create GitHub release
gh release create v1.0.0 --title "v1.0.0" --notes "Initial release"

# Monitor production deployment
gh run list --workflow=deploy-production.yml
railway logs --project production
```

## Step 6: Verify Deployments

### Health Checks
```bash
# Staging
curl https://staging.yourapp.railway.app/health
curl https://staging.yourapp.railway.app/api/stats

# Production
curl https://yourapp.railway.app/health
curl https://yourapp.railway.app/api/stats
```

## Workflow Triggers

### Automatic Triggers

| Workflow | Trigger | Action |
|----------|---------|--------|
| Test Suite | Push to any branch | Runs tests with coverage |
| Code Quality | Push to any branch | Linting and type checking |
| Security Scan | Push to main/develop, Weekly | Security vulnerability scanning |
| Staging Deploy | Push to main/develop | Auto-deploy to staging |
| Production Deploy | Release published | Deploy to production |

### Manual Triggers

```bash
# Trigger staging deployment
gh workflow run deploy-staging.yml

# Trigger production deployment
gh workflow run deploy-production.yml -f version=v1.0.0

# Trigger security scan
gh workflow run security.yml
```

## Monitoring and Alerts

### View Workflow Runs
```bash
# List recent runs
gh run list --limit 10

# Watch a specific run
gh run watch

# View logs
gh run view --log
```

### Railway Logs
```bash
# View staging logs
railway logs --project $RAILWAY_STAGING_PROJECT_ID

# View production logs
railway logs --project $RAILWAY_PRODUCTION_PROJECT_ID

# Follow logs
railway logs --follow
```

### Slack Notifications

If configured, you'll receive Slack notifications for:
- ✅ Successful deployments
- ❌ Failed deployments
- ⚠️ Rollback requirements
- 🔒 Security vulnerabilities

## Troubleshooting

### Common Issues

**1. Workflow fails with "Secret not found"**
- Check that all required secrets are set in GitHub repository settings
- Verify secret names match exactly (case-sensitive)

**2. Railway deployment fails**
- Verify Railway project IDs are correct
- Check that Railway token is valid: `railway whoami`
- Ensure environment variables are set in Railway project

**3. Tests fail in CI but pass locally**
- Check Python version (3.11)
- Verify Redis is available in CI
- Review environment variable configuration

**4. Docker build fails**
- Test build locally: `docker build -f Dockerfile.optimized -t test .`
- Check all dependencies are in requirements.txt
- Verify Dockerfile syntax

**5. Coverage below 80%**
- Run locally: `pytest --cov=app --cov-report=html`
- Open `htmlcov/index.html` to see uncovered lines
- Add tests for uncovered code

### Get Help

- Check workflow logs: `gh run view --log`
- Review Railway logs: `railway logs`
- Check CI/CD documentation: `/home/user/learning_voice_agent/docs/CI_CD.md`
- GitHub Actions docs: https://docs.github.com/en/actions
- Railway docs: https://docs.railway.app

## Best Practices

### Branch Strategy
```
main (protected)
  ↑
develop (auto-deploy to staging)
  ↑
feature/* (run tests + lint)
```

### Release Process
1. Merge features to `develop`
2. Test in staging environment
3. Create release from `main`
4. Tag with semantic version (v1.0.0)
5. GitHub Actions auto-deploys to production

### Security
- Rotate API keys quarterly
- Use different keys for staging/production
- Review security scan results weekly
- Update dependencies regularly

### Performance
- Keep test suite under 5 minutes
- Monitor deployment times
- Review coverage trends
- Track failed deployment rate

## Next Steps

1. ✅ Set up all GitHub secrets
2. ✅ Create Railway projects
3. ✅ Test workflows
4. ✅ Configure Slack notifications
5. ✅ Set up Codecov integration
6. ✅ Install pre-commit hooks
7. ✅ Document custom workflows
8. ✅ Train team on CI/CD process

## Maintenance

### Weekly Tasks
- Review security scan results
- Check coverage trends
- Monitor deployment success rate
- Update dependencies

### Monthly Tasks
- Rotate API keys
- Review workflow performance
- Update documentation
- Audit access permissions

### Quarterly Tasks
- Review and optimize workflows
- Update CI/CD tools
- Conduct disaster recovery test
- Review and update rollback procedures

---

**Need Help?** Contact the DevOps team or open an issue in the repository.

**Last Updated:** 2025-11-21
