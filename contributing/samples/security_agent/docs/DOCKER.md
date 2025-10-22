# Docker Deployment Guide

Complete guide for deploying the GCP Security Intelligence Platform using Docker.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup Steps](#setup-steps)
- [Running with Docker Compose](#running-with-docker-compose)
- [Running with Docker Scripts](#running-with-docker-scripts)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

## 🚀 Quick Start

For users who want to get started immediately:

```bash
# Navigate to the security_agent directory
cd contributing/samples/security_agent

# Run preflight check
./scripts/docker_preflight.sh

# If preflight passes, build and run
docker compose up --build

# Access the interfaces
# - ADK Backend: http://localhost:8000
# - Flask UI: http://localhost:5001
# - Chainlit UI: http://localhost:8001
```

If the preflight check fails, follow the [Setup Steps](#setup-steps) below.

## ✅ Prerequisites

### System Requirements

- **Docker**: Version 20.10+ (includes Docker Compose v2)
- **Docker Compose**: Version 2.0+ or docker-compose v1.29+
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: 4GB+ RAM available for Docker
- **Disk Space**: 3GB+ free space

### GCP Requirements

- **GCP Project**: Active Google Cloud Platform project
- **Service Account**: JSON key with required IAM roles
- **APIs Enabled**:
  - BigQuery API (`bigquery.googleapis.com`)
  - Vertex AI API (`aiplatform.googleapis.com`)

### Required IAM Roles

Your service account needs these roles:

**Minimum (for BigQuery queries):**
- `roles/bigquery.dataViewer`
- `roles/bigquery.jobUser`
- `roles/aiplatform.user`

**Optional (for data collection):**
- `roles/compute.viewer`
- `roles/iam.securityReviewer`
- `roles/storage.objectViewer`

## 🔧 Setup Steps

### Step 1: Create Service Account

```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Create service account
gcloud iam service-accounts create security-agent-sa \
  --display-name="Security Agent Service Account" \
  --project=$PROJECT_ID

# Grant required permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Download JSON key
gcloud iam service-accounts keys create service-account-key.json \
  --iam-account=security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### Step 2: Prepare Configuration

```bash
# Navigate to project directory
cd contributing/samples/security_agent

# Create config directory
mkdir -p config

# Move service account key
mv service-account-key.json config/
chmod 600 config/service-account-key.json

# Copy environment template
cp .env.example .env
```

### Step 3: Configure Environment

Edit `.env` file with your details:

```bash
# Required: GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account-key.json
GOOGLE_CLOUD_LOCATION=us-central1

# Required: BigQuery Configuration
BQ_DEFAULT_DATASET=security_insights
BQ_DEFAULT_TABLE=security_findings

# Required: ADK Configuration
ADK_AGENT_MODEL=gemini-2.5-flash
GOOGLE_GENAI_USE_VERTEXAI=1
ADK_BASE_URL=http://localhost:8000

# Optional: Confluence Integration
CONFLUENCE_URL=
CONFLUENCE_USERNAME=
CONFLUENCE_API_TOKEN=
CONFLUENCE_SPACES=SEC,POLICY,GCP
```

### Step 4: Validate Setup

```bash
# Run preflight check
./scripts/docker_preflight.sh

# Expected output:
# ========================================
#   Docker Preflight Check
# ========================================
# [1/5] Checking config directory...
# ✓ config/ directory exists
# [2/5] Checking .env file...
# ✓ .env file exists
# ✓ All required variables configured
# [3/5] Checking service account credentials...
# ✓ Service account file exists: config/service-account-key.json
# ✓ File permissions are secure: 600
# [4/5] Checking Docker installation...
# ✓ Docker is installed: 24.0.7
# ✓ Docker daemon is running
# [5/5] Checking Docker Compose...
# ✓ Docker Compose is available: v2.23.0
# ========================================
# ✓ All checks passed!
```

## 🐳 Running with Docker Compose

Docker Compose is the recommended approach for local development.

### Start Services

```bash
# Build and start in foreground
docker compose up --build

# Build and start in background (detached mode)
docker compose up -d --build

# Start without rebuilding
docker compose up
```

### View Logs

```bash
# Follow all logs
docker compose logs -f

# Follow specific service
docker compose logs -f security-agent

# View last 100 lines
docker compose logs --tail=100
```

### Stop Services

```bash
# Stop containers (keeps volumes)
docker compose stop

# Stop and remove containers
docker compose down

# Stop, remove containers, and remove volumes
docker compose down -v
```

### Check Status

```bash
# View running containers
docker compose ps

# Check health status
docker compose ps
# Look for "(healthy)" status

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:5001
curl http://localhost:8001
```

## 🔨 Running with Docker Scripts

Alternative approach using build and run scripts.

### Build Container

```bash
# Build with default name
./scripts/docker_build.sh

# Build with custom name
./scripts/docker_build.sh my-security-agent

# Output:
# Building Docker image 'security-agent' from /path/to/security_agent
# [+] Building 45.2s (12/12) FINISHED
```

### Run Container

```bash
# Run with default name
./scripts/docker_run.sh

# Run with custom name
./scripts/docker_run.sh my-security-agent

# Container will:
# - Mount config/ as read-only
# - Mount logs/ as read-write
# - Expose ports 8000, 5001, 8001
# - Auto-remove on stop (--rm flag)
```

### Stop Container

```bash
# Find container ID
docker ps

# Stop container
docker stop <container-id>

# Or use Ctrl+C if running in foreground
```

## ⚙️ Configuration

### Environment Variables

All configuration is done via `.env` file. Required variables:

| Variable | Example | Description |
|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | `my-project-123` | Your GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | `config/service-account-key.json` | Path to service account JSON |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | GCP region for API calls |
| `BQ_DEFAULT_DATASET` | `security_insights` | BigQuery dataset name |
| `BQ_DEFAULT_TABLE` | `security_findings` | BigQuery table name |
| `ADK_AGENT_MODEL` | `gemini-2.5-flash` | Gemini model version |
| `GOOGLE_GENAI_USE_VERTEXAI` | `1` | Use Vertex AI (required) |
| `ADK_BASE_URL` | `http://localhost:8000` | ADK backend URL |

### Volume Mounts

The container uses two volume mounts:

1. **Config (read-only)**: `./config:/app/config:ro`
   - Service account credentials
   - Other configuration files
   - Mounted as read-only for security

2. **Logs (read-write)**: `./logs:/app/logs`
   - Application logs
   - Service logs (ADK, Flask, Chainlit)
   - Persistent across container restarts

### Port Mappings

Three services are exposed:

- **8000**: ADK Backend API
- **5001**: Flask Web UI
- **8001**: Chainlit Chat UI

To change ports, edit `docker-compose.yml`:

```yaml
ports:
  - "9000:8000"  # ADK on 9000
  - "9001:5001"  # Flask on 9001
  - "9002:8001"  # Chainlit on 9002
```

## 🔍 Troubleshooting

### Preflight Check Fails

**Issue**: `docker_preflight.sh` reports errors

**Solutions**:
```bash
# Missing .env file
cp .env.example .env
# Edit .env with your project details

# Missing config directory
mkdir -p config

# Missing service account
# Follow Step 1 in Setup Steps to create service account

# Docker not running
# Start Docker Desktop or run: sudo systemctl start docker
```

### Container Fails to Start

**Issue**: Container exits immediately after starting

**Diagnosis**:
```bash
# Check logs
docker compose logs

# Common issues:
# 1. Invalid credentials
# 2. Missing environment variables
# 3. Port already in use
```

**Solutions**:
```bash
# Verify credentials
./scripts/docker_preflight.sh

# Check if ports are in use
lsof -i :8000
lsof -i :5001
lsof -i :8001

# Kill conflicting processes or change ports
```

### Authentication Errors

**Issue**: "Could not authenticate with GCP"

**Diagnosis**:
```bash
# Check service account file exists
ls -la config/service-account-key.json

# Verify file is valid JSON
cat config/service-account-key.json | jq .

# Check permissions
stat -c "%a" config/service-account-key.json
# Should be 600 or 400
```

**Solutions**:
```bash
# Fix permissions
chmod 600 config/service-account-key.json

# Verify .env points to correct path
grep GOOGLE_APPLICATION_CREDENTIALS .env

# Test credentials outside Docker
gcloud auth activate-service-account \
  --key-file=config/service-account-key.json
```

### Network Issues

**Issue**: Cannot access services on localhost

**Solutions**:
```bash
# Check container is running
docker compose ps

# Check port mappings
docker compose port security-agent 8000

# Try 127.0.0.1 instead of localhost
curl http://127.0.0.1:8000/health

# Check firewall rules
# May need to allow Docker in firewall settings
```

### Performance Issues

**Issue**: Container is slow or unresponsive

**Solutions**:
```bash
# Check resource usage
docker stats

# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory (recommend 4GB+)

# Check logs for errors
docker compose logs -f

# Clear Docker cache
docker system prune -a
```

### Health Check Failing

**Issue**: Container shows as "unhealthy"

**Diagnosis**:
```bash
# Check health status
docker compose ps

# View health check logs
docker inspect <container-id> | jq '.[0].State.Health'

# Manual health check
docker exec <container-id> curl -f http://localhost:8000/health
```

**Solutions**:
```bash
# Increase start_period in docker-compose.yml
# Services may need more time to start
healthcheck:
  start_period: 60s  # Increase from 40s

# Check ADK backend logs
docker compose logs security-agent | grep -i error
```

## 🚀 Production Deployment

### Cloud Run Deployment

```bash
# Set variables
export PROJECT_ID=your-project-id
export REGION=us-central1
export IMAGE_NAME=security-agent

# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
gcloud run deploy $IMAGE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  --set-env-vars BQ_DEFAULT_DATASET=security_insights \
  --set-env-vars BQ_DEFAULT_TABLE=security_findings \
  --set-env-vars ADK_AGENT_MODEL=gemini-2.5-flash \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=1 \
  --allow-unauthenticated
```

### GKE Deployment

```bash
# Create Kubernetes secret for service account
kubectl create secret generic gcp-credentials \
  --from-file=key.json=config/service-account-key.json

# Create ConfigMap for environment
kubectl create configmap security-agent-config \
  --from-env-file=.env

# Deploy to GKE
kubectl apply -f kubernetes/deployment.yaml
```

### Security Best Practices

1. **Credentials**
   - Never commit service account keys to git
   - Rotate keys every 90 days
   - Use Workload Identity in GKE
   - Use Cloud Run service identity

2. **Network**
   - Use Cloud Run ingress controls
   - Enable Cloud Armor for DDoS protection
   - Use VPC connectors for private access

3. **Monitoring**
   - Enable Cloud Logging
   - Set up Cloud Monitoring alerts
   - Monitor container health checks
   - Track error rates and latency

## 📚 Additional Resources

- [Main README](../README.md) - Platform overview
- [Setup Guide](SETUP_AND_TROUBLESHOOTING.md) - Detailed setup instructions
- [config/README.md](../config/README.md) - Service account setup guide
- [Docker Documentation](https://docs.docker.com/) - Official Docker docs
- [Cloud Run Documentation](https://cloud.google.com/run/docs) - Deploy to Cloud Run

## 🆘 Getting Help

If you encounter issues:

1. Run `./scripts/docker_preflight.sh` for diagnostics
2. Check logs: `docker compose logs -f`
3. Verify configuration in `.env`
4. Review this troubleshooting guide
5. Check GitHub issues for similar problems

---

**Last Updated**: October 22, 2025
**Docker Version**: 20.10+
**Docker Compose Version**: 2.0+
