# Deployment Guide - Enhanced UI v2.0

## Overview
This guide covers deploying the enhanced GCP Security Agent with new UI improvements, export functionality, and accessibility features.

## 🆕 New Features in v2.0
- 🛡️ Comprehensive error boundary with user-friendly messages
- 📱 Mobile-responsive design with optimized touch interactions
- ♿ Full accessibility support (ARIA labels, keyboard navigation)
- 📋 Export functionality (Markdown reports + JSON data)
- 🔄 Smart auto-refresh indicators with visual status
- 📝 Collapsible sidebar for better screen usage

## Deployment Options

The ADK Security Agent supports multiple deployment options:

1. **Local Development** - Run directly on your machine
2. **Docker Deployment** - Containerized deployment
3. **Google Cloud Run** - Serverless deployment on GCP
4. **Kubernetes** - Scalable container orchestration

## Local Development

### Quick Start

```bash
# Backend
python run_backend.py

# Frontend (new terminal)
python run_frontend.py
```

### Enhanced Development Configuration

Create a `.env` file with new UI features:
```env
# Core Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
BACKEND_URL=http://localhost:8000

# 🆕 Enhanced UI Settings
SIDEBAR_COLLAPSED=true                    # Default sidebar state
EXPORT_ENABLED=true                       # Enable export functionality
ACCESSIBILITY_MODE=enhanced               # Full accessibility features
MOBILE_OPTIMIZED=true                     # Mobile responsive design
ERROR_BOUNDARY_ENABLED=true               # Enhanced error handling
REFRESH_INDICATOR_ENABLED=true            # Visual refresh status

# Performance Settings
MAX_EXPORT_SIZE=50MB                      # Maximum export file size
CACHE_TIMEOUT=1800                        # Cache timeout in seconds
UI_REFRESH_INTERVAL=30000                 # UI refresh interval (ms)
FRONTEND_URL=http://localhost:8501
LOG_LEVEL=DEBUG
```

## Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Configuration

The `docker-compose.yml` provides:
- Backend service on port 8000
- Frontend service on port 8501
- Persistent volumes for cache and logs
- Health checks for automatic recovery
- Environment variable configuration

### Production Docker Settings

For production, update `docker-compose.yml`:

```yaml
services:
  backend:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Google Cloud Run Deployment

### Prerequisites

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### Deploy Backend

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/security-agent-backend

# Deploy to Cloud Run
gcloud run deploy security-agent-backend \
  --image gcr.io/YOUR_PROJECT_ID/security-agent-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10
```

### Deploy Frontend

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/security-agent-frontend \
  --file Dockerfile.frontend

# Deploy to Cloud Run
gcloud run deploy security-agent-frontend \
  --image gcr.io/YOUR_PROJECT_ID/security-agent-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars BACKEND_URL=https://security-agent-backend-xxxxx.run.app \
  --memory 1Gi \
  --cpu 1
```

### Configure Service Account

```bash
# Create service account
gcloud iam service-accounts create security-agent-sa \
  --display-name="Security Agent Service Account"

# Grant necessary roles
for role in \
  roles/cloudasset.viewer \
  roles/securitycenter.adminViewer \
  roles/storage.admin \
  roles/iam.securityReviewer \
  roles/recommender.viewer \
  roles/secretmanager.viewer \
  roles/monitoring.viewer
do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="$role"
done

# Update Cloud Run service
gcloud run services update security-agent-backend \
  --service-account=security-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Kubernetes Deployment

### Create Namespace

```bash
kubectl create namespace security-agent
```

### Deploy with Helm

```bash
# Add Helm repository
helm repo add security-agent https://your-repo.com/charts

# Install
helm install security-agent security-agent/security-agent \
  --namespace security-agent \
  --set backend.image.tag=latest \
  --set frontend.image.tag=latest \
  --set gcp.projectId=YOUR_PROJECT_ID
```

### Manual Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-agent-backend
  namespace: security-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: security-agent-backend
  template:
    metadata:
      labels:
        app: security-agent-backend
    spec:
      containers:
      - name: backend
        image: gcr.io/YOUR_PROJECT_ID/security-agent-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: YOUR_PROJECT_ID
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: security-agent-backend
  namespace: security-agent
spec:
  selector:
    app: security-agent-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP Project ID | `my-project-123` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | `/app/credentials/key.json` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_PORT` | Backend server port | `8000` |
| `FRONTEND_PORT` | Frontend server port | `8501` |
| `DATABASE_PATH` | SQLite database path | `backend/cache/gcp_data.db` |
| `DATA_REFRESH_INTERVAL` | Cache refresh interval (seconds) | `1800` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RATE_LIMIT_CHAT` | Chat requests per minute | `30` |
| `ENABLE_RATE_LIMITING` | Enable rate limiting | `true` |

## Health Monitoring

### Health Check Endpoints

- `/health` - Basic health check
- `/metrics` - Application metrics
- `/status` - Detailed status



## Security Considerations

### Production Checklist

- [ ] Use HTTPS/TLS for all endpoints
- [ ] Enable authentication (OAuth2/JWT)
- [ ] Implement rate limiting
- [ ] Use secrets management (not env files)
- [ ] Enable audit logging
- [ ] Set up monitoring and alerting
- [ ] Configure backup and recovery
- [ ] Implement network policies
- [ ] Use least-privilege service accounts
- [ ] Enable container scanning

### Secrets Management

For production, use Google Secret Manager:

```bash
# Create secret
echo -n "your-secret-value" | gcloud secrets create api-key --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding api-key \
  --member="serviceAccount:security-agent-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Use in Cloud Run
gcloud run services update security-agent-backend \
  --update-secrets=API_KEY=api-key:latest
```

## Scaling Configuration

### Horizontal Scaling

```yaml
# Cloud Run
gcloud run services update security-agent-backend \
  --min-instances=2 \
  --max-instances=100 \
  --concurrency=100

# Kubernetes HPA
kubectl autoscale deployment security-agent-backend \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

### Vertical Scaling

```yaml
# Cloud Run
gcloud run services update security-agent-backend \
  --memory=4Gi \
  --cpu=4

# Kubernetes
kubectl set resources deployment security-agent-backend \
  --requests=memory=2Gi,cpu=1 \
  --limits=memory=4Gi,cpu=2
```

## Backup and Recovery

### Database Backup

```bash
# Manual backup
sqlite3 backend/cache/gcp_data.db ".backup backup.db"

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
sqlite3 backend/cache/gcp_data.db ".backup $BACKUP_DIR/backup_$TIMESTAMP.db"

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.db" -mtime +7 -delete
```

### Disaster Recovery

1. **Regular Backups**: Schedule hourly database backups
2. **Multi-Region**: Deploy to multiple regions for redundancy
3. **Data Replication**: Use Cloud SQL for automatic replication
4. **Monitoring**: Set up alerts for service degradation
5. **Runbooks**: Document recovery procedures

## Troubleshooting Deployment

See [troubleshooting.md](troubleshooting.md) for common deployment issues and solutions.