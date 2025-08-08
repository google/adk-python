# 🚀 ADK Deployment Guide

This directory contains deployment configurations and documentation for the Agent Development Kit (ADK).

## 📁 Contents

- **`Dockerfile`** - Docker container configuration for ADK applications
- **`cloudbuild.yaml`** - Google Cloud Build configuration for CI/CD
- **`README.md`** - This deployment guide

## 🐳 Docker Deployment

### Basic Docker Build

```bash
# Build ADK application image
docker build -f deploy/Dockerfile -t adk-app .

# Run the container
docker run -p 8000:8000 -p 8501:8501 adk-app
```

### Environment Variables

Set these environment variables for your deployment:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# ADK Configuration
ADK_LOG_LEVEL=INFO
ADK_MODEL_PROVIDER=vertex_ai

# Application Ports
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

## ☁️ Google Cloud Deployment

### Cloud Build

Use the included `cloudbuild.yaml` for automated builds:

```bash
# Submit build to Cloud Build
gcloud builds submit --config=deploy/cloudbuild.yaml .
```

### Cloud Run

Deploy to Google Cloud Run:

```bash
# Build and deploy in one step
gcloud run deploy adk-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🐙 Kubernetes Deployment

### Basic Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adk-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: adk-app
  template:
    metadata:
      labels:
        app: adk-app
    spec:
      containers:
      - name: adk-app
        image: gcr.io/your-project/adk-app:latest
        ports:
        - containerPort: 8000
        - containerPort: 8501
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
```

## 🔧 Production Considerations

### Security
- Use service accounts with minimal required permissions
- Store secrets in Google Secret Manager or Kubernetes secrets
- Enable HTTPS/TLS for production deployments
- Configure network security policies

### Scaling
- Use horizontal pod autoscaling in Kubernetes
- Configure Cloud Run concurrency limits
- Monitor resource usage and adjust limits

### Monitoring
- Enable Cloud Logging and Cloud Monitoring
- Set up health checks and alerting
- Configure distributed tracing

## 🛠️ Development vs Production

### Development
```bash
# Local development with hot reload
docker run -v $(pwd):/app -p 8000:8000 -p 8501:8501 adk-dev
```

### Production
```bash
# Optimized production image
docker run -d --restart=always -p 80:8000 -p 81:8501 adk-prod
```

## 📊 Health Checks

### Docker Health Check
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Kubernetes Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview and quick start
- [Security Agent](../contributing/samples/security_agent/README.md) - Sample application deployment
- [GCP API Explorer](../gcp_api_explorer/README.md) - API explorer deployment

---

For more deployment options and advanced configurations, see the specific application documentation in the `contributing/samples/` directory.