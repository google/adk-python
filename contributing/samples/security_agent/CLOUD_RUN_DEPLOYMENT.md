# Cloud Run Deployment Guide

## Overview

This is a lightweight Flask API that serves as a gateway to Cloud Functions for the GCP Security Agent. The architecture is optimized for serverless deployment with minimal resource usage.

## Architecture

```
Cloud Run (API Gateway)
    ├── /health - Health check
    ├── /api/firewall/rules → Cloud Function
    ├── /api/iam/service-accounts → Cloud Function
    └── /api/security/analyze → Aggregates multiple functions
```

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed and configured
3. **Docker** (for local builds)
4. **APIs Enabled**:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API
   - Cloud Functions API

## Quick Start

### 1. Set Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_REGION=us-central1  # or your preferred region
```

### 2. Deploy

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

Choose option 1 for Cloud Build (recommended) or option 2 for local build.

## Manual Deployment

### Build Container

```bash
# Build locally
docker build -t gcr.io/${GOOGLE_CLOUD_PROJECT}/security-agent-api:latest .

# Push to Container Registry
docker push gcr.io/${GOOGLE_CLOUD_PROJECT}/security-agent-api:latest
```

### Deploy to Cloud Run

```bash
gcloud run deploy security-agent-api \
    --image gcr.io/${GOOGLE_CLOUD_PROJECT}/security-agent-api:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 256Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10
```

## Configuration

### Service Account

The deployment creates a service account `security-agent-api-sa` with minimal permissions:
- `roles/cloudfunctions.invoker` - To call Cloud Functions
- `roles/logging.logWriter` - To write logs

### Environment Variables

Set in Cloud Run:
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `GOOGLE_CLOUD_REGION` - Deployment region

### Resource Limits

Optimized for lightweight operation:
- **Memory**: 256Mi (minimal Flask app)
- **CPU**: 1 vCPU
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10
- **Concurrency**: 100 requests per instance
- **Timeout**: 60 seconds

## Testing

### Health Check

```bash
SERVICE_URL=$(gcloud run services describe security-agent-api --region us-central1 --format 'value(status.url)')
curl ${SERVICE_URL}/health
```

### API Endpoints

```bash
# Get firewall rules
curl ${SERVICE_URL}/api/firewall/rules

# Get service account roles
curl ${SERVICE_URL}/api/iam/service-accounts

# Analyze security
curl -X POST ${SERVICE_URL}/api/security/analyze \
    -H "Content-Type: application/json" \
    -d '{"resource_type": "all"}'
```

## Monitoring

### View Logs

```bash
gcloud run services logs read security-agent-api --region us-central1
```

### View Metrics

Visit the [Cloud Run Console](https://console.cloud.google.com/run) to view:
- Request count
- Latency
- Error rate
- Container CPU/Memory usage

## CI/CD with Cloud Build

### Automatic Deployment

The `cloudbuild.yaml` file enables automatic deployment on git push:

```bash
# Submit build manually
gcloud builds submit --config=cloudbuild.yaml

# Or set up a trigger
gcloud builds triggers create github \
    --repo-name=security-agent \
    --repo-owner=your-github-org \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml
```

## Cost Optimization

This deployment is optimized for cost:

1. **Scales to Zero**: No charges when not in use
2. **Minimal Resources**: 256Mi memory keeps costs low
3. **Request-based Billing**: Pay only for actual usage
4. **No Always-On Instances**: Min instances set to 0

Estimated monthly cost for light usage (< 1000 requests/day): **< $5**

## Troubleshooting

### Common Issues

1. **502 Bad Gateway**
   - Check Cloud Function URLs in `app.py`
   - Verify service account permissions

2. **403 Forbidden**
   - Enable `--allow-unauthenticated` flag
   - Check IAM permissions

3. **Cloud Functions Not Found**
   - Deploy Cloud Functions first
   - Update URLs in `app.py`

4. **Build Fails**
   - Check Docker is running
   - Verify gcloud authentication
   - Enable required APIs

### Debug Commands

```bash
# Check service status
gcloud run services describe security-agent-api --region us-central1

# View recent logs
gcloud run services logs read security-agent-api --region us-central1 --limit=50

# Test locally
docker build -t test-api .
docker run -p 8080:8080 -e PORT=8080 test-api
```

## Cleanup

To remove all resources:

```bash
# Delete Cloud Run service
gcloud run services delete security-agent-api --region us-central1

# Delete container images
gcloud container images delete gcr.io/${GOOGLE_CLOUD_PROJECT}/security-agent-api

# Delete service account
gcloud iam service-accounts delete security-agent-api-sa@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com
```

## Next Steps

1. **Add Authentication**: Implement OAuth2 or API keys
2. **Custom Domain**: Map a custom domain to the service
3. **Monitoring**: Set up alerts and dashboards
4. **Caching**: Add Redis for response caching
5. **API Gateway**: Consider Apigee for advanced API management

## Support

For issues or questions:
- Check logs: `gcloud run services logs read`
- Review [Cloud Run documentation](https://cloud.google.com/run/docs)
- File issues in the project repository