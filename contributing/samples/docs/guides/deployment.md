# Cloud Run Deployment Guide with Service Account Credentials

This guide walks you through deploying the ADK Security Agent to Google Cloud Run with proper service account credentials for security scanning.

## Overview

The deployment process involves:
1. Setting up service account credentials in Secret Manager
2. Building and deploying the Docker container
3. Configuring Cloud Run to access the credentials
4. Testing the deployment

## Prerequisites

- Google Cloud CLI installed and authenticated
- Docker installed (or use Cloud Build)
- Necessary IAM permissions:
  - Secret Manager Admin
  - Cloud Run Admin
  - Service Account Admin
  - Cloud Build Editor

## Step 1: Store Service Account Key in Secret Manager

First, store your local service account key (the one with security scanning permissions) in Google Secret Manager:

```bash
# Create the secret with your local service account key
gcloud secrets create security-agent-sa-key \
    --data-file=path/to/your-service-account-key.json

# Verify the secret was created
gcloud secrets list --filter="name:security-agent-sa-key"
```

## Step 2: Grant Cloud Run Service Account Access

Grant the Cloud Run service account permission to access the secret:

```bash
# Grant secret access to the Cloud Run service account
gcloud secrets add-iam-policy-binding security-agent-sa-key \
    --member="serviceAccount:security-agent@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

Replace `YOUR-PROJECT-ID` with your actual project ID.

## Step 3: Build and Deploy with Cloud Build

From the project root directory:

```bash
# Trigger the Cloud Build deployment
gcloud builds submit \
    --config=deploy/cloudbuild.yaml \
    --substitutions=_SERVICE_NAME=security-agent \
    contributing/samples/security_agent/
```

## Step 4: Alternative Manual Deployment

If you prefer to deploy manually:

```bash
# Build the Docker image
cd contributing/samples/security_agent
docker build -t gcr.io/YOUR-PROJECT-ID/security-agent .

# Push to Google Container Registry
docker push gcr.io/YOUR-PROJECT-ID/security-agent

# Deploy to Cloud Run
gcloud run deploy security-agent \
    --image gcr.io/YOUR-PROJECT-ID/security-agent \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=YOUR-PROJECT-ID" \
    --service-account "security-agent@YOUR-PROJECT-ID.iam.gserviceaccount.com"
```

## Step 5: Test the Deployment

Once deployed, test that the service can access GCP resources:

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe security-agent \
    --region us-central1 \
    --format 'value(status.url)')

# Test the health endpoint
curl $SERVICE_URL/health

# Test the projects endpoint (requires proper credentials)
curl $SERVICE_URL/api/v1/gcp/projects

# Test security analysis (requires scanning permissions)
curl -X POST $SERVICE_URL/api/v1/agent/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "analyze my security posture", "project_id": "YOUR-PROJECT-ID"}'
```

## How the Secret Manager Integration Works

1. **Startup**: When the Cloud Run service starts, the `startup_event()` function is called
2. **Detection**: The service detects it's running in Cloud Run via the `K_SERVICE` environment variable
3. **Fetch**: It uses the Cloud Run service account to fetch the secret from Secret Manager
4. **Setup**: The service account key is written to a temporary file and `GOOGLE_APPLICATION_CREDENTIALS` is set
5. **Usage**: All subsequent GCP API calls use the retrieved service account credentials

## Code Flow

```python
@app.on_event("startup")
async def startup_event():
    """Initialize service account credentials from Secret Manager on startup."""
    setup_service_account_from_secret()

def setup_service_account_from_secret():
    # Only fetch from Secret Manager if running in Cloud Run
    if not os.getenv('K_SERVICE'):
        return  # Use local credentials
        
    # Fetch secret and setup credentials
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(...)
    # ... setup temporary credentials file
```

## Troubleshooting

### Permission Errors

If you get permission errors:

```bash
# Check that Secret Manager API is enabled
gcloud services enable secretmanager.googleapis.com

# Verify the service account has the right permissions
gcloud secrets get-iam-policy security-agent-sa-key

# Verify the secret exists
gcloud secrets versions list security-agent-sa-key
```

### Service Account Issues

If the service account doesn't have scanning permissions:

```bash
# Grant necessary roles to your original service account
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:original-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/cloudasset.viewer"

gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:original-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/securitycenter.adminViewer"

gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:original-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/iam.securityReviewer"
```

### Local Development

For local development, the service will detect it's not in Cloud Run and use your local credentials:

```bash
# Set up local development
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account-key.json
python run_backend.py
```

## Security Considerations

- Service account keys are stored securely in Google Secret Manager
- Keys are only accessible to the Cloud Run service account
- Temporary files are created with secure permissions
- No credentials are logged or exposed in the application

## Next Steps

After successful deployment:

1. Test the security analysis features
2. Configure monitoring and alerting
3. Set up CI/CD pipeline for updates
4. Configure custom domain if needed

## Support

If you encounter issues:

1. Check Cloud Run logs: `gcloud logging read "resource.type=cloud_run_revision"`
2. Verify service account permissions
3. Test Secret Manager access
4. Review the application startup logs