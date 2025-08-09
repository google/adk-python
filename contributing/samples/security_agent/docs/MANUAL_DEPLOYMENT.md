# Manual Deployment Guide (No Cloud Build Required)

Since you're encountering Cloud Build permissions issues, here are alternative deployment methods that don't require additional IAM permissions.

## Method 1: Direct Docker Build and Deploy

This method builds the Docker image locally and pushes it directly to Google Container Registry.

### Step 1: Enable Required APIs

First, ensure the required APIs are enabled (if you have permission):

```bash
# Try to enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
# If you get permission errors, ask your admin to enable these APIs
```

### Step 2: Build and Push Docker Image Locally

From the security_agent directory:

```bash
# Navigate to the project directory
cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent

# Build the Docker image locally
docker build -t gcr.io/mgm-digitalconcierge/security-agent:latest .

# Configure Docker to use gcloud credentials
gcloud auth configure-docker

# Push the image to Google Container Registry
docker push gcr.io/mgm-digitalconcierge/security-agent:latest
```

### Step 3: Deploy to Cloud Run

```bash
# Deploy to Cloud Run directly
gcloud run deploy security-agent \
    --image gcr.io/mgm-digitalconcierge/security-agent:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge" \
    --service-account "security-agent@mgm-digitalconcierge.iam.gserviceaccount.com"
```

## Method 2: Use Cloud Shell (Recommended if you have access)

If you have access to Google Cloud Shell, you can use it to build and deploy:

### Step 1: Upload Code to Cloud Shell

1. Open Google Cloud Shell in the Console
2. Upload your security_agent directory
3. Run the deployment commands from Cloud Shell

### Cloud Shell Commands:

```bash
# In Cloud Shell, navigate to uploaded directory
cd security_agent

# Build the image
docker build -t gcr.io/mgm-digitalconcierge/security-agent:latest .

# Push to registry
docker push gcr.io/mgm-digitalconcierge/security-agent:latest

# Deploy to Cloud Run
gcloud run deploy security-agent \
    --image gcr.io/mgm-digitalconcierge/security-agent:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge" \
    --service-account "security-agent@mgm-digitalconcierge.iam.gserviceaccount.com"
```

## Method 3: Service Account Setup for Secret Manager

Since you don't have permission to create secrets, here's an alternative approach:

### Option A: Ask Admin to Set Up Secret

Ask your project admin to run these commands:

```bash
# Admin creates the secret with your service account key
gcloud secrets create security-agent-sa-key \
    --data-file=path/to/your-service-account-key.json

# Admin grants access to the Cloud Run service account
gcloud secrets add-iam-policy-binding security-agent-sa-key \
    --member="serviceAccount:security-agent@mgm-digitalconcierge.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### Option B: Use Environment Variables (Less Secure)

As a temporary workaround, you can pass the service account key as a base64-encoded environment variable:

```bash
# Encode your service account key
SERVICE_ACCOUNT_KEY=$(base64 -i path/to/your-service-account-key.json)

# Deploy with the key as an environment variable
gcloud run deploy security-agent \
    --image gcr.io/mgm-digitalconcierge/security-agent:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge,SERVICE_ACCOUNT_KEY_BASE64=$SERVICE_ACCOUNT_KEY" \
    --service-account "security-agent@mgm-digitalconcierge.iam.gserviceaccount.com"
```

Then modify the startup function to handle this:

```python
def setup_service_account_from_secret():
    """Setup service account credentials from Secret Manager or environment variable."""
    # Check for base64 encoded key in environment variable (fallback)
    key_base64 = os.getenv('SERVICE_ACCOUNT_KEY_BASE64')
    if key_base64 and not os.getenv('K_SERVICE'):
        import base64
        key_data = base64.b64decode(key_base64).decode('utf-8')
        temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='sa_key_')
        with os.fdopen(temp_fd, 'w') as temp_file:
            temp_file.write(key_data)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
        logging.info("✅ Service account credentials loaded from environment variable")
        return
    
    # Original Secret Manager logic...
```

## Method 4: Test Local Build First

Test that your Docker build works locally:

```bash
# Build locally
docker build -t security-agent-local .

# Run locally to test
docker run -p 8000:8000 \
    -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json \
    -v /path/to/your-service-account-key.json:/path/to/your-service-account-key.json:ro \
    security-agent-local
```

## Troubleshooting

### If Docker Push Fails

```bash
# Make sure you're authenticated
gcloud auth configure-docker

# Check if Container Registry API is enabled
gcloud services list --enabled --filter="name:containerregistry.googleapis.com"
```

### If Cloud Run Deploy Fails

```bash
# Check if Cloud Run API is enabled
gcloud services list --enabled --filter="name:run.googleapis.com"

# Check your current permissions
gcloud auth list
gcloud config list
```

## Next Steps

1. Try Method 1 (Direct Docker build and deploy) first
2. If that fails due to API permissions, use Cloud Shell (Method 2)
3. For the service account secret, either ask your admin or use the environment variable approach temporarily
4. Once deployed, test the service endpoints

## Testing After Deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe security-agent \
    --region us-central1 \
    --format 'value(status.url)')

echo "Service deployed at: $SERVICE_URL"

# Test endpoints
curl $SERVICE_URL/health
curl $SERVICE_URL/api/v1/gcp/projects
```

This approach should work around the Cloud Build permissions issue and get your service deployed successfully.