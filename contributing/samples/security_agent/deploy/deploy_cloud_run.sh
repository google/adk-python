#!/bin/bash
# Deploy GCP Security Agent to Cloud Run

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"mgm-digitalconcierge"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME="gcp-security-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying GCP Security Agent to Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    --project=${PROJECT_ID}

# Build Docker image
echo "🔨 Building Docker image..."
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --project=${PROJECT_ID} \
    --timeout=20m \
    .

# Create secrets if they don't exist
echo "🔐 Managing secrets..."
if ! gcloud secrets describe service-account-key --project=${PROJECT_ID} >/dev/null 2>&1; then
    echo "Creating service account key secret..."
    gcloud secrets create service-account-key \
        --data-file=backend/config/secrets/mgm-digitalconcierge-52fed2a2dac3.json \
        --project=${PROJECT_ID}
fi

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 1 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-env-vars="VERTEX_AI_PROJECT_ID=${PROJECT_ID}" \
    --set-env-vars="VERTEX_AI_LOCATION=${REGION}" \
    --set-secrets="GOOGLE_APPLICATION_CREDENTIALS=service-account-key:latest" \
    --service-account="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --format 'value(status.url)')

echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "Test the deployment:"
echo "curl ${SERVICE_URL}/health"
echo "curl ${SERVICE_URL}/api/v1/asset-inventory/summary?project_id=${PROJECT_ID}"