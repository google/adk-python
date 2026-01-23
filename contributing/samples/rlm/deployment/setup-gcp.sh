#!/bin/bash
# Setup GCP project for ADK-RLM deployment
# This script creates a new project and enables required services

set -e

# Configuration - modify these as needed
PROJECT_ID="${PROJECT_ID:-adk-rlm-$(date +%s)}"
BILLING_ACCOUNT="${BILLING_ACCOUNT:-}"
REGION="${REGION:-us-central1}"

echo "=== ADK-RLM GCP Project Setup ==="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check if billing account is set
if [ -z "$BILLING_ACCOUNT" ]; then
    echo "Available billing accounts:"
    gcloud billing accounts list
    echo ""
    echo "Set BILLING_ACCOUNT environment variable and re-run:"
    echo "  export BILLING_ACCOUNT=<your-billing-account-id>"
    echo "  ./setup-gcp.sh"
    exit 1
fi

echo "Billing Account: $BILLING_ACCOUNT"
echo ""

# Create the project
echo "Creating project $PROJECT_ID..."
gcloud projects create "$PROJECT_ID" --name="ADK-RLM" || {
    echo "Project may already exist, continuing..."
}

# Link billing account
echo "Linking billing account..."
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"

# Set as current project
echo "Setting current project..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    iap.googleapis.com \
    aiplatform.googleapis.com \
    --project="$PROJECT_ID"

# Create Artifact Registry repository for container images
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create adk-rlm \
    --repository-format=docker \
    --location="$REGION" \
    --description="ADK-RLM container images" \
    --project="$PROJECT_ID" || {
    echo "Repository may already exist, continuing..."
}

# Get project number for service accounts
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant permissions to Cloud Build service account
echo "Granting Cloud Build permissions..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/artifactregistry.writer"

# Grant permissions to default compute service account (used by Cloud Build)
echo "Granting default compute service account permissions..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/storage.objectViewer"

# Create service account for Cloud Run
echo "Creating Cloud Run service account..."
gcloud iam service-accounts create adk-rlm-runner \
    --display-name="ADK-RLM Cloud Run Service Account" \
    --project="$PROJECT_ID" || {
    echo "Service account may already exist, continuing..."
}

# Grant the service account permissions for Vertex AI
RUNNER_SA="adk-rlm-runner@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${RUNNER_SA}" \
    --role="roles/aiplatform.user"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""
echo "To deploy, run:"
echo "  export PROJECT_ID=$PROJECT_ID"
echo "  export REGION=$REGION"
echo "  ./deploy.sh"
echo ""
echo "Save these values in your environment or .env file."
