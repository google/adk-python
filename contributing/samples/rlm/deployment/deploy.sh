#!/bin/bash
# Deploy ADK-RLM to Cloud Run with IAP (using Cloud Run's native IAP support)

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-adk-rlm}"

echo "=== ADK-RLM Cloud Run Deployment ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Building and deploying from: $(pwd)"
echo ""

# Submit build to Cloud Build
echo "Submitting build to Cloud Build..."
gcloud builds submit \
    --config=deployment/cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_SERVICE_NAME=${SERVICE_NAME}" \
    --project="$PROJECT_ID" \
    .

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format="value(status.url)")

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Cloud Run service deployed with:"
echo "  - Minimum instances: 1"
echo "  - IAP enabled (requires Google authentication)"
echo "  - Public IAM access disabled"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "To grant access to users, run:"
echo "  gcloud beta run services add-iam-policy-binding $SERVICE_NAME \\"
echo "      --region=$REGION \\"
echo "      --member='user:email@example.com' \\"
echo "      --role='roles/run.invoker' \\"
echo "      --project=$PROJECT_ID"
