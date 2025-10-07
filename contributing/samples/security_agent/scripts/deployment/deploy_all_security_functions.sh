#!/bin/bash

# Deploy ALL Security Cloud Functions
# Complete deployment of security data ingestion infrastructure

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-mgm-digitalconcierge}"
REGION="${REGION:-us-central1}"
DATASET_ID="${DATASET_ID:-security_insights}"

echo "=========================================="
echo "🚀 Deploying ALL Security Cloud Functions"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Dataset: $DATASET_ID"
echo ""

# Check if gcloud is configured
if ! gcloud config get-value project &> /dev/null; then
    echo "Setting up gcloud project..."
    gcloud config set project $PROJECT_ID
fi

# Array of all functions to deploy
FUNCTIONS=(
    "fetch_compute_instances"
    "fetch_firewall_rules"
    "fetch_iam_accounts"
    "fetch_security_findings"
    "fetch_storage_buckets"
)

# Deploy each function
for FUNCTION_NAME in "${FUNCTIONS[@]}"; do
    echo ""
    echo "=========================================="
    echo "📦 Deploying: $FUNCTION_NAME"
    echo "=========================================="

    # Check if directory exists
    if [ ! -d "cloud_functions/$FUNCTION_NAME" ]; then
        echo "❌ Missing cloud_functions/$FUNCTION_NAME directory"
        continue
    fi

    # Check if requirements.txt exists, if not create a basic one
    if [ ! -f "cloud_functions/$FUNCTION_NAME/requirements.txt" ]; then
        echo "Creating requirements.txt for $FUNCTION_NAME..."
        cat > "cloud_functions/$FUNCTION_NAME/requirements.txt" << EOF
google-cloud-bigquery>=3.11.0
google-cloud-compute>=1.13.0
google-cloud-iam>=2.12.0
google-cloud-storage>=2.10.0
google-cloud-securitycenter>=1.23.0
google-cloud-resource-manager>=1.10.0
requests>=2.31.0
EOF
    fi

    # Deploy the Cloud Function
    echo "Deploying $FUNCTION_NAME..."
    gcloud functions deploy $FUNCTION_NAME \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --source=cloud_functions/$FUNCTION_NAME \
        --entry-point=$FUNCTION_NAME \
        --trigger-http \
        --allow-unauthenticated \
        --timeout=540s \
        --memory=512MB \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,BQ_DATASET_ID=$DATASET_ID" \
        --project=$PROJECT_ID \
        && echo "✅ $FUNCTION_NAME deployed successfully" \
        || echo "❌ Failed to deploy $FUNCTION_NAME"
done

# Verify all deployments
echo ""
echo "=========================================="
echo "🔍 Verifying Deployments"
echo "=========================================="

echo ""
echo "Deployed Cloud Functions:"
gcloud functions list --regions=$REGION --project=$PROJECT_ID --format="table(name,state)" | grep -E "fetch_|NAME"

# Count deployed functions
DEPLOYED_COUNT=$(gcloud functions list --regions=$REGION --project=$PROJECT_ID --format="value(name)" | grep -c "fetch_" || echo "0")
echo ""
echo "📊 Deployment Summary:"
echo "   Total expected: 7 functions"
echo "   Currently deployed: $DEPLOYED_COUNT functions"

if [ "$DEPLOYED_COUNT" -eq "7" ]; then
    echo "   ✅ All functions deployed successfully!"
else
    echo "   ⚠️ Some functions may be missing"
fi

# Test each function
echo ""
echo "=========================================="
echo "🧪 Testing Deployed Functions"
echo "=========================================="

for FUNCTION_NAME in "${FUNCTIONS[@]}"; do
    echo ""
    echo "Testing $FUNCTION_NAME..."

    # Get function URL
    FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --format="value(url)" \
        --project=$PROJECT_ID 2>/dev/null || echo "")

    if [ -z "$FUNCTION_URL" ]; then
        echo "❌ $FUNCTION_NAME not found"
        continue
    fi

    # Test the function
    RESPONSE=$(curl -s -X POST "$FUNCTION_URL" \
        -H "Content-Type: application/json" \
        -d '{"test": true}' \
        --max-time 10 || echo '{"error": "timeout or failure"}')

    echo "Response: ${RESPONSE:0:100}..."

    if [[ "$RESPONSE" == *"error"* ]]; then
        echo "❌ $FUNCTION_NAME test failed"
    else
        echo "✅ $FUNCTION_NAME is working"
    fi
done

echo ""
echo "=========================================="
echo "✅ Deployment Script Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Set up Cloud Scheduler for automated refreshes"
echo "  2. Monitor function logs:"
echo "     gcloud functions logs read --region=$REGION"
echo ""
echo "  3. Check BigQuery tables:"
echo "     - compute_instances"
echo "     - firewall_rules"
echo "     - iam_accounts"
echo "     - security_findings"
echo "     - storage_buckets"
echo "     - gcp_release_notes (already deployed)"
echo "     - security_threat_feeds (already deployed)"
echo ""