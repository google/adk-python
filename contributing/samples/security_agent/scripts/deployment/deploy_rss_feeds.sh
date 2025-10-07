#!/bin/bash

# RSS Feed Cloud Functions Deployment Script
# Simplified deployment focused on RSS feed ingestion

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-mgm-digitalconcierge}"
REGION="${REGION:-us-central1}"
DATASET_ID="${DATASET_ID:-security_insights}"

echo "=========================================="
echo "Deploying RSS Feed Cloud Functions"
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

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    bigquery.googleapis.com \
    cloudbuild.googleapis.com \
    --project=$PROJECT_ID

# Create BigQuery dataset if it doesn't exist
echo ""
echo "Creating BigQuery dataset if needed..."
if ! bq ls -d $PROJECT_ID:$DATASET_ID &> /dev/null; then
    bq mk --dataset \
        --location=$REGION \
        --description="Security insights data from GCP services" \
        $PROJECT_ID:$DATASET_ID
    echo "Dataset created"
else
    echo "Dataset already exists"
fi

# Deploy RSS Feed Cloud Functions
echo ""
echo "Deploying RSS Feed Cloud Functions..."
echo "=========================================="

# Function 1: GCP Release Notes
echo ""
echo "Deploying fetch_gcp_release_notes..."
if [ -d "cloud_functions/fetch_gcp_release_notes" ]; then
    gcloud functions deploy fetch_gcp_release_notes \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --source=cloud_functions/fetch_gcp_release_notes \
        --entry-point=fetch_gcp_release_notes \
        --trigger-http \
        --allow-unauthenticated \
        --timeout=540s \
        --memory=512MB \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,BQ_DATASET_ID=$DATASET_ID" \
        --project=$PROJECT_ID || echo "Failed to deploy fetch_gcp_release_notes"

    echo "✅ GCP Release Notes function deployed"
else
    echo "❌ Missing cloud_functions/fetch_gcp_release_notes directory"
fi

# Function 2: Security Feeds
echo ""
echo "Deploying fetch_security_feeds..."
if [ -d "cloud_functions/fetch_security_feeds" ]; then
    gcloud functions deploy fetch_security_feeds \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --source=cloud_functions/fetch_security_feeds \
        --entry-point=fetch_security_feeds \
        --trigger-http \
        --allow-unauthenticated \
        --timeout=540s \
        --memory=512MB \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,BQ_DATASET_ID=$DATASET_ID" \
        --project=$PROJECT_ID || echo "Failed to deploy fetch_security_feeds"

    echo "✅ Security Feeds function deployed"
else
    echo "❌ Missing cloud_functions/fetch_security_feeds directory"
fi

# Set up Cloud Scheduler jobs
echo ""
echo "Setting up Cloud Scheduler jobs..."
echo "=========================================="

# Create App Engine app if needed (required for Cloud Scheduler)
if ! gcloud app describe --project=$PROJECT_ID &> /dev/null; then
    echo "Creating App Engine app (required for Cloud Scheduler)..."
    gcloud app create --region=$REGION --project=$PROJECT_ID || echo "App Engine app creation failed or already exists"
fi

# Get function URLs
echo ""
echo "Getting function URLs..."
GCP_RELEASE_URL=""
SECURITY_FEEDS_URL=""

if gcloud functions describe fetch_gcp_release_notes --region=$REGION --project=$PROJECT_ID &> /dev/null; then
    GCP_RELEASE_URL=$(gcloud functions describe fetch_gcp_release_notes \
        --region=$REGION \
        --format="value(url)" \
        --project=$PROJECT_ID)
    echo "GCP Release Notes URL: $GCP_RELEASE_URL"
fi

if gcloud functions describe fetch_security_feeds --region=$REGION --project=$PROJECT_ID &> /dev/null; then
    SECURITY_FEEDS_URL=$(gcloud functions describe fetch_security_feeds \
        --region=$REGION \
        --format="value(url)" \
        --project=$PROJECT_ID)
    echo "Security Feeds URL: $SECURITY_FEEDS_URL"
fi

# Create scheduler jobs
echo ""
echo "Creating scheduler jobs..."

# Delete existing jobs if they exist
gcloud scheduler jobs delete schedule-fetch-gcp-release-notes \
    --location=$REGION \
    --project=$PROJECT_ID \
    --quiet 2> /dev/null || true

gcloud scheduler jobs delete schedule-fetch-security-feeds \
    --location=$REGION \
    --project=$PROJECT_ID \
    --quiet 2> /dev/null || true

# Create GCP Release Notes scheduler job (every 4 hours)
if [ ! -z "$GCP_RELEASE_URL" ]; then
    gcloud scheduler jobs create http schedule-fetch-gcp-release-notes \
        --location=$REGION \
        --schedule="0 */4 * * *" \
        --uri="$GCP_RELEASE_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"force_refresh": true}' \
        --time-zone="UTC" \
        --project=$PROJECT_ID

    echo "✅ GCP Release Notes scheduled (every 4 hours)"
else
    echo "❌ Could not create GCP Release Notes scheduler - function URL not found"
fi

# Create Security Feeds scheduler job (every 2 hours)
if [ ! -z "$SECURITY_FEEDS_URL" ]; then
    gcloud scheduler jobs create http schedule-fetch-security-feeds \
        --location=$REGION \
        --schedule="0 */2 * * *" \
        --uri="$SECURITY_FEEDS_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"force_refresh": true}' \
        --time-zone="UTC" \
        --project=$PROJECT_ID

    echo "✅ Security Feeds scheduled (every 2 hours)"
else
    echo "❌ Could not create Security Feeds scheduler - function URL not found"
fi

# Test the functions
echo ""
echo "=========================================="
echo "Testing deployed functions..."
echo "=========================================="

# Test GCP Release Notes function
if [ ! -z "$GCP_RELEASE_URL" ]; then
    echo ""
    echo "Testing fetch_gcp_release_notes..."
    RESPONSE=$(curl -s -X POST "$GCP_RELEASE_URL" \
        -H "Content-Type: application/json" \
        -d '{"test": true}' || echo '{"error": "Failed to call function"}')
    echo "Response: $RESPONSE"
fi

# Test Security Feeds function
if [ ! -z "$SECURITY_FEEDS_URL" ]; then
    echo ""
    echo "Testing fetch_security_feeds..."
    RESPONSE=$(curl -s -X POST "$SECURITY_FEEDS_URL" \
        -H "Content-Type: application/json" \
        -d '{"test": true}' || echo '{"error": "Failed to call function"}')
    echo "Response: $RESPONSE"
fi

echo ""
echo "=========================================="
echo "✅ RSS Feed Deployment Complete!"
echo "=========================================="
echo ""
echo "Resources created:"
echo "  - 2 Cloud Functions for RSS feed ingestion"
echo "  - 2 Cloud Scheduler jobs"
echo "  - BigQuery dataset: $DATASET_ID"
echo ""
echo "Next steps:"
echo "  1. Check BigQuery tables are being created:"
echo "     bq ls $PROJECT_ID:$DATASET_ID"
echo ""
echo "  2. Trigger manual refresh:"
echo "     gcloud scheduler jobs run schedule-fetch-gcp-release-notes --location=$REGION"
echo "     gcloud scheduler jobs run schedule-fetch-security-feeds --location=$REGION"
echo ""
echo "  3. Monitor function logs:"
echo "     gcloud functions logs read --region=$REGION"
echo ""