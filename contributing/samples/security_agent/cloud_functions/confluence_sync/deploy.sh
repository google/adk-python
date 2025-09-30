#!/bin/bash

# Deployment script for Confluence to BigQuery sync Cloud Function
# Usage: ./deploy.sh [project-id] [region]

set -e

# Configuration
PROJECT_ID=${1:-"mgm-digitalconcierge"}
REGION=${2:-"us-central1"}
FUNCTION_NAME="confluence-bigquery-sync"
DATASET_ID="security_data"
SERVICE_ACCOUNT="confluence-sync-sa"
TOPIC_NAME="confluence-sync-schedule"
SCHEDULER_JOB="confluence-sync-job"

echo "🚀 Deploying Confluence to BigQuery sync function..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    pubsub.googleapis.com \
    bigquery.googleapis.com \
    secretmanager.googleapis.com \
    cloudrun.googleapis.com

# Create service account if not exists
echo "👤 Setting up service account..."
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
        --display-name="Confluence BigQuery Sync Service Account"
fi

# Grant necessary permissions
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Create secrets if they don't exist
echo "🔐 Setting up secrets..."
create_secret_if_not_exists() {
    local SECRET_NAME=$1
    local SECRET_VALUE=$2

    if ! gcloud secrets describe ${SECRET_NAME} &>/dev/null; then
        echo "Creating secret: ${SECRET_NAME}"
        echo -n "${SECRET_VALUE}" | gcloud secrets create ${SECRET_NAME} \
            --data-file=- \
            --replication-policy="automatic"
    else
        echo "Secret ${SECRET_NAME} already exists"
    fi
}

# Prompt for Confluence credentials if not in environment
if [ -z "$CONFLUENCE_URL" ]; then
    read -p "Enter Confluence URL (e.g., https://yourcompany.atlassian.net): " CONFLUENCE_URL
fi

if [ -z "$CONFLUENCE_USERNAME" ]; then
    read -p "Enter Confluence username/email: " CONFLUENCE_USERNAME
fi

if [ -z "$CONFLUENCE_API_TOKEN" ]; then
    read -sp "Enter Confluence API token: " CONFLUENCE_API_TOKEN
    echo
fi

# Create secrets
create_secret_if_not_exists "confluence-url" "$CONFLUENCE_URL"
create_secret_if_not_exists "confluence-username" "$CONFLUENCE_USERNAME"
create_secret_if_not_exists "confluence-api-token" "$CONFLUENCE_API_TOKEN"

# Create Pub/Sub topic for scheduling
echo "📬 Setting up Pub/Sub topic..."
if ! gcloud pubsub topics describe ${TOPIC_NAME} &>/dev/null; then
    gcloud pubsub topics create ${TOPIC_NAME}
fi

# Deploy the HTTP-triggered function
echo "☁️ Deploying Cloud Function (HTTP trigger)..."
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=sync_confluence_to_bigquery \
    --trigger-http \
    --allow-unauthenticated \
    --service-account=${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},BQ_DATASET_ID=${DATASET_ID},CONFLUENCE_SPACES=SEC,POLICY,GCP" \
    --timeout=540s \
    --memory=512MB \
    --max-instances=10

# Deploy the scheduled function (Pub/Sub trigger)
echo "⏰ Deploying scheduled Cloud Function..."
gcloud functions deploy ${FUNCTION_NAME}-scheduled \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=sync_confluence_scheduled \
    --trigger-topic=${TOPIC_NAME} \
    --service-account=${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},BQ_DATASET_ID=${DATASET_ID},CONFLUENCE_SPACES=SEC,POLICY,GCP" \
    --timeout=540s \
    --memory=512MB \
    --max-instances=2

# Create Cloud Scheduler job for daily sync
echo "⏱️ Setting up Cloud Scheduler..."
if gcloud scheduler jobs describe ${SCHEDULER_JOB} --location=${REGION} &>/dev/null; then
    echo "Updating existing scheduler job..."
    gcloud scheduler jobs update pubsub ${SCHEDULER_JOB} \
        --location=${REGION} \
        --schedule="0 2 * * *" \
        --topic=${TOPIC_NAME} \
        --message-body='{"sync_type":"incremental","spaces":["SEC","POLICY","GCP"]}' \
        --time-zone="America/Los_Angeles"
else
    echo "Creating new scheduler job..."
    gcloud scheduler jobs create pubsub ${SCHEDULER_JOB} \
        --location=${REGION} \
        --schedule="0 2 * * *" \
        --topic=${TOPIC_NAME} \
        --message-body='{"sync_type":"incremental","spaces":["SEC","POLICY","GCP"]}' \
        --time-zone="America/Los_Angeles" \
        --description="Daily incremental sync of Confluence to BigQuery"
fi

# Get the function URL
FUNCTION_URL=$(gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --format="value(serviceConfig.uri)")

echo "✅ Deployment complete!"
echo ""
echo "📌 Function Details:"
echo "   Name: ${FUNCTION_NAME}"
echo "   URL: ${FUNCTION_URL}"
echo "   Region: ${REGION}"
echo "   Schedule: Daily at 2 AM PST"
echo ""
echo "🧪 To test the function:"
echo "   curl -X POST ${FUNCTION_URL} \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"sync_type\":\"incremental\",\"spaces\":[\"SEC\"]}'"
echo ""
echo "📊 To query the data in BigQuery:"
echo "   bq query --use_legacy_sql=false \\"
echo "     'SELECT title, space_key, modified_date FROM ${PROJECT_ID}.${DATASET_ID}.confluence_documents LIMIT 10'"
echo ""
echo "📝 To trigger a manual sync:"
echo "   gcloud scheduler jobs run ${SCHEDULER_JOB} --location=${REGION}"