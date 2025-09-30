#!/bin/bash

# BigQuery Data Refresh Jobs Deployment Script
# Deploys Cloud Functions and sets up Cloud Scheduler for independent data refresh

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-mgm-digitalconcierge}"
REGION="${REGION:-us-central1}"
DATASET_ID="${DATASET_ID:-security_insights}"

echo "=========================================="
echo "Deploying BigQuery Refresh Architecture"
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
    compute.googleapis.com \
    iam.googleapis.com \
    bigquery.googleapis.com \
    pubsub.googleapis.com \
    cloudbuild.googleapis.com \
    --project=$PROJECT_ID

# Create BigQuery dataset if it doesn't exist
echo ""
echo "Creating BigQuery dataset if needed..."
bq mk --dataset \
    --location=$REGION \
    --description="Security insights data from GCP services" \
    $PROJECT_ID:$DATASET_ID || echo "Dataset already exists"

# Create metadata table for tracking refreshes
echo ""
echo "Creating refresh metadata table..."
bq mk --table \
    $PROJECT_ID:$DATASET_ID.refresh_metadata \
    table_name:STRING,refresh_time:TIMESTAMP,record_count:INTEGER,status:STRING,refresh_type:STRING,details:JSON,error_message:STRING \
    || echo "Metadata table already exists"

# Deploy Cloud Functions
echo ""
echo "Deploying Cloud Functions..."
echo "=========================================="

# Array of functions to deploy with their configurations
declare -A FUNCTIONS=(
    ["fetch_compute_instances"]="2h"
    ["fetch_iam_accounts"]="6h"
    ["fetch_firewall_rules"]="4h"
    ["fetch_storage_buckets"]="1h"
    ["fetch_gcp_release_notes"]="4h"
    ["fetch_security_feeds"]="2h"
)

for FUNCTION_NAME in "${!FUNCTIONS[@]}"; do
    SCHEDULE_HOURS="${FUNCTIONS[$FUNCTION_NAME]}"

    echo ""
    echo "Deploying $FUNCTION_NAME (runs every $SCHEDULE_HOURS)..."

    # Deploy the Cloud Function
    gcloud functions deploy $FUNCTION_NAME \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --source=cloud_functions/$FUNCTION_NAME \
        --entry-point=${FUNCTION_NAME} \
        --trigger-http \
        --allow-unauthenticated \
        --timeout=540s \
        --memory=512MB \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,BQ_DATASET_ID=$DATASET_ID" \
        --service-account=$PROJECT_ID@appspot.gserviceaccount.com \
        --project=$PROJECT_ID

    # Get the function URL
    FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --format="value(url)" \
        --project=$PROJECT_ID)

    echo "Function deployed at: $FUNCTION_URL"
done

# Create Cloud Scheduler jobs
echo ""
echo "Setting up Cloud Scheduler jobs..."
echo "=========================================="

# Create App Engine app if it doesn't exist (required for Cloud Scheduler)
if ! gcloud app describe --project=$PROJECT_ID &> /dev/null; then
    echo "Creating App Engine app (required for Cloud Scheduler)..."
    gcloud app create --region=$REGION --project=$PROJECT_ID || true
fi

# Schedule configurations
declare -A SCHEDULES=(
    ["fetch_compute_instances"]="0 */2 * * *"  # Every 2 hours
    ["fetch_iam_accounts"]="0 */6 * * *"       # Every 6 hours
    ["fetch_firewall_rules"]="0 */4 * * *"     # Every 4 hours
    ["fetch_storage_buckets"]="0 * * * *"      # Every hour
    ["fetch_gcp_release_notes"]="0 */4 * * *"  # Every 4 hours
    ["fetch_security_feeds"]="0 */2 * * *"     # Every 2 hours (critical security updates)
)

for FUNCTION_NAME in "${!SCHEDULES[@]}"; do
    SCHEDULE="${SCHEDULES[$FUNCTION_NAME]}"
    JOB_NAME="schedule-$FUNCTION_NAME"

    echo ""
    echo "Creating scheduler job: $JOB_NAME"

    # Delete existing job if it exists
    gcloud scheduler jobs delete $JOB_NAME \
        --location=$REGION \
        --project=$PROJECT_ID \
        --quiet 2> /dev/null || true

    # Get function URL
    FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --format="value(url)" \
        --project=$PROJECT_ID)

    # Create new scheduler job
    gcloud scheduler jobs create http $JOB_NAME \
        --location=$REGION \
        --schedule="$SCHEDULE" \
        --uri="$FUNCTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"force_refresh": true}' \
        --time-zone="UTC" \
        --project=$PROJECT_ID

    echo "Scheduled $FUNCTION_NAME with cron: $SCHEDULE"
done

# Create Pub/Sub topics for real-time updates
echo ""
echo "Setting up Pub/Sub for real-time updates..."
echo "=========================================="

# Create topics
gcloud pubsub topics create security-audit-logs \
    --project=$PROJECT_ID || echo "Topic already exists"

gcloud pubsub topics create security-alerts \
    --project=$PROJECT_ID || echo "Topic already exists"

# Create audit log sink for real-time updates
echo ""
echo "Creating audit log sink for real-time monitoring..."
gcloud logging sinks create security-audit-sink \
    pubsub.googleapis.com/projects/$PROJECT_ID/topics/security-audit-logs \
    --log-filter='
        (protoPayload.serviceName="compute.googleapis.com"
         OR protoPayload.serviceName="iam.googleapis.com"
         OR protoPayload.serviceName="storage.googleapis.com")
        AND (protoPayload.methodName=~".*insert.*"
         OR protoPayload.methodName=~".*delete.*"
         OR protoPayload.methodName=~".*update.*")
    ' \
    --project=$PROJECT_ID || echo "Sink already exists"

# Create BigQuery views for easy querying
echo ""
echo "Creating BigQuery views and materialized views..."
echo "=========================================="

# Security dashboard view
bq query --use_legacy_sql=false --project_id=$PROJECT_ID <<EOF
CREATE OR REPLACE VIEW \`$PROJECT_ID.$DATASET_ID.security_dashboard\` AS
SELECT
    'compute_instances' as resource_type,
    COUNT(*) as total_count,
    COUNTIF(external_ip IS NOT NULL) as exposed_count,
    MAX(last_refreshed) as last_updated
FROM \`$PROJECT_ID.$DATASET_ID.compute_instances\`
UNION ALL
SELECT
    'firewall_rules' as resource_type,
    COUNT(*) as total_count,
    COUNTIF(risk_level IN ('HIGH', 'CRITICAL')) as exposed_count,
    MAX(last_refreshed) as last_updated
FROM \`$PROJECT_ID.$DATASET_ID.firewall_rules\`
UNION ALL
SELECT
    'iam_accounts' as resource_type,
    COUNT(*) as total_count,
    COUNTIF(has_admin_privileges) as exposed_count,
    MAX(last_refreshed) as last_updated
FROM \`$PROJECT_ID.$DATASET_ID.iam_accounts\`
EOF

# Data freshness monitoring view
bq query --use_legacy_sql=false --project_id=$PROJECT_ID <<EOF
CREATE OR REPLACE VIEW \`$PROJECT_ID.$DATASET_ID.data_freshness\` AS
SELECT
    table_name,
    MAX(refresh_time) as last_refresh,
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(refresh_time), MINUTE) as minutes_since_refresh,
    CASE
        WHEN table_name = 'storage_buckets' AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(refresh_time), HOUR) > 2 THEN 'STALE'
        WHEN table_name = 'compute_instances' AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(refresh_time), HOUR) > 3 THEN 'STALE'
        WHEN table_name = 'firewall_rules' AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(refresh_time), HOUR) > 5 THEN 'STALE'
        WHEN table_name = 'iam_accounts' AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(refresh_time), HOUR) > 7 THEN 'STALE'
        ELSE 'FRESH'
    END as freshness_status
FROM \`$PROJECT_ID.$DATASET_ID.refresh_metadata\`
WHERE status = 'success'
GROUP BY table_name
EOF

# Test the functions
echo ""
echo "=========================================="
echo "Testing deployed functions..."
echo "=========================================="

for FUNCTION_NAME in "${!FUNCTIONS[@]}"; do
    echo ""
    echo "Testing $FUNCTION_NAME..."

    FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --format="value(url)" \
        --project=$PROJECT_ID)

    # Make test call
    RESPONSE=$(curl -s -X POST "$FUNCTION_URL" \
        -H "Content-Type: application/json" \
        -d '{"test": true}' || echo '{"error": "Failed to call function"}')

    echo "Response: $RESPONSE"
done

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Resources created:"
echo "  - 4 Cloud Functions for data refresh"
echo "  - 4 Cloud Scheduler jobs"
echo "  - 2 Pub/Sub topics"
echo "  - 1 Logging sink"
echo "  - Multiple BigQuery views"
echo ""
echo "Next steps:"
echo "  1. Check BigQuery tables are being populated:"
echo "     bq query --use_legacy_sql=false 'SELECT * FROM \`$PROJECT_ID.$DATASET_ID.data_freshness\`'"
echo ""
echo "  2. View security dashboard:"
echo "     bq query --use_legacy_sql=false 'SELECT * FROM \`$PROJECT_ID.$DATASET_ID.security_dashboard\`'"
echo ""
echo "  3. Monitor function logs:"
echo "     gcloud functions logs read --region=$REGION"
echo ""
echo "  4. Trigger manual refresh (example):"
echo "     gcloud scheduler jobs run schedule-fetch_compute_instances --location=$REGION"
echo ""