#!/bin/bash

# Comprehensive Deployment Status Check Script
# Shows status of all Cloud Functions and BigQuery tables

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-mgm-digitalconcierge}"
REGION="${REGION:-us-central1}"
DATASET_ID="${DATASET_ID:-security_insights}"

echo "=================================================="
echo "🚀 DEPLOYMENT STATUS REPORT"
echo "=================================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Dataset: $DATASET_ID"
echo "Timestamp: $(date)"
echo ""

# Check Cloud Functions
echo "=================================================="
echo "☁️  CLOUD FUNCTIONS STATUS"
echo "=================================================="
echo ""

# Expected functions
EXPECTED_FUNCTIONS=(
    "fetch_gcp_release_notes"
    "fetch_security_feeds"
    "fetch_compute_instances"
    "fetch_firewall_rules"
    "fetch_iam_accounts"
    "fetch_security_findings"
    "fetch_storage_buckets"
)

# Get deployed functions
echo "Deployed Functions:"
echo "-------------------"
DEPLOYED_COUNT=0

for FUNCTION_NAME in "${EXPECTED_FUNCTIONS[@]}"; do
    STATE=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format="value(state)" 2>/dev/null || echo "NOT_DEPLOYED")

    if [ "$STATE" = "ACTIVE" ]; then
        echo "✅ $FUNCTION_NAME: ACTIVE"
        ((DEPLOYED_COUNT++))
    elif [ "$STATE" = "NOT_DEPLOYED" ]; then
        echo "❌ $FUNCTION_NAME: NOT DEPLOYED"
    else
        echo "⚠️  $FUNCTION_NAME: $STATE"
    fi
done

echo ""
echo "Summary: $DEPLOYED_COUNT / ${#EXPECTED_FUNCTIONS[@]} functions deployed"
echo ""

# Check BigQuery Tables
echo "=================================================="
echo "📊 BIGQUERY TABLES STATUS"
echo "=================================================="
echo ""

# Expected tables
EXPECTED_TABLES=(
    "gcp_release_notes"
    "security_threat_feeds"
    "compute_instances"
    "firewall_rules"
    "iam_accounts"
    "security_findings"
    "storage_buckets"
)

echo "BigQuery Tables:"
echo "----------------"
TABLE_COUNT=0

for TABLE_NAME in "${EXPECTED_TABLES[@]}"; do
    if bq show --project_id=$PROJECT_ID $DATASET_ID.$TABLE_NAME &> /dev/null; then
        ROW_COUNT=$(bq query --project_id=$PROJECT_ID --use_legacy_sql=false \
            "SELECT COUNT(*) as count FROM \`$PROJECT_ID.$DATASET_ID.$TABLE_NAME\`" \
            --format=csv --max_rows=1 2>/dev/null | tail -1 || echo "0")
        echo "✅ $TABLE_NAME: $ROW_COUNT rows"
        ((TABLE_COUNT++))
    else
        echo "❌ $TABLE_NAME: NOT CREATED"
    fi
done

echo ""
echo "Summary: $TABLE_COUNT / ${#EXPECTED_TABLES[@]} tables created"
echo ""

# Check Cloud Scheduler Jobs
echo "=================================================="
echo "⏰ CLOUD SCHEDULER JOBS"
echo "=================================================="
echo ""

echo "Scheduler Jobs:"
echo "--------------"
SCHEDULER_COUNT=0

gcloud scheduler jobs list --location=$REGION --project=$PROJECT_ID 2>/dev/null | grep -E "schedule-fetch" | while read -r line; do
    JOB_NAME=$(echo $line | awk '{print $1}')
    STATE=$(echo $line | awk '{print $2}')
    SCHEDULE=$(echo $line | awk '{print $3,$4,$5,$6,$7}')

    if [ "$STATE" = "ENABLED" ]; then
        echo "✅ $JOB_NAME: $SCHEDULE"
        ((SCHEDULER_COUNT++))
    else
        echo "⚠️  $JOB_NAME: $STATE"
    fi
done || echo "No scheduler jobs found"

echo ""

# Function URLs for testing
echo "=================================================="
echo "🔗 FUNCTION URLS FOR TESTING"
echo "=================================================="
echo ""

for FUNCTION_NAME in "${EXPECTED_FUNCTIONS[@]}"; do
    URL=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format="value(url)" 2>/dev/null || echo "")

    if [ ! -z "$URL" ]; then
        echo "$FUNCTION_NAME:"
        echo "  $URL"
        echo ""
    fi
done

# Next Steps
echo "=================================================="
echo "📝 NEXT STEPS"
echo "=================================================="

if [ "$DEPLOYED_COUNT" -lt "${#EXPECTED_FUNCTIONS[@]}" ]; then
    echo "1. Deploy missing Cloud Functions:"
    echo "   ./scripts/deploy_all_security_functions.sh"
    echo ""
fi

if [ "$TABLE_COUNT" -lt "${#EXPECTED_TABLES[@]}" ]; then
    echo "2. Trigger functions to create missing tables:"
    for FUNCTION_NAME in "${EXPECTED_FUNCTIONS[@]}"; do
        STATE=$(gcloud functions describe $FUNCTION_NAME \
            --region=$REGION \
            --project=$PROJECT_ID \
            --format="value(state)" 2>/dev/null || echo "NOT_DEPLOYED")

        if [ "$STATE" = "ACTIVE" ]; then
            echo "   gcloud functions call $FUNCTION_NAME --region=$REGION"
        fi
    done
    echo ""
fi

echo "3. Set up Cloud Scheduler for automated refreshes:"
echo "   ./scripts/setup_schedulers.sh"
echo ""

echo "4. Test individual functions:"
echo "   curl -X POST [FUNCTION_URL] -H 'Content-Type: application/json' -d '{}'"
echo ""

echo "=================================================="
echo "✅ STATUS CHECK COMPLETE"
echo "==================================================