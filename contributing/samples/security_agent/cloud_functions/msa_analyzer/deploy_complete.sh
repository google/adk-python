#!/bin/bash

################################################################################
#                                                                              #
#   MSA (Multi-Service Analyzer) - Complete Deployment Script                 #
#   All-in-one setup: BigQuery + Cloud Function + Scheduler                   #
#                                                                              #
################################################################################

set -e  # Exit on error

# Configuration
PROJECT_ID=${1:-${GOOGLE_CLOUD_PROJECT}}
REGION=${2:-us-central1}
DATASET_ID="security_data"
FUNCTION_NAME="msa-analyzer"
BUCKET_SUFFIX="msa-cache"
PUBSUB_TOPIC="msa-critical-alerts"
SCHEDULER_JOB="msa-analyzer-daily"
SCHEDULER_CRON="0 9 * * *"  # Daily at 9 AM
TIMEZONE="America/New_York"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     MSA (Multi-Service Analyzer) Complete Deployment          ║"
    echo "║     BigQuery + Cloud Function + Scheduler                     ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${MAGENTA}ℹ️  $1${NC}"
}

check_requirements() {
    print_step "Checking requirements..."

    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check bq
    if ! command -v bq &> /dev/null; then
        print_error "bq command not found. Please install Google Cloud SDK"
        exit 1
    fi

    # Check project ID
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID not provided"
        echo "Usage: $0 <project-id> [region]"
        exit 1
    fi

    print_success "All requirements met"
}

################################################################################
# Main Deployment Steps
################################################################################

deploy_all() {
    print_header

    print_info "Configuration:"
    echo "   Project ID:      $PROJECT_ID"
    echo "   Region:          $REGION"
    echo "   Dataset:         $DATASET_ID"
    echo "   Function:        $FUNCTION_NAME"
    echo "   Schedule:        $SCHEDULER_CRON ($TIMEZONE)"
    echo ""

    check_requirements

    # Set active project
    print_step "Setting active GCP project..."
    gcloud config set project "$PROJECT_ID" --quiet
    print_success "Project set to $PROJECT_ID"

    #---------------------------------------------------------------------------
    # STEP 1: Enable APIs
    #---------------------------------------------------------------------------
    print_section "Step 1/7: Enabling GCP APIs"

    print_step "Enabling required APIs (this may take a minute)..."
    gcloud services enable \
        cloudfunctions.googleapis.com \
        cloudbuild.googleapis.com \
        cloudscheduler.googleapis.com \
        pubsub.googleapis.com \
        bigquery.googleapis.com \
        storage.googleapis.com \
        artifactregistry.googleapis.com \
        run.googleapis.com \
        logging.googleapis.com \
        --project="$PROJECT_ID" \
        --quiet

    print_success "All APIs enabled"

    #---------------------------------------------------------------------------
    # STEP 2: Create Service Accounts
    #---------------------------------------------------------------------------
    print_section "Step 2/7: Creating Service Accounts"

    # Function service account
    SA_NAME="msa-analyzer-sa"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

    print_step "Creating MSA Analyzer service account..."
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="MSA Analyzer Service Account" \
        --description="Service account for MSA Analyzer Cloud Function" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || print_info "Service account already exists"

    # Scheduler service account
    SCHEDULER_SA="msa-scheduler-sa"
    SCHEDULER_SA_EMAIL="${SCHEDULER_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

    print_step "Creating Scheduler service account..."
    gcloud iam service-accounts create "$SCHEDULER_SA" \
        --display-name="MSA Scheduler Service Account" \
        --description="Service account for MSA Cloud Scheduler" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || print_info "Service account already exists"

    print_success "Service accounts ready"

    #---------------------------------------------------------------------------
    # STEP 3: Grant IAM Permissions
    #---------------------------------------------------------------------------
    print_section "Step 3/7: Configuring IAM Permissions"

    print_step "Granting BigQuery permissions..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/bigquery.dataEditor" \
        --condition=None \
        --quiet > /dev/null

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/bigquery.jobUser" \
        --condition=None \
        --quiet > /dev/null

    print_step "Granting Storage permissions..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/storage.objectAdmin" \
        --condition=None \
        --quiet > /dev/null

    print_step "Granting Pub/Sub permissions..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/pubsub.publisher" \
        --condition=None \
        --quiet > /dev/null

    print_success "IAM permissions configured"

    #---------------------------------------------------------------------------
    # STEP 4: Create BigQuery Resources
    #---------------------------------------------------------------------------
    print_section "Step 4/7: Setting up BigQuery"

    print_step "Creating dataset..."
    bq mk --dataset \
        --location=US \
        --description="Security data including MSA analysis results" \
        "${PROJECT_ID}:${DATASET_ID}" 2>/dev/null || print_info "Dataset already exists"

    print_step "Creating msa_analysis_history table..."
    bq mk --table \
        --description="MSA analysis results for GCP release notes monitoring" \
        --time_partitioning_field=timestamp \
        --clustering_fields=risk_level,services_affected \
        "${PROJECT_ID}:${DATASET_ID}.msa_analysis_history" \
        analysis_id:STRING,timestamp:TIMESTAMP,total_changes:INTEGER,services_affected:INTEGER,risk_score:INTEGER,risk_level:STRING,critical_issues:INTEGER,security_risk:STRING,billing_impact:STRING,compliance_impact:STRING,recommendations:STRING,full_report:STRING,created_at:TIMESTAMP \
        2>/dev/null || print_info "Table already exists"

    print_step "Creating active_services table..."
    bq mk --table \
        --description="Active GCP services being monitored by MSA" \
        --clustering_fields=status,service_name \
        "${PROJECT_ID}:${DATASET_ID}.active_services" \
        service_name:STRING,service_type:STRING,status:STRING,project_id:STRING,enabled_date:DATE,last_used:TIMESTAMP,usage_count:INTEGER,created_at:TIMESTAMP,updated_at:TIMESTAMP \
        2>/dev/null || print_info "Table already exists"

    print_step "Populating active_services with common GCP services..."
    cat << 'EOF' | bq query --use_legacy_sql=false --project_id="$PROJECT_ID" 2>/dev/null || print_info "Data may already exist"
INSERT INTO `security_data.active_services`
(service_name, service_type, status, enabled_date, created_at, updated_at)
SELECT * FROM UNNEST([
  STRUCT('BigQuery' AS service_name, 'data-analytics' AS service_type, 'active' AS status, CURRENT_DATE() AS enabled_date, CURRENT_TIMESTAMP() AS created_at, CURRENT_TIMESTAMP() AS updated_at),
  ('Cloud Storage', 'storage', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Compute Engine', 'compute', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud Run', 'compute', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud Functions', 'compute', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud SQL', 'database', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Pub/Sub', 'messaging', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Vertex AI', 'ai-ml', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud KMS', 'security', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Secret Manager', 'security', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('VPC', 'networking', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud Armor', 'security', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Identity Platform', 'security', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Firestore', 'database', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud Spanner', 'database', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
])
WHERE service_name NOT IN (SELECT service_name FROM `security_data.active_services`)
EOF

    print_step "Creating BigQuery views..."

    # Latest summary view
    bq mk --view \
        "SELECT analysis_id, timestamp, total_changes, services_affected, risk_score, risk_level, critical_issues, security_risk, billing_impact, compliance_impact, JSON_EXTRACT_SCALAR(recommendations, '\$[0].action') as top_recommendation, created_at FROM \`${PROJECT_ID}.${DATASET_ID}.msa_analysis_history\` WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) ORDER BY timestamp DESC LIMIT 100" \
        --use_legacy_sql=false \
        --project_id="$PROJECT_ID" \
        "${PROJECT_ID}:${DATASET_ID}.msa_latest_summary" 2>/dev/null || print_info "View already exists"

    # Critical issues view
    bq mk --view \
        "SELECT analysis_id, timestamp, risk_level, critical_issues, security_risk, JSON_EXTRACT_ARRAY(recommendations) as recommendations_array, created_at FROM \`${PROJECT_ID}.${DATASET_ID}.msa_analysis_history\` WHERE critical_issues > 0 OR risk_level = 'high' OR security_risk = 'high' ORDER BY timestamp DESC" \
        --use_legacy_sql=false \
        --project_id="$PROJECT_ID" \
        "${PROJECT_ID}:${DATASET_ID}.msa_critical_issues" 2>/dev/null || print_info "View already exists"

    # Billing trends view
    bq mk --view \
        "SELECT DATE(timestamp) as analysis_date, COUNT(*) as analysis_count, SUM(CASE WHEN billing_impact = 'increase' THEN 1 ELSE 0 END) as price_increases, SUM(CASE WHEN billing_impact = 'decrease' THEN 1 ELSE 0 END) as price_decreases, AVG(risk_score) as avg_risk_score FROM \`${PROJECT_ID}.${DATASET_ID}.msa_analysis_history\` WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) GROUP BY DATE(timestamp) ORDER BY analysis_date DESC" \
        --use_legacy_sql=false \
        --project_id="$PROJECT_ID" \
        "${PROJECT_ID}:${DATASET_ID}.msa_billing_trends" 2>/dev/null || print_info "View already exists"

    print_success "BigQuery fully configured"

    #---------------------------------------------------------------------------
    # STEP 5: Create Cloud Storage & Pub/Sub
    #---------------------------------------------------------------------------
    print_section "Step 5/7: Setting up Storage & Pub/Sub"

    BUCKET_NAME="${PROJECT_ID}-${BUCKET_SUFFIX}"

    print_step "Creating storage bucket for cache..."
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://${BUCKET_NAME}/" 2>/dev/null || print_info "Bucket already exists"

    print_step "Setting bucket lifecycle policy..."
    cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF
    gsutil lifecycle set /tmp/lifecycle.json "gs://${BUCKET_NAME}/"
    rm /tmp/lifecycle.json

    print_step "Creating Pub/Sub topic for alerts..."
    gcloud pubsub topics create "$PUBSUB_TOPIC" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || print_info "Topic already exists"

    print_step "Creating Pub/Sub subscription..."
    gcloud pubsub subscriptions create "${PUBSUB_TOPIC}-subscription" \
        --topic="$PUBSUB_TOPIC" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || print_info "Subscription already exists"

    print_success "Storage and Pub/Sub ready"

    #---------------------------------------------------------------------------
    # STEP 6: Deploy Cloud Function
    #---------------------------------------------------------------------------
    print_section "Step 6/7: Deploying Cloud Function"

    print_step "Copying MSA analyzer module..."
    cp ../../agents/_tools/msa_analyzer.py .

    print_step "Deploying function (this may take 2-3 minutes)..."
    gcloud functions deploy "$FUNCTION_NAME" \
        --gen2 \
        --runtime=python311 \
        --region="$REGION" \
        --source=. \
        --entry-point=analyze_releases \
        --trigger-http \
        --allow-unauthenticated \
        --memory=512MB \
        --timeout=300s \
        --service-account="$SA_EMAIL" \
        --set-env-vars="GCP_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
        --project="$PROJECT_ID" \
        --quiet

    print_success "Cloud Function deployed"

    # Get function URL
    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" \
        --region="$REGION" \
        --format="value(serviceConfig.uri)" \
        --project="$PROJECT_ID")

    print_info "Function URL: $FUNCTION_URL"

    # Cleanup
    rm -f msa_analyzer.py

    #---------------------------------------------------------------------------
    # STEP 7: Setup Cloud Scheduler
    #---------------------------------------------------------------------------
    print_section "Step 7/7: Configuring Cloud Scheduler"

    print_step "Granting scheduler permission to invoke function..."
    gcloud functions add-iam-policy-binding "$FUNCTION_NAME" \
        --region="$REGION" \
        --member="serviceAccount:${SCHEDULER_SA_EMAIL}" \
        --role="roles/cloudfunctions.invoker" \
        --project="$PROJECT_ID" \
        --quiet > /dev/null

    print_step "Creating scheduler job..."
    # Delete if exists
    gcloud scheduler jobs delete "$SCHEDULER_JOB" \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || true

    # Create new job
    gcloud scheduler jobs create http "$SCHEDULER_JOB" \
        --location="$REGION" \
        --schedule="$SCHEDULER_CRON" \
        --time-zone="$TIMEZONE" \
        --uri="$FUNCTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"days_back": 7}' \
        --oidc-service-account-email="$SCHEDULER_SA_EMAIL" \
        --project="$PROJECT_ID" \
        --quiet

    print_success "Scheduler configured (runs daily at 9 AM ET)"

    #---------------------------------------------------------------------------
    # DEPLOYMENT COMPLETE
    #---------------------------------------------------------------------------
    print_section "🎉 DEPLOYMENT COMPLETE!"

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                ║${NC}"
    echo -e "${GREEN}║           MSA Analyzer Successfully Deployed! ✅               ║${NC}"
    echo -e "${GREEN}║                                                                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    print_info "📊 Resources Created:"
    echo ""
    echo "   BigQuery:"
    echo "   • Dataset: ${PROJECT_ID}:${DATASET_ID}"
    echo "   • Tables: msa_analysis_history, active_services"
    echo "   • Views: msa_latest_summary, msa_critical_issues, msa_billing_trends"
    echo ""
    echo "   Cloud Storage:"
    echo "   • Bucket: gs://${BUCKET_NAME}"
    echo ""
    echo "   Pub/Sub:"
    echo "   • Topic: ${PUBSUB_TOPIC}"
    echo "   • Subscription: ${PUBSUB_TOPIC}-subscription"
    echo ""
    echo "   Cloud Function:"
    echo "   • Name: ${FUNCTION_NAME}"
    echo "   • Region: ${REGION}"
    echo "   • URL: ${FUNCTION_URL}"
    echo ""
    echo "   Cloud Scheduler:"
    echo "   • Job: ${SCHEDULER_JOB}"
    echo "   • Schedule: Daily at 9 AM ET"
    echo ""
    echo "   Service Accounts:"
    echo "   • Function SA: ${SA_EMAIL}"
    echo "   • Scheduler SA: ${SCHEDULER_SA_EMAIL}"
    echo ""

    print_info "🧪 Test Your Deployment:"
    echo ""
    echo -e "${CYAN}# Test the function manually${NC}"
    echo "curl -X POST $FUNCTION_URL \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"days_back\": 7}'"
    echo ""
    echo -e "${CYAN}# Run scheduler job immediately${NC}"
    echo "gcloud scheduler jobs run $SCHEDULER_JOB --location=$REGION"
    echo ""
    echo -e "${CYAN}# Query results in BigQuery${NC}"
    echo "bq query --use_legacy_sql=false \\"
    echo "  'SELECT * FROM \`${PROJECT_ID}.${DATASET_ID}.msa_latest_summary\` LIMIT 5'"
    echo ""
    echo -e "${CYAN}# View function logs${NC}"
    echo "gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=50"
    echo ""

    print_info "📚 Documentation:"
    echo "   • Complete guide: README.md"
    echo "   • Testing guide: TESTING.md"
    echo "   • SQL schema: bigquery_setup.sql"
    echo ""

    print_info "💰 Estimated Cost: ~\$0.20/month (with daily runs)"
    echo ""

    print_success "Setup complete! MSA Analyzer is monitoring GCP release notes."
    echo ""
}

################################################################################
# Run Deployment
################################################################################

deploy_all

exit 0
