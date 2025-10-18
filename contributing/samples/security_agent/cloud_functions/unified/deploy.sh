#!/bin/bash

################################################################################
#                                                                              #
#   Unified Cloud Function Deployment Script                                  #
#   Deploy consolidated fetch functions with Vellox                           #
#                                                                              #
################################################################################

set -e

# Configuration
PROJECT_ID=${1:-${GOOGLE_CLOUD_PROJECT}}
REGION=${2:-us-central1}
FUNCTION_NAME="unified-security-fetcher"
ENTRY_POINT="unified_handler"
RUNTIME="python311"
MEMORY="1024MB"
TIMEOUT="540s"
MAX_INSTANCES="10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          Unified Cloud Function Deployment                     ║"
    echo "║          Single Surface for All Fetchers                       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_header

# Validate inputs
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: Project ID not provided${NC}"
    echo "Usage: $0 <project-id> [region]"
    exit 1
fi

echo -e "${CYAN}Deployment Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Function: $FUNCTION_NAME"
echo "  Memory: $MEMORY"
echo "  Timeout: $TIMEOUT"
echo ""

# Set project
echo -e "${YELLOW}Setting project to ${PROJECT_ID}...${NC}"
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    bigquery.googleapis.com \
    iam.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    securitycenter.googleapis.com \
    --project="$PROJECT_ID" || true

# Create BigQuery datasets if they don't exist
echo -e "${YELLOW}Setting up BigQuery datasets...${NC}"
bq mk --location="$REGION" --dataset --project_id="$PROJECT_ID" security_insights 2>/dev/null || true
bq mk --location="$REGION" --dataset --project_id="$PROJECT_ID" security_data 2>/dev/null || true

# Deploy the unified Cloud Function
echo -e "${GREEN}Deploying unified Cloud Function...${NC}"

gcloud functions deploy "$FUNCTION_NAME" \
    --gen2 \
    --runtime="$RUNTIME" \
    --region="$REGION" \
    --source="." \
    --entry-point="$ENTRY_POINT" \
    --trigger-http \
    --allow-unauthenticated \
    --memory="$MEMORY" \
    --timeout="$TIMEOUT" \
    --max-instances="$MAX_INSTANCES" \
    --set-env-vars="PROJECT_ID=$PROJECT_ID,BQ_DATASET_ID=security_insights,BQ_LOCATION=$REGION,ENABLE_SAMPLE_DATA=true" \
    --project="$PROJECT_ID"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" \
    --region="$REGION" \
    --format="value(serviceConfig.uri)" \
    --project="$PROJECT_ID")

echo -e "${GREEN}Function deployed successfully!${NC}"
echo -e "URL: ${BLUE}$FUNCTION_URL${NC}"
echo ""

# Create Cloud Scheduler jobs for each fetcher
echo -e "${YELLOW}Setting up Cloud Scheduler jobs...${NC}"

# Function to create scheduler job
create_scheduler_job() {
    local fetcher=$1
    local schedule=$2
    local description=$3

    job_name="${fetcher//_/-}-schedule"

    echo -e "  Creating job: ${job_name} (${schedule})"

    gcloud scheduler jobs create http "$job_name" \
        --location="$REGION" \
        --schedule="$schedule" \
        --uri="${FUNCTION_URL}/trigger/${fetcher}" \
        --http-method=GET \
        --description="$description" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || \
    gcloud scheduler jobs update http "$job_name" \
        --location="$REGION" \
        --schedule="$schedule" \
        --uri="${FUNCTION_URL}/trigger/${fetcher}" \
        --description="$description" \
        --project="$PROJECT_ID" \
        --quiet
}

# Create scheduler jobs for each fetcher
echo -e "${CYAN}Creating Cloud Scheduler jobs...${NC}"

create_scheduler_job "security_findings" "0 */2 * * *" "Fetch security findings every 2 hours"
create_scheduler_job "custom_roles" "0 9 * * *" "Fetch custom IAM roles daily at 9 AM"
create_scheduler_job "compute_instances" "0 */4 * * *" "Fetch compute instances every 4 hours"
create_scheduler_job "firewall_rules" "0 */4 * * *" "Fetch firewall rules every 4 hours"
create_scheduler_job "storage_buckets" "0 */6 * * *" "Fetch storage buckets every 6 hours"
create_scheduler_job "iam_accounts" "0 */4 * * *" "Fetch IAM accounts every 4 hours"
create_scheduler_job "service_account_roles" "0 */4 * * *" "Fetch service account roles every 4 hours"
create_scheduler_job "standard_roles" "0 9 * * 1" "Fetch standard roles weekly on Monday"
create_scheduler_job "user_roles" "0 */4 * * *" "Fetch user roles every 4 hours"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Deployment Complete!                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Unified Cloud Function deployed successfully${NC}"
echo ""
echo -e "${CYAN}Available Endpoints:${NC}"
echo "  • Health: ${FUNCTION_URL}/health"
echo "  • List fetchers: ${FUNCTION_URL}/fetchers"
echo "  • API Docs: ${FUNCTION_URL}/docs"
echo "  • Individual fetchers: ${FUNCTION_URL}/fetch/{fetcher_name}"
echo "  • Fetch all: ${FUNCTION_URL}/fetch/all"
echo ""
echo -e "${CYAN}Cloud Scheduler Triggers:${NC}"
echo "  • Each fetcher has its own schedule"
echo "  • View schedules: gcloud scheduler jobs list --location=$REGION"
echo "  • Trigger manually: gcloud scheduler jobs run JOB_NAME --location=$REGION"
echo ""
echo -e "${CYAN}Test the deployment:${NC}"
echo "  curl ${FUNCTION_URL}/health"
echo "  curl -X POST ${FUNCTION_URL}/fetch/security_findings"
echo ""
echo -e "${CYAN}View logs:${NC}"
echo "  gcloud functions logs read $FUNCTION_NAME --region=$REGION"
echo ""

exit 0
