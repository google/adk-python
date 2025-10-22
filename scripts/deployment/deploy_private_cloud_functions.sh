#!/bin/bash
# Deploy Private Cloud Functions with Internal-Only Access
# This script deploys the unified security fetcher with private networking

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}
REGION=${REGION:-us-central1}
FUNCTION_NAME="unified-security-fetcher"
SERVICE_ACCOUNT_NAME="security-fetcher-sa"
VPC_NETWORK="default"
VPC_SUBNET="default"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Private Cloud Functions Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Project ID:${NC} $PROJECT_ID"
echo -e "${GREEN}Region:${NC} $REGION"
echo -e "${GREEN}Function:${NC} $FUNCTION_NAME"
echo ""

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Step 1: Enable required APIs
echo -e "${YELLOW}Step 1: Enabling required APIs...${NC}"
gcloud services enable \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  bigquery.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  securitycenter.googleapis.com \
  vpcaccess.googleapis.com \
  --project=$PROJECT_ID

echo -e "${GREEN}✓ APIs enabled${NC}\n"

# Step 2: Create service account
echo -e "${YELLOW}Step 2: Creating dedicated service account...${NC}"

# Check if service account already exists
if gcloud iam service-accounts describe ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ Service account already exists${NC}"
else
  gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
    --display-name="Security Fetcher Function Service Account" \
    --project=$PROJECT_ID
  echo -e "${GREEN}✓ Service account created${NC}"
fi

# Step 3: Grant IAM permissions
echo -e "${YELLOW}Step 3: Granting IAM permissions (principle of least privilege)...${NC}"

ROLES=(
  "roles/bigquery.dataEditor"
  "roles/securitycenter.findingsEditor"
  "roles/compute.viewer"
  "roles/iam.securityReviewer"
  "roles/storage.objectViewer"
)

for ROLE in "${ROLES[@]}"; do
  echo "  Granting $ROLE..."
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="$ROLE" \
    --condition=None \
    --quiet >/dev/null 2>&1 || true
done

echo -e "${GREEN}✓ IAM permissions granted${NC}\n"

# Step 4: Create BigQuery datasets
echo -e "${YELLOW}Step 4: Creating BigQuery datasets...${NC}"

# Security insights dataset
if ! bq ls -d --project_id=$PROJECT_ID | grep -q "security_insights"; then
  bq mk --dataset \
    --location=us-central1 \
    --description="Security insights and findings data" \
    ${PROJECT_ID}:security_insights
  echo -e "${GREEN}✓ Created security_insights dataset${NC}"
else
  echo -e "${GREEN}✓ security_insights dataset already exists${NC}"
fi

# Security data dataset
if ! bq ls -d --project_id=$PROJECT_ID | grep -q "security_data"; then
  bq mk --dataset \
    --location=us-central1 \
    --description="Security analysis and MSA data" \
    ${PROJECT_ID}:security_data
  echo -e "${GREEN}✓ Created security_data dataset${NC}"
else
  echo -e "${GREEN}✓ security_data dataset already exists${NC}"
fi

echo ""

# Step 5: Verify VPC network and subnet
echo -e "${YELLOW}Step 5: Verifying VPC network configuration...${NC}"

if gcloud compute networks describe $VPC_NETWORK --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ VPC network '$VPC_NETWORK' exists${NC}"
else
  echo -e "${RED}✗ VPC network '$VPC_NETWORK' not found${NC}"
  echo "  Creating default network..."
  gcloud compute networks create $VPC_NETWORK --subnet-mode=auto --project=$PROJECT_ID
fi

if gcloud compute networks subnets describe $VPC_SUBNET --region=$REGION --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ Subnet '$VPC_SUBNET' exists in $REGION${NC}"
else
  echo -e "${RED}✗ Subnet '$VPC_SUBNET' not found in $REGION${NC}"
  echo "  Please create the subnet before continuing."
  exit 1
fi

echo ""

# Step 6: Deploy Cloud Function with private settings
echo -e "${YELLOW}Step 6: Deploying Cloud Function with private networking...${NC}"
echo ""

cd "$(dirname "$0")/../../contributing/samples/security_agent/cloud_functions/unified"

echo "Deploying with configuration:"
echo "  - Gen2 function"
echo "  - Internal-only ingress"
echo "  - No unauthenticated access"
echo "  - VPC egress: private-ranges-only"
echo "  - Service account: ${SERVICE_ACCOUNT_NAME}"
echo ""

gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --region=$REGION \
  --runtime=python311 \
  --source=. \
  --entry-point=unified_handler \
  --trigger-http \
  --memory=1024Mi \
  --timeout=540s \
  --max-instances=10 \
  --min-instances=0 \
  --ingress-settings=internal-only \
  --no-allow-unauthenticated \
  --service-account=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
  --vpc-egress=private-ranges-only \
  --network=projects/${PROJECT_ID}/global/networks/${VPC_NETWORK} \
  --subnet=projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${VPC_SUBNET} \
  --set-env-vars=PROJECT_ID=${PROJECT_ID},BQ_DATASET_ID=security_insights,BQ_LOCATION=${REGION} \
  --project=$PROJECT_ID

echo -e "${GREEN}✓ Cloud Function deployed${NC}\n"

# Step 7: Grant Cloud Functions Invoker role
echo -e "${YELLOW}Step 7: Granting Cloud Functions Invoker role...${NC}"

# Allow Cloud Scheduler to invoke
echo "  Granting invoker role to Cloud Scheduler service account..."
gcloud functions add-iam-policy-binding $FUNCTION_NAME \
  --region=$REGION \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker" \
  --project=$PROJECT_ID

# Allow service account to invoke (for testing and agent access)
echo "  Granting invoker role to security agent service account..."
gcloud functions add-iam-policy-binding $FUNCTION_NAME \
  --region=$REGION \
  --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker" \
  --project=$PROJECT_ID

echo -e "${GREEN}✓ Invoker permissions granted${NC}\n"

# Step 8: Create/update Cloud Scheduler jobs with OIDC authentication
echo -e "${YELLOW}Step 8: Creating Cloud Scheduler jobs with OIDC authentication...${NC}"

FUNCTION_URL="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/${FUNCTION_NAME}"
SCHEDULER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Define fetchers and their schedules
declare -A SCHEDULES
SCHEDULES=(
  ["security_findings"]="0 */2 * * *"
  ["custom_roles"]="0 9 * * *"
  ["compute_instances"]="0 */4 * * *"
  ["firewall_rules"]="0 */4 * * *"
  ["storage_buckets"]="0 */6 * * *"
  ["iam_accounts"]="0 */4 * * *"
  ["service_account_roles"]="0 */4 * * *"
  ["standard_roles"]="0 9 * * 1"
  ["user_roles"]="0 */4 * * *"
)

for FETCHER in "${!SCHEDULES[@]}"; do
  JOB_NAME="unified-${FETCHER}-trigger"
  SCHEDULE="${SCHEDULES[$FETCHER]}"

  echo "  Creating/updating job: $JOB_NAME"

  # Try to delete existing job (ignore errors if doesn't exist)
  gcloud scheduler jobs delete $JOB_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --quiet &>/dev/null || true

  # Create new job with OIDC authentication
  gcloud scheduler jobs create http $JOB_NAME \
    --location=$REGION \
    --schedule="$SCHEDULE" \
    --uri="${FUNCTION_URL}/trigger/${FETCHER}" \
    --http-method=GET \
    --oidc-service-account-email=$SCHEDULER_SA \
    --oidc-token-audience=$FUNCTION_URL \
    --time-zone="America/New_York" \
    --project=$PROJECT_ID \
    --quiet

done

echo -e "${GREEN}✓ Cloud Scheduler jobs created${NC}\n"

# Step 9: Display summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Function URL (internal only):${NC}"
echo "  $FUNCTION_URL"
echo ""
echo -e "${GREEN}Service Account:${NC}"
echo "  ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
echo ""
echo -e "${GREEN}Endpoints (require authentication):${NC}"
echo "  POST $FUNCTION_URL/fetch/{fetcher_name}"
echo "  POST $FUNCTION_URL/fetch/all"
echo "  GET  $FUNCTION_URL/fetchers"
echo "  GET  $FUNCTION_URL/health"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Test authentication: ./scripts/testing/test_private_functions.sh"
echo "  2. Verify Cloud Scheduler jobs: gcloud scheduler jobs list --location=$REGION"
echo "  3. Monitor function logs: gcloud functions logs read $FUNCTION_NAME --region=$REGION"
echo "  4. Update Security Agent with authentication logic"
echo ""
echo -e "${GREEN}Deployment script completed successfully!${NC}"
