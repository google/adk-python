#!/bin/bash

# Cloud Run Deployment Script for Security Agent API
# Lightweight deployment optimized for Cloud Functions architecture

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
SERVICE_NAME="security-agent-api"
# Use existing service account when provided (env or key file)
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-}"
if [ -z "$SERVICE_ACCOUNT_EMAIL" ] && [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    if command -v jq >/dev/null 2>&1; then
        SERVICE_ACCOUNT_EMAIL=$(jq -r '.client_email // empty' "$GOOGLE_APPLICATION_CREDENTIALS")
    elif command -v python3 >/dev/null 2>&1; then
        SERVICE_ACCOUNT_EMAIL=$(python3 - <<'PY'
import json, os
path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if path:
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    print(data.get('client_email', ''))
PY
)
        SERVICE_ACCOUNT_EMAIL=$(echo "$SERVICE_ACCOUNT_EMAIL" | tr -d '\n')
    fi
fi

CREATE_SERVICE_ACCOUNT=false
if [ -z "$SERVICE_ACCOUNT_EMAIL" ]; then
    SERVICE_ACCOUNT_EMAIL="${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    CREATE_SERVICE_ACCOUNT=true
fi
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_EMAIL%@*}"
# Data + evaluation configuration
DATASET="${BQ_DEFAULT_DATASET:-security_insights}"
DATASET_LOCATION="${BQ_DATASET_LOCATION:-us-central1}"
INTERACTIONS_TABLE="${AGENT_CONVERSATIONS_TABLE:-agent_conversations}"
EVALUATIONS_TABLE="${AGENT_EVALUATIONS_TABLE:-agent_evaluations}"
VERTEX_LOCATION="${VERTEX_AI_LOCATION:-us-central1}"
EVAL_CANDIDATE="${AGENT_EVALUATION_CANDIDATE_NAME:-${SERVICE_NAME}}"
CLOUDBUILD_CONFIG="${CLOUDBUILD_CONFIG:-cloudbuild-cloudrun.yaml}"

echo -e "${GREEN}=== Security Agent API - Cloud Run Deployment ===${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GOOGLE_CLOUD_PROJECT environment variable not set${NC}"
    echo "Please set: export GOOGLE_CLOUD_PROJECT=your-project-id"
    exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not installed${NC}"
    exit 1
fi

if ! command -v bq &> /dev/null; then
    echo -e "${RED}Error: bq CLI not installed (install via gcloud components install bq)${NC}"
    exit 1
fi

# Set the project
echo "Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Get current user for authentication check
CURRENT_USER=$(gcloud config get-value account 2>/dev/null)
echo "Authenticated as: $CURRENT_USER"
echo ""

# Create service account if needed
echo -e "${YELLOW}Setting up service account...${NC}"
if [ "$CREATE_SERVICE_ACCOUNT" = true ]; then
    if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT_EMAIL} &>/dev/null; then
        echo "Creating service account: ${SERVICE_ACCOUNT_EMAIL}"
        gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
            --display-name="Security Agent API Service Account"
    fi
else
    echo "Using existing service account: ${SERVICE_ACCOUNT_EMAIL}"
fi
echo ""

# Grant necessary permissions (idempotent)
echo "Granting permissions..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/cloudfunctions.invoker"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.user"
echo ""

# Enable required APIs
echo -e "${YELLOW}Enabling required Google Cloud services...${NC}"
gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --project ${PROJECT_ID}

# Ensure BigQuery dataset and tables exist
echo -e "${YELLOW}Ensuring BigQuery dataset (${DATASET}) and tables exist...${NC}"
if ! bq --project_id=${PROJECT_ID} ls --format=none ${DATASET} >/dev/null 2>&1; then
    echo "Creating dataset ${DATASET} in ${DATASET_LOCATION}"
    bq --project_id=${PROJECT_ID} --location=${DATASET_LOCATION} mk --dataset ${DATASET}
fi

if ! bq --project_id=${PROJECT_ID} ls --format=none ${DATASET}.${INTERACTIONS_TABLE} >/dev/null 2>&1; then
    echo "Creating table ${INTERACTIONS_TABLE}"
    bq --project_id=${PROJECT_ID} mk --table ${DATASET}.${INTERACTIONS_TABLE} \
        session_id:STRING,interaction_index:INT64,user_prompt:STRING,agent_response:STRING,created_at:TIMESTAMP
fi

if ! bq --project_id=${PROJECT_ID} ls --format=none ${DATASET}.${EVALUATIONS_TABLE} >/dev/null 2>&1; then
    echo "Creating table ${EVALUATIONS_TABLE}"
    bq --project_id=${PROJECT_ID} mk --table ${DATASET}.${EVALUATIONS_TABLE} \
        evaluation_id:STRING,session_id:STRING,metric_name:STRING,mean_score:FLOAT64,num_cases_total:INT64,num_cases_valid:INT64,num_cases_error:INT64,created_at:TIMESTAMP,summary_json:JSON
fi

echo "Vertex AI location set to: ${VERTEX_LOCATION}"

# Build and deploy options
echo -e "${YELLOW}Deployment options:${NC}"
echo "1) Deploy using Cloud Build (recommended for CI/CD)"
echo "2) Build locally and deploy"
echo "3) Deploy pre-built image"
read -p "Select option (1-3): " DEPLOY_OPTION

case $DEPLOY_OPTION in
    1)
        echo -e "${GREEN}Deploying using Cloud Build...${NC}"
        gcloud builds submit \
            --config=${CLOUDBUILD_CONFIG} \
            --substitutions=_REGION=${REGION},_SERVICE_ACCOUNT=${SERVICE_ACCOUNT_EMAIL}
        ;;

    2)
        echo -e "${GREEN}Building locally...${NC}"

        # Build the Docker image
        IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
        docker build -t ${IMAGE_NAME} .

        echo "Pushing image to Container Registry..."
        docker push ${IMAGE_NAME}

        echo "Deploying to Cloud Run..."
        gcloud run deploy ${SERVICE_NAME} \
            --image ${IMAGE_NAME} \
            --region ${REGION} \
            --platform managed \
            --allow-unauthenticated \
            --port 8080 \
            --memory 256Mi \
            --cpu 1 \
            --min-instances 0 \
            --max-instances 10 \
            --timeout 60 \
            --concurrency 100 \
            --service-account ${SERVICE_ACCOUNT_EMAIL} \
            --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_REGION=${REGION},BQ_DEFAULT_DATASET=${DATASET},AGENT_CONVERSATIONS_TABLE=${INTERACTIONS_TABLE},AGENT_EVALUATIONS_TABLE=${EVALUATIONS_TABLE},VERTEX_AI_LOCATION=${VERTEX_LOCATION},AGENT_EVALUATION_CANDIDATE_NAME=${EVAL_CANDIDATE}
        ;;

    3)
        echo -e "${YELLOW}Enter the image URL (e.g., gcr.io/project/image:tag):${NC}"
        read -p "Image URL: " IMAGE_URL

        echo -e "${GREEN}Deploying pre-built image...${NC}"
        gcloud run deploy ${SERVICE_NAME} \
            --image ${IMAGE_URL} \
            --region ${REGION} \
            --platform managed \
            --allow-unauthenticated \
            --port 8080 \
            --memory 256Mi \
            --cpu 1 \
            --min-instances 0 \
            --max-instances 10 \
            --timeout 60 \
            --concurrency 100 \
            --service-account ${SERVICE_ACCOUNT_EMAIL} \
            --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_REGION=${REGION},BQ_DEFAULT_DATASET=${DATASET},AGENT_CONVERSATIONS_TABLE=${INTERACTIONS_TABLE},AGENT_EVALUATIONS_TABLE=${EVALUATIONS_TABLE},VERTEX_AI_LOCATION=${VERTEX_LOCATION},AGENT_EVALUATION_CANDIDATE_NAME=${EVAL_CANDIDATE}
        ;;

    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

# Get the service URL
echo ""
echo -e "${GREEN}Deployment complete!${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo -e "Service URL: ${GREEN}${SERVICE_URL}${NC}"
echo ""

# Test the deployment
echo -e "${YELLOW}Testing deployment...${NC}"
curl -s ${SERVICE_URL}/health | python3 -m json.tool
echo ""

# Display useful commands
echo -e "${GREEN}=== Useful Commands ===${NC}"
echo "View logs:"
echo "  gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "Update traffic:"
echo "  gcloud run services update-traffic ${SERVICE_NAME} --region ${REGION} --to-latest"
echo ""
echo "Delete service:"
echo "  gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "Test endpoints:"
echo "  curl ${SERVICE_URL}/health"
echo "  curl ${SERVICE_URL}/api/firewall/rules"
echo "  curl ${SERVICE_URL}/api/iam/service-accounts"
echo ""

echo -e "${GREEN}Deployment script completed successfully!${NC}"
