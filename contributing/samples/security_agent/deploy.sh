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
SERVICE_ACCOUNT="${SERVICE_NAME}-sa"

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

# Set the project
echo "Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Get current user for authentication check
CURRENT_USER=$(gcloud config get-value account 2>/dev/null)
echo "Authenticated as: $CURRENT_USER"
echo ""

# Create service account if it doesn't exist
echo -e "${YELLOW}Setting up service account...${NC}"
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    echo "Creating service account: ${SERVICE_ACCOUNT}"
    gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
        --display-name="Security Agent API Service Account"

    # Grant necessary permissions
    echo "Granting permissions..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/cloudfunctions.invoker"

    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/logging.logWriter"
else
    echo "Service account already exists: ${SERVICE_ACCOUNT}"
fi
echo ""

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
            --config=cloudbuild.yaml \
            --substitutions=_REGION=${REGION},_SERVICE_ACCOUNT=${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com
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
            --service-account ${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
            --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_REGION=${REGION}
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
            --service-account ${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
            --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_REGION=${REGION}
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