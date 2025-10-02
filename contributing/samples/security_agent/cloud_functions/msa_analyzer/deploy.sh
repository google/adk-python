#!/bin/bash

# Deploy MSA Analyzer Cloud Function
# Usage: ./deploy.sh [project-id] [region]

PROJECT_ID=${1:-${GOOGLE_CLOUD_PROJECT}}
REGION=${2:-us-central1}
FUNCTION_NAME="msa-analyzer"
MEMORY="512MB"
TIMEOUT="300s"
SOURCE_DIR="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MSA Analyzer Cloud Function Deployment ===${NC}"
echo ""

# Check if project ID is provided
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: Project ID not provided${NC}"
    echo "Usage: $0 <project-id> [region]"
    exit 1
fi

# Set the project
echo -e "${YELLOW}Setting project to: ${PROJECT_ID}${NC}"
gcloud config set project $PROJECT_ID

# Copy the MSA analyzer module
echo -e "${YELLOW}Copying MSA analyzer module...${NC}"
cp ../../agents/_tools/msa_analyzer.py .

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    pubsub.googleapis.com \
    --project=$PROJECT_ID

# Create Pub/Sub topic for critical alerts (if it doesn't exist)
echo -e "${YELLOW}Creating Pub/Sub topic for alerts...${NC}"
gcloud pubsub topics create msa-critical-alerts \
    --project=$PROJECT_ID 2>/dev/null || echo "Topic already exists"

# Deploy the function
echo -e "${YELLOW}Deploying Cloud Function...${NC}"
gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=$SOURCE_DIR \
    --entry-point=analyze_releases \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --set-env-vars="GCP_PROJECT=$PROJECT_ID" \
    --project=$PROJECT_ID

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Cloud Function deployed successfully!${NC}"

    # Get the function URL
    FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
        --region=$REGION \
        --format="value(serviceConfig.uri)" \
        --project=$PROJECT_ID)

    echo ""
    echo -e "${GREEN}Function URL: ${FUNCTION_URL}${NC}"

    # Create Cloud Scheduler job
    echo ""
    echo -e "${YELLOW}Would you like to create a Cloud Scheduler job? (y/n)${NC}"
    read -r CREATE_SCHEDULER

    if [ "$CREATE_SCHEDULER" = "y" ] || [ "$CREATE_SCHEDULER" = "Y" ]; then
        JOB_NAME="msa-analyzer-daily"
        SCHEDULE="0 9 * * *"  # Daily at 9 AM

        echo -e "${YELLOW}Creating Cloud Scheduler job...${NC}"

        # Create service account for scheduler
        SA_NAME="msa-scheduler-sa"
        SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

        gcloud iam service-accounts create $SA_NAME \
            --display-name="MSA Scheduler Service Account" \
            --project=$PROJECT_ID 2>/dev/null || echo "Service account already exists"

        # Grant function invoker role
        gcloud functions add-iam-policy-binding $FUNCTION_NAME \
            --region=$REGION \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="roles/cloudfunctions.invoker" \
            --project=$PROJECT_ID

        # Create scheduler job
        gcloud scheduler jobs create http $JOB_NAME \
            --location=$REGION \
            --schedule="$SCHEDULE" \
            --time-zone="America/New_York" \
            --uri=$FUNCTION_URL \
            --http-method=POST \
            --headers="Content-Type=application/json" \
            --message-body='{"days_back": 7}' \
            --oidc-service-account-email=$SA_EMAIL \
            --project=$PROJECT_ID

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Cloud Scheduler job created!${NC}"
            echo "   Schedule: Daily at 9 AM ET"
            echo "   Job name: $JOB_NAME"
        fi
    fi

    echo ""
    echo -e "${GREEN}=== Deployment Complete ===${NC}"
    echo ""
    echo "Test the function with:"
    echo "curl -X POST $FUNCTION_URL \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"days_back\": 7}'"
    echo ""
    echo "View logs with:"
    echo "gcloud functions logs read $FUNCTION_NAME --region=$REGION"

else
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

# Clean up copied file
rm -f msa_analyzer.py