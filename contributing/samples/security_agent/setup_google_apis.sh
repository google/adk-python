#!/bin/bash

# Setup Google Cloud APIs and Vertex AI for Security Agent
# ==========================================================

echo "🚀 Setting up Google Cloud APIs for Security Agent"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project from .env
PROJECT_ID="mgm-digitalconcierge"
echo "📋 Project ID: $PROJECT_ID"

# Set the project
echo "🔧 Setting active project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo ""
echo "🔌 Enabling required Google Cloud APIs..."
echo "This may take a few minutes..."

APIS=(
    "aiplatform.googleapis.com"           # Vertex AI
    "compute.googleapis.com"               # Compute Engine
    "storage.googleapis.com"               # Cloud Storage
    "iam.googleapis.com"                   # IAM
    "cloudresourcemanager.googleapis.com"  # Resource Manager
    "cloudasset.googleapis.com"            # Cloud Asset Inventory
    "recommender.googleapis.com"           # Recommender
    "secretmanager.googleapis.com"         # Secret Manager
    "monitoring.googleapis.com"            # Cloud Monitoring
    "logging.googleapis.com"               # Cloud Logging
    "securitycenter.googleapis.com"        # Security Command Center (optional)
)

for api in "${APIS[@]}"; do
    echo "  Enabling $api..."
    gcloud services enable $api --project=$PROJECT_ID 2>/dev/null || echo "    ⚠️ Could not enable $api (may require billing or permissions)"
done

echo ""
echo "✅ API enablement complete!"

# Check service account
echo ""
echo "🔐 Checking service account..."
SERVICE_ACCOUNT_FILE="/path/to/your/service-account-key.json"

if [ -f "$SERVICE_ACCOUNT_FILE" ]; then
    echo "✅ Service account file found: $SERVICE_ACCOUNT_FILE"
    
    # Extract service account email
    SA_EMAIL=$(python3 -c "import json; print(json.load(open('$SERVICE_ACCOUNT_FILE'))['client_email'])" 2>/dev/null)
    if [ ! -z "$SA_EMAIL" ]; then
        echo "📧 Service account: $SA_EMAIL"
    fi
else
    echo "⚠️ Service account file not found at: $SERVICE_ACCOUNT_FILE"
    echo "   Please ensure your service account JSON is at this location"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install google-cloud-aiplatform google-cloud-storage google-cloud-iam google-cloud-asset google-cloud-recommender google-cloud-secretmanager

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Ensure your service account has the following roles:"
echo "   - Vertex AI User"
echo "   - Security Reviewer (or Security Admin)"
echo "   - Asset Viewer"
echo "   - IAM Reviewer"
echo "   - Storage Object Viewer"
echo ""
echo "2. Test the agent with: python agent_vertex.py"
echo ""
echo "3. If you get permission errors, grant roles with:"
echo "   gcloud projects add-iam-policy-binding $PROJECT_ID \\"
echo "     --member=\"serviceAccount:\$SA_EMAIL\" \\"
echo "     --role=\"roles/aiplatform.user\""