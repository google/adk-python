#!/bin/bash

# Setup script for Google Cloud credentials
echo "🔐 Google Cloud Credentials Setup"
echo "================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "⚠️  You are not authenticated with gcloud."
    echo "   Running: gcloud auth login"
    gcloud auth login
fi

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)

if [ -z "$CURRENT_PROJECT" ]; then
    echo "⚠️  No default project set."
    echo "   Available projects:"
    gcloud projects list --format="table(projectId,name)"
    echo ""
    read -p "Enter your project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
    CURRENT_PROJECT="$PROJECT_ID"
fi

echo "📁 Current project: $CURRENT_PROJECT"
echo ""

# Options for setting up credentials
echo "Choose how to set up credentials:"
echo "1) Use Application Default Credentials (recommended for development)"
echo "2) Create a new service account key"
echo "3) Use an existing service account key file"
echo "4) Skip (I'll set it up manually)"
echo ""
read -p "Enter your choice (1-4): " CHOICE

case $CHOICE in
    1)
        echo "Setting up Application Default Credentials..."
        gcloud auth application-default login
        echo ""
        echo "✅ ADC configured. No service account key file needed."
        echo ""
        echo "Updating .env file..."
        sed -i.bak "s/^GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT/" .env 2>/dev/null || \
        echo "GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT" >> .env
        sed -i.bak "/^GOOGLE_APPLICATION_CREDENTIALS=/d" .env 2>/dev/null
        echo "✅ .env updated to use ADC"
        ;;
    
    2)
        echo "Creating a new service account..."
        SERVICE_ACCOUNT_NAME="security-agent-sa"
        SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${CURRENT_PROJECT}.iam.gserviceaccount.com"
        
        # Create service account
        gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
            --display-name="Security Agent Service Account" \
            --project="$CURRENT_PROJECT" 2>/dev/null || echo "Service account may already exist"
        
        # Grant necessary roles
        echo "Granting necessary roles..."
        ROLES=(
            "roles/securitycenter.admin"
            "roles/cloudasset.viewer"
            "roles/iam.securityReviewer"
            "roles/storage.admin"
            "roles/monitoring.viewer"
        )
        
        for ROLE in "${ROLES[@]}"; do
            gcloud projects add-iam-policy-binding "$CURRENT_PROJECT" \
                --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
                --role="$ROLE" \
                --quiet 2>/dev/null
        done
        
        # Create key
        KEY_PATH="backend/config/secrets/service-account-key.json"
        mkdir -p backend/config/secrets
        gcloud iam service-accounts keys create "$KEY_PATH" \
            --iam-account="$SERVICE_ACCOUNT_EMAIL" \
            --project="$CURRENT_PROJECT"
        
        echo ""
        echo "✅ Service account created and key saved to $KEY_PATH"
        echo ""
        echo "Updating .env file..."
        sed -i.bak "s/^GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT/" .env 2>/dev/null || \
        echo "GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT" >> .env
        sed -i.bak "s|^GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$KEY_PATH|" .env 2>/dev/null || \
        echo "GOOGLE_APPLICATION_CREDENTIALS=$KEY_PATH" >> .env
        echo "✅ .env updated"
        ;;
    
    3)
        echo ""
        read -p "Enter the path to your service account key file: " KEY_PATH
        if [ -f "$KEY_PATH" ]; then
            # Copy to secrets directory
            DEST_PATH="backend/config/secrets/service-account-key.json"
            mkdir -p backend/config/secrets
            cp "$KEY_PATH" "$DEST_PATH"
            echo "✅ Key file copied to $DEST_PATH"
            echo ""
            echo "Updating .env file..."
            sed -i.bak "s/^GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT/" .env 2>/dev/null || \
            echo "GOOGLE_CLOUD_PROJECT=$CURRENT_PROJECT" >> .env
            sed -i.bak "s|^GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$DEST_PATH|" .env 2>/dev/null || \
            echo "GOOGLE_APPLICATION_CREDENTIALS=$DEST_PATH" >> .env
            echo "✅ .env updated"
        else
            echo "❌ File not found: $KEY_PATH"
            exit 1
        fi
        ;;
    
    4)
        echo "Skipping credential setup."
        echo ""
        echo "📝 Manual setup instructions:"
        echo "1. Set GOOGLE_CLOUD_PROJECT in .env"
        echo "2. Either:"
        echo "   a) Set GOOGLE_APPLICATION_CREDENTIALS to your key file path"
        echo "   b) Use 'gcloud auth application-default login'"
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Start the backend: python run_backend.py"
echo "3. Start the frontend: python run_frontend.py"