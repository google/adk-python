#!/bin/bash
# Quick GCP Connection Check

echo "🔍 Quick GCP Connection Check"
echo "=============================="

# Check authentication
echo -e "\n1. Authentication:"
if gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q '@'; then
    echo "✅ Authenticated as: $(gcloud config get-value account)"
else
    echo "❌ NOT AUTHENTICATED - Run: gcloud auth application-default login"
fi

# Check project
echo -e "\n2. Project:"
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -n "$PROJECT" ]; then
    echo "✅ Project: $PROJECT"
else
    echo "❌ NO PROJECT SET - Run: gcloud config set project YOUR_PROJECT_ID"
fi

# Check required APIs
echo -e "\n3. Required APIs:"
for api in cloudasset.googleapis.com cloudresourcemanager.googleapis.com compute.googleapis.com storage-api.googleapis.com; do
    if gcloud services list --enabled 2>/dev/null | grep -q $api; then
        echo "✅ $api"
    else
        echo "❌ $api - Run: gcloud services enable $api"
    fi
done

# Test Asset API
echo -e "\n4. Testing Asset API:"
if [ -n "$PROJECT" ]; then
    RESPONSE=$(curl -s -X GET \
        -H "Authorization: Bearer $(gcloud auth print-access-token 2>/dev/null)" \
        "https://cloudasset.googleapis.com/v1/projects/$PROJECT:searchAllResources?pageSize=1" 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q '"results"'; then
        echo "✅ Asset API is working!"
    elif echo "$RESPONSE" | grep -q 'PERMISSION_DENIED'; then
        echo "❌ Permission denied - Grant cloudasset.viewer role"
    elif echo "$RESPONSE" | grep -q 'has not been used'; then
        echo "❌ API not enabled - Run: gcloud services enable cloudasset.googleapis.com"
    else
        echo "⚠️  Could not test API"
    fi
fi

echo -e "\n=============================="
echo "If all checks pass ✅, run:"
echo "  python test_endpoints.py"
echo ""
echo "For detailed diagnosis, run:"
echo "  python diagnose_connection.py"
echo "=============================="