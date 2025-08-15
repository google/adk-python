#!/bin/bash
# Deploy GCP Security Agent locally for testing

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"mgm-digitalconcierge"}
REGION=${REGION:-"us-central1"}
PORT=${PORT:-8000}

echo "🚀 Starting GCP Security Agent Local Deployment"
echo "Project: ${PROJECT_ID}"
echo "Port: ${PORT}"
echo ""

# Check for required files
echo "📋 Checking prerequisites..."

if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "Please create a .env file with your GCP credentials"
    exit 1
fi

if [ ! -f "backend/requirements.txt" ]; then
    echo "❌ Error: backend/requirements.txt not found"
    exit 1
fi

# Setup Python virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created new virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Export environment variables
echo "🔧 Loading environment variables..."
export $(cat .env | grep -v '^#' | xargs)
export GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run tests
echo "🧪 Running tests..."
echo "Testing GCP connectivity..."
python3 -c "
import os
from google.cloud import storage
project = os.getenv('GOOGLE_CLOUD_PROJECT')
print(f'✅ Connected to GCP project: {project}')
"

# Test asset inventory
echo "Testing Asset Inventory..."
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from backend.services.gcp_thin_client_service import GCPThinClientService

async def test():
    service = GCPThinClientService('${PROJECT_ID}')
    snapshot = await service.get_asset_inventory_snapshot()
    print(f'✅ Found {snapshot.total_assets} assets')
    for asset_type, count in snapshot.asset_breakdown.items():
        if count > 0:
            print(f'  - {asset_type}: {count}')

asyncio.run(test())
"

# Start the backend server
echo ""
echo "🚀 Starting backend server on port ${PORT}..."
echo "Access the API at: http://localhost:${PORT}"
echo "API Documentation: http://localhost:${PORT}/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --reload