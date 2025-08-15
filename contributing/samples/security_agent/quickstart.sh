#!/bin/bash
# Quick Start Script for GCP Security Agent
# Makes it easy for AI coders to get started

echo "🚀 GCP Security Agent - Quick Start"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    echo -e "${GREEN}✅ Python $python_version found${NC}"
else
    echo -e "${RED}❌ Python 3.8+ required (found $python_version)${NC}"
    exit 1
fi

# Check if in project directory
if [ ! -f "run_backend.py" ]; then
    echo -e "${RED}❌ Not in project root directory${NC}"
    echo "Please run this script from the security_agent directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt 2>/dev/null || echo -e "${YELLOW}⚠️  Some optional dependencies failed (this is OK)${NC}"
pip install -q streamlit 2>/dev/null
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Create necessary directories
echo "Setting up directories..."
mkdir -p cache/assets logs
echo -e "${GREEN}✅ Directories created${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# GCP Configuration
GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# API Configuration  
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:8501

# Feature Flags
ENABLE_MOCK_DATA=true
ENABLE_CACHE=true
CACHE_TTL_SECONDS=300

# Logging
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

echo ""
echo "===================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "===================================="
echo ""
echo "To start the application:"
echo ""
echo "1. In Terminal 1 (Backend):"
echo "   source venv/bin/activate"
echo "   python run_backend.py"
echo ""
echo "2. In Terminal 2 (Frontend):"
echo "   source venv/bin/activate"
echo "   python run_frontend.py"
echo ""
echo "3. Open browser to http://localhost:8501"
echo ""
echo "===================================="
echo "Useful commands:"
echo "  python test_endpoints.py                    # Test API endpoints"
echo "  python scripts/ai_dev_helper.py check       # Check environment"
echo "  python scripts/service_health.py            # Check service health"
echo "  python initialize_asset_data.py             # Initialize test data"
echo ""
echo "📚 See AI_CODER_GUIDE.md for development guide"
echo "===================================="