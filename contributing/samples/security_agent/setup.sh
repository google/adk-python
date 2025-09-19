#!/bin/bash

# ADK Security Agent - Clean Setup Script
# Ensures ONLY ONE instance of each service runs

set -e  # Exit on error

echo "🧹 ADK Security Agent - Clean Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to kill all existing processes
cleanup_processes() {
    echo -e "${YELLOW}🔄 Cleaning up existing processes...${NC}"

    # Kill all ADK web processes
    pkill -f "adk web" 2>/dev/null || true

    # Kill all Python frontend/backend processes
    pkill -f "python.*run_frontend" 2>/dev/null || true
    pkill -f "python.*run_backend" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "streamlit" 2>/dev/null || true

    # Wait for processes to die
    sleep 3

    # Force kill any remaining processes
    ps aux | grep -E "(adk|streamlit|uvicorn)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

    # Verify ports are free
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:8501 | xargs kill -9 2>/dev/null || true

    sleep 2
    echo -e "${GREEN}✅ All processes cleaned up${NC}"
}

# Function to setup environment
setup_environment() {
    echo -e "${YELLOW}📦 Setting up environment...${NC}"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip --quiet

    # Install requirements
    if [ -f "requirements.txt" ]; then
        echo "Installing backend requirements..."
        pip install -r requirements.txt --quiet
    fi

    if [ -f "requirements_frontend.txt" ]; then
        echo "Installing frontend requirements..."
        pip install -r requirements_frontend.txt --quiet
    fi

    # Install ADK
    echo "Installing ADK..."
    pip install google-genai google-genai[adk] --upgrade --quiet

    echo -e "${GREEN}✅ Environment ready${NC}"
}

# Function to setup configuration
setup_config() {
    echo -e "${YELLOW}⚙️ Setting up configuration...${NC}"

    # Check for .env file
    if [ ! -f ".env" ]; then
        echo -e "${RED}❌ .env file not found!${NC}"
        echo "Creating .env.example file..."
        cat > .env.example <<EOF
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account.json

# ADK Configuration
ADK_AGENT_MODEL=gemini-1.5-flash
GOOGLE_GENAI_USE_VERTEXAI=1

# Backend Configuration
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# Frontend Configuration
FRONTEND_PORT=8501

# Database Configuration
DATABASE_PATH=backend/cache/gcp_data.db
EOF
        echo -e "${YELLOW}Please create a .env file with your configuration${NC}"
        exit 1
    fi

    # Load environment variables
    export $(grep -v '^#' .env | xargs)

    # Verify service account file exists
    if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo -e "${RED}❌ Service account file not found: $GOOGLE_APPLICATION_CREDENTIALS${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Configuration loaded${NC}"
}

# Function to initialize database
setup_database() {
    echo -e "${YELLOW}🗄️ Setting up database...${NC}"

    # Create cache directory if it doesn't exist
    mkdir -p backend/cache

    # Run database population script
    if [ -f "utils/populate_sqlite.py" ]; then
        python utils/populate_sqlite.py
    elif [ -f "populate_sqlite.py" ]; then
        python populate_sqlite.py
    fi

    echo -e "${GREEN}✅ Database initialized${NC}"
}

# Function to start backend
start_backend() {
    echo -e "${YELLOW}🚀 Starting backend (ADK Web)...${NC}"

    # Start ADK web in background, capture PID
    python -m dotenv run -- adk web > backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > .backend.pid

    # Wait for backend to be ready
    echo -n "Waiting for backend"
    for i in {1..30}; do
        if curl -s http://localhost:8000/list-apps > /dev/null 2>&1; then
            echo -e "\n${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "\n${RED}❌ Backend failed to start${NC}"
    cat backend.log
    exit 1
}

# Function to start frontend
start_frontend() {
    echo -e "${YELLOW}🎨 Starting frontend (Streamlit)...${NC}"

    # Start frontend in background, capture PID
    python run_frontend.py > frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > .frontend.pid

    # Wait for frontend to be ready
    echo -n "Waiting for frontend"
    for i in {1..30}; do
        if curl -s -I http://localhost:8501 > /dev/null 2>&1; then
            echo -e "\n${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "\n${RED}❌ Frontend failed to start${NC}"
    cat frontend.log
    exit 1
}

# Function to show status
show_status() {
    echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
    echo "=================="
    echo -e "Backend:  ${GREEN}http://localhost:8000${NC}"
    echo -e "Frontend: ${GREEN}http://localhost:8501${NC}"
    echo -e "\n${YELLOW}To stop services:${NC} ./stop.sh"
    echo -e "${YELLOW}To check status:${NC} ./status.sh"
}

# Main execution
main() {
    echo "Starting clean setup..."

    # Always cleanup first
    cleanup_processes

    # Setup steps
    setup_environment
    setup_config
    setup_database

    # Start services
    start_backend
    start_frontend

    # Show status
    show_status
}

# Run main function
main