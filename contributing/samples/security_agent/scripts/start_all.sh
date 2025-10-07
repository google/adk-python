#!/bin/bash
# Unified startup script for GCP Security Intelligence Platform
# Starts all available interfaces for the security agent

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADK_PORT=8000
FLASK_PORT=5001
CHAINLIT_PORT=8001

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GCP Security Intelligence Platform${NC}"
echo -e "${BLUE}  Starting All Services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo -e "${YELLOW}Please copy .env.example to .env and configure it${NC}"
    exit 1
fi

# Load environment variables
source "$PROJECT_DIR/.env"

# Verify required environment variables
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo -e "${RED}❌ Error: GOOGLE_CLOUD_PROJECT not set in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment loaded: Project ${YELLOW}$GOOGLE_CLOUD_PROJECT${NC}"
echo ""

# Run dependency check
echo -e "${BLUE}=== Checking Dependencies ===${NC}"
echo ""

# Quick check of critical modules only
echo -e "${BLUE}Validating critical dependencies...${NC}"

missing_deps=()

# Check Flask
if ! python3 -c "import flask" 2>/dev/null; then
    missing_deps+=("flask")
fi

# Check Google Cloud AI Platform
if ! python3 -c "import google.cloud.aiplatform" 2>/dev/null; then
    missing_deps+=("google-cloud-aiplatform")
fi

# Check Requests
if ! python3 -c "import requests" 2>/dev/null; then
    missing_deps+=("requests")
fi

# Check dotenv
if ! python3 -c "import dotenv" 2>/dev/null; then
    missing_deps+=("python-dotenv")
fi

if [ ${#missing_deps[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All critical dependencies available"
    echo -e "  ${BLUE}→${NC} Run 'python3 tests/test_dependencies.py' for full validation"
else
    echo -e "${RED}❌ Missing critical dependencies: ${missing_deps[*]}${NC}"
    echo -e "  ${BLUE}→${NC} Install with: pip install ${missing_deps[*]}"
    echo -e "  ${BLUE}→${NC} Or run: pip install -r requirements.txt"
    exit 1
fi
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start service in background
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="$PROJECT_DIR/logs/${name}.log"

    mkdir -p "$PROJECT_DIR/logs"

    if check_port $port; then
        echo -e "${YELLOW}⚠️  ${name} already running on port ${port}${NC}"
    else
        echo -e "${BLUE}Starting ${name}...${NC}"
        cd "$PROJECT_DIR"
        eval "$command" > "$log_file" 2>&1 &
        local pid=$!
        sleep 2

        if ps -p $pid > /dev/null; then
            echo -e "${GREEN}✓${NC} ${name} started (PID: ${pid}, Port: ${port})"
            echo -e "  ${BLUE}→${NC} Logs: ${log_file}"
        else
            echo -e "${RED}❌ Failed to start ${name}${NC}"
            echo -e "  ${BLUE}→${NC} Check logs: ${log_file}"
        fi
    fi
    echo ""
}

# Start ADK Backend (required for all interfaces)
echo -e "${BLUE}=== Starting Core Services ===${NC}"
echo ""
start_service "ADK Backend" "adk web" $ADK_PORT

# Start Flask UI
echo -e "${BLUE}=== Starting Web Interfaces ===${NC}"
echo ""
start_service "Flask UI" "python3 app.py --port=$FLASK_PORT" $FLASK_PORT

# Start Chainlit UI (optional)
if command -v chainlit &> /dev/null; then
    start_service "Chainlit UI" "chainlit run chainlit_app.py --port=$CHAINLIT_PORT" $CHAINLIT_PORT
else
    echo -e "${YELLOW}⚠️  Chainlit not installed (optional)${NC}"
    echo -e "  ${BLUE}→${NC} Install with: pip install chainlit==1.0.0"
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Services Started Successfully${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Available Interfaces:${NC}"
echo -e "  ${GREEN}•${NC} ADK Backend:    http://localhost:${ADK_PORT}"
echo -e "  ${GREEN}•${NC} Flask UI:       http://localhost:${FLASK_PORT}"
if command -v chainlit &> /dev/null && ! check_port $CHAINLIT_PORT; then
    echo -e "  ${GREEN}•${NC} Chainlit UI:    http://localhost:${CHAINLIT_PORT}"
fi
echo ""
echo -e "${BLUE}Management:${NC}"
echo -e "  ${GREEN}•${NC} View logs:      tail -f logs/*.log"
echo -e "  ${GREEN}•${NC} Stop services:  ./scripts/stop_all.sh"
echo -e "  ${GREEN}•${NC} Check status:   ps aux | grep -E 'adk|python3.*app.py|chainlit'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and handle graceful shutdown
trap 'echo -e "\n${BLUE}Stopping all services...${NC}"; pkill -P $$; exit 0' SIGINT SIGTERM

# Wait for any background processes
wait
