#!/bin/bash
# Stop all GCP Security Intelligence Platform services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Stopping Security Agent Services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to stop services on specific port
stop_port() {
    local port=$1
    local name=$2

    pids=$(lsof -ti:$port 2>/dev/null)
    if [ -z "$pids" ]; then
        echo -e "${YELLOW}⚠️  ${name} not running on port ${port}${NC}"
    else
        echo -e "${BLUE}Stopping ${name} (port ${port})...${NC}"
        kill $pids 2>/dev/null
        sleep 1

        # Force kill if still running
        pids=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$pids" ]; then
            echo -e "${YELLOW}  Force killing...${NC}"
            kill -9 $pids 2>/dev/null
        fi
        echo -e "${GREEN}✓${NC} ${name} stopped"
    fi
}

# Stop ADK Backend
stop_port 8000 "ADK Backend"

# Stop Flask UI
stop_port 5001 "Flask UI"

# Stop Chainlit UI
stop_port 8001 "Chainlit UI"

echo ""
echo -e "${GREEN}✓ All services stopped${NC}"
echo ""
