#!/bin/bash

# Stop all ADK Security Agent services

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping ADK Security Agent services...${NC}"

# Kill processes using PID files if they exist
if [ -f ".backend.pid" ]; then
    PID=$(cat .backend.pid)
    if ps -p $PID > /dev/null; then
        kill $PID 2>/dev/null
        echo -e "${GREEN}✅ Backend stopped (PID: $PID)${NC}"
    fi
    rm .backend.pid
fi

if [ -f ".frontend.pid" ]; then
    PID=$(cat .frontend.pid)
    if ps -p $PID > /dev/null; then
        kill $PID 2>/dev/null
        echo -e "${GREEN}✅ Frontend stopped (PID: $PID)${NC}"
    fi
    rm .frontend.pid
fi

# Clean up any remaining processes
pkill -f "adk web" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

# Kill processes on ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo -e "${GREEN}✅ All services stopped${NC}"