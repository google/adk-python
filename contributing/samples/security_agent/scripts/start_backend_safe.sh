#!/bin/bash

# Safe Backend Startup Script
# Kills any existing backend processes before starting new one

echo "🛡️ Safe Backend Startup Script"
echo "=============================================="

# Kill any existing processes on port 8000
echo "🔍 Checking for existing processes on port 8000..."
PIDS=$(lsof -ti:8000)

if [ ! -z "$PIDS" ]; then
    echo "⚠️ Found existing processes on port 8000: $PIDS"
    echo "🛑 Killing existing processes..."
    kill -9 $PIDS 2>/dev/null
    sleep 2
    echo "✅ Existing processes terminated"
else
    echo "✅ Port 8000 is free"
fi

# Also kill any existing uvicorn processes
echo "🔍 Checking for existing uvicorn processes..."
UVICORN_PIDS=$(pgrep -f "uvicorn main:app")

if [ ! -z "$UVICORN_PIDS" ]; then
    echo "⚠️ Found existing uvicorn processes: $UVICORN_PIDS"
    echo "🛑 Killing uvicorn processes..."
    kill -9 $UVICORN_PIDS 2>/dev/null
    sleep 1
    echo "✅ Uvicorn processes terminated"
else
    echo "✅ No existing uvicorn processes"
fi

# Now start the backend
echo ""
echo "🚀 Starting Backend Server..."
echo "=============================================="

# Change to backend directory
cd "$(dirname "$0")/../backend" || exit 1

# Start the backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000