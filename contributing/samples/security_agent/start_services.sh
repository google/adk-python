#!/bin/bash

echo "========================================"
echo "Starting Security Agent Services"
echo "========================================"

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "streamlit" 2>/dev/null
pkill -f "run_frontend" 2>/dev/null
sleep 2

# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit in background (headless mode)
echo "Starting Streamlit frontend (headless mode on port 8501)..."
nohup streamlit run frontend/app.py \
  --server.headless=true \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --browser.gatherUsageStats=false \
  > logs/streamlit.log 2>&1 &

echo "Streamlit PID: $!"

# Wait a moment for services to start
sleep 3

# Check status
echo ""
echo "Checking service status..."
python monitoring/simple_monitoring.py

echo ""
echo "========================================"
echo "Services started!"
echo "========================================"
echo "ADK Backend: http://localhost:8000"
echo "Flask API: http://localhost:5000"
echo "Streamlit Frontend: http://localhost:8501"
echo ""
echo "Logs:"
echo "  Streamlit: logs/streamlit.log"
echo ""
echo "To monitor: python monitoring/simple_monitoring.py --loop"
echo "To stop Streamlit: pkill -f streamlit"