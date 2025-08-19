#!/bin/bash
# Standard startup script for Security Agent
# This is the ONLY way to start the services

echo "🚀 Starting Security Agent Services"
echo "===================================="

# Activate virtual environment
source venv/bin/activate

# Clear old logs
> logs/backend.log
> logs/frontend.log

# Start backend
echo "Starting backend on port 8000..."
python run_backend.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend on port 8501..."
python run_frontend.py > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

echo ""
echo "✅ Services Started!"
echo "===================================="
echo "📡 Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "🖥️  Frontend: http://localhost:8501 (PID: $FRONTEND_PID)"
echo "📊 API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Monitor logs with: ./monitor_logs.sh"
echo "🧪 Test config with: python test_agent_config.py"
echo ""
echo "To stop services: pkill -f 'uvicorn|streamlit'"