#!/bin/bash

# Simple script to run the ADK agent Streamlit app

echo "🎯 Starting ADK Agent Interface..."
echo "=================================="

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend is not running. Starting it in background..."
    echo "   Run this in another terminal: python backend/main.py"
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "=================================="

# Run the simple agent app
streamlit run frontend/simple_agent_app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false

echo "✨ App stopped"