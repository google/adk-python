#!/bin/bash

echo "🧹 Clearing all caches and restarting frontend..."
echo "================================================"

# Kill any running Streamlit processes
echo "🛑 Stopping any running Streamlit processes..."
pkill -f "streamlit" 2>/dev/null || echo "No Streamlit processes found"

# Clear Python cache
echo "🐍 Clearing Python cache files..."
find . -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true

# Clear Streamlit cache 
echo "📊 Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache* 2>/dev/null || echo "No Streamlit cache found"
rm -rf .streamlit/cache* 2>/dev/null || echo "No local Streamlit cache found"

# Clear browser cache instruction
echo "🌐 IMPORTANT: Please also clear your browser cache:"
echo "   • Press Ctrl+Shift+R (or Cmd+Shift+R on Mac) to hard refresh"
echo "   • Or open browser dev tools (F12) and right-click refresh button → Empty Cache and Hard Reload"

# Wait a moment
sleep 2

echo ""
echo "✅ Cache cleared! Starting frontend with new dashboard integration..."
echo "🚀 The executive dashboard is now integrated into the front page!"
echo ""

# Start frontend
echo "Starting frontend on http://localhost:8501..."
python run_frontend.py

echo ""
echo "🎉 Frontend started with integrated executive dashboard!"
echo "📊 Key features now on front page:"
echo "   • Security Posture Overview (4 KPIs)"
echo "   • Interactive Security Analytics (charts)"
echo "   • Risk Assessment (Storage/Network/IAM)"
echo "   • Quick Actions (immediate access buttons)"