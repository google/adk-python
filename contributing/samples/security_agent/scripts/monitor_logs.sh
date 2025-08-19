#!/bin/bash
# Real-time log monitoring script for Security Agent debugging

echo "🔍 Security Agent Log Monitor"
echo "=============================="
echo ""
echo "Monitoring logs in real-time..."
echo "Press Ctrl+C to stop"
echo ""
echo "Backend: logs/backend.log"
echo "Frontend: logs/frontend.log"
echo ""
echo "==============================="

# Monitor both log files simultaneously
tail -f logs/backend.log logs/frontend.log