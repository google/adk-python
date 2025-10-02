#!/usr/bin/env python3
"""
Simplified monitoring setup - just tracks basic metrics
"""

import os
import time
import logging
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_services():
    """Check which services are running"""
    services = {
        "ADK Agent": ("http://localhost:8000/health", 8000),
        "Flask API": ("http://localhost:5000/health", 5000),
        "Streamlit": ("http://localhost:8501", 8501)
    }

    print("\n" + "="*60)
    print("🔍 SYSTEM STATUS CHECK")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print()

    running = []
    not_running = []

    for name, (url, port) in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code in [200, 404]:  # 404 is ok, means server is up
                print(f"✅ {name:20} Running on port {port}")
                running.append(name)
            else:
                print(f"⚠️  {name:20} Status {response.status_code}")
                not_running.append(name)
        except requests.exceptions.ConnectionError:
            print(f"❌ {name:20} Not running")
            not_running.append(name)
        except Exception as e:
            print(f"❌ {name:20} Error: {e}")
            not_running.append(name)

    print("\n" + "-"*60)
    print(f"Summary: {len(running)} services UP, {len(not_running)} services DOWN")

    if not_running:
        print(f"Not running: {', '.join(not_running)}")

    return len(running), len(not_running)

def monitor_loop():
    """Run monitoring in a loop"""
    print("Starting monitoring loop (press Ctrl+C to stop)")
    print("Checking every 30 seconds...")

    while True:
        try:
            check_services()
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        monitor_loop()
    else:
        check_services()
        print("\nTo run continuous monitoring: python simple_monitoring.py --loop")