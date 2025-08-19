#!/usr/bin/env python3
"""
Launcher for the minimal thin client frontend.
"""

import subprocess
import sys
import os

def main():
    """Run the minimal Streamlit frontend."""
    print("🚀 Starting Minimal ADK Security Agent Frontend...")
    print("=" * 50)
    
    # Set working directory to frontend
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    os.chdir(frontend_dir)
    
    # Run Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "main_app.py", 
           "--server.port", "8501",
           "--server.address", "localhost"]
    
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()