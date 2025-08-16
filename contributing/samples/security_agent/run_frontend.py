#!/usr/bin/env python3
"""
Run the frontend Streamlit app for the Security Agent.
"""

import subprocess
import sys
import os

def main():
    """Start the Streamlit frontend."""
    print("🚀 Starting Security Agent Frontend (Thin Client)...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['BACKEND_URL'] = 'http://localhost:8000'
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    os.chdir(frontend_dir)
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "main_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]
    
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🔗 Backend URL: {os.environ.get('BACKEND_URL')}")
    print("=" * 50)
    print("Frontend will be available at: http://localhost:8501")
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