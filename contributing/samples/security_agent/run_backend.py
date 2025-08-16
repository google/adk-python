#!/usr/bin/env python3
"""
Run the backend FastAPI server for the Security Agent.
"""

import subprocess
import sys
import os

def main():
    """Start the FastAPI backend server."""
    print("🚀 Starting Security Agent Backend...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'mgm-digitalconcierge'
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    os.chdir(backend_dir)
    
    # Run uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🌍 Project: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    print("=" * 50)
    print("Backend will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Backend stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()