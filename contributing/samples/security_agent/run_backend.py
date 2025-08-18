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
    
    # Load environment variables from .env if it exists
    from pathlib import Path
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Loaded environment from: {env_file}")
    
    # Set environment variables with defaults
    if not os.environ.get('GOOGLE_CLOUD_PROJECT'):
        # Default to a placeholder - user should set their own
        os.environ['GOOGLE_CLOUD_PROJECT'] = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')
        if os.environ['GOOGLE_CLOUD_PROJECT'] == 'your-project-id':
            print("⚠️ GOOGLE_CLOUD_PROJECT not set. Please set it in .env file")
    
    # Set Google Application Credentials if not already set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        backend_dir = os.path.join(os.path.dirname(__file__), "backend")
        
        # Try to find any service account JSON file in the secrets directory
        secrets_dir = os.path.join(backend_dir, "config", "secrets")
        if os.path.exists(secrets_dir):
            import glob
            json_files = glob.glob(os.path.join(secrets_dir, "*.json"))
            if json_files:
                # Use the first JSON file found
                service_account_path = json_files[0]
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
                print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS to: {service_account_path}")
            else:
                print("⚠️ No service account JSON files found in config/secrets/")
        else:
            print("⚠️ Service account directory not found, will use default credentials")
    
    # Change to backend directory
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