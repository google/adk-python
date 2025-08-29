#!/usr/bin/env python3
"""
Run the backend FastAPI server for the Security Agent.
Supports both local development and Cloud Run deployments.

Usage:
    python run_backend.py          # Run locally
    python run_backend.py --cloud  # Deploy to Cloud Run
"""

import subprocess
import sys
import os
import argparse
import signal
import time
import socket

def get_service_account_email():
    """Extract service account email from the key file."""
    import json
    
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("[ERROR] GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        print("Please set GOOGLE_APPLICATION_CREDENTIALS in your .env file")
        sys.exit(1)
    
    # Convert relative path to absolute
    if not os.path.isabs(creds_path):
        creds_path = os.path.join(os.path.dirname(__file__), creds_path)
    
    if not os.path.exists(creds_path):
        print(f"[ERROR] Service account key file not found: {creds_path}")
        print("Please ensure the service account JSON file exists")
        sys.exit(1)
    
    try:
        with open(creds_path, 'r') as f:
            key_data = json.load(f)
            return key_data.get('client_email')
    except Exception as e:
        print(f"[ERROR] Failed to read service account key file: {e}")
        sys.exit(1)

def deploy_to_cloud(project_id):
    """Deploy backend to Cloud Run using Cloud Build."""
    print(f"[CLOUD] Deploying backend to Cloud Run (Project: {project_id})...")
    print("=" * 50)
    
    # Get service account email from key file
    service_account_email = get_service_account_email()
    if not service_account_email:
        print("[ERROR] Could not extract service account email from key file")
        sys.exit(1)
    
    print(f"[AUTH] Using service account: {service_account_email}")
    
    # Change to deploy directory where cloudbuild.yaml is located
    deploy_dir = os.path.join(os.path.dirname(__file__), "deploy")
    
    # Build the Cloud Build command with service account
    cmd = [
        "gcloud", "builds", "submit",
        "--config", os.path.join(deploy_dir, "cloudbuild.yaml"),
        "--project", project_id,
        "--substitutions",
        f"_SERVICE_NAME=security-agent-backend,_REGION=us-central1,_SERVICE_ACCOUNT_EMAIL={service_account_email}",
        "."
    ]
    
    print(f"[CONFIG] Command: {' '.join(cmd)}")
    print("[BUILD] Building and deploying backend service...")
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print("[SUCCESS] Backend deployed successfully!")
        print(f"[URL] Service URL: https://security-agent-backend-<hash>-uc.a.run.app")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Deployment failed: {e}")
        sys.exit(1)

def kill_existing_backend(port='8000'):
    """Kill any existing processes on the specified port."""
    print(f"[CHECK] Checking for existing processes on port {port}...")
    
    try:
        # Check if port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', int(port)))
        sock.close()
        
        if result == 0:
            print(f"[WARNING] Port {port} is already in use")
            
            # Find and kill processes using lsof
            try:
                # Get PIDs using the port
                result = subprocess.run(
                    ['lsof', '-ti', f':{port}'],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid:
                            print(f"[KILL] Killing process {pid} on port {port}")
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    
                    # Wait for processes to die
                    time.sleep(2)
                    print(f"[SUCCESS] Cleared port {port}")
                    
            except Exception as e:
                print(f"[WARNING] Could not kill processes: {e}")
                print("You may need to manually kill the process or use a different port")
                
        else:
            print(f"[OK] Port {port} is free")
            
    except Exception as e:
        print(f"[ERROR] Error checking port: {e}")

def run_local():
    """Start the FastAPI backend server locally."""
    # Detect if running in Cloud Run
    is_cloud_run = os.environ.get('K_SERVICE') is not None
    port = os.environ.get('BACKEND_PORT', os.environ.get('PORT', '8000'))
    
    # Kill any existing backend processes if running locally
    if not is_cloud_run:
        kill_existing_backend(port)
    
    print("[STARTUP] Starting GCP Security Agent Backend (FastAPI)")
    print("=" * 50)
    print("[SERVICES] Backend Services:")
    print("  - Data refresh from GCP APIs")
    print("  - SQLite cache management")
    print("  - Security analysis endpoints")
    print("  - Tool implementations for agent")
    print("=" * 50)
    
    if is_cloud_run:
        print("[ENV] Running in Cloud Run environment")
    else:
        print("[ENV] Running in local development mode")
    
    # Load environment variables from .env if it exists (local dev only)
    if not is_cloud_run:
        from pathlib import Path
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"[CONFIG] Loaded environment from: {env_file}")
    
    # Set environment variables with defaults
    if not os.environ.get('GOOGLE_CLOUD_PROJECT'):
        # Default to a placeholder - user should set their own
        os.environ['GOOGLE_CLOUD_PROJECT'] = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')
        if os.environ['GOOGLE_CLOUD_PROJECT'] == 'your-project-id' and not is_cloud_run:
            print("[WARNING] GOOGLE_CLOUD_PROJECT not set. Please set it in .env file")
    
    # Set Google Application Credentials if not already set (local dev only)
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and not is_cloud_run:
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
                print(f"[CREDS] Set GOOGLE_APPLICATION_CREDENTIALS to: {service_account_path}")
            else:
                print("[WARNING] No service account JSON files found in config/secrets/")
        else:
            print("[WARNING] Service account directory not found, will use default credentials")
    
    # Change to backend directory - IMPORTANT for correct imports
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
        print(f"[DIR] Changed to backend directory: {os.getcwd()}")
    else:
        print(f"[WARNING] Backend directory not found at {backend_dir}")
        print(f"[DIR] Current directory: {os.getcwd()}")
    
    # Configure uvicorn based on environment
    if is_cloud_run:
        # Cloud Run configuration - no reload, bind to PORT env var
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", port
        ]
    else:
        # Local development configuration - with reload
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", port
        ]
    
    print(f"[DIR] Working directory: {os.getcwd()}")
    print(f"[CMD] Command: {' '.join(cmd)}")
    print(f"[PROJECT] Project: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    print("=" * 50)
    
    if is_cloud_run:
        print(f"Backend listening on port: {port}")
        print("Service will be available at Cloud Run URL")
    else:
        print(f"Backend will be available at: http://localhost:{port}")
        print(f"API docs at: http://localhost:{port}/docs")
    
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[STOP] Backend stopped.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error starting backend: {e}")
        sys.exit(1)

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Run Security Agent Backend')
    parser.add_argument('--cloud', action='store_true', 
                       help='Deploy to Cloud Run instead of running locally')
    args = parser.parse_args()
    
    # Load environment variables from .env
    from pathlib import Path
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"[CONFIG] Loaded environment from: {env_file}")
    
    if args.cloud:
        # Get project ID from environment
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project_id or project_id == 'your-project-id':
            print("[ERROR] GOOGLE_CLOUD_PROJECT not set in .env file")
            print("Please set GOOGLE_CLOUD_PROJECT in your .env file")
            sys.exit(1)
        deploy_to_cloud(project_id)
    else:
        run_local()

if __name__ == "__main__":
    main()