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

def deploy_to_cloud(project_id):
    """Deploy backend to Cloud Run using Cloud Build."""
    print(f"☁️  Deploying backend to Cloud Run (Project: {project_id})...")
    print("=" * 50)
    
    # Change to deploy directory where cloudbuild.yaml is located
    deploy_dir = os.path.join(os.path.dirname(__file__), "deploy")
    
    # Build the Cloud Build command
    cmd = [
        "gcloud", "builds", "submit",
        "--config", os.path.join(deploy_dir, "cloudbuild.yaml"),
        "--project", project_id,
        "--substitutions",
        f"_SERVICE_NAME=security-agent-backend,_REGION=us-central1",
        "."
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("📦 Building and deploying backend service...")
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print("✅ Backend deployed successfully!")
        print(f"🔗 Service URL: https://security-agent-backend-<hash>-uc.a.run.app")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def run_local():
    """Start the FastAPI backend server locally."""
    # Detect if running in Cloud Run
    is_cloud_run = os.environ.get('K_SERVICE') is not None
    port = os.environ.get('PORT', '8000')
    
    print("🚀 Starting Security Agent Backend...")
    print("=" * 50)
    
    if is_cloud_run:
        print("☁️  Running in Cloud Run environment")
    else:
        print("💻 Running in local development mode")
    
    # Load environment variables from .env if it exists (local dev only)
    if not is_cloud_run:
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
        if os.environ['GOOGLE_CLOUD_PROJECT'] == 'your-project-id' and not is_cloud_run:
            print("⚠️ GOOGLE_CLOUD_PROJECT not set. Please set it in .env file")
    
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
                print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS to: {service_account_path}")
            else:
                print("⚠️ No service account JSON files found in config/secrets/")
        else:
            print("⚠️ Service account directory not found, will use default credentials")
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    os.chdir(backend_dir)
    
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
    
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🌍 Project: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
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
        print("\n👋 Backend stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")
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
        print(f"✅ Loaded environment from: {env_file}")
    
    if args.cloud:
        # Get project ID from environment
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project_id or project_id == 'your-project-id':
            print("❌ GOOGLE_CLOUD_PROJECT not set in .env file")
            print("Please set GOOGLE_CLOUD_PROJECT in your .env file")
            sys.exit(1)
        deploy_to_cloud(project_id)
    else:
        run_local()

if __name__ == "__main__":
    main()