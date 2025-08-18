#!/usr/bin/env python3
"""
Run the frontend Streamlit app for the Security Agent.
Supports both local development and Cloud Run deployments.

Usage:
    python run_frontend.py          # Run locally
    python run_frontend.py --cloud  # Deploy to Cloud Run
"""

import subprocess
import sys
import os
import argparse

def deploy_to_cloud(project_id):
    """Deploy frontend to Cloud Run using Cloud Build."""
    print(f"☁️  Deploying frontend to Cloud Run (Project: {project_id})...")
    print("=" * 50)
    
    # Change to deploy directory where cloudbuild.yaml is located
    deploy_dir = os.path.join(os.path.dirname(__file__), "deploy")
    
    # Build the Cloud Build command for frontend
    cmd = [
        "gcloud", "builds", "submit",
        "--config", os.path.join(deploy_dir, "cloudbuild.yaml"),
        "--project", project_id,
        "--substitutions",
        f"_SERVICE_NAME=security-agent-frontend,_REGION=us-central1,_IS_FRONTEND=true",
        "."
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("📦 Building and deploying frontend service...")
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print("✅ Frontend deployed successfully!")
        print(f"🔗 Service URL: https://security-agent-frontend-<hash>-uc.a.run.app")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def run_local():
    """Start the Streamlit frontend locally."""
    # Detect if running in Cloud Run
    is_cloud_run = os.environ.get('K_SERVICE') is not None
    port = os.environ.get('PORT', '8501')
    
    print("🚀 Starting Security Agent Frontend (Thin Client)...")
    print("=" * 50)
    
    if is_cloud_run:
        print("☁️  Running in Cloud Run environment")
        # In Cloud Run, backend URL might be set via environment variable
        if not os.environ.get('BACKEND_URL'):
            # Default to internal service communication
            os.environ['BACKEND_URL'] = os.environ.get('BACKEND_SERVICE_URL', 'http://backend-service:8000')
    else:
        print("💻 Running in local development mode")
        # Set default backend URL for local development
        if not os.environ.get('BACKEND_URL'):
            os.environ['BACKEND_URL'] = 'http://localhost:8000'
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    os.chdir(frontend_dir)
    
    # Configure Streamlit based on environment
    if is_cloud_run:
        # Cloud Run configuration - bind to 0.0.0.0 and PORT env var
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "main_app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
    else:
        # Local development configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "main_app.py",
            "--server.port", port,
            "--server.address", "localhost"
        ]
    
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🔗 Backend URL: {os.environ.get('BACKEND_URL')}")
    print("=" * 50)
    
    if is_cloud_run:
        print(f"Frontend listening on port: {port}")
        print("Service will be available at Cloud Run URL")
    else:
        print(f"Frontend will be available at: http://localhost:{port}")
    
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Run Security Agent Frontend')
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