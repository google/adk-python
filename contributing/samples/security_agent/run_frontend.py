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
import signal
import time
import socket
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_service_account_email():
    """Extract service account email from the key file."""
    import json
    
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        print("Please set GOOGLE_APPLICATION_CREDENTIALS in your .env file")
        sys.exit(1)
    
    # Convert relative path to absolute
    if not os.path.isabs(creds_path):
        creds_path = os.path.join(os.path.dirname(__file__), creds_path)
    
    if not os.path.exists(creds_path):
        print(f"❌ Service account key file not found: {creds_path}")
        print("Please ensure the service account JSON file exists")
        sys.exit(1)
    
    try:
        with open(creds_path, 'r') as f:
            key_data = json.load(f)
            return key_data.get('client_email')
    except Exception as e:
        print(f"❌ Failed to read service account key file: {e}")
        sys.exit(1)

def deploy_to_cloud(project_id, extra_args=None):
    """Deploy unified frontend to Cloud Run using Cloud Build.
    
    Args:
        project_id: GCP project ID
        extra_args: Additional gcloud arguments to pass through
    """
    print(f"☁️  Deploying Unified Frontend to Cloud Run (Project: {project_id})...")
    print("=" * 50)
    print("✨ Deploying unified streaming client with executive dashboard")
    
    # Get service account email from key file
    service_account_email = get_service_account_email()
    if not service_account_email:
        print("❌ Could not extract service account email from key file")
        sys.exit(1)
    
    print(f"🔐 Using service account: {service_account_email}")
    
    # Change to deploy directory where cloudbuild.yaml is located
    deploy_dir = os.path.join(os.path.dirname(__file__), "deploy")
    
    # Build the Cloud Build command for frontend with service account
    cmd = [
        "gcloud", "builds", "submit",
        "--config", os.path.join(deploy_dir, "cloudbuild-frontend.yaml"),
        "--project", project_id,
        "--substitutions",
        f"_SERVICE_ACCOUNT_EMAIL={service_account_email}",
        "."
    ]
    
    # Add extra arguments if provided
    if extra_args:
        print(f"🔧 Additional Cloud Run arguments: {' '.join(extra_args)}")
        # We'll need to pass these to the Cloud Build config
        cmd.extend(["--substitutions", f"_EXTRA_ARGS={' '.join(extra_args)}"])
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("📦 Building and deploying frontend service...")
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print("✅ Frontend deployed successfully!")
        print(f"🔗 Service URL: https://security-agent-frontend-<hash>-uc.a.run.app")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def kill_existing_process(port):
    """Kill any existing processes on the specified port."""
    print(f"[CHECK] Checking for existing processes on port {port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', int(port)))
        sock.close()
        
        if result == 0:
            print(f"[WARNING] Port {port} is already in use.")
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
                if result.stdout:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid:
                            print(f"[KILL] Killing process {pid} on port {port}")
                            os.kill(int(pid), signal.SIGKILL)
                    time.sleep(2)
                    print(f"[SUCCESS] Cleared port {port}")
            except Exception as e:
                print(f"[WARNING] Could not kill processes: {e}")
        else:
            print(f"[OK] Port {port} is free")
    except Exception as e:
        print(f"[ERROR] Error checking port: {e}")

def run_local():
    """Start the Streamlit frontend with token streaming."""
    # Check if we should use venv
    from pathlib import Path
    venv_path = Path(__file__).parent / 'venv'
    if venv_path.exists():
        # Use venv Python if available
        python_executable = str(venv_path / 'bin' / 'python')
        print(f"[VENV] Using virtual environment: {venv_path}")
        print(f"[PYTHON] Python executable: {python_executable}")

        # Set VIRTUAL_ENV environment variable to ensure venv is used
        os.environ['VIRTUAL_ENV'] = str(venv_path)
        # Update PATH to prioritize venv binaries
        venv_bin = str(venv_path / 'bin')
        current_path = os.environ.get('PATH', '')
        if venv_bin not in current_path:
            os.environ['PATH'] = f"{venv_bin}:{current_path}"
            print(f"[VENV] Updated PATH to include: {venv_bin}")
    else:
        python_executable = sys.executable
        print(f"[PYTHON] Using system Python: {python_executable}")
        print("[WARNING] Virtual environment not found - install dependencies with: python -m venv venv && source venv/bin/activate && pip install -r requirements_frontend.txt")

    # Detect if running in Cloud Run
    is_cloud_run = os.environ.get('K_SERVICE') is not None
    port = os.environ.get('FRONTEND_PORT', os.environ.get('PORT', '8501'))

    if not is_cloud_run:
        kill_existing_process(port)

    print("🚀 Starting GCP Security Executive Dashboard")
    print("=" * 50)
    print("✨ Unified Features:")
    print("  • Executive dashboard on front page")
    print("  • Token-by-token streaming chat")
    print("  • Real-time security metrics")
    print("  • SQLite database integration")
    print("  • Consolidated security views")
    print("=" * 50)

    # Add frontend directory to Python path BEFORE changing directories
    # This allows proper imports from the frontend module
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    sys.path.insert(0, frontend_dir)
    sys.path.insert(0, os.path.dirname(__file__))  # Also add project root

    # Set critical environment variables with absolute path
    try:
        from config.database import DatabaseConfig
        database_path = DatabaseConfig.get_database_path()
        os.environ['DATABASE_PATH'] = database_path
        print(f"✅ Database path set to: {database_path}")
    except Exception as e:
        print(f"⚠️ Failed to set database path: {e}")
        # Fallback to absolute path
        fallback_path = os.path.join(
            os.path.dirname(__file__), 'backend', 'cache', 'gcp_data.db'
        )
        os.environ['DATABASE_PATH'] = os.path.abspath(fallback_path)
        print(f"Using fallback database path: {os.environ['DATABASE_PATH']}")

    # CRITICAL: Use '1' not 'TRUE' for GOOGLE_GENAI_USE_VERTEXAI
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = '1'

    if is_cloud_run:
        print("☁️  Running in Cloud Run environment")
    else:
        print("💻 Running in local development mode")

    # Change to frontend directory AFTER setting up sys.path
    os.chdir(frontend_dir)
    
    # Use the main app as the frontend entry point
    streamlit_file = "app.py"
    
    # Configure Streamlit based on environment
    if is_cloud_run:
        # Cloud Run configuration - bind to 0.0.0.0 and PORT env var
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            streamlit_file,
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
    else:
        # Local development configuration
        cmd = [
            python_executable, "-m", "streamlit", "run",
            streamlit_file,
            "--server.port", port,
            "--server.address", "localhost"
        ]
    
    # Display configuration summary
    try:
        from config.environment import EnvironmentConfig
        config_summary = EnvironmentConfig.get_configuration_summary()
        
        print(f"📂 Working directory: {os.getcwd()}")
        print(f"🔧 Command: {' '.join(cmd)}")
        print(f"🗄️ Database: {os.environ.get('DATABASE_PATH')}")
        print(f"🤖 Using: ADK Agent with Google ADK Framework")
        print(f"⚙️ Config: {'Valid' if config_summary['is_valid'] else 'Issues Detected'} ({config_summary['valid_count']} vars)")
        print(f"📅 Project: {config_summary['project_id']}")
        print("=" * 50)
        
    except Exception:
        # Fallback to basic display
        print(f"📂 Working directory: {os.getcwd()}")
        print(f"🔧 Command: {' '.join(cmd)}")
        print(f"🗄️ Database: {os.environ.get('DATABASE_PATH')}")
        print(f"🤖 Using: ADK Agent with Google ADK Framework")
        print(f"⚙️ Config: Fallback Mode")
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
    parser = argparse.ArgumentParser(description='Run Security Agent Frontend',
                                   # Allow unknown args to be passed to gcloud
                                   allow_abbrev=False,
                                   add_help=False)
    parser.add_argument('--cloud', action='store_true', 
                       help='Deploy to Cloud Run instead of running locally')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Show this help message and exit')
    
    # Parse known args and capture the rest for gcloud
    args, unknown_args = parser.parse_known_args()
    
    # Handle help manually  
    if args.help:
        print("Usage: python run_frontend.py [OPTIONS] [-- GCLOUD_ARGS]")
        print("\nOptions:")
        print("  --cloud              Deploy to Cloud Run instead of running locally")
        print("  -h, --help           Show this help message and exit")
        print("\nCloud Run Deployment:")
        print("  python run_frontend.py --cloud")
        print("  python run_frontend.py --cloud -- --min-instances=1 --max-instances=5")
        print("\nLocal Development:")
        print("  python run_frontend.py")
        sys.exit(0)
    
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
        
        # Remove '--' separator if present in unknown_args
        extra_args = [arg for arg in unknown_args if arg != '--']
        
        deploy_to_cloud(project_id, extra_args)
    else:
        if unknown_args:
            print(f"⚠️  Ignoring extra arguments for local run: {unknown_args}")
        run_local()

if __name__ == "__main__":
    main()