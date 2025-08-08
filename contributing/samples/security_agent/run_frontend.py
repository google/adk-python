#!/usr/bin/env python3
"""
🖥️ ADK Security Agent - Frontend Only

Simple script to run just the Streamlit frontend.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse
import shutil
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from the local .env file
from dotenv import load_dotenv
dotenv_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Using default configurations.")

def run_cloud_build():
    """Trigger a Cloud Build to build and deploy the application."""
    logger.info("☁️ Starting Cloud Build process...")

    # Ensure gcloud is installed
    if not shutil.which("gcloud"):
        logger.error("gcloud command not found. Please install the Google Cloud SDK.")
        return

    # Get project ID from environment (required)
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set.")
        logger.error("Please set it in your .env file or with `export GOOGLE_CLOUD_PROJECT=your-project-id`")
        return

    logger.info(f"Using project ID: {project_id}")

    # The directory containing the cloudbuild.yaml and source code
    build_context = os.getcwd()

    # Substitutions for the Cloud Build command, read from environment or use defaults
    substitutions = {
        "_REGION": os.getenv("_REGION", "us-central1"),
        "_REPO_NAME": os.getenv("_REPO_NAME", "adk-security-agent"),
        "_IMAGE_NAME": os.getenv("_IMAGE_NAME", "security-agent"),
        "_SERVICE_NAME": os.getenv("_SERVICE_NAME", "security-agent")
    }

    # Validate substitutions
    for key, value in substitutions.items():
        if not value:
            logger.error(f"Build configuration variable '{key}' is missing or empty in your .env file.")
            logger.error("Please ensure all required build variables are set.")
            return
    
    substitutions_str = ",".join([f"{k}={v}" for k, v in substitutions.items()])

    # Cloud Build command (using cloudbuild.yaml from deploy directory)
    adk_root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    cloudbuild_path = os.path.join(adk_root, "deploy", "cloudbuild.yaml")
    if not os.path.exists(cloudbuild_path):
        logger.error(f"cloudbuild.yaml not found at {cloudbuild_path}")
        logger.error("Please ensure cloudbuild.yaml exists in the /deploy directory")
        return
    
    cmd = [
        "gcloud", "builds", "submit",
        str(build_context),
        "--config", cloudbuild_path,
        f"--project={project_id}",
        f"--substitutions={substitutions_str}"
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Cloud Build completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Cloud Build failed: {e}")
    except KeyboardInterrupt:
        logger.info("🛑 Cloud Build stopped by user.")

def main():
    """Run the ADK frontend server."""
    parser = argparse.ArgumentParser(description="🖥️ ADK Security Agent - Frontend Only")
    parser.add_argument("--cloud", action="store_true", help="Trigger a Cloud Build to deploy the application.")
    args = parser.parse_args()

    if args.cloud:
        run_cloud_build()
        return

    print("🖥️ Starting ADK Security Agent Frontend")
    print("=" * 50)
    
    # Check for virtual environment
    venv_path = os.path.join(os.getcwd(), "venv")
    python_exe = sys.executable
    if os.path.exists(venv_path) and os.path.exists(os.path.join(venv_path, "bin", "python")):
        python_exe = os.path.join(venv_path, "bin", "python")
        logger.info(f"Using venv Python: {python_exe}")
    else:
        logger.info(f"Using system Python: {python_exe}")
    
    # Frontend configuration
    frontend_port = os.getenv('FRONTEND_PORT', '8501')
    frontend_host = os.getenv('FRONTEND_HOST', '0.0.0.0')
    
    # Frontend app path
    # Determine the correct path based on current location
    current_dir = os.getcwd()
    if current_dir.endswith("security_agent"):
        # We're in the security_agent directory
        frontend_app = os.path.join("frontend", "main_app.py")
    elif current_dir.endswith("ADK"):
        # We're in the ADK root directory  
        frontend_app = os.path.join("contributing", "samples", "security_agent", "frontend", "main_app.py")
    else:
        # Try to find main_app.py relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_app = os.path.join(script_dir, "frontend", "main_app.py")
    
    # Check if frontend app exists
    if not os.path.exists(frontend_app):
        logger.error(f"❌ Frontend app not found at: {frontend_app}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error("Please run this script from the security_agent directory or the ADK root directory.")
        return False
    
    # Build command
    cmd = [
        python_exe, "-m", "streamlit", "run", 
        str(frontend_app),
        "--server.port", frontend_port,
        "--server.address", frontend_host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    logger.info("🚀 Starting Streamlit frontend...")
    logger.info(f"   • Host: {frontend_host}")
    logger.info(f"   • Port: {frontend_port}")
    logger.info(f"   • App: {frontend_app}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    print(f"""
📊 Frontend will be available at:
   🌐 Web Interface: http://localhost:{frontend_port}
   💬 Chat Interface: http://localhost:{frontend_port} → Click "💬 AI Assistant"

⚠️  Make sure the backend is running first!
   Run in another terminal: python run_backend.py

Press Ctrl+C to stop the frontend.
""")
    
    try:
        # Run the frontend
        subprocess.run(cmd, cwd=os.getcwd(), check=False)
    except KeyboardInterrupt:
        logger.info("🛑 Frontend stopped by user")
    except Exception as e:
        logger.error(f"❌ Frontend failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
