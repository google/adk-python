#!/usr/bin/env python3
"""
🛡️ ADK Security Agent - Backend Only

Simple script to run just the legacy backend server.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from the root .env file
from dotenv import load_dotenv
dotenv_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f".env file not found at project root. Using default configurations.")

import argparse
import shutil

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

    # Cloud Build command
    cmd = [
        "gcloud", "builds", "submit",
        str(build_context),
        "--config", os.path.join(build_context, "contributing/samples/security_agent/cloudbuild.yaml"),
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
    """Run the ADK backend server."""
    parser = argparse.ArgumentParser(description="🛡️ ADK Security Agent - Backend Only")
    parser.add_argument("--cloud", action="store_true", help="Trigger a Cloud Build to deploy the application.")
    args = parser.parse_args()

    if args.cloud:
        run_cloud_build()
        return
        
    print("🛡️ Starting ADK Security Agent Backend")
    print("=" * 50)
    
    # Check for virtual environment
    venv_path = os.path.join(os.getcwd(), "venv")
    python_exe = sys.executable
    if os.path.exists(venv_path) and os.path.exists(os.path.join(venv_path, "bin", "python")):
        python_exe = os.path.join(venv_path, "bin", "python")
        logger.info(f"Using venv Python: {python_exe}")
    else:
        logger.info(f"Using system Python: {python_exe}")
    
    # Set environment
    os.environ['USE_LEGACY'] = 'true'
    
    # Backend configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT', '8000')
    log_level = os.getenv('LOG_LEVEL', 'info')
    reload = os.getenv('RELOAD', 'true').lower() == 'true'
    
    # Build command (run from backend/ directory)
    cmd = [
        python_exe, "-m", "uvicorn",
        "main_legacy:app",
        "--host", host,
        "--port", port,
        "--log-level", log_level
    ]
    
    if reload:
        cmd.append("--reload")
    
    # Set working directory to backend/
    backend_dir = os.path.join("contributing", "samples", "security_agent", "backend")
    
    logger.info("🚀 Starting legacy backend server...")
    logger.info(f"   • Host: {host}")
    logger.info(f"   • Port: {port}")
    logger.info(f"   • Log Level: {log_level}")
    logger.info(f"   • Reload: {reload}")
    logger.info(f"   • Working Directory: {backend_dir}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    print(f"""
📊 Backend will be available at:
   🔧 API Endpoints: http://localhost:{port}
   📖 API Docs: http://localhost:{port}/docs
   🩺 Health Check: http://localhost:{port}/health

Press Ctrl+C to stop the backend.
""")
    
    try:
        # Run the backend from the backend/ directory
        subprocess.run(cmd, cwd=backend_dir, check=False)
    except KeyboardInterrupt:
        logger.info("🛑 Backend stopped by user")
    except Exception as e:
        logger.error(f"❌ Backend failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()