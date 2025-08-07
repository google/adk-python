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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse
import shutil

def run_cloud_build(base_dir):
    """Trigger a Cloud Build to build and deploy the application."""
    logger.info("☁️ Starting Cloud Build process...")

    # Ensure gcloud is installed
    if not shutil.which("gcloud"):
        logger.error("gcloud command not found. Please install the Google Cloud SDK.")
        return

    # Get project ID from environment
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set.")
        logger.error("Please set it to your GCP project ID: `export GOOGLE_CLOUD_PROJECT=your-project-id`")
        return

    logger.info(f"Using project ID: {project_id}")

    # The directory containing the cloudbuild.yaml and source code
    build_context = base_dir

    # Substitutions for the Cloud Build command
    substitutions = [
        "_REGION=us-central1",
        "_REPO_NAME=adk-security-agent",
        "_IMAGE_NAME=security-agent",
        "_SERVICE_NAME=security-agent"
    ]

    # Cloud Build command
    cmd = [
        "gcloud", "builds", "submit",
        str(build_context),
        "--config", str(build_context / "cloudbuild.yaml"),
        f"--project={project_id}",
        f"--substitutions={','.join(substitutions)}"
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

    base_dir = Path(__file__).parent

    if args.cloud:
        run_cloud_build(base_dir)
        return

    print("🖥️ Starting ADK Security Agent Frontend")
    print("=" * 50)
    
    # Check for virtual environment
    venv_path = base_dir / "venv"
    if venv_path.exists() and (venv_path / "bin" / "python").exists():
        python_exe = str(venv_path / "bin" / "python")
        logger.info(f"Using venv Python: {python_exe}")
    else:
        python_exe = sys.executable
        logger.info(f"Using system Python: {python_exe}")
    
    # Frontend configuration
    frontend_port = os.getenv('FRONTEND_PORT', '8501')
    frontend_host = os.getenv('FRONTEND_HOST', '0.0.0.0')
    
    # Frontend app path
    frontend_app = base_dir / "frontend" / "main_app.py"
    
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
        subprocess.run(cmd, cwd=base_dir, check=False)
    except KeyboardInterrupt:
        logger.info("🛑 Frontend stopped by user")
    except Exception as e:
        logger.error(f"❌ Frontend failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()