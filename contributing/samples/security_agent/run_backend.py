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

import argparse

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

    # Cloud Build command
    cmd = [
        "gcloud", "builds", "submit",
        str(build_context),
        "--config", str(build_context / "cloudbuild.yaml"),
        f"--project={project_id}"
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

    base_dir = Path(__file__).parent

    if args.cloud:
        run_cloud_build(base_dir)
        return
        
    print("🛡️ Starting ADK Security Agent Backend")
    print("=" * 50)
    
    # Get the script directory
    
    # Check for virtual environment
    venv_path = base_dir / "venv"
    if venv_path.exists() and (venv_path / "bin" / "python").exists():
        python_exe = str(venv_path / "bin" / "python")
        logger.info(f"Using venv Python: {python_exe}")
    else:
        python_exe = sys.executable
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
    backend_dir = base_dir / "backend"
    
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