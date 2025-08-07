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

def main():
    """Run the ADK backend server."""
    print("🛡️ Starting ADK Security Agent Backend")
    print("=" * 50)
    
    # Get the script directory
    base_dir = Path(__file__).parent
    
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