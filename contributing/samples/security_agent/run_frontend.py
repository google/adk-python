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

def main():
    """Run the ADK frontend server."""
    print("🖥️ Starting ADK Security Agent Frontend")
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