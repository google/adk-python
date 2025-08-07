#!/usr/bin/env python3
"""
🛡️ Enhanced GCP API Security Evaluation Agent - One-Command Startup

This script provides a single entry point to run the security agent with different configurations:
- Modular architecture (default) - Services can be enabled/disabled independently
- Legacy architecture - Original monolithic backend
- Docker deployment - Containerized deployment

Usage:
    python run.py                    # Run modular architecture (default)
    python run.py --legacy           # Run legacy monolithic backend  
    python run.py --docker           # Run in Docker container
    python run.py --help             # Show help
"""

import os
import sys
import time
import subprocess
import signal
import logging
import argparse
import json
import webbrowser
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global process tracking
processes = []

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("🛑 Shutting down all services...")
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
    
    # Wait for graceful shutdown
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Force killing process {process.pid}")
            process.kill()
    
    logger.info("✅ All services stopped")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def print_banner():
    """Print startup banner."""
    print("""
🛡️  Enhanced GCP API Security Evaluation Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Comprehensive security analysis for GCP APIs
    with modular service architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('streamlit', 'Streamlit'),
        ('google.cloud.resourcemanager_v3', 'Google Cloud Resource Manager'),
        ('requests', 'Requests'),
        ('pydantic', 'Pydantic')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True


def setup_environment():
    """Set up environment variables and configuration."""
    base_dir = Path(__file__).parent
    backend_dir = base_dir / "backend"
    
    # Ensure directories exist
    (base_dir / "logs").mkdir(exist_ok=True)
    (backend_dir / "config").mkdir(exist_ok=True)
    
    # Set default service config path if not set
    if not os.getenv('SERVICE_CONFIG_PATH'):
        config_path = backend_dir / "config" / "services.json"
        os.environ['SERVICE_CONFIG_PATH'] = str(config_path)
        logger.info(f"📋 Service config: {config_path}")
    
    # Check for GCP credentials
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        # Look for service account key in config/secrets
        secrets_dir = backend_dir / "config" / "secrets"
        if secrets_dir.exists():
            for key_file in secrets_dir.glob("*.json"):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(key_file)
                logger.info(f"🔑 Found service account key: {key_file.name}")
                break
        
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.warning("⚠️  No GOOGLE_APPLICATION_CREDENTIALS found")
            logger.warning("   Some services may not work without GCP authentication")
    
    # Set Python path
    sys.path.insert(0, str(backend_dir))
    
    logger.info("✅ Environment setup complete")


def wait_for_service(url: str, timeout: int = 30, service_name: str = "service") -> bool:
    """Wait for a service to be ready."""
    import requests
    
    logger.info(f"⏳ Waiting for {service_name} to be ready...")
    
    for attempt in range(timeout):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ {service_name.title()} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
    
    logger.error(f"❌ {service_name.title()} failed to start within {timeout} seconds")
    return False


def start_backend(modular: bool = True) -> Optional[subprocess.Popen]:
    """Start the backend server."""
    backend_dir = Path(__file__).parent / "backend"
    
    if modular:
        main_module = "main_modular:app"
        logger.info("🚀 Starting modular backend server...")
    else:
        main_module = "main:app"
        logger.info("🚀 Starting legacy backend server...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        main_module,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ]
    
    try:
        # Start backend in background
        process = subprocess.Popen(
            cmd,
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        processes.append(process)
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        return None


def start_frontend() -> Optional[subprocess.Popen]:
    """Start the Streamlit frontend."""
    logger.info("🌐 Starting frontend server...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(frontend_dir / "main_app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        processes.append(process)
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start frontend: {e}")
        return None


def run_docker():
    """Run the application in Docker."""
    logger.info("🐳 Starting Docker deployment...")
    
    # Check if Docker is available
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, check=True)
        logger.info(f"✅ Docker found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ Docker is not installed or not available")
        logger.error("Please install Docker: https://docs.docker.com/get-docker/")
        return False
    
    base_dir = Path(__file__).parent
    
    # Check for Dockerfile
    dockerfile = base_dir / "Dockerfile"
    if not dockerfile.exists():
        logger.error("❌ Dockerfile not found")
        logger.error("Please ensure Dockerfile exists in the project root")
        return False
    
    try:
        # Build Docker image
        logger.info("🔨 Building Docker image...")
        build_cmd = ["docker", "build", "-t", "security-agent", "."]
        subprocess.run(build_cmd, cwd=str(base_dir), check=True)
        
        # Run Docker container
        logger.info("🚀 Starting Docker container...")
        run_cmd = [
            "docker", "run",
            "-p", "8000:8000",
            "-p", "8501:8501",
            "--rm",
            "-it",
            "security-agent"
        ]
        
        subprocess.run(run_cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Docker command failed: {e}")
        return False


def open_browser_tabs():
    """Open browser tabs for the application."""
    urls = [
        ("Frontend UI", "http://localhost:8501"),
        ("Backend API", "http://localhost:8000"),
        ("API Documentation", "http://localhost:8000/docs")
    ]
    
    logger.info("🌐 Opening browser tabs...")
    for name, url in urls:
        try:
            webbrowser.open(url)
            logger.info(f"   📖 {name}: {url}")
        except Exception as e:
            logger.warning(f"Failed to open {name}: {e}")


def show_status_info(modular: bool = True):
    """Show application status and access information."""
    print("""
🎉 Security Agent is now running!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Access Points:
   🌐 Frontend UI:       http://localhost:8501
   🔧 Backend API:       http://localhost:8000
   📖 API Documentation: http://localhost:8000/docs
   🩺 Health Check:      http://localhost:8000/health
""")
    
    if modular:
        print("""⚙️  Modular Features:
   🔧 Service Management: http://localhost:8501 -> Service Management
   📊 Service Status:     http://localhost:8000/api/v1/services/status/summary
   🩺 Health Monitoring:  http://localhost:8000/api/v1/services/{service}/health
""")
    
    print("""🚀 Getting Started:
   1. Go to http://localhost:8501
   2. Select your GCP project
   3. Start with the Dashboard
   4. Use Service Management to enable/disable features (modular mode)

⏹️  To stop: Press Ctrl+C

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(
        description="Enhanced GCP API Security Evaluation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run with modular architecture (default)
  python run.py --legacy           # Run with legacy monolithic backend
  python run.py --docker           # Run in Docker container
  python run.py --no-browser       # Don't open browser automatically
        """
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy monolithic backend instead of modular architecture'
    )
    
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Run in Docker container'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser tabs'
    )
    
    parser.add_argument(
        '--frontend-only',
        action='store_true',
        help='Start only the frontend (backend must be running separately)'
    )
    
    parser.add_argument(
        '--backend-only',
        action='store_true',
        help='Start only the backend'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Docker mode
    if args.docker:
        return run_docker()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup environment
    setup_environment()
    
    modular = not args.legacy
    mode_name = "Modular" if modular else "Legacy"
    logger.info(f"🔧 Running in {mode_name} mode")
    
    try:
        # Start backend (unless frontend-only)
        backend_process = None
        if not args.frontend_only:
            backend_process = start_backend(modular=modular)
            if not backend_process:
                logger.error("❌ Failed to start backend")
                return False
            
            # Wait for backend to be ready
            if not wait_for_service("http://localhost:8000/health", service_name="backend"):
                logger.error("❌ Backend failed to start properly")
                return False
        
        # Start frontend (unless backend-only)
        frontend_process = None
        if not args.backend_only:
            frontend_process = start_frontend()
            if not frontend_process:
                logger.error("❌ Failed to start frontend")
                return False
            
            # Wait for frontend to be ready
            if not wait_for_service("http://localhost:8501", service_name="frontend"):
                logger.error("❌ Frontend failed to start properly")
                return False
        
        # Open browser tabs
        if not args.no_browser and not args.backend_only:
            time.sleep(2)  # Give services a moment to fully start
            open_browser_tabs()
        
        # Show status
        show_status_info(modular=modular)
        
        # Keep the script running and monitor processes
        while True:
            time.sleep(1)
            
            # Check if any process has died
            if backend_process and backend_process.poll() is not None:
                logger.error("❌ Backend process has stopped")
                break
            
            if frontend_process and frontend_process.poll() is not None:
                logger.error("❌ Frontend process has stopped")
                break
        
    except KeyboardInterrupt:
        # Handled by signal handler
        pass
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)