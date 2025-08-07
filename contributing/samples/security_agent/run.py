#!/usr/bin/env python3
"""
🛡️ Enhanced GCP API Security Evaluation Agent - Universal Startup Script

This script provides a single entry point for all deployment scenarios:
- Local development with virtual environment
- Docker deployment (local and Cloud Run)
- Direct Cloud Run deployment
- Production server mode

Usage:
    # Local Development
    python run.py                    # Run legacy backend (default)
    python run.py --modular          # Run modular architecture
    
    # Cloud/Production
    python run.py --cloud            # Cloud Run compatible mode
    python run.py --production       # Production server (no reload)
    
    # Docker
    python run.py --docker           # Build and run in Docker
    python run.py --docker-build     # Build Docker image only
    
    # Deployment
    python run.py --deploy PROJECT_ID # Deploy to Cloud Run
    python run.py --help             # Show help

Environment Variables:
    PORT: Port to bind to (set by Cloud Run, default 8000)
    GOOGLE_CLOUD_PROJECT: GCP project ID
    K_SERVICE: Cloud Run service name (auto-detected)
    K_REVISION: Cloud Run revision (auto-detected)
    USE_MODULAR: Use modular backend (true/false)
    SERVICE_CONFIG_PATH: Path to service configuration
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

# Load environment variables from .env file first
base_dir = Path(__file__).parent

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    
    # Load .env file from project root
    env_file = base_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"✅ Loaded environment from {env_file}")
    elif (base_dir / ".env.example").exists():
        print("⚠️  No .env file found. Copy .env.example to .env and update with your values.")
except ImportError:
    # python-dotenv not installed, try manual loading
    env_file = base_dir / ".env"
    if env_file.exists():
        print("⚠️  python-dotenv not installed. Installing for better .env support...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv", "--quiet"], check=True)
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
            print(f"✅ Loaded environment from {env_file}")
        except:
            # Manual parsing as fallback
            print("ℹ️  Loading .env file manually (install python-dotenv for better support)")
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Detect environment before anything else
def is_cloud_run():
    """Check if running on Cloud Run."""
    return os.getenv('K_SERVICE') is not None

def is_docker():
    """Check if running inside Docker."""
    return os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER') == 'true'

# Add virtual environment activation for local development
venv_path = base_dir / "venv"

# Only activate venv if not in Cloud Run or Docker
if not is_cloud_run() and not is_docker() and venv_path.exists():
    # Update PATH to use venv Python
    venv_bin = venv_path / "bin"
    if venv_bin.exists():  # Unix/Mac
        os.environ['PATH'] = f"{venv_bin}:{os.environ['PATH']}"
        sys.executable = str(venv_bin / "python")
    else:  # Windows
        venv_scripts = venv_path / "Scripts"
        if venv_scripts.exists():
            os.environ['PATH'] = f"{venv_scripts};{os.environ['PATH']}"
            sys.executable = str(venv_scripts / "python.exe")

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
    # Ensure we're using the right Python interpreter
    if venv_path.exists():
        python_exe = str(venv_path / "bin" / "python") if (venv_path / "bin" / "python").exists() else sys.executable
    else:
        python_exe = sys.executable
    
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
            # Use subprocess to check with the correct Python
            result = subprocess.run(
                [python_exe, "-c", f"import {package}"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                missing_packages.append(name)
        except Exception:
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True


def setup_environment():
    """Set up environment variables and configuration."""
    backend_dir = base_dir / "backend"
    
    # Ensure directories exist
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    (backend_dir / "config").mkdir(exist_ok=True)
    
    # Set default service config path if not set
    if not os.getenv('SERVICE_CONFIG_PATH'):
        config_path = backend_dir / "config" / "services.json"
        os.environ['SERVICE_CONFIG_PATH'] = str(config_path)
    
    logger.info(f"📋 Service config: {os.getenv('SERVICE_CONFIG_PATH')}")
    
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
            logger.warning("   Set it in .env or place key in backend/config/secrets/")
            logger.warning("   Some services may not work without GCP authentication")
    else:
        # Resolve relative paths to absolute
        creds_path = Path(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
        if not creds_path.is_absolute():
            creds_path = base_dir / creds_path
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)
        
        if creds_path.exists():
            logger.info(f"🔑 Using service account key: {creds_path.name}")
        else:
            logger.warning(f"⚠️  Service account key not found: {creds_path}")
    
    # Set Google Cloud Project if specified
    if os.getenv('GOOGLE_CLOUD_PROJECT'):
        logger.info(f"☁️  Using GCP project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    
    # Set Python path
    sys.path.insert(0, str(backend_dir))
    
    # Configure logging based on environment
    if os.getenv('LOG_TO_FILE', 'true').lower() == 'true':
        log_file = log_dir / os.getenv('LOG_FILE_PATH', 'app.log')
        log_file.parent.mkdir(exist_ok=True)
        
        # Add file handler to existing logger
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    logger.info("✅ Environment setup complete")


def wait_for_service(url: str, timeout: int = 30, service_name: str = "service") -> bool:
    """Wait for a service to be ready."""
    import requests
    
    logger.info(f"⏳ Waiting for {service_name} to be ready at {url}...")
    
    for attempt in range(timeout):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {service_name.title()} is ready!")
                return True
            else:
                logger.debug(f"Attempt {attempt + 1}: Got status {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Attempt {attempt + 1}: Connection refused (service starting...)")
        except requests.exceptions.Timeout:
            logger.debug(f"Attempt {attempt + 1}: Request timeout")
        except requests.exceptions.RequestException as e:
            logger.debug(f"Attempt {attempt + 1}: Request failed: {e}")
        
        # Show progress every 10 seconds
        if (attempt + 1) % 10 == 0:
            logger.info(f"Still waiting for {service_name}... ({attempt + 1}/{timeout})")
        
        time.sleep(1)
    
    logger.error(f"❌ {service_name.title()} failed to start within {timeout} seconds")
    return False


def start_backend(modular: bool = None) -> Optional[subprocess.Popen]:
    """Start the backend server."""
    # Use environment variable if modular not specified
    if modular is None:
        modular = os.getenv('USE_MODULAR', 'false').lower() == 'true'
    
    if modular:
        # Modular backend: use backend/main:app and run from backend/ directory
        main_module = "main:app"
        working_dir = base_dir / "backend"
        logger.info("🚀 Starting modular backend server...")
        logger.info("   • Full service architecture with 16+ services")
        logger.info("   • Service registry and dynamic router management")
    else:
        # Legacy backend: use main_legacy:app and run from root directory
        main_module = "main_legacy:app"
        working_dir = base_dir
        logger.info("🚀 Starting legacy backend server...")
        logger.info("   • Simple monolithic backend")
    
    # Use venv Python if available, otherwise use system Python
    if venv_path.exists() and (venv_path / "bin" / "python").exists():
        python_exe = str(venv_path / "bin" / "python")
        logger.info(f"Using venv Python: {python_exe}")
    else:
        python_exe = sys.executable
        logger.info(f"Using system Python: {python_exe}")
        if not venv_path.exists():
            logger.warning("⚠️  Virtual environment not found. Consider creating one:")
            logger.warning("   python -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT', '8000')
    log_level = os.getenv('LOG_LEVEL', 'info')
    reload = os.getenv('RELOAD', 'true').lower() == 'true' and not is_cloud_run() and not is_docker()
    
    cmd = [
        python_exe, "-m", "uvicorn",
        main_module,
        "--host", host,
        "--port", port,
        "--log-level", log_level
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        # Start backend in background
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info(f"Working directory: {working_dir}")
        process = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        processes.append(process)
        
        # Check if process started successfully
        import time
        time.sleep(1)  # Give it a moment to start
        
        if process.poll() is not None:
            # Process has already terminated
            stdout, stderr = process.communicate()
            logger.error(f"❌ Backend process terminated immediately")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return None
        
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        return None


def start_frontend() -> Optional[subprocess.Popen]:
    """Start the Streamlit frontend."""
    logger.info("🌐 Starting frontend server...")
    
    frontend_dir = base_dir / "frontend"
    
    # Use venv Python if available, otherwise use system Python
    if venv_path.exists() and (venv_path / "bin" / "python").exists():
        python_exe = str(venv_path / "bin" / "python")
    else:
        python_exe = sys.executable
    
    # Get configuration from environment
    frontend_host = os.getenv('FRONTEND_HOST', '0.0.0.0')
    frontend_port = os.getenv('FRONTEND_PORT', '8501')
    
    cmd = [
        python_exe, "-m", "streamlit", "run",
        str(frontend_dir / "main_app.py"),
        "--server.port", frontend_port,
        "--server.address", frontend_host,
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
    backend_port = os.getenv('PORT', '8000')
    frontend_port = os.getenv('FRONTEND_PORT', '8501')
    
    urls = [
        ("Frontend UI", f"http://localhost:{frontend_port}"),
        ("Backend API", f"http://localhost:{backend_port}"),
        ("API Documentation", f"http://localhost:{backend_port}/docs")
    ]
    
    logger.info("🌐 Opening browser tabs...")
    for name, url in urls:
        try:
            webbrowser.open(url)
            logger.info(f"   📖 {name}: {url}")
        except Exception as e:
            logger.warning(f"Failed to open {name}: {e}")


def show_status_info(modular: bool = None):
    """Show application status and access information."""
    if modular is None:
        modular = os.getenv('USE_MODULAR', 'false').lower() == 'true'
    
    backend_port = os.getenv('PORT', '8000')
    frontend_port = os.getenv('FRONTEND_PORT', '8501')
    
    print(f"""
🎉 Security Agent is now running!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Access Points:
   🌐 Frontend UI:       http://localhost:{frontend_port}
   🔧 Backend API:       http://localhost:{backend_port}
   📖 API Documentation: http://localhost:{backend_port}/docs
   🩺 Health Check:      http://localhost:{backend_port}/health
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


def run_cloud_mode():
    """Run in Cloud Run compatible mode (direct uvicorn)."""
    logger.info("🚀 Starting in Cloud Run mode...")
    
    # Setup environment
    backend_dir = Path(__file__).parent / "backend"
    # Don't change directory - we'll specify the working directory in subprocess
    
    # Set service config path if not set
    if not os.getenv('SERVICE_CONFIG_PATH'):
        config_path = backend_dir / "config" / "services.json"
        os.environ['SERVICE_CONFIG_PATH'] = str(config_path)
    
    # Get port from environment (Cloud Run sets this)
    port = int(os.getenv('PORT', 8000))
    
    # Determine which backend to use
    use_modular = os.getenv('USE_MODULAR', 'false').lower() == 'true'
    
    if use_modular:
        logger.info("Using modular backend...")
        app_module = "main:app"
    else:
        logger.info("Using legacy backend...")
        app_module = "main_legacy:app"
    
    # Determine Python executable
    if venv_path.exists() and not is_docker() and not os.getenv('K_SERVICE'):
        python_exe = str(venv_path / "bin" / "python")
    else:
        python_exe = sys.executable
    
    # Run uvicorn
    cmd = [
        python_exe, "-m", "uvicorn",
        app_module,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "info",
        "--access-log"
    ]
    
    # No reload in production
    if not is_cloud_run() and not is_docker():
        cmd.append("--reload")
    
    logger.info(f"Starting server on port {port}...")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server failed: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True


def deploy_to_cloud_run(project_id: str, region: str = "us-central1"):
    """Deploy the application to Cloud Run."""
    logger.info(f"🚀 Deploying to Cloud Run...")
    logger.info(f"   Project: {project_id}")
    logger.info(f"   Region: {region}")
    
    service_name = "security-agent"
    
    # Check for gcloud
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ gcloud CLI is not installed")
        logger.error("Install from: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Set the project
    subprocess.run(["gcloud", "config", "set", "project", project_id], check=True)
    
    # Enable required APIs
    logger.info("Enabling required APIs...")
    apis = [
        "run.googleapis.com",
        "cloudbuild.googleapis.com",
        "containerregistry.googleapis.com",
        "artifactregistry.googleapis.com"
    ]
    
    for api in apis:
        subprocess.run(
            ["gcloud", "services", "enable", api, "--quiet"],
            capture_output=True
        )
    
    # Deploy using source deployment
    logger.info("Deploying application...")
    cmd = [
        "gcloud", "run", "deploy", service_name,
        "--source", ".",
        "--region", region,
        "--platform", "managed",
        "--allow-unauthenticated",
        "--port", "8000",
        "--memory", "2Gi",
        "--cpu", "2",
        "--timeout", "300",
        "--max-instances", "10",
        "--min-instances", "0",
        "--set-env-vars", f"USE_MODULAR={os.getenv('USE_MODULAR', 'true')}",
        "--set-env-vars", f"GOOGLE_CLOUD_PROJECT={project_id}",
        "--quiet"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Get the service URL
        get_url_cmd = [
            "gcloud", "run", "services", "describe", service_name,
            "--region", region,
            "--format", "value(status.url)"
        ]
        
        url_result = subprocess.run(get_url_cmd, capture_output=True, text=True, check=True)
        service_url = url_result.stdout.strip()
        
        logger.info("✅ Deployment successful!")
        logger.info(f"📍 Service URL: {service_url}")
        logger.info(f"🩺 Health Check: {service_url}/health")
        logger.info(f"📖 API Docs: {service_url}/docs")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Deployment failed: {e.stderr}")
        return False


def build_docker_image(tag: str = "security-agent:latest"):
    """Build Docker image for the application."""
    logger.info(f"🐳 Building Docker image: {tag}")
    
    # Check for Dockerfile
    if not (base_dir / "Dockerfile").exists():
        logger.error("❌ Dockerfile not found")
        return False
    
    try:
        cmd = ["docker", "build", "-t", tag, "."]
        subprocess.run(cmd, cwd=str(base_dir), check=True)
        logger.info(f"✅ Docker image built: {tag}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Docker build failed: {e}")
        return False


def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(
        description="Enhanced GCP API Security Evaluation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local Development
  python run.py                           # Run legacy backend (default)
  python run.py --modular                 # Run modular backend
  python run.py --backend-only            # Run only backend
  python run.py --frontend-only           # Run only frontend
  
  # Cloud/Production
  python run.py --cloud                   # Cloud Run mode (for container)
  python run.py --production              # Production mode (no reload)
  
  # Docker
  python run.py --docker                  # Build and run Docker locally
  python run.py --docker-build            # Build Docker image only
  
  # Deployment
  python run.py --deploy PROJECT_ID       # Deploy to Cloud Run
  python run.py --deploy PROJECT_ID --region us-west1  # Deploy to specific region
        """
    )
    
    parser.add_argument(
        '--modular',
        action='store_true',
        help='Use modular architecture (default: legacy)'
    )
    
    parser.add_argument(
        '--cloud',
        action='store_true',
        help='Run in Cloud Run mode (for containers)'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Production mode (no reload, optimized)'
    )
    
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Build and run with Docker locally'
    )
    
    parser.add_argument(
        '--docker-build',
        action='store_true',
        help='Build Docker image only'
    )
    
    parser.add_argument(
        '--deploy',
        metavar='PROJECT_ID',
        help='Deploy to Cloud Run with specified project ID'
    )
    
    parser.add_argument(
        '--region',
        default='us-central1',
        help='Cloud Run region (default: us-central1)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    parser.add_argument(
        '--frontend-only',
        action='store_true',
        help='Start only the frontend'
    )
    
    parser.add_argument(
        '--backend-only',
        action='store_true',
        help='Start only the backend'
    )
    
    args = parser.parse_args()
    
    # Cloud Run mode (when running in container)
    if args.cloud or is_cloud_run():
        return run_cloud_mode()
    
    # Deploy to Cloud Run
    if args.deploy:
        os.environ['USE_MODULAR'] = 'true' if args.modular else 'false'
        return deploy_to_cloud_run(args.deploy, args.region)
    
    # Docker build only
    if args.docker_build:
        return build_docker_image()
    
    # Docker run
    if args.docker:
        if not build_docker_image():
            return False
        return run_docker()
    
    # Production mode
    if args.production:
        os.environ['PRODUCTION'] = 'true'
    
    # Local development mode
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup environment
    setup_environment()
    
    # Check environment variable if not specified via command line
    if not args.modular and os.getenv('USE_MODULAR', 'false').lower() == 'true':
        modular = True
    else:
        modular = args.modular
    
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
            backend_port = os.getenv('PORT', '8000')
            if not wait_for_service(f"http://localhost:{backend_port}/health", service_name="backend"):
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
            frontend_port = os.getenv('FRONTEND_PORT', '8501')
            if not wait_for_service(f"http://localhost:{frontend_port}", service_name="frontend"):
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