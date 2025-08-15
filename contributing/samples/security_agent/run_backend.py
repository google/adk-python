#!/usr/bin/env python3
"""
Run Backend Server
Starts the GCP Security Agent backend with optional cloud deployment
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_environment():
    """Setup environment variables from .env file"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Set default environment variables
    os.environ.setdefault("PYTHONPATH", str(project_root))
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
    os.environ.setdefault("PORT", "8000")

def run_local():
    """Run backend server locally"""
    print("🚀 Starting GCP Security Agent Backend (Local)")
    print(f"Project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"Port: {os.getenv('PORT', '8000')}")
    print("")
    
    # Kill any existing backend processes
    port = os.getenv('PORT', '8000')
    print(f"🔄 Checking for existing processes on port {port}...")
    try:
        # Find and kill processes using the port
        if sys.platform == "darwin" or sys.platform == "linux":
            # macOS/Linux
            result = subprocess.run(
                f"lsof -ti:{port}", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    print(f"  Killing existing process: {pid}")
                    subprocess.run(f"kill -9 {pid}", shell=True)
                print("  ✅ Existing processes terminated")
                # Wait a moment for port to be released
                import time
                time.sleep(2)
            else:
                print("  ✅ No existing processes found")
        elif sys.platform == "win32":
            # Windows
            subprocess.run(
                f"netstat -ano | findstr :{port}", 
                shell=True
            )
            # Note: Windows requires different approach
            print("  ⚠️  Please manually close any process using port {port}")
    except Exception as e:
        print(f"  ⚠️  Could not check for existing processes: {e}")
    
    # Check for virtual environment
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Use virtual environment Python
    venv_python = "venv/bin/python" if os.path.exists("venv/bin/python") else sys.executable
    
    # Install dependencies
    print("📦 Checking dependencies...")
    pip_cmd = "venv/bin/pip" if os.path.exists("venv/bin/pip") else "pip"
    # Install quietly, suppress output unless there's an error
    result = subprocess.run([pip_cmd, "install", "-q", "-r", "backend/requirements.txt"], 
                          capture_output=True, text=True)
    if result.returncode != 0 and "already satisfied" not in result.stdout:
        print("Installing missing dependencies...")
        subprocess.run([pip_cmd, "install", "-q", "-r", "backend/requirements.txt"])
    
    # Test GCP connectivity (optional - don't fail if not available)
    print("🔧 Checking GCP connectivity...")
    test_cmd = f"""
import os
try:
    from google.cloud import storage
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    print(f'✅ Connected to GCP project: {{project}}')
except ImportError:
    print('⚠️  Some GCP libraries not available - will use available services only')
"""
    # Use venv Python for the test
    subprocess.run([venv_python, "-c", test_cmd], check=False)
    
    print("\n🚀 Starting backend server...")
    port = os.getenv('PORT', '8000')
    print(f"Access the API at: http://localhost:{port}")
    print(f"API Documentation: http://localhost:{port}/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run the server using virtual environment
    subprocess.run([
        venv_python, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", os.getenv("PORT", "8000"),
        "--reload"
    ])

def deploy_cloud():
    """Deploy backend to Google Cloud Run"""
    print("☁️  Deploying GCP Security Agent Backend to Cloud Run")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
    region = os.getenv("REGION", "us-central1")
    service_name = "gcp-security-agent"
    
    print(f"Project: {project_id}")
    print(f"Region: {region}")
    print(f"Service: {service_name}")
    print("")
    
    # Check if gcloud is installed
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: gcloud CLI not found. Please install Google Cloud SDK")
        print("Visit: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)
    
    # Enable required APIs
    print("📦 Enabling required Google Cloud APIs...")
    apis = [
        "run.googleapis.com",
        "cloudbuild.googleapis.com",
        "containerregistry.googleapis.com",
        "secretmanager.googleapis.com",
        "compute.googleapis.com",
        "iam.googleapis.com",
        "storage.googleapis.com",
        "bigquery.googleapis.com",
        "pubsub.googleapis.com",
        "container.googleapis.com",
        "cloudfunctions.googleapis.com",
        "recommender.googleapis.com",
        "securitycenter.googleapis.com"
    ]
    
    for api in apis:
        print(f"  Enabling {api}...")
        subprocess.run([
            "gcloud", "services", "enable", api,
            "--project", project_id
        ], capture_output=True)
    
    # Check if Docker is available
    print("\n🐳 Checking Docker...")
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        use_docker = True
        print("✅ Docker found - using local build")
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_docker = False
        print("⚠️  Docker not found - using Cloud Build")
    
    # Build and deploy
    if use_docker and os.path.exists("deploy/Dockerfile"):
        # Build locally with Docker
        print("\n🔨 Building Docker image locally...")
        image_name = f"gcr.io/{project_id}/{service_name}"
        
        subprocess.run([
            "docker", "build",
            "-t", image_name,
            "-f", "deploy/Dockerfile",
            "."
        ], check=True)
        
        print("\n📤 Pushing image to Container Registry...")
        subprocess.run(["docker", "push", image_name], check=True)
        
        print("\n☁️  Deploying to Cloud Run...")
        subprocess.run([
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", region,
            "--project", project_id,
            "--allow-unauthenticated",
            "--memory", "2Gi",
            "--cpu", "2",
            "--timeout", "300",
            "--max-instances", "10",
            "--min-instances", "1",
            "--set-env-vars", f"GOOGLE_CLOUD_PROJECT={project_id}",
            "--set-env-vars", f"VERTEX_AI_PROJECT_ID={project_id}",
            "--set-env-vars", f"VERTEX_AI_LOCATION={region}"
        ], check=True)
    else:
        # Use Cloud Build
        print("\n🔨 Building with Cloud Build...")
        
        # Create a simple Dockerfile if it doesn't exist
        if not os.path.exists("Dockerfile"):
            print("Creating Dockerfile...")
            dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
"""
            with open("Dockerfile", "w") as f:
                f.write(dockerfile_content)
        
        # Submit to Cloud Build and deploy
        print("\n☁️  Building and deploying with Cloud Build...")
        subprocess.run([
            "gcloud", "run", "deploy", service_name,
            "--source", ".",
            "--platform", "managed",
            "--region", region,
            "--project", project_id,
            "--allow-unauthenticated",
            "--memory", "2Gi",
            "--cpu", "2",
            "--timeout", "300",
            "--max-instances", "10",
            "--min-instances", "1",
            "--set-env-vars", f"GOOGLE_CLOUD_PROJECT={project_id}",
            "--set-env-vars", f"VERTEX_AI_PROJECT_ID={project_id}",
            "--set-env-vars", f"VERTEX_AI_LOCATION={region}"
        ], check=True)
    
    # Get service URL
    result = subprocess.run([
        "gcloud", "run", "services", "describe", service_name,
        "--platform", "managed",
        "--region", region,
        "--project", project_id,
        "--format", "value(status.url)"
    ], capture_output=True, text=True, check=True)
    
    service_url = result.stdout.strip()
    
    print("\n✅ Deployment complete!")
    print(f"🌐 Service URL: {service_url}")
    print(f"\nTest the deployment:")
    print(f"  curl {service_url}/health")
    print(f"  curl {service_url}/api/v1/asset-inventory/summary?project_id={project_id}")

def main():
    parser = argparse.ArgumentParser(description="Run GCP Security Agent Backend")
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Deploy to Google Cloud Run instead of running locally"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (local only)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Google Cloud project ID"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="Google Cloud region for deployment"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Override with command line arguments
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project
    if args.region:
        os.environ["REGION"] = args.region
    
    # Run appropriate mode
    if args.cloud:
        deploy_cloud()
    else:
        run_local()

if __name__ == "__main__":
    main()