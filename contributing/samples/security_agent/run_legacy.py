import os
import sys
import subprocess
import time
import shutil
import requests # For verify_service
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
if os.path.exists('.env'):
    load_dotenv(dotenv_path='.env')
    print_status("Loaded environment variables from .env file.")
else:
    print_warning(".env file not found. GCP integrations might not work as expected.")

# Import the stop script for pre-flight cleanup
import stop # Assuming stop.py is in the same directory

# Define colors for console output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m' # No Color

def print_status(message):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def start_service(service_name, command, pid_file, log_file, cwd=None):
    """Start a service with the given command and track its PID."""
    print_status(f"Starting {service_name}...")
    
    # Ensure log directory exists
    log_dir = os.path.join("contributing", "samples", "security_agent", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Full paths for PID and log files
    full_pid_file = os.path.join(os.path.dirname(__file__), pid_file)
    full_log_file = os.path.join(log_dir, log_file)
    
    # Set up environment with PYTHONPATH and loaded .env variables
    env = os.environ.copy()
    project_root = os.getcwd()
    
    # Add project root to PYTHONPATH
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
        
    # Manually load from .env again to ensure all variables are in this process's environment
    load_dotenv(dotenv_path=os.path.join(project_root, '.env'), override=True)
    
    # Update the environment for the subprocess with all current os.environ values
    env.update(os.environ)

    try:
        # Start the service with proper process management
        with open(full_log_file, 'a') as log:
            if sys.platform == "win32":
                # Windows: Use process groups for proper backgrounding
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    env=env,
                    cwd=cwd
                )
            else:
                # Unix-like: Use nohup and process session for proper backgrounding
                process = subprocess.Popen(
                    ['nohup'] + command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                    env=env,
                    cwd=cwd
                )
        
        # Write PID to file
        with open(full_pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print_success(f"{service_name} started (PID: {process.pid})")
        print_status(f"{service_name} log: logs/{log_file}")
        
        # Give the process a moment to start
        time.sleep(2)
        
    except Exception as e:
        print_error(f"Failed to start {service_name}: {e}")
        return False
    
    return True

def verify_service(url, service_name, timeout=10, max_attempts=5):
    """Verify that a service is running by checking its endpoint."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                print_success(f"{service_name} is running at {url}")
                return True
            else:
                print_warning(f"{service_name} returned status {response.status_code}")
        except requests.exceptions.RequestException:
            if attempt < max_attempts:
                print_status(f"Waiting for {service_name}... (attempt {attempt}/{max_attempts})")
                time.sleep(2)
            else:
                print_warning(f"{service_name} may not have started correctly")
                return False
    return False

def verify_gcp_integrations():
    """Verify that GCP integrations are working properly."""
    print_status("Verifying GCP service integrations...")
    
    integration_endpoints = [
        ("http://localhost:8000/api/v1/security/health", "Security Center Integration"),
        ("http://localhost:8000/api/v1/tracing/health", "Cloud Trace Integration"),
        ("http://localhost:8000/api/v1/monitoring/monitoring-health", "Cloud Monitoring Integration"),
        ("http://localhost:8000/api/v1/apihub/health", "API Hub Integration"),
        ("http://localhost:8000/api/v1/cloud-logs/health", "Cloud Logging Integration")
    ]
    
    integration_results = []
    for endpoint, integration_name in integration_endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("healthy"):
                    print_success(f"✅ {integration_name} is healthy")
                    integration_results.append(True)
                else:
                    print_warning(f"⚠️ {integration_name} is not healthy: {data.get('error', 'Unknown error')}")
                    integration_results.append(False)
            else:
                print_warning(f"⚠️ {integration_name} health check failed (HTTP {response.status_code})")
                integration_results.append(False)
        except requests.exceptions.RequestException:
            print_warning(f"⚠️ Could not check {integration_name} health (service may not be ready)")
            integration_results.append(False)
    
    healthy_count = sum(integration_results)
    total_count = len(integration_results)
    
    if healthy_count == total_count:
        print_success(f"🎉 All {total_count} GCP integrations are healthy!")
    elif healthy_count > 0:
        print_warning(f"⚠️ {healthy_count}/{total_count} GCP integrations are healthy. Some features may use mock data.")
    else:
        print_warning("⚠️ No GCP integrations are healthy. All features will use mock data.")
    
    print_status("💡 To enable real GCP integrations:")
    print("   1. Ensure all required APIs are enabled (run script will do this)")
    print("   2. Set up Application Default Credentials: gcloud auth application-default login")
    print("   3. Set GOOGLE_CLOUD_PROJECT environment variable")
    print("   4. Ensure service account has proper permissions")
    
    return healthy_count > 0

def command_exists(cmd):
    return shutil.which(cmd) is not None

def check_python_version():
    print_status("Checking Python version...")
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} is compatible.")
        return True
    else:
        print_error(f"Python 3.8+ is required, found {sys.version_info.major}.{sys.version_info.minor}.")
        return False

def create_venv():
    print_status("Creating virtual environment...")
    if not os.path.exists("venv"):
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print_success("Virtual environment created.")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print_status("Virtual environment already exists.")

def activate_venv_command():
    if sys.platform == "win32":
        return os.path.join("venv", "Scripts", "activate")
    else:
        return os.path.join("venv", "bin", "activate")

def install_dependencies():
    print_status("Installing dependencies...")
    pip_executable = os.path.join("venv", "Scripts", "pip") if sys.platform == "win32" else os.path.join("venv", "bin", "pip")

    try:
        # Upgrade pip
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True, capture_output=True, text=True)
        print_success("pip upgraded.")

        # Install main dependencies
        main_reqs_path = os.path.join("contributing", "samples", "security_agent", "requirements.txt")
        subprocess.run([pip_executable, "install", "-r", main_reqs_path], check=True, capture_output=True, text=True)
        print_success("Main dependencies installed.")

        # Install backend dependencies
        backend_reqs_path = os.path.join("contributing", "samples", "security_agent", "backend", "requirements.txt")
        subprocess.run([pip_executable, "install", "-r", backend_reqs_path], check=True, capture_output=True, text=True)
        print_success("Backend dependencies installed.")

    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError as e:
        print_error(f"Required requirements file not found: {e}. Please ensure 'contributing/samples/security_agent/requirements.txt' and 'contributing/samples/security_agent/backend/requirements.txt' exist.")
        sys.exit(1)



def enable_gcp_api(project_id, service_name):
    print_status(f"Enabling {service_name} API for project {project_id}...")
    try:
        # --async makes the command non-blocking and suitable for scripts
        subprocess.run(['gcloud', 'services', 'enable', service_name, '--project', project_id, '--async'], check=True, capture_output=True, text=True)
        print_success(f"Enabled {service_name} API.")
    except subprocess.CalledProcessError as e:
        if "ALREADY_ENABLED" in e.stderr.upper() or "ALREADY ENABLED" in e.stderr.upper(): # Check for already enabled messages
            print_warning(f"{service_name} API is already enabled for project {project_id}.")
        elif "PERMISSION_DENIED" in e.stderr.upper():
            print_error(f"Permission denied to enable {service_name} API. Please ensure the authenticated account has 'serviceusage.services.enable' permission on project {project_id}. Details: {e.stderr.strip()}")
            sys.exit(1)
        else:
            print_error(f"Failed to enable {service_name} API: {e.stderr}")
            sys.exit(1)
    except Exception as e:
        print_error(f"An unexpected error occurred while enabling {service_name} API: {e}")
        sys.exit(1)

def provision_secret_manager_secret(project_id, secret_name, secret_value, description="Secret provisioned by ADK setup script"):
    print_status(f"Provisioning Secret Manager secret '{secret_name}' in project {project_id}...")
    try:
        # Check if secret already exists
        subprocess.run(['gcloud', 'secrets', 'describe', secret_name, '--project', project_id, '--quiet'], check=True, capture_output=True, text=True)
        print_warning(f"Secret '{secret_name}' already exists in project {project_id}.")
    except subprocess.CalledProcessError:
        # Secret does not exist, create it
        try:
            print_status(f"Creating secret '{secret_name}'...")
            subprocess.run(['gcloud', 'secrets', 'create', secret_name, '--project', project_id, '--data-file=-', '--description', description], input=secret_value, text=True, check=True, capture_output=True)
            print_success(f"Secret '{secret_name}' created.")
            # Add a version to the secret
            print_status(f"Adding secret version to '{secret_name}'...")
            subprocess.run(['gcloud', 'secrets', 'versions', 'add', secret_name, '--data-file=-', '--project', project_id], input=secret_value, text=True, check=True, capture_output=True)
            print_success(f"Secret version added for '{secret_name}'.")
        except subprocess.CalledProcessError as e:
            if "PERMISSION_DENIED" in e.stderr.upper():
                print_error(f"Permission denied to create/manage secret '{secret_name}'. Ensure the account has 'secretmanager.secrets.create' and 'secretmanager.versions.add' permissions on project {project_id}. Details: {e.stderr.strip()}")
            else:
                print_error(f"Failed to create secret '{secret_name}': {e.stderr}")
            sys.exit(1)
    except Exception as e:
        print_error(f"An unexpected error occurred while provisioning secret '{secret_name}': {e}")
        sys.exit(1)

def provision_cloud_storage_bucket(project_id, bucket_name, location='US', default_class='STANDARD'):
    print_status(f"Provisioning Cloud Storage bucket '{bucket_name}' in project {project_id}...")
    try:
        # Check if bucket exists. gsutil ls gs://bucket_name returns 0 if exists, non-zero if not.
        subprocess.run(['gsutil', 'ls', f'gs://{bucket_name}'], check=True, capture_output=True, text=True)
        print_warning(f"Bucket '{bucket_name}' already exists.")
    except subprocess.CalledProcessError:
        # Bucket does not exist, create it
        try:
            print_status(f"Creating bucket '{bucket_name}'...")
            subprocess.run(['gsutil', 'mb', '-p', project_id, '-l', location, '-c', default_class, f'gs://{bucket_name}'], check=True, capture_output=True, text=True)
            print_success(f"Bucket '{bucket_name}' created.")
        except subprocess.CalledProcessError as e:
            if "PERMISSION_DENIED" in e.stderr.upper():
                print_error(f"Permission denied to create bucket '{bucket_name}'. Ensure the account has 'storage.buckets.create' permission on project {project_id}. Details: {e.stderr.strip()}")
            else:
                print_error(f"Failed to create bucket '{bucket_name}': {e.stderr}")
            sys.exit(1)
    except Exception as e:
        print_error(f"An unexpected error occurred while provisioning bucket '{bucket_name}': {e}")
        sys.exit(1)

def provision_gcp_resources(project_id):
    print("\n")
    print_status(f"⚙️ Starting GCP Resource Provisioning for project: {project_id}")
    print("==================================================================")

    if not command_exists("gcloud"):
        print_error("'gcloud' command not found. Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)
    
    # Optional: Check gcloud authentication and project configuration
    try:
        # Verify if gcloud is authenticated and project is set
        # Try to get default project from gcloud directly for provisioning context
        gcloud_project_output = subprocess.run(['gcloud', 'config', 'get-value', 'project'], check=True, capture_output=True, text=True).stdout.strip()
        if not gcloud_project_output:
            print_error(f"gcloud default project is not set. Please run 'gcloud config set project {project_id}' or ensure GOOGLE_CLOUD_PROJECT is set and gcloud is authenticated.")
            sys.exit(1)
        if gcloud_project_output != project_id:
            print_warning(f"gcloud default project is '{gcloud_project_output}', but provisioning target is '{project_id}'. Ensure this is intended.")

    except Exception as e:
        print_error(f"Failed to verify gcloud setup for provisioning. Ensure `gcloud auth application-default login` has been run and project is set. Error: {e}")
        sys.exit(1)


    # 1. Enable required APIs
    print_status("Enabling necessary Google Cloud APIs...")
    required_apis = [
        # Core infrastructure APIs
        'aiplatform.googleapis.com',          # For Vertex AI models and agent chat
        'secretmanager.googleapis.com',       # For Secret Manager
        'cloudresourcemanager.googleapis.com',# For project listing and resource management
        'serviceusage.googleapis.com',        # For listing/enabling services
        'storage.googleapis.com',             # For Cloud Storage
        'compute.googleapis.com',             # For Compute Engine interactions
        'container.googleapis.com',           # For GKE interactions
        
        # Security and monitoring APIs (NEW INTEGRATIONS)
        'securitycenter.googleapis.com',      # For Security Center findings and incident response
        'iam.googleapis.com',                 # For IAM Policy Analyzer and permissions
        'logging.googleapis.com',             # For Cloud Logging and usage analytics
        'monitoring.googleapis.com',          # For Cloud Monitoring and performance metrics
        'cloudtrace.googleapis.com',          # For Cloud Trace and distributed tracing
        
        # API management and discovery
        'apihub.googleapis.com',              # For API Hub discovery and management
        
        # Additional services that may be used
        'cloudasset.googleapis.com',          # For Cloud Asset Inventory (compliance)
        'cloudkms.googleapis.com',            # For Cloud KMS (security recommendations)
        'dns.googleapis.com',                 # For Cloud DNS (infrastructure analysis)
        'networkmanagement.googleapis.com',  # For network connectivity analysis
    ]
    for api in required_apis:
        enable_gcp_api(project_id, api)
    print_success("All required APIs checked/enabled.")

    # 2. Provision Secret Manager Secrets (Example - uncomment and configure if needed)
    print_status("Checking and provisioning Secret Manager secrets (if not existing)...")
    # Example: provision_secret_manager_secret(project_id, 'my-api-key', 'your-api-key-value')
    # Replace with actual secrets your application needs to store/retrieve
    # For instance, if your APIHubService uses a secret named 'api-hub-credentials':
    # api_hub_secret_value = '{"client_id": "your_client_id", "client_secret": "your_client_secret"}'
    # provision_secret_manager_secret(project_id, 'api-hub-credentials', api_hub_secret_value, "API Hub credentials for ADK agent")

    # 3. Provision Cloud Storage Buckets (Example - uncomment and configure if needed)
    print_status("Checking and provisioning Cloud Storage buckets (if not existing)...")
    # Example: provision_cloud_storage_bucket(project_id, 'my-data-bucket')
    # Replace with actual buckets your application needs for data persistence
    # For instance, if your MSA service stores parsed documents in a bucket:
    # msa_data_bucket_name = f'{project_id}-msa-parsed-data'
    # provision_cloud_storage_bucket(project_id, msa_data_bucket_name, location='US-CENTRAL1')

    print("\n")
    print_success("GCP resource provisioning steps completed.")
    print_warning("Review the output above for any permissions errors or failed steps. Manual steps may be required for certain resources.")
    print_warning("Note: Base Vertex AI models like 'gemini-2.0-flash-exp' are generally available by default and are not provisioned here.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced GCP Security Agent.")
    parser.add_argument("--docker", action="store_true", help="Run in Docker mode.")
    # Removed --provision-gcp flag
    args = parser.parse_args()

    print("🚀 Enhanced GCP Security Agent - One-Command Deployment")
    print("==================================================================")

    # Pre-flight cleanup
    print_status("Running pre-flight checks and stopping existing services...")
    stop.main() # Call the stop script's main function
    time.sleep(2) # Give a moment for processes to terminate

    # GCP provisioning is now default for local workflow (if not in Docker mode)
    if not args.docker:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            print_error("GOOGLE_CLOUD_PROJECT environment variable must be set for local development and GCP provisioning.")
            sys.exit(1)
        if not provision_gcp_resources(project_id):
            print_error("GCP resource provisioning failed during default setup. Exiting.")
            sys.exit(1)

    # Determine if Docker workflow is requested
    if args.docker:
        if not command_exists("docker"):
            print_error("Docker is not installed or not in PATH.")
            print_status("Please install Docker from https://www.docker.com/get-started and ensure it's running.")
            sys.exit(1)
        if sys.platform == "win32":
            # Ensure Docker Desktop is running on Windows
            try:
                subprocess.run(['docker', 'info'], check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            except subprocess.CalledProcessError:
                print_error("Docker daemon is not running. Please start Docker Desktop.")
                sys.exit(1)
        
        print_status("Building Docker image...")
        try:
            # Use the directory containing the Dockerfile as context
            dockerfile_dir = os.path.join("contributing", "samples", "security_agent")
            subprocess.run(['docker', 'build', '-t', 'security-agent', dockerfile_dir], check=True)
            print_success("Docker image built successfully.")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to build Docker image: {e.stderr}")
            sys.exit(1)

        print_status("Running Docker container...")
        try:
            subprocess.run(['docker', 'run', '-p', '8000:8000', '-p', '8501:8501', '-d', '--name', 'security-agent', 'security-agent'], check=True)
            print_success("Docker container is running.")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to run Docker container: {e.stderr}")
            sys.exit(1)
        print_status("Note: When running with Docker, logs are usually accessed via `docker logs security-agent`.")
        print_status("Use `docker stop security-agent` and `docker rm security-agent` to stop and remove the container.")

    else:
        # Local development workflow (GCP provisioning handled above)

        
        if not check_python_version():
            sys.exit(1)
        
        create_venv()
        install_dependencies()

        # Define base command for venv execution - use absolute path
        python_executable = os.path.join(os.getcwd(), "venv", "Scripts", "python") if sys.platform == "win32" else os.path.join(os.getcwd(), "venv", "bin", "python")

        # Define paths relative to the project root
        backend_main = os.path.join("contributing", "samples", "security_agent", "backend", "main.py")
        frontend_app = os.path.join("contributing", "samples", "security_agent", "frontend", "enhanced_security_agent_app.py")
        adk_cli_module = "google.adk.cli"

        # Start Backend
        start_service(
            "Backend", 
            [python_executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"], 
            "backend.pid", 
            "backend.log",
            cwd=os.path.join("contributing", "samples", "security_agent", "backend") # Run uvicorn from backend dir
        )
        
        # Start Frontend
        start_service(
            "Frontend", 
            [python_executable, "-m", "streamlit", "run", "frontend/enhanced_security_agent_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"], 
            "frontend.pid", 
            "frontend.log",
            cwd=os.path.join("contributing", "samples", "security_agent") # Run streamlit from security_agent dir
        )

        # Start ADK Web Interface
        start_service(
            "ADK Web Interface", 
            [python_executable, "-m", adk_cli_module, "web", "--port", "8080"], 
            "adk_web.pid", 
            "adk_web.log",
            cwd=os.path.join("contributing", "samples", "security_agent") # Run adk cli from security_agent dir
        )
        
        # Verify services (using requests, which is cross-platform)
        print_status("Verifying services...")
        backend_ok = verify_service("http://localhost:8000/health", "Backend")
        frontend_ok = verify_service("http://localhost:8501", "Frontend")
        adk_web_ok = verify_service("http://localhost:8080", "ADK Web")

        if not (backend_ok and frontend_ok and adk_web_ok):
            print_error("One or more services failed to start. Check logs/ for details.")
            # sys.exit(1) # Keep services running for debugging, but indicate failure
        else:
            print_success("All services verified as running.")
        
        # Additional verification for GCP integrations
        if backend_ok:
            print_status("\nVerifying GCP integrations...")
            gcp_integrations_ok = verify_gcp_integrations()
            if gcp_integrations_ok:
                print_success("GCP integrations are working! You'll see real data from your GCP project.")
            else:
                print_warning("GCP integrations are not fully configured. The app will work with demo data.")

    # Show startup info
    print("\n")
    print("🎉 Enhanced GCP Security Agent is Ready!")
    print("====================================================")
    print("\n🌐 Access Points:")
    print("   • Frontend (Streamlit): http://localhost:8501")
    print("   • Backend API: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • ADK Web Interface: http://localhost:8080")
    print("\n🚀 Quick Start:")
    print("   1. Open http://localhost:8501 in your browser")
    print("   2. Use the ADK Chat or explore other features")
    print("   3. Access native ADK interface at http://localhost:8080")
    print("\n💡 Press Ctrl+C to stop all services")
    print("\n")
    
    # Keep the script running to maintain services
    try:
        print_status("Services are running. Press Ctrl+C to stop...")
        while True:
            time.sleep(60)  # Check every minute if services are still running
            # Optional: Add health checks here
    except KeyboardInterrupt:
        print_status("\nStopping all services...")
        # Call the stop script that was already imported
        stop.main()
        print_success("All services stopped. Goodbye!")

if __name__ == "__main__":
    main()
