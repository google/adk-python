import os
import sys
import subprocess
import time
import shutil

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

def start_service(name, cmd_parts, pid_file, log_file, cwd=None):
    print_status(f"Starting {name} server...")
    log_path = os.path.join("logs", log_file)
    os.makedirs("logs", exist_ok=True)

    env = os.environ.copy()
    # Ensure the current project directory is in PYTHONPATH for module imports
    # This assumes the project root is the CWD when run.py is executed
    project_root = os.getcwd()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env["PYTHONPATH"]}"
    else:
        env["PYTHONPATH"] = project_root

    try:
        with open(log_path, 'w') as log_f:
            if sys.platform == "win32":
                # Use subprocess.Popen without shell=True for better control
                # CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS for true backgrounding
                process = subprocess.Popen(cmd_parts, stdout=log_f, stderr=log_f, stdin=subprocess.DEVNULL, 
                                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS, env=env, cwd=cwd)
            else:
                # nohup equivalent for Unix-like systems
                process = subprocess.Popen(['nohup'] + cmd_parts, stdout=log_f, stderr=log_f, stdin=subprocess.DEVNULL, 
                                           preexec_fn=os.setsid if hasattr(os, 'setsid') else None, env=env, cwd=cwd)

        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        print_success(f"{name} server started (PID: {process.pid}). Log: {log_path}")
        return process.pid
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")
        sys.exit(1)

def verify_service(url, service_name, max_retries=10, delay=5):
    print_status(f"Verifying {service_name} at {url}...")
    for i in range(max_retries):
        try:
            # Use requests directly. In a real scenario, consider more robust health checks.
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print_success(f"{service_name} is running at {url}.")
                return True
        except requests.exceptions.ConnectionError:
            pass # Connection refused, keep retrying
        except Exception as e:
            print_warning(f"Error checking {service_name} health: {e}")
        
        print_status(f"Waiting for {service_name}... (attempt {i+1}/{max_retries})")
        time.sleep(delay)
    print_error(f"{service_name} failed to start after {max_retries} attempts.")
    return False

def main():
    print("🚀 Enhanced GCP Security Agent - One-Command Deployment")
    print("==================================================================")

    # Pre-flight cleanup
    print_status("Running pre-flight checks and stopping existing services...")
    stop.main() # Call the stop script's main function
    time.sleep(2) # Give a moment for processes to terminate

    # Determine if Docker workflow is requested
    if "--docker" in sys.argv:
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
        # Local development workflow
        if not check_python_version():
            sys.exit(1)
        
        create_venv()
        install_dependencies()

        # Define base command for venv execution
        python_executable = os.path.join("venv", "Scripts", "python") if sys.platform == "win32" else os.path.join("venv", "bin", "python")

        # Define paths relative to the project root
        backend_main = os.path.join("contributing", "samples", "security_agent", "backend", "main.py")
        frontend_app = os.path.join("contributing", "samples", "security_agent", "frontend", "enhanced_security_agent_app.py")
        adk_cli_module = "google.adk.cli"

        # Start Backend
        start_service(
            "Backend", 
            [python_executable, "-m", "uvicorn", f"{backend_main.replace(os.sep, '.').replace('.py', '')}:app", "--host", "0.0.0.0", "--port", "8000"], 
            "backend.pid", 
            "backend.log",
            cwd=os.path.join("contributing", "samples", "security_agent", "backend") # Run uvicorn from backend dir
        )
        
        # Start Frontend
        start_service(
            "Frontend", 
            [python_executable, "-m", "streamlit", "run", frontend_app, "--server.port", "8501", "--server.address", "0.0.0.0"], 
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
    print("\n")

if __name__ == "__main__":
    main()
