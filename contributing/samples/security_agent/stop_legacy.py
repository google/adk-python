import os
import sys
import subprocess
import time
import signal

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

def is_process_running(pid):
    """Check if a process with the given PID is running."""
    if pid is None:
        return False
    try:
        if sys.platform == "win32":
            # On Windows, taskkill with /F (force) and /T (tree kill) is used to check existence and kill
            # We use subprocess.run with check=False to avoid raising CalledProcessError for non-existent PID
            result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            return f" {pid} " in result.stdout # Check if PID appears in tasklist output
        else:
            os.kill(pid, 0) # Send no signal, just check for process existence
        return True
    except OSError:
        return False
    except subprocess.CalledProcessError: # For windows if tasklist fails (e.g., PID not found)
        return False

def terminate_process(pid):
    """Terminates a process given its PID, gracefully then forcefully if needed."""
    if not is_process_running(pid):
        return True # Process is already gone

    try:
        if sys.platform == "win32":
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.kill(pid, signal.SIGTERM) # Graceful shutdown
            time.sleep(2) # Give it some time to shut down
            if is_process_running(pid):
                os.kill(pid, signal.SIGKILL) # Forceful kill
        return True
    except Exception as e:
        print_warning(f"Failed to terminate process {pid}: {e}")
        return False

def stop_service_by_pid_file(pid_file, service_name):
    """Stop a service by reading its PID from a file."""
    full_pid_file_path = os.path.join(os.path.dirname(__file__), pid_file)
    if os.path.exists(full_pid_file_path):
        try:
            with open(full_pid_file_path, 'r') as f:
                pid = int(f.read().strip())
            
            if is_process_running(pid):
                print_status(f"Stopping {service_name} (PID: {pid})...")
                if terminate_process(pid):
                    print_success(f"{service_name} stopped.")
                else:
                    print_error(f"Failed to stop {service_name} (PID: {pid}).")
            else:
                print_status(f"{service_name} was not running (stale PID file).")
            os.remove(full_pid_file_path)
        except (ValueError, IOError) as e:
            print_warning(f"Could not read PID file {pid_file} for {service_name}: {e}")
    else:
        print_status(f"No PID file found for {service_name} at {full_pid_file_path}.")

def stop_processes_by_port(port, service_name):
    """Stop processes listening on a given port."""
    print_status(f"Checking for processes on port {port} ({service_name})...")
    pids_to_kill = []
    try:
        if sys.platform == "win32":
            # On Windows, use netstat and findstr
            command = ['netstat', '-ano']
            result = subprocess.run(command, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            for line in result.stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.strip().split()
                    try:
                        pid = int(parts[-1])
                        pids_to_kill.append(pid)
                    except ValueError:
                        pass
        else:
            # On Unix-like, use lsof
            command = ['lsof', '-ti', f':{port}']
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                pids_to_kill = [int(p) for p in result.stdout.decode().strip().splitlines() if p.strip()]
    except Exception as e:
        print_warning(f"Could not check port {port}: {e}")

    if pids_to_kill:
        print_status(f"Found processes on port {port}: {pids_to_kill}")
        for pid in pids_to_kill:
            if is_process_running(pid):
                print_status(f"Stopping process {pid} on port {port}...")
                terminate_process(pid)
        print_success(f"Stopped all processes on port {port}.")
    else:
        print_status(f"No processes found on port {port}.")

def stop_processes_by_name(pattern, service_name):
    """Stop processes matching a given name pattern."""
    print_status(f"Checking for {service_name} processes matching: {pattern}")
    pids_to_kill = []
    try:
        if sys.platform == "win32":
            # On Windows, use tasklist with image name filter
            command = ['tasklist', '/FI', f'IMAGENAME eq {pattern}*']
            result = subprocess.run(command, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            for line in result.stdout.splitlines():
                if "PID" in line and "Image Name" not in line: # Avoid header
                    parts = line.strip().split()
                    try:
                        pid = int(parts[1]) # PID is usually the second column
                        pids_to_kill.append(pid)
                    except ValueError:
                        pass
        else:
            # On Unix-like, use pgrep
            command = ['pgrep', '-f', pattern]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                pids_to_kill = [int(p) for p in result.stdout.decode().strip().splitlines() if p.strip()]
    except Exception as e:
        print_warning(f"Could not check for processes matching pattern '{pattern}': {e}")

    if pids_to_kill:
        print_status(f"Found {service_name} processes: {pids_to_kill}")
        for pid in pids_to_kill:
            if is_process_running(pid):
                print_status(f"Stopping {service_name} process {pid}...")
                terminate_process(pid)
        print_success(f"Stopped all {service_name} processes.")
    else:
        print_status(f"No {service_name} processes found.")

def cleanup_docker():
    """Stop and remove Docker containers."""
    print_status("Checking for Docker containers...")
    try:
        subprocess.run(['docker', 'info'], check=True, capture_output=True, text=True) # Check if docker daemon is running
        
        running_containers = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'], 
                                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
        
        if 'security-agent' in running_containers:
            print_status("Stopping security-agent Docker container...")
            subprocess.run(['docker', 'stop', 'security-agent'], check=True, capture_output=True, text=True)
            subprocess.run(['docker', 'rm', 'security-agent'], check=True, capture_output=True, text=True)
            print_success("Docker container cleaned up.")
        else:
            print_status("No 'security-agent' Docker container found.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print_status(f"Docker not available or daemon not running, skipping container cleanup: {e}")
    except Exception as e:
        print_error(f"An unexpected error occurred during Docker cleanup: {e}")

def main():
    print_status("🛑 Stopping ADK Services")
    print("==================================")
    
    # Stop services by PID files
    stop_service_by_pid_file("backend.pid", "Backend")
    stop_service_by_pid_file("frontend.pid", "Frontend")
    stop_service_by_pid_file("adk_web.pid", "ADK Web")
    
    # Stop services by port (in case PID files are missing or for other processes)
    stop_processes_by_port(8000, "Backend API")
    stop_processes_by_port(8501, "Streamlit Frontend")
    stop_processes_by_port(8080, "ADK Web Interface")
    
    # Stop specific processes by name (more generic for potentially non-python processes)
    stop_processes_by_name("uvicorn", "Uvicorn Backend")
    stop_processes_by_name("streamlit", "Streamlit Frontend")
    stop_processes_by_name("google.adk.cli", "ADK Web") # Can be 'python -m google.adk.cli'
    
    # Clean up Docker containers
    cleanup_docker()
    
    # Clean up log files if requested
    if "--clean-logs" in sys.argv:
        print_status("Cleaning up log files...")
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith(".log"):
                    os.remove(os.path.join(log_dir, filename))
            print_success("Log files cleaned.")
        else:
            print_status("Logs directory not found, skipping log cleanup.")
    
    print("\n")
    print_success("All ADK services have been stopped.")
    print("==================================")

if __name__ == "__main__":
    main()
