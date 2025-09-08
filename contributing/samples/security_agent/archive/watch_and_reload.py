#!/usr/bin/env python3
"""
Auto-reload watcher for Claude Security Agent
==============================================

This script watches for file changes and automatically:
1. Restarts the backend server when backend files change
2. Reloads the Streamlit frontend when frontend files change
3. Refreshes ADK agents when agent files change
4. Updates the database when data files change
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue

# Configuration
WATCH_EXTENSIONS = {'.py', '.yaml', '.yml', '.json', '.md', '.sql', '.env'}
IGNORE_PATTERNS = {'__pycache__', '.git', '.pytest_cache', '*.pyc', '.DS_Store', 'node_modules'}

class ColoredOutput:
    """Colored terminal output for better visibility."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def info(msg):
        print(f"{ColoredOutput.CYAN}[INFO]{ColoredOutput.ENDC} {msg}")
    
    @staticmethod
    def success(msg):
        print(f"{ColoredOutput.GREEN}[SUCCESS]{ColoredOutput.ENDC} {msg}")
    
    @staticmethod
    def warning(msg):
        print(f"{ColoredOutput.WARNING}[WARNING]{ColoredOutput.ENDC} {msg}")
    
    @staticmethod
    def error(msg):
        print(f"{ColoredOutput.FAIL}[ERROR]{ColoredOutput.ENDC} {msg}")
    
    @staticmethod
    def header(msg):
        print(f"\n{ColoredOutput.HEADER}{ColoredOutput.BOLD}{'='*60}{ColoredOutput.ENDC}")
        print(f"{ColoredOutput.HEADER}{ColoredOutput.BOLD}{msg}{ColoredOutput.ENDC}")
        print(f"{ColoredOutput.HEADER}{ColoredOutput.BOLD}{'='*60}{ColoredOutput.ENDC}\n")

class ServiceManager:
    """Manages backend and frontend services."""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.adk_process = None
        self.log = ColoredOutput()
        
    def start_backend(self):
        """Start the FastAPI backend server."""
        self.stop_backend()
        self.log.info("Starting backend server...")
        
        try:
            self.backend_process = subprocess.Popen(
                ["python", "run_backend.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            time.sleep(3)  # Give it time to start
            self.log.success("Backend server started on port 8000")
        except Exception as e:
            self.log.error(f"Failed to start backend: {e}")
    
    def stop_backend(self):
        """Stop the backend server."""
        if self.backend_process:
            self.log.info("Stopping backend server...")
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGTERM)
                else:
                    self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
            except:
                self.backend_process.kill()
            self.backend_process = None
            self.log.success("Backend server stopped")
    
    def start_frontend(self):
        """Start the Streamlit frontend."""
        self.stop_frontend()
        self.log.info("Starting frontend...")
        
        try:
            self.frontend_process = subprocess.Popen(
                ["python", "run_frontend.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            time.sleep(3)  # Give it time to start
            self.log.success("Frontend started on port 8501")
        except Exception as e:
            self.log.error(f"Failed to start frontend: {e}")
    
    def stop_frontend(self):
        """Stop the frontend."""
        if self.frontend_process:
            self.log.info("Stopping frontend...")
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(self.frontend_process.pid), signal.SIGTERM)
                else:
                    self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
            except:
                self.frontend_process.kill()
            self.frontend_process = None
            self.log.success("Frontend stopped")
    
    def restart_adk_web(self):
        """Restart ADK web interface."""
        self.log.info("Restarting ADK web...")
        
        # Kill existing ADK web processes
        try:
            subprocess.run(["pkill", "-f", "adk web"], capture_output=True)
            time.sleep(2)
        except:
            pass
        
        # Start ADK web from the agent directory
        agent_dir = Path(__file__).parent / "agents" / "gcp_security"
        if agent_dir.exists():
            try:
                self.adk_process = subprocess.Popen(
                    ["adk", "web"],
                    cwd=str(agent_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.log.success("ADK web restarted")
            except Exception as e:
                self.log.error(f"Failed to restart ADK web: {e}")
    
    def refresh_database(self):
        """Refresh the SQLite database."""
        self.log.info("Refreshing database...")
        try:
            subprocess.run(
                ["python", "populate_sqlite.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            self.log.success("Database refreshed")
        except subprocess.TimeoutExpired:
            self.log.warning("Database refresh timed out")
        except Exception as e:
            self.log.error(f"Failed to refresh database: {e}")

class ProjectWatcher(FileSystemEventHandler):
    """Watches for file changes in the project."""
    
    def __init__(self, service_manager):
        self.service_manager = service_manager
        self.log = ColoredOutput()
        self.change_queue = queue.Queue()
        self.last_reload = {}
        self.debounce_seconds = 2  # Prevent rapid reloads
        
        # Start the change processor thread
        self.processor_thread = threading.Thread(target=self._process_changes, daemon=True)
        self.processor_thread.start()
    
    def should_ignore(self, path):
        """Check if path should be ignored."""
        path_str = str(path)
        for pattern in IGNORE_PATTERNS:
            if pattern in path_str:
                return True
        return False
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if we should ignore this file
        if self.should_ignore(file_path):
            return
        
        # Check file extension
        if file_path.suffix not in WATCH_EXTENSIONS:
            return
        
        # Add to change queue
        self.change_queue.put(file_path)
    
    def _process_changes(self):
        """Process file changes with debouncing."""
        while True:
            try:
                file_path = self.change_queue.get(timeout=1)
                
                # Determine what needs to be reloaded
                relative_path = file_path.relative_to(Path.cwd())
                path_parts = relative_path.parts
                
                now = time.time()
                
                # Backend changes
                if 'backend' in path_parts:
                    last = self.last_reload.get('backend', 0)
                    if now - last > self.debounce_seconds:
                        self.log.warning(f"Backend file changed: {relative_path}")
                        self.service_manager.start_backend()
                        self.last_reload['backend'] = now
                
                # Frontend changes
                elif 'frontend' in path_parts or 'unified_streaming_client.py' in str(file_path):
                    last = self.last_reload.get('frontend', 0)
                    if now - last > self.debounce_seconds:
                        self.log.warning(f"Frontend file changed: {relative_path}")
                        self.service_manager.start_frontend()
                        self.last_reload['frontend'] = now
                
                # Agent changes
                elif 'agents' in path_parts or 'vertex_sqlite' in str(file_path):
                    last = self.last_reload.get('adk', 0)
                    if now - last > self.debounce_seconds:
                        self.log.warning(f"Agent file changed: {relative_path}")
                        self.service_manager.restart_adk_web()
                        self.last_reload['adk'] = now
                
                # Database or data changes
                elif 'populate_sqlite' in str(file_path) or 'data_fetcher' in str(file_path):
                    last = self.last_reload.get('database', 0)
                    if now - last > self.debounce_seconds * 5:  # Less frequent for DB
                        self.log.warning(f"Data file changed: {relative_path}")
                        self.service_manager.refresh_database()
                        self.last_reload['database'] = now
                
                # Configuration changes
                elif file_path.suffix in {'.env', '.yaml', '.yml'}:
                    self.log.warning(f"Config file changed: {relative_path}")
                    self.log.info("Restarting all services...")
                    self.service_manager.start_backend()
                    self.service_manager.start_frontend()
                    self.last_reload['backend'] = now
                    self.last_reload['frontend'] = now
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log.error(f"Error processing change: {e}")

def main():
    """Main entry point for the watcher."""
    log = ColoredOutput()
    log.header("Claude Security Agent - Auto Reload Watcher")
    
    # Initialize service manager
    service_manager = ServiceManager()
    
    # Start services initially
    log.info("Starting initial services...")
    service_manager.start_backend()
    service_manager.start_frontend()
    
    # Set up file watcher
    event_handler = ProjectWatcher(service_manager)
    observer = Observer()
    
    # Watch directories
    watch_paths = [
        Path.cwd() / "backend",
        Path.cwd() / "frontend",
        Path.cwd() / "agents",
        Path.cwd() / "config",
        Path.cwd(),  # For root level files like .env
    ]
    
    for path in watch_paths:
        if path.exists():
            observer.schedule(event_handler, str(path), recursive=True)
            log.info(f"Watching: {path}")
    
    # Start watching
    observer.start()
    log.success("File watcher started. Press Ctrl+C to stop.")
    log.info("Services will auto-reload when files change.")
    
    # Show status
    print("\n" + "="*60)
    print("🚀 Services Running:")
    print("  - Backend:  http://localhost:8000")
    print("  - Frontend: http://localhost:8501")
    print("  - ADK Web:  http://localhost:3000 (if configured)")
    print("\n📁 Watching for changes in:")
    print("  - Backend files  → Auto-restart backend")
    print("  - Frontend files → Auto-reload Streamlit")
    print("  - Agent files    → Restart ADK web")
    print("  - Config files   → Restart all services")
    print("="*60 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.warning("\nStopping watcher...")
        observer.stop()
        service_manager.stop_backend()
        service_manager.stop_frontend()
    
    observer.join()
    log.success("Watcher stopped.")

if __name__ == "__main__":
    # Check for required package
    try:
        import watchdog
    except ImportError:
        print("Installing required package: watchdog...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
        print("Package installed. Please run the script again.")
        sys.exit(1)
    
    main()