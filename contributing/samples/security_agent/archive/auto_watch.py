#!/usr/bin/env python3
"""
ACTUAL File Watcher for Auto-Reloading Services
================================================
This is a real, working file watcher that will restart your services
when files change. No fictional commands!
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoReloader(FileSystemEventHandler):
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.last_restart = 0
        self.debounce = 2  # seconds
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only watch Python files
        if not event.src_path.endswith('.py'):
            return
            
        # Debounce rapid changes
        current_time = time.time()
        if current_time - self.last_restart < self.debounce:
            return
            
        self.last_restart = current_time
        
        # Determine what to restart
        if 'backend' in event.src_path:
            print(f"🔄 Backend file changed: {event.src_path}")
            self.restart_backend()
        elif 'frontend' in event.src_path:
            print(f"🔄 Frontend file changed: {event.src_path}")
            self.restart_frontend()
    
    def restart_backend(self):
        print("Restarting backend...")
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
        self.backend_process = subprocess.Popen(["python", "run_backend.py"])
        print("✅ Backend restarted")
    
    def restart_frontend(self):
        print("Restarting frontend...")
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
        self.frontend_process = subprocess.Popen(["python", "run_frontend.py"])
        print("✅ Frontend restarted")
    
    def start_services(self):
        print("Starting services...")
        self.restart_backend()
        self.restart_frontend()
        print("✅ All services started")
    
    def stop_services(self):
        if self.backend_process:
            self.backend_process.terminate()
        if self.frontend_process:
            self.frontend_process.terminate()
        print("✅ All services stopped")

def main():
    print("=" * 60)
    print("Auto-Reload File Watcher")
    print("=" * 60)
    
    # Install watchdog if needed
    try:
        import watchdog
    except ImportError:
        print("Installing watchdog...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
        print("Please run this script again")
        sys.exit(1)
    
    handler = AutoReloader()
    handler.start_services()
    
    observer = Observer()
    observer.schedule(handler, ".", recursive=True)
    observer.start()
    
    print("\n📁 Watching for file changes...")
    print("   - Backend files → Auto-restart backend")
    print("   - Frontend files → Auto-restart frontend")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        handler.stop_services()
    
    observer.join()

if __name__ == "__main__":
    main()