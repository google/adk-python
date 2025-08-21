#!/usr/bin/env python3
"""
Comprehensive service startup script for Security Agent.
Handles process cleanup, starts backend and frontend, and monitors health.
"""

import subprocess
import sys
import os
import time
import signal
import socket
import argparse
from pathlib import Path
from typing import Optional, Dict

class ServiceManager:
    """Manages Security Agent services with automatic cleanup and monitoring."""
    
    def __init__(self):
        self.processes = {}
        self.project_root = Path(__file__).parent
        
    def check_port(self, port: int) -> bool:
        """Check if a port is in use."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    def kill_port_process(self, port: int) -> bool:
        """Kill any process using the specified port."""
        print(f"🔍 Checking port {port}...")
        
        if not self.check_port(port):
            print(f"✅ Port {port} is free")
            return True
        
        print(f"⚠️ Port {port} is in use, killing process...")
        
        try:
            # Use lsof to find PIDs
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"  🛑 Killed process {pid}")
                        except:
                            pass
                
                time.sleep(2)
                print(f"✅ Port {port} cleared")
                return True
            
        except Exception as e:
            print(f"❌ Failed to kill process on port {port}: {e}")
            return False
        
        return True
    
    def start_backend(self, port: int = 8000) -> Optional[subprocess.Popen]:
        """Start the backend FastAPI server."""
        print("\n🚀 Starting Backend Server...")
        print("=" * 50)
        
        # Kill existing backend
        if not self.kill_port_process(port):
            print("⚠️ Warning: Could not clear backend port")
        
        try:
            # Load environment
            env = os.environ.copy()
            env_file = self.project_root / '.env'
            
            if env_file.exists():
                from dotenv import load_dotenv
                load_dotenv(env_file)
                print(f"✅ Loaded environment from {env_file}")
            
            # Start backend
            backend_dir = self.project_root / 'backend'
            process = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'main:app', 
                 '--reload', '--host', '0.0.0.0', '--port', str(port)],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # Wait for startup
            print("⏳ Waiting for backend to start...")
            for i in range(30):  # 30 second timeout
                if self.check_port(port):
                    print(f"✅ Backend started on port {port}")
                    print(f"📚 API docs: http://localhost:{port}/docs")
                    self.processes['backend'] = process
                    return process
                time.sleep(1)
                
                # Check if process died
                if process.poll() is not None:
                    print("❌ Backend process died during startup")
                    # Print last few lines of output
                    for line in process.stdout:
                        print(f"  {line.strip()}")
                    return None
            
            print("⚠️ Backend started but not responding yet")
            self.processes['backend'] = process
            return process
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return None
    
    def start_frontend(self, port: int = 8501) -> Optional[subprocess.Popen]:
        """Start the frontend Streamlit app."""
        print("\n🎨 Starting Frontend App...")
        print("=" * 50)
        
        # Kill existing frontend
        if not self.kill_port_process(port):
            print("⚠️ Warning: Could not clear frontend port")
        
        try:
            # Start frontend
            frontend_dir = self.project_root / 'frontend'
            process = subprocess.Popen(
                [sys.executable, '-m', 'streamlit', 'run', 
                 'main_app.py', '--server.port', str(port)],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for startup
            print("⏳ Waiting for frontend to start...")
            for i in range(30):  # 30 second timeout
                if self.check_port(port):
                    print(f"✅ Frontend started on port {port}")
                    print(f"🌐 App URL: http://localhost:{port}")
                    self.processes['frontend'] = process
                    return process
                time.sleep(1)
                
                # Check if process died
                if process.poll() is not None:
                    print("❌ Frontend process died during startup")
                    return None
            
            print("⚠️ Frontend started but not responding yet")
            self.processes['frontend'] = process
            return process
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return None
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        health = {}
        
        # Check backend
        health['backend'] = self.check_port(8000)
        
        # Check frontend
        health['frontend'] = self.check_port(8501)
        
        # Check processes
        for name, process in self.processes.items():
            if process:
                health[f'{name}_process'] = process.poll() is None
            else:
                health[f'{name}_process'] = False
        
        return health
    
    def monitor_services(self):
        """Monitor services and restart if needed."""
        print("\n📊 Service Monitoring Active")
        print("Press Ctrl+C to stop all services")
        print("-" * 50)
        
        try:
            while True:
                time.sleep(10)  # Check every 10 seconds
                
                health = self.health_check()
                
                # Restart backend if needed
                if not health.get('backend', False):
                    print("\n⚠️ Backend is down, restarting...")
                    self.start_backend()
                
                # Restart frontend if needed
                if not health.get('frontend', False):
                    print("\n⚠️ Frontend is down, restarting...")
                    self.start_frontend()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping all services...")
    
    def cleanup(self):
        """Clean up all processes."""
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        # Final port cleanup
        self.kill_port_process(8000)
        self.kill_port_process(8501)
        
        print("✅ All services stopped")
    
    def run(self, backend_only=False, frontend_only=False, no_monitor=False):
        """Run the service manager."""
        print("🤖 Security Agent Service Manager")
        print("=" * 60)
        
        try:
            # Start services
            if not frontend_only:
                backend = self.start_backend()
                if not backend:
                    print("❌ Backend failed to start")
                    if not frontend_only:
                        return False
            
            if not backend_only:
                frontend = self.start_frontend()
                if not frontend:
                    print("❌ Frontend failed to start")
                    if not backend_only:
                        return False
            
            # Show status
            print("\n✅ Services Started Successfully!")
            print("=" * 60)
            
            health = self.health_check()
            print("\n📊 Service Status:")
            for service, status in health.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {service}: {'Running' if status else 'Not running'}")
            
            print("\n🔗 Access Points:")
            if health.get('backend', False):
                print("  Backend API: http://localhost:8000")
                print("  API Docs: http://localhost:8000/docs")
            if health.get('frontend', False):
                print("  Frontend App: http://localhost:8501")
            
            print("\n💡 Tips:")
            print("  - Press Ctrl+C to stop all services")
            print("  - Services will auto-restart if they crash")
            print("  - Check logs above for any errors")
            
            # Monitor services
            if not no_monitor:
                self.monitor_services()
            else:
                print("\n⚠️ Monitoring disabled. Services running in background.")
                print("Run 'python start_all_services.py --stop' to stop them.")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
        
        finally:
            if not no_monitor:
                self.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Security Agent Service Manager')
    parser.add_argument('--backend-only', action='store_true', 
                       help='Start only the backend server')
    parser.add_argument('--frontend-only', action='store_true',
                       help='Start only the frontend app')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Start services without monitoring')
    parser.add_argument('--stop', action='store_true',
                       help='Stop all running services')
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    if args.stop:
        print("🛑 Stopping all services...")
        manager.cleanup()
        return
    
    success = manager.run(
        backend_only=args.backend_only,
        frontend_only=args.frontend_only,
        no_monitor=args.no_monitor
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()