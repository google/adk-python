#!/usr/bin/env python3
"""
Standalone GCP Security Dashboard Launcher
=========================================

Launches the executive dashboard as a standalone Streamlit application.
This provides a dedicated dashboard view for executives and security teams.

Usage:
    python run_dashboard.py              # Run on default port 8502
    python run_dashboard.py --port 8503  # Run on custom port
    python run_dashboard.py --cloud      # Run for Cloud Run deployment
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Setup environment for dashboard execution"""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Change to the project directory
    os.chdir(current_dir)
    
    # Verify dashboard module exists
    dashboard_path = current_dir / "frontend" / "dashboard.py"
    if not dashboard_path.exists():
        print(f"❌ Error: Dashboard module not found at {dashboard_path}")
        sys.exit(1)
    
    print(f"✅ Dashboard module found: {dashboard_path}")
    return current_dir

def run_dashboard(port: int = 8502, cloud_mode: bool = False):
    """Run the standalone dashboard"""
    
    setup_dir = setup_environment()
    
    # Set environment variables for dashboard
    os.environ.setdefault("DATABASE_PATH", 
                         str(setup_dir / "backend" / "cache" / "gcp_data.db"))
    
    # Streamlit command configuration
    streamlit_args = [
        "streamlit", "run", 
        str(setup_dir / "frontend" / "dashboard.py"),
        "--server.port", str(port),
        "--server.address", "0.0.0.0" if cloud_mode else "localhost",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true" if cloud_mode else "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#FF6B6B",
        "--theme.backgroundColor", "#0E1117",
        "--theme.secondaryBackgroundColor", "#262730"
    ]
    
    if cloud_mode:
        print(f"🌐 Starting GCP Security Dashboard in Cloud Mode on port {port}")
        print(f"📊 Dashboard will be available at: http://0.0.0.0:{port}")
    else:
        print(f"🖥️  Starting GCP Security Dashboard on port {port}")
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
    
    print("🔧 Dashboard Features:")
    print("   • Executive security overview")
    print("   • Interactive security findings analysis")
    print("   • Storage security metrics")
    print("   • Network security visualization")
    print("   • Asset analytics and trends")
    print("   • Real-time data pivoting")
    
    try:
        # Run Streamlit dashboard
        subprocess.run(streamlit_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard shutdown requested")
        print("✅ GCP Security Dashboard stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch GCP Security Executive Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dashboard.py                    # Default port 8502
  python run_dashboard.py --port 8503       # Custom port
  python run_dashboard.py --cloud           # Cloud Run mode
  
Environment Variables:
  DATABASE_PATH                              # SQLite database path
  GOOGLE_CLOUD_PROJECT                       # GCP project ID
  
The dashboard provides comprehensive security analytics including:
- Executive security posture overview
- Interactive security findings analysis
- Storage and network security metrics
- Asset analytics with trend visualization
- Real-time data filtering and pivoting
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8502,
        help="Port to run dashboard on (default: 8502)"
    )
    
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Run in cloud mode (binds to 0.0.0.0, headless)"
    )
    
    args = parser.parse_args()
    
    # Validate port range
    if not (1024 <= args.port <= 65535):
        print("❌ Error: Port must be between 1024 and 65535")
        sys.exit(1)
    
    print("🛡️  GCP Security Executive Dashboard")
    print("=" * 50)
    
    run_dashboard(port=args.port, cloud_mode=args.cloud)

if __name__ == "__main__":
    main()