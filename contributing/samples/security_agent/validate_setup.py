#!/usr/bin/env python3
"""
Validate GCP Security Agent Setup
=================================

This script validates that the environment is properly configured
and all required components are working.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv


def check_env_file():
    """Check if .env file exists and is properly configured."""
    print("🔍 Checking .env configuration...")
    
    if not Path('.env').exists():
        print("❌ .env file not found")
        print("💡 Run: python setup.py")
        return False
    
    load_dotenv()
    
    required_vars = [
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        return False
    
    print("✅ .env file configured correctly")
    return True


def check_service_account():
    """Check if service account key file exists and is valid."""
    print("🔑 Checking service account key...")
    
    key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not key_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        return False
    
    if not os.path.exists(key_path):
        print(f"❌ Service account key file not found: {key_path}")
        return False
    
    try:
        with open(key_path, 'r') as f:
            data = json.load(f)
            if data.get('type') != 'service_account':
                print("❌ Invalid service account key file")
                return False
    except Exception as e:
        print(f"❌ Error reading service account key: {e}")
        return False
    
    print("✅ Service account key file valid")
    return True


def check_gcp_auth():
    """Check if GCP authentication is working."""
    print("🔐 Testing GCP authentication...")
    
    try:
        from google.cloud import asset_v1
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        client = asset_v1.AssetServiceClient()
        parent = f"projects/{project_id}"
        
        # Test with minimal request
        request = asset_v1.ListAssetsRequest(parent=parent, page_size=1)
        results = list(client.list_assets(request=request))
        
        print("✅ GCP authentication working")
        return True
        
    except Exception as e:
        print(f"❌ GCP authentication failed: {e}")
        print()
        print("Common fixes:")
        print("• Check service account permissions")
        print("• Enable required APIs in your project")
        print("• Verify project ID is correct")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print("📦 Checking Python dependencies...")
    
    required_packages = [
        'google-cloud-asset',
        'google-cloud-storage', 
        'google-cloud-iam',
        'google-cloud-monitoring',
        'google-cloud-secret-manager',
        'google-cloud-aiplatform',
        'fastapi',
        'streamlit',
        'sqlite3'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True


def check_database():
    """Check if database directory exists."""
    print("💾 Checking database setup...")
    
    db_path = Path("backend/cache")
    if not db_path.exists():
        print("📁 Creating database directory...")
        db_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Database directory ready")
    return True


def run_basic_test():
    """Run a basic integration test."""
    print("🧪 Running basic integration test...")
    
    try:
        # Test data fetcher import
        sys.path.append('backend')
        from services.data_fetcher import DataFetcher
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        fetcher = DataFetcher(project_id)
        
        # Test database initialization
        stats = fetcher.get_summary_stats()
        
        print("✅ Basic integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    print()
    print("📋 System Information")
    print("-" * 30)
    print(f"Project ID: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"Backend URL: {os.getenv('BACKEND_URL', 'http://localhost:8000')}")
    print(f"Frontend URL: {os.getenv('FRONTEND_URL', 'http://localhost:8501')}")
    print(f"Service Account: {os.path.basename(os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set'))}")
    print()


def main():
    """Main validation function."""
    print("🔐 GCP Security Agent Validation")
    print("=" * 40)
    print()
    
    checks = [
        check_env_file,
        check_service_account,
        check_dependencies,
        check_database,
        check_gcp_auth,
        run_basic_test
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
            passed += 1
        print()
    
    print_system_info()
    
    if passed == total:
        print("🎉 All checks passed! System is ready.")
        print()
        print("Next steps:")
        print("1. python run_backend.py")
        print("2. python run_frontend.py") 
        print("3. Open http://localhost:8501")
    else:
        print(f"❌ {total - passed} checks failed. Please fix issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()