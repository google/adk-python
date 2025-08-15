#!/usr/bin/env python3
"""
AI Developer Helper Script
Assists AI coders with common development tasks
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

class AIDevHelper:
    """Helper utilities for AI coders working on this project"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "frontend"
        
    def check_environment(self) -> Dict[str, bool]:
        """Check if development environment is properly set up"""
        checks = {
            "python_version": sys.version_info >= (3, 8),
            "venv_active": "VIRTUAL_ENV" in os.environ,
            "backend_exists": self.backend_dir.exists(),
            "frontend_exists": self.frontend_dir.exists(),
            "env_file": (self.project_root / ".env").exists(),
            "cache_dir": (self.project_root / "cache").exists(),
        }
        
        # Check for required Python packages
        try:
            import fastapi
            checks["fastapi_installed"] = True
        except ImportError:
            checks["fastapi_installed"] = False
            
        try:
            import streamlit
            checks["streamlit_installed"] = True
        except ImportError:
            checks["streamlit_installed"] = False
            
        return checks
    
    def setup_environment(self):
        """Set up the development environment"""
        print("🚀 Setting up development environment...")
        
        # Create necessary directories
        dirs_to_create = [
            self.project_root / "cache" / "assets",
            self.project_root / "logs",
            self.project_root / "tests",
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_template = """# GCP Configuration
GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# API Configuration  
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:8501

# Feature Flags
ENABLE_MOCK_DATA=true
ENABLE_CACHE=true
CACHE_TTL_SECONDS=300

# Logging
LOG_LEVEL=INFO
"""
            env_file.write_text(env_template)
            print("✅ Created .env file with defaults")
        
        print("✅ Environment setup complete!")
    
    def analyze_errors(self) -> List[Dict[str, str]]:
        """Analyze recent errors from logs"""
        errors = []
        log_patterns = [
            "ERROR",
            "CRITICAL", 
            "Exception",
            "Traceback",
            "Failed",
            "not found",
            "not available"
        ]
        
        # Check backend logs
        backend_log = self.project_root / "logs" / "backend.log"
        if backend_log.exists():
            with open(backend_log, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    for pattern in log_patterns:
                        if pattern in line:
                            errors.append({
                                "file": "backend.log",
                                "line": i + 1,
                                "error": line.strip(),
                                "pattern": pattern
                            })
        
        return errors
    
    def generate_mock_data(self, data_type: str) -> Dict:
        """Generate mock data for testing"""
        mock_data = {
            "assets": {
                "total_assets": 150,
                "asset_breakdown": {
                    "Compute Instances": 45,
                    "Storage Buckets": 30,
                    "IAM Accounts": 25,
                    "Networks": 10,
                    "Databases": 15,
                    "Cloud Functions": 10,
                    "BigQuery Datasets": 5,
                    "Pub/Sub Topics": 5,
                    "GKE Clusters": 3,
                    "Cloud Run Services": 2
                },
                "high_risk_assets": ["bucket-public-123", "instance-exposed-456"],
                "security_findings": [
                    {"severity": "HIGH", "category": "PUBLIC_ACCESS", "resource": "bucket-123"},
                    {"severity": "MEDIUM", "category": "ENCRYPTION", "resource": "instance-456"}
                ]
            },
            "recommendations": [
                {
                    "id": "rec-1",
                    "title": "Enable bucket encryption",
                    "severity": "high",
                    "affected_resources": ["bucket-123", "bucket-456"]
                },
                {
                    "id": "rec-2", 
                    "title": "Restrict IAM permissions",
                    "severity": "critical",
                    "affected_resources": ["project-owner-role"]
                }
            ],
            "iam": {
                "users": ["user1@example.com", "user2@example.com"],
                "service_accounts": ["sa1@project.iam", "sa2@project.iam"],
                "roles": ["owner", "editor", "viewer"],
                "risky_bindings": [
                    {"member": "allUsers", "role": "viewer"},
                    {"member": "user1@example.com", "role": "owner"}
                ]
            }
        }
        
        return mock_data.get(data_type, {})
    
    def check_service_availability(self) -> Dict[str, bool]:
        """Check which GCP services are available"""
        services = {}
        
        # Check if services can be imported
        gcp_services = [
            ("compute", "google.cloud.compute_v1"),
            ("storage", "google.cloud.storage"),
            ("iam", "google.iam.credentials_v1"),
            ("asset", "google.cloud.asset_v1"),
            ("recommender", "google.cloud.recommender_v1"),
            ("securitycenter", "google.cloud.securitycenter_v1"),
            ("functions", "google.cloud.functions_v1"),
            ("bigquery", "google.cloud.bigquery"),
            ("pubsub", "google.cloud.pubsub_v1"),
            ("container", "google.cloud.container_v1"),
            ("run", "google.cloud.run_v2"),
        ]
        
        for service_name, module_name in gcp_services:
            try:
                __import__(module_name)
                services[service_name] = True
            except ImportError:
                services[service_name] = False
        
        return services
    
    def generate_test_data_file(self):
        """Generate test data JSON file for development"""
        test_data = {
            "project_id": "test-project-123",
            "assets": self.generate_mock_data("assets"),
            "recommendations": self.generate_mock_data("recommendations"),
            "iam": self.generate_mock_data("iam"),
            "timestamp": "2024-01-15T12:00:00Z",
            "api_metadata": {
                "source": "mock",
                "call_duration": 0.5
            }
        }
        
        output_file = self.project_root / "cache" / "test_data.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✅ Generated test data file: {output_file}")
        return output_file
    
    def quick_test(self) -> bool:
        """Run quick tests to verify system is working"""
        print("🧪 Running quick tests...")
        
        tests_passed = True
        
        # Test 1: Check if backend can start
        try:
            result = subprocess.run(
                ["python", "-c", "from backend.main import app; print('OK')"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.project_root
            )
            if result.returncode == 0:
                print("✅ Backend imports successfully")
            else:
                print(f"❌ Backend import failed: {result.stderr}")
                tests_passed = False
        except Exception as e:
            print(f"❌ Backend test failed: {e}")
            tests_passed = False
        
        # Test 2: Check if frontend can start
        try:
            result = subprocess.run(
                ["python", "-c", "from frontend.main_app import main; print('OK')"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.project_root
            )
            if result.returncode == 0:
                print("✅ Frontend imports successfully")
            else:
                print(f"❌ Frontend import failed: {result.stderr}")
                tests_passed = False
        except Exception as e:
            print(f"❌ Frontend test failed: {e}")
            tests_passed = False
        
        return tests_passed
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 AI Developer Helper Commands:

1. Check Environment:
   python ai_dev_helper.py check
   
2. Setup Environment:
   python ai_dev_helper.py setup
   
3. Analyze Errors:
   python ai_dev_helper.py errors
   
4. Generate Mock Data:
   python ai_dev_helper.py mock [assets|recommendations|iam]
   
5. Check Service Availability:
   python ai_dev_helper.py services
   
6. Generate Test Data File:
   python ai_dev_helper.py testdata
   
7. Run Quick Tests:
   python ai_dev_helper.py test
   
8. Show This Help:
   python ai_dev_helper.py help
"""
        print(help_text)

def main():
    """Main entry point"""
    helper = AIDevHelper()
    
    if len(sys.argv) < 2:
        helper.show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "check":
        print("🔍 Checking environment...")
        checks = helper.check_environment()
        for check, status in checks.items():
            emoji = "✅" if status else "❌"
            print(f"{emoji} {check}: {status}")
    
    elif command == "setup":
        helper.setup_environment()
    
    elif command == "errors":
        print("🔍 Analyzing errors...")
        errors = helper.analyze_errors()
        if errors:
            print(f"Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10
                print(f"  - {error['file']}:{error['line']} - {error['pattern']}")
                print(f"    {error['error'][:100]}...")
        else:
            print("✅ No errors found!")
    
    elif command == "mock":
        data_type = sys.argv[2] if len(sys.argv) > 2 else "assets"
        mock_data = helper.generate_mock_data(data_type)
        print(json.dumps(mock_data, indent=2))
    
    elif command == "services":
        print("🔍 Checking GCP service availability...")
        services = helper.check_service_availability()
        for service, available in services.items():
            emoji = "✅" if available else "❌"
            print(f"{emoji} {service}: {'Available' if available else 'Not Available (using mock)'}")
    
    elif command == "testdata":
        helper.generate_test_data_file()
    
    elif command == "test":
        if helper.quick_test():
            print("✅ All quick tests passed!")
        else:
            print("❌ Some tests failed - check output above")
    
    elif command == "help":
        helper.show_help()
    
    else:
        print(f"❌ Unknown command: {command}")
        helper.show_help()

if __name__ == "__main__":
    main()