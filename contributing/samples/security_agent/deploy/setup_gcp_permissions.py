#!/usr/bin/env python3
"""
setup_gcp_permissions.py - GCP Security Agent Setup Script

This script automates the setup of required GCP APIs and service account permissions
for the GCP Security Agent application.

Usage:
    python setup_gcp_permissions.py --project-id YOUR_PROJECT_ID

Requirements:
    pip install google-cloud-resource-manager google-cloud-iam google-cloud-service-usage
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    from google.cloud import resourcemanager_v3
    from google.cloud import iam
    from google.cloud import service_usage_v1
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("📦 Install with: pip install google-cloud-resource-manager google-cloud-iam google-cloud-service-usage")
    sys.exit(1)


class GCPSecurityAgentSetup:
    """Setup class for GCP Security Agent permissions and APIs."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.service_account_name = "security-agent"
        self.service_account_email = f"{self.service_account_name}@{project_id}.iam.gserviceaccount.com"
        
        # Required APIs for the Security Agent
        self.required_apis = [
            "cloudresourcemanager.googleapis.com",
            "serviceusage.googleapis.com", 
            "iam.googleapis.com",
            "iamcredentials.googleapis.com",
            "securitycenter.googleapis.com",
            "cloudkms.googleapis.com",
            "secretmanager.googleapis.com",
            "monitoring.googleapis.com",
            "logging.googleapis.com",
            "cloudtrace.googleapis.com",
            "clouderrorreporting.googleapis.com",
            "compute.googleapis.com",
            "container.googleapis.com",
            "run.googleapis.com",
            "appengine.googleapis.com",
            "storage.googleapis.com",
            "bigquery.googleapis.com",
            "sql.googleapis.com",
            "firestore.googleapis.com",
            "aiplatform.googleapis.com",
            "ml.googleapis.com",
            "dns.googleapis.com",
            "servicenetworking.googleapis.com",
            "cloudbuild.googleapis.com",
            "sourcerepo.googleapis.com",
            "artifactregistry.googleapis.com",
            "recommender.googleapis.com"
        ]
        
        # Required IAM roles for the service account
        self.required_roles = [
            "roles/viewer",
            "roles/resourcemanager.projectViewer",
            "roles/serviceusage.serviceUsageViewer",
            "roles/recommender.viewer",
            "roles/securitycenter.findingsViewer",
            "roles/compute.securityAdmin",
            "roles/cloudkms.viewer",
            "roles/iam.securityReviewer",
            "roles/resourcemanager.projectIamAdmin",
            "roles/securitycenter.complianceViewer",
            "roles/logging.viewer",
            "roles/monitoring.viewer",
            "roles/cloudtrace.user",
            "roles/aiplatform.user"
        ]
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated with GCP."""
        try:
            credentials, project = default()
            print(f"✅ Authenticated with GCP (default project: {project})")
            return True
        except DefaultCredentialsError:
            print("❌ Not authenticated with GCP")
            print("🔧 Run: gcloud auth application-default login")
            return False
    
    def enable_apis(self) -> bool:
        """Enable required GCP APIs."""
        print("📡 Enabling required APIs...")
        
        try:
            # Use gcloud command for API enablement (most reliable method)
            cmd = ["gcloud", "services", "enable"] + self.required_apis + [f"--project={self.project_id}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully enabled {len(self.required_apis)} APIs")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to enable APIs: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ gcloud CLI not found. Please install Google Cloud SDK")
            return False
    
    def create_service_account(self) -> bool:
        """Create the security agent service account."""
        print("👤 Creating service account...")
        
        try:
            cmd = [
                "gcloud", "iam", "service-accounts", "create", self.service_account_name,
                "--display-name=GCP Security Agent",
                "--description=Service account for GCP Security Agent",
                f"--project={self.project_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Created service account: {self.service_account_email}")
            return True
            
        except subprocess.CalledProcessError as e:
            if "already exists" in e.stderr:
                print(f"ℹ️  Service account already exists: {self.service_account_email}")
                return True
            else:
                print(f"❌ Failed to create service account: {e.stderr}")
                return False
    
    def assign_roles(self) -> bool:
        """Assign required IAM roles to the service account."""
        print("🔐 Assigning IAM roles...")
        
        success_count = 0
        for role in self.required_roles:
            try:
                cmd = [
                    "gcloud", "projects", "add-iam-policy-binding", self.project_id,
                    f"--member=serviceAccount:{self.service_account_email}",
                    f"--role={role}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"  ✅ Assigned role: {role}")
                success_count += 1
                
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to assign role {role}: {e.stderr}")
        
        print(f"🎯 Successfully assigned {success_count}/{len(self.required_roles)} roles")
        return success_count == len(self.required_roles)
    
    def create_service_account_key(self) -> bool:
        """Create and download service account key."""
        print("🔑 Creating service account key...")
        
        key_file = "service-account-key.json"
        
        try:
            cmd = [
                "gcloud", "iam", "service-accounts", "keys", "create", key_file,
                f"--iam-account={self.service_account_email}",
                f"--project={self.project_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if Path(key_file).exists():
                print(f"✅ Service account key saved as '{key_file}'")
                return True
            else:
                print("❌ Service account key file not found after creation")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create service account key: {e.stderr}")
            return False
    
    def generate_env_config(self) -> None:
        """Generate .env configuration instructions."""
        key_path = Path("service-account-key.json").absolute()
        
        print("\n📝 Add the following to your .env file:")
        print("=" * 50)
        print(f'GOOGLE_APPLICATION_CREDENTIALS="{key_path}"')
        print(f'GOOGLE_CLOUD_PROJECT="{self.project_id}"')
        print("ADK_EVALUATION_ENABLED=true")
        print(f'VERTEX_AI_PROJECT_ID="{self.project_id}"')
        print('VERTEX_AI_LOCATION="us-central1"')
        print("=" * 50)
        
        # Optionally create .env file
        create_env = input("\n❓ Create .env file automatically? (y/N): ").lower().strip()
        if create_env == 'y':
            try:
                with open('.env', 'w') as f:
                    f.write(f'GOOGLE_APPLICATION_CREDENTIALS="{key_path}"\n')
                    f.write(f'GOOGLE_CLOUD_PROJECT="{self.project_id}"\n')
                    f.write('ADK_EVALUATION_ENABLED=true\n')
                    f.write(f'VERTEX_AI_PROJECT_ID="{self.project_id}"\n')
                    f.write('VERTEX_AI_LOCATION="us-central1"\n')
                
                print("✅ .env file created successfully!")
            except Exception as e:
                print(f"❌ Failed to create .env file: {e}")
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        print(f"🔧 Setting up GCP Security Agent permissions for project: {self.project_id}\n")
        
        # Check authentication
        if not self.check_authentication():
            return False
        
        # Enable APIs
        if not self.enable_apis():
            return False
        
        # Create service account
        if not self.create_service_account():
            return False
        
        # Assign roles
        if not self.assign_roles():
            print("⚠️  Some roles failed to assign. The application may have limited functionality.")
        
        # Create service account key
        if not self.create_service_account_key():
            return False
        
        # Generate .env configuration
        self.generate_env_config()
        
        print("\n🎉 Setup completed successfully!")
        print("🚀 You can now run the GCP Security Agent with: ./run.py")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up GCP APIs and service account permissions for the Security Agent"
    )
    parser.add_argument(
        "--project-id", 
        required=True,
        help="GCP Project ID to set up permissions for"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        setup = GCPSecurityAgentSetup(args.project_id)
        print(f"📡 Would enable {len(setup.required_apis)} APIs")
        print(f"🔐 Would assign {len(setup.required_roles)} IAM roles")
        print(f"👤 Would create service account: {setup.service_account_email}")
        return
    
    setup = GCPSecurityAgentSetup(args.project_id)
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()