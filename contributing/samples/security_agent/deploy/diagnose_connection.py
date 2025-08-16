#!/usr/bin/env python3
"""
GCP Connection Diagnostic Tool
Helps identify and fix issues preventing live data connection
"""

import subprocess
import json
import sys
import os
from typing import Tuple, Dict, List

class GCPConnectionDiagnostic:
    """Diagnose and fix GCP connection issues"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        self.project_id = None
        
    def run_command(self, cmd: str) -> Tuple[bool, str, str]:
        """Run command and return success, stdout, stderr"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated with GCP"""
        print("\n🔐 Checking GCP Authentication...")
        
        # Check active account
        success, stdout, stderr = self.run_command(
            "gcloud auth list --filter=status:ACTIVE --format='value(account)'"
        )
        
        if success and stdout.strip():
            print(f"  ✅ Authenticated as: {stdout.strip()}")
            
            # Check application default credentials
            adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            if os.path.exists(adc_path):
                print(f"  ✅ Application default credentials exist")
            else:
                print(f"  ⚠️  Application default credentials missing")
                self.issues_found.append({
                    "issue": "Missing application default credentials",
                    "fix": "gcloud auth application-default login"
                })
            return True
        else:
            print("  ❌ Not authenticated with GCP")
            self.issues_found.append({
                "issue": "Not authenticated",
                "fix": "gcloud auth application-default login"
            })
            return False
    
    def check_project(self) -> bool:
        """Check if a valid project is set"""
        print("\n📁 Checking Project Configuration...")
        
        success, stdout, stderr = self.run_command("gcloud config get-value project")
        
        if success and stdout.strip():
            self.project_id = stdout.strip()
            print(f"  ✅ Project ID: {self.project_id}")
            
            # Verify project exists and is accessible
            success, stdout, stderr = self.run_command(
                f"gcloud projects describe {self.project_id} --format='value(projectId)'"
            )
            
            if success:
                print(f"  ✅ Project is accessible")
                return True
            else:
                print(f"  ❌ Cannot access project {self.project_id}")
                self.issues_found.append({
                    "issue": f"Cannot access project {self.project_id}",
                    "fix": "Check project ID and permissions"
                })
                return False
        else:
            print("  ❌ No project configured")
            self.issues_found.append({
                "issue": "No project configured",
                "fix": "gcloud config set project YOUR_PROJECT_ID"
            })
            return False
    
    def check_apis(self) -> Dict[str, bool]:
        """Check which APIs are enabled"""
        print("\n🔌 Checking Required APIs...")
        
        required_apis = {
            "cloudasset.googleapis.com": "Asset Inventory (REQUIRED)",
            "cloudresourcemanager.googleapis.com": "Resource Manager (REQUIRED)",
            "compute.googleapis.com": "Compute Engine",
            "storage-api.googleapis.com": "Storage",
            "recommender.googleapis.com": "Recommender (Optional)",
            "iam.googleapis.com": "IAM (Optional)"
        }
        
        api_status = {}
        
        for api, description in required_apis.items():
            success, stdout, stderr = self.run_command(
                f"gcloud services list --enabled --filter='name:{api}' --format='value(name)'"
            )
            
            is_enabled = success and api in stdout
            api_status[api] = is_enabled
            
            if is_enabled:
                print(f"  ✅ {description}: Enabled")
            else:
                if "REQUIRED" in description:
                    print(f"  ❌ {description}: NOT ENABLED")
                    self.issues_found.append({
                        "issue": f"{description} API not enabled",
                        "fix": f"gcloud services enable {api}"
                    })
                else:
                    print(f"  ⚠️  {description}: Not enabled (optional)")
        
        return api_status
    
    def check_permissions(self) -> bool:
        """Check if user has required permissions"""
        print("\n🔑 Checking IAM Permissions...")
        
        if not self.project_id:
            print("  ⚠️  Skipping - no project configured")
            return False
        
        # Get current user
        success, stdout, stderr = self.run_command("gcloud config get-value account")
        if not success:
            return False
        
        user_email = stdout.strip()
        
        # Check IAM roles
        success, stdout, stderr = self.run_command(
            f"gcloud projects get-iam-policy {self.project_id} "
            f"--flatten='bindings[].members' "
            f"--filter='bindings.members:user:{user_email}' "
            f"--format='value(bindings.role)'"
        )
        
        if success and stdout:
            roles = stdout.strip().split('\n')
            print(f"  ℹ️  Your roles: {', '.join(roles)}")
            
            # Check for minimum required roles
            required_roles = [
                "roles/viewer",
                "roles/cloudasset.viewer",
                "roles/browser"
            ]
            
            has_required = any(
                any(req in role for req in required_roles)
                for role in roles
            )
            
            if has_required or "roles/owner" in roles or "roles/editor" in roles:
                print(f"  ✅ Sufficient permissions")
                return True
            else:
                print(f"  ⚠️  May need additional permissions")
                self.issues_found.append({
                    "issue": "Insufficient permissions",
                    "fix": f"gcloud projects add-iam-policy-binding {self.project_id} --member='user:{user_email}' --role='roles/cloudasset.viewer'"
                })
                return False
        else:
            print(f"  ❌ No IAM roles found")
            self.issues_found.append({
                "issue": "No IAM roles",
                "fix": f"gcloud projects add-iam-policy-binding {self.project_id} --member='user:{user_email}' --role='roles/viewer'"
            })
            return False
    
    def test_asset_api(self) -> bool:
        """Test direct access to Asset Inventory API"""
        print("\n🧪 Testing Asset Inventory API...")
        
        if not self.project_id:
            print("  ⚠️  Skipping - no project configured")
            return False
        
        # Test searchAllResources endpoint
        cmd = (
            f'curl -s -X GET '
            f'-H "Authorization: Bearer $(gcloud auth print-access-token)" '
            f'"https://cloudasset.googleapis.com/v1/projects/{self.project_id}:searchAllResources?pageSize=1"'
        )
        
        success, stdout, stderr = self.run_command(cmd)
        
        if success:
            try:
                data = json.loads(stdout)
                
                if "results" in data:
                    print(f"  ✅ API is working! Found {len(data['results'])} resources")
                    if data['results']:
                        first_asset = data['results'][0]
                        print(f"  ℹ️  Example asset: {first_asset.get('assetType', 'Unknown')}")
                    return True
                    
                elif "error" in data:
                    error = data['error']
                    status = error.get('status', 'UNKNOWN')
                    message = error.get('message', 'Unknown error')
                    
                    print(f"  ❌ API Error ({status}): {message}")
                    
                    if status == "PERMISSION_DENIED":
                        self.issues_found.append({
                            "issue": "Permission denied for Asset API",
                            "fix": f"gcloud projects add-iam-policy-binding {self.project_id} --member='user:$(gcloud config get-value account)' --role='roles/cloudasset.viewer'"
                        })
                    elif "API has not been used" in message:
                        self.issues_found.append({
                            "issue": "Asset API not enabled",
                            "fix": "gcloud services enable cloudasset.googleapis.com"
                        })
                    return False
                    
                else:
                    print("  ⚠️  No resources found (project might be empty)")
                    return True
                    
            except json.JSONDecodeError:
                print(f"  ❌ Invalid API response")
                return False
        else:
            print(f"  ❌ Could not call API")
            return False
    
    def check_backend_connection(self) -> bool:
        """Check if backend is running and can connect to GCP"""
        print("\n🖥️  Testing Backend Connection...")
        
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("  ✅ Backend is running")
                
                # Test asset endpoint
                response = requests.get(
                    f"http://localhost:8000/api/v1/assets/snapshot/{self.project_id or 'test'}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        total_assets = data.get("data", {}).get("summary", {}).get("total_assets", 0)
                        if total_assets > 0:
                            print(f"  ✅ Backend connected to GCP! Found {total_assets} assets")
                            return True
                        else:
                            print(f"  ⚠️  Backend connected but found 0 assets")
                            return False
                    else:
                        print(f"  ❌ Backend cannot fetch assets")
                        return False
                else:
                    print(f"  ❌ Asset endpoint returned {response.status_code}")
                    return False
            else:
                print("  ❌ Backend health check failed")
                return False
                
        except requests.exceptions.ConnectionError:
            print("  ❌ Backend is not running")
            self.issues_found.append({
                "issue": "Backend not running",
                "fix": "python run_backend.py"
            })
            return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def suggest_fixes(self):
        """Suggest fixes for found issues"""
        if not self.issues_found:
            print("\n✅ No issues found! Your GCP connection should be working.")
            print("\nTry running: python test_endpoints.py")
        else:
            print("\n🔧 Suggested Fixes:")
            print("=" * 50)
            
            for i, issue in enumerate(self.issues_found, 1):
                print(f"\n{i}. Issue: {issue['issue']}")
                print(f"   Fix: {issue['fix']}")
            
            print("\n" + "=" * 50)
            print("\n📋 Run these commands to fix issues:")
            for issue in self.issues_found:
                print(f"   {issue['fix']}")
    
    def auto_fix(self) -> bool:
        """Attempt to automatically fix some issues"""
        print("\n🔧 Attempting Auto-Fix...")
        
        if not self.issues_found:
            print("  ℹ️  No issues to fix")
            return True
        
        print(f"  Found {len(self.issues_found)} issues")
        response = input("  Attempt to fix automatically? (y/n): ").lower()
        
        if response != 'y':
            return False
        
        for issue in self.issues_found:
            fix_cmd = issue['fix']
            
            # Only auto-fix safe commands
            safe_commands = [
                "gcloud auth application-default login",
                "gcloud services enable"
            ]
            
            is_safe = any(safe in fix_cmd for safe in safe_commands)
            
            if is_safe:
                print(f"\n  Running: {fix_cmd}")
                success, stdout, stderr = self.run_command(fix_cmd)
                if success:
                    print(f"  ✅ Fixed: {issue['issue']}")
                    self.fixes_applied.append(issue['issue'])
                else:
                    print(f"  ❌ Could not fix: {issue['issue']}")
            else:
                print(f"  ⚠️  Skipping (requires manual action): {fix_cmd}")
        
        return len(self.fixes_applied) > 0
    
    def run_diagnostic(self):
        """Run complete diagnostic"""
        print("=" * 60)
        print("🔍 GCP Connection Diagnostic Tool")
        print("=" * 60)
        
        # Run all checks
        auth_ok = self.check_authentication()
        project_ok = self.check_project()
        api_status = self.check_apis()
        perms_ok = self.check_permissions()
        api_test_ok = self.test_asset_api()
        backend_ok = self.check_backend_connection()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Diagnostic Summary")
        print("=" * 60)
        
        all_ok = (
            auth_ok and project_ok and 
            api_status.get("cloudasset.googleapis.com", False) and
            api_test_ok
        )
        
        if all_ok:
            print("\n✅ GCP connection is properly configured!")
            if backend_ok:
                print("✅ Backend is successfully fetching live data!")
            else:
                print("⚠️  Backend needs to be started or restarted")
        else:
            print(f"\n❌ Found {len(self.issues_found)} issues preventing live data connection")
        
        # Suggest fixes
        self.suggest_fixes()
        
        # Offer auto-fix
        if self.issues_found:
            self.auto_fix()
        
        print("\n" + "=" * 60)
        print("Done! Re-run this tool after fixing issues.")
        print("=" * 60)

def main():
    diagnostic = GCPConnectionDiagnostic()
    diagnostic.run_diagnostic()

if __name__ == "__main__":
    main()