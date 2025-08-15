# 🔌 GCP Live Data Connection Guide

## Quick Diagnosis

Run this command to check your GCP connection:
```bash
python test_endpoints.py
```

If you see "Total assets: 0" or errors, follow this guide.

## ✅ Step-by-Step Setup for Live GCP Data

### 1. **Authenticate with GCP**

```bash
# Check if you're authenticated
gcloud auth list

# If not authenticated, login:
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 2. **Enable Required GCP APIs**

These are the MINIMUM required APIs for live data:

```bash
# Core APIs (REQUIRED)
gcloud services enable cloudasset.googleapis.com          # Asset Inventory
gcloud services enable cloudresourcemanager.googleapis.com # Resource Manager
gcloud services enable compute.googleapis.com              # Compute Engine
gcloud services enable storage-api.googleapis.com          # Storage

# Optional but recommended
gcloud services enable recommender.googleapis.com          # Recommendations
gcloud services enable iam.googleapis.com                  # IAM
```

### 3. **Check Your Permissions**

You need these IAM roles:

```bash
# Check your current roles
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:YOUR_EMAIL"

# Minimum required role (gives read access to everything)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/viewer"

# For asset inventory specifically
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/cloudasset.viewer"
```

### 4. **Test Direct API Access**

Test if you can access the Asset API directly:

```bash
# Test Asset Inventory API
curl -X GET \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://cloudasset.googleapis.com/v1/projects/YOUR_PROJECT_ID/assets"

# Test searching resources
curl -X GET \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://cloudasset.googleapis.com/v1/projects/YOUR_PROJECT_ID:searchAllResources"
```

### 5. **Configure the Application**

Create/update your `.env` file:

```env
# CRITICAL: Set your actual project ID
GOOGLE_CLOUD_PROJECT=your-actual-project-id

# This tells the app to use live data
ENABLE_MOCK_DATA=false
ENABLE_CACHE=true
```

### 6. **Run the Backend with Debug Logging**

```bash
# Start backend with debug logging to see what's happening
LOG_LEVEL=DEBUG python run_backend.py
```

## 🔍 Troubleshooting Connection Issues

### Issue: "403 Permission Denied"

**Error**: `403 Permission 'cloudasset.assets.searchAllResources' denied`

**Solution**:
```bash
# Grant the required permission
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:$(gcloud config get-value account)" \
  --role="roles/cloudasset.viewer"
```

### Issue: "API Not Enabled"

**Error**: `API cloudasset.googleapis.com is not enabled`

**Solution**:
```bash
# Enable the API
gcloud services enable cloudasset.googleapis.com

# Verify it's enabled
gcloud services list --enabled | grep cloudasset
```

### Issue: "No Assets Found"

**Error**: Getting 0 assets even though project has resources

**Check these**:
1. Are you in the right project?
   ```bash
   gcloud config get-value project
   ```

2. Do you actually have resources?
   ```bash
   gcloud compute instances list
   gcloud storage buckets list
   ```

3. Can you see assets via gcloud?
   ```bash
   gcloud asset search-all-resources --scope=projects/YOUR_PROJECT_ID
   ```

### Issue: "Authentication Failed"

**Error**: `Could not automatically determine credentials`

**Solution**:
```bash
# Re-authenticate
gcloud auth application-default login

# Verify credentials exist
ls ~/.config/gcloud/application_default_credentials.json
```

## 📊 Verify Live Data is Working

Once connected, you should see:

```
Testing Asset Snapshot Endpoint...
   Status: 200
   ✅ SUCCESS: Got snapshot data
   Total assets: 444  # <-- Real number of your assets
   Data source: live_api  # <-- Confirms using live data
```

In the dashboard, you'll see:
- Real asset counts (not 0 or mock numbers like 100/150)
- Actual resource types from your project
- Real storage buckets, compute instances, etc.

## 🚀 Quick Test Script

Create `test_gcp_connection.py`:

```python
#!/usr/bin/env python3
"""Test GCP connection and show what's working/not working"""

import subprocess
import json
import sys

def run_command(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔍 GCP Connection Diagnostic")
    print("=" * 50)
    
    # Check authentication
    print("\n1. Checking authentication...")
    success, stdout, stderr = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'")
    if success and stdout.strip():
        print(f"✅ Authenticated as: {stdout.strip()}")
    else:
        print("❌ Not authenticated. Run: gcloud auth application-default login")
        return
    
    # Check project
    print("\n2. Checking project...")
    success, stdout, stderr = run_command("gcloud config get-value project")
    if success and stdout.strip():
        project_id = stdout.strip()
        print(f"✅ Project: {project_id}")
    else:
        print("❌ No project set. Run: gcloud config set project YOUR_PROJECT_ID")
        return
    
    # Check APIs
    print("\n3. Checking required APIs...")
    apis = [
        "cloudasset.googleapis.com",
        "cloudresourcemanager.googleapis.com",
        "compute.googleapis.com",
        "storage-api.googleapis.com"
    ]
    
    for api in apis:
        success, stdout, stderr = run_command(f"gcloud services list --enabled --filter='name:{api}' --format='value(name)'")
        if success and api in stdout:
            print(f"✅ {api}")
        else:
            print(f"❌ {api} - Run: gcloud services enable {api}")
    
    # Test Asset API
    print("\n4. Testing Asset API access...")
    cmd = f'curl -s -X GET -H "Authorization: Bearer $(gcloud auth print-access-token)" "https://cloudasset.googleapis.com/v1/projects/{project_id}:searchAllResources?pageSize=1"'
    success, stdout, stderr = run_command(cmd)
    if success:
        try:
            data = json.loads(stdout)
            if "results" in data:
                print(f"✅ Asset API working! Found resources in project")
            elif "error" in data:
                print(f"❌ API Error: {data['error'].get('message', 'Unknown error')}")
            else:
                print("⚠️  No resources found (project might be empty)")
        except:
            print(f"❌ Could not parse API response")
    else:
        print(f"❌ Could not call Asset API")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("If all checks pass, the app should show live data.")
    print("If any fail, fix them and try again.")

if __name__ == "__main__":
    main()
```

## 💡 Tips for Live Data

1. **Always use a real project ID** - Not "test" or "demo"
2. **Start with read-only access** - Use "Viewer" role initially  
3. **Check logs** - Backend logs show exactly which API calls fail
4. **Test incrementally** - Get one API working before enabling others
5. **Use gcloud first** - If gcloud works, the app should work

## 🎯 Expected Results with Live Data

When properly connected, you'll see:
- **Asset counts**: Real numbers from your project (e.g., 444 assets)
- **Asset types**: Actual GCP resources (Storage Buckets, Compute Instances, etc.)
- **Locations**: Real regions where your resources are deployed
- **Security findings**: Actual issues if Security Command Center is enabled
- **Recommendations**: Real suggestions from Recommender API if enabled

## 🚨 Common Mistakes to Avoid

1. ❌ Using a fake project ID like "test-project"
2. ❌ Not running `gcloud auth application-default login`
3. ❌ Forgetting to enable APIs
4. ❌ Not having proper IAM permissions
5. ❌ Having ENABLE_MOCK_DATA=true in .env

## ✅ Success Indicators

You know it's working when:
1. `test_endpoints.py` shows non-zero asset counts
2. Dashboard displays "Data source: live_api"
3. Asset types match your actual GCP resources
4. No "mock" or "fallback" indicators in the UI
5. Backend logs show successful API calls without 403/404 errors

---

**Remember**: The goal is to connect to YOUR real GCP project and see YOUR actual resources. No mock data, just real insights into your cloud infrastructure!