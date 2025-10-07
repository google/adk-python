# Setup and Troubleshooting Guide

Complete guide for setting up and troubleshooting the GCP Security Intelligence Platform.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Dependency Validation](#dependency-validation)
- [Configuration](#configuration)
- [Starting Services](#starting-services)
- [Common Issues](#common-issues)
- [Verification](#verification)

## ✅ Prerequisites

### System Requirements

- **Python**: 3.11+ (tested on 3.12.7)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: 4GB+ RAM recommended
- **Disk Space**: 2GB+ free space

### Required Accounts

- **GCP Project**: Active Google Cloud Platform project
- **Service Account**: JSON key file with required permissions
- **Confluence** (optional): API access token for documentation sync

### Required Permissions

Your service account needs these IAM roles:

**Required (for BigQuery access):**
- `roles/bigquery.dataViewer` - Read BigQuery data
- `roles/bigquery.jobUser` - Execute BigQuery queries

**Optional (for live data collection):**
- `roles/compute.viewer` - Read compute resources
- `roles/iam.securityReviewer` - Read IAM data
- `roles/storage.objectViewer` - Read GCS data
- `roles/cloudasset.viewer` - Read Cloud Asset inventory

## 🚀 Installation Steps

### Step 1: Install ADK via pipx

```bash
# Install pipx (if not already installed)
pip install --user pipx
pipx ensurepath

# Install ADK
pipx install google-adk

# Verify installation
adk --version
# Expected: adk, version 1.14.1 (or higher)
```

### Step 2: Clone Repository

```bash
# Clone the repository
git clone https://github.com/stuagano/adk-python.git
cd adk-python/contributing/samples/security_agent
```

### Step 3: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# This installs:
# - google-cloud-bigquery, google-cloud-compute, google-cloud-iam
# - google-cloud-storage, google-cloud-resource-manager
# - flask, flask-cors, gunicorn
# - chainlit (for modern UI)
# - mcp (for Model Context Protocol)
# - beautifulsoup4, lxml, feedparser
# - pandas, tabulate, python-dotenv
```

### Step 4: Install Tool Dependencies in ADK Environment

```bash
# Install required packages in ADK's pipx environment
~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# Or use the path for your system:
# macOS/Linux: ~/.local/pipx/venvs/google-adk/bin/python3.13
# Windows: %USERPROFILE%\.local\pipx\venvs\google-adk\Scripts\python.exe
```

### Step 5: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env file with your details
nano .env  # or vim, code, etc.
```

**Required configuration in `.env`:**

```bash
# GCP Project
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account-key.json
GOOGLE_CLOUD_LOCATION=us-central1

# BigQuery
BQ_DEFAULT_DATASET=security_insights
BQ_DEFAULT_TABLE=security_insights

# ADK
ADK_AGENT_MODEL=gemini-2.5-flash
GOOGLE_GENAI_USE_VERTEXAI=1
ADK_BASE_URL=http://localhost:8000
```

### Step 6: Add Service Account Key

```bash
# Create config directory if it doesn't exist
mkdir -p config

# Copy your service account JSON key
cp /path/to/your-service-account-key.json config/

# Secure the file
chmod 600 config/service-account-key.json

# Update .env to point to this file
# GOOGLE_APPLICATION_CREDENTIALS=config/your-service-account-key.json
```

## 🧪 Dependency Validation

Run the comprehensive dependency validation test:

```bash
python3 test_dependencies.py
```

**Expected output:**
```
✓ Python version: 3.12.7
✓ google-cloud-bigquery: 3.36.0
✓ flask: 3.0.3
✓ chainlit: 2.8.3
✓ mcp: 1.16.0
...
Total: 41 passed, 2 failed
```

**Note:** 2 failures are acceptable:
- `google-adk` not in main environment (it's installed via pipx - this is correct!)
- Minor Chainlit import warning (doesn't affect functionality)

## ⚙️ Configuration

### Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | ✅ | - | Your GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | ✅ | - | Path to service account JSON |
| `GOOGLE_CLOUD_LOCATION` | ✅ | us-central1 | GCP region |
| `BQ_DEFAULT_DATASET` | ✅ | security_insights | BigQuery dataset name |
| `BQ_DEFAULT_TABLE` | ✅ | security_insights | BigQuery table name |
| `ADK_AGENT_MODEL` | ✅ | gemini-2.5-flash | Gemini model to use |
| `GOOGLE_GENAI_USE_VERTEXAI` | ✅ | 1 | Use Vertex AI (required) |
| `ADK_BASE_URL` | ✅ | http://localhost:8000 | ADK backend URL |
| `CONFLUENCE_URL` | ⚠️ | - | Confluence instance URL (optional) |
| `CONFLUENCE_USERNAME` | ⚠️ | - | Confluence username (optional) |
| `CONFLUENCE_API_TOKEN` | ⚠️ | - | Confluence API token (optional) |
| `CONFLUENCE_SPACES` | ⚠️ | SEC,POLICY,GCP | Spaces to sync (optional) |

### BigQuery Setup

```bash
# Create the BigQuery dataset
bq mk --dataset \
  --location=US \
  your-project-id:security_insights

# Verify dataset exists
bq ls

# Grant service account access
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:your-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:your-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"
```

## 🎯 Starting Services

### Option 1: ADK Backend Only

```bash
# Terminal 1: Start ADK backend
adk web

# Access at: http://localhost:8000
```

### Option 2: Flask Web UI

```bash
# Terminal 1: Start ADK backend
adk web

# Terminal 2: Start Flask UI
python3 app.py --port=5001

# Access at: http://localhost:5001
```

### Option 3: Chainlit UI (Recommended)

```bash
# Terminal 1: Start ADK backend
adk web

# Terminal 2: Start Chainlit
chainlit run chainlit_app.py

# Access at: http://localhost:8001
```

### Option 4: MCP Server (for Claude Desktop, Continue, Cursor)

```bash
# Run standalone (testing)
python3 mcp_server.py

# Or configure in Claude Desktop (see docs/MCP_SERVER_INTEGRATION.md)
```

## 🔧 Common Issues

### Issue 1: Port 5000 Already in Use

**Symptoms:**
```
Error: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use port 5001 instead
python3 app.py --port=5001

# Or kill the process using port 5000
lsof -ti:5000 | xargs kill -9
```

**Cause:** macOS AirPlay Receiver uses port 5000 by default.

---

### Issue 2: No module named 'bs4'

**Symptoms:**
```
ModuleNotFoundError: No module named 'bs4'
```

**Solution:**
```bash
# Install in ADK environment
~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# Verify installation
python3 test_dependencies.py
```

**Cause:** Tool dependencies not installed in ADK's pipx environment.

---

### Issue 3: GOOGLE_CLOUD_PROJECT not set

**Symptoms:**
```
✗ GOOGLE_CLOUD_PROJECT: NOT SET
```

**Solution:**
```bash
# Edit .env file
nano .env

# Add this line:
GOOGLE_CLOUD_PROJECT=your-project-id

# Verify
grep GOOGLE_CLOUD_PROJECT .env
```

**Cause:** Environment variable not configured in `.env` file.

---

### Issue 4: Service Account Authentication Failed

**Symptoms:**
```
google.auth.exceptions.DefaultCredentialsError
```

**Solution:**
```bash
# Verify file exists
ls -l config/service-account-key.json

# Check permissions
chmod 600 config/service-account-key.json

# Verify .env points to correct file
grep GOOGLE_APPLICATION_CREDENTIALS .env

# Test authentication
gcloud auth activate-service-account --key-file=config/service-account-key.json
gcloud auth list
```

**Cause:** Service account key file not found or incorrect path in `.env`.

---

### Issue 5: BigQuery Dataset Not Found

**Symptoms:**
```
404 Not found: Dataset your-project-id:security_insights
```

**Solution:**
```bash
# Create the dataset
bq mk --dataset \
  --location=US \
  your-project-id:security_insights

# Verify
bq ls

# Deploy Cloud Functions to populate data (optional)
cd cloud_functions/fetch_iam_accounts
./deploy.sh your-project-id us-central1
```

**Cause:** BigQuery dataset doesn't exist yet.

---

### Issue 6: Chainlit Import Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'chainlit'
```

**Solution:**
```bash
# Install chainlit
pip install chainlit

# Or reinstall all dependencies
pip install -r requirements.txt

# Verify
pip show chainlit
```

**Cause:** Chainlit not installed.

---

### Issue 7: ADK CLI Not Found

**Symptoms:**
```
command not found: adk
```

**Solution:**
```bash
# Install ADK via pipx
pipx install google-adk

# Ensure pipx is in PATH
pipx ensurepath

# Restart terminal or reload PATH
source ~/.bashrc  # or ~/.zshrc

# Verify
adk --version
```

**Cause:** ADK not installed or not in PATH.

---

### Issue 8: Confluence Tools Not Working

**Symptoms:**
```
WARNING: Confluence service not available
```

**Solution:**

This is **expected** if you haven't configured Confluence credentials. The platform works in **cache-only mode** without Confluence API access.

To enable Confluence integration:
```bash
# Edit .env
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACES=SEC,POLICY,GCP

# Deploy Confluence sync Cloud Function (optional)
cd cloud_functions/confluence_sync
./deploy.sh your-project-id us-central1
```

**Cause:** Confluence credentials not configured (optional feature).

---

### Issue 9: Type Annotation Errors

**Symptoms:**
```
TypeError: Default value None of parameter release_notes_url: str = None
```

**Solution:**

These errors have been fixed in the current version. Update to latest:
```bash
git pull origin main
```

**Cause:** Old code version with incompatible type annotations.

---

### Issue 10: Permission Denied on BigQuery

**Symptoms:**
```
403 Access Denied: Project your-project-id
```

**Solution:**
```bash
# Grant required roles
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:your-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:your-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"

# Verify
gcloud projects get-iam-policy your-project-id \
  --flatten="bindings[].members" \
  --filter="bindings.members:your-sa@your-project-id.iam.gserviceaccount.com"
```

**Cause:** Service account lacks required IAM permissions.

## ✅ Verification

### Test ADK Backend

```bash
# Start ADK backend
adk web

# In another terminal, test the API
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### Test Flask UI

```bash
# Start Flask
python3 app.py --port=5001

# Open browser
open http://localhost:5001
```

### Test Chainlit UI

```bash
# Start Chainlit
chainlit run chainlit_app.py

# Open browser
open http://localhost:8001
```

### Test BigQuery Connection

```bash
# Test query
python3 -c "
from agents._tools.bigquery_tools import hello_world
result = hello_world()
print(result)
"
# Expected: Success message with query results
```

### Test All Tools

```bash
# Run comprehensive test
python3 test_dependencies.py

# Expected: 41+ passed, 2 or fewer failed
```

## 📚 Additional Resources

- [Complete Tool Reference](TOOLS.md) - All 32 tools documented
- [Chainlit UI Integration](CHAINLIT_INTEGRATION.md) - Modern chat interface
- [MCP Server Integration](MCP_SERVER_INTEGRATION.md) - Model Context Protocol
- [Confluence Integration](CONFLUENCE_BIGQUERY_INTEGRATION.md) - Documentation sync
- [Cloud Functions Guide](../cloud_functions/README.md) - Data collection setup

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Run dependency validation**: `python3 test_dependencies.py`
2. **Check logs**: Look for error messages in terminal output
3. **Verify configuration**: Ensure all required variables in `.env`
4. **Test authentication**: `gcloud auth list`
5. **Check permissions**: Verify service account IAM roles
6. **Update code**: `git pull origin main`
7. **File an issue**: https://github.com/stuagano/adk-python/issues

## 🎉 Success Checklist

- [ ] Python 3.11+ installed
- [ ] ADK CLI working (`adk --version`)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Tool dependencies in ADK environment
- [ ] `.env` file configured
- [ ] Service account key in place
- [ ] BigQuery dataset created
- [ ] Dependency test passes (`python3 test_dependencies.py`)
- [ ] ADK backend starts (`adk web`)
- [ ] At least one UI option works (Flask/Chainlit)
- [ ] Can query BigQuery successfully

Once all checks pass, you're ready to use the platform! 🚀
