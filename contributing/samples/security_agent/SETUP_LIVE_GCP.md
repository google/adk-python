# Setup Guide: Enable Live GCP Data

This guide shows you how to configure the Security Agent to use **real GCP data** instead of mock/test data.

## Prerequisites

1. **GCP Project** with resources (Storage buckets, IAM policies, etc.)
2. **Service Account** with appropriate permissions
3. **Service Account Key** downloaded as JSON file

## Step 1: Create GCP Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** > **Service Accounts**
3. Click **Create Service Account**
4. Name it `security-agent-service-account`
5. Grant these roles:
   - **Viewer** (read access to all resources)
   - **Storage Object Viewer** (for bucket analysis)
   - **Security Reviewer** (for security findings)
   - **Cloud Asset Viewer** (for asset inventory)

## Step 2: Download Service Account Key

1. Click on your service account
2. Go to **Keys** tab
3. Click **Add Key** > **Create new key**
4. Choose **JSON** format
5. Download and save as `service-account-key.json`

## Step 3: Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and update these values:
   ```bash
   # Replace with your actual GCP project ID
   GOOGLE_CLOUD_PROJECT=your-real-project-id

   # Replace with absolute path to your service account JSON file
   GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account-key.json
   ```

## Step 4: Install Required Dependencies

The GCP libraries should already be installed, but verify:

```bash
./venv/bin/pip install google-cloud-storage google-cloud-asset google-cloud-resource-manager
```

## Step 5: Test the Configuration

1. **Start the backend:**
   ```bash
   python run_backend.py
   ```

2. **Check the logs** for these indicators:
   - ✅ `GCP Live Data Tool initialized successfully`
   - 🔴 `Using LIVE GCP data for storage buckets` (when querying storage)

3. **Test through frontend:**
   ```bash
   python run_frontend.py
   ```

4. **Ask about storage buckets** in the chat:
   - "Tell me about storage buckets"
   - "What storage buckets do I have?"

## How It Works

### Data Source Detection

The system automatically detects whether to use live or cached data:

- **🔴 Live GCP Data**: Used when credentials are properly configured
- **📁 Cached SQLite Data**: Used as fallback when GCP is unavailable

### Live Data Features

When using live GCP data, you get:

- **Real-time bucket information** from your actual GCP project
- **Security analysis** with live IAM policy checks
- **Public access detection** based on actual bucket configurations
- **Current bucket properties** (location, storage class, versioning, etc.)

### Fallback Behavior

If GCP credentials are missing or invalid:
- System automatically falls back to SQLite cached data
- No errors or interruptions to user experience
- Clear logging indicates which data source is being used

## Troubleshooting

### Common Issues

1. **"GCP client initialization failed"**
   - Check that `GOOGLE_APPLICATION_CREDENTIALS` points to valid JSON file
   - Verify the service account has required permissions
   - Ensure the project ID is correct

2. **"Using SQLite cached data"**
   - This means GCP credentials aren't configured - normal for testing
   - Follow steps 1-3 above to enable live data

3. **"Live GCP query failed, falling back to SQLite"**
   - Temporary GCP API issue
   - Check internet connectivity
   - Verify service account permissions

### Verification Commands

Test your configuration:

```bash
# Test service account access
gcloud auth activate-service-account --key-file=/path/to/service-account-key.json

# List storage buckets (should work if permissions are correct)
gsutil ls

# Check current project
gcloud config get-value project
```

## Security Notes

- **Never commit** service account keys to version control
- **Rotate keys** regularly for security
- **Use least privilege** - only grant necessary permissions
- **Store keys securely** outside the project directory

## What You'll See

### With Live Data:
```
🔴 Using LIVE GCP data for storage buckets
Storage Security Analysis:
- Total Buckets: 5
- Public Buckets: 1 ⚠️ HIGH RISK

Storage Buckets:
• **my-production-bucket**
  - Location: US
  - Storage Class: STANDARD
  - Access: 🔒 Private
```

### With Cached Data:
```
📁 Using SQLite cached data for storage buckets
Storage Security Analysis:
- Total Buckets: 3
- Public Buckets: 1 ⚠️ HIGH RISK

Storage Buckets:
• **mgm-digitalconcierge-logs** (test data)
```

Now your Security Agent will analyze your real GCP environment!