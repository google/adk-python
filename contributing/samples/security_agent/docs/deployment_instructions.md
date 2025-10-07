# GCP Security Intelligence Platform - Deployment Instructions


## Overview

The GCP Security Intelligence Platform is an AI-powered security monitoring system with three components:

1. **BigQuery Datasets** - Central data storage (required)
2. **ADK Backend (Cloud Run)** - AI agent with 23 security analysis tools (required)
3. **Cloud Functions** - Automated data collectors that populate BigQuery on schedules (modular - deploy what you need)

**Architecture Flow:** Cloud Functions → BigQuery ← ADK Agent (Cloud Run)

---

## Prerequisites

### GCP Project Requirements
- Active GCP project with billing enabled
- Project Owner or Editor role for deployment

### Service Account Roles

The deployment script (`./deploy.sh` below in step 3) automatically creates service account `security-agent-api-sa` and grants all necessary roles:
- `roles/cloudfunctions.invoker` - Invoke Cloud Functions
- `roles/logging.logWriter` - Write logs
- `roles/bigquery.dataEditor` - Read/write BigQuery data
- `roles/bigquery.jobUser` - Run BigQuery queries
- `roles/resourcemanager.organizationViewer` - View organization resources
- `roles/iam.securityReviewer` - View IAM policies and roles

**Your deployment user** must have these roles to deploy Cloud Functions:
- `roles/cloudfunctions.developer`
- `roles/cloudscheduler.admin`
- `roles/run.admin` (or Project Owner/Editor)

### Required APIs
```bash
gcloud services enable \
  bigquery.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  run.googleapis.com
```

---

## Deployment Steps

### Step 1: Environment Setup

```bash
# Set your project configuration
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_REGION=us-central1

# Authenticate and set project
gcloud auth login
gcloud config set project ${GOOGLE_CLOUD_PROJECT}
```

---

### Step 2: Create BigQuery Datasets

```bash
# Create primary dataset
bq mk --dataset \
  --location=US \
  --description="Security insights and findings" \
  ${GOOGLE_CLOUD_PROJECT}:security_insights

# Create MSA results dataset
bq mk --dataset \
  --location=US \
  --description="Multi-Service Analyzer results" \
  ${GOOGLE_CLOUD_PROJECT}:security_data

# Verify datasets created
bq ls --project_id=${GOOGLE_CLOUD_PROJECT}
```

---

### Step 3: Deploy ADK Backend to Cloud Run

```bash
# Clone repository
git clone https://github.com/stuagano/adk-python.git
cd adk-python/contributing/samples/security_agent

# Run deployment script (interactive)
./deploy.sh
```

**When prompted, select option 1:** Deploy using Cloud Build (recommended)

**What the script does:**
- Creates service account `security-agent-api-sa`
- Builds Docker container via Cloud Build
- Deploys to Cloud Run with optimized settings (256Mi memory, auto-scaling)
- Configures environment variables

**Expected output:**
```
Deployment complete!
Service URL: https://security-agent-api-HASH-uc.a.run.app

Testing deployment...
{
  "status": "healthy",
  "agent": "Security Agent",
  "model": "gemini-2.5-flash"
}
```

**Save the Service URL** - you'll need it for Flask UI configuration (if deploying).

---

### Step 4: Deploy Cloud Functions (Modular)

Choose which data collectors to deploy based on your needs:

#### Option A: Deploy MSA Analyzer Only (Recommended Start)
```bash
cd cloud_functions/msa_analyzer
./deploy_complete.sh ${GOOGLE_CLOUD_PROJECT} ${GOOGLE_CLOUD_REGION}
```
**Cost:** ~$0.20/month
**Purpose:** Monitors GCP release notes for security/compliance impacts

#### Option B: Interactive Deployment Menu
```bash
cd cloud_functions
./deploy_selected.sh ${GOOGLE_CLOUD_PROJECT} ${GOOGLE_CLOUD_REGION}
```
**Menu options:**
- IAM & Security functions (7) - $1.60/month
- Infrastructure functions (2) - $0.60/month
- Feeds & Documentation (3) - $0.60/month
- MSA Analyzer (1) - $0.20/month
- Everything (13) - $2.80/month

#### Option C: Deploy Specific Function Manually
```bash
cd cloud_functions/FUNCTION_NAME

gcloud functions deploy FUNCTION_NAME \
  --runtime python311 \
  --trigger-http \
  --entry-point main \
  --region ${GOOGLE_CLOUD_REGION} \
  --memory 512MB \
  --timeout 540s \
  --set-env-vars GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},DATASET_ID=security_insights
```

---

### Step 5: Set Up Automated Schedules

Each Cloud Function needs a Cloud Scheduler job:

```bash
# Example: Schedule MSA Analyzer daily at 2 AM
gcloud scheduler jobs create http msa-analyzer-daily \
  --location ${GOOGLE_CLOUD_REGION} \
  --schedule="0 2 * * *" \
  --uri="https://${GOOGLE_CLOUD_REGION}-${GOOGLE_CLOUD_PROJECT}.cloudfunctions.net/msa-analyzer" \
  --http-method=POST \
  --project ${GOOGLE_CLOUD_PROJECT}
```

**Recommended schedules:**
- MSA Analyzer: `0 2 * * *` (daily 2 AM)
- IAM functions: `0 */6 * * *` (every 6 hours)
- Infrastructure: `0 */2 * * *` (every 2 hours)
- Security findings: `*/30 * * * *` (every 30 minutes)

---

### Step 6: Optional - Confluence Integration

If using Confluence documentation sync:

```bash
# Add environment variables
cd cloud_functions/confluence_sync

gcloud functions deploy sync-confluence-to-bigquery \
  --runtime python311 \
  --trigger-http \
  --entry-point sync_confluence_to_bigquery \
  --region ${GOOGLE_CLOUD_REGION} \
  --memory 512MB \
  --timeout 540s \
  --set-env-vars \
    CONFLUENCE_URL=https://your-domain.atlassian.net,\
    CONFLUENCE_USERNAME=your-email@example.com,\
    CONFLUENCE_API_TOKEN=your-api-token,\
    CONFLUENCE_SPACES=SEC,POLICY,GCP

# Schedule daily sync
gcloud scheduler jobs create http sync-confluence \
  --location ${GOOGLE_CLOUD_REGION} \
  --schedule="0 2 * * *" \
  --uri="https://${GOOGLE_CLOUD_REGION}-${GOOGLE_CLOUD_PROJECT}.cloudfunctions.net/sync-confluence-to-bigquery" \
  --http-method=POST
```

---

## Post-Deployment Validation

### 1. Verify Cloud Run Deployment
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe security-agent-api \
  --region ${GOOGLE_CLOUD_REGION} \
  --format 'value(status.url)')

# Test health endpoint
curl ${SERVICE_URL}/health

# Expected: {"status": "healthy", "agent": "Security Agent"}
```

### 2. Verify Cloud Functions
```bash
# List deployed functions
gcloud functions list --project ${GOOGLE_CLOUD_PROJECT}

# Check scheduler jobs
gcloud scheduler jobs list --location ${GOOGLE_CLOUD_REGION}

# Trigger test run
gcloud scheduler jobs run msa-analyzer-daily --location ${GOOGLE_CLOUD_REGION}

# View logs
gcloud functions logs read msa-analyzer --region ${GOOGLE_CLOUD_REGION} --limit 20
```

### 3. Verify BigQuery Data
```bash
# Check data freshness
bq query --use_legacy_sql=false \
  "SELECT table_name, last_refresh, record_count
   FROM \`${GOOGLE_CLOUD_PROJECT}.security_insights.refresh_metadata\`
   ORDER BY last_refresh DESC"
```

### 4. Test End-to-End Query
```bash
# Query via ADK agent
curl -X POST ${SERVICE_URL}/run \
  -H "Content-Type: application/json" \
  -d '{
    "newMessage": {
      "parts": [{"text": "What security data do I have?"}],
      "role": "user"
    }
  }'
```

---

## Operational Commands

### Monitor Data Freshness
```sql
-- Check which tables are stale
SELECT
  table_name,
  last_refresh,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), last_refresh, HOUR) as hours_stale,
  record_count
FROM `PROJECT.security_insights.refresh_metadata`
ORDER BY hours_stale DESC;
```

### View Cloud Run Logs
```bash
gcloud run services logs read security-agent-api \
  --region ${GOOGLE_CLOUD_REGION} \
  --limit 50
```

### View Cloud Function Logs
```bash
gcloud functions logs read FUNCTION_NAME \
  --region ${GOOGLE_CLOUD_REGION} \
  --limit 50
```

### Manually Trigger Cloud Function
```bash
gcloud scheduler jobs run JOB_NAME \
  --location ${GOOGLE_CLOUD_REGION}
```

### Update Cloud Run Environment Variables
```bash
gcloud run services update security-agent-api \
  --region ${GOOGLE_CLOUD_REGION} \
  --set-env-vars "KEY=VALUE,KEY2=VALUE2"
```

### Scale Cloud Run
```bash
# Set min/max instances
gcloud run services update security-agent-api \
  --region ${GOOGLE_CLOUD_REGION} \
  --min-instances 0 \
  --max-instances 10
```

---

## Troubleshooting

### Cloud Run Deployment Fails
**Check:**
- APIs enabled: `gcloud services list --enabled | grep run`
- IAM permissions: Verify you have `roles/run.admin`
- Service account exists: Check Cloud Build logs

**Fix:**
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

### Cloud Function Returns Errors
**Check logs:**
```bash
gcloud functions logs read FUNCTION_NAME --limit 50
```

**Common issues:**
- Permission denied → Add required IAM roles to service account
- Timeout → Increase timeout or optimize function
- BigQuery error → Verify dataset exists and schema matches

### No Data in BigQuery
**Steps:**
1. Verify scheduler job is enabled and running
2. Manually trigger the job
3. Check function execution logs for errors
4. Verify service account has BigQuery write permissions

```bash
# Check scheduler status
gcloud scheduler jobs describe JOB_NAME --location ${REGION}

# Trigger manually
gcloud scheduler jobs run JOB_NAME --location ${REGION}
```

### Agent Returns Empty Responses
**Steps:**
1. Check Cloud Run is healthy: `curl ${SERVICE_URL}/health`
2. Verify BigQuery has data: `bq ls PROJECT:security_insights`
3. Test direct query to BigQuery
4. Review Cloud Run logs for errors

---

## Security Best Practices

### Service Account Key Management
```bash
# Rotate keys every 90 days
gcloud iam service-accounts keys create new-key.json \
  --iam-account=SA@PROJECT.iam.gserviceaccount.com

# Delete old keys
gcloud iam service-accounts keys delete OLD_KEY_ID \
  --iam-account=SA@PROJECT.iam.gserviceaccount.com

# List keys and check age
gcloud iam service-accounts keys list \
  --iam-account=SA@PROJECT.iam.gserviceaccount.com
```

### Enable Audit Logging
```bash
# Enable BigQuery audit logs
gcloud logging sinks create bigquery-audit-sink \
  bigquery.googleapis.com/projects/${GOOGLE_CLOUD_PROJECT}/datasets/audit_logs \
  --log-filter='resource.type="bigquery_resource"'
```

### Configure VPC (High-Security Environments)
```bash
# Deploy Cloud Run with VPC connector (if required)
gcloud run services update security-agent-api \
  --vpc-connector=YOUR_CONNECTOR \
  --vpc-egress=private-ranges-only
```

---

## Backup & Recovery

### Backup BigQuery Data (Weekly)
```bash
# Export tables to Cloud Storage
bq extract --destination_format=AVRO \
  ${GOOGLE_CLOUD_PROJECT}:security_insights.TABLE_NAME \
  gs://YOUR-BACKUP-BUCKET/security_insights/$(date +%Y%m%d)/TABLE_NAME/*.avro
```

### Restore BigQuery Table
```bash
bq load --source_format=AVRO \
  ${GOOGLE_CLOUD_PROJECT}:security_insights.TABLE_NAME \
  gs://YOUR-BACKUP-BUCKET/security_insights/DATE/TABLE_NAME/*.avro
```

---

## Quick Reference

### Deployed Resources
| Resource | Type | Name | Purpose |
|----------|------|------|---------|
| BigQuery Dataset | Dataset | `security_insights` | Primary data storage |
| BigQuery Dataset | Dataset | `security_data` | MSA results |
| Cloud Run Service | Service | `security-agent-api` | ADK agent backend |
| Cloud Functions | Functions | 1-13 functions | Data collectors |
| Cloud Scheduler | Jobs | 1-13 jobs | Function triggers |

### Key URLs
- Cloud Run Service: `https://security-agent-api-HASH-uc.a.run.app`
- Health Check: `${SERVICE_URL}/health`
- BigQuery Console: `https://console.cloud.google.com/bigquery?project=${PROJECT}`

### Environment Variables (Cloud Run)
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
BQ_DEFAULT_DATASET=security_insights
ADK_AGENT_MODEL=gemini-2.5-flash
```
