# Deployment Guide - GCP Security Intelligence Platform

## Overview

This platform uses a modular architecture with three main components:
1. **ADK Backend** - AI agent with 23 tools (required)
2. **Flask Wrapper** - Optional web UI (optional)
3. **Cloud Functions** - Data fetchers (deploy what you need)

---

## Prerequisites

### Required
- Google Cloud Project with billing enabled
- BigQuery API enabled
- Service account with permissions:
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
  - `roles/resourcemanager.organizationViewer`
  - `roles/iam.securityReviewer`

### Optional (for Cloud Functions)
- Cloud Functions API enabled
- Cloud Scheduler API enabled
- Pub/Sub API enabled

---

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/stuagano/adk-python.git
cd contributing/samples/security_agent
```

### 2. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export DEFAULT_DATASET=security_insights
```

### 4. Run ADK Backend
```bash
# Terminal 1: Start ADK backend (port 8000)
adk web
```

The agent is now running at http://localhost:8000

### 5. Run Flask Wrapper (Optional)
```bash
# Terminal 2: Start Flask wrapper (port 5000)
python3 app.py
```

Web UI available at http://localhost:5000

---

## Cloud Functions Deployment

### Overview
Deploy only the Cloud Functions you need. Each function is independent.

**Categories:**
- 🔒 **IAM & Security** (7 functions) - User/service account roles, custom roles
- ☁️ **Infrastructure** (2 functions) - Compute instances, firewall rules
- 📰 **Feeds & Documentation** (3 functions) - RSS feeds, release notes, Confluence
- 🎯 **Analysis** (1 function) - MSA release notes analyzer

### Interactive Deployment
```bash
cd cloud_functions
./deploy_selected.sh
```

This launches an interactive menu to choose which functions to deploy.

### Deploy Specific Function
```bash
cd cloud_functions/msa_analyzer
./deploy_complete.sh your-project-id us-central1
```

### Deploy All Functions
```bash
cd cloud_functions
./deploy_selected.sh
# Select option 5: "🎁 Everything"
```

---

## Cloud Run Deployment (Flask Wrapper)

### 1. Build Container
```bash
# Build Docker image
docker build -t gcr.io/your-project-id/security-agent:latest .

# Push to Container Registry
docker push gcr.io/your-project-id/security-agent:latest
```

### 2. Deploy to Cloud Run
```bash
gcloud run deploy security-agent \
  --image gcr.io/your-project-id/security-agent:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300
```

### 3. Configure ADK Backend
The Flask wrapper needs to connect to your ADK backend. Update `app.py`:

```python
# Change this line:
session_url = "http://localhost:8000/apps/agents/users/web-user/sessions"

# To your deployed ADK endpoint:
session_url = "https://your-adk-backend.run.app/apps/agents/users/web-user/sessions"
```

---

## BigQuery Setup

### 1. Create Dataset
```bash
bq mk --dataset \
  --location=US \
  --description="Security insights and findings" \
  your-project-id:security_insights
```

### 2. Create Tables (for MSA)
```bash
cd cloud_functions/msa_analyzer
bq query --use_legacy_sql=false < bigquery_setup.sql
```

### 3. Verify Setup
```bash
# List tables
bq ls your-project-id:security_insights

# Query sample data
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) as total FROM \`your-project-id.security_insights.assets\`"
```

---

## Architecture Validation

### Test ADK Backend
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "agent": "Security Agent",
  "model": "gemini-2.5-flash"
}
```

### Test Flask Wrapper
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "agent": "Security Agent",
  "model": "gemini-2.5-flash"
}
```

### Test BigQuery Connection
```bash
# From agent
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "agents",
    "userId": "test-user",
    "sessionId": "test-session",
    "newMessage": {
      "parts": [{"text": "List all BigQuery datasets"}],
      "role": "user"
    }
  }'
```

---

## Monitoring & Maintenance

### Check Cloud Function Logs
```bash
gcloud functions logs read msa-analyzer --limit 50
```

### Monitor Cloud Scheduler Jobs
```bash
gcloud scheduler jobs list
gcloud scheduler jobs describe msa-analyzer-daily
```

### Query BigQuery Metadata
```sql
-- Check data freshness
SELECT
  table_name,
  MAX(refresh_timestamp) as last_updated,
  COUNT(*) as total_records
FROM `your-project-id.security_insights.refresh_metadata`
GROUP BY table_name
ORDER BY last_updated DESC;
```

### View Agent Logs
```bash
# ADK backend logs
tail -f logs/adk.log

# Flask wrapper logs
tail -f logs/app.log
```

---

## Cost Estimation

### Cloud Functions
- **Per function**: ~$0.20/month (with scheduled runs)
- **All 13 functions**: ~$2.60/month

### BigQuery
- **Storage**: $0.02/GB/month
- **Queries**: $5/TB processed
- **Estimated for this platform**: $5-10/month

### Cloud Run (Flask Wrapper - Optional)
- **Always-on**: ~$20/month
- **On-demand**: ~$2-5/month

### Total Estimated Cost
- **Minimal setup** (ADK only): $0/month (local)
- **With Cloud Functions**: $2.60-10/month
- **Full deployment**: $10-30/month

---

## Troubleshooting

### ADK Backend Won't Start
```bash
# Check Python version (requires 3.11+)
python3 --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check environment variables
env | grep GOOGLE
```

### Flask Can't Connect to ADK
```bash
# Verify ADK is running
curl http://localhost:8000/health

# Check app.py session_url setting
grep "session_url" app.py
```

### BigQuery Permission Errors
```bash
# Test service account permissions
gcloud projects get-iam-policy your-project-id \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:*" \
  --format="table(bindings.role)"
```

### Cloud Function Deployment Fails
```bash
# Check enabled APIs
gcloud services list --enabled

# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
```

---

## Security Best Practices

1. **Service Account Keys**
   - Never commit `.json` key files to git
   - Use Workload Identity when possible
   - Rotate keys every 90 days

2. **API Access**
   - Limit service account permissions (principle of least privilege)
   - Use separate service accounts for dev/prod

3. **Data Access**
   - Enable BigQuery audit logging
   - Use VPC Service Controls for sensitive data
   - Set up data retention policies

4. **Network Security**
   - Use Cloud Run ingress controls
   - Enable VPC connector for private access
   - Configure Cloud Armor for DDoS protection

---

## Next Steps

1. **Deploy Cloud Functions** - Start with MSA analyzer
2. **Configure Confluence** - If you use Confluence documentation
3. **Set Up Monitoring** - Cloud Monitoring alerts
4. **Train Users** - Share agent capabilities with team
5. **Iterate** - Add custom tools based on your needs

---

## Support & Resources

- **Main README**: [README.md](README.md)
- **Cloud Functions Guide**: [cloud_functions/README.md](cloud_functions/README.md)
- **MSA Quick Start**: [cloud_functions/msa_analyzer/QUICKSTART.md](cloud_functions/msa_analyzer/QUICKSTART.md)
- **Todo List**: [docs/todo.md](docs/todo.md)

---

**Last Updated**: October 2, 2025
