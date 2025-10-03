# MSA Analyzer - 5-Minute Quick Start

## What You'll Get

A fully automated system that:
- 📰 Monitors GCP release notes daily
- 🔍 Analyzes security, billing, and compliance impacts
- 📊 Stores results in BigQuery
- 🚨 Alerts on critical issues via Pub/Sub
- ⏰ Runs automatically every morning at 9 AM

## Prerequisites

✅ GCP Project with billing enabled
✅ `gcloud` CLI installed
✅ Owner permissions on the project

## Deploy Everything (One Command)

```bash
cd cloud_functions/msa_analyzer
./deploy_complete.sh mgm-digitalconcierge us-central1
```

**That's it!** The script handles everything automatically:
- Enables all APIs
- Creates all resources
- Configures permissions
- Deploys the function
- Sets up daily scheduler

**Time:** ~5 minutes | **Interaction:** None required

## What Gets Created

```
Project: mgm-digitalconcierge
│
├── BigQuery
│   ├── Dataset: security_data
│   ├── Table: msa_analysis_history (results storage)
│   ├── Table: active_services (monitored services)
│   └── Views: latest_summary, critical_issues, billing_trends
│
├── Cloud Storage
│   └── Bucket: mgm-digitalconcierge-msa-cache (processed notes cache)
│
├── Pub/Sub
│   ├── Topic: msa-critical-alerts
│   └── Subscription: msa-critical-alerts-subscription
│
├── Cloud Functions
│   └── msa-analyzer (us-central1, Python 3.11, 512MB)
│
└── Cloud Scheduler
    └── msa-analyzer-daily (runs at 9 AM ET)
```

## Test It

### 1. Manual Test

```bash
# Get function URL
FUNCTION_URL=$(gcloud functions describe msa-analyzer \
  --region=us-central1 \
  --format="value(serviceConfig.uri)")

# Run analysis
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}'
```

**Expected:** JSON response with analysis summary

### 2. Check BigQuery Results

```bash
bq query --use_legacy_sql=false \
  'SELECT * FROM `security_data.msa_latest_summary` LIMIT 5'
```

**Expected:** Recent analysis results

### 3. View Logs

```bash
gcloud functions logs read msa-analyzer --region=us-central1 --limit=50
```

**Expected:** Function execution logs

### 4. Test Scheduler

```bash
gcloud scheduler jobs run msa-analyzer-daily --location=us-central1
```

**Expected:** Scheduler triggers function

## Customize Active Services

Update which GCP services to monitor:

```bash
bq query --use_legacy_sql=false "
INSERT INTO \`security_data.active_services\`
(service_name, service_type, status, enabled_date, created_at, updated_at)
VALUES
  ('Cloud Logging', 'operations', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
  ('Cloud Trace', 'operations', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
"
```

## Dashboard Query Examples

### Recent Analyses
```sql
SELECT
  timestamp,
  total_changes,
  services_affected,
  risk_level,
  critical_issues
FROM `security_data.msa_latest_summary`
ORDER BY timestamp DESC
LIMIT 10
```

### Critical Issues Only
```sql
SELECT
  timestamp,
  risk_level,
  critical_issues,
  JSON_EXTRACT_SCALAR(recommendations_array[0], '$.action') as urgent_action
FROM `security_data.msa_critical_issues`
```

### Cost Impact Trends
```sql
SELECT
  analysis_date,
  price_increases,
  price_decreases,
  avg_risk_score
FROM `security_data.msa_billing_trends`
ORDER BY analysis_date DESC
LIMIT 30
```

## Set Up Alerts (Optional)

### Email Alerts via Pub/Sub

1. Create email subscription:
```bash
gcloud pubsub subscriptions create msa-email-alerts \
  --topic=msa-critical-alerts \
  --push-endpoint=https://your-email-service.com/webhook
```

2. Or pull messages manually:
```bash
gcloud pubsub subscriptions pull msa-critical-alerts-subscription \
  --auto-ack --limit=10
```

### Slack Integration

Create a Cloud Function to forward alerts to Slack:

```python
# slack_alert.py
import json
import requests
from google.cloud import pubsub_v1

def forward_to_slack(event, context):
    alert = json.loads(base64.b64decode(event['data']))

    slack_webhook = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    message = {
        "text": f"🚨 MSA Critical Alert",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert['action']}*\n{alert.get('details', '')}"
                }
            }
        ]
    }

    requests.post(slack_webhook, json=message)
```

## Troubleshooting

### Function Fails
```bash
# Check logs for errors
gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=20
```

### No Data in BigQuery
```bash
# Verify tables exist
bq ls security_data

# Check function permissions
gcloud projects get-iam-policy mgm-digitalconcierge \
  --flatten="bindings[].members" \
  --filter="bindings.members:msa-analyzer-sa"
```

### Re-deploy
```bash
# If something goes wrong, just re-run:
./deploy_complete.sh mgm-digitalconcierge us-central1
```
The script is idempotent - safe to run multiple times!

## Cost

**~$0.20/month** with daily runs:
- Cloud Functions: ~$0.05
- Cloud Scheduler: ~$0.10
- BigQuery: ~$0.02
- Storage: ~$0.01
- Pub/Sub: ~$0.02

## Next Steps

1. ✅ Run the deployment script
2. ✅ Test with manual invocation
3. ✅ Review first analysis results
4. ✅ Customize active_services table
5. ✅ Set up Slack/email alerts
6. ✅ Create BigQuery dashboard
7. ✅ Integrate with incident management

## Full Documentation

- **Complete Guide:** [README.md](README.md)
- **Testing Guide:** [TESTING.md](TESTING.md)
- **SQL Schema:** [bigquery_setup.sql](bigquery_setup.sql)

## Support

Questions? Check the logs first:
```bash
gcloud functions logs read msa-analyzer --region=us-central1 --limit=100
```

## Success!

Your MSA Analyzer is now monitoring GCP release notes 24/7. You'll get daily analysis results in BigQuery and alerts for critical issues via Pub/Sub.

🎉 **Happy Monitoring!**
