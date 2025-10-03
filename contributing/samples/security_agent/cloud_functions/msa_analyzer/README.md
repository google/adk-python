# MSA (Multi-Service Analyzer) - Cloud Function Setup

## Overview

The MSA Analyzer monitors GCP release notes and automatically analyzes impacts on:
- **Security** - Critical updates, authentication changes, encryption changes
- **Billing** - Pricing changes, cost optimizations
- **Compliance** - Certifications, regulatory changes

## Quick Start

### Prerequisites
- GCP Project with billing enabled
- `gcloud` CLI installed and authenticated
- Permissions: `roles/owner` or equivalent

### One-Command Complete Deployment

```bash
./deploy_complete.sh <your-project-id> us-central1
```

**This single script does EVERYTHING:**
1. ✅ Enables all required GCP APIs
2. ✅ Creates service accounts with IAM permissions
3. ✅ Creates BigQuery dataset, tables, and views
4. ✅ Sets up Cloud Storage bucket for caching
5. ✅ Configures Pub/Sub topic and subscription for alerts
6. ✅ Deploys Cloud Function with proper configuration
7. ✅ Creates Cloud Scheduler job for daily automated runs

**Total time:** ~5 minutes | **No interaction required** - fully automated!

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GCP Release Notes (RSS)                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Cloud Scheduler (Daily at 9 AM)                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Cloud Function: MSA Analyzer                        │
│  • Fetches release notes                                         │
│  • Filters for active services                                   │
│  • Analyzes security/billing/compliance impacts                  │
│  • Generates recommendations                                     │
└──────┬──────────────────┬────────────────────┬──────────────────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌─────────────┐  ┌──────────────┐   ┌─────────────────┐
│  BigQuery   │  │ Cloud Storage│   │    Pub/Sub      │
│  (Results)  │  │   (Cache)    │   │ (Alerts)        │
└─────────────┘  └──────────────┘   └─────────────────┘
```

## BigQuery Tables

### `security_data.msa_analysis_history`
Stores complete analysis results:
- Analysis ID, timestamp, risk scores
- Security, billing, compliance impacts
- Recommendations
- Full JSON report

### `security_data.active_services`
Lists GCP services to monitor:
```sql
SELECT service_name, service_type, status
FROM `security_data.active_services`
WHERE status = 'active'
```

**Customize this table** with your organization's services!

### Views
- `msa_latest_summary` - Recent analyses
- `msa_critical_issues` - High-priority items
- `msa_billing_trends` - Cost impact trends

## Usage

### Manual Trigger

Test the function manually:

```bash
# Get function URL
FUNCTION_URL=$(gcloud functions describe msa-analyzer \
  --region=us-central1 \
  --format="value(serviceConfig.uri)")

# Trigger analysis
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}'
```

### Scheduled Execution

The Cloud Scheduler job runs daily at 9 AM ET:

```bash
# Run scheduler job immediately
gcloud scheduler jobs run msa-analyzer-daily \
  --location=us-central1

# View scheduler logs
gcloud scheduler jobs describe msa-analyzer-daily \
  --location=us-central1
```

### Query Results

```bash
# Latest analysis summary
bq query --use_legacy_sql=false '
SELECT
  timestamp,
  total_changes,
  services_affected,
  risk_level,
  critical_issues
FROM `security_data.msa_latest_summary`
ORDER BY timestamp DESC
LIMIT 5'

# Critical issues
bq query --use_legacy_sql=false '
SELECT
  timestamp,
  risk_level,
  critical_issues,
  JSON_EXTRACT_SCALAR(recommendations, "$[0].action") as top_action
FROM `security_data.msa_critical_issues`
LIMIT 10'
```

### View Logs

```bash
# Cloud Function logs
gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --limit=50

# Filter for errors
gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=20
```

## Configuration

### Environment Variables

Set in Cloud Function:
- `GCP_PROJECT` - Your GCP project ID
- `GOOGLE_CLOUD_PROJECT` - Alternative project ID

### Customize Active Services

Update the services you want to monitor:

```sql
-- Add new service
INSERT INTO `security_data.active_services`
(service_name, service_type, status, enabled_date)
VALUES
  ('Cloud Logging', 'operations', 'active', CURRENT_DATE());

-- Disable monitoring for a service
UPDATE `security_data.active_services`
SET status = 'inactive'
WHERE service_name = 'Some Old Service';
```

### Adjust Schedule

Change the Cloud Scheduler frequency:

```bash
# Modify schedule (e.g., twice daily)
gcloud scheduler jobs update http msa-analyzer-daily \
  --location=us-central1 \
  --schedule="0 9,17 * * *"
```

## Alerts

### Pub/Sub Integration

Critical issues are published to `msa-critical-alerts` topic:

```bash
# Subscribe to alerts
gcloud pubsub subscriptions pull msa-alerts-subscription \
  --auto-ack \
  --limit=10
```

### Custom Alert Handlers

Create a subscriber function:

```python
from google.cloud import pubsub_v1

def handle_alert(event, context):
    import json
    alert = json.loads(base64.b64decode(event['data']))

    if alert['priority'] == 'critical':
        # Send to Slack, PagerDuty, email, etc.
        send_notification(alert)
```

## Monitoring

### Key Metrics

- Analysis execution time
- Number of changes detected
- Risk score trends
- Critical issues over time

### Dashboards

Query for dashboard data:

```sql
-- Weekly risk trend
SELECT
  DATE_TRUNC(timestamp, WEEK) as week,
  AVG(risk_score) as avg_risk,
  SUM(critical_issues) as total_critical
FROM `security_data.msa_analysis_history`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY week
ORDER BY week DESC;
```

## Troubleshooting

### Function Fails to Deploy

```bash
# Check enabled APIs
gcloud services list --enabled

# View build logs
gcloud builds list --limit=5
```

### No Results in BigQuery

1. Check function logs for errors
2. Verify BigQuery permissions
3. Ensure dataset/tables exist:
   ```bash
   bq ls security_data
   ```

### No Release Notes Fetched

- GCP RSS feed may be unavailable (check logs)
- Function falls back to web scraping
- Verify internet access from Cloud Function

### Cache Issues

```bash
# Clear cache
gsutil rm -r gs://YOUR-PROJECT-msa-cache/processed_notes.json
```

## Cost Estimates

Typical monthly costs (based on daily runs):

| Resource | Usage | Est. Cost |
|----------|-------|-----------|
| Cloud Functions | 30 invocations/month, 512MB, ~60s | $0.05 |
| Cloud Scheduler | 1 job, 30 runs/month | $0.10 |
| BigQuery | 1GB storage, minimal queries | $0.02 |
| Cloud Storage | 1MB cache | $0.00 |
| **Total** | | **~$0.20/month** |

## Security

- Function uses service account with least-privilege access
- No public data exposure
- Results stored in your BigQuery project
- Optional: Enable VPC connector for private access

## Development

### Local Testing

```bash
cd cloud_functions/msa_analyzer
cp ../../agents/_tools/msa_analyzer.py .

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
export GOOGLE_CLOUD_PROJECT="your-project"

# Run locally
python main.py
```

### Update Function

```bash
# After code changes
gcloud functions deploy msa-analyzer \
  --region=us-central1 \
  --source=.
```

## Integration with Security Agent

The MSA Analyzer can be called from the ADK Security Agent:

```python
from agents._tools.msa_analyzer import analyze_gcp_releases

# Run analysis
report = analyze_gcp_releases(days_back=7)

# Access results
print(f"Risk Level: {report['summary']['overall_risk_level']}")
print(f"Critical Issues: {report['summary']['critical_issues']}")
```

## Support

- View logs: `gcloud functions logs read msa-analyzer --region=us-central1`
- Check status: `gcloud functions describe msa-analyzer --region=us-central1`
- File issues: Create ticket with logs and error messages

## Next Steps

1. ✅ Complete setup with `./setup_msa.sh`
2. ✅ Customize `active_services` table
3. ✅ Run manual test
4. ✅ Review first analysis results
5. ✅ Setup alerting (Slack, email, etc.)
6. ✅ Create BigQuery dashboard
7. ✅ Integrate with incident management

## Resources

- [GCP Release Notes](https://cloud.google.com/release-notes)
- [Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)
