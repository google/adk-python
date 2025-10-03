# MSA Analyzer - Testing Guide

## Quick Test Checklist

After running `./setup_msa.sh`, verify everything works:

### ✅ 1. BigQuery Tables Created

```bash
# Check dataset exists
bq ls | grep security_data

# Check tables
bq ls security_data

# Expected output:
#   msa_analysis_history
#   active_services
```

### ✅ 2. Cloud Function Deployed

```bash
# Check function status
gcloud functions describe msa-analyzer --region=us-central1

# Should show: state: ACTIVE
```

### ✅ 3. Storage Bucket Created

```bash
# Verify bucket exists
gsutil ls | grep msa-cache

# Expected: gs://YOUR-PROJECT-msa-cache/
```

### ✅ 4. Pub/Sub Topic Created

```bash
# Check topic
gcloud pubsub topics list | grep msa-critical-alerts

# Check subscription
gcloud pubsub subscriptions list | grep msa-alerts
```

### ✅ 5. Service Account Permissions

```bash
# Verify service account
gcloud iam service-accounts list | grep msa-analyzer-sa

# Check IAM bindings
gcloud projects get-iam-policy YOUR-PROJECT \
  --flatten="bindings[].members" \
  --filter="bindings.members:msa-analyzer-sa"
```

## Functional Tests

### Test 1: Manual Function Invocation

```bash
# Get function URL
FUNCTION_URL=$(gcloud functions describe msa-analyzer \
  --region=us-central1 \
  --format="value(serviceConfig.uri)")

# Invoke function
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}' | jq .
```

**Expected Response:**
```json
{
  "success": true,
  "analysis_id": "abc123def456",
  "timestamp": "2025-10-02T17:00:00Z",
  "summary": {
    "total_changes": 15,
    "services_affected": 8,
    "risk_level": "medium",
    "critical_issues": 0
  },
  "top_recommendations": [...],
  "message": "MSA analyzed 15 changes affecting 8 services..."
}
```

### Test 2: BigQuery Data Verification

```bash
# Check if data was written
bq query --use_legacy_sql=false '
SELECT
  analysis_id,
  timestamp,
  total_changes,
  risk_level,
  critical_issues
FROM `security_data.msa_analysis_history`
ORDER BY timestamp DESC
LIMIT 5'
```

**Expected:** At least 1 row with your recent analysis

### Test 3: View Active Services

```bash
bq query --use_legacy_sql=false '
SELECT service_name, status
FROM `security_data.active_services`
WHERE status = "active"'
```

**Expected:** List of active GCP services

### Test 4: Scheduler Job (if enabled)

```bash
# Trigger scheduler manually
gcloud scheduler jobs run msa-analyzer-daily \
  --location=us-central1

# Wait 30 seconds, then check logs
sleep 30

gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --limit=20
```

### Test 5: Pub/Sub Alerts

```bash
# Trigger function with mock critical issue
# (Would need to modify code or have actual critical GCP update)

# Check for messages
gcloud pubsub subscriptions pull msa-alerts-subscription \
  --auto-ack \
  --limit=5
```

## Performance Tests

### Execution Time

```bash
# Time the function
time curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}'
```

**Expected:** < 60 seconds for 7 days of data

### Memory Usage

```bash
# Check function metrics
gcloud functions describe msa-analyzer \
  --region=us-central1 \
  --format="value(buildConfig.runtime,serviceConfig.availableMemory)"
```

**Expected:** python311, 512MB

### Historical Analysis

```bash
# Test with larger time window
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 30}'
```

**Expected:** Still completes within 5 minutes

## Error Scenarios

### Test Error Handling

#### 1. Invalid Parameters

```bash
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": -1}'

# Should return 4xx error with message
```

#### 2. Missing Permissions

```bash
# Temporarily remove BigQuery permissions
gcloud projects remove-iam-policy-binding YOUR-PROJECT \
  --member="serviceAccount:msa-analyzer-sa@YOUR-PROJECT.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# Trigger function - should log warning but continue
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}'

# Restore permissions
gcloud projects add-iam-policy-binding YOUR-PROJECT \
  --member="serviceAccount:msa-analyzer-sa@YOUR-PROJECT.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

#### 3. Network Issues

Check logs for RSS feed fallback:

```bash
gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --filter="textPayload:fallback OR textPayload:scraping" \
  --limit=10
```

## Integration Tests

### Test with ADK Agent

```python
# In your ADK agent environment
import sys
sys.path.append('/path/to/security_agent/agents/_tools')

from msa_analyzer import MSAAnalyzer

# Initialize
analyzer = MSAAnalyzer(project_id='YOUR-PROJECT')

# Run analysis
report = analyzer.analyze_release_notes(days_back=7)

# Verify results
assert report['analysis_id']
assert report['summary']['total_changes_analyzed'] >= 0
assert report['summary']['overall_risk_level'] in ['low', 'medium', 'high']

print("✅ ADK Integration test passed!")
```

### Test Cache Functionality

```bash
# Run once
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}' > /tmp/run1.json

# Wait 1 minute
sleep 60

# Run again - should use cache
curl -X POST $FUNCTION_URL \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}' > /tmp/run2.json

# Check cache file was created
gsutil ls gs://YOUR-PROJECT-msa-cache/processed_notes.json

# Compare results - total_changes should be different
# (or 0 if all notes already processed)
```

## Load Testing

### Concurrent Requests

```bash
# Send 5 concurrent requests
for i in {1..5}; do
  curl -X POST $FUNCTION_URL \
    -H 'Content-Type: application/json' \
    -d '{"days_back": 7}' &
done
wait

# Check logs for concurrency issues
gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --limit=50
```

## Monitoring Verification

### Check Cloud Monitoring Metrics

```bash
# View invocations
gcloud monitoring time-series list \
  --filter='metric.type="cloudfunctions.googleapis.com/function/execution_count"' \
  --format=json

# View errors
gcloud monitoring time-series list \
  --filter='metric.type="cloudfunctions.googleapis.com/function/execution_error_count"' \
  --format=json
```

### Setup Alert Policy

```bash
# Create alert for function failures
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="MSA Analyzer Failures" \
  --condition-display-name="High error rate" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=60s \
  --condition-filter='resource.type="cloud_function" AND metric.type="cloudfunctions.googleapis.com/function/execution_error_count"'
```

## Cleanup (After Testing)

If you want to remove everything:

```bash
# Delete function
gcloud functions delete msa-analyzer --region=us-central1

# Delete scheduler job
gcloud scheduler jobs delete msa-analyzer-daily --location=us-central1

# Delete pub/sub
gcloud pubsub subscriptions delete msa-alerts-subscription
gcloud pubsub topics delete msa-critical-alerts

# Delete BigQuery dataset (WARNING: deletes all data)
bq rm -r -f security_data

# Delete storage bucket
gsutil rm -r gs://YOUR-PROJECT-msa-cache

# Delete service accounts
gcloud iam service-accounts delete msa-analyzer-sa@YOUR-PROJECT.iam.gserviceaccount.com
gcloud iam service-accounts delete msa-scheduler-sa@YOUR-PROJECT.iam.gserviceaccount.com
```

## Troubleshooting

### Function times out

```bash
# Increase timeout
gcloud functions deploy msa-analyzer \
  --region=us-central1 \
  --timeout=540s  # Max is 540s (9 minutes)
```

### Out of memory

```bash
# Increase memory
gcloud functions deploy msa-analyzer \
  --region=us-central1 \
  --memory=1024MB
```

### BigQuery quota exceeded

```bash
# Check quotas
gcloud compute project-info describe --project=YOUR-PROJECT

# Request quota increase if needed
```

### No release notes fetched

- Check GCP RSS feed is accessible: https://cloud.google.com/feeds/gcp-release-notes.xml
- Function falls back to web scraping automatically
- Verify function has internet access

## Success Criteria

Your MSA setup is successful when:

- ✅ Function deploys without errors
- ✅ Manual invocation returns valid JSON
- ✅ BigQuery tables populated with analysis data
- ✅ Active services table contains your GCP services
- ✅ Function completes in < 60 seconds for 7 days
- ✅ Scheduler job runs successfully (if enabled)
- ✅ No errors in function logs
- ✅ Cache file created in storage bucket
- ✅ Results queryable in BigQuery

## Getting Help

1. Check logs: `gcloud functions logs read msa-analyzer --region=us-central1 --limit=100`
2. Verify setup: Re-run `./setup_msa.sh` (it's idempotent)
3. Test components individually using tests above
4. Review README.md for configuration options
