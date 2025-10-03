# MSA Analyzer - Quick Reference Card

## 🚀 One-Command Deploy

```bash
./deploy_complete.sh mgm-digitalconcierge us-central1
```

**Time:** 5 minutes | **Cost:** ~$0.20/month

---

## 📊 What Gets Created

| Resource | Name | Purpose |
|----------|------|---------|
| **BigQuery Dataset** | `security_data` | Stores MSA results |
| **Table** | `msa_analysis_history` | All analysis results |
| **Table** | `active_services` | Monitored services |
| **View** | `msa_latest_summary` | Last 30 days |
| **View** | `msa_critical_issues` | High-priority only |
| **View** | `msa_billing_trends` | Cost impacts |
| **Cloud Function** | `msa-analyzer` | Runs analysis |
| **Cloud Scheduler** | `msa-analyzer-daily` | Daily trigger at 9 AM |
| **Storage Bucket** | `{project}-msa-cache` | Processed notes cache |
| **Pub/Sub Topic** | `msa-critical-alerts` | Critical alerts |

---

## 🧪 Quick Tests

### 1. Manual Trigger
```bash
curl -X POST $(gcloud functions describe msa-analyzer \
  --region=us-central1 \
  --format="value(serviceConfig.uri)") \
  -H 'Content-Type: application/json' \
  -d '{"days_back": 7}'
```

### 2. Query Results
```bash
bq query --use_legacy_sql=false \
  'SELECT * FROM `security_data.msa_latest_summary` LIMIT 5'
```

### 3. Check Logs
```bash
gcloud functions logs read msa-analyzer --region=us-central1 --limit=20
```

---

## 🤖 Agent Integration

**The Security Agent can now:**

### Analyze Release Notes
```python
analyze_gcp_releases(days_back=7)
```

### Query MSA Data
```sql
-- Latest analyses
SELECT * FROM security_data.msa_latest_summary
ORDER BY timestamp DESC LIMIT 10

-- Critical issues
SELECT * FROM security_data.msa_critical_issues
WHERE risk_level = 'high'

-- Billing trends
SELECT * FROM security_data.msa_billing_trends
WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)

-- Active services
SELECT * FROM security_data.active_services
WHERE status = 'active'
```

### Agent Queries
```
"Analyze recent GCP release notes"
"Show me critical updates"
"What GCP services are we monitoring?"
"Any pricing changes recently?"
"What's the risk level of recent changes?"
```

---

## 📋 Common Queries

### Latest Analysis Summary
```sql
SELECT
  timestamp,
  total_changes,
  services_affected,
  risk_level,
  critical_issues,
  top_recommendation
FROM `security_data.msa_latest_summary`
ORDER BY timestamp DESC
LIMIT 10
```

### Critical Issues with Actions
```sql
SELECT
  timestamp,
  risk_level,
  JSON_EXTRACT_SCALAR(recommendations_array[0], '$.action') as urgent_action,
  JSON_EXTRACT_SCALAR(recommendations_array[0], '$.deadline') as deadline
FROM `security_data.msa_critical_issues`
```

### Price Changes This Month
```sql
SELECT
  analysis_date,
  price_increases,
  price_decreases
FROM `security_data.msa_billing_trends`
WHERE analysis_date >= DATE_TRUNC(CURRENT_DATE(), MONTH)
ORDER BY analysis_date DESC
```

### Services Being Monitored
```sql
SELECT
  service_name,
  service_type,
  COUNT(*) as mention_count
FROM `security_data.active_services`
WHERE status = 'active'
GROUP BY service_name, service_type
ORDER BY service_name
```

---

## 🔧 Maintenance

### Add Service to Monitor
```sql
INSERT INTO `security_data.active_services`
(service_name, service_type, status, enabled_date, created_at, updated_at)
VALUES
  ('Cloud Logging', 'operations', 'active', CURRENT_DATE(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
```

### Update Scheduler Frequency
```bash
gcloud scheduler jobs update http msa-analyzer-daily \
  --location=us-central1 \
  --schedule="0 9,17 * * *"  # Twice daily
```

### Clear Cache
```bash
gsutil rm gs://mgm-digitalconcierge-msa-cache/processed_notes.json
```

---

## 📈 Monitoring

### Function Metrics
```bash
gcloud monitoring time-series list \
  --filter='metric.type="cloudfunctions.googleapis.com/function/execution_count"'
```

### Error Rate
```bash
gcloud functions logs read msa-analyzer \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=20
```

---

## 🚨 Alerts Setup

### Pull Critical Alerts
```bash
gcloud pubsub subscriptions pull msa-critical-alerts-subscription \
  --auto-ack --limit=10
```

### Slack Webhook (Example)
```python
import requests
alert = {
    "text": f"🚨 {message}",
    "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": details}}]
}
requests.post("https://hooks.slack.com/YOUR/WEBHOOK", json=alert)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Function fails | Check logs: `gcloud functions logs read msa-analyzer --region=us-central1` |
| No data in BQ | Verify permissions: `gcloud projects get-iam-policy mgm-digitalconcierge` |
| Timeout | Increase timeout: `--timeout=540s` in deploy command |
| Out of memory | Increase memory: `--memory=1024MB` in deploy command |

---

## 📚 Full Documentation

- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Complete Guide:** [README.md](README.md)
- **Testing:** [TESTING.md](TESTING.md)
- **Integration:** [../../../MSA_INTEGRATION_COMPLETE.md](../../../MSA_INTEGRATION_COMPLETE.md)

---

## 💡 Pro Tips

1. **Customize active_services table** for your specific GCP services
2. **Set up Slack alerts** for critical issues via Pub/Sub
3. **Create BigQuery dashboard** using Data Studio or Looker
4. **Schedule runs** at off-peak hours to reduce costs
5. **Use views** for common queries (already created!)

---

## ✅ Success Checklist

- [ ] Deployed via `./deploy_complete.sh`
- [ ] Manual test returns valid JSON
- [ ] BigQuery tables have data
- [ ] Scheduler runs successfully
- [ ] Agent can query MSA data
- [ ] Active services customized
- [ ] Alerts configured

---

**Questions?** Check logs first:
```bash
gcloud functions logs read msa-analyzer --region=us-central1 --limit=100
```
