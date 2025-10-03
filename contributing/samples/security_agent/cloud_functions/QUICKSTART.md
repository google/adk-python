# Cloud Functions - Quick Start Guide

## Choose Your Deployment

### Option 1: Interactive Deployment (Easiest)

```bash
cd cloud_functions
./deploy_selected.sh mgm-digitalconcierge us-central1
```

**Interactive menu lets you choose:**
- 🔒 IAM & Security functions only
- ☁️ Infrastructure functions only
- 📰 Feeds & Documentation only
- 🎯 MSA Analyzer only (recommended to start!)
- 🎁 Everything
- 🎨 Custom selection

### Option 2: Deploy Individual Functions

```bash
# Deploy just MSA Analyzer (recommended first step)
cd msa_analyzer
./deploy_complete.sh mgm-digitalconcierge us-central1

# Deploy specific function
cd ../fetch_custom_roles
./deploy.sh mgm-digitalconcierge us-central1
```

### Option 3: Deploy Everything

```bash
cd cloud_functions
./deploy_selected.sh mgm-digitalconcierge us-central1
# Choose option 5 (Everything)
```

---

## Available Functions

### 🎯 Start Here (Recommended)

**MSA Analyzer** - Multi-Service Analyzer
- Analyzes GCP release notes
- Security, billing, compliance impacts
- Daily automated runs
- **Cost:** ~$0.20/month

```bash
cd msa_analyzer
./deploy_complete.sh mgm-digitalconcierge us-central1
```

### 🔒 IAM & Security (7 functions)

Pick which IAM data you need:

| Function | What it does | Update frequency |
|----------|--------------|------------------|
| `fetch_custom_roles` | Custom IAM role analysis | Daily |
| `fetch_standard_roles` | All GCP predefined roles | Weekly |
| `fetch_iam_accounts` | IAM bindings (who has what) | Every 4 hours |
| `fetch_service_account_roles` | Service account permissions | Every 4 hours |
| `fetch_user_roles` | User permissions | Every 4 hours |
| `fetch_security_findings` | Security Command Center | Every 2 hours |
| `fetch_firewall_rules` | VPC firewall rules | Every 4 hours |

### ☁️ Infrastructure (2 functions)

| Function | What it does | Update frequency |
|----------|--------------|------------------|
| `fetch_compute_instances` | VM inventory | Every 2 hours |
| `fetch_storage_buckets` | GCS bucket info | Every 4 hours |

### 📰 Feeds (3 functions)

| Function | What it does | Update frequency |
|----------|--------------|------------------|
| `fetch_gcp_release_notes` | GCP release notes RSS | Every 4 hours |
| `fetch_security_feeds` | CVE & threat intel | Every 2 hours |
| `confluence_sync` | Confluence docs | Daily |

---

## Cost Guide

**Pay only for what you deploy:**

| Deployment | Monthly Cost |
|------------|--------------|
| MSA only | ~$0.20 |
| MSA + IAM (5 functions) | ~$1.20 |
| MSA + Everything | ~$2.80 |

---

## Common Patterns

### Pattern 1: Start Small (Recommended)

**Week 1:**
```bash
cd msa_analyzer
./deploy_complete.sh mgm-digitalconcierge us-central1
```

**Week 2 (add IAM monitoring):**
```bash
cd ../fetch_custom_roles && ./deploy.sh mgm-digitalconcierge us-central1
cd ../fetch_iam_accounts && ./deploy.sh mgm-digitalconcierge us-central1
```

**Week 3 (add infrastructure):**
```bash
cd ../fetch_compute_instances && ./deploy.sh mgm-digitalconcierge us-central1
cd ../fetch_firewall_rules && ./deploy.sh mgm-digitalconcierge us-central1
```

### Pattern 2: IAM-Focused Team

```bash
./deploy_selected.sh mgm-digitalconcierge us-central1
# Choose option 1 (IAM & Security)
```

### Pattern 3: Full Security Posture

```bash
./deploy_selected.sh mgm-digitalconcierge us-central1
# Choose option 5 (Everything)
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│          You Choose What to Deploy                      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│      Cloud Functions (run on schedule)                  │
│      • Fetch data from GCP APIs                         │
│      • Write to BigQuery                                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    BigQuery     │
              │  (your tables)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Security Agent │
              │  (queries data) │
              └─────────────────┘
```

**Key:** Agent reads BigQuery, not the functions!

---

## Quick Commands

### Deploy MSA Only
```bash
cd msa_analyzer && ./deploy_complete.sh mgm-digitalconcierge us-central1
```

### Deploy with Interactive Menu
```bash
./deploy_selected.sh mgm-digitalconcierge us-central1
```

### Check Deployed Functions
```bash
gcloud functions list --project=mgm-digitalconcierge
```

### View Function Logs
```bash
gcloud functions logs read msa-analyzer --region=us-central1 --limit=20
```

### Check BigQuery Tables
```bash
bq ls security_insights
bq ls security_data
```

### Test Agent Access
```bash
python -c "from agents.agent import root_agent; print(root_agent.chat('What datasets do I have?'))"
```

---

## Troubleshooting

### Function deployment fails
```bash
# Check you have the right permissions
gcloud projects get-iam-policy mgm-digitalconcierge | grep $(gcloud config get-value account)

# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com cloudscheduler.googleapis.com
```

### No data in BigQuery
```bash
# Trigger function manually
gcloud scheduler jobs run function-name-job --location=us-central1

# Check function logs
gcloud functions logs read function-name --region=us-central1
```

### Agent can't find data
```bash
# List available tables
bq ls security_insights
bq ls security_data

# Query directly
bq query --use_legacy_sql=false 'SELECT * FROM security_insights.iam_custom_roles LIMIT 5'
```

---

## What's Next?

1. **Deploy MSA** - Start with the analyzer
2. **Test it** - Check BigQuery has data
3. **Query with agent** - Ask the agent about your data
4. **Add more functions** - Deploy additional fetchers as needed
5. **Monitor** - Check logs and costs weekly

---

## Full Documentation

- **Detailed Overview:** [README.md](README.md)
- **MSA Analyzer:** [msa_analyzer/README.md](msa_analyzer/README.md)
- **All Functions Analysis:** [CLOUD_FUNCTIONS_ANALYSIS.md](CLOUD_FUNCTIONS_ANALYSIS.md)

---

## Support

**Quick checks:**
```bash
# List functions
gcloud functions list --project=mgm-digitalconcierge

# Check logs
gcloud functions logs read <function-name> --region=us-central1

# Query BigQuery
bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM security_insights.iam_custom_roles'
```

**Remember:** Deploy only what you need. You can always add more later!
