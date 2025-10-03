# Cloud Functions - Modular Data Fetchers

## Overview

This directory contains **modular, independent Cloud Functions** that fetch GCP data and store it in BigQuery. Each function runs independently on its own schedule, giving you complete control over what data to collect.

**Key Principle:** The Security Agent queries BigQuery directly - it never calls these functions. These are background data fetchers that keep your BigQuery tables fresh.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Modular Architecture                          │
└─────────────────────────────────────────────────────────────────┘

Customer Choice: Deploy only what you need!

┌───────────────────┐
│ Cloud Scheduler   │  ← You control schedule for each function
└────────┬──────────┘
         │
         │ (triggers independently)
         │
    ┌────┴────────────────────────────────────────────────┐
    │                                                      │
    ▼                          ▼                          ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Function 1 │         │  Function 2 │   ...   │  Function N │
│ (you pick)  │         │ (you pick)  │         │ (you pick)  │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                        │
       │ (writes to specific table)                     │
       │                       │                        │
       └───────────────────────┴────────────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │    BigQuery     │
                      │  (your tables)  │
                      └────────┬────────┘
                               │
                         (agent reads)
                               │
                               ▼
                      ┌─────────────────┐
                      │  Security Agent │
                      │   (run_query)   │
                      └─────────────────┘
```

---

## Available Functions

### 🔒 IAM & Security Functions

| Function | Purpose | BigQuery Table | Schedule Suggestion |
|----------|---------|----------------|---------------------|
| **fetch_custom_roles** | Custom IAM roles analysis | `iam_custom_roles` | Daily |
| **fetch_standard_roles** | Predefined GCP IAM roles | `iam_standard_roles` | Weekly |
| **fetch_iam_accounts** | All IAM bindings at project level | `iam_bindings` | Every 4 hours |
| **fetch_service_account_roles** | Service account role assignments | `service_account_roles` | Every 4 hours |
| **fetch_user_roles** | User IAM role assignments | `user_roles` | Every 4 hours |
| **fetch_security_findings** | Security Command Center findings | `security_findings` | Every 2 hours |
| **fetch_firewall_rules** | VPC firewall rules | `firewall_rules` | Every 4 hours |

### ☁️ Infrastructure Functions

| Function | Purpose | BigQuery Table | Schedule Suggestion |
|----------|---------|----------------|---------------------|
| **fetch_compute_instances** | Compute Engine VM instances | `compute_instances` | Every 2 hours |
| **fetch_storage_buckets** | Cloud Storage bucket info | `storage_buckets` | Every 4 hours |

### 📰 Feed & Documentation Functions

| Function | Purpose | BigQuery Table | Schedule Suggestion |
|----------|---------|----------------|---------------------|
| **fetch_gcp_release_notes** | GCP release notes from RSS | `gcp_release_notes` | Every 4 hours |
| **fetch_security_feeds** | CVE and threat intelligence | `security_threat_feeds` | Every 2 hours |
| **confluence_sync** | Confluence documentation | `confluence_documents` | Daily |

### 🎯 Analysis Functions

| Function | Purpose | BigQuery Table | Schedule Suggestion |
|----------|---------|----------------|---------------------|
| **msa_analyzer** ⭐ | Multi-service release notes analysis | `msa_analysis_history` | Daily at 9 AM |

---

## Deployment Options

### Option 1: Deploy Only What You Need (Recommended)

Pick and choose which functions to deploy:

```bash
# Deploy only IAM-related functions
cd fetch_custom_roles && ./deploy.sh mgm-digitalconcierge us-central1
cd ../fetch_iam_accounts && ./deploy.sh mgm-digitalconcierge us-central1

# Deploy MSA analyzer
cd ../msa_analyzer && ./deploy_complete.sh mgm-digitalconcierge us-central1
```

### Option 2: Deploy Everything

Use the provided deployment script:

```bash
cd scripts
./deploy_all_security_functions.sh mgm-digitalconcierge us-central1
```

This will:
- Deploy all 13 functions
- Set up default schedules
- Create BigQuery tables
- Configure IAM permissions

### Option 3: Deploy by Category

```bash
# IAM functions only
./deploy_iam_functions.sh mgm-digitalconcierge us-central1

# Infrastructure functions only
./deploy_infrastructure_functions.sh mgm-digitalconcierge us-central1

# Feed functions only
./deploy_feed_functions.sh mgm-digitalconcierge us-central1
```

---

## Function Independence

Each function is **completely independent**:

✅ **Independent deployment** - Deploy any function without others
✅ **Independent scheduling** - Set different schedules per function
✅ **Independent failure** - One function failing doesn't affect others
✅ **Independent cost** - Only pay for what you deploy
✅ **Independent updates** - Update one without touching others

---

## How the Agent Uses This Data

The Security Agent **never calls these Cloud Functions directly**. Instead:

1. **Cloud Functions** (background) → Write data to BigQuery
2. **Security Agent** (runtime) → Query BigQuery with `run_query()`

**Example agent usage:**

```python
# Agent queries BigQuery table populated by fetch_custom_roles
run_query("""
    SELECT role_name, permissions_count
    FROM security_insights.iam_custom_roles
    WHERE risk_level = 'HIGH'
""")

# Agent queries MSA results
run_query("""
    SELECT * FROM security_data.msa_latest_summary
    ORDER BY timestamp DESC LIMIT 10
""")
```

The agent doesn't care how the data got into BigQuery - it just queries it!

---

## Customization Guide

### Deploy Only What You Need

**Scenario 1: IAM-focused security team**
```bash
# Deploy only IAM functions
fetch_custom_roles/
fetch_standard_roles/
fetch_iam_accounts/
fetch_service_account_roles/
fetch_user_roles/
```

**Scenario 2: Infrastructure monitoring**
```bash
# Deploy compute + storage functions
fetch_compute_instances/
fetch_storage_buckets/
fetch_firewall_rules/
```

**Scenario 3: Release notes tracking only**
```bash
# Deploy just MSA analyzer
msa_analyzer/
```

**Scenario 4: Full security posture**
```bash
# Deploy everything
all 13 functions
```

### Adjust Schedules

Each function can run on its own schedule:

```bash
# Critical security data - every 2 hours
gcloud scheduler jobs create http fetch-security-findings \
  --schedule="0 */2 * * *"

# Less critical data - daily
gcloud scheduler jobs create http fetch-standard-roles \
  --schedule="0 9 * * *"

# Real-time needs - every 30 minutes
gcloud scheduler jobs create http fetch-firewall-rules \
  --schedule="*/30 * * * *"
```

---

## Cost Optimization

**Pay only for what you deploy!**

### Cost per Function (approximate)

| Deployment | Functions | Est. Monthly Cost |
|------------|-----------|-------------------|
| **Minimal** (MSA only) | 1 | $0.20 |
| **IAM-focused** (5 functions) | 5 | $1.00 |
| **Infrastructure** (3 functions) | 3 | $0.60 |
| **Full deployment** (all 13) | 13 | $2.60 |

### Cost Factors
- Function invocations (scheduler triggers)
- Execution time (usually <60 seconds)
- Memory (512MB default)
- BigQuery storage (~$0.02/GB/month)

---

## Common Deployment Patterns

### Pattern 1: Start Small, Grow Later
```bash
# Week 1: Deploy MSA only
cd msa_analyzer && ./deploy_complete.sh

# Week 2: Add IAM monitoring
cd ../fetch_custom_roles && ./deploy.sh
cd ../fetch_iam_accounts && ./deploy.sh

# Week 3: Add infrastructure
cd ../fetch_compute_instances && ./deploy.sh
cd ../fetch_firewall_rules && ./deploy.sh
```

### Pattern 2: Deploy by Priority
```bash
# High priority (deploy immediately)
msa_analyzer/
fetch_security_findings/
fetch_firewall_rules/

# Medium priority (deploy next week)
fetch_custom_roles/
fetch_iam_accounts/

# Low priority (deploy later)
fetch_standard_roles/
confluence_sync/
```

### Pattern 3: Compliance-Driven
```bash
# Required for SOC2/ISO27001
fetch_iam_accounts/
fetch_user_roles/
fetch_security_findings/
fetch_firewall_rules/

# Nice to have
everything else
```

---

## Function Details

### Each Function Directory Contains:

```
function_name/
├── main.py              # Function entry point
├── requirements.txt     # Python dependencies
├── deploy.sh           # Deployment script
├── README.md           # Function-specific docs
└── cloudbuild.yaml     # Build configuration (some functions)
```

### Standard Deployment Pattern

Every function follows the same deployment pattern:

```bash
cd function_name/
./deploy.sh <project-id> <region>
```

Example:
```bash
cd fetch_custom_roles/
./deploy.sh mgm-digitalconcierge us-central1
```

---

## BigQuery Tables Created

Each function writes to its own BigQuery table:

```
Dataset: security_insights
├── iam_custom_roles            (fetch_custom_roles)
├── iam_standard_roles          (fetch_standard_roles)
├── iam_bindings                (fetch_iam_accounts)
├── service_account_roles       (fetch_service_account_roles)
├── user_roles                  (fetch_user_roles)
├── security_findings           (fetch_security_findings)
├── firewall_rules              (fetch_firewall_rules)
├── compute_instances           (fetch_compute_instances)
├── storage_buckets             (fetch_storage_buckets)
├── gcp_release_notes           (fetch_gcp_release_notes)
├── security_threat_feeds       (fetch_security_feeds)
└── confluence_documents        (confluence_sync)

Dataset: security_data
├── msa_analysis_history        (msa_analyzer)
└── active_services             (msa_analyzer)
```

The agent can query **any of these tables** using `run_query()`.

---

## Monitoring

### Check Function Status

```bash
# List all deployed functions
gcloud functions list --project=mgm-digitalconcierge

# Check specific function
gcloud functions describe fetch-custom-roles \
  --region=us-central1

# View logs
gcloud functions logs read fetch-custom-roles \
  --region=us-central1 \
  --limit=50
```

### Check Scheduler Status

```bash
# List all scheduler jobs
gcloud scheduler jobs list --location=us-central1

# Check last execution
gcloud scheduler jobs describe fetch-custom-roles-daily \
  --location=us-central1
```

### Query BigQuery Data

```bash
# Check when data was last updated
bq query --use_legacy_sql=false '
SELECT table_name, TIMESTAMP_MILLIS(last_modified_time) as last_updated
FROM `mgm-digitalconcierge.security_insights.__TABLES__`
ORDER BY last_updated DESC
'
```

---

## Troubleshooting

### Function Not Deployed
```bash
# Re-deploy
cd function_name/
./deploy.sh mgm-digitalconcierge us-central1
```

### No Data in BigQuery
```bash
# Check function logs
gcloud functions logs read function-name --region=us-central1

# Trigger manually
gcloud scheduler jobs run function-name-job --location=us-central1
```

### Function Failing
```bash
# Check error logs
gcloud functions logs read function-name \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=20
```

---

## Best Practices

1. **Start small** - Deploy 2-3 critical functions first
2. **Monitor costs** - Check billing after first week
3. **Adjust schedules** - Tune based on your needs
4. **Review logs** - Check for failures weekly
5. **Update regularly** - Keep functions up to date
6. **Document choices** - Note which functions you deployed and why

---

## Agent Integration

The Security Agent automatically has access to **all BigQuery tables**, regardless of which functions are deployed.

**Agent capabilities:**
- ✅ Query any table: `run_query("SELECT * FROM security_insights.iam_custom_roles")`
- ✅ List available tables: `list_tables("security_insights")`
- ✅ Explore schema: `get_table_schema("security_insights", "iam_custom_roles")`
- ✅ Join across tables: Complex multi-table queries work fine

**The agent is completely decoupled from the Cloud Functions!**

---

## Summary

✅ **Modular Design** - Pick and choose what to deploy
✅ **Independent** - Each function operates separately
✅ **Cost Effective** - Pay only for what you use
✅ **Flexible Scheduling** - Different schedules per function
✅ **Agent-Ready** - Agent queries BigQuery, not functions
✅ **Easy Management** - Standard deployment pattern
✅ **Production Ready** - Monitoring, logging, error handling included

Deploy what makes sense for YOUR use case!
