# Cloud Functions Analysis & Consolidation Plan

## Current State: 13 Cloud Functions 😱

You're absolutely right - this is way too many! Here's what we have:

### Data Fetchers (10 functions - all doing similar things)
1. **fetch_compute_instances** - Gets VM instances → BigQuery
2. **fetch_custom_roles** - Gets custom IAM roles → BigQuery
3. **fetch_firewall_rules** - Gets firewall rules → BigQuery
4. **fetch_iam_accounts** - Gets IAM bindings → BigQuery
5. **fetch_security_findings** - Gets Security Command Center findings → BigQuery
6. **fetch_service_account_roles** - Gets SA role assignments → BigQuery
7. **fetch_standard_roles** - Gets predefined IAM roles → BigQuery
8. **fetch_storage_buckets** - Gets GCS bucket info → BigQuery
9. **fetch_user_roles** - Gets user IAM bindings → BigQuery
10. **fetch_gcp_release_notes** - Gets RSS feeds → BigQuery

### Special Purpose (3 functions)
11. **fetch_security_feeds** - CVE/threat intel → BigQuery
12. **confluence_sync** - Confluence docs → BigQuery
13. **msa_analyzer** - Analyzes release notes, stores results → BigQuery

---

## Key Insight: Agent Never Calls These Functions! 🎯

**The agent queries BigQuery directly using `run_query()`.**

The Cloud Functions are just **scheduled data fetchers** that run periodically to keep BigQuery tables fresh.

```
┌─────────────────────────────────────────────────────────────┐
│                    Current Architecture                      │
└─────────────────────────────────────────────────────────────┘

Cloud Scheduler                    Cloud Functions (13)
      │                                    │
      │ (triggers every X hours)          │
      ├────────────────────────────────────┤
      │                                    │
      │                              ┌─────▼─────┐
      │                              │  fetch_*  │
      │                              │ functions │
      │                              └─────┬─────┘
      │                                    │
      │                              (writes data)
      │                                    │
      │                                    ▼
      │                           ┌─────────────────┐
      │                           │    BigQuery     │
      │                           │  (all tables)   │
      │                           └────────┬────────┘
      │                                    │
      │                              (reads data)
      │                                    │
      │                                    ▼
      │                           ┌─────────────────┐
      │                           │  Security Agent │
      │                           │   (run_query)   │
      │                           └─────────────────┘
```

---

## The Problem

1. **Maintenance Nightmare** - 13 separate functions to deploy, monitor, debug
2. **Duplicate Code** - Each fetch_* function has similar BigQuery write logic
3. **Cost** - 13 function deployments, 13 scheduler jobs
4. **Complexity** - Hard to understand what data is available

---

## Proposed Solution: Consolidate into ONE Data Fetcher

### New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Consolidated Architecture                  │
└─────────────────────────────────────────────────────────────┘

Cloud Scheduler
      │
      │ (triggers daily/hourly)
      │
      ▼
┌────────────────────────────────────────────────────────────┐
│      Unified Data Fetcher Function                          │
│                                                              │
│  • Compute instances                                        │
│  • Firewall rules                                           │
│  • IAM (users, SAs, roles, bindings)                       │
│  • Storage buckets                                          │
│  • Security findings                                        │
│  • GCP release notes (RSS)                                  │
│  • Security feeds (CVE)                                     │
│                                                              │
│  All in ONE function with modular fetchers                  │
└──────────────────────┬─────────────────────────────────────┘
                       │
                 (batch write)
                       │
                       ▼
              ┌─────────────────┐
              │    BigQuery     │
              │   (all tables)  │
              └────────┬────────┘
                       │
                 (reads data)
                       │
                       ▼
              ┌─────────────────┐
              │  Security Agent │
              │   (run_query)   │
              └─────────────────┘

Keep Separate:
  • MSA Analyzer (complex analysis logic)
  • Confluence Sync (external API, different schedule)
```

---

## Consolidation Plan

### Phase 1: Create Unified Data Fetcher

**File:** `cloud_functions/unified_data_fetcher/main.py`

```python
"""
Unified GCP Data Fetcher
Consolidates 10 separate fetch functions into one modular function
"""

import os
from google.cloud import bigquery, compute_v1, storage, iam_admin_v1
from typing import Dict, List

# Modular fetchers
from fetchers import (
    fetch_compute_instances,
    fetch_firewall_rules,
    fetch_iam_data,
    fetch_storage_data,
    fetch_security_findings,
    fetch_release_notes,
    fetch_security_feeds
)

def unified_data_fetch(request):
    """
    Single entry point for all data fetching.
    Can fetch all data or specific resources based on request.
    """
    request_json = request.get_json(silent=True)

    # Default: fetch everything
    resources = request_json.get('resources', 'all') if request_json else 'all'

    results = {}

    if resources == 'all' or 'compute' in resources:
        results['compute'] = fetch_compute_instances()

    if resources == 'all' or 'firewall' in resources:
        results['firewall'] = fetch_firewall_rules()

    if resources == 'all' or 'iam' in resources:
        results['iam'] = fetch_iam_data()

    if resources == 'all' or 'storage' in resources:
        results['storage'] = fetch_storage_data()

    if resources == 'all' or 'security' in resources:
        results['security'] = fetch_security_findings()

    if resources == 'all' or 'feeds' in resources:
        results['release_notes'] = fetch_release_notes()
        results['security_feeds'] = fetch_security_feeds()

    return {
        'success': True,
        'fetched': list(results.keys()),
        'details': results
    }
```

### Phase 2: Deployment

**Single deployment command:**
```bash
./deploy_unified_fetcher.sh mgm-digitalconcierge us-central1
```

**Single scheduler job:**
```bash
gcloud scheduler jobs create http unified-data-fetch \
  --schedule="0 */2 * * *" \  # Every 2 hours
  --uri=$FUNCTION_URL \
  --message-body='{"resources": "all"}'
```

### Phase 3: Keep Separate (3 functions)

1. **msa_analyzer** - Complex analysis logic, daily schedule
2. **confluence_sync** - External Confluence API, different schedule
3. **unified_data_fetcher** - All GCP resource fetching

**From 13 → 3 functions!**

---

## Benefits

### Before (13 functions)
- ❌ 13 separate deployments
- ❌ 13 scheduler jobs
- ❌ Duplicate BigQuery write code
- ❌ Hard to know what data exists
- ❌ 13 sets of logs to check
- ❌ ~13 × $0.20 = $2.60/month

### After (3 functions)
- ✅ 3 deployments
- ✅ 3 scheduler jobs
- ✅ Shared code, easier maintenance
- ✅ Clear data inventory
- ✅ 3 sets of logs
- ✅ ~3 × $0.20 = $0.60/month

**Savings: 10 functions, ~$2/month, huge maintenance reduction**

---

## Migration Strategy

### Option A: Big Bang (Recommended)
1. Build unified_data_fetcher with all 10 fetchers
2. Deploy it
3. Test thoroughly
4. Delete old 10 functions
5. Update scheduler jobs

**Time:** 1-2 days

### Option B: Gradual
1. Start with unified_data_fetcher (empty)
2. Migrate one fetcher at a time
3. Test each migration
4. Delete old function when ready

**Time:** 1 week

### Option C: Leave MSA Only (Quick Win)
1. Keep only **msa_analyzer** (the one we just built)
2. Delete all other 12 functions
3. Agent still works (queries BigQuery)
4. Manually refresh BigQuery data when needed

**Time:** 1 hour

---

## What Does the Agent Actually Need?

The agent only needs **BigQuery tables with data**. It doesn't care how the data gets there!

**Current BigQuery tables (that agent queries):**
```
security_insights dataset:
  • security_findings
  • firewall_rules
  • compute_instances
  • storage_buckets
  • iam_bindings
  • custom_roles
  • gcp_release_notes
  • security_threat_feeds

security_data dataset:
  • msa_analysis_history (MSA results)
  • active_services (monitored services)
```

**Agent tools:**
- `run_query()` - Query ANY BigQuery table
- `list_datasets()` - See available datasets
- `list_tables()` - See available tables

**The agent is completely decoupled from Cloud Functions!**

---

## Recommendation

### Immediate Action (1 hour)
1. **Keep:** `msa_analyzer` (we just built this)
2. **Keep:** `confluence_sync` (if you use Confluence)
3. **Archive:** All 11 other functions → `cloud_functions/_archived/`
4. **Document:** Which BigQuery tables exist and how to refresh them manually

### Long-term (1-2 days when needed)
Build `unified_data_fetcher` when you need automated refreshes.

---

## Current Function Status

| Function | Purpose | Agent Uses? | Keep? |
|----------|---------|-------------|-------|
| fetch_compute_instances | VM data → BQ | No (queries BQ) | Consolidate |
| fetch_custom_roles | IAM roles → BQ | No (queries BQ) | Consolidate |
| fetch_firewall_rules | Firewall → BQ | No (queries BQ) | Consolidate |
| fetch_gcp_release_notes | RSS → BQ | No (queries BQ) | **Delete** (MSA does this) |
| fetch_iam_accounts | IAM → BQ | No (queries BQ) | Consolidate |
| fetch_security_feeds | CVE → BQ | No (queries BQ) | Consolidate |
| fetch_security_findings | SCC → BQ | No (queries BQ) | Consolidate |
| fetch_service_account_roles | SA roles → BQ | No (queries BQ) | Consolidate |
| fetch_standard_roles | Roles → BQ | No (queries BQ) | Consolidate |
| fetch_storage_buckets | GCS → BQ | No (queries BQ) | Consolidate |
| fetch_user_roles | User IAM → BQ | No (queries BQ) | Consolidate |
| **msa_analyzer** | **Release notes analysis** | **Yes!** | ✅ **KEEP** |
| confluence_sync | Confluence → BQ | No (queries BQ) | Keep if using Confluence |

---

## Decision Time

**What do you want to do?**

### Option 1: Clean Slate (Recommended)
- Keep **MSA only** (1 function)
- Archive other 12 functions
- Agent works perfectly (queries BigQuery)
- Build unified fetcher later if needed

### Option 2: Consolidate Now
- Build unified_data_fetcher (1-2 days)
- Keep MSA (1 function)
- Keep Confluence if needed (1 function)
- Delete old 10+ functions
- Total: 2-3 functions

### Option 3: Leave As-Is
- Keep all 13 functions
- Continue maintaining separately
- Not recommended

**Which option?**
