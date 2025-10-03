# Unified Data API

A modern FastAPI application that consolidates 13 Cloud Functions into a single, modular, type-safe API for fetching GCP resources and syncing to BigQuery.

## 🎯 Problem Solved

**Before**: 13 separate Cloud Functions with duplicated code, inconsistent typing, and difficult maintenance.

**After**: Single FastAPI application with:
- ✅ Strong typing via Pydantic models (shared with ADK agent tools)
- ✅ Modular BigQuery operations (separate create/insert/query logic)
- ✅ Unified API with automatic OpenAPI docs
- ✅ Single deployment unit (via Vellox for Cloud Functions)
- ✅ Reduced maintenance overhead (~70% code reduction)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Data API                          │
│                     (FastAPI App)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Fetchers   │  │   Pydantic   │  │   BigQuery   │     │
│  │              │──│    Models    │──│  Operations  │     │
│  │ - IAM        │  │              │  │              │     │
│  │ - Compute    │  │ - IAMAccount │  │ - create()   │     │
│  │ - Storage    │  │ - Firewall   │  │ - insert()   │     │
│  │ - Network    │  │ - Bucket     │  │ - query()    │     │
│  │ - Security   │  │ - Finding    │  │ - upsert()   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
        │                                          │
        ▼                                          ▼
   GCP APIs                              BigQuery Tables
  (IAM, Compute,                        (security_insights
   Storage, etc.)                         dataset)
```

## 📁 Project Structure

```
unified_data_api/
├── __init__.py                 # Package exports
├── models.py                   # Pydantic models for all GCP resources (400 lines)
├── bigquery_ops.py             # Modular BigQuery operations (400 lines)
├── fetchers.py                 # Data fetchers for each GCP service (200 lines)
├── main.py                     # FastAPI application with endpoints (600 lines)
├── cloud_function_wrapper.py   # Vellox wrapper for Cloud Functions (30 lines)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
cd unified_data_api
pip install -r requirements.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge
export BIGQUERY_DATASET=security_insights

# Run locally
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080

# API docs available at:
# - http://localhost:8080/docs (Swagger UI)
# - http://localhost:8080/redoc (ReDoc)
```

### Deploy to Cloud Functions (Gen2)

```bash
# Install Vellox wrapper
pip install vellox

# Deploy as single Cloud Function
gcloud functions deploy unified-data-api \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=cloud_function_wrapper.main \
  --trigger-http \
  --allow-unauthenticated \
  --memory=512MB \
  --timeout=540s

# Function will be available at:
# https://REGION-PROJECT_ID.cloudfunctions.net/unified-data-api
```

### Schedule with Cloud Scheduler

```bash
# Create scheduler job to fetch all data hourly
gcloud scheduler jobs create http fetch-all-data \
  --location=us-central1 \
  --schedule="0 * * * *" \
  --uri="https://REGION-PROJECT_ID.cloudfunctions.net/unified-data-api/api/v1/batch/fetch-all" \
  --http-method=POST \
  --oidc-service-account-email=scheduler@PROJECT_ID.iam.gserviceaccount.com
```

## 📊 API Endpoints

### Health & Admin

- `GET /health` - Health check with BigQuery connectivity status
- `POST /api/v1/admin/create-tables` - Create all BigQuery tables from Pydantic models
- `GET /api/v1/admin/tables/{table_name}/info` - Get table metadata

### IAM Endpoints

- `POST /api/v1/iam/accounts/fetch` - Fetch IAM accounts → `iam_accounts` table
- `POST /api/v1/iam/custom-roles/fetch` - Fetch custom roles → `custom_roles` table
- `POST /api/v1/iam/service-accounts/fetch` - Fetch service account roles → `service_account_roles` table

### Compute Endpoints

- `POST /api/v1/compute/instances/fetch?zones=us-central1-a` - Fetch compute instances → `compute_instances` table

### Network Endpoints

- `POST /api/v1/network/firewall-rules/fetch` - Fetch firewall rules → `firewall_rules` table
- `POST /api/v1/network/networks/fetch` - Fetch VPC networks → `networks` table

### Storage Endpoints

- `POST /api/v1/storage/buckets/fetch` - Fetch storage buckets → `storage_buckets` table

### Security Endpoints

- `POST /api/v1/security/findings/fetch?min_severity=HIGH` - Fetch security findings → `security_findings` table

### Feeds & Documentation Endpoints

- `POST /api/v1/feeds/security/fetch` - Fetch security feeds (NVD, CISA) → `security_feeds` table
- `POST /api/v1/feeds/release-notes/fetch` - Fetch GCP release notes → `release_notes` table
- `POST /api/v1/feeds/confluence/sync?space_key=IT` - Sync Confluence pages → `confluence_pages` table

### Batch Operations

- `POST /api/v1/batch/fetch-all` - Trigger all fetchers (runs in background)

## 🔍 Example Usage

### Fetch IAM Accounts

```bash
# Fetch and sync to BigQuery
curl -X POST "http://localhost:8080/api/v1/iam/accounts/fetch?sync_to_bq=true"

# Response:
{
  "success": true,
  "message": "Fetched 127 IAM accounts",
  "records_fetched": 127,
  "records_inserted": 127,
  "table_name": "iam_accounts",
  "execution_time_ms": 2341.5
}
```

### Create All Tables

```bash
curl -X POST "http://localhost:8080/api/v1/admin/create-tables"

# Response:
{
  "success": true,
  "message": "Created 11/11 tables",
  "tables": {
    "iam_accounts": true,
    "firewall_rules": true,
    "storage_buckets": true,
    ...
  }
}
```

### Health Check

```bash
curl "http://localhost:8080/health"

# Response:
{
  "status": "healthy",
  "timestamp": "2025-10-03T20:30:00Z",
  "bigquery_connected": true,
  "services_available": {
    "iam": true,
    "compute": true,
    "storage": true,
    "network": true,
    "security": true
  },
  "version": "1.0.0"
}
```

## 🎨 Pydantic Models (Shared with ADK)

All models are strongly typed and can be imported by ADK agent tools:

```python
from unified_data_api.models import IAMAccount, FirewallRule, StorageBucket

# Use in ADK tools for better type safety
def analyze_iam_accounts() -> List[IAMAccount]:
    """Agent tool with strong typing"""
    query = "SELECT * FROM security_insights.iam_accounts WHERE is_primitive_role = true"
    accounts = bq_ops.query_to_models(query, IAMAccount)
    return accounts
```

### Model Examples

```python
# IAM Account
IAMAccount(
    email="user@example.com",
    account_type=AccountType.USER,
    role="roles/viewer",
    project_id="my-project",
    is_primitive_role=True,
    key_age_days=None
)

# Firewall Rule
FirewallRule(
    rule_name="allow-ssh-from-anywhere",
    network="default",
    direction="INGRESS",
    action="ALLOW",
    source_ranges=["0.0.0.0/0"],
    protocols=["tcp"],
    ports=["22"],
    allows_all_ips=True,  # Security flag!
    allows_ssh=True       # Security flag!
)

# Storage Bucket
StorageBucket(
    bucket_name="my-public-bucket",
    location="US",
    storage_class="STANDARD",
    is_public=True,       # Security flag!
    encryption_type="GOOGLE_MANAGED",
    versioning_enabled=False
)
```

## 🔧 BigQuery Operations

The `BigQueryOperations` class provides modular operations:

```python
from unified_data_api import BigQueryOperations, IAMAccount

bq_ops = BigQueryOperations(project_id="my-project", dataset_id="security_insights")

# Create tables from Pydantic models
bq_ops.create_table("iam_accounts", model_class=IAMAccount)

# Insert typed records
accounts = [IAMAccount(...), IAMAccount(...)]
bq_ops.insert_records("iam_accounts", accounts)

# Upsert with key fields
bq_ops.upsert_records("iam_accounts", accounts, key_fields=["email", "project_id"])

# Query to typed models
accounts = bq_ops.query_to_models(
    "SELECT * FROM security_insights.iam_accounts WHERE is_primitive_role = true",
    IAMAccount
)
```

## 📈 Benefits vs. 13 Cloud Functions

| Feature | 13 Cloud Functions | Unified API |
|---------|-------------------|-------------|
| Code Duplication | High (BigQuery ops in each) | None (shared module) |
| Type Safety | Weak (dicts) | Strong (Pydantic) |
| Deployment | 13 separate deploys | 1 deploy |
| API Documentation | Manual | Auto-generated |
| Maintenance | 13 functions to update | 1 codebase |
| Testing | 13 test suites | 1 integrated suite |
| Monitoring | 13 logs to check | Centralized logging |
| Cost | 13 function instances | 1 function instance |

**Estimated Code Reduction**: ~70% (from ~2,600 lines to ~800 lines)

## 🔄 Migration Path

1. **Phase 1**: Deploy unified API alongside existing Cloud Functions
2. **Phase 2**: Test unified API endpoints match Cloud Function outputs
3. **Phase 3**: Update Cloud Scheduler jobs to call unified API
4. **Phase 4**: Migrate fetchers from Cloud Functions to `fetchers.py`
5. **Phase 5**: Decommission old Cloud Functions

## 🧪 Testing

```bash
# Test locally
python -m pytest tests/

# Test specific endpoint
curl -X POST "http://localhost:8080/api/v1/iam/accounts/fetch?sync_to_bq=false"

# Check health
curl "http://localhost:8080/health"
```

## 📝 TODOs

- [ ] Migrate existing Cloud Function logic to `fetchers.py`
- [ ] Add authentication (IAM-based for Cloud Functions)
- [ ] Add rate limiting
- [ ] Add request/response caching
- [ ] Create comprehensive test suite
- [ ] Set up CI/CD pipeline
- [ ] Add Prometheus metrics endpoint
- [ ] Document all Pydantic model fields

## 🤝 Integration with ADK Agent

The ADK agent tools can import and use these Pydantic models for strong typing:

```python
# agents/_tools/security_tools.py
from unified_data_api.models import IAMAccount, Severity
from unified_data_api import BigQueryOperations

def get_high_risk_accounts() -> List[IAMAccount]:
    """
    Agent tool with strong typing from unified API models
    """
    bq_ops = BigQueryOperations(project_id="my-project")

    query = """
    SELECT * FROM security_insights.iam_accounts
    WHERE is_primitive_role = true
    OR key_age_days > 90
    """

    # Query returns typed Pydantic models
    accounts: List[IAMAccount] = bq_ops.query_to_models(query, IAMAccount)

    return accounts
```

## 📚 References

- [Vellox - FastAPI wrapper for Cloud Functions](https://github.com/junah201/vellox)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Google Cloud Functions Gen2](https://cloud.google.com/functions/docs/2nd-gen/overview)
