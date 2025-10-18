# Unified Cloud Functions with Vellox

## Overview

This unified Cloud Function consolidates all security data fetchers into a single deployable ASGI application using [Vellox](https://github.com/junah201/vellox) for Cloud Functions compatibility. This architecture provides:

- **Single deployment surface** - One Cloud Function instead of dozens
- **Shared utilities** - Reusable authentication, BigQuery, and response handling
- **Per-endpoint triggers** - Cloud Scheduler can still trigger individual fetchers
- **FastAPI documentation** - Auto-generated API docs at `/docs`
- **Improved maintainability** - Centralized configuration and error handling
- **Cost optimization** - Reduced cold starts and better resource utilization

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Unified Cloud Function                         │
│                        (Vellox + FastAPI)                        │
└─────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┴──────────────────────┐
         │                                            │
    Cloud Scheduler                              Direct HTTP
         │                                            │
         ▼                                            ▼
   /trigger/{fetcher}                          /fetch/{fetcher}
         │                                            │
         └──────────────────┬─────────────────────────┘
                           │
                    FastAPI Router
                           │
         ┌─────────────────┴──────────────────────┐
         │                                         │
    Fetcher Registry                        Shared Utilities
         │                                         │
    ┌────┴────┐                          ┌────────┴────────┐
    │ Fetchers│                          │ Auth │ BQ │ Resp│
    └─────────┘                          └─────────────────┘
         │                                         │
         └──────────────────┬──────────────────────┘
                           │
                      BigQuery Tables
```

## Features

### 🚀 Unified Deployment
- Single Cloud Function replaces multiple independent functions
- Vellox adapter provides seamless ASGI-to-Cloud Functions translation
- FastAPI handles routing and request validation

### 🔧 Modular Fetchers
Each fetcher is a self-contained module that:
- Extends `BaseFetcher` for consistent behavior
- Defines its own BigQuery schema
- Implements data fetching logic
- Handles errors gracefully with sample data fallback

### 🛠️ Shared Utilities
Reusable components for all fetchers:
- **Authentication** - Centralized GCP service client creation
- **BigQuery** - Dataset/table management and batch insertion
- **Configuration** - Environment-based settings with validation
- **Response** - Standardized response formatting

### 📊 Available Fetchers

| Fetcher | Description | BigQuery Table |
|---------|-------------|----------------|
| `security_findings` | Security Command Center findings | `security_insights.security_findings` |
| `custom_roles` | Custom IAM roles with risk analysis | `security_insights.iam_custom_roles` |
| `compute_instances` | Compute Engine VM instances | `security_insights.compute_instances` |
| `firewall_rules` | VPC firewall rules | `security_insights.firewall_rules` |
| `storage_buckets` | Cloud Storage bucket info | `security_insights.storage_buckets` |
| `iam_accounts` | IAM bindings at project level | `security_insights.iam_bindings` |
| `service_account_roles` | Service account role assignments | `security_insights.service_account_roles` |
| `standard_roles` | Predefined GCP IAM roles | `security_insights.iam_standard_roles` |
| `user_roles` | User IAM role assignments | `security_insights.user_roles` |

## Quick Start

### Prerequisites

1. Google Cloud Project with billing enabled
2. Required APIs enabled:
   - Cloud Functions
   - Cloud Build
   - Cloud Scheduler
   - BigQuery
   - IAM
   - Compute Engine
   - Cloud Storage
   - Security Command Center

3. Authentication configured:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

### Deploy the Unified Function

```bash
# Navigate to unified directory
cd cloud_functions/unified

# Deploy using the deployment script
./deploy.sh YOUR_PROJECT_ID us-central1

# Or deploy manually
gcloud functions deploy unified-security-fetcher \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=. \
    --entry-point=unified_handler \
    --trigger-http \
    --allow-unauthenticated \
    --memory=1024MB \
    --timeout=540s \
    --set-env-vars="PROJECT_ID=YOUR_PROJECT_ID,BQ_DATASET_ID=security_insights"
```

### Test the Deployment

```bash
# Get the function URL
FUNCTION_URL=$(gcloud functions describe unified-security-fetcher \
    --region=us-central1 \
    --format="value(serviceConfig.uri)")

# Test health endpoint
curl $FUNCTION_URL/health

# List available fetchers
curl $FUNCTION_URL/fetchers

# Trigger a specific fetcher
curl -X POST $FUNCTION_URL/fetch/security_findings

# View API documentation
open $FUNCTION_URL/docs
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and available endpoints |
| `/health` | GET | Health check with configuration validation |
| `/fetchers` | GET | List all available fetchers |
| `/docs` | GET | Interactive API documentation (FastAPI) |
| `/redoc` | GET | Alternative API documentation |

### Fetcher Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fetch/{fetcher_name}` | POST | Execute a specific fetcher |
| `/fetch/all` | POST | Execute all fetchers sequentially |
| `/trigger/{fetcher_name}` | GET | Cloud Scheduler trigger endpoint |

### Example Requests

```bash
# Execute specific fetcher
curl -X POST $FUNCTION_URL/fetch/custom_roles \
  -H "Content-Type: application/json" \
  -d '{"async_mode": false, "force_refresh": true}'

# Execute all fetchers
curl -X POST $FUNCTION_URL/fetch/all

# Cloud Scheduler trigger (GET request)
curl $FUNCTION_URL/trigger/security_findings
```

## Cloud Scheduler Integration

The deployment script automatically creates Cloud Scheduler jobs for each fetcher:

```bash
# List all scheduler jobs
gcloud scheduler jobs list --location=us-central1

# Run a job manually
gcloud scheduler jobs run security-findings-schedule --location=us-central1

# Update schedule
gcloud scheduler jobs update http custom-roles-schedule \
    --location=us-central1 \
    --schedule="0 */6 * * *"
```

### Default Schedules

| Fetcher | Schedule | Cron Expression |
|---------|----------|-----------------|
| security_findings | Every 2 hours | `0 */2 * * *` |
| custom_roles | Daily at 9 AM | `0 9 * * *` |
| compute_instances | Every 4 hours | `0 */4 * * *` |
| firewall_rules | Every 4 hours | `0 */4 * * *` |
| storage_buckets | Every 6 hours | `0 */6 * * *` |
| iam_accounts | Every 4 hours | `0 */4 * * *` |
| service_account_roles | Every 4 hours | `0 */4 * * *` |
| standard_roles | Weekly on Monday | `0 9 * * 1` |
| user_roles | Every 4 hours | `0 */4 * * *` |

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PROJECT_ID=your-project-id
export BQ_DATASET_ID=security_insights
export ENABLE_SAMPLE_DATA=true

# Run with uvicorn
uvicorn app.main:app --reload --port 8000

# Access local API
curl http://localhost:8000/health
curl http://localhost:8000/fetchers
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio fastapi[test]

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fetchers.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Adding a New Fetcher

1. Create fetcher module in `fetchers/`:
```python
# fetchers/new_fetcher.py
from typing import List, Dict, Any
from google.cloud import bigquery
from .base import BaseFetcher

class NewFetcher(BaseFetcher):
    @property
    def table_name(self) -> str:
        return 'new_table'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            # Add your schema fields
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        # Implement data fetching logic
        return []
```

2. Register in `fetchers/__init__.py`:
```python
from .new_fetcher import NewFetcher

FETCHERS_REGISTRY = {
    # ... existing fetchers ...
    'new_fetcher': NewFetcher
}
```

3. Add Cloud Scheduler job in `deploy.sh`:
```bash
create_scheduler_job "new_fetcher" "0 */4 * * *" "Description"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ID` | Google Cloud Project ID | Required |
| `BQ_DATASET_ID` | BigQuery dataset for tables | `security_insights` |
| `BQ_LOCATION` | BigQuery dataset location | `us-central1` |
| `ENABLE_SAMPLE_DATA` | Use sample data if fetch fails | `true` |
| `MAX_RETRIES` | Maximum retry attempts | `3` |
| `TIMEOUT_SECONDS` | Function timeout | `300` |

### BigQuery Datasets

The function uses two datasets:
- `security_insights` - Main dataset for security data
- `security_data` - Additional dataset for MSA analyzer

Tables are created automatically with appropriate schemas when first accessed.

## Monitoring

### View Logs

```bash
# Function logs
gcloud functions logs read unified-security-fetcher \
    --region=us-central1 \
    --limit=50

# Filter by severity
gcloud functions logs read unified-security-fetcher \
    --region=us-central1 \
    --filter="severity>=ERROR"

# Stream logs
gcloud functions logs tail unified-security-fetcher \
    --region=us-central1
```

### Metrics

```bash
# View metrics in Cloud Console
gcloud monitoring metrics-descriptors list \
    --filter="metric.type:cloudfunctions.googleapis.com"

# Function execution count
gcloud monitoring time-series list \
    --filter='metric.type="cloudfunctions.googleapis.com/function/execution_count"'
```

### BigQuery Usage

```sql
-- Check last update times
SELECT
  table_name,
  TIMESTAMP_MILLIS(last_modified_time) as last_updated
FROM `YOUR_PROJECT.security_insights.__TABLES__`
ORDER BY last_updated DESC;

-- Count records per table
SELECT
  table_name,
  row_count,
  size_bytes / (1024*1024) as size_mb
FROM `YOUR_PROJECT.security_insights.__TABLES__`;
```

## Cost Optimization

### Estimated Costs

| Component | Monthly Cost (Approximate) |
|-----------|----------------------------|
| Cloud Function (1GB, 540s timeout) | $5-15 |
| Cloud Scheduler (9 jobs) | $0.27 |
| BigQuery Storage (10GB) | $0.20 |
| BigQuery Queries (100GB scanned) | $5.00 |
| **Total** | **~$10-20/month** |

### Cost Reduction Tips

1. **Adjust schedules** - Reduce frequency for less critical data
2. **Optimize queries** - Use partitioning and clustering
3. **Set data retention** - Delete old data automatically
4. **Use regional resources** - Keep everything in same region
5. **Monitor usage** - Set up billing alerts

## Troubleshooting

### Common Issues

#### Function Deployment Fails
```bash
# Check Cloud Build logs
gcloud builds list --limit=5

# Verify APIs are enabled
gcloud services list --enabled
```

#### No Data in BigQuery
```bash
# Check function execution
gcloud functions logs read unified-security-fetcher \
    --region=us-central1 \
    --filter="fetch_data"

# Verify permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

#### Scheduler Jobs Not Triggering
```bash
# Check job status
gcloud scheduler jobs describe JOB_NAME --location=us-central1

# Run manually
gcloud scheduler jobs run JOB_NAME --location=us-central1
```

## Security Considerations

1. **Authentication** - Function allows unauthenticated access by default. Add IAM for production.
2. **Secrets** - Use Secret Manager for sensitive configuration
3. **Network** - Consider VPC Service Controls for enhanced security
4. **Monitoring** - Enable audit logging and alerts
5. **Data** - Implement data classification and retention policies

## Migration from Modular Functions

If migrating from individual Cloud Functions:

1. **Deploy unified function** alongside existing functions
2. **Update Cloud Scheduler** jobs to use new endpoints
3. **Monitor both** for a transition period
4. **Delete old functions** once verified

```bash
# List old functions
gcloud functions list --filter="name:fetch-"

# Delete old function
gcloud functions delete FUNCTION_NAME --region=REGION
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the ADK Security Agent and follows the same license terms.

## Support

For issues or questions:
1. Check this documentation
2. Review function logs
3. Open an issue in the repository
4. Contact the maintainers

---

**Note**: This unified architecture significantly simplifies deployment and maintenance compared to managing dozens of independent functions while maintaining the same functionality and flexibility for Cloud Scheduler integration.
