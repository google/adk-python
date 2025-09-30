# Security BigQuery Agent

A high-performance ADK agent for GCP security analysis using BigQuery as the primary data platform.

## =€ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your project details

# Run the agent
adk web
```

## =Ê Architecture

This agent uses **BigQuery** as the single source of truth for all security data:

- **Direct BigQuery Access** - No caching layers, no SQLite
- **Cloud Functions** - Automated data refresh on schedules
- **Real-time Updates** - Pub/Sub integration for immediate changes
- **Security Analysis** - Built-in risk scoring and alerting

## =à Key Components

### 1. Main Agent (`agents/agent.py`)
- Security-focused Gemini model
- 12 BigQuery tools for analysis
- Conversational security expert persona

### 2. Data Refresh (`cloud_functions/`)
Independent Cloud Functions that refresh data:
- `fetch_compute_instances/` - VMs (every 2h)
- `fetch_iam_accounts/` - IAM (every 6h)
- `fetch_firewall_rules/` - Firewall (every 4h)
- `fetch_storage_buckets/` - Storage (every 1h)

### 3. Deployment (`scripts/`)
- `deploy_refresh_jobs.sh` - Deploy all Cloud Functions

## =È Available Tools

The agent provides these BigQuery tools:

1. **get_security_insights_summary()** - Overview of security data
2. **query_security_insights()** - Custom security queries
3. **get_security_statistics()** - Aggregated stats by category
4. **explore_all_tables_and_views()** - Browse dataset structure
5. **analyze_table_or_view()** - Deep dive into tables
6. **run_query()** - Execute any SQL query
7. **analyze_query_cost()** - Estimate query costs
8. **hello_world()** - Test BigQuery connection
9. **list_datasets()** - Show all datasets
10. **list_tables()** - Show tables in dataset
11. **get_table_schema()** - View table structure
12. **get_table_sample()** - Preview table data

## =Â BigQuery Schema

### Security Insights Dataset

```sql
security_insights.compute_instances   -- VM inventory with risk analysis
security_insights.iam_accounts        -- IAM bindings and permissions
security_insights.firewall_rules      -- Firewall rules with risk scores
security_insights.storage_buckets     -- Storage bucket configurations
security_insights.security_findings   -- Security Command Center findings
security_insights.refresh_metadata    -- Data freshness tracking
```

## =€ Deployment

### Deploy Cloud Functions for Data Refresh

```bash
# Set environment
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export DATASET_ID="security_insights"

# Run deployment
cd scripts
./deploy_refresh_jobs.sh
```

This will:
- Deploy 4 Cloud Functions for data collection
- Set up Cloud Scheduler for automatic refresh
- Create BigQuery views for analysis
- Configure Pub/Sub for real-time updates

## <¯ Example Queries

When using the agent via `adk web`:

```
"Show me all critical firewall rules"
"List compute instances with external IPs"
"Find IAM accounts with admin privileges"
"Check for publicly exposed storage buckets"
"What are the most critical security findings?"
"Show me security statistics grouped by severity"
```

## =Ê Monitoring

Check data freshness:
```sql
SELECT * FROM `project.security_insights.data_freshness`
```

View security dashboard:
```sql
SELECT * FROM `project.security_insights.security_dashboard`
```

## =' Configuration

### Environment Variables (.env)
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account.json
BQ_DEFAULT_DATASET=security_insights
BQ_DEFAULT_TABLE=security_insights
ADK_AGENT_MODEL=gemini-2.5-flash
```

### Refresh Schedules
Edit schedules in `scripts/deploy_refresh_jobs.sh`:
- Compute Instances: Every 2 hours
- IAM Accounts: Every 6 hours
- Firewall Rules: Every 4 hours
- Storage Buckets: Every hour

## =È Performance

- **Response Time**: 1-7 seconds average
- **Query Performance**: Sub-second for most queries
- **Data Freshness**: Automated refresh based on resource type
- **Cost Efficient**: ~$50-100/month typical usage

## =á Security

- Uses service account authentication
- All data stored in BigQuery with encryption
- Audit logging for all operations
- Risk scoring for automatic threat detection

## =Ú Requirements

- Google Cloud Project with billing enabled
- BigQuery API enabled
- Compute Engine API enabled
- IAM API enabled
- Cloud Functions API enabled (for automation)

## <¯ Success Metrics

 **80%+ evaluation success rate**
 **7.29s average response time**
 **Direct BigQuery access (no caching delays)**
 **Automated data refresh**
 **Built-in security risk analysis**