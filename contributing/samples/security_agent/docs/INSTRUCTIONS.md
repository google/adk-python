# GCP Security Agent - Developer Instructions

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Development Setup](#development-setup)
4. [Running Locally](#running-locally)
5. [Adding New Tools](#adding-new-tools)
6. [Deploying Cloud Functions](#deploying-cloud-functions)
7. [Testing Guidelines](#testing-guidelines)
8. [Common Development Tasks](#common-development-tasks)
9. [Troubleshooting](#troubleshooting)
10. [Code Structure](#code-structure)

---

## Project Overview

The GCP Security Intelligence Platform is a comprehensive security monitoring and analysis system for Google Cloud Platform. It combines:

- **ADK-Powered AI Agent** - Natural language interface with 32 specialized tools
- **Automated Data Collection** - 13 Cloud Functions for scheduled data ingestion
- **BigQuery Data Platform** - Centralized security intelligence storage
- **Flask Web UI** - Interactive dashboard and chat interface
- **Confluence Integration** - Documentation and policy synchronization

### Key Statistics
- **84% Code Reduction**: 19,402 lines added / 53,520 lines removed
- **32 Tools**: Comprehensive security analysis capabilities
- **13 Cloud Functions**: Modular data collection architecture
- **2 BigQuery Datasets**: `security_insights` and `security_data`

### Primary Use Cases
1. **Service Onboarding** - Automated security assessment for new GCP services
2. **IAM Analysis** - Custom role analysis and permission auditing
3. **Release Notes Monitoring** - Track GCP changes for security/billing/compliance impacts
4. **Security Findings** - Centralized view of all security issues
5. **Compliance Tracking** - Automated compliance validation against security controls

---

## Architecture Overview

### System Components

```
┌──────────────────────────────────────────────────────────────┐
│                          User                                 │
└────────────┬─────────────────────────────────────────────────┘
             │
    ┌────────▼────────┐
    │  Flask Web UI   │  (Web UI on port 5001)
    │    (app.py)     │  - Chat interface
    └────────┬────────┘  - Dashboard & metrics
             │ HTTP
    ┌────────▼────────────────────────────────────────────────┐
    │              ADK Backend (port 8000)                     │
    │    ADK Security Agent - 32 Tools                        │
    │    (Gemini 2.5 Flash - Natural Language Interface)      │
    └────────┬──────────────────────────┬─────────────────────┘
             │                          │
    ┌────────▼────────┐        ┌───────▼─────────┐
    │  BigQuery Tools │        │ Service Tools   │
    │  - Analysis     │        │ - Discovery     │
    │  - Queries      │        │ - Evaluation    │
    │  - Exploration  │        │ - Onboarding    │
    │  - 12 tools     │        │ - 20 tools      │
    └────────┬────────┘        └────────┬────────┘
             │                          │
    ┌────────▼──────────────────────────▼────────┐
    │        BigQuery Data Platform               │
    │      (Single Source of Truth)               │
    │   - security_insights (primary)             │
    │   - security_data (MSA results)             │
    └────────────────────┬───────────────────────┘
                         │
    ┌────────────────────▼───────────────────────┐
    │      Cloud Functions (13) - Modular        │
    │   Customer Chooses Which to Deploy         │
    │                                             │
    │  🔒 IAM & Security (7 functions)           │
    │  ☁️ Infrastructure (2 functions)            │
    │  📰 Feeds & Docs (3 functions)              │
    │  🎯 Analysis (1 function - MSA)             │
    └────────────────────┬───────────────────────┘
                         │
    ┌────────────────────▼───────────────────────┐
    │       External Data Sources                 │
    │  - GCP APIs & Services                     │
    │  - Confluence Documentation                │
    │  - RSS Security Feeds                       │
    │  - GCP Release Notes                       │
    └─────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Separation of Concerns**
   - Agent queries BigQuery (never calls Cloud Functions directly)
   - Cloud Functions populate data (never queried by agent)
   - UI calls ADK backend API (never runs agent locally)

2. **Modular Deployment**
   - Deploy only the Cloud Functions you need
   - Each function is independent and self-contained
   - Scheduled via Cloud Scheduler

3. **Direct Access**
   - Agent has full BigQuery access via `run_query()` tool
   - No intermediate API layers for data access
   - Real-time query capabilities

4. **No Coupling**
   - Frontend → Backend API only
   - Backend → BigQuery only
   - Cloud Functions → BigQuery only
   - No circular dependencies

---

## Development Setup

### Prerequisites

```bash
# Required tools
- Python 3.11+
- Google Cloud SDK (gcloud CLI)
- Node.js 18+ (for ADK installation)
- Git

# Required GCP APIs
gcloud services enable compute.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable securitycenter.googleapis.com
```

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/stuagano/adk-python.git
cd adk-python/contributing/samples/security_agent

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install ADK globally
npm install -g @google/adk

# 4. Install required packages in ADK environment
# (Required for web scraping and RSS parsing)
~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# 5. Configure environment
cp .env.example .env
# Edit .env with your GCP project details

# 6. Set up service account credentials
# Place service account JSON in config/service-account.json
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/config/service-account.json"
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account.json
GOOGLE_CLOUD_LOCATION=us-central1

# BigQuery Configuration
BQ_DEFAULT_DATASET=security_insights
BQ_DEFAULT_TABLE=security_insights

# ADK Configuration
ADK_AGENT_MODEL=gemini-2.5-flash
GOOGLE_GENAI_USE_VERTEXAI=1
ADK_BASE_URL=http://localhost:8000

# Confluence Configuration (Optional)
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACES=SEC,POLICY,GCP
CONFLUENCE_CACHE_DB=backend/cache/confluence_cache.db
CONFLUENCE_CACHE_TTL_HOURS=6

# Flask Configuration
FLASK_PORT=5001
STREAM_CHUNK_SIZE=200
```

---

## Running Locally

### Two-Process Setup

The application requires **two terminal windows** running simultaneously:

#### Terminal 1: ADK Backend (Port 8000)

```bash
# Start the ADK backend with the agent and all tools
adk web

# You should see:
# ✓ Server started on http://localhost:8000
# ✓ Agent: security_bigquery_agent
# ✓ Model: gemini-2.5-flash
# ✓ Tools: 32 tools loaded
```

**What this does:**
- Starts the ADK web server on port 8000
- Loads the security agent with 32 tools
- Provides HTTP API endpoints for agent interaction
- Handles all natural language processing and tool execution

#### Terminal 2: Flask Web UI (Port 5001)

```bash
# Start the Flask web UI
python3 app.py --port=5001

# You should see:
# 🚀 Starting Flask app for BigQuery Security Agent
#    Agent: security_bigquery_agent
#    Model: gemini-2.5-flash
#    Tools: 32 tools available
#
# 📍 Server running at: http://localhost:5001
#    Health check: http://localhost:5001/health
#    Agent info: http://localhost:5001/agent-info
```

**Why port 5001?**
- Port 5000 conflicts with macOS AirPlay Receiver
- Port 5001 is the default to avoid this conflict

**What this does:**
- Provides web interface for chat and dashboard
- Calls ADK backend API (port 8000) for agent interactions
- Displays metrics and visualizations
- Handles SSE streaming for real-time responses

### Verify Setup

```bash
# Check ADK backend health
curl http://localhost:8000/health

# Check Flask UI health
curl http://localhost:5001/health

# Open web interface
open http://localhost:5001
```

### Quick Test Queries

Once both servers are running, try these queries in the web UI:

```
"Show me all security findings"
"What GCP services are enabled?"
"Analyze Cloud Storage security"
"What are the latest release notes?"
"Search Confluence for security policies"
```

---

## Adding New Tools

### Tool Development Process

1. **Create Tool Function** in `agents/_tools/`
2. **Add Type Annotations** (required by ADK)
3. **Write Docstring** (becomes tool description)
4. **Register in agent.py**
5. **Test Locally**
6. **Document in TOOLS.md**

### Example: Creating a New Tool

```python
# File: agents/_tools/my_new_tool.py

from typing import Optional
from .base import get_bq_client, PROJECT_ID, DEFAULT_DATASET

def analyze_networking_security(
    min_risk_score: int = 50,
    include_recommendations: bool = True
) -> str:
    """
    Analyze network security configurations for GCP resources.

    This tool examines firewall rules, VPC settings, and network policies
    to identify potential security risks in network configurations.

    Args:
        min_risk_score: Minimum risk score to include (0-100). Default: 50
        include_recommendations: Include remediation recommendations. Default: True

    Returns:
        Formatted analysis with findings and recommendations
    """
    client = get_bq_client()

    # Query BigQuery for network security data
    query = f"""
        SELECT
            firewall_rule_name,
            direction,
            source_ranges,
            allowed_ports,
            risk_score,
            risk_factors
        FROM `{PROJECT_ID}.{DEFAULT_DATASET}.firewall_rules`
        WHERE risk_score >= {min_risk_score}
        ORDER BY risk_score DESC
        LIMIT 100
    """

    results = client.query(query).result()

    # Format results
    output = [f"Network Security Analysis (Risk Score >= {min_risk_score})\n"]
    output.append("=" * 70 + "\n")

    for row in results:
        output.append(f"\n🔥 {row.firewall_rule_name}")
        output.append(f"   Direction: {row.direction}")
        output.append(f"   Risk Score: {row.risk_score}/100")
        output.append(f"   Source Ranges: {', '.join(row.source_ranges)}")

        if include_recommendations and row.risk_factors:
            output.append(f"   ⚠️  Risk Factors: {', '.join(row.risk_factors)}")

    return "\n".join(output)
```

### Register the Tool

Edit `agents/agent.py`:

```python
# Import your new tool
from ._tools.my_new_tool import analyze_networking_security

# Add to tools list
tools = [
    # ... existing tools ...
    FunctionTool(analyze_networking_security),
]
```

### Type Annotation Requirements

ADK requires proper type annotations for all tool parameters:

```python
# ✅ CORRECT
def my_tool(
    param1: str,                    # Required parameter
    param2: int = 10,               # Optional with default
    param3: Optional[bool] = None   # Explicitly optional
) -> str:
    pass

# ❌ WRONG
def my_tool(param1, param2=10):     # No type annotations
    pass
```

### Testing Your Tool

```bash
# 1. Restart ADK backend
# Press Ctrl+C in Terminal 1, then:
adk web

# 2. Test via web UI or Python
python3 -c "
from agents.agent import root_agent
from agents._tools.my_new_tool import analyze_networking_security

# Direct test
result = analyze_networking_security(min_risk_score=75)
print(result)
"
```

---

## Deploying Cloud Functions

### Overview of Cloud Functions

The platform includes 13 independent Cloud Functions for automated data collection:

| Category | Functions | Purpose |
|----------|-----------|---------|
| IAM & Security | 7 functions | User roles, service accounts, custom roles |
| Infrastructure | 2 functions | Compute instances, storage buckets |
| Feeds & Docs | 3 functions | RSS feeds, release notes, Confluence sync |
| Analysis | 1 function | Multi-Service Analyzer (MSA) |

### Deployment Methods

#### Option 1: Deploy All Functions

```bash
cd cloud_functions
./deploy_selected.sh

# Interactive menu will appear:
# Select which functions to deploy
# Choose schedules (hourly, daily, etc.)
# Set environment variables
```

#### Option 2: Deploy Individual Function

```bash
cd cloud_functions/fetch_iam_accounts

# Deploy function
gcloud functions deploy fetch-iam-accounts \
  --runtime python311 \
  --trigger-http \
  --entry-point fetch_iam_accounts \
  --memory 512MB \
  --timeout 540s \
  --set-env-vars "PROJECT_ID=your-project,DATASET_ID=security_insights"

# Create schedule (every 6 hours)
gcloud scheduler jobs create http fetch-iam-accounts-schedule \
  --schedule="0 */6 * * *" \
  --uri="https://REGION-PROJECT.cloudfunctions.net/fetch-iam-accounts" \
  --http-method=POST \
  --message-body='{"sync_type":"incremental"}'
```

#### Option 3: Deploy via Cloud Build

```bash
# Deploy using cloudbuild.yaml
cd cloud_functions
gcloud builds submit --config=cloudbuild.yaml
```

### Recommended Schedules

```bash
# Critical security data (frequent updates)
Security Findings:     */30 * * * *  # Every 30 minutes
Storage Buckets:       0 * * * *     # Every hour
Compute Instances:     0 */2 * * *   # Every 2 hours

# IAM data (moderate updates)
IAM Accounts:          0 */6 * * *   # Every 6 hours
Service Account Roles: 0 */4 * * *   # Every 4 hours
User Roles:            0 */6 * * *   # Every 6 hours

# Configuration data (infrequent changes)
Custom Roles:          0 0 * * *     # Daily
Standard Roles:        0 0 * * 0     # Weekly
Firewall Rules:        0 */4 * * *   # Every 4 hours

# External feeds
RSS Security Feeds:    0 */2 * * *   # Every 2 hours
GCP Release Notes:     0 */6 * * *   # Every 6 hours
Confluence Sync:       0 2 * * *     # Daily at 2 AM
```

### Environment Variables for Cloud Functions

Each function requires these environment variables:

```bash
PROJECT_ID=your-gcp-project-id
DATASET_ID=security_insights
GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# For Confluence sync only:
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACES=SEC,POLICY,GCP
```

### Testing Cloud Functions Locally

```bash
cd cloud_functions/fetch_iam_accounts

# Install dependencies
pip install -r requirements.txt

# Test function
python main.py --test

# Or use Functions Framework
functions-framework --target=fetch_iam_accounts --debug
```

### Monitoring Deployments

```bash
# View logs
gcloud functions logs read fetch-iam-accounts --limit 50

# Check schedules
gcloud scheduler jobs list

# View function details
gcloud functions describe fetch-iam-accounts
```

---

## Testing Guidelines

### Unit Testing

```bash
# Run all tests
cd cloud_functions/tests
python -m pytest unit/ -v

# Run specific test file
python -m pytest unit/test_iam_tools.py -v

# Run with coverage
python -m pytest unit/ --cov=agents --cov-report=html
```

### Integration Testing

```bash
# Test BigQuery integration
cd cloud_functions/tests
python -m pytest integration/test_bigquery.py -v

# Test Confluence integration
python tests/test_confluence_integration.py

# Test ADK agent
python test_adk_query.py
```

### Performance Testing

```bash
# Run performance benchmarks
cd cloud_functions/tests
python -m pytest performance/ -v

# Test specific scenarios
python performance/test_query_performance.py
```

### Manual Testing Checklist

- [ ] ADK backend starts without errors
- [ ] Flask UI loads successfully
- [ ] Chat interface responds to queries
- [ ] BigQuery tools return data
- [ ] Service discovery works
- [ ] Confluence search returns results (if configured)
- [ ] Dashboard metrics display correctly
- [ ] Health endpoints return 200 OK

### Testing New Tools

```python
# File: tests/test_my_new_tool.py

import pytest
from agents._tools.my_new_tool import analyze_networking_security

def test_analyze_networking_security():
    """Test basic functionality."""
    result = analyze_networking_security(min_risk_score=50)
    assert result is not None
    assert "Network Security Analysis" in result

def test_analyze_networking_security_high_threshold():
    """Test with high risk threshold."""
    result = analyze_networking_security(min_risk_score=90)
    assert result is not None
    # Should return fewer results with higher threshold

def test_analyze_networking_security_no_recommendations():
    """Test without recommendations."""
    result = analyze_networking_security(
        min_risk_score=50,
        include_recommendations=False
    )
    assert "Risk Factors" not in result
```

---

## Common Development Tasks

### Task 1: Update Agent Instructions

```bash
# Edit the instruction file
vi docs/agent_instructions.md

# The agent automatically reloads instructions on startup
# Restart ADK backend to apply changes
# (Ctrl+C in Terminal 1, then: adk web)
```

### Task 2: Add New BigQuery Table

```python
# 1. Create Cloud Function to populate table
# File: cloud_functions/fetch_my_data/main.py

def fetch_my_data(request):
    from google.cloud import bigquery

    client = bigquery.Client()
    table_id = f"{PROJECT_ID}.security_insights.my_new_table"

    # Define schema
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]

    # Create or update table
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)

    # Fetch and insert data
    rows_to_insert = [
        {"id": "1", "name": "Example", "created_at": "2025-01-01T00:00:00"}
    ]

    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        raise Exception(f"Errors: {errors}")

    return {"status": "success", "rows_inserted": len(rows_to_insert)}

# 2. Deploy the function (see Deploying Cloud Functions section)

# 3. Create tool to query the table (see Adding New Tools section)
```

### Task 3: Update Dashboard Metrics

```python
# Edit app.py to add new metric endpoint

@app.route("/api/my-new-metric")
def get_my_new_metric():
    """Return new metric data."""
    try:
        # Query BigQuery for data
        from agents._tools.bigquery_tools import run_query

        result = run_query("""
            SELECT
                COUNT(*) as total,
                AVG(risk_score) as avg_risk
            FROM `security_insights.my_new_table`
        """)

        return jsonify({
            "success": True,
            "data": {
                "total": result[0]["total"],
                "average_risk": result[0]["avg_risk"]
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

### Task 4: Add New Service to Discovery

```python
# Use the learn_service_from_url tool
from agents._tools.service_discovery import learn_service_from_url

# Learn about a new service from documentation
result = learn_service_from_url(
    documentation_url="https://cloud.google.com/new-service/docs"
)

# The service is automatically added to the catalog
# and can be analyzed with analyze_gcp_service()
```

### Task 5: Create Custom Analysis Query

```python
# Use the run_query tool directly in chat UI:
"Run this query:
SELECT
    service_name,
    COUNT(*) as issue_count,
    AVG(risk_score) as avg_risk
FROM security_insights.security_findings
WHERE severity = 'HIGH'
GROUP BY service_name
ORDER BY issue_count DESC
LIMIT 10"
```

---

## Troubleshooting

### Problem: ADK Backend Won't Start

**Symptoms:**
```
Error: Module not found: beautifulsoup4
```

**Solution:**
```bash
# Install missing packages in ADK environment
~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# Restart ADK
adk web
```

### Problem: Flask UI Can't Connect to Backend

**Symptoms:**
```
Error communicating with ADK backend
Connection refused to localhost:8000
```

**Solution:**
```bash
# 1. Verify ADK backend is running
curl http://localhost:8000/health

# 2. Check ADK_BASE_URL in .env
echo $ADK_BASE_URL  # Should be http://localhost:8000

# 3. Restart both services
# Terminal 1: adk web
# Terminal 2: python3 app.py --port=5001
```

### Problem: BigQuery Permission Denied

**Symptoms:**
```
403 Forbidden: Access Denied
```

**Solution:**
```bash
# Verify service account has required roles
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/bigquery.jobUser"

# Check credentials are set
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### Problem: Cloud Function Deployment Fails

**Symptoms:**
```
ERROR: Build failed
```

**Solution:**
```bash
# Check requirements.txt includes all dependencies
cd cloud_functions/my_function
cat requirements.txt

# Verify function name matches entry-point
# In main.py:
# def my_function_name(request):  # Must match --entry-point

# Check for syntax errors
python -m py_compile main.py

# Deploy with verbose logging
gcloud functions deploy my-function \
  --runtime python311 \
  --trigger-http \
  --entry-point my_function_name \
  --verbosity=debug
```

### Problem: Confluence Tools Not Working

**Symptoms:**
```
Error: Confluence API authentication failed
```

**Solution:**
```bash
# Verify credentials
echo $CONFLUENCE_URL
echo $CONFLUENCE_USERNAME
# Don't echo API token (security)

# Test connection
curl -u "$CONFLUENCE_USERNAME:$CONFLUENCE_API_TOKEN" \
  "$CONFLUENCE_URL/rest/api/content?limit=1"

# Refresh cache
python3 -c "
from agents._tools.confluence_tools import refresh_confluence_cache
refresh_confluence_cache(force=True)
"
```

### Problem: Port 5001 Already in Use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find and kill process using port 5001
lsof -ti:5001 | xargs kill -9

# Or use a different port
python3 app.py --port=5002
```

### Problem: Agent Gives Empty Responses

**Symptoms:**
- Agent responds with "I don't have that data"
- Tools return no results

**Solution:**
```bash
# 1. Check if Cloud Functions have run
# Query BigQuery for data freshness
python3 -c "
from google.cloud import bigquery
client = bigquery.Client()
query = '''
SELECT
  table_name,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), last_refresh, MINUTE) as minutes_old
FROM \`security_insights.refresh_metadata\`
ORDER BY minutes_old DESC
'''
for row in client.query(query):
    print(f'{row.table_name}: {row.minutes_old} minutes old')
"

# 2. Manually trigger Cloud Function
gcloud functions call fetch-iam-accounts --data '{}'

# 3. Check agent has access to tools
curl http://localhost:8000/apps/agents/users/web-user/tools
```

---

## Code Structure

### Directory Layout

```
security_agent/
├── agents/                      # ADK agent configuration
│   ├── agent.py                # Main agent definition with tool registration
│   └── _tools/                 # Tool implementations
│       ├── __init__.py         # Tool exports
│       ├── base.py             # Shared utilities and config
│       ├── bigquery_tools.py   # Standard BigQuery operations (7 tools)
│       ├── security_tools.py   # Security-focused queries (3 tools)
│       ├── exploration_tools.py # Dataset exploration (2 tools)
│       ├── feed_tools.py       # RSS feeds and release notes (4 tools)
│       ├── confluence_tools.py # Confluence integration (5 tools)
│       ├── service_discovery.py # GCP service discovery (8 tools)
│       ├── msa_analyzer.py     # Multi-Service Analyzer (1 tool)
│       └── service_evaluation/ # Service evaluation framework (2 tools)
│           ├── evaluator.py    # Service security assessment
│           └── compliance_checker.py # Compliance validation
│
├── app.py                      # Flask web UI application
├── templates/                  # HTML templates
│   └── index.html             # Main dashboard
├── static/                     # CSS, JavaScript, images
│
├── cloud_functions/            # 13 Cloud Functions for data collection
│   ├── fetch_iam_accounts/    # IAM accounts and bindings
│   ├── fetch_service_account_roles/ # Service account permissions
│   ├── fetch_user_roles/      # User role assignments
│   ├── fetch_custom_roles/    # Custom IAM roles
│   ├── fetch_standard_roles/  # Google-managed roles
│   ├── fetch_compute_instances/ # VM inventory
│   ├── fetch_firewall_rules/  # Network security rules
│   ├── fetch_storage_buckets/ # Storage bucket configs
│   ├── fetch_security_findings/ # Security Command Center
│   ├── fetch_security_feeds/  # RSS threat feeds
│   ├── fetch_gcp_release_notes/ # GCP platform updates
│   ├── confluence_sync/       # Documentation sync
│   ├── msa_analyzer/          # Release notes impact analysis
│   ├── deploy_selected.sh     # Interactive deployment script
│   └── tests/                 # Cloud function tests
│
├── unified_data_api/          # Unified data access layer
│   ├── main.py               # FastAPI application
│   ├── bigquery_ops.py       # BigQuery operations
│   ├── fetchers.py           # Data fetchers
│   └── models.py             # Pydantic models
│
├── docs/                      # Documentation
│   ├── agent_instructions.md # Agent behavior contract
│   ├── TOOLS.md              # Comprehensive tool reference
│   ├── CONFLUENCE_BIGQUERY_INTEGRATION.md
│   └── IAM_ANALYSIS_ARCHITECTURE.md
│
├── config/                    # Configuration files
│   └── service-account.json  # GCP credentials (not in repo)
│
├── backend/                   # Backend utilities
│   └── cache/                # Cache storage
│       └── confluence_cache.db # Confluence document cache
│
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in repo)
└── README.md                 # Project overview
```

### Key Files Explained

#### `agents/agent.py`
- Imports all 32 tools from `_tools/` directory
- Wraps functions as `FunctionTool` objects
- Creates `LlmAgent` with Gemini 2.5 Flash model
- Loads behavioral instructions from `docs/agent_instructions.md`

#### `agents/_tools/base.py`
- Shared configuration (PROJECT_ID, DEFAULT_DATASET, DEFAULT_TABLE)
- BigQuery client initialization
- `StructuredToolResponse` class for consistent tool outputs
- Common utility functions

#### `app.py`
- Flask web application for UI
- Calls ADK backend API (port 8000) for agent interactions
- Provides HTTP endpoints for dashboard metrics
- Handles SSE streaming for real-time chat responses
- Service discovery endpoints for GCP services

#### `cloud_functions/*/main.py`
- Independent data collection functions
- Fetch data from GCP APIs or external sources
- Transform and insert into BigQuery tables
- Error handling and logging
- Typically 150-800 lines each

#### `unified_data_api/main.py`
- FastAPI application for unified data access
- Provides REST API for cloud function results
- Aggregates data from multiple BigQuery tables
- Used by monitoring and reporting tools

### Tool Organization

Tools are organized by category in separate files:

1. **Security Tools** (`security_tools.py`)
   - Direct security findings analysis
   - High-level summaries and statistics

2. **BigQuery Tools** (`bigquery_tools.py`)
   - Standard database operations
   - Schema inspection, queries, cost analysis

3. **Exploration Tools** (`exploration_tools.py`)
   - Dataset and table discovery
   - Detailed object analysis

4. **Feed Tools** (`feed_tools.py`)
   - RSS feed parsing and analysis
   - Release notes monitoring

5. **Confluence Tools** (`confluence_tools.py`)
   - Documentation search and retrieval
   - Coverage analysis

6. **Service Discovery** (`service_discovery.py`)
   - GCP service catalog
   - Dynamic service analysis
   - Documentation parsing

7. **Service Evaluation** (`service_evaluation/`)
   - Security assessment framework
   - Compliance validation

8. **MSA Analyzer** (`msa_analyzer.py`)
   - Release notes impact analysis
   - Risk scoring and recommendations

### Adding to the Codebase

When adding new functionality:

1. **New Tool** → Create in appropriate `_tools/*.py` file
2. **New Cloud Function** → Add directory in `cloud_functions/`
3. **New API Endpoint** → Add route in `app.py`
4. **New BigQuery Table** → Create schema in Cloud Function
5. **New Documentation** → Add markdown in `docs/`

### Code Style Guidelines

- **Type Annotations**: Required for all function parameters
- **Docstrings**: Google-style docstrings for all public functions
- **Error Handling**: Use try/except with specific exceptions
- **Logging**: Use Python logging module, not print statements
- **Configuration**: Use environment variables, not hardcoded values
- **SQL**: Use parameterized queries, not string concatenation
- **Line Length**: Max 100 characters
- **Imports**: Organize as stdlib, third-party, local

---

## Next Steps

Now that you understand the development environment:

1. **Explore the Tools** - Read `docs/TOOLS.md` for detailed tool reference
2. **Try Example Queries** - Use the web UI to interact with the agent
3. **Deploy Cloud Functions** - Set up automated data collection
4. **Add Your Own Tool** - Follow the "Adding New Tools" section
5. **Customize Agent Behavior** - Edit `docs/agent_instructions.md`

For questions or issues:
- Check the Troubleshooting section
- Review existing tools as examples
- Consult the main README.md for architecture overview

Happy developing! 🚀
