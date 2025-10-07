# GCP Security Intelligence Platform

**Version 1.0.1** | Production Ready ✅

A comprehensive security monitoring and analysis platform for Google Cloud Platform, featuring an ADK-powered AI agent with BigQuery integration and multiple user interfaces.

## 🎯 Overview

The GCP Security Intelligence Platform provides a unified AI agent that queries BigQuery security data through natural language. It supports multiple interfaces (ADK Backend, Flask UI, Chainlit UI, MCP Server) and includes modular Cloud Functions for automated data collection.

### Key Features

- 🤖 **AI-Powered Security Analysis** - Natural language queries to BigQuery security data
- 📊 **BigQuery Native** - Centralized data platform with real-time analysis
- 🔌 **Multiple Interfaces** - ADK, Flask, Chainlit, MCP Server
- ☁️ **Modular Cloud Functions** - Deploy only what you need
- 📚 **Documentation Sync** - Confluence → BigQuery integration
- 🔒 **Security Tools** - 3 specialized security analysis tools

## 🚀 Quick Start

### Prerequisites

```bash
# Required software
- Python 3.11+
- Google Cloud SDK
- ADK CLI (installed via pipx)

# Required GCP APIs
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/stuagano/adk-python.git
cd contributing/samples/security_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install ADK tool dependencies (BeautifulSoup, lxml, feedparser)
~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# 4. Configure environment
cp .env.example .env
# Edit .env with your GCP project details:
#   GOOGLE_CLOUD_PROJECT=your-project-id
#   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
#   BQ_DEFAULT_DATASET=security_insights
#   BQ_DEFAULT_TABLE=security_findings
```

### Start Services

```bash
# Start all services with one command
./scripts/start_all.sh

# This starts:
# ✅ ADK Backend (port 8000) - Agent orchestration & API
# ✅ Flask UI (port 5001) - Web interface
# ✅ Chainlit UI (port 8001) - Chat interface

# Stop all services
./scripts/stop_all.sh
```

### Access Interfaces

| Interface | URL | Purpose |
|-----------|-----|---------|
| **ADK Backend** | http://localhost:8000 | Direct API access, programmatic integration |
| **Flask UI** | http://localhost:5001 | Custom web interface, dashboards |
| **Chainlit UI** | http://localhost:8001 | Modern chat interface (recommended for end users) |

## 🛠️ Security Analysis Tools

The platform includes 3 specialized security tools that query BigQuery:

### 1. `get_security_insights_summary()`
Returns overview of security findings table with metrics:
- Total records, categories, severity levels
- Unique resources affected
- Date range of findings

### 2. `query_security_insights(query_filter, limit)`
Query security findings with SQL WHERE clause filtering.

**Available columns:**
- `id` (INTEGER) - Unique identifier
- `name` (STRING) - Finding name
- `category` (STRING) - Security category
- `severity` (STRING) - Severity level (HIGH, MEDIUM, LOW)
- `resource_name` (STRING) - Affected resource
- `description` (STRING) - Finding description
- `recommendation` (STRING) - Remediation steps
- `state` (STRING) - Current state
- `created_at` (STRING) - Creation timestamp
- `project_id` (STRING) - GCP project ID

**Example filters:**
```python
query_security_insights("severity = 'HIGH'")
query_security_insights("created_at >= '2025-10-06'")
query_security_insights("category = 'VULNERABILITY'", limit=10)
```

### 3. `get_security_statistics(group_by)`
Aggregated statistics grouped by field.

**Valid group_by values:**
- `severity` - Group by severity level
- `category` - Group by security category
- `state` - Group by finding state
- `project_id` - Group by GCP project

## 📊 BigQuery Schema

### Security Findings Table

**Dataset:** `security_insights`
**Table:** `security_findings`

**Columns:**
```sql
CREATE TABLE security_insights.security_findings (
  id INTEGER,
  name STRING,
  category STRING,
  severity STRING,
  resource_name STRING,
  description STRING,
  recommendation STRING,
  state STRING,
  created_at STRING,
  project_id STRING
)
```

### Example Queries

```sql
-- High severity findings
SELECT * FROM `project.security_insights.security_findings`
WHERE severity = 'HIGH'
ORDER BY created_at DESC;

-- Findings by category
SELECT category, COUNT(*) as count
FROM `project.security_insights.security_findings`
GROUP BY category
ORDER BY count DESC;

-- Recent findings (last 24 hours)
SELECT * FROM `project.security_insights.security_findings`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);
```

## 🔌 Chainlit Integration

### Standalone Usage

```bash
chainlit run chainlit_app.py --port 8001
```

### Integrate with Existing Chainlit App

**Method 1: One-Line Integration**
```python
from chainlit_agent import register_security_agent

@cl.set_chat_profiles
async def chat_profile():
    return register_security_agent(get_my_profiles())
```

**Method 2: Manual Integration**
```python
from chainlit_agent import SecurityAgentProfile

@cl.set_chat_profiles
async def chat_profile():
    profiles = SecurityAgentProfile.get_profiles()
    # Add your profiles here
    return profiles

@cl.on_chat_start
async def start():
    await SecurityAgentProfile.on_chat_start()

@cl.on_message
async def main(message: cl.Message):
    await SecurityAgentProfile.on_message(message)
```

See [docs/CHAINLIT_PLUGIN_INTEGRATION.md](docs/CHAINLIT_PLUGIN_INTEGRATION.md) for details.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           User Interfaces                │
│  Flask UI | Chainlit UI | MCP Server    │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   ADK Backend      │
         │  (port 8000)       │
         │  Gemini 2.5 Flash  │
         └─────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐   ┌────▼─────┐   ┌───▼────┐
│Security│   │BigQuery  │   │Service │
│Tools   │   │Tools     │   │Discovery│
│(3)     │   │(~10)     │   │(~10)   │
└───┬────┘   └────┬─────┘   └───┬────┘
    │             │              │
    └─────────────▼──────────────┘
                  │
         ┌────────▼────────┐
         │    BigQuery     │
         │  Data Platform  │
         └────────┬────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
┌───▼──────────┐    ┌──────────▼───┐
│Cloud Functions│   │External APIs │
│(IAM, Compute, │   │(GCP, RSS,    │
│ Storage, etc.)│   │ Confluence)  │
└──────────────┘    └──────────────┘
```

### Key Architectural Principles

1. **Separation of Concerns**: Agent queries BigQuery, Cloud Functions populate data
2. **Modular Deployment**: Deploy only the Cloud Functions you need
3. **Direct Access**: Agent has full BigQuery access for flexible queries
4. **No Coupling**: Agent never calls Cloud Functions directly
5. **Scheduled Updates**: Cloud Functions run on schedules via Cloud Scheduler

## ☁️ Cloud Functions (Optional)

Deploy modular Cloud Functions to populate BigQuery with security data:

### IAM & Security (5 functions)
- `fetch_iam_accounts` - Users, groups, service accounts
- `fetch_service_account_roles` - Service account permissions
- `fetch_user_roles` - User role assignments
- `fetch_custom_roles` - Custom IAM roles
- `fetch_standard_roles` - Google-managed roles

### Infrastructure (3 functions)
- `fetch_compute_instances` - VM security analysis
- `fetch_firewall_rules` - Network security, risk scoring
- `fetch_storage_buckets` - Storage security

### Feeds & Documentation (4 functions)
- `fetch_security_findings` - Security Command Center
- `fetch_security_feeds` - RSS security feeds
- `fetch_gcp_release_notes` - Platform updates
- `confluence_sync` - Documentation → BigQuery

See [cloud_functions/README.md](cloud_functions/README.md) for deployment instructions.

## 🧪 Testing & Validation

### Dependency Check

```bash
# Quick validation (runs in startup script)
python3 -c "import flask, google.cloud.aiplatform, requests, dotenv"

# Comprehensive validation
python3 tests/test_dependencies.py
```

### Test Services

```bash
# Start services
./scripts/start_all.sh

# Test ADK Backend
curl http://localhost:8000/health

# Test Flask UI
curl http://localhost:5001

# Test Chainlit UI
curl http://localhost:8001
```

## 📚 Documentation

### Getting Started
- [FINAL_STATUS.md](FINAL_STATUS.md) - Complete platform status and features
- [CHANGELOG.md](CHANGELOG.md) - Version history and recent changes
- [docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md) - Comprehensive setup guide

### Integration Guides
- [docs/CHAINLIT_INTEGRATION.md](docs/CHAINLIT_INTEGRATION.md) - Chainlit UI setup
- [docs/CHAINLIT_PLUGIN_INTEGRATION.md](docs/CHAINLIT_PLUGIN_INTEGRATION.md) - Plug-and-play integration
- [docs/MCP_SERVER_INTEGRATION.md](docs/MCP_SERVER_INTEGRATION.md) - Model Context Protocol
- [docs/TOOLS.md](docs/TOOLS.md) - Complete tool reference

### Architecture & Development
- [docs/agent_instructions.md](docs/agent_instructions.md) - Agent behavior contract
- [cloud_functions/README.md](cloud_functions/README.md) - Cloud Functions guide
- [cloud_functions/tests/README.md](cloud_functions/tests/README.md) - Testing guide

## 🔧 Configuration

### Environment Variables (.env)

```bash
# GCP Configuration (Required)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account.json
GOOGLE_CLOUD_LOCATION=us-central1

# BigQuery Configuration (Required)
BQ_DEFAULT_DATASET=security_insights
BQ_DEFAULT_TABLE=security_findings

# ADK Configuration (Required)
ADK_BASE_URL=http://localhost:8000
ADK_AGENT_MODEL=gemini-2.5-flash
GOOGLE_GENAI_USE_VERTEXAI=1

# Confluence Configuration (Optional)
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACES=SEC,POLICY,GCP
```

### Chainlit Configuration

Located in [.chainlit/config.toml](.chainlit/config.toml):

```toml
[project]
enable_telemetry = false
user_env = []  # Empty for local development

[UI]
name = "GCP Security Agent"
default_collapse_content = true
```

## 🔧 Recent Fixes (v1.0.1)

### ADK Compatibility
- ✅ Fixed return types: `StructuredToolResponse` → `str` for ADK automatic function calling
- ✅ ADK requires simple types (str, dict, int) - custom dataclasses not supported

### BigQuery Schema
- ✅ Fixed column reference: `resource_type` → `resource_name`
- ✅ Added schema documentation to tool docstrings

### Chainlit
- ✅ Fixed directory structure: `.chainlit` file → `.chainlit/config.toml` directory
- ✅ Configured `user_env = []` for local development
- ✅ Prevented duplicate ADK session creation

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## 📝 Example Usage

### Natural Language Queries (via Chainlit)

```
"Show me security findings from the last 24 hours"
"List all HIGH severity vulnerabilities"
"Get security statistics grouped by category"
"Find findings related to storage buckets"
"What are the most common security issues?"
```

### Programmatic Access (via Python)

```python
import requests

# Query ADK backend
response = requests.post('http://localhost:8000/run', json={
    'user_id': 'test-user',
    'message': 'Show me high severity findings'
})

results = response.json()
print(results)
```

## 🚢 Production Deployment

### Deploy to Cloud Run

```bash
# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/security-agent

# Deploy ADK backend
gcloud run deploy security-agent \
  --image gcr.io/$PROJECT_ID/security-agent \
  --port 8000 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID

# Deploy Flask UI
gcloud run deploy security-ui \
  --image gcr.io/$PROJECT_ID/security-agent \
  --port 5001 \
  --set-env-vars ADK_BASE_URL=https://security-agent-xxx.run.app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the Google ADK Python samples.

## 🙏 Acknowledgments

- Google Cloud Platform team for the ADK framework
- Gemini team for powerful language models
- All contributors to the security platform

---

**Status**: ✅ Production Ready (v1.0.1)
**Last Updated**: October 7, 2025
**Built with ❤️ for GCP Security**
