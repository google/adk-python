# GCP Security Intelligence Platform v1.0.1

<div align="center">

[![Security Agent](https://img.shields.io/badge/Status-v1.0.1-green.svg)](contributing/samples/security_agent/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Production-green.svg)](https://cloud.google.com/vertex-ai)
[![ADK](https://img.shields.io/badge/Built%20with-ADK-blue.svg)](https://github.com/stuagano/adk-python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Production-ready GCP security analysis platform with ADK-powered AI agent and BigQuery integration**

[🚀 Quick Start](#-quick-start) • [✨ Features](#-features) • [📖 Documentation](#-documentation) • [🔧 Recent Fixes](#-recent-fixes-v101)

</div>

---

## 🎯 Overview

The GCP Security Intelligence Platform v1.0.1 is a production-ready security analysis platform that provides natural language queries to BigQuery security data. Built on the ADK (Agent Development Kit) framework with multiple user interfaces (ADK Backend, Flask UI, Chainlit UI, MCP Server).

## ✨ Features

### 🤖 **AI-Powered Security Analysis**
- Natural language queries to BigQuery security data
- 3 specialized security tools for analysis
- Gemini 2.5 Flash powered conversational AI
- ADK automatic function calling

### 🔌 **Multiple Interfaces**
- **ADK Backend** (port 8000) - Direct API access
- **Flask UI** (port 5001) - Web interface
- **Chainlit UI** (port 8001) - Modern chat interface
- **MCP Server** - Claude Desktop integration

### 📊 **BigQuery Native**
- Centralized security data platform
- Real-time analysis and queries
- Correct schema with proper column names
- Support for custom SQL queries

### ☁️ **Modular Cloud Functions**
- Deploy only what you need
- IAM & Security (5 functions)
- Infrastructure (3 functions)
- Feeds & Documentation (4 functions)

## 🚀 Quick Start

**Get started in under 5 minutes:**

```bash
# 1. Clone the repository
git clone https://github.com/stuagano/adk-python.git
cd adk-python/contributing/samples/security_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install ADK tool dependencies
~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# 4. Configure environment
cp .env.example .env
# Edit .env with your GCP project details

# 5. Start all services
./scripts/start_all.sh

# Services now running:
# ✅ ADK Backend: http://localhost:8000
# ✅ Flask UI: http://localhost:5001
# ✅ Chainlit UI: http://localhost:8001
```

## 🛠️ Security Tools

The platform includes 3 specialized security analysis tools:

### 1. `get_security_insights_summary()`
Overview of security findings with metrics:
- Total records, categories, severity levels
- Unique resources affected
- Date range of findings

### 2. `query_security_insights(query_filter, limit)`
Query security findings with SQL WHERE filtering

**Available columns:**
- id, name, category, severity
- resource_name, description, recommendation
- state, created_at, project_id

### 3. `get_security_statistics(group_by)`
Aggregated statistics grouped by field
- severity, category, state, project_id

## 📊 BigQuery Schema

**Dataset:** `security_insights`
**Table:** `security_findings`

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

## 🔧 Recent Fixes (v1.0.1)

### ADK Compatibility
- ✅ Fixed return types: `StructuredToolResponse` → `str`
- ✅ ADK automatic function calling requires simple types
- ✅ All security tools now compatible

### BigQuery Schema
- ✅ Fixed column reference: `resource_type` → `resource_name`
- ✅ Added schema documentation to tool docstrings
- ✅ Accurate SQL queries

### Chainlit
- ✅ Fixed directory structure: `.chainlit/config.toml`
- ✅ Configured for local development
- ✅ Prevented duplicate session creation

See [CHANGELOG.md](contributing/samples/security_agent/CHANGELOG.md) for complete version history.

## 📚 Documentation

### Getting Started
- **[README.md](contributing/samples/security_agent/README.md)** - Complete platform documentation
- **[FINAL_STATUS.md](contributing/samples/security_agent/FINAL_STATUS.md)** - Platform status and features
- **[CHANGELOG.md](contributing/samples/security_agent/CHANGELOG.md)** - Version history

### Integration Guides
- **[Chainlit Integration](contributing/samples/security_agent/docs/CHAINLIT_INTEGRATION.md)** - Chat UI setup
- **[Chainlit Plugin](contributing/samples/security_agent/docs/CHAINLIT_PLUGIN_INTEGRATION.md)** - Plug-and-play integration
- **[MCP Server](contributing/samples/security_agent/docs/MCP_SERVER_INTEGRATION.md)** - Model Context Protocol
- **[Tools Reference](contributing/samples/security_agent/docs/TOOLS.md)** - Complete tool documentation

### Architecture & Development
- **[Agent Instructions](contributing/samples/security_agent/docs/agent_instructions.md)** - Agent behavior contract
- **[Cloud Functions](contributing/samples/security_agent/cloud_functions/README.md)** - Data collection guide
- **[Testing](contributing/samples/security_agent/cloud_functions/tests/README.md)** - Testing guide

## 🧪 Example Queries

### Via Chainlit (Natural Language)
```
"Show me security findings from the last 24 hours"
"List all HIGH severity vulnerabilities"
"Get security statistics grouped by category"
"Find findings related to storage buckets"
```

### Via BigQuery (SQL)
```sql
-- High severity findings
SELECT * FROM `project.security_insights.security_findings`
WHERE severity = 'HIGH'
ORDER BY created_at DESC;

-- Recent findings
SELECT * FROM `project.security_insights.security_findings`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);
```

## 🏗️ Architecture

```
User Interfaces (Flask, Chainlit, MCP)
              ↓
      ADK Backend (port 8000)
              ↓
    Security Tools (3) + BigQuery Tools
              ↓
     BigQuery Data Platform
              ↓
Cloud Functions (Optional) + External APIs
```

### Key Principles
1. **Separation of Concerns** - Agent queries BigQuery, Cloud Functions populate data
2. **Modular Deployment** - Deploy only needed Cloud Functions
3. **Direct Access** - Agent has full BigQuery access
4. **No Coupling** - Agent never calls Cloud Functions directly

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
