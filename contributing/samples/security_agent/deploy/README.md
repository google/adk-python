# GCP Security Agent with ADK Integration

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()
[![ADK](https://img.shields.io/badge/Built%20with-ADK-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../../../LICENSE)

A comprehensive security management system for Google Cloud Platform, featuring chat-centric architecture and Google ADK (Agent Development Kit) integration for intelligent agent orchestration.

[Quick Start](#quick-start) • [Documentation](#documentation) • [Architecture](#architecture) • [API Reference](#api-reference)

---

## Overview

This security agent provides an intuitive chat interface for managing GCP security operations. It leverages Google's ADK to coordinate specialized AI agents that handle various security, compliance, and infrastructure management tasks.

## Key Features

- **Chat-Centric Interface**: Natural language interaction for all security operations
- **ADK-Powered Orchestration**: Intelligent routing to specialized agents (Security, IAM, Storage, Compliance)
- **Real-Time GCP Integration**: Direct connection to your GCP projects with live data
- **Conversation Memory**: Context persistence across sessions
- **Multi-Agent Architecture**: Coordinated specialist agents for different domains
- **Advanced Security Analysis**: Real-time risk assessment & vulnerability scanning
- **IAM Compliance**: Deep permissions analysis & policy compliance
- **Live Monitoring**: Performance dashboards with health monitoring
- **Compliance Frameworks**: SOC2, ISO27001, GDPR evaluation

## Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud Project with ADK access
- 4GB RAM (8GB recommended)
- Google Cloud SDK (gcloud CLI)

### Installation

```bash
# Clone repository
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent

# Install dependencies
pip install -r requirements.txt

# Setup authentication
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the application
python run_backend.py
python run_frontend.py
```

### Access Points

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Project Structure

```
security_agent/
├── agents/                 # ADK agent implementations
│   ├── coordinator_agent.py
│   ├── security_agent.py
│   └── direct_adk_agent.py
├── backend/               # FastAPI backend services
│   ├── api/              # API endpoints
│   ├── models/           # Data models
│   └── services/         # Business logic
├── frontend/             # Streamlit UI components
│   ├── components/       # UI modules
│   └── main_app.py      # Main application
├── docs/                 # Documentation
│   ├── architecture/     # Architecture details
│   └── guides/          # Setup and deployment guides
├── tools/                # Utility tools
│   ├── gcp_tools/       # GCP integrations
│   └── security_tools/  # Security utilities
└── config/              # Configuration files
```

## Documentation

### Setup & Configuration
- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Get running in 5 minutes
- **[Environment Setup](docs/guides/ENV_SETUP.md)** - Configuration details
- **[ADK Setup Guide](docs/ADK_SETUP_GUIDE.md)** - Google ADK configuration

### Architecture & Design
- **[Architecture Overview](docs/architecture/ARCHITECTURE.md)** - System design and patterns
- **[Chat Architecture](docs/architecture/CHAT_CENTRIC_ARCHITECTURE.md)** - Chat-first design
- **[Implementation Roadmap](docs/architecture/IMPLEMENTATION_ROADMAP.md)** - Development roadmap

### Deployment & Operations
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Manual Deployment](docs/MANUAL_DEPLOYMENT.md)** - Step-by-step deployment
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation

## Architecture

The system uses a chat-centric architecture with ADK-powered agent coordination:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chat UI       │───▶│  Coordinator     │───▶│  Specialized    │
│   (Frontend)    │    │  Agent (ADK)     │    │  Agents (ADK)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Conversation   │    │  LLM-Driven      │    │  • Security     │
│  Memory         │    │  Routing         │    │  • IAM          │
└─────────────────┘    └──────────────────┘    │  • Storage      │
                                               │  • Compliance   │
                                               └─────────────────┘
```

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/chat` | POST | Chat interaction endpoint |
| `/api/agents/status` | GET | Agent status monitoring |
| `/api/security/scan` | POST | Security vulnerability scan |
| `/api/iam/analyze` | GET | IAM permissions analysis |
| `/api/compliance/evaluate` | GET | Compliance framework evaluation |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/chat` | Real-time chat communication |
| `/ws/monitoring` | Live monitoring updates |

Full API documentation available at http://localhost:8000/docs when running locally.

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Key configuration options:

```env
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# ADK Configuration
ADK_EVALUATION_ENABLED=true
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=8501

# Security Settings
ENABLE_AUTHENTICATION=false
API_KEY=your-api-key  # If authentication enabled
```

### Required GCP APIs

Enable required APIs with this command:

```bash
gcloud services enable \
  cloudresourcemanager.googleapis.com \
  serviceusage.googleapis.com \
  iam.googleapis.com \
  securitycenter.googleapis.com \
  cloudkms.googleapis.com \
  secretmanager.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com
```

### Required IAM Roles

- `roles/viewer` - Basic project access
- `roles/iam.securityReviewer` - IAM analysis
- `roles/securitycenter.findingsViewer` - Security findings
- `roles/logging.viewer` - Cloud logging access
- `roles/monitoring.viewer` - Monitoring data

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Code Style

```bash
# Format code
black .

# Lint
pylint agents/ backend/ frontend/

# Type checking
mypy agents/ backend/ frontend/
```

### Local Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run in development mode
export ENVIRONMENT=development
python run_backend.py --reload
python run_frontend.py --reload
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Missing APIs**
   ```bash
   # Check enabled APIs
   gcloud services list --enabled
   
   # Enable missing APIs
   gcloud services enable [API_NAME]
   ```

3. **Port Conflicts**
   - Change ports in `.env` file
   - Or kill existing processes: `lsof -ti:8000 | xargs kill -9`

4. **Memory Issues**
   - Increase Docker memory allocation
   - Close unnecessary applications
   - Use cloud deployment for production workloads

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) file for details.

## Support

- [Google ADK Documentation](https://cloud.google.com/agent-development-kit)
- [Issue Tracker](https://github.com/google/adk-python/issues)
- [Discussion Forum](https://github.com/google/adk-python/discussions)

---
**Version**: 2.0 | **Last Updated**: August 2025