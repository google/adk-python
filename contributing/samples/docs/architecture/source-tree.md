# Source Tree Documentation
# ADK Security Agent Project Structure

## Version 4.0 | Last Updated: 2025-01-18

## Project Root Structure

```
security_agent/
├── agent.py                     # Single-agent ADK implementation (Cloud Run compatible)
├── backend/                     # Backend intelligence layer
├── frontend/                    # Frontend thin client
├── tests/                       # Test suites
├── deploy/                      # Deployment configurations
├── docs/                        # Documentation
├── evaluation/                  # ADK pattern evaluation
├── coordination/                # Multi-agent coordination (future)
├── memory/                      # Persistent memory storage
├── scripts/                     # Utility scripts
├── run_backend.py              # Backend launcher
├── run_frontend.py             # Frontend launcher
└── venv/                       # Virtual environment
```

## Detailed Structure

### `/agent.py` - Single Agent Implementation
```python
"""
Purpose: Standalone ADK agent for Cloud Run deployment
Key Components:
- discover_and_analyze_resources(): RADAR Recognition + Assessment
- generate_recommendations(): RADAR Decision + Action  
- root_agent: Main agent with Gemini 2.0 Flash
"""
```

### `/backend/` - Intelligence Layer
```
backend/
├── __init__.py
├── main.py                     # FastAPI application entry point
├── agents/                     # Agent implementations
│   ├── __init__.py
│   ├── radar_coordinator.py   # Main RADAR orchestrator
│   ├── recognition_agent.py   # Resource discovery
│   ├── assessment_agent.py    # Security assessment
│   ├── decision_agent.py      # Priority and recommendations
│   ├── action_agent.py        # Remediation execution
│   └── review_agent.py        # Verification and monitoring
├── api/                        # API endpoints
│   ├── __init__.py
│   ├── gcp.py                 # GCP service integration
│   ├── gcp_direct.py          # Direct GCP API calls
│   ├── asset_inventory.py     # Asset discovery endpoints
│   ├── security.py            # Security analysis endpoints
│   ├── iam.py                 # IAM management endpoints
│   ├── monitoring.py          # Logging and metrics
│   ├── recommendations.py     # Recommender API integration
│   ├── storage.py             # Cloud Storage operations
│   ├── keys.py                # API key management
│   ├── advisory_notifications.py  # Security bulletins
│   ├── conversation_context.py    # Session management
│   ├── logs.py                # Centralized logging
│   ├── org_policy.py          # Organization policies
│   └── service_management.py  # Service control
├── config/                     # Configuration files
│   └── secrets/               # Credential storage
│       ├── README.md          # Security instructions
│       └── *.json             # Service account keys (gitignored)
├── data/                      # Local data storage
│   └── chat_sessions.db      # Session persistence
├── requirements.txt           # Python dependencies
└── backend.log               # Application logs
```

### `/frontend/` - Thin Client Layer
```
frontend/
├── __init__.py
├── main_app.py                # Main Streamlit application
├── thin_client.py             # Pure thin client implementation
├── minimal_client.py          # Minimal UI version
├── config.py                  # Frontend configuration
├── startup_status.py          # Startup health checks
├── unified_api_client.py      # Backend API client
├── archive/                   # Deprecated components
│   └── components/            # Old UI components
│       ├── chat/             # Chat components
│       │   ├── chat_view.py
│       │   ├── chat_manager.py
│       │   └── chat_commands.py
│       ├── dashboard/        # Dashboard views
│       │   ├── dashboard_view.py
│       │   ├── asset_charts.py
│       │   └── multi_agent_graph_view.py
│       ├── radar/            # RADAR UI components
│       │   ├── radar_coordinator_view.py
│       │   ├── radar_state_manager.py
│       │   └── *_chat_view.py
│       ├── security/         # Security views
│       │   ├── iam_analyzer_view.py
│       │   └── security_evaluation_view.py
│       └── shared/           # Shared components
│           ├── chat_streaming_base.py
│           └── gcp_api_explorer_view.py
└── logs/
    └── frontend.log          # Frontend logs
```

### `/tests/` - Test Suites
```
tests/
├── __init__.py
├── conftest.py                # Pytest configuration
├── pytest.ini                 # Pytest settings
├── TEST_PLAN.md              # Test strategy document
├── TEST_RESULTS.md           # Test execution results
├── test_agent_llm.py         # Agent LLM tests
├── test_agent_tool_routing.py # Tool routing tests
├── test_api_endpoints.py     # API endpoint tests
├── test_asset_inventory_integration.py
├── test_chat_responses.py    # Chat functionality tests
├── test_complete_integration.py # E2E integration tests
├── test_dashboard_integration.py
├── test_frontend_backend_integration.py
├── test_security.py          # Security tests
├── test_session_persistence.py # Session management tests
└── frontend/                 # Frontend-specific tests
    └── components/
        └── radar/
            └── test_radar_websocket_client.py
```

### `/deploy/` - Deployment Configuration
```
deploy/
├── README.md                  # Deployment instructions
├── CHANGELOG.md              # Version history
├── Dockerfile                # Container definition
├── docker-compose.yml        # Local development setup
├── cloudbuild.yaml          # Cloud Build CI/CD
├── requirements.txt         # Deployment dependencies
├── run_backend.py           # Backend startup script
├── run_frontend.py          # Frontend startup script
├── setup_gcp_permissions.py # GCP IAM setup
└── diagnose_connection.py   # Connection troubleshooting
```

### `/docs/` - Documentation
```
docs/
├── prd.md                    # Product Requirements Document
├── architecture.md           # System Architecture
├── architecture/            # Detailed architecture docs
│   ├── tech-stack.md       # Technology stack
│   ├── source-tree.md      # This document
│   ├── coding-standards.md # Coding guidelines
│   ├── ADK_INTEGRATION.md  # ADK integration guide
│   ├── API_REFERENCE.md    # API documentation
│   ├── DEPLOYMENT_GUIDE.md # Deployment procedures
│   └── ... (other docs)
├── guides/                  # User guides
│   ├── QUICK_START.md     # Getting started
│   └── ENV_SETUP.md       # Environment setup
└── qa/                     # QA documentation
    └── ... (test docs)
```

### `/evaluation/` - ADK Pattern Evaluation
```
evaluation/
├── README.md                 # Evaluation framework docs
├── adk_evaluator.py         # Main evaluation script
├── validate_adk_compliance.py # ADK compliance checker
├── config/                  # Evaluation configuration
│   ├── evaluation_config.yaml
│   └── test_config.json
├── datasets/                # Test datasets
│   ├── compliance_check.test.json
│   ├── incident_response.test.json
│   └── vulnerability_assessment.test.json
├── evaluators/              # Evaluation modules
│   ├── compliance_evaluator.py
│   ├── performance_evaluator.py
│   └── security_evaluator.py
├── metrics/                 # Custom metrics
│   ├── custom_metrics.py
│   └── security_metrics.py
└── examples/                # Usage examples
    ├── simple_agent_test.py
    └── web_ui_example.py
```

### `/coordination/` - Multi-Agent Coordination (Future)
```
coordination/
├── memory_bank/             # Shared memory for agents
│   └── __init__.py
├── orchestration/           # Orchestration logic
│   └── __init__.py
└── subtasks/               # Task decomposition
    └── __init__.py
```

### `/memory/` - Persistent Storage
```
memory/
├── claude-flow-data.json   # Claude Flow integration data
└── memory-store.json      # Agent memory persistence
```

### `/scripts/` - Utility Scripts
```
scripts/
└── service_health.py      # Health check utilities
```

## File Naming Conventions

### Python Files
- **Modules**: `snake_case.py` (e.g., `radar_coordinator.py`)
- **Classes**: `PascalCase` in files (e.g., `class RadarCoordinator`)
- **Tests**: `test_*.py` prefix (e.g., `test_security.py`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)

### Documentation Files
- **Markdown**: `UPPER_CASE.md` for primary docs
- **Guides**: `kebab-case.md` for sub-documents
- **API Docs**: `API_*.md` prefix

### Configuration Files
- **YAML**: `snake_case.yaml` or `.yml`
- **JSON**: `snake_case.json`
- **Environment**: `.env` (no suffix)

## Import Organization

### Standard Import Order
```python
# 1. Standard library imports
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party imports
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 3. Google/GCP imports
from google.cloud import asset_v1
from google.genai.adk import Agent, Tool

# 4. Local application imports
from backend.agents.radar_coordinator import RadarCoordinator
from backend.api.security import SecurityAnalyzer
```

## Module Responsibilities

### Core Modules

#### `backend/main.py`
- FastAPI application initialization
- Router registration
- Middleware configuration
- Health check endpoints
- Startup/shutdown events

#### `backend/agents/radar_coordinator.py`
- Main orchestration logic
- Agent delegation
- RADAR methodology implementation
- Response synthesis

#### `frontend/thin_client.py`
- Streamlit UI rendering
- User input handling
- Backend API communication
- Response streaming display

### API Modules

#### `backend/api/asset_inventory.py`
- Resource discovery endpoints
- Asset search functionality
- Inventory export capabilities

#### `backend/api/security.py`
- Security analysis endpoints
- Vulnerability detection
- Risk scoring

#### `backend/api/iam.py`
- IAM policy analysis
- Permission auditing
- Service account management

## Data Flow Paths

### Request Flow
```
1. User Input → frontend/thin_client.py
2. API Call → backend/main.py
3. Route → backend/api/*.py
4. Agent → backend/agents/*.py
5. Tool → GCP API calls
6. Response → backend/api/*.py
7. Stream → frontend/thin_client.py
8. Display → User
```

### Session Flow
```
1. Session Create → frontend/config.py
2. Session Store → backend/api/conversation_context.py
3. Context Load → backend/agents/radar_coordinator.py
4. Context Use → Agent decision making
5. History Update → backend/data/chat_sessions.db
```

## Configuration Files

### `.env` Template
```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
VERTEX_AI_LOCATION=us-central1

# Application Configuration
BACKEND_URL=http://localhost:8000
BACKEND_PORT=8000
FRONTEND_PORT=8501

# Feature Flags
ENABLE_CACHING=true
ENABLE_MONITORING=true
DEBUG_MODE=false
```

### `requirements.txt` Structure
```
# Core Dependencies
fastapi==0.104.1
streamlit==1.28.2
google-genai==0.4.0

# Google Cloud
google-cloud-asset==3.19.1
google-cloud-iam==2.12.1
google-cloud-logging==3.8.0

# Utilities
pydantic==2.5.0
httpx==0.25.2
python-dotenv==1.0.0
```

## Testing Structure

### Unit Tests
- Location: `tests/test_*.py`
- Coverage: Individual functions and methods
- Framework: pytest

### Integration Tests
- Location: `tests/test_*_integration.py`
- Coverage: Module interactions
- Framework: pytest with fixtures

### End-to-End Tests
- Location: `tests/test_complete_integration.py`
- Coverage: Full user workflows
- Framework: pytest + httpx

## Build Artifacts

### Docker Images
```
security-agent-frontend:latest
security-agent-backend:latest
```

### Cloud Build Outputs
```
gs://your-bucket/builds/
├── frontend/
│   └── latest/
└── backend/
    └── latest/
```

## Deprecated/Legacy Code

### Archive Policy
- Deprecated code moved to `archive/` directories
- Retention period: 6 months
- Documentation of deprecation reasons required

### Migration Paths
- Old component → New component mapping documented
- Gradual deprecation with warnings
- Backward compatibility period: 3 months

## Security Considerations

### Sensitive Files
```
Never commit:
- *.json (service accounts)
- .env (environment variables)
- *.key (private keys)
- *.pem (certificates)
```

### Access Control
```
Read-only:
- docs/
- tests/

Restricted:
- backend/config/secrets/
- deploy/

Public:
- README.md
- LICENSE
```

## Maintenance Guidelines

### Regular Updates
- Dependencies: Monthly
- Documentation: With each feature
- Tests: With each change
- Security scans: Weekly

### Code Review Checklist
- [ ] Follows naming conventions
- [ ] Includes tests
- [ ] Updates documentation
- [ ] No sensitive data
- [ ] Passes linting

## Future Structure Plans

### Phase 2 Additions
```
security_agent/
├── graphql/          # GraphQL API layer
├── workers/          # Background job processors
├── cache/           # Redis cache layer
└── monitoring/      # Custom monitoring
```

### Phase 3 Evolution
```
security_agent/
├── microservices/   # Service decomposition
├── events/          # Event-driven components
├── ml/             # ML model management
└── edge/           # Edge computing modules
```