# GCP Security Agent

A single-agent implementation that provides comprehensive GCP security analysis through backend API integration.

## Architecture

This agent follows a clean client-server architecture:
- **Agent (Frontend)**: Single agent with lightweight tool wrappers
- **Backend APIs**: FastAPI server providing the actual implementation

## Available Tools

The agent has access to 12 security analysis tools:

### Discovery & Inventory
- `discover_assets` - Complete inventory of GCP resources
- `analyze_service_usage` - Check enabled APIs and services

### Security Analysis  
- `analyze_security` - Security Command Center findings
- `analyze_iam` - IAM policies and permissions review
- `analyze_storage` - Storage bucket security audit
- `manage_api_keys` - API key usage and restrictions

### Compliance & Monitoring
- `check_org_policies` - Organization policy compliance
- `analyze_monitoring` - Monitoring and alerting configuration
- `analyze_logs` - Logging configuration and events
- `check_advisory_notifications` - Security advisories

### Recommendations
- `get_security_recommendations` - Prioritized action items
- `run_comprehensive_security_scan` - Full security assessment

## Backend API Modules

Each tool maps to a backend API module in `/backend/api/`:

| Tool | Backend Module | Endpoint |
|------|---------------|----------|
| discover_assets | asset_inventory.py | /api/v1/assets/list |
| analyze_security | security.py | /api/v1/security/findings |
| analyze_iam | iam.py | /api/v1/iam/analyze |
| analyze_storage | storage.py | /api/v1/storage/analyze |
| analyze_monitoring | monitoring.py | /api/v1/monitoring/analyze |
| analyze_logs | logs.py | /api/v1/logs/analyze |
| check_org_policies | org_policy.py | /api/v1/org-policy/check |
| analyze_service_usage | service_management.py | /api/v1/services/analyze |
| check_advisory_notifications | advisory_notifications.py | /api/v1/advisory/check |
| manage_api_keys | keys.py | /api/v1/keys/analyze |
| get_security_recommendations | recommendations.py | /api/v1/recommendations/security |

## Configuration

Set these environment variables:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export BACKEND_API_URL=http://localhost:8000  # Backend FastAPI server
export API_TIMEOUT=30  # API call timeout in seconds
```

## Deployment

### Local Development

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Run the agent:
```bash
python agent.py
```

### Cloud Run Deployment

```bash
gcloud run deploy gcp-security-agent \
  --source . \
  --port 8080 \
  --project $GOOGLE_CLOUD_PROJECT \
  --set-env-vars BACKEND_API_URL=$BACKEND_API_URL \
  --allow-unauthenticated \
  --region us-central1
```

## Usage Example

```python
from agent import agent
from google.adk import Runner

# Initialize runner
runner = Runner(agent)

# Run comprehensive scan
response = runner.run("Run a comprehensive security scan of my GCP project")
print(response)

# Get specific analysis
response = runner.run("Check my IAM permissions for security issues")
print(response)
```

## Clean Architecture Benefits

- **Single Agent**: No complex multi-agent coordination
- **Clear Separation**: Agent handles UI, backend handles logic
- **Maintainable**: Each API module is independent
- **Scalable**: Easy to add new tools by creating API endpoints
- **Testable**: Each component can be tested in isolation
- **Secure**: Backend holds credentials, frontend is unprivileged