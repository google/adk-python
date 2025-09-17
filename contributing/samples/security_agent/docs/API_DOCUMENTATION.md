# GCP Security Agent API Documentation

## Overview

The GCP Security Agent provides a comprehensive REST API for security analysis, resource discovery, and compliance monitoring. This document provides detailed information about all available endpoints, request/response formats, authentication, and usage examples.

## Base URL

```
Local Development: http://localhost:8000
Production: https://your-security-agent.run.app
```

## Authentication

The API uses Google Cloud Service Account authentication. Ensure your service account has the following IAM roles:

- Cloud Asset Viewer
- Security Center Admin Viewer  
- Storage Admin
- IAM Security Reviewer
- Recommender Viewer
- Secret Manager Viewer
- Monitoring Viewer

### Environment Variables

```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

## Rate Limits

- **Chat Operations**: 30 requests/minute
- **Heavy Operations**: 5 requests/minute  
- **Default Operations**: 100 requests/minute
- **Rate Limit Window**: 60 seconds

Check rate limit status: `GET /api/v1/rate-limit/status`

## Core API Endpoints

### 1. Chat & Agent Interface

#### POST /api/v1/chat/message
Process security queries through the ADK agent with session persistence.

**Request Body:**
```json
{
  "query": "What are my security vulnerabilities?",
  "session_id": "session_123",
  "user_id": "user_456"
}
```

**Response:**
```json
{
  "response": "Security analysis shows 3 critical findings...",
  "session_id": "session_123",
  "user_id": "user_456",
  "success": true
}
```

**Example Usage:**
```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me my IAM security issues", "session_id": "demo"}'
```

### 2. Health & Monitoring

#### GET /health
Quick health check with comprehensive monitoring.

**Response:**
```json
{
  "status": "healthy",
  "message": "System operational",
  "timestamp": "2025-09-08T18:55:00Z",
  "version": "1.13.0",
  "system_mode": "robust_fallback_enabled",
  "components": {
    "agent_llm": "available",
    "iam_analysis": "available",
    "recommendations": "available",
    "websocket_streaming": "available",
    "database": "healthy",
    "gcp_apis": "operational"
  },
  "features": {
    "comprehensive_monitoring": true,
    "rate_limiting": true,
    "adk_session_management": true,
    "websockets": true,
    "real_time_streaming": true,
    "background_data_refresh": true
  }
}
```

#### GET /api/v1/health/comprehensive
Detailed health monitoring with performance metrics.

#### GET /status
Detailed service status with system metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-08T18:55:00Z",
  "uptime": {
    "seconds": 86400,
    "hours": 24.0,
    "human_readable": "24h 0m"
  },
  "system": {
    "cpu": {"usage_percent": 15.2, "status": "normal"},
    "memory": {"usage_percent": 45.8, "available_gb": 2.1},
    "disk": {"usage_percent": 68.3, "free_gb": 12.7}
  }
}
```

### 3. Session Management

#### GET /api/v1/sessions
List all active sessions.

#### POST /api/v1/sessions
Create a new session.

**Request Body:**
```json
{
  "user_id": "user_123",
  "metadata": {"context": "security_analysis"}
}
```

#### GET /api/v1/sessions/{session_id}
Get session details and conversation history.

#### DELETE /api/v1/sessions/{session_id}
Delete a session and its history.

### 4. GCP Resource Management

#### GET /api/v1/gcp/projects
List accessible GCP projects.

**Response:**
```json
{
  "success": true,
  "projects": [
    {
      "id": "my-project-123",
      "name": "Production Project",
      "state": "ACTIVE"
    }
  ],
  "total": 1
}
```

#### GET /api/v1/gcp/projects/{project_id}
Get detailed project information.

### 5. Asset Inventory & Discovery

#### GET /api/v1/assets/discover/{project_id}
Discover all assets in a project.

**Response:**
```json
{
  "success": true,
  "assets": [
    {
      "name": "instance-1",
      "asset_type": "compute.googleapis.com/Instance",
      "location": "us-central1-a",
      "state": "RUNNING",
      "created": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 145,
  "categories": {
    "compute": 12,
    "storage": 8,
    "network": 15
  }
}
```

#### GET /api/v1/assets/security/analyze/{project_id}
Analyze security posture of all assets.

**Response:**
```json
{
  "success": true,
  "findings": [
    {
      "severity": "CRITICAL",
      "category": "IAM_MISCONFIGURATION",
      "resource": "projects/my-project/instances/vm-1",
      "description": "Instance has overly broad IAM permissions",
      "recommendation": "Apply principle of least privilege"
    }
  ],
  "summary": {
    "critical": 2,
    "high": 8,
    "medium": 15,
    "low": 23
  }
}
```

### 6. IAM Analysis

#### GET /api/v1/iam/policies/{project_id}
Analyze IAM policies for security issues.

**Response:**
```json
{
  "success": true,
  "policies": [
    {
      "resource": "projects/my-project",
      "bindings": [
        {
          "role": "roles/editor",
          "members": ["user:admin@company.com"],
          "risk_level": "HIGH",
          "reason": "Overly broad permissions"
        }
      ]
    }
  ],
  "recommendations": [
    "Replace Editor role with specific permissions",
    "Remove unused service accounts"
  ]
}
```

#### GET /api/v1/iam/service-accounts/{project_id}
List and analyze service accounts.

#### GET /api/v1/iam/custom-roles/{project_id}
Analyze custom IAM roles.

### 7. Security Analysis

#### GET /api/v1/security/findings/{project_id}
Get security findings from Security Command Center.

**Query Parameters:**
- `severity`: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
- `category`: Filter by finding category
- `state`: Filter by finding state (ACTIVE, INACTIVE)

**Response:**
```json
{
  "success": true,
  "findings": [
    {
      "name": "organizations/123/sources/456/findings/789",
      "severity": "HIGH",
      "category": "OPEN_FIREWALL",
      "state": "ACTIVE",
      "resource_name": "projects/my-project/global/firewalls/allow-all",
      "description": "Firewall rule allows unrestricted access",
      "recommendation": "Restrict source IP ranges"
    }
  ],
  "total": 42
}
```

#### GET /api/v1/security/vulnerabilities/{project_id}
Scan for vulnerabilities across all resources.

### 8. Storage Security

#### GET /api/v1/storage/buckets/{project_id}
List and analyze Cloud Storage buckets for security.

**Response:**
```json
{
  "success": true,
  "buckets": [
    {
      "name": "my-bucket",
      "location": "US",
      "public_access": false,
      "encryption": "GOOGLE_MANAGED",
      "security_score": 85,
      "issues": [
        "Bucket versioning disabled",
        "No lifecycle policies configured"
      ]
    }
  ]
}
```

### 9. Monitoring & Logging

#### GET /api/v1/monitoring/metrics/{project_id}
Get security-relevant monitoring metrics.

**Query Parameters:**
- `metric_type`: Specific metric type to retrieve
- `start_time`: Start time for metrics (ISO format)
- `end_time`: End time for metrics (ISO format)

#### GET /api/v1/monitoring/alerts/{project_id}
List active security alerts.

### 10. Recommendations

#### GET /api/v1/recommendations/{project_id}
Get security recommendations from Google Cloud Recommender.

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "name": "projects/123/locations/global/recommenders/google.iam.policy.Recommender/recommendations/abc",
      "description": "Remove unused IAM role binding",
      "recommender_subtype": "REMOVE_ROLE",
      "priority": "HIGH",
      "impact": {
        "security_projection": {
          "details": "Reduces attack surface"
        }
      }
    }
  ],
  "total": 15
}
```

### 11. Knowledge Base

#### GET /api/v1/knowledge/search
Search enterprise policies and coding standards.

**Query Parameters:**
- `q`: Search query
- `type`: Content type (policy, standard, compliance)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "title": "Data Encryption Policy",
      "content": "All data must be encrypted at rest and in transit...",
      "type": "policy",
      "severity": "CRITICAL"
    }
  ]
}
```

### 12. Advanced IAM Features

#### GET /api/v1/iam-recommendations/{project_id}
Get intelligent IAM recommendations.

#### GET /api/v1/least-privilege/{project_id}
Analyze least privilege compliance.

#### GET /api/v1/cross-project/permissions
Analyze cross-project permission dependencies.

### 13. Custom Roles Analysis

#### GET /api/v1/custom-roles/analyze/{project_id}
Analyze custom IAM roles for optimization.

**Response:**
```json
{
  "success": true,
  "custom_roles": [
    {
      "name": "projects/my-project/roles/customRole1",
      "permissions": ["storage.objects.get", "storage.objects.list"],
      "usage": "ACTIVE",
      "optimization": "Can be replaced with predefined role",
      "risk_score": 25
    }
  ]
}
```

### 14. MSA (Monthly Service Announcements) Analysis

#### GET /api/v1/msa/impact-analysis/{project_id}
Analyze impact of Monthly Service Announcements on your project.

**Response:**
```json
{
  "success": true,
  "msa_impacts": [
    {
      "msa_id": "2024-08-bigquery-permissions",
      "title": "BigQuery datasets.get permission split",
      "impact_level": "HIGH",
      "affected_roles": ["projects/my-project/roles/customBigQueryRole"],
      "remediation": "Update custom role to include new permission"
    }
  ]
}
```

### 15. Data Refresh & Caching

#### POST /api/data/refresh
Trigger comprehensive data refresh for a project.

**Request Body:**
```json
{
  "project_id": "my-project-123",
  "force_refresh": false,
  "fetch_types": ["compute", "storage", "security"]
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "my-project-123_20250908_120000",
  "project_id": "my-project-123",
  "message": "Data refresh started in background",
  "status_url": "/api/v1/data/refresh/status/my-project-123_20250908_120000"
}
```

#### GET /api/data/refresh/status/{job_id}
Check refresh job status.

**Response:**
```json
{
  "status": "completed",
  "started_at": "2025-09-08T12:00:00Z",
  "completed_at": "2025-09-08T12:05:30Z",
  "project_id": "my-project-123",
  "result": {
    "compute_instances": 15,
    "storage_buckets": 8,
    "security_findings": 12
  }
}
```

#### GET /api/data/assets/{project_id}
Get cached assets with fast local query.

**Query Parameters:**
- `asset_type`: Filter by asset type (optional)
- `limit`: Maximum results to return (default: 100)

**Response:**
```json
{
  "success": true,
  "project_id": "my-project-123",
  "assets": [
    {
      "name": "instance-1",
      "asset_type": "compute.googleapis.com/Instance",
      "location": "us-central1-a",
      "state": "RUNNING",
      "resource_data": {
        "machine_type": "n1-standard-1",
        "internal_ip": "10.0.0.5",
        "external_ip": "34.123.45.67"
      }
    }
  ],
  "total_count": 15,
  "from_cache": true,
  "fast_query": true
}
```

#### POST /api/data/warmup/{project_id}
Warm up cache with essential data (lighter than full refresh).

#### DELETE /api/data/cache/{project_id}
Clear cached data for a project.

### 16. Networking & Connectivity

#### POST /api/v1/networking/connectivity/test
Test network connectivity between resources.

**Request Body:**
```json
{
  "source": "projects/my-project/zones/us-central1-a/instances/vm-1",
  "destination": "10.0.1.100",
  "port": 443,
  "protocol": "TCP"
}
```

#### GET /api/v1/vpc-errors/{project_id}
Analyze VPC configuration errors.

### 17. Support & Feedback

#### POST /api/v1/feedback
Submit feedback on agent responses.

**Request Body:**
```json
{
  "session_id": "session_123",
  "query": "Original query",
  "response": "Agent response",
  "rating": 4,
  "feedback": "Very helpful, but could be more specific"
}
```

#### GET /api/v1/support-tickets/{project_id}
List Google Cloud support tickets related to security.

### 18. Statistics & Analytics

#### GET /api/v1/statistics/usage
Get API usage statistics.

#### GET /api/v1/statistics/security-trends/{project_id}
Get security trend analysis over time.

## WebSocket Connections

### Real-time Chat Streaming

#### WS /api/v1/agent/ws
Real-time streaming chat interface for token-by-token responses.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/agent/ws');

// Send chat message
ws.send(JSON.stringify({
  "type": "chat",
  "query": "What are my security risks?",
  "session_id": "demo",
  "user_id": "user_123"
}));

// Receive streaming tokens
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'token') {
    console.log('Token:', data.content);
  } else if (data.type === 'complete') {
    console.log('Response complete');
  }
};

// Handle connection events
ws.onopen = () => console.log('WebSocket connected');
ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = () => console.log('WebSocket disconnected');
```

#### WS /api/v1/ws/chat/{connection_id}
Alternative WebSocket endpoint with connection ID.

#### GET /api/v1/ws/stats
WebSocket connection statistics.

**Response:**
```json
{
  "active_connections": 5,
  "total_messages": 1250,
  "average_response_time_ms": 850,
  "uptime_seconds": 86400
}
```

#### GET /api/v1/ws/health
WebSocket service health check.

## Error Handling

### Standard Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Project not found or insufficient permissions",
    "details": {
      "project_id": "invalid-project",
      "required_permissions": ["cloudasset.assets.searchAllResources"]
    }
  }
}
```

### Common Error Codes

- `AUTHENTICATION_ERROR` - Invalid or missing credentials
- `PERMISSION_DENIED` - Insufficient IAM permissions  
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INVALID_REQUEST` - Malformed request data
- `INTERNAL_ERROR` - Server-side error
- `SERVICE_UNAVAILABLE` - Dependent service unavailable

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```bash
   # Check service account permissions
   gcloud projects get-iam-policy PROJECT_ID
   
   # Verify credentials file
   echo $GOOGLE_APPLICATION_CREDENTIALS
   ```

2. **Rate Limiting**
   ```bash
   # Check current limits
   curl http://localhost:8000/api/v1/rate-limit/status
   ```

3. **Database Connection Issues**
   ```bash
   # Check database path
   sqlite3 $DATABASE_PATH ".tables"
   ```

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python run_backend.py
```

### Health Diagnostics

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed system status  
curl http://localhost:8000/status

# Comprehensive health monitoring
curl http://localhost:8000/api/v1/health/comprehensive
```

## SDK Examples

### Python SDK Usage

```python
import httpx

class SecurityAgentClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def chat(self, query, session_id="default"):
        response = await self.client.post(
            f"{self.base_url}/api/v1/chat/message",
            json={
                "query": query,
                "session_id": session_id
            }
        )
        return response.json()
    
    async def get_security_findings(self, project_id, severity=None):
        params = {"severity": severity} if severity else {}
        response = await self.client.get(
            f"{self.base_url}/api/v1/security/findings/{project_id}",
            params=params
        )
        return response.json()

# Usage
client = SecurityAgentClient()
result = await client.chat("What are my security vulnerabilities?")
findings = await client.get_security_findings("my-project", "HIGH")
```

### JavaScript SDK Usage

```javascript
class SecurityAgentClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async chat(query, sessionId = 'default') {
        const response = await fetch(`${this.baseUrl}/api/v1/chat/message`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                session_id: sessionId
            })
        });
        return response.json();
    }
    
    async getAssets(projectId) {
        const response = await fetch(
            `${this.baseUrl}/api/v1/assets/discover/${projectId}`
        );
        return response.json();
    }
}

// Usage
const client = new SecurityAgentClient();
const result = await client.chat('Show me my GCP resources');
const assets = await client.getAssets('my-project');
```



## Deployment Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | - | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | - | Service account key path |
| `DATABASE_PATH` | No | `backend/cache/gcp_data.db` | SQLite database path |
| `BACKEND_URL` | No | `http://localhost:8000` | Backend API URL |
| `BACKEND_PORT` | No | `8000` | Backend server port |
| `DATA_REFRESH_INTERVAL` | No | `1800` | Cache refresh interval (seconds) |
| `RATE_LIMIT_CHAT` | No | `30` | Chat rate limit (requests/minute) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  security-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_CLOUD_PROJECT=my-project
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    volumes:
      - ./credentials.json:/app/credentials.json:ro
      - ./cache:/app/cache
```

### Cloud Run Deployment

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/security-agent
gcloud run deploy security-agent \
  --image gcr.io/PROJECT_ID/security-agent \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=PROJECT_ID
```

## Changelog

### Version 1.13.0 (Latest) - Integration Fixes Release
- ✅ **NEW**: Enhanced health endpoint with version field and detailed component status
- ✅ **NEW**: Background data refresh API at `/api/data/refresh` with job tracking
- ✅ **NEW**: Fast cached asset queries at `/api/data/assets/{project_id}`
- ✅ **NEW**: WebSocket endpoints for real-time streaming chat
- ✅ **FIXED**: WebSocket connection handling and error recovery
- ✅ **FIXED**: SecurityDashboard constructor compatibility issues
- ✅ **FIXED**: Background cache refresh "list index out of range" errors
- ✅ **IMPROVED**: Comprehensive health monitoring with fallback mechanisms
- ✅ **IMPROVED**: Real-time streaming chat interface with token-by-token responses
- ✅ **IMPROVED**: MSA impact analysis with database integration
- ✅ **IMPROVED**: Advanced IAM features with cross-project analysis
- ✅ **IMPROVED**: Knowledge base integration with persistent storage
- ✅ **IMPROVED**: Performance optimizations and caching strategies

### Breaking Changes in v1.13.0
- Health endpoint now includes `version` field
- Data refresh moved to `/api/data/refresh` (was `/api/v1/data/refresh/{project_id}`)
- WebSocket endpoint structure updated for better connection management
- SecurityDashboard constructor now requires database path parameter

### Version 1.12.0
- Added custom roles analysis
- Implemented feedback system
- Enhanced error handling
- Added networking diagnostics

### Version 1.11.0
- Initial API release
- Basic security scanning
- Asset discovery
- IAM analysis

---

*API Documentation Last Updated: September 8, 2025*  
*Version: 1.13.0*  
*Status: Production Ready*