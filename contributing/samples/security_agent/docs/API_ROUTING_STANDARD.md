# API Routing Standard - ADK Security Agent

## Base URL Structure
All API endpoints follow this pattern:
```
http://localhost:8000/api/v1/{service}/{resource}
```

## Service Routing Map

### 1. Agent Service (`/api/v1/agent`)
**Router:** `backend/api/agent.py`
**Prefix:** `/api/v1` (Note: agent endpoints are at root of v1)
```
POST /api/v1/agent/chat           - Chat with AI agent
GET  /api/v1/agent/ws             - WebSocket connection
GET  /api/v1/sessions             - List sessions
POST /api/v1/sessions/{user_id}   - Create session
GET  /api/v1/conversations        - List conversations
```

### 2. GCP Service (`/api/v1/gcp`)
**Router:** `backend/api/gcp.py`
**Prefix:** `/api/v1/gcp`
```
GET  /api/v1/gcp/projects                    - List all projects
GET  /api/v1/gcp/projects/{project_id}       - Get project details
GET  /api/v1/gcp/projects/{project_id}/services - List enabled services
POST /api/v1/gcp/call                        - Generic API call
GET  /api/v1/gcp/discovery/apis              - Discover available APIs
```

### 3. Security Service (`/api/v1/security`)
**Router:** `backend/api/security.py`
**Prefix:** `/api/v1/security`
```
POST /api/v1/security/evaluate               - Evaluate security posture
GET  /api/v1/security/score                  - Get security score
GET  /api/v1/security/findings               - List security findings
GET  /api/v1/security/enabled-apis           - List enabled security APIs
POST /api/v1/security/vulnerability/evaluate - Evaluate vulnerabilities
```

### 4. IAM Service (`/api/v1/iam`)
**Router:** `backend/api/iam.py`
**Prefix:** `/api/v1/iam`
```
GET  /api/v1/iam/project/{project_id}/analyze-user/{user_email} - Analyze user permissions
GET  /api/v1/iam/project/{project_id}/analyze-all-users        - Analyze all users
GET  /api/v1/iam/project/{project_id}/policy                   - Get IAM policy
```

### 5. Compliance Service (`/api/v1/compliance`)
**Router:** `backend/api/compliance.py`
**Prefix:** `/api/v1/compliance`
```
POST /api/v1/compliance/evaluate   - Evaluate compliance
GET  /api/v1/compliance/frameworks - List compliance frameworks
```

### 6. Monitoring Service (`/api/v1/monitoring`)
**Router:** `backend/api/monitoring.py`
**Prefix:** `/api/v1/monitoring`
```
GET  /api/v1/monitoring/summary                     - Get monitoring summary
GET  /api/v1/monitoring/logs/{project_id}          - Get logs
GET  /api/v1/monitoring/metrics/{project_id}       - Get metrics
GET  /api/v1/monitoring/traces/{project_id}        - Get traces
GET  /api/v1/monitoring/dashboard/{project_id}     - Dashboard data
```

### 7. Recommendations Service (`/api/v1/recommendations`)
**Router:** `backend/api/recommendations.py`
**Prefix:** `/api/v1/recommendations`
```
POST /api/v1/recommendations/dashboard           - Get dashboard recommendations
GET  /api/v1/recommendations/priority/{priority} - Get by priority
```

### 8. MSA Service (`/api/v1/msa`)
**Router:** `backend/api/msa.py`
**Prefix:** `/api/v1/msa`
```
POST /api/v1/msa/parse           - Parse MSA document
GET  /api/v1/msa/records         - Get MSA records
POST /api/v1/msa/impact-analysis - Analyze MSA impact
```

### 9. Performance Service (`/api/v1/performance`)
**Router:** `backend/api/performance_monitor.py`
**Prefix:** `/api/v1/performance`
```
GET  /api/v1/performance/metrics  - Get performance metrics
GET  /api/v1/performance/status   - Get performance status
```

### 10. Context Service (`/api/v1/context`)
**Router:** `backend/api/context_manager.py`
**Prefix:** `/api/v1/context`
```
GET  /api/v1/context/user/{user_id}        - Get user context
POST /api/v1/context/analyze               - Analyze context
GET  /api/v1/context/suggestions/{user_id} - Get suggestions
```

## Router Registration in `backend/main.py`

```python
# Router registration order and prefixes
app.include_router(agent_router, prefix="/api/v1")           # Agent at root
app.include_router(gcp_router, prefix="/api/v1/gcp")         # GCP service
app.include_router(security_router, prefix="/api/v1/security") # Security service
app.include_router(monitoring_router, prefix="/api/v1/monitoring") # Monitoring
app.include_router(iam_router, prefix="/api/v1/iam")         # IAM service
app.include_router(compliance_router, prefix="/api/v1/compliance") # Compliance
app.include_router(recommendations_router, prefix="/api/v1/recommendations") # Recommendations
app.include_router(msa_router, prefix="/api/v1/msa")         # MSA service
app.include_router(performance_router, prefix="/api/v1/performance") # Performance
app.include_router(context_router, prefix="/api/v1/context") # Context
```

## Frontend API Client Paths

The frontend uses these exact paths in `frontend/api_client_consolidated.py`:

```python
# Core endpoints used by frontend
GET  /api/v1/gcp/projects
GET  /api/v1/security/score
POST /api/v1/recommendations/dashboard
POST /api/v1/agent/chat
GET  /api/v1/security/findings
GET  /api/v1/security/enabled-apis
GET  /api/v1/monitoring/summary
GET  /api/v1/gcp/projects/{project_id}
GET  /api/v1/iam/project/{project_id}/analyze-all-users
```

## Important Notes

1. **Agent Router Exception**: The agent router is included at `/api/v1` (not `/api/v1/agent`) because its endpoints already include `/agent` in their paths.

2. **Path Pattern**: All other services follow the pattern `/api/v1/{service_name}/{resource}`

3. **Project ID**: Most endpoints accept `project_id` as either:
   - Path parameter: `/api/v1/gcp/projects/{project_id}`
   - Query parameter: `?project_id=mgm-digitalconcierge`

4. **Response Format**: All endpoints return JSON with this structure:
   ```json
   {
     "success": true,
     "data": {...},
     "error": null
   }
   ```

5. **Error Handling**: All errors return appropriate HTTP status codes:
   - 200: Success
   - 400: Bad Request
   - 404: Not Found
   - 500: Internal Server Error

## Testing Endpoints

To test if an endpoint is working:
```bash
# Test GCP projects endpoint
curl http://localhost:8000/api/v1/gcp/projects

# Test security score
curl http://localhost:8000/api/v1/security/score

# Test monitoring summary
curl http://localhost:8000/api/v1/monitoring/summary
```

## Troubleshooting

1. **404 Not Found**: Check that the router is registered in `backend/main.py` with the correct prefix
2. **500 Internal Error**: Check that the service class has the required method
3. **Import Error**: Ensure the router file exists and exports `router`
4. **Method Not Allowed**: Verify HTTP method (GET/POST/PUT/DELETE) matches the endpoint definition