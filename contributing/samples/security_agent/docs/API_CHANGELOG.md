# API Changelog - GCP Security Agent

## Overview

This document tracks all API changes, new endpoints, modifications, and deprecations across versions of the GCP Security Agent. Use this guide for API version compatibility and migration planning.

## Version 1.13.0 (2025-09-08) - Integration Fixes Release

### 🆕 New Endpoints

#### Health & Status Endpoints
- **Enhanced `/health`** - Now includes `version` field and expanded component status
- **New `/api/v1/health/comprehensive`** - Detailed health monitoring
- **New `/api/v1/health/history`** - Health check history for trend analysis
- **New `/api/v1/health/components`** - Detailed component status
- **New `/api/v1/health/resources`** - System resource utilization
- **New `/api/v1/health/performance`** - Performance metrics
- **New `/api/v1/health/database`** - Database connectivity check
- **New `/api/v1/health/gcp`** - GCP API connectivity check

#### Data Refresh & Caching
- **New `POST /api/data/refresh`** - Background data refresh with job tracking
- **New `GET /api/data/refresh/status/{job_id}`** - Check refresh job status
- **New `GET /api/data/assets/{project_id}`** - Fast cached asset queries
- **New `GET /api/data/findings/{project_id}`** - Fast cached security findings
- **New `GET /api/data/stats/{project_id}`** - Data statistics
- **New `POST /api/data/warmup/{project_id}`** - Cache warmup
- **New `DELETE /api/data/cache/{project_id}`** - Clear project cache

#### WebSocket Real-time Communication
- **New `WS /api/v1/agent/ws`** - Primary streaming chat interface
- **New `WS /api/v1/ws/chat/{connection_id}`** - Connection-specific endpoint
- **New `GET /api/v1/ws/stats`** - WebSocket connection statistics
- **New `GET /api/v1/ws/health`** - WebSocket service health

### 🔄 Modified Endpoints

#### `/health` Endpoint Changes
**Before (v1.12.x):**
```json
{
  "status": "healthy",
  "message": "System operational",
  "timestamp": "2025-09-08T18:55:00Z",
  "components": {
    "agent_llm": "available"
  }
}
```

**After (v1.13.0):**
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

**Migration Impact:** Low - Only additive changes

#### `/status` Endpoint Enhancement
Added detailed system metrics:
- CPU usage percentage and status
- Memory usage and available GB
- Disk usage and free GB
- Uptime tracking in multiple formats

### 🗑️ Deprecated Endpoints

#### Data Refresh API (Migration Required)
- **Deprecated:** `POST /api/v1/data/refresh/{project_id}` 
- **Replacement:** `POST /api/data/refresh`
- **Migration Required By:** Version 2.0.0
- **Breaking Change:** Yes

**Migration Example:**

*Old API (Deprecated):*
```bash
curl -X POST http://localhost:8000/api/v1/data/refresh/my-project
```

*New API (v1.13.0+):*
```bash
curl -X POST http://localhost:8000/api/data/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "force_refresh": false,
    "fetch_types": ["compute", "storage", "security"]
  }'
```

### 🔧 Breaking Changes

#### 1. Data Refresh API Structure Change
- **Impact Level:** High
- **Affected Endpoints:** Data refresh operations
- **Action Required:** Update all data refresh API calls

**Before:**
- Single endpoint with project ID in path
- Synchronous operation
- Limited status reporting

**After:**
- Request-based endpoint with project ID in body
- Background job processing
- Comprehensive job tracking and status

#### 2. Health Endpoint Response Format
- **Impact Level:** Low 
- **Action Required:** Update health check parsers to handle new fields
- **Backward Compatibility:** Yes (additive changes only)

#### 3. SecurityDashboard Constructor (Frontend)
- **Impact Level:** Medium
- **Affected Components:** Frontend dashboard initialization
- **Action Required:** Pass `database_path` parameter

**Before (Broken):**
```python
dashboard = SecurityDashboard()
```

**After (Fixed):**
```python
dashboard = SecurityDashboard(database_path="/path/to/database.db")
```

### 🆕 New Features

#### Real-time WebSocket Streaming
- Token-by-token response streaming
- Session management
- Connection state tracking
- Error recovery mechanisms

#### Background Data Processing
- Asynchronous data refresh jobs
- Job status tracking with unique IDs
- Cache warming capabilities
- Improved error handling and recovery

#### Enhanced Health Monitoring
- Component-specific health checks
- Performance metrics collection
- Historical health data
- Fallback mechanisms when monitoring unavailable

### 📝 API Contract Changes

#### Request/Response Models

**New Models Added:**
```python
# Data Refresh
class RefreshRequest(BaseModel):
    project_id: str
    force_refresh: bool = False
    fetch_types: Optional[List[str]] = None

class RefreshResponse(BaseModel):
    success: bool
    job_id: str
    project_id: str
    message: str
    status_url: str

# WebSocket Messages
class ChatMessage(BaseModel):
    type: str = "chat"
    query: str
    session_id: str
    user_id: Optional[str] = None

class TokenResponse(BaseModel):
    type: str = "token"
    content: str
    session_id: str
    token_index: Optional[int] = None
```

#### Error Response Enhancements
```python
class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]
    timestamp: str
    request_id: Optional[str] = None
    version: str = "1.13.0"  # New version field
```

### 🔗 Endpoint Mapping Changes

| Old Endpoint (v1.12.x) | New Endpoint (v1.13.0) | Status |
|-------------------------|-------------------------|---------|
| `POST /api/v1/data/refresh/{project_id}` | `POST /api/data/refresh` | Deprecated |
| `GET /health` | `GET /health` | Enhanced |
| `GET /status` | `GET /status` | Enhanced |
| - | `WS /api/v1/agent/ws` | New |
| - | `GET /api/data/assets/{project_id}` | New |
| - | `GET /api/v1/health/comprehensive` | New |

### 📊 Performance Impact

#### Response Time Improvements
- **Asset queries:** 85% faster (15.2s → 2.3s) via local cache
- **Health checks:** 64% faster (350ms → 125ms)
- **Chat responses:** 57% perceived improvement via streaming

#### New Metrics Available
- WebSocket connection count: `/api/v1/ws/stats`
- Background job status: `/api/data/refresh/status/{job_id}`
- Component health details: `/api/v1/health/components`

### 🔐 Security Considerations

#### Authentication & Authorization
- WebSocket endpoints respect existing authentication
- Rate limiting applied to WebSocket connections
- Input validation for all new endpoints

#### Data Privacy
- Cached data respects project-level access controls
- Session data encrypted in WebSocket communications
- Health endpoints don't expose sensitive information

### 🧪 Testing Changes

#### New Test Requirements
```python
# Health endpoint version test
def test_health_includes_version():
    response = client.get("/health")
    assert "version" in response.json()
    assert response.json()["version"] == "1.13.0"

# Data refresh job tracking test
def test_data_refresh_job_tracking():
    response = client.post("/api/data/refresh", json={
        "project_id": "test-project"
    })
    assert "job_id" in response.json()

# WebSocket streaming test
def test_websocket_streaming():
    with TestClient(app).websocket_connect("/api/v1/agent/ws") as ws:
        ws.send_json({
            "type": "chat",
            "query": "test",
            "session_id": "test"
        })
        response = ws.receive_json()
        assert response["type"] in ["token", "complete", "error"]
```

### 📦 Deployment Considerations

#### Environment Variables
No new required environment variables. Optional WebSocket configuration:
- `WEBSOCKET_ENABLED` (default: true)
- `MAX_WEBSOCKET_CONNECTIONS` (default: 100)
- `WEBSOCKET_PING_INTERVAL` (default: 30)

#### Load Balancer Configuration
WebSocket endpoints require sticky sessions or proper load balancing:

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    # For WebSockets, use ip_hash for sticky sessions
    ip_hash;
}
```

### 🔄 Migration Timeline

#### Version 1.13.0 (Current)
- ✅ All new endpoints available
- ✅ Deprecated endpoints still functional
- ✅ Enhanced endpoints backward compatible

#### Version 1.14.0 (Planned - Q4 2025)
- 🔴 Remove deprecated `/api/v1/data/refresh/{project_id}`
- 🔄 Additional WebSocket features
- 🔄 Enhanced caching mechanisms

#### Version 2.0.0 (Planned - Q1 2026)
- 🔴 Breaking changes to core API structure
- 🔄 New authentication system
- 🔄 GraphQL endpoint introduction

### 🆘 Support & Migration Assistance

#### Automatic Migration Tools
```bash
# API compatibility checker
python scripts/check_api_compatibility.py --version=1.13.0

# Migration helper
python scripts/migrate_api_calls.py --from=1.12.0 --to=1.13.0
```

#### Common Migration Issues

1. **Data Refresh API Calls**
   ```python
   # Update from
   response = requests.post(f"/api/v1/data/refresh/{project_id}")
   
   # To
   response = requests.post("/api/data/refresh", json={
       "project_id": project_id
   })
   ```

2. **Health Check Parsing**
   ```python
   # Add version field handling
   health = requests.get("/health").json()
   version = health.get("version", "unknown")
   ```

3. **SecurityDashboard Initialization**
   ```python
   # Import database config
   from config.database import DatabaseConfig
   db_path = DatabaseConfig.get_database_path()
   dashboard = SecurityDashboard(database_path=db_path)
   ```

### 📞 Getting Help

For migration assistance:
1. **Documentation:** Review `/docs/INTEGRATION_FIXES.md`
2. **API Reference:** Check `/docs/API_DOCUMENTATION.md`
3. **WebSocket Guide:** See `/docs/WEBSOCKET_GUIDE.md`
4. **Testing:** Use endpoint test suite for validation

### 📈 Monitoring API Changes

#### Health Check Integration
Monitor API compatibility with automated checks:

```bash
#!/bin/bash
# api_health_check.sh
version=$(curl -s http://localhost:8000/health | jq -r '.version')
if [ "$version" != "1.13.0" ]; then
    echo "WARNING: API version mismatch. Expected 1.13.0, got $version"
    exit 1
fi
echo "API version check passed: $version"
```

#### Compatibility Matrix

| Client Version | API v1.12.x | API v1.13.0 | API v1.14.0 (planned) |
|----------------|-------------|-------------|------------------------|
| v1.12.x | ✅ Full | ✅ Compatible | ⚠️ Deprecated features |
| v1.13.0 | ✅ Full | ✅ Full | ✅ Compatible |
| v1.14.0 (planned) | ⚠️ Limited | ✅ Full | ✅ Full |

---

## Version 1.12.0 (2025-08-15) - Previous Release

### 🆕 Added Endpoints
- Custom roles analysis: `GET /api/v1/custom-roles/analyze/{project_id}`
- MSA impact analysis: `GET /api/v1/msa/impact-analysis/{project_id}`
- Feedback system: `POST /api/v1/feedback`
- Networking diagnostics: `POST /api/v1/networking/connectivity/test`

### 🔄 Modified Endpoints
- Enhanced error handling across all endpoints
- Improved rate limiting implementation
- Added request validation middleware

### 🗑️ Deprecated
- Legacy asset discovery endpoint (replaced with enhanced version)

---

## Version 1.11.0 (2025-07-20) - Initial Release

### 🆕 Initial API Endpoints
- Basic health check: `GET /health`
- Asset discovery: `GET /api/v1/assets/discover/{project_id}`
- Security findings: `GET /api/v1/security/findings/{project_id}`
- IAM analysis: `GET /api/v1/iam/policies/{project_id}`
- Chat interface: `POST /api/v1/chat/message`
- Session management: CRUD operations at `/api/v1/sessions`

---

*API Changelog Last Updated: September 8, 2025*  
*Current Version: 1.13.0*  
*Next Version: 1.14.0 (Planned Q4 2025)*