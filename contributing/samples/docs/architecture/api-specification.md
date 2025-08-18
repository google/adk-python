# GCP Security Agent - API Specification

## 1. API Overview

### 1.1 Base Information
- **API Version**: v1
- **Base URL**: `http://localhost:8000/api/v1` (local) / `https://your-service.run.app/api/v1` (cloud)
- **Protocol**: HTTP/HTTPS
- **Data Format**: JSON
- **Authentication**: Google Cloud IAM / Application Default Credentials
- **API Standard**: OpenAPI 3.0.3

### 1.2 API Design Principles
- **RESTful Architecture**: Resource-based URLs with standard HTTP methods
- **Consistent Response Format**: Standardized error handling and response structure
- **Stateless Communication**: Each request contains all necessary information
- **Idempotent Operations**: Safe retry behavior for all operations
- **Versioned API**: Backward compatibility through versioning

### 1.3 Content Types
- **Request Content-Type**: `application/json`
- **Response Content-Type**: `application/json`
- **WebSocket**: `application/json` over WebSocket protocol

## 2. Authentication and Authorization

### 2.1 Authentication Methods

#### 2.1.1 Google Cloud IAM (Production)
```http
Authorization: Bearer <access_token>
```

#### 2.1.2 Application Default Credentials (Local Development)
```bash
gcloud auth application-default login
```

#### 2.1.3 Service Account Key
```json
{
  "type": "service_account",
  "project_id": "your-project",
  "private_key_id": "key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "service-account@project.iam.gserviceaccount.com"
}
```

### 2.2 Required Permissions
```yaml
required_roles:
  - roles/cloudasset.viewer
  - roles/compute.viewer
  - roles/storage.objectViewer
  - roles/iam.securityReviewer
  - roles/recommender.viewer
  - roles/monitoring.viewer
```

### 2.3 API Key (Optional)
For enhanced rate limiting and analytics:
```http
X-API-Key: your-api-key
```

## 3. Rate Limiting and Quotas

### 3.1 Rate Limits
- **Global Rate Limit**: 1000 requests/hour per authenticated user
- **Asset Discovery**: 100 requests/hour per project
- **Chat Interface**: 500 requests/hour per session
- **WebSocket**: 10 connections per user

### 3.2 Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 856
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 3600
```

### 3.3 Quota Exceeded Response
```json
{
  "error": {
    "code": 429,
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": "1 hour",
      "reset_time": "2024-01-01T12:00:00Z"
    }
  }
}
```

## 4. Standard Response Format

### 4.1 Success Response
```json
{
  "success": true,
  "data": {},
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345",
    "api_version": "v1"
  }
}
```

### 4.2 Error Response
```json
{
  "success": false,
  "error": {
    "code": 400,
    "message": "Invalid request",
    "details": {
      "field": "project_id",
      "error": "Project ID is required"
    }
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345"
  }
}
```

### 4.3 Pagination Response
```json
{
  "success": true,
  "data": [],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 150,
    "total_pages": 8,
    "has_next": true,
    "has_previous": false
  }
}
```

## 5. API Endpoints

### 5.1 Health and Status Endpoints

#### 5.1.1 Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "features": {
    "secret_manager": true,
    "adk_session_management": true,
    "websockets": true,
    "performance_monitoring": true,
    "context_awareness": true
  },
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "websocket": "/api/v1/agent/ws",
    "chat": "/api/v1/agent/chat",
    "asset_inventory": "/api/v1/asset-inventory"
  }
}
```

#### 5.1.2 API Documentation
```http
GET /docs
```

**Response:** OpenAPI/Swagger UI interface

### 5.2 Asset Inventory Endpoints

#### 5.2.1 Asset Inventory Summary
```http
GET /api/v1/asset-inventory/summary
```

**Query Parameters:**
- `project_id` (optional): GCP project ID

**Response:**
```json
{
  "success": true,
  "data": {
    "total_assets": 42,
    "asset_types": {
      "Compute Instances": 8,
      "Storage Buckets": 15,
      "IAM Accounts": 12,
      "Networks": 4,
      "Databases": 3
    },
    "security_findings": 7,
    "high_risk_assets": 3,
    "active_recommendations": 5
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 5.2.2 Natural Language Asset Discovery
```http
POST /api/v1/asset-inventory/discover
```

**Request Body:**
```json
{
  "query": "show me my compute instances",
  "project_id": "mgm-digitalconcierge",
  "include_security_analysis": true,
  "include_cost_analysis": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "query_type": "compute_instances",
    "assets": [
      {
        "name": "mgm-web-server-01",
        "type": "compute.googleapis.com/Instance",
        "zone": "us-central1-a",
        "machine_type": "e2-medium",
        "status": "RUNNING",
        "security_findings": [
          {
            "severity": "MEDIUM",
            "description": "Instance missing OS security patches"
          }
        ]
      }
    ],
    "total_found": 2,
    "security_summary": {
      "high_risk": 0,
      "medium_risk": 1,
      "low_risk": 1
    }
  },
  "api_calls_made": [
    {
      "service": "cloudasset.googleapis.com",
      "method": "searchAllResources",
      "timestamp": "2024-01-01T12:00:00Z"
    }
  ]
}
```

#### 5.2.3 Get Compute Instances
```http
GET /api/v1/asset-inventory/compute/instances
```

**Query Parameters:**
- `project_id` (optional): GCP project ID
- `zone` (optional): Specific zone filter
- `status` (optional): Instance status filter

**Response:**
```json
{
  "success": true,
  "data": {
    "instances": [
      {
        "name": "mgm-web-server-01",
        "id": "1234567890123456789",
        "zone": "us-central1-a",
        "machine_type": "e2-medium",
        "status": "RUNNING",
        "creation_timestamp": "2024-01-01T10:00:00Z",
        "network_interfaces": [
          {
            "name": "nic0",
            "network": "projects/mgm-digitalconcierge/global/networks/default",
            "subnetwork": "projects/mgm-digitalconcierge/regions/us-central1/subnetworks/default"
          }
        ],
        "disks": [
          {
            "device_name": "persistent-disk-0",
            "source": "projects/mgm-digitalconcierge/zones/us-central1-a/disks/mgm-web-server-01",
            "boot": true
          }
        ],
        "security_analysis": {
          "os_login_enabled": false,
          "shielded_vm_enabled": true,
          "firewall_rules": 5,
          "public_ip": true,
          "recommendations": [
            "Enable OS Login for centralized SSH key management",
            "Review firewall rules for least privilege access"
          ]
        }
      }
    ],
    "total_instances": 2
  }
}
```

#### 5.2.4 Get Storage Buckets
```http
GET /api/v1/asset-inventory/storage/buckets
```

**Response:**
```json
{
  "success": true,
  "data": {
    "buckets": [
      {
        "name": "mgm-data-lake-raw",
        "location": "US",
        "storage_class": "STANDARD",
        "creation_time": "2024-01-01T09:00:00Z",
        "updated_time": "2024-01-01T11:30:00Z",
        "lifecycle_rules": [],
        "versioning_enabled": false,
        "security_analysis": {
          "public_access": true,
          "uniform_bucket_level_access": false,
          "encryption": "Google-managed",
          "iam_policies": 3,
          "recommendations": [
            "Enable uniform bucket-level access",
            "Review and restrict public access",
            "Enable versioning for data protection"
          ]
        },
        "size_gb": 150.5,
        "object_count": 1250
      }
    ],
    "total_buckets": 10
  }
}
```

#### 5.2.5 Security Asset Analysis
```http
GET /api/v1/asset-inventory/security/analyze
```

**Response:**
```json
{
  "success": true,
  "data": {
    "security_overview": {
      "total_assets_analyzed": 42,
      "high_risk_assets": 3,
      "medium_risk_assets": 8,
      "low_risk_assets": 31,
      "security_score": 72
    },
    "high_risk_findings": [
      {
        "asset_name": "mgm-data-lake-raw",
        "asset_type": "storage.googleapis.com/Bucket",
        "finding": "Bucket has public read access",
        "severity": "HIGH",
        "impact": "Data exposure risk",
        "remediation": "Remove public access and implement IAM-based access control"
      }
    ],
    "recommendations": [
      {
        "id": "rec_001",
        "title": "Enable uniform bucket-level access for all storage buckets",
        "priority": "HIGH",
        "affected_assets": 10,
        "estimated_effort": "LOW",
        "compliance_frameworks": ["SOC2", "ISO27001"]
      }
    ],
    "compliance_status": {
      "SOC2": {
        "compliant": 35,
        "non_compliant": 7,
        "compliance_percentage": 83.3
      }
    }
  }
}
```

### 5.3 Chat Interface Endpoints

#### 5.3.1 Chat Message Processing
```http
POST /api/v1/agent/chat
```

**Request Body:**
```json
{
  "query": "tell me about the buckets in the project",
  "user_id": "user123",
  "project_id": "mgm-digitalconcierge",
  "session_id": "session_456",
  "context": {
    "topic": "storage_analysis",
    "entities": ["buckets", "security"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "response": "I found 10 storage buckets in your project. Here's a security analysis:\n\n**High Risk Buckets:**\n• mgm-data-lake-raw - Has public read access\n\n**Recommendations:**\n• Enable uniform bucket-level access\n• Review IAM policies for least privilege",
  "agent_used": "AssetDiscoveryAgent",
  "session_id": "session_456",
  "suggestions": [
    "Show me more details about the high-risk bucket",
    "What are the recommended IAM policies?",
    "Analyze other storage security settings"
  ],
  "performance_metrics": {
    "response_time_ms": 1250,
    "agent_processing_time_ms": 800,
    "api_calls_made": 2
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_789"
  }
}
```

#### 5.3.2 WebSocket Chat Interface
```websocket
WebSocket /api/v1/agent/ws
```

**Connection Parameters:**
- `user_id`: User identifier
- `session_id`: Session identifier
- `project_id`: GCP project ID

**Message Format:**
```json
{
  "type": "chat_message",
  "data": {
    "query": "analyze my IAM permissions",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

**Response Format:**
```json
{
  "type": "chat_response",
  "data": {
    "response": "Analysis complete...",
    "agent_used": "SecurityAgent",
    "suggestions": ["..."],
    "timestamp": "2024-01-01T12:00:05Z"
  }
}
```

#### 5.3.3 Follow-up Suggestions
```http
GET /api/v1/agent/suggestions
```

**Query Parameters:**
- `session_id`: Session identifier
- `last_query`: Previous query for context

**Response:**
```json
{
  "success": true,
  "suggestions": [
    "Show me more details about the high-risk assets",
    "What are the recommended security improvements?",
    "Analyze my network security configuration",
    "Check my IAM policy compliance"
  ],
  "context": {
    "topic": "security_analysis",
    "last_agent": "SecurityAgent"
  }
}
```

### 5.4 Session Management Endpoints

#### 5.4.1 Create Session
```http
POST /api/v1/sessions/create
```

**Request Body:**
```json
{
  "user_id": "user123",
  "project_id": "mgm-digitalconcierge",
  "metadata": {
    "client_type": "streamlit_thin_client",
    "adk_compliant": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "session_456",
  "created_at": "2024-01-01T12:00:00Z",
  "expires_at": "2024-01-01T20:00:00Z",
  "status": "active"
}
```

#### 5.4.2 Get Session Status
```http
GET /api/v1/sessions/{session_id}/status
```

**Response:**
```json
{
  "success": true,
  "session_id": "session_456",
  "status": "active",
  "created_at": "2024-01-01T12:00:00Z",
  "last_activity": "2024-01-01T12:15:00Z",
  "analytics": {
    "total_messages": 15,
    "total_queries": 8,
    "agents_used": ["SecurityAgent", "AssetDiscoveryAgent"],
    "average_response_time": 2.3
  }
}
```

#### 5.4.3 Get Session Messages
```http
GET /api/v1/sessions/{session_id}/messages
```

**Query Parameters:**
- `limit` (optional): Number of messages to return (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "messages": [
    {
      "id": "msg_001",
      "session_id": "session_456",
      "sender_type": "user",
      "content": "tell me about my buckets",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "id": "msg_002",
      "session_id": "session_456",
      "sender_type": "assistant",
      "content": "I found 10 storage buckets...",
      "agent_used": "AssetDiscoveryAgent",
      "timestamp": "2024-01-01T12:00:05Z"
    }
  ],
  "total_messages": 15
}
```

### 5.5 Recommendations Endpoints

#### 5.5.1 Get Recommendations
```http
GET /api/v1/recommendations
```

**Query Parameters:**
- `project_id` (optional): GCP project ID
- `priority` (optional): Filter by priority (LOW, MEDIUM, HIGH, CRITICAL)
- `category` (optional): Filter by category
- `status` (optional): Filter by status

**Response:**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "id": "rec_001",
        "title": "Enable uniform bucket-level access",
        "description": "Uniform bucket-level access provides better security and consistency for bucket access control.",
        "category": "storage_security",
        "priority": "HIGH",
        "implementation_effort": "LOW",
        "cost_impact": "NONE",
        "affected_assets": [
          "mgm-data-lake-raw",
          "mgm-data-lake-processed"
        ],
        "implementation_steps": [
          "Navigate to Cloud Storage in the GCP Console",
          "Select each bucket individually",
          "Go to Permissions tab",
          "Enable 'Uniform bucket-level access'"
        ],
        "status": "NEW",
        "created_date": "2024-01-01T12:00:00Z",
        "compliance_frameworks": ["SOC2", "ISO27001"]
      }
    ],
    "summary": {
      "total_recommendations": 12,
      "by_priority": {
        "CRITICAL": 1,
        "HIGH": 3,
        "MEDIUM": 5,
        "LOW": 3
      },
      "by_status": {
        "NEW": 8,
        "IN_PROGRESS": 3,
        "COMPLETED": 1
      }
    }
  }
}
```

#### 5.5.2 Update Recommendation Status
```http
POST /api/v1/recommendations/{recommendation_id}/status
```

**Request Body:**
```json
{
  "status": "IN_PROGRESS",
  "notes": "Started implementation on 2024-01-01",
  "assigned_to": "security-team@company.com"
}
```

**Response:**
```json
{
  "success": true,
  "recommendation": {
    "id": "rec_001",
    "status": "IN_PROGRESS",
    "updated_date": "2024-01-01T12:30:00Z",
    "notes": "Started implementation on 2024-01-01",
    "assigned_to": "security-team@company.com"
  }
}
```

#### 5.5.3 Get Prioritized Recommendations
```http
GET /api/v1/recommendations/prioritized
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prioritized_recommendations": [
      {
        "rank": 1,
        "recommendation": {
          "id": "rec_003",
          "title": "Remediate public bucket access",
          "priority": "CRITICAL",
          "security_impact": 9.5,
          "implementation_effort": "LOW"
        }
      }
    ],
    "prioritization_criteria": {
      "security_impact": 40,
      "compliance_requirement": 30,
      "implementation_effort": 20,
      "cost_impact": 10
    }
  }
}
```

## 6. Error Codes and Handling

### 6.1 HTTP Status Codes
- `200 OK` - Success
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### 6.2 Error Response Format
```json
{
  "success": false,
  "error": {
    "code": 400,
    "type": "VALIDATION_ERROR",
    "message": "Invalid project ID format",
    "details": {
      "field": "project_id",
      "value": "invalid-project",
      "expected_format": "lowercase letters, numbers, and hyphens"
    },
    "help_url": "https://docs.example.com/api/errors/validation"
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345",
    "trace_id": "trace_67890"
  }
}
```

### 6.3 Common Error Types
- `VALIDATION_ERROR` - Request validation failed
- `AUTHENTICATION_ERROR` - Authentication failed
- `AUTHORIZATION_ERROR` - Insufficient permissions
- `RESOURCE_NOT_FOUND` - Requested resource not found
- `RATE_LIMIT_EXCEEDED` - Rate limit exceeded
- `GCP_API_ERROR` - Google Cloud API error
- `INTERNAL_ERROR` - Internal server error

## 7. Performance Considerations

### 7.1 Response Time Targets
- **Asset queries**: < 2 seconds
- **Chat responses**: < 5 seconds
- **Session operations**: < 1 second
- **Recommendations**: < 3 seconds

### 7.2 Caching Strategy
- **Asset data**: 5-minute cache TTL
- **Recommendations**: 15-minute cache TTL
- **Session data**: In-memory with Redis backup
- **API responses**: HTTP cache headers

### 7.3 Pagination
All list endpoints support pagination:
```http
GET /api/v1/recommendations?page=2&page_size=20
```

## 8. SDK and Client Libraries

### 8.1 Python Client Example
```python
from gcp_security_agent import SecurityAgentClient

client = SecurityAgentClient(
    base_url="http://localhost:8000",
    project_id="mgm-digitalconcierge"
)

# Discover assets
assets = client.discover_assets("show me my compute instances")

# Chat interface
response = client.chat("analyze my security posture")

# Get recommendations
recommendations = client.get_recommendations(priority="HIGH")
```

### 8.2 cURL Examples
```bash
# Get asset summary
curl -X GET "http://localhost:8000/api/v1/asset-inventory/summary" \
  -H "Content-Type: application/json"

# Chat query
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "show me my storage buckets",
    "user_id": "user123",
    "project_id": "mgm-digitalconcierge"
  }'
```

This API specification provides comprehensive documentation for all endpoints, request/response formats, authentication requirements, and error handling patterns used in the GCP Security Agent system.