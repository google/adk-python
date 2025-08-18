# ADK Security Agent - API Reference

## 📚 Table of Contents
1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Core APIs](#core-apis)
4. [Security APIs](#security-apis)
6. [Monitoring APIs](#monitoring-apis)
7. [Integration APIs](#integration-apis)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [API Examples](#api-examples)

## 🎯 API Overview

The ADK Security Agent provides a comprehensive RESTful API with direct endpoints for security evaluation, monitoring, and management of Google Cloud Platform resources.

### Base URL
```
http://localhost:8000/api/v1
```

### API Standards
- **Protocol**: HTTP/1.1, HTTP/2
- **Format**: JSON
- **Authentication**: Bearer token (GCP Service Account)
- **Versioning**: URL path versioning (v1)
- **OpenAPI**: 3.0.3 specification available at `/docs`

### Response Format
All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {
    // Response data specific to the endpoint
  },
  "metadata": {
    "timestamp": "2025-01-08T10:30:00Z",
    "service": "security",
    "version": "1.0.0",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "error": null
}
```

### Error Response Format
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": {
      "resource_type": "project",
      "resource_id": "my-project-123"
    }
  },
  "metadata": {
    "timestamp": "2025-01-08T10:30:00Z",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

## 🔐 Authentication

### Service Account Authentication

The API uses Google Cloud service account authentication. Set up authentication using one of these methods:

#### Method 1: Application Default Credentials
```bash
# For local development
gcloud auth application-default login

# For service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Method 2: Bearer Token
```bash
# Get access token
TOKEN=$(gcloud auth print-access-token)

# Use in API calls
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/gcp/projects
```

### Required Permissions

| API Category | Required IAM Roles |
|--------------|-------------------|
| Project Management | `resourcemanager.projects.get` |
| IAM Analysis | `iam.roles.list`, `iam.policies.get` |
| Security Evaluation | `securitycenter.findings.list` |
| Logging | `logging.logEntries.list` |
| Monitoring | `monitoring.timeSeries.list` |

## 📋 Service Management APIs

### List All Services

```http
GET /api/v1/services/status/summary
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_services": 16,
    "enabled_services": 10,
    "running_services": 9,
    "disabled_services": 6,
    "error_services": 1,
    "services": {
      "security": {
        "status": "running",
        "enabled": true,
        "health": "healthy"
      },
      "iam": {
        "status": "running",
        "enabled": true,
        "health": "healthy"
      }
    }
  }
}
```

### Get Service Status

```http
GET /api/v1/services/{service_name}/status
```

**Path Parameters:**
- `service_name` (string): Name of the service

**Response:**
```json
{
  "success": true,
  "data": {
    "service_name": "security",
    "status": "running",
    "enabled": true,
    "initialized": true,
    "health": {
      "healthy": true,
      "last_check": "2025-01-08T10:30:00Z",
      "checks": {
        "api_connectivity": "pass",
        "authentication": "pass",
        "dependencies": "pass"
      }
    },
    "dependencies": ["gcp"],
    "configuration": {
      "scan_depth": "comprehensive",
      "cache_ttl": 300
    }
  }
}
```

### Enable Service

```http
POST /api/v1/services/{service_name}/enable
```

**Response:**
```json
{
  "success": true,
  "data": {
    "service_name": "threat_intelligence",
    "previous_status": "disabled",
    "new_status": "running",
    "initialization_time_ms": 1250
  }
}
```

### Disable Service

```http
POST /api/v1/services/{service_name}/disable
```

**Response:**
```json
{
  "success": true,
  "data": {
    "service_name": "threat_intelligence",
    "previous_status": "running",
    "new_status": "disabled"
  }
}
```

**Error Response (Required Service):**
```json
{
  "success": false,
  "error": {
    "code": "CANNOT_DISABLE_REQUIRED",
    "message": "Cannot disable required service: security"
  }
}
```

### Restart Service

```http
POST /api/v1/services/{service_name}/restart
```

**Response:**
```json
{
  "success": true,
  "data": {
    "service_name": "iam",
    "downtime_ms": 523,
    "status": "running"
  }
}
```

### Get Service Health

```http
GET /api/v1/services/{service_name}/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "healthy": true,
    "service": "iam",
    "timestamp": "2025-01-08T10:30:00Z",
    "checks": {
      "api_connectivity": {
        "status": "pass",
        "latency_ms": 45,
        "details": "Connected to iam.googleapis.com"
      },
      "authentication": {
        "status": "pass",
        "details": "Service account authenticated"
      },
      "cache": {
        "status": "pass",
        "hit_rate": 0.92,
        "size_mb": 12.5
      }
    },
    "metrics": {
      "requests_per_minute": 145,
      "error_rate": 0.002,
      "average_latency_ms": 120
    }
  }
}
```

## 🔧 Core Service APIs

### GCP Service

#### List Projects

```http
GET /api/v1/gcp/projects
```

**Query Parameters:**
- `filter` (string, optional): Filter expression
- `page_size` (integer, optional): Number of results per page (default: 50)
- `page_token` (string, optional): Token for pagination

**Response:**
```json
{
  "success": true,
  "data": {
    "projects": [
      {
        "project_id": "my-project-123",
        "project_number": "123456789",
        "display_name": "My Project",
        "state": "ACTIVE",
        "create_time": "2023-01-15T10:00:00Z",
        "labels": {
          "environment": "production",
          "team": "security"
        }
      }
    ],
    "total_count": 15,
    "next_page_token": "eyJzdGFydCI6MTB9"
  }
}
```

#### Get Project Info

```http
GET /api/v1/gcp/project/{project_id}/info
```

**Path Parameters:**
- `project_id` (string): GCP project ID

**Response:**
```json
{
  "success": true,
  "data": {
    "project_id": "my-project-123",
    "project_number": "123456789",
    "display_name": "My Project",
    "state": "ACTIVE",
    "create_time": "2023-01-15T10:00:00Z",
    "parent": {
      "type": "organization",
      "id": "123456789012"
    },
    "labels": {
      "environment": "production",
      "team": "security"
    },
    "lifecycle_state": "ACTIVE"
  }
}
```

#### List Enabled Services

```http
GET /api/v1/gcp/project/{project_id}/services
```

**Response:**
```json
{
  "success": true,
  "data": {
    "project_id": "my-project-123",
    "services": [
      "compute.googleapis.com",
      "storage.googleapis.com",
      "iam.googleapis.com",
      "cloudresourcemanager.googleapis.com",
      "securitycenter.googleapis.com"
    ],
    "total_count": 25
  }
}
```

## 🛡️ Security Service APIs

### Security Evaluation

#### Run Security Evaluation

```http
POST /api/v1/security/evaluate
```

**Request Body:**
```json
{
  "project_id": "my-project-123",
  "scan_types": ["iam", "network", "storage", "compute"],
  "scan_depth": "comprehensive",
  "include_recommendations": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "evaluation_id": "eval_550e8400",
    "project_id": "my-project-123",
    "timestamp": "2025-01-08T10:30:00Z",
    "overall_score": 78,
    "risk_level": "medium",
    "summary": {
      "total_findings": 23,
      "critical": 2,
      "high": 5,
      "medium": 10,
      "low": 6
    },
    "findings": [
      {
        "finding_id": "finding_001",
        "category": "iam",
        "severity": "critical",
        "title": "Overly permissive IAM policy",
        "description": "Service account has Owner role",
        "resource": "serviceAccount:sa@project.iam",
        "recommendation": "Apply principle of least privilege"
      }
    ],
    "recommendations": [
      {
        "priority": "high",
        "category": "iam",
        "action": "Remove Owner role from service accounts",
        "impact": "Reduces risk of privilege escalation",
        "effort": "low"
      }
    ]
  }
}
```

#### Get Security Score

```http
GET /api/v1/security/score?project_id={project_id}
```

**Query Parameters:**
- `project_id` (string): GCP project ID

**Response:**
```json
{
  "success": true,
  "data": {
    "project_id": "my-project-123",
    "current_score": 78,
    "previous_score": 72,
    "trend": "improving",
    "last_evaluation": "2025-01-08T10:30:00Z",
    "score_breakdown": {
      "iam": 85,
      "network": 70,
      "storage": 80,
      "compute": 75
    },
    "risk_level": "medium",
    "improvements_since_last": [
      "Enabled VPC Flow Logs",
      "Removed default service account usage"
    ]
  }
}
```

#### Get Security Recommendations

```http
GET /api/v1/security/recommendations?project_id={project_id}
```

**Query Parameters:**
- `project_id` (string): GCP project ID
- `category` (string, optional): Filter by category (iam, network, storage, compute)
- `priority` (string, optional): Filter by priority (critical, high, medium, low)

**Response:**
```json
{
  "success": true,
  "data": {
    "project_id": "my-project-123",
    "recommendations": [
      {
        "recommendation_id": "rec_001",
        "category": "iam",
        "priority": "critical",
        "title": "Implement least privilege access",
        "description": "Review and reduce excessive permissions",
        "impact": {
          "security_improvement": 15,
          "effort_hours": 4,
          "risk_reduction": "high"
        },
        "steps": [
          "Audit current IAM policies",
          "Identify overprivileged accounts",
          "Create custom roles with minimal permissions",
          "Test and apply new roles"
        ],
        "resources": [
          {
            "type": "serviceAccount",
            "name": "sa@project.iam",
            "current_roles": ["roles/owner"],
            "recommended_roles": ["roles/compute.admin"]
          }
        ]
      }
    ],
    "total_count": 15,
    "estimated_improvement": 25
  }
}
```

### IAM Analysis

#### Analyze User Permissions

```http
GET /api/v1/iam/project/{project_id}/analyze-user/{user_email}
```

**Path Parameters:**
- `project_id` (string): GCP project ID
- `user_email` (string): User email address

**Response:**
```json
{
  "success": true,
  "data": {
    "user": "user@example.com",
    "project_id": "my-project-123",
    "analysis_timestamp": "2025-01-08T10:30:00Z",
    "summary": {
      "total_roles": 5,
      "total_permissions": 347,
      "risk_score": 72,
      "privileged_permissions": 15
    },
    "roles": [
      {
        "role": "roles/editor",
        "type": "primitive",
        "source": "direct",
        "permissions_count": 200
      },
      {
        "role": "roles/storage.admin",
        "type": "predefined",
        "source": "inherited",
        "inherited_from": "folder:123456"
      }
    ],
    "effective_permissions": [
      "compute.instances.create",
      "storage.buckets.delete"
    ],
    "risky_permissions": [
      {
        "permission": "iam.serviceAccounts.actAs",
        "risk_level": "high",
        "reason": "Allows impersonation of service accounts"
      }
    ],
    "recommendations": [
      "Remove Editor role and use specific predefined roles",
      "Consider using custom role with limited permissions"
    ]
  }
}
```

#### Get IAM Testing Scenarios

```http
GET /api/v1/iam/testing/scenarios
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scenarios": [
      {
        "scenario_id": "least_privilege_test",
        "name": "Least Privilege Validation",
        "description": "Tests if users have minimal required permissions",
        "category": "best_practices",
        "severity": "high",
        "estimated_duration": "2 minutes"
      },
      {
        "scenario_id": "service_account_keys",
        "name": "Service Account Key Rotation",
        "description": "Checks age and usage of service account keys",
        "category": "security_hygiene",
        "severity": "medium",
        "estimated_duration": "1 minute"
      }
    ],
    "categories": [
      "best_practices",
      "security_hygiene",
      "compliance",
      "incident_response"
    ]
  }
}
```

#### Run IAM Scenario Test

```http
POST /api/v1/iam/testing/run-scenario/{scenario_id}
```

**Path Parameters:**
- `scenario_id` (string): Scenario identifier

**Request Body:**
```json
{
  "project_id": "my-project-123",
  "parameters": {
    "include_service_accounts": true,
    "check_inherited_roles": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scenario_id": "least_privilege_test",
    "project_id": "my-project-123",
    "execution_time": "2025-01-08T10:30:00Z",
    "duration_ms": 1847,
    "result": "fail",
    "score": 65,
    "findings": [
      {
        "finding_type": "excessive_permissions",
        "severity": "high",
        "resource": "user:admin@example.com",
        "details": "User has Owner role at project level"
      }
    ],
    "passed_checks": 8,
    "failed_checks": 3,
    "remediation_steps": [
      {
        "step": 1,
        "action": "Remove Owner role from user:admin@example.com",
        "command": "gcloud projects remove-iam-policy-binding..."
      }
    ]
  }
}
```

### Compliance Service

#### List Compliance Frameworks

```http
GET /api/v1/compliance/frameworks
```

**Response:**
```json
{
  "success": true,
  "data": {
    "frameworks": [
      {
        "id": "soc2",
        "name": "SOC 2",
        "version": "2017",
        "description": "Service Organization Control 2",
        "categories": ["security", "availability", "confidentiality"],
        "controls_count": 64
      },
      {
        "id": "iso27001",
        "name": "ISO 27001",
        "version": "2013",
        "description": "Information Security Management",
        "categories": ["risk_management", "access_control", "cryptography"],
        "controls_count": 114
      },
      {
        "id": "gdpr",
        "name": "GDPR",
        "version": "2018",
        "description": "General Data Protection Regulation",
        "categories": ["privacy", "data_protection", "consent"],
        "controls_count": 99
      }
    ]
  }
}
```

#### Run Compliance Evaluation

```http
POST /api/v1/compliance/evaluate
```

**Request Body:**
```json
{
  "project_id": "my-project-123",
  "frameworks": ["soc2", "iso27001"],
  "include_evidence": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "evaluation_id": "comp_eval_123",
    "project_id": "my-project-123",
    "timestamp": "2025-01-08T10:30:00Z",
    "overall_compliance": 82,
    "results": {
      "soc2": {
        "compliance_score": 85,
        "total_controls": 64,
        "passed_controls": 54,
        "failed_controls": 6,
        "not_applicable": 4,
        "gaps": [
          {
            "control_id": "CC6.1",
            "control_name": "Logical Access Controls",
            "status": "fail",
            "gap": "MFA not enforced for all users",
            "severity": "high",
            "remediation": "Enable MFA requirement in IAM"
          }
        ]
      },
      "iso27001": {
        "compliance_score": 79,
        "total_controls": 114,
        "passed_controls": 90,
        "failed_controls": 15,
        "not_applicable": 9
      }
    },
    "recommendations": [
      {
        "framework": "soc2",
        "priority": "high",
        "action": "Implement MFA for all user accounts",
        "controls_addressed": ["CC6.1", "CC6.2"],
        "estimated_score_improvement": 5
      }
    ]
  }
}
```

## 📊 Monitoring APIs

### Cloud Logging Service

#### Query Security Events

```http
GET /api/v1/cloud-logging/events
```

**Query Parameters:**
- `project_id` (string): GCP project ID
- `severity` (string, optional): Minimum severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `time_range` (string, optional): Time range (1h, 24h, 7d, 30d)
- `resource_type` (string, optional): Filter by resource type
- `limit` (integer, optional): Maximum results (default: 100)
- `page_token` (string, optional): Pagination token

**Response:**
```json
{
  "success": true,
  "data": {
    "events": [
      {
        "timestamp": "2025-01-08T10:25:00Z",
        "severity": "ERROR",
        "resource": {
          "type": "gce_instance",
          "labels": {
            "instance_id": "1234567890",
            "zone": "us-central1-a"
          }
        },
        "log_name": "compute.googleapis.com/activity",
        "text_payload": "Failed login attempt detected",
        "labels": {
          "security_event": "true",
          "event_type": "authentication_failure"
        },
        "source_location": {
          "file": "auth_handler.py",
          "line": 145,
          "function": "validate_credentials"
        }
      }
    ],
    "total_count": 234,
    "next_page_token": "eyJvZmZzZXQiOjEwMH0="
  }
}
```

#### Get Log Analytics

```http
GET /api/v1/cloud-logging/analytics
```

**Query Parameters:**
- `project_id` (string): GCP project ID
- `metric` (string): Metric type (error_rate, event_frequency, severity_distribution)
- `time_range` (string): Time range for analysis

**Response:**
```json
{
  "success": true,
  "data": {
    "metric": "severity_distribution",
    "time_range": "24h",
    "data": {
      "CRITICAL": 5,
      "ERROR": 23,
      "WARNING": 145,
      "INFO": 3421,
      "DEBUG": 0
    },
    "trends": {
      "error_rate_change": "+15%",
      "most_common_error": "Permission denied",
      "peak_error_hour": "2025-01-08T03:00:00Z"
    }
  }
}
```

### OpenTelemetry Tracing

#### Get Trace Summary

```http
GET /api/v1/tracing/traces
```

**Query Parameters:**
- `project_id` (string): GCP project ID
- `service` (string, optional): Filter by service name
- `time_range` (string): Time range
- `min_duration_ms` (integer, optional): Minimum trace duration

**Response:**
```json
{
  "success": true,
  "data": {
    "traces": [
      {
        "trace_id": "5f8a9b2c3d4e5f6a7b8c9d0e1f2a3b4c",
        "root_span": {
          "operation": "security.evaluate",
          "service": "security",
          "duration_ms": 2341,
          "status": "OK"
        },
        "span_count": 15,
        "services_involved": ["security", "iam", "gcp"],
        "start_time": "2025-01-08T10:25:00Z",
        "end_time": "2025-01-08T10:25:02.341Z"
      }
    ],
    "summary": {
      "total_traces": 1523,
      "average_duration_ms": 156,
      "p95_duration_ms": 423,
      "p99_duration_ms": 892,
      "error_rate": 0.023
    }
  }
}
```

## 🔌 Integration APIs

### Agent Service

#### Chat with AI Agent

```http
POST /api/v1/agent/chat
```

**Request Body:**
```json
{
  "message": "What are the main security risks in my project?",
  "context": {
    "project_id": "my-project-123",
    "include_findings": true
  },
  "session_id": "session_123"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Based on my analysis of project 'my-project-123', I've identified several security risks:\n\n1. **Overly Permissive IAM Policies** (Critical)\n   - 3 service accounts have Owner role\n   - 12 users have Editor role at project level\n\n2. **Exposed Cloud Storage Buckets** (High)\n   - 2 buckets allow public read access\n   - No bucket versioning enabled\n\n3. **Missing Security Controls** (Medium)\n   - VPC Flow Logs not enabled\n   - No Cloud Armor policies configured\n\nWould you like me to provide specific remediation steps for any of these issues?",
    "suggestions": [
      "Review IAM policies for least privilege",
      "Secure Cloud Storage buckets",
      "Enable security monitoring"
    ],
    "actions": [
      {
        "type": "quick_fix",
        "label": "Remove public access from buckets",
        "command": "gsutil iam ch -d allUsers gs://bucket-name"
      }
    ],
    "session_id": "session_123",
    "tokens_used": 245
  }
}
```

#### Get Chat Sessions

```http
GET /api/v1/agent/sessions
```

**Query Parameters:**
- `limit` (integer, optional): Maximum results (default: 20)
- `offset` (integer, optional): Pagination offset

**Response:**
```json
{
  "success": true,
  "data": {
    "sessions": [
      {
        "session_id": "session_123",
        "created_at": "2025-01-08T10:00:00Z",
        "last_activity": "2025-01-08T10:30:00Z",
        "message_count": 15,
        "context": {
          "project_id": "my-project-123"
        },
        "summary": "Security risk analysis and remediation"
      }
    ],
    "total": 42,
    "limit": 20,
    "offset": 0
  }
}
```

### API Hub Service

#### Discover APIs

```http
GET /api/v1/apihub/apis
```

**Query Parameters:**
- `project_id` (string): GCP project ID
- `filter` (string, optional): Search filter
- `include_external` (boolean, optional): Include external APIs

**Response:**
```json
{
  "success": true,
  "data": {
    "apis": [
      {
        "api_id": "compute-v1",
        "name": "Compute Engine API",
        "version": "v1",
        "type": "REST",
        "status": "GA",
        "description": "Create and manage compute resources",
        "documentation_url": "https://cloud.google.com/compute/docs/reference/rest/v1",
        "usage": {
          "last_7_days": 15234,
          "last_30_days": 67891,
          "trend": "increasing"
        }
      }
    ],
    "total_count": 127,
    "categories": [
      "compute",
      "storage",
      "networking",
      "security",
      "ai_ml"
    ]
  }
}
```

## ❌ Error Handling

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | Invalid or missing credentials | 401 |
| `PERMISSION_DENIED` | Insufficient permissions | 403 |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist | 404 |
| `VALIDATION_ERROR` | Invalid request parameters | 400 |
| `SERVICE_UNAVAILABLE` | Service is disabled or down | 503 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Unexpected server error | 500 |

### Error Response Example

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid project ID format",
    "details": {
      "field": "project_id",
      "value": "invalid project id",
      "constraint": "Must match pattern: ^[a-z][a-z0-9-]{4,28}[a-z0-9]$"
    },
    "request_id": "req_123456",
    "documentation_url": "https://docs.example.com/errors/validation"
  }
}
```

## ⚡ Rate Limiting

### Default Limits

| Endpoint Category | Rate Limit | Burst |
|-------------------|------------|-------|
| Service Management | 10 req/min | 20 |
| Security Evaluation | 5 req/min | 10 |
| IAM Analysis | 20 req/min | 40 |
| Log Queries | 30 req/min | 60 |
| Agent Chat | 10 req/min | 15 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1673175000
X-RateLimit-Retry-After: 30
```

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "API rate limit exceeded",
    "details": {
      "limit": 60,
      "window": "1 minute",
      "retry_after": 30
    }
  }
}
```

## 💡 API Examples

### Complete Security Evaluation Flow

```python
import requests
import time

# Base configuration
BASE_URL = "http://localhost:8000/api/v1"
PROJECT_ID = "my-project-123"
headers = {
    "Authorization": f"Bearer {get_access_token()}",
    "Content-Type": "application/json"
}

# 1. Check service status
services_response = requests.get(
    f"{BASE_URL}/services/status/summary",
    headers=headers
)
print("Service Status:", services_response.json())

# 2. Run security evaluation
eval_response = requests.post(
    f"{BASE_URL}/security/evaluate",
    headers=headers,
    json={
        "project_id": PROJECT_ID,
        "scan_types": ["iam", "network", "storage"],
        "scan_depth": "comprehensive"
    }
)
evaluation = eval_response.json()
print("Evaluation ID:", evaluation["data"]["evaluation_id"])

# 3. Get recommendations
rec_response = requests.get(
    f"{BASE_URL}/security/recommendations",
    headers=headers,
    params={"project_id": PROJECT_ID, "priority": "high"}
)
recommendations = rec_response.json()

# 4. Run IAM analysis for specific user
user_response = requests.get(
    f"{BASE_URL}/iam/project/{PROJECT_ID}/analyze-user/admin@example.com",
    headers=headers
)
user_analysis = user_response.json()

# 5. Check compliance
compliance_response = requests.post(
    f"{BASE_URL}/compliance/evaluate",
    headers=headers,
    json={
        "project_id": PROJECT_ID,
        "frameworks": ["soc2"],
        "include_evidence": True
    }
)
compliance_results = compliance_response.json()

print(f"Security Score: {evaluation['data']['overall_score']}")
print(f"High Priority Recommendations: {len(recommendations['data']['recommendations'])}")
print(f"SOC2 Compliance: {compliance_results['data']['results']['soc2']['compliance_score']}%")
```

### Streaming Logs with Pagination

```python
def stream_security_events(project_id, time_range="1h"):
    """Stream security events from Cloud Logging."""
    page_token = None
    
    while True:
        params = {
            "project_id": project_id,
            "severity": "WARNING",
            "time_range": time_range,
            "limit": 50
        }
        
        if page_token:
            params["page_token"] = page_token
        
        response = requests.get(
            f"{BASE_URL}/cloud-logging/events",
            headers=headers,
            params=params
        )
        
        data = response.json()["data"]
        
        for event in data["events"]:
            print(f"[{event['severity']}] {event['timestamp']}: {event['text_payload']}")
        
        page_token = data.get("next_page_token")
        if not page_token:
            break
        
        time.sleep(1)  # Rate limiting

# Stream events
stream_security_events(PROJECT_ID)
```

### Interactive Agent Session

```python
def chat_with_agent(session_id=None):
    """Interactive chat session with security agent."""
    if not session_id:
        session_id = f"session_{int(time.time())}"
    
    print("Security Agent Chat (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        message = input("You: ")
        if message.lower() == 'exit':
            break
        
        response = requests.post(
            f"{BASE_URL}/agent/chat",
            headers=headers,
            json={
                "message": message,
                "context": {"project_id": PROJECT_ID},
                "session_id": session_id
            }
        )
        
        data = response.json()["data"]
        print(f"Agent: {data['response']}")
        
        if data.get("suggestions"):
            print("\nSuggestions:")
            for i, suggestion in enumerate(data["suggestions"], 1):
                print(f"  {i}. {suggestion}")
        
        print("-" * 50)

# Start chat
chat_with_agent()
```

## 📚 Additional Resources

- **OpenAPI Specification**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Postman Collection**: [Download](http://localhost:8000/api/v1/openapi.json)
- **SDK Libraries**: Coming soon
- **Webhook Integration**: See [Webhooks Guide](./WEBHOOKS.md)

---

**API Version:** 1.0.0  
**Last Updated:** January 2025  
**Contact:** api-support@example.com