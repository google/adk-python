# Remediation API Documentation

## Overview

The Remediation API provides automated vulnerability remediation capabilities with safety mechanisms, approval workflows, and rollback support. This API is part of STORY-210: Automated Remediation Engine.

**Base URL**: `/api/v1/remediation`

## Endpoints

### 1. Execute Remediation

Execute automated remediation for a security vulnerability.

**Endpoint**: `POST /api/v1/remediation/execute`

**Request Body**:
```json
{
  "vulnerability_id": "vuln-001",
  "remediation_template": "PUBLIC_BUCKET_REMEDIATION",
  "parameters": {
    "bucket_name": "my-public-bucket",
    "project_id": "my-project"
  },
  "auto_approve": false,
  "dry_run": true,
  "priority": "HIGH"
}
```

**Response**:
```json
{
  "remediation_id": "rem-abc123",
  "status": "SUCCESS",
  "vulnerability_id": "vuln-001",
  "resource_name": "//storage.googleapis.com/my-public-bucket",
  "changes_made": [
    {
      "action": "MODIFY_BUCKET_IAM",
      "description": "Removed public access",
      "status": "SUCCESS"
    }
  ],
  "rollback_point": "snapshot-xyz789",
  "execution_time": 12.5,
  "validation_results": {
    "bucket_not_public": true,
    "uniform_access_enabled": true
  }
}
```

**Status Codes**:
- `200 OK`: Remediation executed successfully
- `400 Bad Request`: Invalid parameters
- `403 Forbidden`: Approval required but not provided
- `500 Internal Server Error`: Execution failed

---

### 2. Get Remediation Status

Get the current status of a remediation execution.

**Endpoint**: `GET /api/v1/remediation/status/{remediation_id}`

**Response**:
```json
{
  "remediation_id": "rem-abc123",
  "status": "IN_PROGRESS",
  "vulnerability_id": "vuln-001",
  "resource_name": "//storage.googleapis.com/my-bucket",
  "progress": 75,
  "changes_made": [],
  "error_message": null,
  "execution_time": 8.3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### 3. Rollback Remediation

Rollback a completed remediation to its previous state.

**Endpoint**: `POST /api/v1/remediation/rollback`

**Request Body**:
```json
{
  "remediation_id": "rem-abc123",
  "rollback_point": "snapshot-xyz789",
  "reason": "Unexpected side effects detected"
}
```

**Response**:
```json
{
  "success": true,
  "remediation_id": "rem-abc123",
  "rollback_point": "snapshot-xyz789",
  "reason": "Unexpected side effects detected",
  "timestamp": "2024-01-15T11:00:00Z"
}
```

---

### 4. List Remediation Templates

Get all available remediation templates.

**Endpoint**: `GET /api/v1/remediation/templates`

**Response**:
```json
{
  "templates": [
    {
      "id": "PUBLIC_BUCKET_REMEDIATION",
      "name": "Remove Public Access from Storage Bucket",
      "description": "Removes public access and enables uniform bucket-level access",
      "vulnerability_types": ["PUBLIC_STORAGE_NO_AUTH", "PUBLIC_BUCKET"],
      "risk_level": "HIGH",
      "requires_approval": true
    },
    {
      "id": "EXCESSIVE_IAM_REMEDIATION",
      "name": "Remove Excessive IAM Permissions",
      "description": "Replaces overly broad roles with least-privilege alternatives",
      "vulnerability_types": ["EXCESSIVE_IAM_PERMISSIONS"],
      "risk_level": "CRITICAL",
      "requires_approval": true
    }
  ],
  "total": 4
}
```

---

### 5. Batch Remediation

Execute remediation for multiple vulnerabilities.

**Endpoint**: `POST /api/v1/remediation/batch`

**Request Body**:
```json
{
  "vulnerabilities": ["vuln-001", "vuln-002", "vuln-003"],
  "template": "PUBLIC_BUCKET_REMEDIATION",
  "auto_approve": false
}
```

**Response**:
```json
{
  "success": true,
  "total_processed": 3,
  "results": [
    {
      "vulnerability_id": "vuln-001",
      "remediation_id": "rem-001",
      "status": "SUCCESS"
    },
    {
      "vulnerability_id": "vuln-002",
      "remediation_id": "rem-002",
      "status": "SUCCESS"
    },
    {
      "vulnerability_id": "vuln-003",
      "status": "FAILED",
      "error": "Resource not found"
    }
  ],
  "timestamp": "2024-01-15T12:00:00Z"
}
```

---

### 6. Get Pending Approvals

Get list of remediations pending approval.

**Endpoint**: `GET /api/v1/remediation/approval/pending`

**Response**:
```json
{
  "success": true,
  "pending_count": 2,
  "approvals": [
    {
      "request_id": "req-123",
      "remediation_id": "rem-abc",
      "template_name": "Remove Excessive IAM Permissions",
      "risk_level": "CRITICAL",
      "resource_name": "//iam.googleapis.com/projects/test/serviceAccounts/admin",
      "requested_at": "2024-01-15T09:00:00Z",
      "timeout": "2024-01-15T11:00:00Z",
      "approvers": ["security-lead@company.com"]
    }
  ]
}
```

---

### 7. Approve Remediation

Approve a pending remediation request.

**Endpoint**: `POST /api/v1/remediation/approval/{request_id}/approve`

**Request Body**:
```json
{
  "approver": "security-lead@company.com",
  "comments": "Approved after review"
}
```

---

### 8. Reject Remediation

Reject a pending remediation request.

**Endpoint**: `POST /api/v1/remediation/approval/{request_id}/reject`

**Request Body**:
```json
{
  "rejector": "security-lead@company.com",
  "reason": "Needs additional review"
}
```

---

### 9. Get Remediation Metrics

Get system-wide remediation metrics and statistics.

**Endpoint**: `GET /api/v1/remediation/metrics`

**Response**:
```json
{
  "total_remediations": 142,
  "success_rate": 95.3,
  "average_execution_time": 23.5,
  "rollback_count": 3,
  "pending_approvals": 2,
  "by_status": {
    "SUCCESS": 135,
    "FAILED": 4,
    "ROLLED_BACK": 3
  },
  "by_vulnerability_type": {
    "PUBLIC_STORAGE_NO_AUTH": 45,
    "EXCESSIVE_IAM_PERMISSIONS": 38,
    "MISSING_ENCRYPTION": 32,
    "WEAK_NETWORK_SECURITY": 27
  },
  "mttr": 12.3,
  "timestamp": "2024-01-15T13:00:00Z"
}
```

---

## Remediation Templates

### Available Templates

1. **PUBLIC_BUCKET_REMEDIATION**
   - Removes public access from storage buckets
   - Enables uniform bucket-level access
   - Risk Level: HIGH
   - Requires Approval: Yes

2. **EXCESSIVE_IAM_REMEDIATION**
   - Removes owner/editor roles
   - Replaces with viewer role
   - Risk Level: CRITICAL
   - Requires Approval: Yes

3. **MISSING_ENCRYPTION_REMEDIATION**
   - Enables default encryption
   - Risk Level: HIGH
   - Requires Approval: No

4. **WEAK_NETWORK_SECURITY_REMEDIATION**
   - Restricts overly permissive firewall rules
   - Removes 0.0.0.0/0 access
   - Risk Level: HIGH
   - Requires Approval: Yes

## Status Values

- `PENDING`: Awaiting approval
- `APPROVED`: Approved, ready to execute
- `IN_PROGRESS`: Currently executing
- `SUCCESS`: Completed successfully
- `FAILED`: Execution failed
- `ROLLED_BACK`: Changes rolled back
- `REJECTED`: Approval rejected
- `UNSAFE`: Failed safety checks

## Error Codes

| Code | Description |
|------|-------------|
| `REM001` | Template not found |
| `REM002` | Invalid parameters |
| `REM003` | Approval required |
| `REM004` | Rollback point not found |
| `REM005` | Execution timeout |
| `REM006` | Resource not accessible |
| `REM007` | Validation failed |

## Rate Limits

- Execute remediation: 10 requests per minute
- Batch remediation: 1 request per minute
- Status checks: 60 requests per minute

## Best Practices

1. **Always run dry-run first** before actual remediation
2. **Save rollback points** for critical remediations
3. **Use batch API** for multiple vulnerabilities
4. **Monitor metrics** to track remediation effectiveness
5. **Set up approval workflows** for high-risk changes

## ADK Agent Integration

The remediation capabilities are fully integrated with the ADK agent. Example commands:

```
"Fix the public bucket vulnerability vuln-001"
"Show me available remediation templates"
"What's the status of remediation rem-abc123?"
"Rollback the last remediation"
"Show remediation metrics"
```

## Security Considerations

- All remediation actions are logged for audit
- Rollback points expire after 7 days
- Critical remediations require multi-level approval
- Service account needs appropriate GCP permissions
- All API calls require authentication