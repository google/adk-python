# API Response Formats - ADK Security Agent

## Standard Response Structure

All API endpoints return JSON responses with this structure:

### Success Response
```json
{
  "success": true,
  "data": {...},  // Optional: Main response data
  "total": 0,     // Optional: Total count for list responses
  "error": null
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed error information"
}
```

## Service-Specific Response Formats

### 1. GCP Projects (`/api/v1/gcp/projects`)
```json
{
  "success": true,
  "projects": [
    {
      "id": "project-id",        // IMPORTANT: Use 'id' not 'project_id'
      "name": "Project Name",
      "state": "ACTIVE"           // Optional
    }
  ],
  "total": 1
}
```

### 2. Security Score (`/api/v1/security/score`)
```json
{
  "score": 85,
  "category": "Good",
  "findings": {
    "critical": 0,
    "high": 2,
    "medium": 5,
    "low": 10
  }
}
```

### 3. IAM Analysis (`/api/v1/iam/project/{project_id}/analyze-all-users`)
```json
{
  "project": "project-id",
  "total_users": 10,
  "analyzed_users": 10,
  "high_risk_users": 1,
  "medium_risk_users": 3,
  "low_risk_users": 6,
  "users": [
    {
      "email": "user@example.com",
      "roles": ["roles/viewer"],
      "risk_level": "low"
    }
  ]
}
```

### 4. Recommendations (`/api/v1/recommendations/dashboard`)
```json
{
  "project_id": "project-id",
  "total_recommendations": 15,
  "critical": 3,
  "high": 5,
  "medium": 4,
  "low": 3,
  "recommendations": [
    {
      "id": "rec-001",
      "title": "Enable MFA",
      "description": "Multi-factor authentication not enabled",
      "priority": "critical",
      "category": "IAM",
      "impact": "High",
      "effort": "Low",
      "remediation": "gcloud command..."
    }
  ]
}
```

### 5. Monitoring Summary (`/api/v1/monitoring/summary`)
```json
{
  "project_id": "project-id",
  "status": "healthy",
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "disk_usage": 38.5,
    "network_io": 125.6
  },
  "alerts": {
    "critical": 0,
    "warning": 2,
    "info": 5
  },
  "services": {
    "running": 12,
    "stopped": 0,
    "degraded": 1
  },
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 6. Chat Response (`/api/v1/agent/chat`)
```json
{
  "success": true,
  "response": "AI agent response text...",
  "agent_used": "SecurityAgent",
  "suggestions": [
    "Suggested follow-up question 1",
    "Suggested follow-up question 2"
  ],
  "gcp_api_calls": [],
  "execution_time": 1.23
}
```

### 7. Compliance Evaluation (`/api/v1/compliance/evaluate`)
```json
{
  "project_id": "project-id",
  "overall_score": 78,
  "frameworks": {
    "SOC2": {
      "score": 82,
      "status": "partial",
      "findings": 12,
      "critical": 2
    }
  },
  "recommendations": [
    "Enable audit logging",
    "Implement encryption"
  ]
}
```

## Frontend Expectations

The frontend (`frontend/main_app.py`) expects these specific field names:

1. **Projects**: Looks for `id` field (not `project_id`)
2. **Success Flag**: Checks `success` field to determine if request succeeded
3. **Error Handling**: Looks for `error` field when `success` is false
4. **Lists**: Expects array fields like `projects`, `users`, `recommendations`

## Common Issues and Solutions

### Issue: "No projects available" in sidebar
**Cause**: Frontend looking for wrong field name
**Solution**: Ensure backend returns `id` not `project_id` in project objects

### Issue: API returns 500 error
**Cause**: Missing method in service class
**Solution**: Ensure service class has all required methods (e.g., `get_projects()`)

### Issue: Frontend shows error but API works
**Cause**: Response format mismatch
**Solution**: Check that response includes `success: true` field

## Testing Response Formats

```bash
# Test project list format
curl -s http://localhost:8000/api/v1/gcp/projects | jq

# Expected output:
{
  "success": true,
  "projects": [
    {
      "id": "mgm-digitalconcierge",
      "name": "Project mgm-digitalconcierge"
    }
  ],
  "total": 1
}
```

## Memory Notes

- Always use `id` for project identifier, not `project_id`
- Include `success: true/false` in all responses
- Frontend expects arrays for list endpoints
- Error messages go in `error` field when `success: false`