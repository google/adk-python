# API Endpoint Mapping Analysis

## Current State Analysis

### UnifiedAPIClient Calls vs Backend Routes

| UnifiedAPIClient Call | Expected Backend Route | Actual Backend Route | Status |
|----------------------|------------------------|---------------------|---------|
| GET `/api/v1/assets/snapshot/{project_id}` | Asset Inventory | ❌ Not exists | MISSING |
| GET `/api/v1/assets/summary` | Asset Inventory | ✅ `/api/v1/assets/summary` | OK |
| GET `/api/v1/asset-inventory/summary` | Asset Inventory (old) | ❌ Wrong prefix | MISMATCH |
| POST `/api/v1/assets/discover` | Asset Inventory | ❌ Not exists (has `/search`) | MISMATCH |
| GET `/api/v1/security/score` | Security | ❌ Not implemented | MISSING |
| GET `/api/v1/security/findings` | Security | ❌ Not implemented | MISSING |
| POST `/api/v1/recommendations/dashboard` | Recommendations | ✅ `/api/v1/recommendations/dashboard` | OK |
| GET `/api/v1/iam/project/{project_id}/analyze-all-users` | IAM | ❌ Not implemented | MISSING |
| GET `/api/v1/iam/project/{project_id}/policy` | IAM | ❌ Not implemented | MISSING |
| GET `/api/v1/iam/project/{project_id}/user/{user_email}` | IAM | ❌ Not implemented | MISSING |
| POST `/api/v1/compliance/evaluate` | Compliance | ❌ Router not available | MISSING |
| GET `/api/v1/security/enabled-apis` | Security | ❌ Not implemented | MISSING |
| POST `/api/v1/agent/chat` | Agent | ✅ Should exist (fallback available) | OK |
| GET `/api/v1/gcp/projects` | GCP | ❌ Not implemented | MISSING |
| GET `/api/v1/gcp/projects/{project_id}` | GCP | ❌ Not implemented | MISSING |
| GET `/api/v1/monitoring/summary` | Monitoring | ❌ Not implemented | MISSING |

## Required Fixes

### 1. UnifiedAPIClient Updates
- Change `/api/v1/asset-inventory/summary` to `/api/v1/assets/summary`
- Change `/api/v1/assets/discover` to `/api/v1/assets/search`

### 2. Backend Implementation Needed
- `/api/v1/assets/snapshot/{project_id}` endpoint
- Security router endpoints (`/score`, `/findings`, `/enabled-apis`)
- IAM router endpoints for project/user analysis
- GCP router endpoints for projects
- Monitoring summary endpoint
- Compliance evaluation endpoint

### 3. Current Working Endpoints
- ✅ `/api/v1/assets/summary` - Asset inventory summary
- ✅ `/api/v1/assets/list` - List assets (POST)
- ✅ `/api/v1/assets/search` - Search assets (POST)
- ✅ `/api/v1/assets/export` - Export assets (POST)
- ✅ `/api/v1/assets/asset-types` - Get asset types
- ✅ `/api/v1/recommendations/dashboard` - Get recommendations
- ✅ `/api/v1/agent/chat` - Chat endpoint (with fallback)
- ✅ `/health` - Health check
- ✅ `/` - Root endpoint

## Implementation Priority
1. **High Priority** - Fix mismatched endpoints in UnifiedAPIClient
2. **Medium Priority** - Implement missing security endpoints
3. **Low Priority** - Add remaining endpoints as needed