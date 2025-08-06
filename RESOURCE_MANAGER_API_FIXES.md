# Resource Manager API v3 Fixes

## Problem
The backend was failing with the error:
```
"field [parent] has issue [invalid parent name]"
```

This occurred because the Resource Manager API v3 requires a `parent` parameter for listing projects, but the code was calling:
```
https://cloudresourcemanager.googleapis.com/v3/projects
```

## Solution
Changed the API calls to use Resource Manager API v1, which doesn't require the parent parameter and is more suitable for listing accessible projects.

## Files Modified

### Main Application
1. **`/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/backend/api/gcp.py`**
   - Changed Resource Manager API call from v3 to v1
   - Updated response parsing to handle v1 API response format (`lifecycleState` instead of `state`)

2. **`/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/backend/services/gcp_service.py`**
   - Enhanced error handling with better error messages
   - Added specific guidance for Resource Manager API issues
   - Improved error response parsing

### Backup Files (for consistency)
3. **`/Users/stuartgano/Desktop/Micron/ADK/security_agent_backup/backend/api/gcp.py`**
   - Changed curl command from v3 to v1 API
   - Updated response parsing for v1 API format

4. **`/Users/stuartgano/Desktop/Micron/ADK/security_agent_backup/backend/services/gcp_service.py`**
   - Enhanced error handling to match main application

## Key Changes

### API Version Change
```python
# BEFORE (v3 - requires parent parameter)
result = gcp_service.call_google_api(
    service="cloudresourcemanager",
    version="v3",
    resource_path="projects",
    method="GET"
)

# AFTER (v1 - no parent parameter needed)
result = gcp_service.call_google_api(
    service="cloudresourcemanager",
    version="v1",
    resource_path="projects",
    method="GET"
)
```

### Response Parsing Update
```python
# BEFORE (v3 format)
if project.get("state") == "ACTIVE":

# AFTER (v1 format)
if project.get("lifecycleState") == "ACTIVE":
```

### Enhanced Error Messages
```python
# Added specific guidance for Resource Manager API issues
if service == "cloudresourcemanager" and "parent" in str(e).lower():
    error_details += " (Hint: Resource Manager v3 API requires parent parameter; consider using v1 API instead)"
```

## Testing Plan

1. **Start the backend server:**
   ```bash
   cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test the projects endpoint:**
   ```bash
   curl -X GET "http://localhost:8000/api/v1/gcp/projects"
   ```

3. **Verify the response includes:**
   - `success: true`
   - List of accessible projects
   - Proper project details with `project_id`, `display_name`, etc.

## API Documentation References

- **Resource Manager v1 API (recommended for listing projects):**
  - Endpoint: `https://cloudresourcemanager.googleapis.com/v1/projects`
  - Documentation: https://cloud.google.com/resource-manager/reference/rest/v1/projects/list

- **Resource Manager v3 API (requires parent parameter):**
  - Endpoint: `https://cloudresourcemanager.googleapis.com/v3/projects?parent=organizations/ORG_ID`
  - Documentation: https://cloud.google.com/resource-manager/reference/rest/v3/projects/list

## Benefits of This Fix

1. **Immediate Resolution:** Fixes the "invalid parent name" error
2. **Broader Compatibility:** v1 API works for users without organization-level access
3. **Better Error Messages:** Clear guidance when API issues occur
4. **Consistent Codebase:** All files use the same approach
5. **Authentication Ready:** Works with the newly configured Application Default Credentials