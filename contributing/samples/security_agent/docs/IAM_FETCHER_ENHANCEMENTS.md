# Enhanced IAM Fetcher Cloud Function

## Overview
The fetch_iam_accounts Cloud Function has been significantly enhanced to provide comprehensive IAM analysis capabilities, including custom role mapping to predefined roles and permission risk analysis.

## Key Enhancements

### 1. Custom Role Analysis
- **Fetches all custom roles** in the project with full permission lists
- **Maps custom roles to similar predefined roles** using permission overlap analysis
- **Calculates similarity percentages** to identify the best predefined role replacements
- **Provides top 3 suggestions** for each custom role with similarity scores

### 2. Permission Risk Analysis
- **Analyzes each permission** to extract:
  - Service (e.g., compute, storage, iam)
  - Resource type
  - Verb (action being performed)
  - Risk level (HIGH/MEDIUM/LOW)
  - Admin vs data access classification
- **Risk scoring** based on permission patterns:
  - HIGH: delete, setIamPolicy, admin, actAs, create, update
  - MEDIUM: write, modify, edit
  - LOW: get, list, read operations

### 3. Enhanced Data Tables

#### iam_accounts Table
- All IAM bindings at project level
- Service account details
- Risk levels for each role assignment
- External vs internal account tracking
- Google-managed service account identification

#### custom_roles Table (NEW)
- Complete custom role inventory
- All included permissions (JSON)
- Similar predefined roles with similarity percentages
- Risk analysis for each custom role
- Permission count metrics

#### role_permissions Table (NEW)
- Granular permission-to-role mapping
- Service and resource type breakdown
- Risk level per permission
- Tracks both custom and predefined roles
- Enables permission usage analysis

### 4. BigQuery Analysis Views

#### iam_stats_view
- Summary statistics for all IAM accounts
- Risk distribution metrics
- Service account vs human account breakdown

#### custom_role_analysis_view
- Custom role to predefined role mapping
- Top 3 similar roles with percentages
- Suggested replacements for optimization

#### permission_risk_view
- Service-level permission risk analysis
- High-risk permission identification
- Permission usage across roles

## Use Cases

### 1. Custom Role Optimization
```sql
-- Find custom roles that could be replaced with predefined roles
SELECT
  custom_role_name,
  title,
  permission_count,
  suggested_replacements,
  max_similarity_percentage
FROM `project.dataset.custom_role_analysis_view`
WHERE max_similarity_percentage > 80
ORDER BY max_similarity_percentage DESC
```

### 2. High-Risk Permission Audit
```sql
-- Identify high-risk permissions in custom roles
SELECT
  role_name,
  permission,
  risk_level,
  service
FROM `project.dataset.role_permissions`
WHERE role_type = 'CUSTOM'
  AND risk_level = 'HIGH'
ORDER BY role_name
```

### 3. Service-Level Risk Assessment
```sql
-- Analyze risk by GCP service
SELECT
  service,
  total_permissions,
  high_risk_permissions,
  ROUND(high_risk_permissions / total_permissions * 100, 2) as risk_percentage
FROM `project.dataset.permission_risk_view`
ORDER BY high_risk_permissions DESC
```

## Deployment

### Function Configuration
- **Runtime**: Python 3.11
- **Memory**: 1024MB (increased for comprehensive analysis)
- **Timeout**: 540 seconds
- **Region**: us-central1

### Environment Variables
- `PROJECT_ID`: Target GCP project
- `BQ_DATASET_ID`: BigQuery dataset for storing results

## Benefits

1. **Custom Role Rationalization**: Identify custom roles that could be replaced with predefined roles
2. **Permission Risk Management**: Understand and mitigate high-risk permission assignments
3. **Compliance Reporting**: Comprehensive IAM audit trails and analysis
4. **Cost Optimization**: Reduce custom role maintenance overhead
5. **Security Posture**: Identify overly permissive role assignments

## API Response Format

```json
{
  "status": "success",
  "tables_updated": {
    "iam_accounts": 45,
    "custom_roles": 12,
    "role_permissions": 850
  },
  "statistics": {
    "iam_accounts": {
      "total": 45,
      "service_accounts": 28,
      "admin_accounts": 8,
      "high_risk": 5,
      "external": 2
    },
    "custom_roles": {
      "total": 12,
      "with_high_risk_permissions": 3
    },
    "permissions": {
      "total_mappings": 850,
      "high_risk_permissions": 125,
      "admin_permissions": 95
    }
  }
}
```

## Next Steps

1. **Set up Cloud Scheduler** for automated refreshes (every 6 hours recommended)
2. **Create alerting** on high-risk role assignments
3. **Build dashboards** for IAM governance visualization
4. **Implement approval workflows** for custom role creation
5. **Regular reviews** of custom role usage and optimization opportunities