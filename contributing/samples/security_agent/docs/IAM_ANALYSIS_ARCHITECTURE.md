# IAM Analysis Architecture

## Overview
The IAM analysis system has been refactored into four specialized Cloud Functions, each focusing on a specific aspect of IAM security analysis. This modular approach provides better maintainability, clearer separation of concerns, and avoids the serialization issues encountered with the monolithic design.

## Cloud Functions

### 1. fetch_custom_roles
**Purpose**: Analyze custom IAM roles created by the organization
**BigQuery Table**: `security_insights.custom_roles`
**Key Features**:
- Lists all custom roles in the project
- Analyzes permissions for risk levels (HIGH/MEDIUM/LOW)
- Suggests similar predefined roles for optimization
- Calculates similarity percentages to standard roles

**Key Fields**:
- `role_id`: Custom role identifier
- `included_permissions`: JSON array of permissions
- `high_risk_permissions`: Count of dangerous permissions
- `similar_predefined_roles`: JSON array of similar standard roles with percentages

### 2. fetch_user_roles
**Purpose**: Track human user IAM bindings
**BigQuery Table**: `security_insights.user_roles`
**Key Features**:
- Identifies all human users with project access
- Flags admin and owner privileges
- Detects external users (outside project domain)
- Tracks conditional access policies

**Key Fields**:
- `user_email`: User's email address
- `is_admin`: Boolean flag for admin privileges
- `is_owner`: Boolean flag for owner role
- `is_external`: Boolean flag for external users
- `domain`: Email domain for organization tracking

### 3. fetch_service_account_roles
**Purpose**: Analyze service account permissions and keys
**BigQuery Table**: `security_insights.service_account_roles`
**Key Features**:
- Lists all service account IAM bindings
- Tracks service account keys and status
- Distinguishes Google-managed vs user-managed accounts
- Identifies disabled service accounts
- Counts user-managed keys for security assessment

**Key Fields**:
- `service_account_email`: Service account identifier
- `has_keys`: Boolean flag for key existence
- `key_count`: Number of user-managed keys
- `is_google_managed`: Boolean for Google service accounts
- `disabled`: Service account status

### 4. fetch_standard_roles
**Purpose**: Catalog and analyze predefined GCP roles
**BigQuery Table**: `security_insights.standard_roles`
**Key Features**:
- Lists all available predefined roles
- Categorizes roles (PRIMITIVE/ADMIN/READ_ONLY/WRITE/SERVICE_SPECIFIC)
- Analyzes permissions and capabilities
- Groups roles by GCP service
- Identifies high-privilege roles

**Key Fields**:
- `role_name`: Predefined role identifier
- `category`: Role classification
- `services_accessed`: JSON array of GCP services
- `capabilities`: JSON array (READ/WRITE/DELETE/IAM_ADMIN)
- `is_primitive`: Boolean for basic roles (owner/editor/viewer)

## Benefits of Modular Architecture

1. **Improved Performance**: Each function runs independently with focused scope
2. **Better Error Handling**: Failures isolated to specific analysis types
3. **Simpler Maintenance**: Clear separation of concerns
4. **Easier Testing**: Individual functions can be tested in isolation
5. **Scalability**: Functions can be scaled independently based on usage

## SQL Query Examples

### Find Over-Privileged Custom Roles
```sql
SELECT
  role_id,
  permission_count,
  high_risk_permissions,
  similar_predefined_roles
FROM `security_insights.custom_roles`
WHERE high_risk_permissions > 5
ORDER BY high_risk_permissions DESC
```

### Identify External Admin Users
```sql
SELECT
  user_email,
  role,
  domain
FROM `security_insights.user_roles`
WHERE is_external = TRUE
  AND (is_admin = TRUE OR is_owner = TRUE)
```

### Service Accounts with Multiple Keys
```sql
SELECT
  service_account_email,
  role,
  key_count
FROM `security_insights.service_account_roles`
WHERE is_user_managed = TRUE
  AND key_count > 1
ORDER BY key_count DESC
```

### Compare Custom to Standard Roles
```sql
WITH custom_perms AS (
  SELECT
    role_id,
    JSON_EXTRACT_ARRAY(included_permissions) as perms
  FROM `security_insights.custom_roles`
),
standard_perms AS (
  SELECT
    role_name,
    JSON_EXTRACT_ARRAY(included_permissions) as perms
  FROM `security_insights.standard_roles`
)
-- Compare permission overlap between custom and standard roles
SELECT
  c.role_id,
  s.role_name,
  ARRAY_LENGTH(
    ARRAY(
      SELECT perm
      FROM UNNEST(c.perms) as perm
      WHERE perm IN UNNEST(s.perms)
    )
  ) as common_permissions
FROM custom_perms c
CROSS JOIN standard_perms s
WHERE ARRAY_LENGTH(c.perms) > 0
  AND ARRAY_LENGTH(s.perms) > 0
ORDER BY common_permissions DESC
```

## Deployment Status

| Function | Table | Status |
|----------|-------|--------|
| fetch_custom_roles | custom_roles | Deploying |
| fetch_user_roles | user_roles | Deploying |
| fetch_service_account_roles | service_account_roles | Deploying |
| fetch_standard_roles | standard_roles | Deploying |

## Next Steps

1. **Testing**: Invoke each function to populate BigQuery tables
2. **Verification**: Query tables to ensure data integrity
3. **Integration**: Update security dashboard to use new tables
4. **Automation**: Schedule regular executions via Cloud Scheduler
5. **Alerting**: Set up monitoring for anomalous IAM changes