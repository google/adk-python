---
name: bigquery-governance
description: Implement data governance in BigQuery - IAM access control, column/row-level security, data masking, encryption, audit logging, and data catalog integration. Use when securing data, managing access, or implementing compliance requirements.
license: Apache-2.0
compatibility: BigQuery, IAM, Data Catalog, DLP
metadata:
  author: Google Cloud
  version: "1.0"
  category: governance
adk:
  config:
    timeout_seconds: 300
    max_parallel_calls: 3
  allowed_callers:
    - bigquery_agent
    - security_agent
    - compliance_agent
---

# BigQuery Governance Skill

Implement comprehensive data governance in BigQuery including access control, data masking, encryption, audit logging, and compliance management.

## When to Use This Skill

Use this skill when you need to:
- Configure IAM permissions for datasets and tables
- Implement column-level and row-level security
- Set up data masking and anonymization
- Manage encryption (CMEK)
- Enable and analyze audit logs
- Integrate with Data Catalog for metadata management
- Meet compliance requirements (GDPR, HIPAA, PCI-DSS)

## Governance Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **IAM** | Identity-based access control | User/group permissions |
| **Column Security** | Hide sensitive columns | PII protection |
| **Row Security** | Filter rows by user | Multi-tenant data |
| **Data Masking** | Mask/redact values | Privacy compliance |
| **CMEK** | Customer-managed keys | Key control |
| **Audit Logs** | Activity tracking | Compliance auditing |

## Quick Start

### 1. Grant Dataset Access

```sql
-- Grant viewer access to dataset
GRANT `roles/bigquery.dataViewer`
ON SCHEMA `project.dataset`
TO 'user:analyst@company.com';
```

### 2. Create Column-Level Policy

```sql
-- Create policy tag taxonomy
-- (Done via Data Catalog API or Console)

-- Apply policy tag to column
ALTER TABLE `project.dataset.customers`
ALTER COLUMN ssn SET OPTIONS (
  policy_tags = ['projects/project/locations/us/taxonomies/123/policyTags/456']
);
```

### 3. Create Row-Level Policy

```sql
CREATE ROW ACCESS POLICY region_filter
ON `project.dataset.sales`
GRANT TO ('user:regional_manager@company.com')
FILTER USING (region = 'West');
```

## IAM Access Control

### Predefined Roles

| Role | Description | Typical Use |
|------|-------------|-------------|
| `bigquery.admin` | Full BigQuery access | Administrators |
| `bigquery.dataOwner` | Full dataset access | Dataset owners |
| `bigquery.dataEditor` | Read/write tables | Data engineers |
| `bigquery.dataViewer` | Read-only access | Analysts |
| `bigquery.jobUser` | Run queries | Query users |
| `bigquery.user` | List datasets, run jobs | Basic access |

### Grant Permissions

```sql
-- Grant role on dataset
GRANT `roles/bigquery.dataViewer`
ON SCHEMA `project.dataset`
TO 'user:user@company.com';

-- Grant role on table
GRANT `roles/bigquery.dataViewer`
ON TABLE `project.dataset.table`
TO 'group:analysts@company.com';

-- Grant to service account
GRANT `roles/bigquery.dataEditor`
ON SCHEMA `project.dataset`
TO 'serviceAccount:etl@project.iam.gserviceaccount.com';

-- Grant to all authenticated users
GRANT `roles/bigquery.dataViewer`
ON TABLE `project.dataset.public_data`
TO 'allAuthenticatedUsers';
```

### Revoke Permissions

```sql
REVOKE `roles/bigquery.dataViewer`
ON SCHEMA `project.dataset`
FROM 'user:former_employee@company.com';
```

### View Permissions

```sql
-- List dataset permissions
SELECT * FROM `project.dataset.INFORMATION_SCHEMA.OBJECT_PRIVILEGES`;

-- List table permissions
SELECT *
FROM `project.dataset.INFORMATION_SCHEMA.OBJECT_PRIVILEGES`
WHERE object_name = 'table_name';
```

## Column-Level Security

### Setup Policy Tags

Policy tags are created in Data Catalog and applied to columns.

```python
# Using Data Catalog API
from google.cloud import datacatalog_v1

client = datacatalog_v1.PolicyTagManagerClient()

# Create taxonomy
taxonomy = client.create_taxonomy(
    parent=f"projects/{project}/locations/{location}",
    taxonomy=datacatalog_v1.Taxonomy(
        display_name="PII_Taxonomy",
        description="Policy tags for PII data",
        activated_policy_types=[
            datacatalog_v1.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL
        ]
    )
)

# Create policy tag
policy_tag = client.create_policy_tag(
    parent=taxonomy.name,
    policy_tag=datacatalog_v1.PolicyTag(
        display_name="SSN",
        description="Social Security Numbers"
    )
)
```

### Apply Policy Tags

```sql
-- Apply policy tag to column
ALTER TABLE `project.dataset.customers`
ALTER COLUMN ssn SET OPTIONS (
  policy_tags = ['projects/project/locations/us/taxonomies/123/policyTags/ssn']
);

-- Apply to multiple columns
ALTER TABLE `project.dataset.customers`
ALTER COLUMN email SET OPTIONS (
  policy_tags = ['projects/project/locations/us/taxonomies/123/policyTags/email']
),
ALTER COLUMN phone SET OPTIONS (
  policy_tags = ['projects/project/locations/us/taxonomies/123/policyTags/phone']
);
```

### Grant Fine-Grained Access

```sql
-- Grant access to policy tag
GRANT `roles/datacatalog.categoryFineGrainedReader`
ON POLICY TAG `projects/project/locations/us/taxonomies/123/policyTags/ssn`
TO 'user:compliance_officer@company.com';
```

## Row-Level Security

### Create Row Access Policy

```sql
-- Basic filter policy
CREATE ROW ACCESS POLICY sales_region_policy
ON `project.dataset.sales`
GRANT TO ('user:west_manager@company.com')
FILTER USING (region = 'West');

-- Policy using function
CREATE ROW ACCESS POLICY dept_policy
ON `project.dataset.employees`
GRANT TO ('group:managers@company.com')
FILTER USING (
  department IN (
    SELECT department FROM `project.dataset.manager_departments`
    WHERE manager_email = SESSION_USER()
  )
);

-- Policy for multiple groups
CREATE ROW ACCESS POLICY multi_region_policy
ON `project.dataset.sales`
GRANT TO (
  'user:ceo@company.com',
  'group:executives@company.com'
)
FILTER USING (TRUE);  -- Full access
```

### Manage Policies

```sql
-- View existing policies
SELECT *
FROM `project.dataset.INFORMATION_SCHEMA.ROW_ACCESS_POLICIES`
WHERE table_name = 'sales';

-- Drop policy
DROP ROW ACCESS POLICY sales_region_policy
ON `project.dataset.sales`;

-- Drop all policies on table
DROP ALL ROW ACCESS POLICIES ON `project.dataset.sales`;
```

### Best Practices

1. **Use groups instead of individuals** for easier management
2. **Test policies** with different users before production
3. **Document policies** for audit purposes
4. **Consider performance** - complex filters add overhead

## Data Masking

### Dynamic Data Masking

```sql
-- Create masking rule (preview feature)
CREATE DATA MASKING RULE email_mask
ON `project.dataset.customers` (email)
USING MASK_FUNCTION('email');

-- Custom masking function
CREATE DATA MASKING RULE ssn_mask
ON `project.dataset.customers` (ssn)
USING MASK_FUNCTION(
  'partial',
  STRUCT(
    'show_first' AS INT64(0),
    'show_last' AS INT64(4),
    'mask_char' AS STRING('X')
  )
);
```

### SHA256 Hashing

```sql
-- Hash sensitive values for analytics
SELECT
  TO_HEX(SHA256(email)) AS email_hash,
  TO_HEX(SHA256(phone)) AS phone_hash,
  purchase_amount
FROM `project.dataset.transactions`;
```

### Tokenization

```sql
-- Create tokenized view
CREATE VIEW `project.dataset.tokenized_customers` AS
SELECT
  customer_id,
  FARM_FINGERPRINT(email) AS email_token,
  CONCAT(
    SUBSTR(ssn, 1, 3),
    '-XX-XXXX'
  ) AS masked_ssn,
  state,
  signup_date
FROM `project.dataset.customers`;
```

### Data Redaction with DLP

```python
# Using Cloud DLP API for redaction
from google.cloud import dlp_v2

dlp = dlp_v2.DlpServiceClient()

# De-identify configuration
deidentify_config = {
    "record_transformations": {
        "field_transformations": [
            {
                "fields": [{"name": "email"}],
                "primitive_transformation": {
                    "character_mask_config": {
                        "masking_character": "*",
                        "number_to_mask": 0,
                        "characters_to_ignore": [
                            {"characters_to_skip": "@."}
                        ]
                    }
                }
            }
        ]
    }
}
```

## Encryption

### Default Encryption

BigQuery encrypts all data at rest by default using Google-managed keys.

### Customer-Managed Encryption Keys (CMEK)

```sql
-- Create dataset with CMEK
CREATE SCHEMA `project.dataset`
OPTIONS (
  default_kms_key_name = 'projects/project/locations/us/keyRings/ring/cryptoKeys/key'
);

-- Create table with CMEK
CREATE TABLE `project.dataset.secure_table`
(id INT64, data STRING)
OPTIONS (
  kms_key_name = 'projects/project/locations/us/keyRings/ring/cryptoKeys/key'
);

-- Query encryption info
SELECT
  table_name,
  kms_key_name
FROM `project.dataset.INFORMATION_SCHEMA.TABLE_OPTIONS`
WHERE option_name = 'kms_key_name';
```

### Key Rotation

CMEK keys can be rotated in Cloud KMS. BigQuery automatically uses new key versions for encryption while maintaining access to data encrypted with previous versions.

## Audit Logging

### Enable Audit Logs

Audit logs are enabled at the project level in Cloud Console or via gcloud:

```bash
# Enable data access logs
gcloud projects set-iam-policy PROJECT policy.yaml
```

### Query Audit Logs

```sql
-- Query job audit logs
SELECT
  timestamp,
  protopayload_auditlog.methodName,
  protopayload_auditlog.resourceName,
  protopayload_auditlog.authenticationInfo.principalEmail
FROM `project.region.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);

-- Query from Cloud Logging export
SELECT
  timestamp,
  JSON_VALUE(protopayload_auditlog, '$.methodName') AS method,
  JSON_VALUE(protopayload_auditlog, '$.resourceName') AS resource,
  JSON_VALUE(protopayload_auditlog, '$.authenticationInfo.principalEmail') AS user
FROM `project.dataset.cloudaudit_googleapis_com_data_access`
WHERE DATE(timestamp) = CURRENT_DATE();
```

### Key Audit Events

| Event | Method Name | Description |
|-------|-------------|-------------|
| Query Run | `jobservice.jobcompleted` | Query executed |
| Data Read | `tabledata.list` | Table data accessed |
| Table Create | `tables.insert` | Table created |
| Permission Change | `setIamPolicy` | Permissions modified |

## Data Catalog Integration

### Tag Tables with Business Metadata

```sql
-- Add table description
ALTER TABLE `project.dataset.customers`
SET OPTIONS (description = 'Customer master data. Contains PII.');

-- Add labels
ALTER TABLE `project.dataset.customers`
SET OPTIONS (
  labels = [
    ('data_classification', 'confidential'),
    ('data_owner', 'customer_team'),
    ('pii', 'true')
  ]
);
```

### Search Data Catalog

```python
from google.cloud import datacatalog_v1

client = datacatalog_v1.DataCatalogClient()

# Search for PII tables
scope = datacatalog_v1.SearchCatalogRequest.Scope(
    include_project_ids=["my-project"]
)

results = client.search_catalog(
    scope=scope,
    query="tag:pii=true"
)

for result in results:
    print(result.relative_resource_name)
```

## Compliance Patterns

### GDPR - Right to Erasure

```sql
-- Delete user data
DELETE FROM `project.dataset.customers`
WHERE customer_id = @customer_id;

DELETE FROM `project.dataset.orders`
WHERE customer_id = @customer_id;

DELETE FROM `project.dataset.activity_logs`
WHERE user_id = @customer_id;
```

### GDPR - Data Export

```sql
-- Export user data
EXPORT DATA OPTIONS(
  uri='gs://bucket/exports/user_*.json',
  format='JSON'
) AS
SELECT * FROM `project.dataset.customers`
WHERE customer_id = @customer_id;
```

### HIPAA - Minimum Necessary

```sql
-- Create limited view for specific use case
CREATE VIEW `project.dataset.treatment_summary` AS
SELECT
  patient_id,  -- De-identified
  treatment_category,
  treatment_date,
  outcome
FROM `project.dataset.treatments`;
-- Excludes PHI columns
```

## References

- `IAM_ROLES.md` - Complete IAM role reference
- `POLICY_TAGS.md` - Policy tag setup guide
- `AUDIT_QUERIES.md` - Common audit log queries

## Scripts

- `audit_report.py` - Generate access audit report
- `permission_scanner.py` - Scan for permission issues

## Limitations

- Row-level policies: Max 100 per table
- Column-level security: Requires Data Catalog
- Data masking: Preview feature
- Audit logs: 30-day retention (default)
