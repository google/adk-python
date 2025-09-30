-- Scheduled Query: Aggregate Security Metrics
-- Schedule: Every 30 minutes
-- This query aggregates security metrics from all tables for quick analysis

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.security_metrics_hourly`
PARTITION BY DATE(timestamp)
CLUSTER BY metric_category, metric_name
AS
WITH compute_metrics AS (
    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'compute' as metric_category,
        'total_instances' as metric_name,
        COUNT(*) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.compute_instances`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'compute' as metric_category,
        'public_instances' as metric_name,
        COUNTIF(external_ip IS NOT NULL) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.compute_instances`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'compute' as metric_category,
        'unprotected_instances' as metric_name,
        COUNTIF(NOT deletion_protection) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.compute_instances`
),

firewall_metrics AS (
    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'firewall' as metric_category,
        'total_rules' as metric_name,
        COUNT(*) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.firewall_rules`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'firewall' as metric_category,
        'critical_risk_rules' as metric_name,
        COUNTIF(risk_level = 'CRITICAL') as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.firewall_rules`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'firewall' as metric_category,
        'high_risk_rules' as metric_name,
        COUNTIF(risk_level = 'HIGH') as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.firewall_rules`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'firewall' as metric_category,
        'internet_exposed_rules' as metric_name,
        COUNTIF(allows_all_traffic) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.firewall_rules`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'firewall' as metric_category,
        'average_risk_score' as metric_name,
        AVG(risk_score) as metric_value,
        'score' as metric_unit
    FROM `{project_id}.{dataset_id}.firewall_rules`
),

iam_metrics AS (
    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'iam' as metric_category,
        'total_accounts' as metric_name,
        COUNT(DISTINCT email) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.iam_accounts`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'iam' as metric_category,
        'admin_accounts' as metric_name,
        COUNT(DISTINCT email) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.iam_accounts`
    WHERE has_admin_privileges

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'iam' as metric_category,
        'service_accounts' as metric_name,
        COUNT(DISTINCT email) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.iam_accounts`
    WHERE is_service_account

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'iam' as metric_category,
        'external_accounts' as metric_name,
        COUNT(DISTINCT email) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.iam_accounts`
    WHERE is_external

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'iam' as metric_category,
        'critical_risk_accounts' as metric_name,
        COUNT(DISTINCT email) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.iam_accounts`
    WHERE risk_level = 'CRITICAL'
),

storage_metrics AS (
    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'storage' as metric_category,
        'total_buckets' as metric_name,
        COUNT(*) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.storage_buckets`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'storage' as metric_category,
        'public_buckets' as metric_name,
        COUNTIF(is_public) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.storage_buckets`

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'storage' as metric_category,
        'unencrypted_buckets' as metric_name,
        COUNTIF(NOT default_encryption) as metric_value,
        'count' as metric_unit
    FROM `{project_id}.{dataset_id}.storage_buckets`
),

overall_metrics AS (
    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'overall' as metric_category,
        'security_score' as metric_name,
        100 - (
            (SELECT COUNTIF(risk_level IN ('HIGH', 'CRITICAL')) FROM `{project_id}.{dataset_id}.firewall_rules`) * 5 +
            (SELECT COUNTIF(has_admin_privileges) FROM `{project_id}.{dataset_id}.iam_accounts`) * 2 +
            (SELECT COUNTIF(external_ip IS NOT NULL) FROM `{project_id}.{dataset_id}.compute_instances`) * 1
        ) as metric_value,
        'percentage' as metric_unit

    UNION ALL

    SELECT
        CURRENT_TIMESTAMP() as timestamp,
        'overall' as metric_category,
        'critical_issues' as metric_name,
        (SELECT COUNTIF(risk_level = 'CRITICAL') FROM `{project_id}.{dataset_id}.firewall_rules`) +
        (SELECT COUNTIF(risk_level = 'CRITICAL') FROM `{project_id}.{dataset_id}.iam_accounts`) as metric_value,
        'count' as metric_unit
)

SELECT * FROM compute_metrics
UNION ALL
SELECT * FROM firewall_metrics
UNION ALL
SELECT * FROM iam_metrics
UNION ALL
SELECT * FROM storage_metrics
UNION ALL
SELECT * FROM overall_metrics;