-- BigQuery table schema for MSA Analysis History
-- Run this to create the necessary tables for MSA Analyzer

-- Create dataset (if not exists)
CREATE SCHEMA IF NOT EXISTS `security_data`
OPTIONS(
  location="US",
  description="Security data including MSA analysis results"
);

-- Create MSA analysis history table
CREATE TABLE IF NOT EXISTS `security_data.msa_analysis_history` (
  analysis_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  total_changes INT64,
  services_affected INT64,
  risk_score INT64,
  risk_level STRING,
  critical_issues INT64,
  security_risk STRING,
  billing_impact STRING,
  compliance_impact STRING,
  recommendations STRING,  -- JSON string
  full_report STRING,      -- JSON string with complete report
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY risk_level, services_affected
OPTIONS(
  description="MSA analysis results for GCP release notes monitoring",
  require_partition_filter=false
);

-- Create active services table (tracks which GCP services are in use)
CREATE TABLE IF NOT EXISTS `security_data.active_services` (
  service_name STRING NOT NULL,
  service_type STRING,
  status STRING,
  project_id STRING,
  enabled_date DATE,
  last_used TIMESTAMP,
  usage_count INT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY status, service_name
OPTIONS(
  description="Active GCP services being monitored by MSA"
);

-- Insert sample active services (customize for your organization)
INSERT INTO `security_data.active_services`
(service_name, service_type, status, enabled_date)
VALUES
  ('BigQuery', 'data-analytics', 'active', CURRENT_DATE()),
  ('Cloud Storage', 'storage', 'active', CURRENT_DATE()),
  ('Compute Engine', 'compute', 'active', CURRENT_DATE()),
  ('Cloud Run', 'compute', 'active', CURRENT_DATE()),
  ('Cloud Functions', 'compute', 'active', CURRENT_DATE()),
  ('Cloud SQL', 'database', 'active', CURRENT_DATE()),
  ('Pub/Sub', 'messaging', 'active', CURRENT_DATE()),
  ('Vertex AI', 'ai-ml', 'active', CURRENT_DATE()),
  ('Cloud KMS', 'security', 'active', CURRENT_DATE()),
  ('Secret Manager', 'security', 'active', CURRENT_DATE()),
  ('VPC', 'networking', 'active', CURRENT_DATE()),
  ('Cloud Armor', 'security', 'active', CURRENT_DATE()),
  ('Identity Platform', 'security', 'active', CURRENT_DATE()),
  ('Firestore', 'database', 'active', CURRENT_DATE()),
  ('Cloud Spanner', 'database', 'active', CURRENT_DATE())
-- Add more services your organization uses
;

-- Create view for latest analysis summary
CREATE OR REPLACE VIEW `security_data.msa_latest_summary` AS
SELECT
  analysis_id,
  timestamp,
  total_changes,
  services_affected,
  risk_score,
  risk_level,
  critical_issues,
  security_risk,
  billing_impact,
  compliance_impact,
  JSON_EXTRACT_SCALAR(recommendations, '$[0].action') as top_recommendation,
  created_at
FROM `security_data.msa_analysis_history`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
ORDER BY timestamp DESC
LIMIT 100;

-- Create view for critical issues tracking
CREATE OR REPLACE VIEW `security_data.msa_critical_issues` AS
SELECT
  analysis_id,
  timestamp,
  risk_level,
  critical_issues,
  security_risk,
  JSON_EXTRACT_ARRAY(recommendations) as recommendations_array,
  created_at
FROM `security_data.msa_analysis_history`
WHERE critical_issues > 0
  OR risk_level = 'high'
  OR security_risk = 'high'
ORDER BY timestamp DESC;

-- Create view for billing impact tracking
CREATE OR REPLACE VIEW `security_data.msa_billing_trends` AS
SELECT
  DATE(timestamp) as analysis_date,
  COUNT(*) as analysis_count,
  SUM(CASE WHEN billing_impact = 'increase' THEN 1 ELSE 0 END) as price_increases,
  SUM(CASE WHEN billing_impact = 'decrease' THEN 1 ELSE 0 END) as price_decreases,
  AVG(risk_score) as avg_risk_score
FROM `security_data.msa_analysis_history`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY DATE(timestamp)
ORDER BY analysis_date DESC;

-- Grant permissions (customize for your service account)
-- Replace YOUR_PROJECT with actual project ID
-- GRANT `roles/bigquery.dataEditor` ON SCHEMA `security_data`
-- TO 'serviceAccount:msa-analyzer@YOUR_PROJECT.iam.gserviceaccount.com';
