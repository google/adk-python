#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# Setup Cloud Logging Export to BigQuery
# =============================================================================
# This script sets up the infrastructure to export Cloud Logging data to
# BigQuery for analytics. This demonstrates the EXTRA EFFORT required to
# achieve analytics similar to the BigQuery Agent Analytics Plugin.
#
# COMPARISON:
#   BigQuery Plugin:  pip install + 10 lines of config
#   Cloud Logging:    This entire script + ongoing maintenance
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Billing enabled on project
#   - Cloud Logging API enabled
#   - BigQuery API enabled
#
# Usage:
#   export PROJECT_ID="your-project-id"
#   ./setup_cloud_logging_export.sh
# =============================================================================

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-test-project-0728-467323}"
REGION="us-central1"
DATASET_ID="cloud_logging_export"
SINK_NAME="adk-logs-to-bigquery"
LOG_FILTER='logName="projects/'${PROJECT_ID}'/logs/adk-otel"'

echo "============================================================"
echo "Cloud Logging to BigQuery Export Setup"
echo "============================================================"
echo "Project:  $PROJECT_ID"
echo "Dataset:  $DATASET_ID"
echo "Sink:     $SINK_NAME"
echo "============================================================"
echo ""
echo "EFFORT COMPARISON:"
echo "  BigQuery Plugin:     ~10 lines of Python config"
echo "  This approach:       60+ lines of shell + IAM + ongoing maintenance"
echo ""
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: Create BigQuery Dataset for exported logs
# -----------------------------------------------------------------------------
echo ""
echo "[Step 1/5] Creating BigQuery dataset..."
echo "  Command: bq mk --dataset"

if bq ls --project_id="$PROJECT_ID" | grep -q "$DATASET_ID"; then
    echo "  Dataset already exists: $DATASET_ID"
else
    bq mk \
        --dataset \
        --location=US \
        --description="Exported Cloud Logging data for ADK agent analytics" \
        "${PROJECT_ID}:${DATASET_ID}"
    echo "  Created dataset: $DATASET_ID"
fi

# -----------------------------------------------------------------------------
# Step 2: Create Log Sink to export to BigQuery
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2/5] Creating Log Sink..."
echo "  Command: gcloud logging sinks create"

# Check if sink already exists
if gcloud logging sinks describe "$SINK_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo "  Sink already exists: $SINK_NAME"
    echo "  Updating sink..."
    gcloud logging sinks update "$SINK_NAME" \
        "bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/${DATASET_ID}" \
        --project="$PROJECT_ID" \
        --log-filter="$LOG_FILTER"
else
    gcloud logging sinks create "$SINK_NAME" \
        "bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/${DATASET_ID}" \
        --project="$PROJECT_ID" \
        --log-filter="$LOG_FILTER" \
        --description="Export ADK agent logs to BigQuery for analytics"
    echo "  Created sink: $SINK_NAME"
fi

# -----------------------------------------------------------------------------
# Step 3: Get the sink's service account and grant BigQuery permissions
# -----------------------------------------------------------------------------
echo ""
echo "[Step 3/5] Configuring IAM permissions..."

# Get the writer identity (service account) for the sink
SINK_SA=$(gcloud logging sinks describe "$SINK_NAME" \
    --project="$PROJECT_ID" \
    --format='value(writerIdentity)')

echo "  Sink service account: $SINK_SA"
echo "  Granting BigQuery Data Editor role..."

# Grant the service account permission to write to BigQuery
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="$SINK_SA" \
    --role="roles/bigquery.dataEditor" \
    --condition=None \
    --quiet

echo "  IAM permissions configured"

# -----------------------------------------------------------------------------
# Step 4: Create a view to normalize the exported log data
# -----------------------------------------------------------------------------
echo ""
echo "[Step 4/5] Creating normalized view..."
echo "  This view attempts to extract structured data from unstructured logs"
echo "  NOTE: This is FRAGILE and requires maintenance when log format changes"

# The exported logs have a complex nested structure
# We need to create a view that extracts relevant fields
bq query --use_legacy_sql=false --project_id="$PROJECT_ID" "
CREATE OR REPLACE VIEW \`${PROJECT_ID}.${DATASET_ID}.normalized_agent_events\` AS
SELECT
  timestamp,
  -- Log metadata
  logName,
  severity,
  -- Try to extract trace context
  trace,
  spanId,
  -- The actual log content is in different places depending on log type
  COALESCE(
    textPayload,
    JSON_EXTRACT_SCALAR(jsonPayload, '$.message'),
    TO_JSON_STRING(jsonPayload)
  ) AS log_content,
  -- Try to extract structured fields (BEST EFFORT - may be null)
  JSON_EXTRACT_SCALAR(jsonPayload, '$.event_type') AS event_type,
  JSON_EXTRACT_SCALAR(jsonPayload, '$.agent') AS agent,
  JSON_EXTRACT_SCALAR(jsonPayload, '$.session_id') AS session_id,
  JSON_EXTRACT_SCALAR(jsonPayload, '$.tool') AS tool_name,
  -- Resource info
  resource.type AS resource_type,
  resource.labels.project_id AS project_id
FROM \`${PROJECT_ID}.${DATASET_ID}.adk_otel_*\`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
"

echo "  Created view: normalized_agent_events"

# -----------------------------------------------------------------------------
# Step 5: Create Gemini model connection (for AI analytics)
# -----------------------------------------------------------------------------
echo ""
echo "[Step 5/5] Setting up Gemini model connection..."
echo "  This requires additional setup for AI-powered analytics"

# Check if connection exists
if bq ls --connection --project_id="$PROJECT_ID" --location="US" 2>/dev/null | grep -q "gemini_conn"; then
    echo "  Connection already exists: gemini_conn"
else
    echo "  Creating Cloud Resource connection..."
    bq mk --connection \
        --location=US \
        --project_id="$PROJECT_ID" \
        --connection_type=CLOUD_RESOURCE \
        gemini_conn 2>/dev/null || echo "  Connection may already exist"
fi

# Get connection service account
CONN_SA=$(bq show --connection --project_id="$PROJECT_ID" --location="US" gemini_conn 2>/dev/null | grep "serviceAccountId" | awk '{print $2}' || echo "")

if [ -n "$CONN_SA" ]; then
    echo "  Connection service account: $CONN_SA"
    echo "  Granting Vertex AI User role..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$CONN_SA" \
        --role="roles/aiplatform.user" \
        --condition=None \
        --quiet 2>/dev/null || echo "  Role may already be granted"
fi

# Create the remote model
echo "  Creating Gemini remote model..."
bq query --use_legacy_sql=false --project_id="$PROJECT_ID" "
CREATE OR REPLACE MODEL \`${PROJECT_ID}.${DATASET_ID}.gemini_model\`
REMOTE WITH CONNECTION \`${PROJECT_ID}.us.gemini_conn\`
OPTIONS (endpoint = 'gemini-2.0-flash')
" 2>/dev/null || echo "  Model may already exist"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "WHAT WAS CREATED:"
echo "  1. BigQuery dataset: ${DATASET_ID}"
echo "  2. Log sink: ${SINK_NAME}"
echo "  3. IAM bindings for sink service account"
echo "  4. Normalized view: normalized_agent_events"
echo "  5. Gemini model connection: gemini_model"
echo ""
echo "NEXT STEPS:"
echo "  1. Run the agent with --otel_to_cloud to generate logs"
echo "  2. Wait 1-5 minutes for logs to be exported"
echo "  3. Query the exported data (see cloud_logging_export_queries.sql)"
echo ""
echo "IMPORTANT LIMITATIONS:"
echo "  - Log export has 1-5 minute delay (vs 1-2 sec for BQ plugin)"
echo "  - Schema is unstructured (requires complex JSON parsing)"
echo "  - Token usage NOT captured in OTel logs"
echo "  - Tool arguments NOT captured in OTel logs"
echo "  - Multimodal content NOT supported"
echo "  - View requires maintenance when log format changes"
echo ""
echo "============================================================"
echo "EFFORT SUMMARY:"
echo "============================================================"
echo ""
echo "To achieve PARTIAL analytics parity with BigQuery Plugin:"
echo ""
echo "BigQuery Agent Analytics Plugin:"
echo "  - Setup: 10 lines of Python"
echo "  - Time: 5 minutes"
echo "  - Maintenance: None (schema is stable)"
echo "  - Data completeness: 100%"
echo ""
echo "Cloud Logging Export (this approach):"
echo "  - Setup: This script (~150 lines) + IAM configuration"
echo "  - Time: 30+ minutes"
echo "  - Maintenance: Ongoing (log format changes, view updates)"
echo "  - Data completeness: ~40% (no tokens, no tool args, no multimodal)"
echo ""
echo "============================================================"
