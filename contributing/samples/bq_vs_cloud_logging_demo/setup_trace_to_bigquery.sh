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
# Setup Cloud Trace to BigQuery Export Pipeline
# =============================================================================
# This script sets up a custom pipeline to export Cloud Trace data to BigQuery.
#
# IMPORTANT: Cloud Trace does NOT have native BigQuery export!
# This requires a custom ETL pipeline, demonstrating the extra complexity
# compared to the BigQuery Agent Analytics Plugin.
#
# COMPARISON:
#   BigQuery Plugin:     pip install + 10 lines of config
#   Cloud Trace Export:  This script + Python ETL script + scheduling
#
# =============================================================================

set -e

PROJECT_ID="${PROJECT_ID:-test-project-0728-467323}"
DATASET_ID="cloud_trace_export"
TABLE_ID="agent_traces"
REGION="US"

echo "============================================================"
echo "Cloud Trace to BigQuery Export Setup"
echo "============================================================"
echo "Project:  $PROJECT_ID"
echo "Dataset:  $DATASET_ID"
echo "Table:    $TABLE_ID"
echo "============================================================"
echo ""
echo "EFFORT COMPARISON:"
echo "  BigQuery Plugin:     ~10 lines of Python config"
echo "  This approach:       150+ lines shell + 200+ lines Python ETL"
echo ""
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: Enable required APIs
# -----------------------------------------------------------------------------
echo ""
echo "[Step 1/5] Enabling required APIs..."

gcloud services enable cloudtrace.googleapis.com --project="$PROJECT_ID" 2>/dev/null || true
gcloud services enable telemetry.googleapis.com --project="$PROJECT_ID" 2>/dev/null || true
gcloud services enable bigquery.googleapis.com --project="$PROJECT_ID" 2>/dev/null || true

echo "  APIs enabled"

# -----------------------------------------------------------------------------
# Step 2: Create BigQuery Dataset
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2/5] Creating BigQuery dataset..."

if bq ls --project_id="$PROJECT_ID" 2>/dev/null | grep -q "$DATASET_ID"; then
    echo "  Dataset already exists: $DATASET_ID"
else
    bq mk \
        --dataset \
        --location="$REGION" \
        --description="Exported Cloud Trace data for ADK agent analytics" \
        "${PROJECT_ID}:${DATASET_ID}"
    echo "  Created dataset: $DATASET_ID"
fi

# -----------------------------------------------------------------------------
# Step 3: Create BigQuery Table with Schema
# -----------------------------------------------------------------------------
echo ""
echo "[Step 3/5] Creating BigQuery table with trace schema..."

# Create table schema JSON
cat > /tmp/trace_schema.json << 'SCHEMA_EOF'
[
  {"name": "trace_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique trace identifier"},
  {"name": "span_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique span identifier"},
  {"name": "parent_span_id", "type": "STRING", "mode": "NULLABLE", "description": "Parent span ID"},
  {"name": "span_name", "type": "STRING", "mode": "REQUIRED", "description": "Name of the span"},
  {"name": "start_time", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Span start time"},
  {"name": "end_time", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Span end time"},
  {"name": "duration_ms", "type": "FLOAT64", "mode": "NULLABLE", "description": "Span duration in milliseconds"},
  {"name": "service_name", "type": "STRING", "mode": "NULLABLE", "description": "Service name from OTel"},
  {"name": "agent_name", "type": "STRING", "mode": "NULLABLE", "description": "ADK agent name"},
  {"name": "session_id", "type": "STRING", "mode": "NULLABLE", "description": "Session/conversation ID"},
  {"name": "operation_name", "type": "STRING", "mode": "NULLABLE", "description": "Operation type (invoke_agent, call_llm, execute_tool)"},
  {"name": "tool_name", "type": "STRING", "mode": "NULLABLE", "description": "Tool name for execute_tool spans"},
  {"name": "tool_call_id", "type": "STRING", "mode": "NULLABLE", "description": "Tool call identifier"},
  {"name": "tool_args", "type": "STRING", "mode": "NULLABLE", "description": "Tool arguments (JSON)"},
  {"name": "tool_response", "type": "STRING", "mode": "NULLABLE", "description": "Tool response (JSON)"},
  {"name": "input_tokens", "type": "INT64", "mode": "NULLABLE", "description": "Input token count"},
  {"name": "output_tokens", "type": "INT64", "mode": "NULLABLE", "description": "Output token count"},
  {"name": "total_tokens", "type": "INT64", "mode": "NULLABLE", "description": "Total token count"},
  {"name": "llm_request", "type": "STRING", "mode": "NULLABLE", "description": "LLM request (JSON)"},
  {"name": "llm_response", "type": "STRING", "mode": "NULLABLE", "description": "LLM response (JSON)"},
  {"name": "model", "type": "STRING", "mode": "NULLABLE", "description": "Model name"},
  {"name": "finish_reason", "type": "STRING", "mode": "NULLABLE", "description": "LLM finish reason"},
  {"name": "labels", "type": "STRING", "mode": "NULLABLE", "description": "All span labels (JSON)"},
  {"name": "exported_at", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "When this record was exported"}
]
SCHEMA_EOF

# Create or update table
if bq show "${PROJECT_ID}:${DATASET_ID}.${TABLE_ID}" >/dev/null 2>&1; then
    echo "  Table already exists: $TABLE_ID"
else
    bq mk \
        --table \
        --description="Exported Cloud Trace spans for ADK agent analytics" \
        "${PROJECT_ID}:${DATASET_ID}.${TABLE_ID}" \
        /tmp/trace_schema.json
    echo "  Created table: $TABLE_ID"
fi

# -----------------------------------------------------------------------------
# Step 4: Create Gemini Model Connection (for AI analytics)
# -----------------------------------------------------------------------------
echo ""
echo "[Step 4/5] Setting up Gemini model connection..."

# Check if connection exists
if bq ls --connection --project_id="$PROJECT_ID" --location="$REGION" 2>/dev/null | grep -q "gemini_conn"; then
    echo "  Connection already exists: gemini_conn"
else
    echo "  Creating Cloud Resource connection..."
    bq mk --connection \
        --location="$REGION" \
        --project_id="$PROJECT_ID" \
        --connection_type=CLOUD_RESOURCE \
        gemini_conn 2>/dev/null || echo "  Connection may already exist"
fi

# Get connection service account and grant permissions
CONN_SA=$(bq show --connection --project_id="$PROJECT_ID" --location="$REGION" gemini_conn 2>/dev/null | grep "serviceAccountId" | awk '{print $2}' || echo "")

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
REMOTE WITH CONNECTION \`${PROJECT_ID}.${REGION}.gemini_conn\`
OPTIONS (endpoint = 'gemini-2.0-flash')
" 2>/dev/null || echo "  Model may already exist"

# -----------------------------------------------------------------------------
# Step 5: Summary
# -----------------------------------------------------------------------------
echo ""
echo "[Step 5/5] Setup complete!"
echo ""
echo "============================================================"
echo "WHAT WAS CREATED:"
echo "============================================================"
echo "  1. BigQuery dataset: ${DATASET_ID}"
echo "  2. BigQuery table: ${TABLE_ID} (with schema for trace data)"
echo "  3. Gemini model connection: gemini_model"
echo ""
echo "============================================================"
echo "NEXT STEPS (MANUAL - This is the extra complexity!):"
echo "============================================================"
echo ""
echo "  1. Run the ETL script to export traces to BigQuery:"
echo "     python export_traces_to_bigquery.py"
echo ""
echo "  2. Query the exported data:"
echo "     See trace_export_queries.sql"
echo ""
echo "============================================================"
echo "TOTAL SETUP EFFORT:"
echo "============================================================"
echo ""
echo "  BigQuery Agent Analytics Plugin:"
echo "    - Setup: 10 lines of Python"
echo "    - Time: 5 minutes"
echo "    - Ongoing: Nothing (automatic)"
echo ""
echo "  Cloud Trace to BigQuery (this approach):"
echo "    - Setup: This script (~150 lines)"
echo "    - ETL Script: export_traces_to_bigquery.py (~200 lines)"
echo "    - Time: 30+ minutes initial + ongoing maintenance"
echo "    - Ongoing: Must run ETL periodically (cron/Cloud Scheduler)"
echo ""
echo "============================================================"
