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

# Run the demo agent with Cloud Logging via OpenTelemetry
#
# Prerequisites:
#   - Install required packages:
#     pip install opentelemetry-exporter-gcp-logging \
#                 opentelemetry-exporter-gcp-monitoring \
#                 opentelemetry-exporter-otlp-proto-grpc
#   - Cloud Logging API enabled in your GCP project
#   - Authenticated with gcloud (gcloud auth application-default login)
#
# Usage:
#   export PROJECT_ID="your-gcp-project"
#   ./run_with_cloud_logging.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR"

# Check for required environment variables
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: PROJECT_ID environment variable is required"
    exit 1
fi

echo "============================================================"
echo "Cloud Logging Demo (via OpenTelemetry)"
echo "============================================================"
echo "Project: $PROJECT_ID"
echo "Agent:   $AGENT_DIR"
echo "============================================================"

# Set OpenTelemetry environment variables
export OTEL_SERVICE_NAME="bq_vs_cloud_logging_demo"
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
# Prevent PII in span attributes (content captured in logs instead)
export ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"

echo ""
echo "OpenTelemetry Configuration:"
echo "  OTEL_SERVICE_NAME=$OTEL_SERVICE_NAME"
echo "  OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=$OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED"
echo "  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=$OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
echo ""

# Run ADK web UI with Cloud Logging enabled
echo "Starting ADK web server with --otel_to_cloud flag..."
echo "Access the web UI at http://localhost:8000"
echo ""
echo "After interacting with the agent, view logs in Cloud Console:"
echo "  https://console.cloud.google.com/logs/query;query=logName%3D%22projects%2F${PROJECT_ID}%2Flogs%2Fadk-otel%22"
echo ""

adk web "$AGENT_DIR" --otel_to_cloud
