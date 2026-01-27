#!/bin/bash
# Run the Agent Analytics Dashboard
#
# Usage:
#   ./run_dashboard.sh [--simulate] [--port PORT]
#
# Options:
#   --simulate    Run simulation to populate sample data first
#   --port PORT   Port to run on (default: 8080)
#   --help        Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8080
SIMULATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --simulate)
            SIMULATE=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--simulate] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --simulate    Run simulation to populate sample data first"
            echo "  --port PORT   Port to run on (default: 8080)"
            echo "  --help        Show this help message"
            echo ""
            echo "Environment Variables (required):"
            echo "  BQ_AGENT_ANALYTICS_PROJECT  Google Cloud project ID"
            echo "  BQ_AGENT_ANALYTICS_DATASET  BigQuery dataset ID"
            echo ""
            echo "Environment Variables (optional):"
            echo "  BQ_AGENT_ANALYTICS_TABLE    BigQuery table name (default: agent_events_v2)"
            echo "  VERTEXAI_PROJECT            Vertex AI project ID"
            echo "  VERTEXAI_LOCATION           Vertex AI location (default: us-central1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required environment variables
if [ -z "$BQ_AGENT_ANALYTICS_PROJECT" ]; then
    echo "Error: BQ_AGENT_ANALYTICS_PROJECT environment variable is not set"
    echo "Please set it with: export BQ_AGENT_ANALYTICS_PROJECT=your-project-id"
    exit 1
fi

if [ -z "$BQ_AGENT_ANALYTICS_DATASET" ]; then
    echo "Error: BQ_AGENT_ANALYTICS_DATASET environment variable is not set"
    echo "Please set it with: export BQ_AGENT_ANALYTICS_DATASET=your-dataset-id"
    exit 1
fi

# Set defaults for optional variables
export BQ_AGENT_ANALYTICS_TABLE="${BQ_AGENT_ANALYTICS_TABLE:-agent_events_v2}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$BQ_AGENT_ANALYTICS_PROJECT}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-central1}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$BQ_AGENT_ANALYTICS_PROJECT}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"

echo "==================================="
echo "Agent Analytics Dashboard"
echo "==================================="
echo "Project:  $BQ_AGENT_ANALYTICS_PROJECT"
echo "Dataset:  $BQ_AGENT_ANALYTICS_DATASET"
echo "Table:    $BQ_AGENT_ANALYTICS_TABLE"
echo "Port:     $PORT"
echo "==================================="
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run simulation if requested
if [ "$SIMULATE" = true ]; then
    echo "Running simulation to generate sample data..."
    python simulate_agent_data.py --num-sessions 10
    echo ""
    echo "Simulation complete. Starting dashboard..."
    echo ""
fi

# Start the dashboard
echo "Starting dashboard at http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
