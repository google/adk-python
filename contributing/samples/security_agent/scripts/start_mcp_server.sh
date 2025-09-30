#!/bin/bash
"""
Startup script for ADK Security Agent MCP Server
"""

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting ADK Security Agent MCP Server..."
echo "Project directory: $PROJECT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install MCP dependencies if not already installed
echo "Installing MCP dependencies..."
pip install -q mcp

# Check if .env file exists and load it
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default environment variables if not set
export DATABASE_PATH="${DATABASE_PATH:-backend/cache/gcp_data.db}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-mgm-digitalconcierge}"

# Verify required files exist
if [ ! -f "mcp_server.py" ]; then
    echo "Error: mcp_server.py not found in $PROJECT_DIR"
    exit 1
fi

if [ ! -f "agents/agent.py" ]; then
    echo "Error: agents/agent.py not found. ADK agent is required."
    exit 1
fi

echo "Environment setup complete."
echo "Database path: $DATABASE_PATH"
echo "GCP Project: $GOOGLE_CLOUD_PROJECT"
echo ""
echo "Starting MCP server..."
echo "Use Ctrl+C to stop the server."
echo ""

# Start the MCP server
exec python mcp_server.py