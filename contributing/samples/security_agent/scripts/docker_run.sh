#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME=${1:-security-agent}

ENV_FILE="${PROJECT_ROOT}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found at ${ENV_FILE}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"

docker run --rm \
  --env-file "$ENV_FILE" \
  -p 8000:8000 -p 5001:5001 -p 8001:8001 \
  -v "${PROJECT_ROOT}/config:/app/config:ro" \
  -v "${PROJECT_ROOT}/logs:/app/logs" \
  "$IMAGE_NAME"
