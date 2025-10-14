#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME=${1:-security-agent}

echo "Building Docker image '${IMAGE_NAME}' from ${PROJECT_ROOT}"
docker build "${PROJECT_ROOT}" -t "${IMAGE_NAME}"
