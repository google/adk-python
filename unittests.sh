#!/bin/bash
# Copyright 2026 Google LLC
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


# Runs all unit tests for adk codebase. Sets up dev environment re: Contributing.md
# Usage: ./unittests.sh [format]
#   format - Optional argument to run autoformat.sh if tests pass

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the environment is restored if the script is interrupted.
cleanup_on_interrupt() {
    echo -e "\nScript interrupted. Restoring full development environment..."
    uv sync --all-extras
    exit 130 # Standard exit code for Ctrl+C
}
trap cleanup_on_interrupt INT TERM

RUN_FORMAT=false
if [[ "${1:-}" == "format" ]]; then
    RUN_FORMAT=true
fi

echo "Setting up test environment..."
uv sync --extra test --extra eval --extra a2a

echo "Running unit tests..."
TEST_EXIT_CODE=0
pytest ./tests/unittests || TEST_EXIT_CODE=$?

echo "Restoring full development environment..."
uv sync --all-extras

if [[ $TEST_EXIT_CODE -ne 0 ]]; then
    echo "Unit tests failed with exit code $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi

echo "Unit tests passed!"

if [[ "$RUN_FORMAT" == "true" ]]; then
    echo "Running autoformat..."
    ./autoformat.sh
fi
