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


# Runs all unit tests for adk codebase. Sets up test environment according to
# CONTRIBUTING.md.
# Usage: ./unittests.sh [--version <version>] [pytest_args]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
cd ..

# Argument Parsing
ALL_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")
versions_to_run=()
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      if [[ -z "${2:-}" ]]; then
        echo "Error: Missing version for --version flag." >&2
        echo "Usage: $0 [--version <version>] [pytest_args]" >&2
        exit 1
      fi
      # Validate version
      if ! [[ " ${ALL_VERSIONS[*]} " =~ " $2 " ]]; then
        echo "Error: Invalid version '$2'. Supported versions: ${ALL_VERSIONS[*]}" >&2
        exit 1
      fi
      versions_to_run=("$2")
      shift 2
      ;;
    *)
      PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#versions_to_run[@]} -eq 0 ]]; then
  versions_to_run=("${ALL_VERSIONS[@]}")
fi


# Capture original venv for restoration
ORIGINAL_VENV="${VIRTUAL_ENV:-}"

restore_venv() {
    # Deactivate the unittest_venv if it is active
    if command -v deactivate &> /dev/null; then
        deactivate
    fi

    echo "Cleaning up temporary directories..."
    [[ -n "${VENV_DIR:-}" ]] && rm -rf "$VENV_DIR"
    [[ -n "${UV_CACHE_DIR:-}" ]] && rm -rf "$UV_CACHE_DIR"

    if [[ -n "$ORIGINAL_VENV" ]]; then
        echo "Reactivating pre-existing venv: $ORIGINAL_VENV"
        source "$ORIGINAL_VENV/bin/activate"
    fi
}

# Ensure the environment is restored when the script exits.
trap restore_venv EXIT

# Temporary directory for the virtual environment.
VENV_DIR=$(mktemp -d "${TMPDIR:-/tmp}/unittest_venv.XXXXXX")

# Move uv cache to temp to prevent workspace bloat and IDE performance issues.
export UV_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/uv_cache.XXXXXX")

# Force 'copy' mode; hardlinks (uv default) fail on many virtual filesystems.
export UV_LINK_MODE=copy

# 1. deactivate the current venv
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Deactivating current venv: $VIRTUAL_ENV"
    if command -v deactivate &> /dev/null; then
        deactivate
    fi

fi

for version in "${versions_to_run[@]}"; do
    echo ""
    echo "=================================================="
    echo " RUNNING TESTS FOR PYTHON $version"
    echo "=================================================="

    # 2. create a unittest_venv just for unit tests
    echo "Creating/Using unittest_venv for python${version} in $VENV_DIR..."
    uv venv --python "${version}" "$VENV_DIR" --clear
    source "$VENV_DIR/bin/activate"

    # 3. perform the unit tests
    echo "Setting up test environment in $VENV_DIR..."
    uv sync --extra test --active --frozen

    # Determine if PYTEST_ARGS contains explicit test targets (e.g. file, dir, or node ID)
    HAS_TEST_TARGET=false
    for arg in "${PYTEST_ARGS[@]:-}"; do
        [[ "$arg" == *.py || -d "$arg" || "$arg" == *::* ]] && HAS_TEST_TARGET=true && break
    done

    echo "Running unit tests..."
    TEST_EXIT_CODE=0
    if [[ "$HAS_TEST_TARGET" == true ]]; then
        pytest "${PYTEST_ARGS[@]}" || TEST_EXIT_CODE=$?
    else
        pytest ./tests/unittests "${PYTEST_ARGS[@]:-}" || TEST_EXIT_CODE=$?
    fi
    # 4. report the unit tests status as is
    if [[ $TEST_EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "--------------------------------------------------"
        echo "Unit tests failed for Python $version with exit code $TEST_EXIT_CODE"
        echo "--------------------------------------------------"
        exit $TEST_EXIT_CODE
    fi
done


# 5. reactivate the pre-existing venv if the unit test succeeds
echo ""
echo "--------------------------------------------------"
echo "Unit tests passed for all specified versions!"
echo "--------------------------------------------------"
