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
#
# Convenience wrapper for scripts/run_precommit_checks.py.
#
#   scripts/run_precommit.sh            # auto-fix everything (default)
#   scripts/run_precommit.sh --check    # verify only, no changes (like CI)
#   scripts/run_precommit.sh src tests  # auto-fix specific paths
#
# Picks an interpreter that already has the dev tools installed, in order:
#   1. the repo's .venv (created by `uv sync --extra dev`)
#   2. an active virtualenv ($VIRTUAL_ENV)
#   3. `uv run --extra dev` (resolves/syncs on the fly; slower)
#   4. plain python3
# This avoids re-running `uv run` (which re-resolves the environment) when a
# synced interpreter is already available.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checker="${repo_root}/scripts/run_precommit_checks.py"

if [[ -x "${repo_root}/.venv/bin/python" ]]; then
  exec "${repo_root}/.venv/bin/python" "${checker}" "$@"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  exec "${VIRTUAL_ENV}/bin/python" "${checker}" "$@"
elif command -v uv >/dev/null 2>&1; then
  exec uv run --extra dev python "${checker}" "$@"
else
  exec python3 "${checker}" "$@"
fi
