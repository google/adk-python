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

"""Constants for the environment toolset."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default limits
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 30
"""Default execution timeout in seconds."""

MAX_OUTPUT_CHARS = 30_000
"""Maximum characters returned to the LLM per tool call."""

# ---------------------------------------------------------------------------
# System instruction templates
# ---------------------------------------------------------------------------

ENVIRONMENT_INSTRUCTION = """\
Your environment is at {working_dir}/

# Environment Rules

DO:
- Chain sequential, dependent commands with `&&` in a single `Execute` call
- To read existing files, always use the `ReadFile` tool. Use `EditFile` to modify existing files.

DON'T:
- Use `Execute` to run cat, head, or tail when `ReadFile` tools can do the job
- Combine `EditFile` or `ReadFile` with `Execute` in the same response (Instead, call the file tool first, then `Execute` in the next turn)
- Use multiple `Execute` calls for dependent commands (they run in parallel)
"""
