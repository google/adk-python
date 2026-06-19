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

"""Agent that uses Perseus for live workspace context resolution.

Perseus (github.com/Perseus-Computing-LLC/perseus) is an open-source (MIT)
live context engine that resolves workspace state at inference time.  Instead
of baking static instructions into prompts, agents use Perseus directives
(``@file``, ``@search``, ``@memory``, etc.) to pull in exactly what they
need.

This agent demonstrates the pattern:
1. A ``before_agent_callback`` resolves Perseus context before each run.
2. The resolved context is injected into the agent's instruction template.
3. The agent "knows" about workspace state without it being hardcoded.
"""

from __future__ import annotations

import os
import subprocess

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

_PERSEUS_BINARY = os.environ.get("PERSEUS_BINARY", "perseus")


def resolve_perseus_context(callback_context: CallbackContext) -> None:
  """Resolves Perseus directives and stores the result in agent state.

  Called before each agent run.  Reads directives from a state key
  (defaulting to a sensible set) and resolves them via the Perseus CLI.
  """
  directives = callback_context.state.get(
      "_perseus_directives",
      "@file AGENTS.md @file README.md",
  )

  # Optionally scope to a workspace directory.
  workspace = callback_context.state.get("_perseus_workspace", os.getcwd())

  try:
    result = subprocess.run(
        [_PERSEUS_BINARY, "resolve", directives],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=workspace,
    )
    if result.returncode == 0 and result.stdout.strip():
      callback_context.state["_perseus_context"] = result.stdout.strip()
    else:
      callback_context.state["_perseus_context"] = (
          f"(Perseus: no context resolved for directives: {directives})"
      )
  except FileNotFoundError:
    callback_context.state["_perseus_context"] = (
        "(Perseus CLI not installed. Install with: pip install perseus-ctx)"
    )
  except subprocess.TimeoutExpired:
    callback_context.state["_perseus_context"] = (
        "(Perseus resolution timed out)"
    )


root_agent = Agent(
    name="perseus_context_agent",
    description=(
        "Agent with live workspace context via Perseus.  Knows about "
        "project files, git state, and workspace structure without "
        "hardcoded instructions."
    ),
    before_agent_callback=resolve_perseus_context,
    instruction="""\
You are a helpful assistant with live context about the current workspace.

The following context was resolved by Perseus from the workspace files
and state.  Use it to answer questions accurately.

--- BEGIN PERSEUS CONTEXT ---
{_perseus_context}
--- END PERSEUS CONTEXT ---

Use this context to give grounded, file-aware answers.  If the context
is empty or unavailable, let the user know and fall back to general
knowledge.
""",
)
