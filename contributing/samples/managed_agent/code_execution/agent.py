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

"""A ManagedAgent that runs server-side code execution.

``ManagedAgent`` calls the Managed Agents API directly from its run loop instead
of running a local model loop. It currently supports server-side tools only.

Unlike ``LlmAgent``, ``ManagedAgent`` has no ``code_executor`` field, so code
execution is enabled by passing the raw built-in tool config
``types.Tool(code_execution=types.ToolCodeExecution())`` in ``tools`` -- the
same config ``BuiltInCodeExecutor`` produces under the hood. The model writes
and runs code on the server to compute answers.

A fresh remote sandbox is provisioned via ``environment={'type': 'remote'}``;
the environment id is recovered from prior events so multi-turn conversations
reuse the same sandbox.

Run with ``adk web`` /
``adk run contributing/samples/managed_agent/code_execution``. See the README
for the required environment / auth setup.
"""

import os

from google.adk.agents import ManagedAgent
from google.genai import types

# The Managed Agent id served by the Managed Agents API. Override with the
# MANAGED_AGENT_ID environment variable if your project has access to a
# different agent.
_DEFAULT_AGENT_ID = 'antigravity-preview-05-2026'

root_agent = ManagedAgent(
    name='managed_code_execution_agent',
    agent_id=os.environ.get('MANAGED_AGENT_ID', _DEFAULT_AGENT_ID),
    # Provision a remote sandbox for the agent. The environment id is recovered
    # from prior events, so follow-up turns reuse the same sandbox.
    environment={'type': 'remote'},
    # ManagedAgent has no `code_executor` field; enable server-side code
    # execution by passing the raw built-in tool config. This is the same config
    # BuiltInCodeExecutor appends for a regular LlmAgent.
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)
