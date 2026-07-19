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

"""A coordinator LlmAgent that calls ManagedAgent specialists as single-turn tools.

This sample shows a local ``LlmAgent`` orchestrating two server-backed
``ManagedAgent`` specialists exposed as single-turn sub-agents
(``mode='single_turn'``). ADK auto-wraps each single-turn sub-agent as an inline
tool: the coordinator calls a specialist like a tool, receives the result, and
may call several specialists within a single turn before composing the final
answer. The specialists' internal events are preserved in the shared session.

Each managed call is stateless: single-turn runs are isolated, so the
coordinator should pass a self-contained request to each specialist.

Two specialists are configured:

- ``managed_search_agent`` -- a ``ManagedAgent`` with the server-side
  ``google_search`` tool, for questions that require web search results.
- ``managed_code_execution_agent`` -- a ``ManagedAgent`` with server-side code
  execution, for questions that require computation.

Run with ``adk web`` /
``adk run contributing/samples/managed_agent/single_turn``. See the README
for the required environment / auth setup.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.agents import ManagedAgent
from google.adk.tools import google_search
from google.genai import types

# The Managed Agent id served by the Managed Agents API. Override with the
# MANAGED_AGENT_ID environment variable if your project has access to a
# different agent.
_DEFAULT_AGENT_ID = 'antigravity-preview-05-2026'
_AGENT_ID = os.environ.get('MANAGED_AGENT_ID', _DEFAULT_AGENT_ID)

# A ManagedAgent specialist for questions that require web search results.
# mode='single_turn' exposes it to the coordinator as an inline tool.
managed_search_agent = ManagedAgent(
    name='managed_search_agent',
    mode='single_turn',
    description=(
        'Answers questions that require up-to-date information from the web.'
        ' Uses server-side Google Search.'
    ),
    agent_id=_AGENT_ID,
    environment={'type': 'remote'},
    tools=[google_search],
)

# A ManagedAgent specialist that solves computational questions by running code
# server-side. mode='single_turn' exposes it to the coordinator as an inline
# tool.
managed_code_execution_agent = ManagedAgent(
    name='managed_code_execution_agent',
    mode='single_turn',
    description=(
        'Solves computational, math, or data questions by writing and running'
        ' code server-side. Use for arithmetic, numeric, and other tasks best'
        ' handled by executing code.'
    ),
    agent_id=_AGENT_ID,
    environment={'type': 'remote'},
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)

# The local coordinator. No `model` is set, so ADK uses the default model. The
# two managed specialists are single-turn sub-agents, so ADK exposes each as an
# inline tool; the coordinator calls them and keeps control of the turn (it can
# call both before answering).
root_agent = LlmAgent(
    name='managed_tool_coordinator',
    description='Calls managed specialists as tools and composes the answer.',
    instruction=(
        'You are an assistant with two specialist tools.\n'
        '- Use `managed_search_agent` to look up current information from the'
        ' web.\n'
        '- Use `managed_code_execution_agent` to compute results by running'
        ' code.\n'
        'You may call both tools in a single turn -- for example, look up a'
        ' value and then compute with it -- and then write the final answer'
        ' yourself.'
    ),
    sub_agents=[managed_search_agent, managed_code_execution_agent],
)
