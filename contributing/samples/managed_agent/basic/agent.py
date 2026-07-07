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

"""A ManagedAgent backed by the Managed Agents API (interactions.create).

``ManagedAgent`` calls the Managed Agents API directly from its run loop instead
of running a local model loop. It currently supports server-side tools only
(ADK built-in tools and raw ``google.genai.types.Tool`` configs); here we wire
up ``google_search``, which runs entirely on the server.

A fresh remote sandbox is provisioned via ``environment={'type': 'remote'}``;
the environment id is recovered from prior events so multi-turn conversations
reuse the same sandbox.

Run with ``adk web`` / ``adk run contributing/samples/managed_agent/basic``. See
the README for the required environment / auth setup.
"""

import os

from google.adk.agents import ManagedAgent
from google.adk.tools import google_search

# The Managed Agent id served by the Managed Agents API. Override with the
# MANAGED_AGENT_ID environment variable if your project has access to a
# different agent.
_DEFAULT_AGENT_ID = 'antigravity-preview-05-2026'

root_agent = ManagedAgent(
    name='managed_search_agent',
    agent_id=os.environ.get('MANAGED_AGENT_ID', _DEFAULT_AGENT_ID),
    # Provision a remote sandbox for the agent. The environment id is recovered
    # from prior events, so follow-up turns reuse the same sandbox.
    environment={'type': 'remote'},
    # Only server-side tools are supported today. google_search is an ADK
    # built-in tool that executes on the server.
    tools=[google_search],
)
