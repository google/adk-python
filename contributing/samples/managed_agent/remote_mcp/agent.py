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

"""A ManagedAgent wired to a remote MCP server (Maps Grounding Lite).

``ManagedAgent`` executes MCP **server-side**: it forwards the MCP server
URL and headers to the Managed Agents API, and the backend opens the MCP
session and runs the tools. Unlike ``LlmAgent``'s ``McpToolset``
(client-side execution), ADK never connects to the MCP server here.

Authentication uses a ``header_provider`` callback that mints the request
header at runtime (invoked by the Runner during resolution). Here it reads
``GOOGLE_MAPS_API_KEY`` from the environment and sends it as
``X-Goog-Api-Key``, the header the Maps Grounding Lite MCP server expects.

Run with ``adk web`` or
``adk run contributing/samples/managed_agent/remote_mcp``. See the README
for the required environment / auth setup (enable the Maps Grounding Lite
service and set GOOGLE_MAPS_API_KEY).
"""

import os

from google.adk.agents import ManagedAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import RemoteMcpServer

# The Managed Agent id served by the Managed Agents API. Override with the
# MANAGED_AGENT_ID environment variable if your project has access to a
# different agent.
_DEFAULT_AGENT_ID = 'antigravity-preview-05-2026'

# The Maps Grounding Lite MCP server (Streamable HTTP transport). Uses the mTLS
# endpoint host (mapstools.mtls.googleapis.com); the backend opens the session.
_MAPS_MCP_URL = 'https://mapstools.mtls.googleapis.com/mcp'


def _maps_headers(ctx: ReadonlyContext) -> dict[str, str]:
  """Mint the Maps auth header at request time (runner-invoked).

  Reads GOOGLE_MAPS_API_KEY from the environment. Raising here surfaces loudly
  during tool resolution rather than becoming a silent error event.
  """
  api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
  if not api_key:
    raise ValueError(
        'GOOGLE_MAPS_API_KEY is not set. Enable the Maps Grounding Lite service'
        ' and export GOOGLE_MAPS_API_KEY (see the README).'
    )
  return {'X-Goog-Api-Key': api_key}


root_agent = ManagedAgent(
    name='managed_maps_agent',
    agent_id=os.environ.get('MANAGED_AGENT_ID', _DEFAULT_AGENT_ID),
    # Server-side remote MCP: ADK forwards the URL + headers; the backend runs
    # the MCP tools. The header_provider mints the auth header per turn.
    tools=[
        RemoteMcpServer(
            name='maps_grounding_lite',
            url=_MAPS_MCP_URL,
            header_provider=_maps_headers,
        )
    ],
)
