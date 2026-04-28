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

"""Bridge from DNS-AID AgentRecord to ADK McpToolset."""

from __future__ import annotations

try:
  from dns_aid.core.models import AgentRecord
  from dns_aid.core.models import Protocol
except ImportError as e:
  raise ImportError(
      'dns_aid is not installed. Please install it with '
      '`pip install "google-adk[dns-aid]"`.'
  ) from e

from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def mcp_toolset_from_record(record: AgentRecord) -> McpToolset:
  """Convert a DNS-AID AgentRecord (protocol=mcp) into an ADK McpToolset.

  Uses the streamable-HTTP MCP transport pointing at the record's endpoint
  URL.

  Args:
    record: An AgentRecord from dns_aid.discover() with protocol=mcp.

  Returns:
    An McpToolset ready to be attached to an ADK agent.

  Raises:
    ValueError: If the record's protocol is not MCP.
  """
  if record.protocol != Protocol.MCP:
    raise ValueError(
        f'Agent {record.name} uses protocol {record.protocol}, not MCP'
    )
  return McpToolset(
      connection_params=StreamableHTTPConnectionParams(
          url=record.endpoint_url,
      ),
  )


__all__ = ['mcp_toolset_from_record']
