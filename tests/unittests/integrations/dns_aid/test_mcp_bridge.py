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

from __future__ import annotations

import pytest

pytest.importorskip('dns_aid')

from unittest.mock import MagicMock

from dns_aid.core.models import Protocol
from google.adk.integrations.dns_aid.mcp_bridge import mcp_toolset_from_record
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def _make_record(
    *,
    protocol: Protocol,
    name: str = 'mcp-server',
    endpoint_url: str = 'https://mcp.example.com/',
    description: str | None = 'mcp tools',
) -> MagicMock:
  """Build a fake AgentRecord-like mock."""
  record = MagicMock()
  record.protocol = protocol
  record.name = name
  record.endpoint_url = endpoint_url
  record.description = description
  return record


class TestMcpToolsetFromRecord:

  def test_mcp_toolset_from_record_happy_path(self):
    record = _make_record(
        protocol=Protocol.MCP,
        endpoint_url='https://mcp.example.com/',
    )

    toolset = mcp_toolset_from_record(record)

    assert isinstance(toolset, McpToolset)
    assert toolset._connection_params.url == 'https://mcp.example.com/'

  def test_mcp_toolset_from_record_wrong_protocol(self):
    record = _make_record(protocol=Protocol.A2A)

    with pytest.raises(ValueError):
      mcp_toolset_from_record(record)
