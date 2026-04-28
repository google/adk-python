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
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.integrations.dns_aid.a2a_bridge import remote_a2a_agent_from_record


def _make_record(
    *,
    protocol: Protocol,
    name: str = 'chat',
    endpoint_url: str = 'https://chat.example.com/',
    description: str | None = 'hello',
) -> MagicMock:
  """Build a fake AgentRecord-like mock."""
  record = MagicMock()
  record.protocol = protocol
  record.name = name
  record.endpoint_url = endpoint_url
  record.description = description
  return record


class TestRemoteA2aAgentFromRecord:

  def test_remote_a2a_agent_from_record_happy_path(self):
    record = _make_record(
        protocol=Protocol.A2A,
        name='chat',
        endpoint_url='https://chat.example.com/',
        description='hello',
    )

    agent = remote_a2a_agent_from_record(record)

    assert isinstance(agent, RemoteA2aAgent)
    assert agent.name == 'chat'
    assert agent.description == 'hello'
    expected_card = f'https://chat.example.com{AGENT_CARD_WELL_KNOWN_PATH}'
    assert agent._agent_card_source == expected_card

  def test_remote_a2a_agent_from_record_strips_trailing_slash(self):
    record = _make_record(
        protocol=Protocol.A2A,
        endpoint_url='https://example.com',
    )

    agent = remote_a2a_agent_from_record(record)

    expected_card = f'https://example.com{AGENT_CARD_WELL_KNOWN_PATH}'
    assert agent._agent_card_source == expected_card
    assert '//' not in agent._agent_card_source.replace('https://', '')

  def test_remote_a2a_agent_from_record_wrong_protocol(self):
    record = _make_record(protocol=Protocol.MCP)

    with pytest.raises(ValueError):
      remote_a2a_agent_from_record(record)

  @pytest.mark.parametrize('description', [None, ''])
  def test_remote_a2a_agent_from_record_empty_description(self, description):
    record = _make_record(
        protocol=Protocol.A2A,
        description=description,
    )

    agent = remote_a2a_agent_from_record(record)

    assert agent.description == ''
