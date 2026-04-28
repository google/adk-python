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

"""Bridge from DNS-AID AgentRecord to ADK RemoteA2aAgent."""

from __future__ import annotations

try:
  from dns_aid.core.models import AgentRecord
  from dns_aid.core.models import Protocol
except ImportError as e:
  raise ImportError(
      'dns_aid is not installed. Please install it with '
      '`pip install "google-adk[dns-aid]"`.'
  ) from e

try:
  from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
  from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
except ImportError as e:
  raise ImportError(
      'a2a-sdk is not installed. Please install it with '
      '`pip install "google-adk[a2a]"`. The dns-aid extra pulls this in '
      'transitively but a partial install can still hit this.'
  ) from e


def remote_a2a_agent_from_record(record: AgentRecord) -> RemoteA2aAgent:
  """Convert a DNS-AID AgentRecord (protocol=a2a) into a RemoteA2aAgent.

  RemoteA2aAgent fetches the agent card lazily on first invocation, so this
  function performs no I/O and is intentionally synchronous.

  Args:
    record: An AgentRecord from dns_aid.discover() with protocol=a2a.

  Returns:
    A RemoteA2aAgent ready to be added to an ADK runner.

  Raises:
    ValueError: If the record's protocol is not A2A.
  """
  if record.protocol != Protocol.A2A:
    raise ValueError(
        f'Agent {record.name} uses protocol {record.protocol}, not A2A'
    )
  agent_card_url = (
      f'{record.endpoint_url.rstrip("/")}{AGENT_CARD_WELL_KNOWN_PATH}'
  )
  return RemoteA2aAgent(
      name=record.name,
      agent_card=agent_card_url,
      description=record.description or '',
  )


__all__ = ['remote_a2a_agent_from_record']
