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
"""Backward-compatibility module re-exporting RemoteA2aAgent."""

from __future__ import annotations

from ..a2a.agent._remote_a2a_agent import A2A_METADATA_PREFIX as A2A_METADATA_PREFIX
from ..a2a.agent._remote_a2a_agent import A2AClientError as A2AClientError
from ..a2a.agent._remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH as AGENT_CARD_WELL_KNOWN_PATH
from ..a2a.agent._remote_a2a_agent import AgentCardResolutionError as AgentCardResolutionError
from ..a2a.agent._remote_a2a_agent import DEFAULT_TIMEOUT as DEFAULT_TIMEOUT
from ..a2a.agent._remote_a2a_agent import RemoteA2aAgent as RemoteA2aAgent

__all__ = [
    "A2AClientError",
    "AGENT_CARD_WELL_KNOWN_PATH",
    "AgentCardResolutionError",
    "RemoteA2aAgent",
]
