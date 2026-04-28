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

"""DNS-AID integration for Google ADK.

DNS-AID enables agents to discover and publish themselves via DNS SVCB
records. This integration is the decentralized counterpart to
``integrations/agent_registry`` (which uses Google Cloud's centralized
agent registry).

Install: ``pip install "google-adk[dns-aid]"``
"""

from .a2a_bridge import remote_a2a_agent_from_record
from .dns_aid import discover_agents
from .dns_aid import get_dns_aid_tools
from .dns_aid import publish_agent
from .dns_aid import unpublish_agent
from .mcp_bridge import mcp_toolset_from_record

__all__ = [
    'discover_agents',
    'get_dns_aid_tools',
    'mcp_toolset_from_record',
    'publish_agent',
    'remote_a2a_agent_from_record',
    'unpublish_agent',
]
