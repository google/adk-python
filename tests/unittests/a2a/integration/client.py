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

"""A2A Client for integration tests."""

from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.types import TransportProtocol as A2ATransport
from google.adk.a2a.agent.interceptors.new_integration_extension import _NEW_A2A_ADK_INTEGRATION_EXTENSION
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
import httpx

from .server import agent_card


def create_client(app, streaming: bool = False) -> RemoteA2aAgent:
  """Creates a RemoteA2aAgent connected to the provided FastAPI app.

  Args:
    app: The FastAPI application (server) to connect to.
    streaming: Whether to enable streaming mode in the client.

  Returns:
    A RemoteA2aAgent instance.
  """

  client = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=app), base_url="http://test"
  )

  client_config = A2AClientConfig(
      httpx_client=client,
      streaming=streaming,
      polling=False,
      supported_transports=[A2ATransport.jsonrpc],
  )
  factory = A2AClientFactory(config=client_config)

  # use_legacy=False forces the new implementation
  agent = RemoteA2aAgent(
      name="remote_agent",
      agent_card=agent_card,
      a2a_client_factory=factory,
      use_legacy=False,
  )

  return agent


def create_a2a_client(app, streaming: bool = False):
  """Creates a bare A2A Client connected to the provided FastAPI app.

  This is in contrast to create_client, which wraps the a2a_client into a
  RemoteA2aAgent for the standard runner framework ecosystem execution.

  Args:
    app: The FastAPI application (server) to connect to.
    streaming: Whether to enable streaming mode in the client.

  Returns:
    An A2A Client instance.
  """
  client = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=app),
      base_url="http://test",
      headers={HTTP_EXTENSION_HEADER: _NEW_A2A_ADK_INTEGRATION_EXTENSION},
  )

  client_config = A2AClientConfig(
      httpx_client=client,
      streaming=streaming,
      polling=False,
      supported_transports=[A2ATransport.jsonrpc],
  )
  factory = A2AClientFactory(config=client_config)
  return factory.create(agent_card)
