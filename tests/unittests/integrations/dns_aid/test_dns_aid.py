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

import inspect
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations.dns_aid.dns_aid import discover_agents
from google.adk.integrations.dns_aid.dns_aid import get_dns_aid_tools
from google.adk.integrations.dns_aid.dns_aid import publish_agent
from google.adk.integrations.dns_aid.dns_aid import unpublish_agent
from google.adk.tools.function_tool import FunctionTool


def _make_discovery_result(payload: dict) -> MagicMock:
  """Build a fake DiscoveryResult-like mock with a model_dump method."""
  result = MagicMock()
  result.model_dump = MagicMock(return_value=payload)
  return result


def _make_publish_result(payload: dict) -> MagicMock:
  """Build a fake PublishResult-like mock with a model_dump method."""
  result = MagicMock()
  result.model_dump = MagicMock(return_value=payload)
  return result


class TestDiscoverAgents:

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.discover',
      new_callable=AsyncMock,
  )
  async def test_discover_agents_happy_path(self, mock_discover):
    payload = {
        'agents': [{
            'name': 'chat',
            'protocol': 'a2a',
            'endpoint_url': 'https://chat.example.com/',
        }],
        'count': 1,
    }
    mock_discover.return_value = _make_discovery_result(payload)

    result = await discover_agents(domain='example.com')

    assert result == payload
    assert not isinstance(result, str)
    mock_discover.assert_awaited_once()

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.discover',
      new_callable=AsyncMock,
  )
  async def test_discover_agents_empty(self, mock_discover):
    payload = {'agents': [], 'count': 0}
    mock_discover.return_value = _make_discovery_result(payload)

    result = await discover_agents(domain='example.com')

    assert result == payload
    assert result['count'] == 0


class TestPublishAgent:

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.create_backend',
  )
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.publish',
      new_callable=AsyncMock,
  )
  async def test_publish_agent_happy_path(
      self, mock_publish, mock_create_backend
  ):
    payload = {
        'success': True,
        'agent_name': 'chat',
        'domain': 'example.com',
    }
    mock_publish.return_value = _make_publish_result(payload)
    mock_create_backend.return_value = MagicMock()

    result = await publish_agent(
        agent_name='chat',
        domain='example.com',
        protocol='mcp',
        endpoint='https://chat.example.com',
        port=443,
        backend_name='mock',
    )

    assert result == payload
    mock_create_backend.assert_called_once_with('mock')
    mock_publish.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_publish_agent_invalid_agent_name(self):
    with pytest.raises((ValueError, Exception)):
      await publish_agent(
          agent_name='_invalid',
          domain='example.com',
          endpoint='https://chat.example.com',
      )

  @pytest.mark.asyncio
  async def test_publish_agent_invalid_protocol(self):
    with pytest.raises((ValueError, Exception)):
      await publish_agent(
          agent_name='chat',
          domain='example.com',
          protocol='ftp',
          endpoint='https://chat.example.com',
      )

  @pytest.mark.asyncio
  @pytest.mark.parametrize('bad_port', [0, 70000])
  async def test_publish_agent_invalid_port(self, bad_port):
    with pytest.raises((ValueError, Exception)):
      await publish_agent(
          agent_name='chat',
          domain='example.com',
          endpoint='https://chat.example.com',
          port=bad_port,
      )

  @pytest.mark.asyncio
  async def test_publish_agent_invalid_ttl(self):
    with pytest.raises((ValueError, Exception)):
      await publish_agent(
          agent_name='chat',
          domain='example.com',
          endpoint='https://chat.example.com',
          ttl=10,
      )

  @pytest.mark.asyncio
  async def test_publish_agent_empty_endpoint(self):
    with pytest.raises((ValueError, Exception)):
      await publish_agent(
          agent_name='chat',
          domain='example.com',
          endpoint='',
      )

  @pytest.mark.asyncio
  async def test_publish_agent_unknown_backend(self):
    with pytest.raises((ValueError, Exception)):
      await publish_agent(
          agent_name='chat',
          domain='example.com',
          endpoint='https://chat.example.com',
          backend_name='not-a-real-backend',
      )


class _PermissionDenied(Exception):
  """Throwaway permission-style exception."""


class _ConnectionFailure(Exception):
  """Throwaway connection-style exception."""


class _ThrottleExceeded(Exception):
  """Throwaway throttle-style exception."""


class TestUnpublishAgent:

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.unpublish',
      new_callable=AsyncMock,
  )
  async def test_unpublish_agent_success(self, mock_unpublish):
    mock_unpublish.return_value = True

    result = await unpublish_agent(
        agent_name='chat',
        domain='example.com',
    )

    assert result['success'] is True
    assert result['status'] == 'ok'
    assert result['agent_name'] == 'chat'
    assert result['domain'] == 'example.com'

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.unpublish',
      new_callable=AsyncMock,
  )
  async def test_unpublish_agent_not_found_via_false(self, mock_unpublish):
    mock_unpublish.return_value = False

    result = await unpublish_agent(
        agent_name='chat',
        domain='example.com',
    )

    assert result['success'] is False
    assert result['status'] == 'not_found'

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.unpublish',
      new_callable=AsyncMock,
  )
  async def test_unpublish_agent_permission_denied(self, mock_unpublish):
    mock_unpublish.side_effect = _PermissionDenied('nope')

    result = await unpublish_agent(
        agent_name='chat',
        domain='example.com',
    )

    assert result['success'] is False
    assert result['status'] == 'permission_denied'

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.unpublish',
      new_callable=AsyncMock,
  )
  async def test_unpublish_agent_backend_unavailable(self, mock_unpublish):
    mock_unpublish.side_effect = _ConnectionFailure('down')

    result = await unpublish_agent(
        agent_name='chat',
        domain='example.com',
    )

    assert result['success'] is False
    assert result['status'] == 'backend_unavailable'

  @pytest.mark.asyncio
  @patch(
      'google.adk.integrations.dns_aid.dns_aid.dns_aid.unpublish',
      new_callable=AsyncMock,
  )
  async def test_unpublish_agent_throttled(self, mock_unpublish):
    mock_unpublish.side_effect = _ThrottleExceeded('slow down')

    result = await unpublish_agent(
        agent_name='chat',
        domain='example.com',
    )

    assert result['success'] is False
    assert result['status'] == 'throttled'


class TestGetDnsAidTools:

  def test_get_dns_aid_tools_no_backend(self):
    # get_dns_aid_tools always returns three tools (discover, publish,
    # unpublish) regardless of backend; with no backend they're bound
    # to backend_name=None.
    tools = get_dns_aid_tools()

    assert len(tools) == 3
    assert all(isinstance(t, FunctionTool) for t in tools)

  def test_get_dns_aid_tools_with_backend(self):
    tools = get_dns_aid_tools(backend_name='mock')

    assert len(tools) == 3
    assert all(isinstance(t, FunctionTool) for t in tools)

  def test_get_dns_aid_tools_unknown_backend_raises(self):
    with pytest.raises(ValueError):
      get_dns_aid_tools(backend_name='not-a-real-backend')

  def test_get_dns_aid_tools_signature_parity(self):
    tools = get_dns_aid_tools(backend_name='mock')

    publish_params = set(inspect.signature(publish_agent).parameters) - {
        'backend_name'
    }
    unpublish_params = set(inspect.signature(unpublish_agent).parameters) - {
        'backend_name'
    }

    funcs = [t.func for t in tools]
    publish_closure = next(
        f for f in funcs if 'endpoint' in inspect.signature(f).parameters
    )
    unpublish_closure = next(
        f
        for f in funcs
        if 'endpoint' not in inspect.signature(f).parameters
        and 'agent_name' in inspect.signature(f).parameters
    )

    assert set(inspect.signature(publish_closure).parameters) == publish_params
    assert (
        set(inspect.signature(unpublish_closure).parameters) == unpublish_params
    )
