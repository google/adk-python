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

"""Tests that the Google credential tools do not block the event loop."""

import asyncio
import time
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools._google_credentials import BaseGoogleCredentialsConfig
from google.adk.tools._google_credentials import GoogleCredentialsManager
from google.adk.tools.tool_context import ToolContext
import pytest


def _create_tool_context():
  return ToolContext(
      function_call_id='test-fc-id',
      invocation_context=InvocationContext(
          agent=LlmAgent(name='test'),
          session=Session(app_name='test', user_id='123', id='123'),
          invocation_id='123',
          session_service=InMemorySessionService(),
      ),
  )


@pytest.mark.asyncio
async def test_oauth_flow_exchange_does_not_block_the_event_loop():
  """Test that redeeming the auth code does not freeze the event loop.

  The Google tools refresh the credentials in a thread already, but the
  authorization code exchange was still run on the calling thread, which
  stalled every other request served by the same event loop.
  """
  config = BaseGoogleCredentialsConfig(
      client_id='test_client_id',
      client_secret='test_client_secret',
      scopes=['https://www.googleapis.com/auth/calendar'],
  )
  manager = GoogleCredentialsManager(config)
  tool_context = _create_tool_context()

  # The first pass finds no stored response and asks the user to authorize,
  # which tells us the key the response is stored under.
  assert await manager._perform_oauth_flow(tool_context) is None
  requested = tool_context.actions.requested_auth_configs['test-fc-id']
  credential_key = requested.credential_key

  stored = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          redirect_uri='https://example.com/callback',
          auth_code='test_auth_code',
          auth_response_uri='https://example.com/callback?code=test_auth_code',
      ),
  )
  tool_context.state['temp:' + credential_key] = stored

  mock_client = MagicMock()

  def slow_fetch_token(*args, **kwargs):
    time.sleep(0.5)
    return {'access_token': 'slow_access_token', 'token_type': 'bearer'}

  mock_client.fetch_token.side_effect = slow_fetch_token

  ticks = 0
  done = False

  async def ticker():
    nonlocal ticks
    while not done:
      await asyncio.sleep(0.01)
      ticks += 1

  ticker_task = asyncio.create_task(ticker())
  await asyncio.sleep(0)
  try:
    with patch(
        'google.adk.auth.oauth2_credential_util.OAuth2Session',
        lambda *args, **kwargs: mock_client,
    ):
      creds = await manager._perform_oauth_flow(tool_context)
  finally:
    done = True
    await ticker_task

  assert creds is not None
  assert creds.token == 'slow_access_token'
  # The loop kept running while the token request was in flight. A blocking
  # exchange leaves a single tick, the one that ran before it started.
  assert ticks > 1
