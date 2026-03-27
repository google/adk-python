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

"""Tests for run_live() reconnection logic in BaseLlmFlow.

These tests verify the exception handling and reconnection behavior of
the outer while-True loop in run_live(), which was fixed in #4996.
"""

import asyncio
import contextlib
from typing import AsyncGenerator
from typing import Optional
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import MAX_RECONNECT_ATTEMPTS
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.context_utils import Aclosing
from google.genai import errors as genai_errors
from websockets.exceptions import ConnectionClosedOK
import pytest

from ... import testing_utils

# Patch target for asyncio.sleep inside base_llm_flow.
_SLEEP_PATCH = 'google.adk.flows.llm_flows.base_llm_flow.asyncio.sleep'


async def _noop_sleep(_delay):
  """No-op replacement for asyncio.sleep in tests."""
  pass


class _LoopBreak(Exception):
  """Sentinel exception to break out of the while-True loop in tests."""
  pass


class _StubConnection(BaseLlmConnection):
  """Minimal connection stub that tracks send_history calls."""

  def __init__(self):
    self.send_history_called = False

  async def send_history(self, history):
    self.send_history_called = True

  async def send_content(self, content):
    pass

  async def send(self, data):
    pass

  async def send_realtime(self, blob):
    pass

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    return
    yield

  async def close(self):
    pass


async def _create_live_context(
    agent: LlmAgent,
    resumption_handle: Optional[str] = None,
) -> InvocationContext:
  """Create an invocation context configured for live testing."""
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='hello',
  )
  ctx.live_request_queue = LiveRequestQueue()
  ctx.live_request_queue.close()
  if resumption_handle:
    ctx.live_session_resumption_handle = resumption_handle
  return ctx


def _setup_flow(agent):
  """Get the agent's flow and patch _send_to_model / _receive_from_model
  to be no-ops so only the connect/exception logic is exercised."""
  flow = agent._llm_flow

  async def noop_send(conn, ctx_arg):
    pass

  async def noop_receive(conn, eid, ctx_arg, req):
    return
    yield

  flow._send_to_model = noop_send
  flow._receive_from_model = noop_receive
  return flow


def _make_connect_fn(behaviors):
  """Create a mock connect() context manager that steps through ``behaviors``.

  Each entry in ``behaviors`` is either:
  - an Exception to raise, or
  - a _StubConnection instance to yield.

  After all behaviors are exhausted, _LoopBreak is raised to terminate
  the while-True loop deterministically.
  """
  call_count = 0

  @contextlib.asynccontextmanager
  async def mock_connect(llm_request):
    nonlocal call_count
    idx = call_count
    call_count += 1
    if idx >= len(behaviors):
      raise _LoopBreak('all behaviors exhausted')
    behavior = behaviors[idx]
    if isinstance(behavior, Exception):
      raise behavior
    yield behavior

  mock_connect.call_count = lambda: call_count
  return mock_connect


# ---------- exception handler tests ----------


@pytest.mark.asyncio
async def test_reconnects_on_connection_closed_with_handle():
  """ConnectionClosedOK + resumption handle => loop continues and reconnects."""
  stub_conn = _StubConnection()
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle='test-handle')
  flow = _setup_flow(agent)

  mock_connect = _make_connect_fn([
      ConnectionClosedOK(None, None),  # 1st: error, should be caught
      stub_conn,                        # 2nd: reconnect succeeds
      # 3rd: _LoopBreak auto-raised
  ])

  with mock.patch(_SLEEP_PATCH, side_effect=_noop_sleep):
    with mock.patch.object(
        type(agent.canonical_model), 'connect', side_effect=mock_connect
    ):
      with pytest.raises(_LoopBreak):
        async with Aclosing(flow.run_live(ctx)) as agen:
          async for event in agen:
            pass

  # Verify the reconnection happened: 1st failed, 2nd succeeded, 3rd broke
  assert mock_connect.call_count() == 3


@pytest.mark.asyncio
async def test_reconnects_on_api_error_with_handle():
  """APIError + resumption handle => loop continues and reconnects."""
  stub_conn = _StubConnection()
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle='test-handle')
  flow = _setup_flow(agent)

  mock_connect = _make_connect_fn([
      genai_errors.APIError(503, {'error': 'connection lost'}),
      stub_conn,
  ])

  with mock.patch(_SLEEP_PATCH, side_effect=_noop_sleep):
    with mock.patch.object(
        type(agent.canonical_model), 'connect', side_effect=mock_connect
    ):
      with pytest.raises(_LoopBreak):
        async with Aclosing(flow.run_live(ctx)) as agen:
          async for event in agen:
            pass

  assert mock_connect.call_count() == 3


@pytest.mark.asyncio
async def test_raises_after_max_retries_connection_closed():
  """ConnectionClosedOK should propagate after MAX_RECONNECT_ATTEMPTS."""
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle='test-handle')
  flow = _setup_flow(agent)

  # Generate MAX_RECONNECT_ATTEMPTS + 1 errors to exhaust retries.
  errors = [
      ConnectionClosedOK(None, None)
      for _ in range(MAX_RECONNECT_ATTEMPTS + 1)
  ]
  mock_connect = _make_connect_fn(errors)

  with mock.patch(_SLEEP_PATCH, side_effect=_noop_sleep):
    with mock.patch.object(
        type(agent.canonical_model), 'connect', side_effect=mock_connect
    ):
      with pytest.raises(ConnectionClosedOK):
        async with Aclosing(flow.run_live(ctx)) as agen:
          async for event in agen:
            pass

  # Should have attempted MAX_RECONNECT_ATTEMPTS times before giving up.
  assert mock_connect.call_count() == MAX_RECONNECT_ATTEMPTS


@pytest.mark.asyncio
async def test_raises_after_max_retries_api_error():
  """APIError should propagate after MAX_RECONNECT_ATTEMPTS."""
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle='test-handle')
  flow = _setup_flow(agent)

  errors = [
      genai_errors.APIError(503, {'error': 'down'})
      for _ in range(MAX_RECONNECT_ATTEMPTS + 1)
  ]
  mock_connect = _make_connect_fn(errors)

  with mock.patch(_SLEEP_PATCH, side_effect=_noop_sleep):
    with mock.patch.object(
        type(agent.canonical_model), 'connect', side_effect=mock_connect
    ):
      with pytest.raises(genai_errors.APIError):
        async with Aclosing(flow.run_live(ctx)) as agen:
          async for event in agen:
            pass

  assert mock_connect.call_count() == MAX_RECONNECT_ATTEMPTS


@pytest.mark.asyncio
async def test_raises_connection_closed_without_handle():
  """ConnectionClosedOK WITHOUT handle => should propagate immediately."""
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle=None)

  mock_connect = _make_connect_fn([ConnectionClosedOK(None, None)])

  with mock.patch.object(
      type(agent.canonical_model), 'connect', side_effect=mock_connect
  ):
    with pytest.raises(ConnectionClosedOK):
      async for _ in agent._llm_flow.run_live(ctx):
        pass


@pytest.mark.asyncio
async def test_raises_api_error_without_handle():
  """APIError WITHOUT handle => should propagate immediately."""
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle=None)

  mock_connect = _make_connect_fn([
      genai_errors.APIError(503, {'error': 'connection lost'}),
  ])

  with mock.patch.object(
      type(agent.canonical_model), 'connect', side_effect=mock_connect
  ):
    with pytest.raises(genai_errors.APIError):
      async for _ in agent._llm_flow.run_live(ctx):
        pass


@pytest.mark.asyncio
async def test_raises_non_api_error_with_handle():
  """Non-APIError + handle => should still propagate."""
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle='test-handle')

  mock_connect = _make_connect_fn([RuntimeError('unexpected')])

  with mock.patch.object(
      type(agent.canonical_model), 'connect', side_effect=mock_connect
  ):
    with pytest.raises(RuntimeError, match='unexpected'):
      async for _ in agent._llm_flow.run_live(ctx):
        pass


# ---------- send_history tests ----------


@pytest.mark.asyncio
async def test_skips_history_on_reconnect():
  """send_history should NOT be called when a resumption handle exists."""
  stub_conn = _StubConnection()
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle='test-handle')
  flow = _setup_flow(agent)

  mock_connect = _make_connect_fn([stub_conn])

  with mock.patch.object(
      type(agent.canonical_model), 'connect', side_effect=mock_connect
  ):
    with pytest.raises(_LoopBreak):
      async with Aclosing(flow.run_live(ctx)) as agen:
        async for event in agen:
          pass

  assert not stub_conn.send_history_called


@pytest.mark.asyncio
async def test_sends_history_without_handle():
  """send_history SHOULD be called when no resumption handle exists."""
  stub_conn = _StubConnection()
  agent = LlmAgent(name='test', model=testing_utils.MockModel.create([]))
  ctx = await _create_live_context(agent, resumption_handle=None)
  flow = _setup_flow(agent)

  mock_connect = _make_connect_fn([stub_conn])

  with mock.patch.object(
      type(agent.canonical_model), 'connect', side_effect=mock_connect
  ):
    with pytest.raises(_LoopBreak):
      async with Aclosing(flow.run_live(ctx)) as agen:
        async for event in agen:
          pass

  assert stub_conn.send_history_called
