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

"""Tests for _receive_from_model connection-close handling in BaseLlmFlow.

google-genai >= 1.62.0 converts websockets.ConnectionClosedOK into
google.genai.errors.APIError with WebSocket close code 1000 (RFC 6455 Normal
Closure). These tests verify that both ConnectionClosedOK and APIError(1000)
are treated as a clean session end rather than an unexpected error, while
APIError with any other code is still re-raised.
"""

from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.genai import errors as genai_errors
import pytest
from websockets.exceptions import ConnectionClosedOK

from ... import testing_utils


class _TestFlow(BaseLlmFlow):
  """Minimal concrete subclass of BaseLlmFlow for testing."""

  pass


async def _collect(agen):
  """Drain an async generator and return all yielded items."""
  items = []
  async for item in agen:
    items.append(item)
  return items


def _make_raising_connection(exc):
  """Return a mock LLM connection whose receive() raises *exc* immediately."""

  async def _raise():
    raise exc
    yield  # make it an async generator

  connection = mock.MagicMock()
  connection.receive = _raise
  return connection


@pytest.fixture
def flow():
  return _TestFlow()


@pytest.fixture
async def invocation_context():
  agent = Agent(name='test_agent', model='mock')
  return await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )


@pytest.fixture
def llm_request():
  return LlmRequest()


# ---------------------------------------------------------------------------
# ConnectionClosedOK — pre-existing behaviour, must remain silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_from_model_connection_closed_ok_is_silent(
    flow, invocation_context, llm_request
):
  """ConnectionClosedOK must be swallowed so the live session ends cleanly."""
  connection = _make_raising_connection(ConnectionClosedOK(None, None))

  events = await _collect(
      flow._receive_from_model(
          connection, 'evt-1', invocation_context, llm_request
      )
  )

  assert events == []


# ---------------------------------------------------------------------------
# APIError(code=1000) — new behaviour for google-genai >= 1.62.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_from_model_api_error_1000_is_silent(
    flow, invocation_context, llm_request
):
  """APIError with code 1000 (Normal Closure) must be swallowed.

  google-genai >= 1.62.0 wraps ConnectionClosedOK as APIError(1000).
  This should be treated identically to ConnectionClosedOK.
  """
  error = genai_errors.APIError(1000, {}, None)
  connection = _make_raising_connection(error)

  events = await _collect(
      flow._receive_from_model(
          connection, 'evt-2', invocation_context, llm_request
      )
  )

  assert events == []


# ---------------------------------------------------------------------------
# APIError with a non-1000 code — must still propagate as a real error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_from_model_api_error_non_1000_is_raised(
    flow, invocation_context, llm_request
):
  """APIError with a code other than 1000 must propagate unchanged."""
  error = genai_errors.APIError(500, {}, None)
  connection = _make_raising_connection(error)

  with pytest.raises(genai_errors.APIError) as exc_info:
    await _collect(
        flow._receive_from_model(
            connection, 'evt-3', invocation_context, llm_request
        )
    )

  assert exc_info.value.code == 500
