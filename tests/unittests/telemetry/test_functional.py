# Copyright 2025 Google LLC
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

import gc
import sys
from unittest import mock

from google.adk.agents import base_agent
from google.adk.agents.llm_agent import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.telemetry import tracing
from google.adk.tools import FunctionTool
from google.adk.utils.context_utils import Aclosing
from google.genai.types import Part
from opentelemetry.version import __version__
import pytest

from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner


@pytest.fixture
def test_model() -> BaseLlm:
  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name='some_tool', args={}),
          Part.from_text(text='text response'),
      ]
  )
  return mock_model


@pytest.fixture
def test_agent(test_model: BaseLlm) -> Agent:
  def some_tool():
    pass

  root_agent = Agent(
      name='some_root_agent',
      model=test_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )
  return root_agent


@pytest.fixture
async def test_runner(test_agent: Agent) -> TestInMemoryRunner:
  runner = TestInMemoryRunner(test_agent)
  return runner


@pytest.fixture
def mock_start_as_current_span(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
  mock_context_manager = mock.MagicMock()
  mock_context_manager.__enter__.return_value = mock.Mock()
  mock_start_as_current_span = mock.Mock()
  mock_start_as_current_span.return_value = mock_context_manager

  def do_replace(tracer):
    monkeypatch.setattr(
        tracer, 'start_as_current_span', mock_start_as_current_span
    )

  do_replace(tracing.tracer)
  do_replace(base_agent.tracer)

  return mock_start_as_current_span


@pytest.mark.asyncio
async def test_tracer_start_as_current_span(
    test_runner: TestInMemoryRunner,
    mock_start_as_current_span: mock.Mock,
):
  """Test creation of multiple spans in an E2E runner invocation.

  Additionally tests if each async generator invoked is wrapped in Aclosing.
  This is necessary because instrumentation utilizes contextvars, which ran into "ContextVar was created in a different Context" errors,
  when a given coroutine gets indeterminitely suspended.
  """
  firstiter, finalizer = sys.get_asyncgen_hooks()

  def wrapped_firstiter(coro):
    nonlocal firstiter
    assert any(
        isinstance(referrer, Aclosing)
        or isinstance(indirect_referrer, Aclosing)
        for referrer in gc.get_referrers(coro)
        # Some coroutines have a layer of indirection in python 3.9 and 3.10
        for indirect_referrer in gc.get_referrers(referrer)
    ), f'Coro `{coro.__name__}` is not wrapped with Aclosing'
    firstiter(coro)

  sys.set_asyncgen_hooks(wrapped_firstiter, finalizer)

  # Act
  async with Aclosing(test_runner.run_async_with_new_session_agen('')) as agen:
    async for _ in agen:
      pass

  # Assert
  expected_start_as_current_span_calls = [
      mock.call('invocation'),
      mock.call('execute_tool some_tool'),
      mock.call('invoke_agent some_root_agent'),
      mock.call('call_llm'),
      mock.call('call_llm'),
  ]

  mock_start_as_current_span.assert_has_calls(
      expected_start_as_current_span_calls,
      any_order=True,
  )
  assert mock_start_as_current_span.call_count == len(
      expected_start_as_current_span_calls
  )
