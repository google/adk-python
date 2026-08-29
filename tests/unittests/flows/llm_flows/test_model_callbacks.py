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

from typing import Any
from typing import Optional
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel
import pytest

from ... import testing_utils


class MockBeforeModelCallback(BaseModel):
  mock_response: str

  def __call__(
      self,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> LlmResponse:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.mock_response)]
        )
    )


class MockAfterModelCallback(BaseModel):
  mock_response: str

  def __call__(
      self,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
  ) -> LlmResponse:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.mock_response)]
        )
    )


class MockOnModelCallback(BaseModel):
  mock_response: str

  def __call__(
      self,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> LlmResponse:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.mock_response)]
        )
    )


def noop_callback(**kwargs) -> Optional[LlmResponse]:
  pass


def test_before_model_callback():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=MockBeforeModelCallback(
          mock_response='before_model_callback'
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'before_model_callback'),
  ]


def test_before_model_callback_noop():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=noop_callback,
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'model_response'),
  ]


def test_before_model_callback_end():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=MockBeforeModelCallback(
          mock_response='before_model_callback',
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'before_model_callback'),
  ]


def test_after_model_callback():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_model_callback=MockAfterModelCallback(
          mock_response='after_model_callback'
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'after_model_callback'),
  ]


@pytest.mark.asyncio
async def test_after_model_callback_noop():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_model_callback=noop_callback,
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [('root_agent', 'model_response')]


@pytest.mark.asyncio
async def test_on_model_callback_model_error_noop():
  """Test that the on_model_error_callback is a no-op when the model returns an error."""
  mock_model = testing_utils.MockModel.create(
      responses=[], error=SystemError('error')
  )
  agent = Agent(
      name='root_agent',
      model=mock_model,
      on_model_error_callback=noop_callback,
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  with pytest.raises(SystemError):
    await runner.run_async_with_new_session('test')


@pytest.mark.asyncio
async def test_on_model_callback_model_error_modify_model_response():
  """Test that the on_model_error_callback can modify the model response."""
  mock_model = testing_utils.MockModel.create(
      responses=[], error=SystemError('error')
  )
  agent = Agent(
      name='root_agent',
      model=mock_model,
      on_model_error_callback=MockOnModelCallback(
          mock_response='on_model_error_callback_response'
      ),
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [('root_agent', 'on_model_error_callback_response')]


@pytest.mark.asyncio
async def test_on_model_error_callback_chain_stops_on_recovery_response():
  """Test that model error recovery stops after a non-None response."""
  recovery_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='recovered_model_response')]
      )
  )
  noop_error_callback = mock.Mock(return_value=None)
  recovery_callback = mock.AsyncMock(return_value=recovery_response)
  unexpected_callback = mock.Mock(
      side_effect=AssertionError('callback chain should have stopped')
  )
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(
          responses=[], error=SystemError('error')
      ),
      on_model_error_callback=[
          noop_error_callback,
          recovery_callback,
          unexpected_callback,
      ],
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  events = await runner.run_async_with_new_session('test')

  assert testing_utils.simplify_events(events) == [
      ('root_agent', 'recovered_model_response')
  ]
  noop_error_callback.assert_called_once()
  recovery_callback.assert_awaited_once()
  unexpected_callback.assert_not_called()
