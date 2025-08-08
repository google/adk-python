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

from typing import Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils


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


class MockAsyncBeforeModelCallback(BaseModel):
  mock_response: str

  async def __call__(
      self,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> LlmResponse:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.mock_response)]
        )
    )


class MockAsyncAfterModelCallback(BaseModel):
  mock_response: str

  async def __call__(
      self,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
  ) -> LlmResponse:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.mock_response)]
        )
    )


def noop_callback(**kwargs) -> Optional[LlmResponse]:
  pass


async def async_noop_callback(**kwargs) -> Optional[LlmResponse]:
  pass


@pytest.mark.asyncio
async def test_before_model_callback():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=MockBeforeModelCallback(
          mock_response='before_model_callback'
      ),
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [
      ('root_agent', 'before_model_callback'),
  ]


@pytest.mark.asyncio
async def test_before_model_callback_noop():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=noop_callback,
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [
      ('root_agent', 'model_response'),
  ]


@pytest.mark.asyncio
async def test_after_model_callback():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_model_callback=MockAfterModelCallback(
          mock_response='after_model_callback'
      ),
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [
      ('root_agent', 'after_model_callback'),
  ]


@pytest.mark.asyncio
async def test_async_before_model_callback():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=MockAsyncBeforeModelCallback(
          mock_response='async_before_model_callback'
      ),
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [
      ('root_agent', 'async_before_model_callback'),
  ]


@pytest.mark.asyncio
async def test_async_before_model_callback_noop():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=async_noop_callback,
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [
      ('root_agent', 'model_response'),
  ]


@pytest.mark.asyncio
async def test_async_after_model_callback():
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_model_callback=MockAsyncAfterModelCallback(
          mock_response='async_after_model_callback'
      ),
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  assert testing_utils.simplify_events(
      await runner.run_async_with_new_session('test')
  ) == [
      ('root_agent', 'async_after_model_callback'),
  ]
