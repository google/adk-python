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

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from google.genai.errors import ClientError
import pytest

from ... import testing_utils

mock_error = ClientError(
    code=429,
    response_json={
        'error': {
            'code': 429,
            'message': 'Quota exceeded.',
            'status': 'RESOURCE_EXHAUSTED',
        }
    },
)


class MockPlugin(BasePlugin):
  before_model_text = 'before_model_text from MockPlugin'
  after_model_text = 'after_model_text from MockPlugin'
  on_model_error_text = 'on_model_error_text from MockPlugin'

  def __init__(self, name='mock_plugin'):
    self.name = name
    self.enable_before_model_callback = False
    self.enable_after_model_callback = False
    self.enable_on_model_error_callback = False
    self.before_model_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.before_model_text)]
        )
    )
    self.after_model_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.after_model_text)]
        )
    )
    self.on_model_error_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.on_model_error_text)]
        )
    )

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    if not self.enable_before_model_callback:
      return None
    return self.before_model_response

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    if not self.enable_after_model_callback:
      return None
    return self.after_model_response

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    if not self.enable_on_model_error_callback:
      return None
    return self.on_model_error_response


CANONICAL_MODEL_CALLBACK_CONTENT = 'canonical_model_callback_content'


def canonical_agent_model_callback(**kwargs) -> Optional[LlmResponse]:
  return LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text=CANONICAL_MODEL_CALLBACK_CONTENT)]
      )
  )


@pytest.fixture
def mock_plugin():
  return MockPlugin()


def test_before_model_callback_with_plugin(mock_plugin):
  """Tests that the model response is overridden by before_model_callback from the plugin."""
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  mock_plugin.enable_before_model_callback = True
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', mock_plugin.before_model_text),
  ]


def test_before_model_fallback_canonical_callback(mock_plugin):
  """Tests that when plugin returns empty response, the model response is overridden by the canonical agent model callback."""
  responses = ['model_response']
  mock_plugin.enable_before_model_callback = False
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=canonical_agent_model_callback,
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', CANONICAL_MODEL_CALLBACK_CONTENT),
  ]


def test_before_model_callback_fallback_model(mock_plugin):
  """Tests that the model response is executed normally when both plugin and canonical agent model callback return empty response."""
  responses = ['model_response']
  mock_plugin.enable_before_model_callback = False
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'model_response'),
  ]


def test_on_model_error_callback_with_plugin(mock_plugin):
  """Tests that the model error is handled by the plugin."""
  mock_model = testing_utils.MockModel.create(error=mock_error, responses=[])
  mock_plugin.enable_on_model_error_callback = True
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])

  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', mock_plugin.on_model_error_text),
  ]


def test_on_model_error_callback_fallback_to_runner(mock_plugin):
  """Tests that the model error is not handled and falls back to raise from runner."""
  mock_model = testing_utils.MockModel.create(error=mock_error, responses=[])
  mock_plugin.enable_on_model_error_callback = False
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  try:
    testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  except Exception as e:
    assert e == mock_error


if __name__ == '__main__':
  pytest.main([__file__])
