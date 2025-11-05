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
import asyncio
import dataclasses
from unittest import mock

from gepa import adk_agent
from google.adk import runners
from google.adk.agents import base_agent
from google.adk.events import event as event_lib
from google.adk.plugins import base_plugin
from google.genai import types


class _TestPlugin(base_plugin.BasePlugin):

  def __init__(self, outputs):
    super().__init__(name="test-plugin")
    self._model_output_idx = 0
    self.got_llm_requests = []
    self._outputs = outputs

  async def before_model_callback(self, *, callback_context, llm_request):
    self.got_llm_requests.append(llm_request)
    if self._model_output_idx < len(self._outputs):
      out = self._outputs[self._model_output_idx]
      self._model_output_idx += 1
      return out
    return event_lib.Event(
        error_code="empty test list",
        author="agent",
    )


@dataclasses.dataclass
class EnvResponse:
  observation: str
  done: bool
  reward: float


class _TestEnv:

  def __init__(self, responses):
    self._responses = responses
    self._idx = 0

  def step(self, action):
    del action
    if self._idx < len(self._responses):
      resp = self._responses[self._idx]
      self._idx += 1
    else:
      resp = EnvResponse("out-of-bound", done=True, reward=0)
    return resp

  def reset(self, task_index: int):
    del task_index
    return EnvResponse("reset-obs", done=False, reward=42)


def test_default_flow():
  model_outputs = [
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="ab")],
              role="model",
          ),
          author="agent",
      ),
      event_lib.Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="test_tool",
                          args=dict(tool_inputs="fake-tool-inputs"),
                      )
                  )
              ],
              role="model",
          ),
          author="agent",
      ),
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="cd")],
              role="model",
          ),
          author="agent",
      ),
  ]
  events = adk_agent.run_environment_loop(
      instruction="some-instruction",
      env=_TestEnv([
          EnvResponse("some-obs-1", done=False, reward=123),
          EnvResponse("tool-response", done=False, reward=45),
          EnvResponse("some-obs-2", done=False, reward=67),
      ]),
      temperature=0,
      tools=[
          types.FunctionDeclaration(
              name="test_tool",
              description="test_tool",
              parameters={
                  "type": "object",
                  "properties": {
                      "tool_inputs": {
                          "type": "string",
                          "description": "tool_inputs",
                      }
                  },
              },
          )
      ],
      task_index=0,
      max_num_steps=3,
      plugins=[
          _TestPlugin(model_outputs),
      ],
  )
  events = list(events)
  want = [
      "reset-obs",
      "ab",
      "some-obs-1",
      "test_tool",
      "tool-response",
      "cd",
      "some-obs-2",
  ]

  def _extract_from_event(event):
    if not event.content:
      return ""
    if len(event.content.parts) != 1:
      return ""
    part = event.content.parts[0]
    if part.function_call:
      return part.function_call.name
    if part.function_response:
      return part.function_response.response.get("result")
    return part.text

  got = [_extract_from_event(e) for e in events]
  assert got == want

  got_rewards = [e.actions.state_delta.get("reward") for e in events]
  assert got_rewards == [None, None, 123, None, 45, None, 67]


def test_intermediary_step_is_done():
  model_outputs = [
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="ab")],
              role="model",
          ),
          author="agent",
      ),
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="cd")],
              role="model",
          ),
          author="agent",
      ),
  ]
  events = adk_agent.run_environment_loop(
      instruction="some-instruction",
      env=_TestEnv([
          EnvResponse("some-obs-1", done=True, reward=0),
          EnvResponse("some-obs-2", done=False, reward=0),
      ]),
      temperature=0,
      tools=[],
      task_index=0,
      max_num_steps=5,
      plugins=[
          _TestPlugin(model_outputs),
      ],
  )
  want_text = ["reset-obs", "ab", "some-obs-1"]
  got = [e.content.parts[0].text for e in events]
  assert got == want_text


def test_intermediary_tool_step_is_done():
  model_outputs = [
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="ab")],
              role="model",
          ),
          author="agent",
      ),
      event_lib.Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="test_tool",
                          args=dict(tool_inputs="fake-tool-inputs"),
                      )
                  )
              ],
              role="model",
          ),
          author="agent",
      ),
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="cd")],
              role="model",
          ),
          author="agent",
      ),
  ]
  events = adk_agent.run_environment_loop(
      instruction="some-instruction",
      env=_TestEnv([
          EnvResponse("some-obs-1", done=False, reward=123),
          EnvResponse("tool-response", done=True, reward=45),
          EnvResponse("some-obs-2", done=False, reward=67),
      ]),
      temperature=0,
      tools=[
          types.FunctionDeclaration(
              name="test_tool",
              description="test_tool",
              parameters={
                  "type": "object",
                  "properties": {
                      "tool_inputs": {
                          "type": "string",
                          "description": "tool_inputs",
                      }
                  },
              },
          )
      ],
      task_index=0,
      max_num_steps=3,
      plugins=[
          _TestPlugin(model_outputs),
      ],
  )
  events = list(events)
  want = ["reset-obs", "ab", "some-obs-1", "test_tool", "tool-response"]

  def _extract_from_event(event):
    if not event.content:
      return ""
    if len(event.content.parts) != 1:
      return ""
    part = event.content.parts[0]
    if part.function_call:
      return part.function_call.name
    if part.function_response:
      return part.function_response.response.get("result")
    return part.text

  got = [_extract_from_event(e) for e in events]
  assert got == want


def test_llm_request():
  model_outputs = [
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="ab")],
              role="model",
          ),
          author="agent",
      ),
      event_lib.Event(
          content=types.Content(
              parts=[types.Part(text="cd")],
              role="model",
          ),
          author="agent",
      ),
  ]
  test_plugin = _TestPlugin(model_outputs)
  events = adk_agent.run_environment_loop(
      instruction="some-instruction",
      env=_TestEnv([
          EnvResponse("some-obs-1", done=False, reward=123),
          EnvResponse("some-obs-2", done=False, reward=67),
      ]),
      temperature=0.123,
      tools=[],
      task_index=0,
      max_num_steps=2,
      plugins=[test_plugin],
  )
  _ = list(events)

  assert len(test_plugin.got_llm_requests) == 2
  got = test_plugin.got_llm_requests[-1]
  assert "some-instruction" in got.config.system_instruction
  assert got.config.temperature == 0.123
  got_parts = [c.parts[0].text for c in got.contents]
  assert got_parts == ["reset-obs", "ab", "some-obs-1"]


def test_model_name_is_set():
  class _MockAgent(base_agent.BaseAgent):

    async def _run_async_impl(self, ctx):
      pass

  async def _mock_create_session(*args, **kwargs):
    del args, kwargs
    await asyncio.sleep(0.1)
    mock_session = mock.Mock()
    mock.user_id = "fake-user=id"
    mock.id = "fake-session-id"
    return mock_session

  with mock.patch.object(runners, "InMemoryRunner") as mock_runner_cls:
    mock_runner = mock_runner_cls.return_value
    mock_runner.session_service.create_session.side_effect = (
        _mock_create_session
    )
    mock_runner.run.return_value = []
    adk_agent.run_environment_loop(
        instruction="some-instruction",
        env=_TestEnv([]),
        temperature=0.123,
        tools=[],
        task_index=0,
        agent_model="some-test-model",
        plugins=[_TestPlugin([])],
    )
    mock_runner_cls.assert_called_once()
    _, runner_kwargs = mock_runner_cls.call_args
  assert runner_kwargs["agent"].sub_agents[0].model.model == "some-test-model"
