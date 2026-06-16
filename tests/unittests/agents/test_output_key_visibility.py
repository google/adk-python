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

"""Tests for LlmAgent output_key visibility in callbacks."""

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.auto_flow import AutoFlow
from google.genai import types
import pytest
from pytest_mock import MockerFixture

from .. import testing_utils

# Standard MockModel will be used instead of SafeMockModel


@pytest.mark.asyncio
async def test_output_key_visibility_in_after_agent_callback():
  """Test that output_key state delta is visible in after_agent_callback."""
  mock_response = "Hello! How can I help you?"
  mock_model = testing_utils.MockModel.create(responses=[mock_response])

  callback_called = False
  captured_state_value = None
  captured_session_state_value = None

  async def check_output_key(callback_context: CallbackContext):
    nonlocal callback_called, captured_state_value, captured_session_state_value
    callback_called = True
    captured_state_value = callback_context.state.get("result", "NOT_FOUND")
    captured_session_state_value = callback_context.session.state.get(
        "result", "NOT_IN_RAW"
    )

  agent = LlmAgent(
      name="my_agent",
      model=mock_model,
      instruction="Reply with a short greeting.",
      output_key="result",
      after_agent_callback=check_output_key,
  )

  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async(new_message="hello")

  assert callback_called, "Callback was not called"

  assert (
      captured_state_value == mock_response
  ), f"Expected {mock_response}, got {captured_state_value}"
  assert (
      captured_session_state_value == mock_response
  ), f"Expected {mock_response}, got {captured_session_state_value}"


@pytest.mark.asyncio
async def test_output_key_visibility_in_run_live(mocker: MockerFixture):
  """Test that output_key state delta is visible in after_agent_callback in run_live."""
  mock_response = "Hello! How can I help you?"
  mock_model = testing_utils.MockModel.create(responses=[mock_response])

  callback_called = False
  captured_state_value = None
  captured_session_state_value = None

  async def check_output_key(callback_context: CallbackContext):
    nonlocal callback_called, captured_state_value, captured_session_state_value
    callback_called = True
    captured_state_value = callback_context.state.get("result", "NOT_FOUND")
    captured_session_state_value = callback_context.session.state.get(
        "result", "NOT_IN_RAW"
    )

  agent = LlmAgent(
      name="my_agent",
      model=mock_model,
      instruction="Reply with a short greeting.",
      output_key="result",
      after_agent_callback=check_output_key,
  )

  async def mock_auto_flow_run_live(self, ctx):
    yield Event(
        id=Event.new_id(),
        invocation_id=ctx.invocation_id,
        author=ctx.agent.name,
        content=types.Content(parts=[types.Part(text=mock_response)]),
    )

  mocker.patch.object(AutoFlow, "run_live", mock_auto_flow_run_live)

  runner = testing_utils.InMemoryRunner(root_agent=agent)
  live_queue = LiveRequestQueue()

  agen = runner.runner.run_live(
      user_id="test_user",
      session_id=runner.session.id,
      live_request_queue=live_queue,
  )

  # Send a message to trigger the agent
  live_queue.send_content(
      types.Content(role="user", parts=[types.Part(text="hello")])
  )

  live_queue.close()

  async for event in agen:
    pass

  assert callback_called, "Callback was not called"
  assert (
      captured_state_value == mock_response
  ), f"Expected {mock_response}, got {captured_state_value}"
  assert (
      captured_session_state_value == mock_response
  ), f"Expected {mock_response}, got {captured_session_state_value}"


@pytest.mark.asyncio
async def test_output_key_visibility_in_sequential_agent():
  """Test that output_key state delta is visible in next agent's before_agent_callback."""
  mock_response = "Hello from agent 1!"
  mock_model = testing_utils.MockModel.create(
      responses=[mock_response, "Hello from agent 2!"]
  )

  callback_called = False
  captured_session_state_value = None

  async def check_before_agent(callback_context: CallbackContext):
    nonlocal callback_called, captured_session_state_value
    callback_called = True
    captured_session_state_value = callback_context.session.state.get(
        "result", "NOT_FOUND"
    )

  agent_1 = LlmAgent(
      name="agent_1",
      model=mock_model,
      instruction="Reply with a short greeting.",
      output_key="result",
  )

  agent_2 = LlmAgent(
      name="agent_2",
      model=mock_model,
      instruction="Reply with a short greeting.",
      before_agent_callback=check_before_agent,
  )

  sequential_agent = SequentialAgent(
      name="seq_agent",
      sub_agents=[agent_1, agent_2],
  )

  runner = testing_utils.InMemoryRunner(root_agent=sequential_agent)

  events = await runner.run_async(new_message="hello")

  assert callback_called, "Callback was not called"
  assert (
      captured_session_state_value == mock_response
  ), f"Expected {mock_response}, got {captured_session_state_value}"
