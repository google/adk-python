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
import contextlib
from typing import AsyncGenerator

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest
from typing_extensions import override  # <-- FIX: Add this import
from websockets import frames  # <-- FIX 1: Import the frames module
from websockets.exceptions import ConnectionClosed

from .. import testing_utils


def test_live_streaming_multi_agent_single_tool():
  """Test live streaming with multi-agent delegation for a single tool call."""
  # --- 1. Mock LLM Responses ---

  # Mock response for the root_agent to delegate the task to the roll_agent.
  # FIX: Use from_function_call to represent delegation to a sub-agent.
  delegation_to_roll_agent = types.Part.from_function_call(
      name='transfer_to_agent', args={'agent_name': 'roll_agent'}
  )

  root_response1 = LlmResponse(
      content=types.Content(role='model', parts=[delegation_to_roll_agent]),
      turn_complete=False,
  )
  root_response2 = LlmResponse(turn_complete=True)
  mock_root_model = testing_utils.MockModel.create(
      [root_response1, root_response2]
  )

  # Mock response for the roll_agent to call its `roll_die` tool.
  function_call = types.Part.from_function_call(
      name='roll_die', args={'sides': 20}
  )
  roll_agent_response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  roll_agent_response2 = LlmResponse(turn_complete=True)
  mock_roll_model = testing_utils.MockModel.create(
      [roll_agent_response1, roll_agent_response2]
  )

  # --- 2. Mock Tools and Agents ---

  def roll_die(sides: int) -> int:
    """Rolls a die and returns a fixed result for testing."""
    return 15

  mock_roll_sub_agent = Agent(
      name='roll_agent',
      model=mock_roll_model,
      tools=[roll_die],
  )

  main_agent = Agent(
      name='root_agent',
      model=mock_root_model,
      sub_agents=[mock_roll_sub_agent],
  )

  # --- 3. Test Runner Setup ---
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )
        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 5:
            return

      try:
        session = self.session
        asyncio.run(asyncio.wait_for(consume_responses(session), timeout=5.0))
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      return collected_responses

  runner = CustomTestRunner(root_agent=main_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Roll a 20-sided die', mime_type='audio/pcm')
  )

  # --- 4. Run and Assert ---
  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, but got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  delegation_found = False
  tool_call_found = False
  tool_response_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call:
          # FIX: Check for the function call that represents delegation.
          if part.function_call.name == 'transfer_to_agent':
            delegation_found = True
            assert part.function_call.args == {'agent_name': 'roll_agent'}

          # Check for the function call made by the roll_agent.
          if part.function_call.name == 'roll_die':
            tool_call_found = True
            assert part.function_call.args['sides'] == 20

        # Check for the result from the executed function.
        if part.function_response and part.function_response.name == 'roll_die':
          tool_response_found = True
          assert part.function_response.response['result'] == 15

  assert delegation_found, 'A function_call event for delegation was not found.'
  assert tool_call_found, 'A function_call event for roll_die was not found.'
  assert tool_response_found, 'A function_response for roll_die was not found.'


def test_live_streaming_connection_error_on_connect():
  """
  Tests that the runner correctly handles a ConnectionClosed exception
  raised from the model's `connect` method during a live run.
  """

  # 1. Create a mock model that fails during the connection phase.
  class MockModelThatFailsToConnect(testing_utils.MockModel):

    @contextlib.asynccontextmanager
    @override
    async def connect(self, llm_request: testing_utils.LlmRequest):
      """Override connect to simulate an immediate connection failure."""

      # FIX 2: Create a proper `Close` frame object first.
      close_frame = frames.Close(
          1007,
          'gemini-live-2.5-flash-preview is not supported in the live api.',
      )

      # FIX 3: Pass the frame object to the `rcvd` parameter of the exception.
      raise ConnectionClosed(rcvd=close_frame, sent=None)

      yield  # pragma: no cover

  # 2. Instantiate the custom mock model.
  mock_model = MockModelThatFailsToConnect(responses=[])

  # 3. Set up the agent and runner.
  agent = Agent(name='test_agent_for_connection_failure', model=mock_model)
  runner = testing_utils.InMemoryRunner(root_agent=agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Initial audio chunk', mime_type='audio/pcm')
  )

  # 4. Assert that `run_live` raises `ConnectionClosed`.
  with pytest.raises(ConnectionClosed) as excinfo:
    runner.run_live(live_request_queue)

  # 5. Verify the details of the exception. The `code` and `reason` are
  #    attributes of the received frame (`rcvd`), not the exception itself.
  assert excinfo.value.rcvd.code == 1007
  assert (
      'is not supported in the live api' in excinfo.value.rcvd.reason
  ), 'The exception reason should match the simulated server error.'
