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

"""Tests for Progressive SSE Streaming Stage 1 implementation."""

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.genai import types
import pytest


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
  monkeypatch.setenv("ADK_ENABLE_PROGRESSIVE_SSE_STREAMING", "1")
  yield
  monkeypatch.delenv("ADK_ENABLE_PROGRESSIVE_SSE_STREAMING")


def get_weather(location: str) -> dict[str, Any]:
  """Mock weather function for testing.

  Args:
    location: The location to get the weather for.

  Returns:
    A dictionary containing the weather information.
  """
  return {
      "temperature": 22,
      "condition": "sunny",
      "location": location,
  }


class StreamingMockModel(BaseLlm):
  """A mock model that properly streams multiple chunks in a single call."""

  model: str = "streaming-mock"
  stream_chunks: list[LlmResponse] = []
  call_count: int = 0

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["streaming-mock"]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Yield all chunks in a single streaming call."""
    self.call_count += 1

    # Only stream on the first call
    if self.call_count > 1:
      # On subsequent calls, return a simple final response
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text="Task completed.")],
          ),
          partial=False,
      )
      return

    aggregator = StreamingResponseAggregator()

    # Process each chunk through the aggregator
    for chunk in self.stream_chunks:
      # Convert LlmResponse to types.GenerateContentResponse
      # Since we don't have the full response object, we'll simulate it
      async for processed_chunk in aggregator.process_response(
          self._llm_response_to_generate_content_response(chunk)
      ):
        yield processed_chunk

    # Call close() to get the final aggregated response
    if final_response := aggregator.close():
      yield final_response

  def _llm_response_to_generate_content_response(
      self, llm_response: LlmResponse
  ) -> types.GenerateContentResponse:
    """Convert LlmResponse to GenerateContentResponse for aggregator."""
    # Create a minimal GenerateContentResponse that the aggregator can process
    candidates = []
    if llm_response.content:
      candidates.append(
          types.Candidate(
              content=llm_response.content,
              finish_reason=llm_response.finish_reason,
              finish_message=llm_response.error_message,
          )
      )

    return types.GenerateContentResponse(
        candidates=candidates,
        usage_metadata=llm_response.usage_metadata,
    )


def test_progressive_sse_streaming_function_calls():
  """Test that function calls are buffered and executed in parallel."""

  # Setup: Create mock responses simulating streaming chunks
  response1 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="Checking weather...")]
      ),
  )

  response2 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name="get_weather", args={"location": "Tokyo"}
              )
          ],
      ),
  )

  response3 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name="get_weather", args={"location": "New York"}
              )
          ],
      ),
      finish_reason=types.FinishReason.STOP,
  )

  # Create a streaming mock that yields all chunks in one call
  mock_model = StreamingMockModel(
      stream_chunks=[response1, response2, response3]
  )

  agent = Agent(
      name="weather_agent",
      model=mock_model,
      tools=[get_weather],
  )

  run_config = RunConfig(streaming_mode=StreamingMode.SSE)

  # Use the real InMemoryRunner to get access to run_config parameter
  runner = InMemoryRunner(agent=agent)

  # Create session manually
  session = runner.session_service.create_session_sync(
      app_name=runner.app_name, user_id="test_user"
  )

  events = []
  for event in runner.run(
      user_id="test_user",
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[types.Part.from_text(text="What is the weather?")],
      ),
      run_config=run_config,
  ):
    events.append(event)

  # Verify event structure (Stage 1 expectations)
  # Expected events:
  # 0-2: Partial events (text + 2 FCs) - not executed
  # 3: Final aggregated model event (text + 2 FCs) - partial=False
  # 4: Aggregated function response (both get_weather results executed in
  # parallel)
  # 5: Final model response after FCs
  assert len(events) == 6

  assert events[0].partial
  assert events[0].content.parts[0].text == "Checking weather..."

  assert events[1].partial
  assert events[1].content.parts[0].function_call.name == "get_weather"
  assert events[1].content.parts[0].function_call.args["location"] == "Tokyo"

  assert events[2].partial
  assert events[2].content.parts[0].function_call.name == "get_weather"
  assert events[2].content.parts[0].function_call.args["location"] == "New York"

  assert not events[3].partial
  assert events[3].content.parts[0].text == "Checking weather..."
  assert events[3].content.parts[1].function_call.name == "get_weather"
  assert events[3].content.parts[1].function_call.args["location"] == "Tokyo"
  assert events[3].content.parts[2].function_call.name == "get_weather"
  assert events[3].content.parts[2].function_call.args["location"] == "New York"

  assert not events[4].partial
  assert events[4].content.parts[0].function_response.name == "get_weather"
  assert (
      events[4].content.parts[0].function_response.response["location"]
      == "Tokyo"
  )
  assert events[4].content.parts[1].function_response.name == "get_weather"
  assert (
      events[4].content.parts[1].function_response.response["location"]
      == "New York"
  )

  assert not events[5].partial
  assert events[5].content.parts[0].text == "Task completed."


def test_progressive_sse_preserves_part_ordering():
  """Test that part ordering is preserved, especially for thought parts.

  This test verifies that when the model outputs:
  - chunk1(thought1_1)
  - chunk2(thought1_2)
  - chunk3(text1_1)
  - chunk4(text1_2)
  - chunk5(FC1)
  - chunk6(thought2_1)
  - chunk7(thought2_2)
  - chunk8(FC2)

  The final aggregated output should be:
  - Part(thought1)  # thought1_1 + thought1_2 merged
  - Part(text1)     # text1_1 + text1_2 merged
  - Part(FC1)
  - Part(thought2)  # thought2_1 + thought2_2 merged
  - Part(FC2)
  """

  # Create streaming chunks that test the ordering requirement
  chunk1 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part(text="Initial thought part 1. ", thought=True)],
      )
  )

  chunk2 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part(text="Initial thought part 2.", thought=True)],
      )
  )

  chunk3 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_text(text="Let me check Tokyo. ")],
      )
  )

  chunk4 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="And New York too.")]
      )
  )

  chunk5 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name="get_weather", args={"location": "Tokyo"}
              )
          ],
      )
  )

  chunk6 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  text="Now processing second thought part 1. ", thought=True
              )
          ],
      )
  )

  chunk7 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part(text="Second thought part 2.", thought=True)],
      )
  )

  chunk8 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name="get_weather", args={"location": "New York"}
              )
          ],
      ),
      finish_reason=types.FinishReason.STOP,
  )

  mock_model = StreamingMockModel(
      stream_chunks=[
          chunk1,
          chunk2,
          chunk3,
          chunk4,
          chunk5,
          chunk6,
          chunk7,
          chunk8,
      ]
  )

  agent = Agent(
      name="ordering_test_agent",
      model=mock_model,
      tools=[get_weather],
  )

  run_config = RunConfig(streaming_mode=StreamingMode.SSE)

  # Use the real InMemoryRunner to get access to run_config parameter
  runner = InMemoryRunner(agent=agent)

  # Create session manually
  session = runner.session_service.create_session_sync(
      app_name=runner.app_name, user_id="test_user"
  )

  events = []
  for event in runner.run(
      user_id="test_user",
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[types.Part.from_text(text="What is the weather?")],
      ),
      run_config=run_config,
  ):
    events.append(event)

  # Find the final aggregated model event (partial=False, from model)
  aggregated_event = None
  for event in events:
    if (
        not event.partial
        and event.author == "ordering_test_agent"
        and event.content
        and len(event.content.parts) > 2
    ):
      aggregated_event = event
      break

  assert aggregated_event is not None, "Should find an aggregated model event"

  # Verify the part ordering
  parts = aggregated_event.content.parts
  assert len(parts) == 5, f"Expected 5 parts, got {len(parts)}"

  # Part 0: First thought (merged from chunk1 + chunk2)
  assert parts[0].thought
  assert parts[0].text == "Initial thought part 1. Initial thought part 2."

  # Part 1: Regular text (merged from chunk3 + chunk4)
  assert not parts[1].thought
  assert parts[1].text == "Let me check Tokyo. And New York too."

  # Part 2: First function call (from chunk5)
  assert parts[2].function_call.name == "get_weather"
  assert parts[2].function_call.args["location"] == "Tokyo"

  # Part 3: Second thought (merged from chunk6 + chunk7)
  assert parts[3].thought
  assert (
      parts[3].text
      == "Now processing second thought part 1. Second thought part 2."
  )

  # Part 4: Second function call (from chunk8)
  assert parts[4].function_call.name == "get_weather"
  assert parts[4].function_call.args["location"] == "New York"
