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


def test_progressive_sse_streaming_function_call_arguments():
  """Test streaming function call arguments feature.

  This test simulates the streamFunctionCallArguments feature where a function
  call's arguments are streamed incrementally across multiple chunks:

  Chunk 1: FC name + partial location argument ("New ")
  Chunk 2: Continue location argument ("York") -> concatenated to "New York"
  Chunk 3: Add unit argument ("celsius"), willContinue=False -> FC complete

  Expected result: FunctionCall(name="get_weather",
                                 args={"location": "New York", "unit":
                                 "celsius"},
                                 id="fc_001")
  """

  aggregator = StreamingResponseAggregator()

  # Chunk 1: FC name + partial location argument
  chunk1_fc = types.FunctionCall(
      name="get_weather",
      id="fc_001",
      partial_args=[
          types.PartialArg(json_path="$.location", string_value="New ")
      ],
      will_continue=True,
  )
  chunk1 = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role="model", parts=[types.Part(function_call=chunk1_fc)]
              )
          )
      ]
  )

  # Chunk 2: Continue streaming location argument
  chunk2_fc = types.FunctionCall(
      partial_args=[
          types.PartialArg(json_path="$.location", string_value="York")
      ],
      will_continue=True,
  )
  chunk2 = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role="model", parts=[types.Part(function_call=chunk2_fc)]
              )
          )
      ]
  )

  # Chunk 3: Add unit argument, FC complete
  chunk3_fc = types.FunctionCall(
      partial_args=[
          types.PartialArg(json_path="$.unit", string_value="celsius")
      ],
      will_continue=False,  # FC complete
  )
  chunk3 = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role="model", parts=[types.Part(function_call=chunk3_fc)]
              ),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )

  # Process all chunks through aggregator
  processed_chunks = []
  for chunk in [chunk1, chunk2, chunk3]:

    async def process():
      results = []
      async for response in aggregator.process_response(chunk):
        results.append(response)
      return results

    import asyncio

    chunk_results = asyncio.run(process())
    processed_chunks.extend(chunk_results)

  # Get final aggregated response
  final_response = aggregator.close()

  # Verify final aggregated response has complete FC
  assert final_response is not None
  assert len(final_response.content.parts) == 1

  fc_part = final_response.content.parts[0]
  assert fc_part.function_call is not None
  assert fc_part.function_call.name == "get_weather"
  assert fc_part.function_call.id == "fc_001"

  # Verify arguments were correctly assembled from streaming chunks
  args = fc_part.function_call.args
  assert args["location"] == "New York"  # "New " + "York" concatenated
  assert args["unit"] == "celsius"


def test_progressive_sse_preserves_thought_signature():
  """Test that thought_signature is preserved when streaming FC arguments.

  This test verifies that when a streaming function call has a thought_signature
  in the Part, it is correctly preserved in the final aggregated FunctionCall.
  """

  aggregator = StreamingResponseAggregator()

  # Create a thought signature (simulating what Gemini returns)
  # thought_signature is bytes (base64 encoded)
  test_thought_signature = b"test_signature_abc123"

  # Chunk with streaming FC args and thought_signature
  chunk_fc = types.FunctionCall(
      name="add_5_numbers",
      id="fc_003",
      partial_args=[
          types.PartialArg(json_path="$.num1", number_value=10),
          types.PartialArg(json_path="$.num2", number_value=20),
      ],
      will_continue=False,
  )

  # Create Part with both function_call AND thought_signature
  chunk_part = types.Part(
      function_call=chunk_fc, thought_signature=test_thought_signature
  )

  chunk = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(role="model", parts=[chunk_part]),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )

  # Process chunk through aggregator
  async def process():
    results = []
    async for response in aggregator.process_response(chunk):
      results.append(response)
    return results

  import asyncio

  asyncio.run(process())

  # Get final aggregated response
  final_response = aggregator.close()

  # Verify thought_signature was preserved in the Part
  assert final_response is not None
  assert len(final_response.content.parts) == 1

  fc_part = final_response.content.parts[0]
  assert fc_part.function_call is not None
  assert fc_part.function_call.name == "add_5_numbers"

  assert fc_part.thought_signature == test_thought_signature


def test_progressive_sse_handles_empty_function_call():
  """Test that empty function calls are skipped.

  When using streamFunctionCallArguments, Gemini may send an empty
  functionCall: {} as the final chunk to signal streaming completion.
  This test verifies that such empty function calls are properly skipped
  and don't cause errors.
  """

  aggregator = StreamingResponseAggregator()

  # Chunk 1: Streaming FC with partial args
  chunk1_fc = types.FunctionCall(
      name="concat_number_and_string",
      id="fc_001",
      partial_args=[
          types.PartialArg(json_path="$.num", number_value=100),
          types.PartialArg(json_path="$.s", string_value="ADK"),
      ],
      will_continue=False,
  )
  chunk1 = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role="model", parts=[types.Part(function_call=chunk1_fc)]
              )
          )
      ]
  )

  # Chunk 2: Empty function call (streaming end marker)
  chunk2_fc = types.FunctionCall()  # Empty function call
  chunk2 = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role="model", parts=[types.Part(function_call=chunk2_fc)]
              ),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )

  # Process all chunks through aggregator
  async def process():
    results = []
    for chunk in [chunk1, chunk2]:
      async for response in aggregator.process_response(chunk):
        results.append(response)
    return results

  import asyncio

  asyncio.run(process())

  # Get final aggregated response
  final_response = aggregator.close()

  # Verify final response only has the real FC, not the empty one
  assert final_response is not None
  assert len(final_response.content.parts) == 1

  fc_part = final_response.content.parts[0]
  assert fc_part.function_call is not None
  assert fc_part.function_call.name == "concat_number_and_string"
  assert fc_part.function_call.id == "fc_001"

  # Verify arguments
  args = fc_part.function_call.args
  assert args["num"] == 100
  assert args["s"] == "ADK"
