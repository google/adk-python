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

from __future__ import annotations

from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.flows.llm_flows.functions import AF_FUNCTION_CALL_ID_PREFIX
from google.adk.utils import streaming_utils
from google.genai import types
import pytest


class TestStreamingResponseAggregator:

  @pytest.mark.asyncio
  async def test_process_response_with_text(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Hello")])
            )
        ]
    )
    results = []
    async for r in aggregator.process_response(response):
      results.append(r)
    assert len(results) == 1
    assert results[0].content.parts[0].text == "Hello"
    assert results[0].partial

  @pytest.mark.asyncio
  async def test_process_response_with_thought(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="Thinking...", thought=True)]
                )
            )
        ]
    )
    results = []
    async for r in aggregator.process_response(response):
      results.append(r)
    assert len(results) == 1
    assert results[0].content.parts[0].text == "Thinking..."
    assert results[0].content.parts[0].thought
    assert results[0].partial

  @pytest.mark.asyncio
  async def test_process_response_multiple(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    response1 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Hello ")])
            )
        ]
    )
    response2 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="World!")])
            )
        ]
    )
    async for _ in aggregator.process_response(response1):
      pass
    results = []
    async for r in aggregator.process_response(response2):
      results.append(r)
    assert len(results) == 1
    assert results[0].content.parts[0].text == "World!"

    closed_response = aggregator.close()
    assert closed_response is not None
    assert closed_response.content.parts[0].text == "Hello World!"

  @pytest.mark.asyncio
  async def test_process_response_interleaved_thought_and_text(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    response1 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="I am thinking...", thought=True)]
                )
            )
        ]
    )
    response2 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="Okay, I have a result.")]
                )
            )
        ]
    )
    response3 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text=" The result is 42.")]
                )
            )
        ]
    )

    async for _ in aggregator.process_response(response1):
      pass
    async for _ in aggregator.process_response(response2):
      pass
    async for _ in aggregator.process_response(response3):
      pass

    closed_response = aggregator.close()
    assert closed_response is not None
    assert len(closed_response.content.parts) == 2
    assert closed_response.content.parts[0].text == "I am thinking..."
    assert closed_response.content.parts[0].thought
    assert (
        closed_response.content.parts[1].text
        == "Okay, I have a result. The result is 42."
    )
    assert not closed_response.content.parts[1].thought

  def test_close_with_no_responses(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    closed_response = aggregator.close()
    assert closed_response is None

  @pytest.mark.asyncio
  async def test_close_with_finish_reason(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Hello")]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    async for _ in aggregator.process_response(response):
      pass
    closed_response = aggregator.close()
    assert closed_response is not None
    assert closed_response.content.parts[0].text == "Hello"
    assert closed_response.error_code is None
    assert closed_response.error_message is None

  @pytest.mark.asyncio
  async def test_close_with_error(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Error")]),
                finish_reason=types.FinishReason.RECITATION,
                finish_message="Recitation error",
            )
        ]
    )
    async for _ in aggregator.process_response(response):
      pass
    closed_response = aggregator.close()
    assert closed_response is not None
    assert closed_response.content.parts[0].text == "Error"
    assert closed_response.error_code == types.FinishReason.RECITATION
    assert closed_response.error_message == "Recitation error"

  @pytest.mark.asyncio
  @pytest.mark.parametrize("use_progressive_sse", [True, False])
  async def test_empty_content_produces_empty_final_frame(
      self, use_progressive_sse
  ):
    """A candidate with an empty parts list produces an empty final frame."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, use_progressive_sse
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      response = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[]),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )
      results = []
      async for r in aggregator.process_response(response):
        results.append(r)
      closed_response = aggregator.close()

      assert len(results) == 1
      assert results[0].content is not None
      assert closed_response is not None
      assert closed_response.partial is False
      assert closed_response.content is None
      assert closed_response.finish_reason == types.FinishReason.STOP

  @pytest.mark.asyncio
  @pytest.mark.parametrize("use_progressive_sse", [True, False])
  async def test_prompt_feedback_block_returns_error_frame(
      self, use_progressive_sse
  ):
    """A prompt-level safety block produces a final frame with the error code."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, use_progressive_sse
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      response = types.GenerateContentResponse(
          prompt_feedback=types.GenerateContentResponsePromptFeedback(
              block_reason=types.BlockedReason.SAFETY,
              block_reason_message="Blocked by safety",
          )
      )
      results = []
      async for r in aggregator.process_response(response):
        results.append(r)
      closed_response = aggregator.close()

      assert len(results) == 1
      assert closed_response is not None
      assert closed_response.partial is False
      assert closed_response.error_code == types.BlockedReason.SAFETY
      assert closed_response.error_message == "Blocked by safety"
      assert closed_response.content is None

  @pytest.mark.asyncio
  @pytest.mark.parametrize("use_progressive_sse", [True, False])
  async def test_pure_function_call_behavior_differs_by_mode(
      self, use_progressive_sse
  ):
    """A pure function call yields the part in progressive mode and an empty frame otherwise."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, use_progressive_sse
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      response = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name="my_tool",
                                  args={"x": 1},
                              )
                          )
                      ]
                  ),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )

      results = []
      async for r in aggregator.process_response(response):
        results.append(r)
      closed_response = aggregator.close()

      assert closed_response is not None
      assert closed_response.partial is False

      if use_progressive_sse:
        assert closed_response.content is not None
        assert len(closed_response.content.parts) == 1
        assert closed_response.content.parts[0].function_call.name == "my_tool"
      else:
        assert closed_response.content is None

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "test_id, use_progressive_sse, metadata_type",
      [
          ("grounding_default", False, "grounding"),
          ("grounding_progressive", True, "grounding"),
          ("citation_default", False, "citation"),
          ("citation_progressive", True, "citation"),
      ],
  )
  async def test_close_preserves_metadata(
      self, test_id, use_progressive_sse, metadata_type
  ):
    """close() should carry metadata into the aggregated response."""
    aggregator = streaming_utils.StreamingResponseAggregator()

    metadata = None
    response1 = None
    response2 = None

    if metadata_type == "grounding":
      metadata = types.GroundingMetadata(
          grounding_chunks=[
              types.GroundingChunk(
                  retrieved_context=types.GroundingChunkRetrievedContext(
                      uri="https://example.com/doc1",
                      title="Source",
                  )
              )
          ],
      )
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="Hello ")]),
                  grounding_metadata=metadata,
              )
          ]
      )
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="World!")]),
                  finish_reason=types.FinishReason.STOP,
                  grounding_metadata=metadata,
              )
          ]
      )
    elif metadata_type == "citation":
      metadata = types.CitationMetadata(
          citations=[
              types.Citation(
                  start_index=0,
                  end_index=10,
                  uri="https://example.com/source",
                  title="Source",
              )
          ]
      )
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="Cited text")]),
              )
          ]
      )
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[]),
                  finish_reason=types.FinishReason.STOP,
                  citation_metadata=metadata,
              )
          ]
      )

    async def run_test():
      async for _ in aggregator.process_response(response1):
        pass
      async for _ in aggregator.process_response(response2):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      if use_progressive_sse:
        assert closed_response.partial is False

      if metadata_type == "grounding":
        assert closed_response.grounding_metadata is not None
        assert len(closed_response.grounding_metadata.grounding_chunks) == 1
      elif metadata_type == "citation":
        assert closed_response.citation_metadata is not None
        assert len(closed_response.citation_metadata.citations) == 1

    if use_progressive_sse:
      with temporary_feature_override(
          FeatureName.PROGRESSIVE_SSE_STREAMING, True
      ):
        await run_test()
    else:
      await run_test()

  @pytest.mark.asyncio
  @pytest.mark.parametrize("use_progressive_sse", [False, True])
  async def test_close_propagates_model_version(self, use_progressive_sse):
    """close() should carry model_version into the aggregated response."""
    aggregator = streaming_utils.StreamingResponseAggregator()
    response1 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Hello ")]),
            )
        ],
        model_version="gemini-test-1.0",
    )
    response2 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="World!")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        model_version="gemini-test-1.0",
    )

    async def run_test():
      async for _ in aggregator.process_response(response1):
        pass
      async for _ in aggregator.process_response(response2):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      assert closed_response.model_version == "gemini-test-1.0"

    if use_progressive_sse:
      with temporary_feature_override(
          FeatureName.PROGRESSIVE_SSE_STREAMING, True
      ):
        await run_test()
    else:
      await run_test()

  @pytest.mark.asyncio
  async def test_non_progressive_merged_yield_propagates_model_version(self):
    """The mid-stream merged text event should carry model_version forward.

    In non-progressive mode, when a new non-text response arrives after buffered
    text, the aggregator yields a synthesized merged-text LlmResponse before
    yielding the current partial. That merged event must preserve fields from
    the source response (model_version, grounding_metadata, citation_metadata,
    finish_reason).
    """
    # PROGRESSIVE_SSE_STREAMING defaults to on; explicitly disable it to
    # exercise the non-progressive merged-yield code path under test.
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, False
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      # First: buffer some text.
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[types.Part(text="Hello World!")]
                  ),
              )
          ],
          model_version="gemini-test-2.0",
      )
      # Second: a response without text triggers the merged yield path.
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[]),
                  finish_reason=types.FinishReason.STOP,
              )
          ],
          model_version="gemini-test-2.0",
      )

      results = []
      async for r in aggregator.process_response(response1):
        results.append(r)
      async for r in aggregator.process_response(response2):
        results.append(r)

      # The synthesized merged-text event should carry model_version.
      merged_events = [
          r
          for r in results
          if r.content
          and r.content.parts
          and r.content.parts[0].text == "Hello World!"
          and not r.partial
      ]
      assert merged_events, "expected a merged non-partial text event"
      assert merged_events[0].model_version == "gemini-test-2.0"


class TestFunctionCallIdGeneration:
  """Tests for function call ID generation in streaming mode."""

  @pytest.mark.asyncio
  async def test_non_streaming_fc_generates_id_when_empty(self):
    """Non-streaming function call should get an adk-* ID if LLM didn't provide one."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()

      response = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name="my_tool",
                                  args={"x": 1},
                                  id=None,  # No ID from LLM
                              )
                          )
                      ]
                  ),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )

      async for _ in aggregator.process_response(response):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      fc = closed_response.content.parts[0].function_call
      assert fc.id is not None
      assert fc.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)

  @pytest.mark.asyncio
  async def test_non_streaming_fc_preserves_llm_assigned_id(self):
    """Non-streaming function call should preserve ID if LLM provided one."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()

      response = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name="my_tool",
                                  args={"x": 1},
                                  id="llm-assigned-id",
                              )
                          )
                      ]
                  ),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )

      async for _ in aggregator.process_response(response):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      fc = closed_response.content.parts[0].function_call
      assert fc.id == "llm-assigned-id"

  @pytest.mark.asyncio
  async def test_streaming_fc_generates_consistent_id_across_chunks(self):
    """Streaming function call should have the same ID in partial and final responses."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()

      # First chunk: function call starts
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name="my_tool",
                                  id=None,
                                  partial_args=[
                                      types.PartialArg(
                                          json_path="$.x",
                                          string_value="hello",
                                      )
                                  ],
                                  will_continue=True,
                              )
                          )
                      ]
                  )
              )
          ]
      )

      # Second chunk: function call continues
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name=None,
                                  id=None,
                                  partial_args=[
                                      types.PartialArg(
                                          json_path="$.x",
                                          string_value=" world",
                                      )
                                  ],
                                  will_continue=False,  # Complete
                              )
                          )
                      ]
                  ),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )

      partial_results = []
      async for r in aggregator.process_response(response1):
        partial_results.append(r)
      async for r in aggregator.process_response(response2):
        partial_results.append(r)

      closed_response = aggregator.close()
      assert closed_response is not None
      final_fc = closed_response.content.parts[0].function_call
      assert final_fc.id is not None
      assert final_fc.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      assert final_fc.args == {"x": "hello world"}

      # Verify partial and final events share the same ID
      partial_fc = partial_results[0].content.parts[0].function_call
      assert (
          partial_fc.id == final_fc.id
      ), f"Partial FC ID ({partial_fc.id!r}) != Final FC ID ({final_fc.id!r})"

  @pytest.mark.asyncio
  async def test_multiple_streaming_fcs_get_different_ids(self):
    """Multiple function calls arriving in separate chunks should get different IDs."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()

      # First FC
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name="tool_a",
                                  id=None,
                                  partial_args=[
                                      types.PartialArg(
                                          json_path="$.a", string_value="val_a"
                                      )
                                  ],
                                  will_continue=False,
                              )
                          )
                      ]
                  )
              )
          ]
      )

      # Second FC
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      parts=[
                          types.Part(
                              function_call=types.FunctionCall(
                                  name="tool_b",
                                  id=None,
                                  partial_args=[
                                      types.PartialArg(
                                          json_path="$.b", string_value="val_b"
                                      )
                                  ],
                                  will_continue=False,
                              )
                          )
                      ]
                  ),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )

      async for _ in aggregator.process_response(response1):
        pass
      async for _ in aggregator.process_response(response2):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      assert len(closed_response.content.parts) == 2

      fc_a = closed_response.content.parts[0].function_call
      fc_b = closed_response.content.parts[1].function_call

      assert fc_a.id is not None
      assert fc_b.id is not None
      assert fc_a.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      assert fc_b.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      assert fc_a.id != fc_b.id  # Different IDs for different FCs
