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
    """A candidate with empty parts + STOP passes through without an error.

    A terminal empty STOP chunk must not be classified as an error at the
    streaming layer; consumers that batch parts across chunks rely on it
    passing through cleanly.
    """
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
      assert results[0].error_code is None
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
  async def test_close_preserves_usage_metadata_from_earlier_chunk(
      self, use_progressive_sse
  ):
    """A later chunk without usage must not erase an earlier chunk's counts.

    Providers typically report token usage on a single chunk; the trailing
    chunks of the same turn carry none. The aggregated response is the one
    that gets persisted, so it must retain the counts it already saw.
    """
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, use_progressive_sse
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      # First chunk carries the token counts.
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="Hello ")]),
              )
          ],
          usage_metadata=types.GenerateContentResponseUsageMetadata(
              prompt_token_count=10,
              candidates_token_count=5,
              total_token_count=15,
          ),
      )
      # Second chunk carries none.
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="World!")]),
                  finish_reason=types.FinishReason.STOP,
              )
          ],
      )

      async for _ in aggregator.process_response(response1):
        pass
      async for _ in aggregator.process_response(response2):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      assert closed_response.usage_metadata is not None
      assert closed_response.usage_metadata.prompt_token_count == 10
      assert closed_response.usage_metadata.candidates_token_count == 5
      assert closed_response.usage_metadata.total_token_count == 15

  @pytest.mark.asyncio
  @pytest.mark.parametrize("use_progressive_sse", [False, True])
  async def test_close_uses_latest_reported_usage_metadata(
      self, use_progressive_sse
  ):
    """When several chunks report usage, the most recent one wins."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, use_progressive_sse
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      response1 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="Hello ")]),
              )
          ],
          usage_metadata=types.GenerateContentResponseUsageMetadata(
              prompt_token_count=10,
              candidates_token_count=5,
              total_token_count=15,
          ),
      )
      response2 = types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(parts=[types.Part(text="World!")]),
                  finish_reason=types.FinishReason.STOP,
              )
          ],
          usage_metadata=types.GenerateContentResponseUsageMetadata(
              prompt_token_count=10,
              candidates_token_count=9,
              total_token_count=19,
          ),
      )

      async for _ in aggregator.process_response(response1):
        pass
      async for _ in aggregator.process_response(response2):
        pass

      closed_response = aggregator.close()
      assert closed_response is not None
      assert closed_response.usage_metadata is not None
      assert closed_response.usage_metadata.total_token_count == 19

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


def _text_chunk(
    text: str,
    *,
    thought: bool = False,
    signature: bytes | None = None,
    finish: types.FinishReason | None = None,
) -> types.GenerateContentResponse:
  part = types.Part(text=text, thought=thought or None)
  if signature:
    part.thought_signature = signature
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(role="model", parts=[part]),
              finish_reason=finish,
          )
      ]
  )


class TestStreamingThoughtSignature:
  """Signatures must survive the merge of streamed text chunks.

  Consecutive text chunks are joined into a single part that the aggregator
  builds from scratch, so anything the source chunks carried is lost unless
  it is copied across. The model expects its signature back verbatim, and
  without it the reasoning the signature stood for is redone.
  """

  @pytest.mark.asyncio
  async def test_signature_on_merged_text_is_preserved(self):
    aggregator = streaming_utils.StreamingResponseAggregator()
    chunks = [
        _text_chunk("At minute 5 ", signature=b"text-signature"),
        _text_chunk("the presenter speaks.", finish=types.FinishReason.STOP),
    ]
    for chunk in chunks:
      async for _ in aggregator.process_response(chunk):
        pass

    closed = aggregator.close()
    assert closed is not None
    parts = closed.content.parts
    assert len(parts) == 1
    assert parts[0].text == "At minute 5 the presenter speaks."
    assert parts[0].thought_signature == b"text-signature"

  @pytest.mark.asyncio
  async def test_signature_on_a_later_chunk_is_preserved(self):
    """The signature can land on any chunk of the run, not just the first."""
    aggregator = streaming_utils.StreamingResponseAggregator()
    chunks = [
        _text_chunk("At minute 5 "),
        _text_chunk(
            "the presenter speaks.",
            signature=b"late-signature",
            finish=types.FinishReason.STOP,
        ),
    ]
    for chunk in chunks:
      async for _ in aggregator.process_response(chunk):
        pass

    closed = aggregator.close()
    assert closed is not None
    assert closed.content.parts[0].thought_signature == b"late-signature"

  @pytest.mark.asyncio
  async def test_thought_and_answer_keep_their_own_signatures(self):
    """A thought run and an answer run flush separately and must not swap."""
    aggregator = streaming_utils.StreamingResponseAggregator()
    chunks = [
        _text_chunk("Let me check.", thought=True, signature=b"thought-sig"),
        _text_chunk(
            "It is a dog.",
            signature=b"answer-sig",
            finish=types.FinishReason.STOP,
        ),
    ]
    for chunk in chunks:
      async for _ in aggregator.process_response(chunk):
        pass

    closed = aggregator.close()
    assert closed is not None
    parts = closed.content.parts
    assert len(parts) == 2
    assert parts[0].thought
    assert parts[0].thought_signature == b"thought-sig"
    assert parts[1].thought_signature == b"answer-sig"

  @pytest.mark.asyncio
  async def test_content_free_signature_parts_are_kept(self):
    """Server-side media tools return signatures on parts holding nothing."""
    aggregator = streaming_utils.StreamingResponseAggregator()
    sig_only = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(thought_signature=b"call-context")],
                )
            )
        ]
    )
    chunks = [
        _text_chunk("At minute 5 the presenter speaks."),
        sig_only,
        _text_chunk("", finish=types.FinishReason.STOP),
    ]
    for chunk in chunks:
      async for _ in aggregator.process_response(chunk):
        pass

    closed = aggregator.close()
    assert closed is not None
    signatures = [
        p.thought_signature for p in closed.content.parts if p.thought_signature
    ]
    assert signatures == [b"call-context"]


def _streaming_fc_response(
    partial_args: list[types.PartialArg],
    *,
    name: str | None = None,
    will_continue: bool = True,
) -> types.GenerateContentResponse:
  """Builds a response carrying a single streaming function-call chunk."""
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              name=name,
                              partial_args=partial_args,
                              will_continue=will_continue,
                          )
                      )
                  ]
              )
          )
      ]
  )


class TestStreamingFunctionCallArgs:
  """Tests for how streamed function call arguments are accumulated."""

  @pytest.mark.asyncio
  async def test_many_chunks_into_one_arg_accumulate_linearly(self):
    """Chunks of one argument are joined once, not re-concatenated per chunk."""
    chunk_count = 500
    chunk = "x" * 64
    expected = chunk * chunk_count

    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()

      # Total characters written into the args dict. An accumulator that
      # concatenates per chunk writes back every prefix, so the total is
      # quadratic in the final length; buffering writes the value once.
      written_chars = 0
      set_value = aggregator._set_value_by_json_path

      def counting_set_value(json_path, value):
        nonlocal written_chars
        if isinstance(value, str):
          written_chars += len(value)
        set_value(json_path, value)

      aggregator._set_value_by_json_path = counting_set_value

      for i in range(chunk_count):
        response = _streaming_fc_response(
            [types.PartialArg(json_path="$.document", string_value=chunk)],
            name="write_file" if i == 0 else None,
            will_continue=i < chunk_count - 1,
        )
        async for _ in aggregator.process_response(response):
          pass

      closed_response = aggregator.close()

    assert closed_response is not None
    fc = closed_response.content.parts[0].function_call
    assert fc.name == "write_file"
    assert fc.args == {"document": expected}
    assert written_chars <= 2 * len(expected)

  @pytest.mark.asyncio
  async def test_scalar_replaces_buffered_string_on_same_path(self):
    """A scalar arriving after string chunks on the same path wins."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      responses = [
          _streaming_fc_response(
              [types.PartialArg(json_path="$.value", string_value="12")],
              name="my_tool",
          ),
          _streaming_fc_response(
              [types.PartialArg(json_path="$.value", string_value="34")]
          ),
          _streaming_fc_response(
              [types.PartialArg(json_path="$.value", number_value=99)],
              will_continue=False,
          ),
      ]
      for response in responses:
        async for _ in aggregator.process_response(response):
          pass

      closed_response = aggregator.close()

    assert closed_response is not None
    assert closed_response.content.parts[0].function_call.args == {"value": 99}

  @pytest.mark.asyncio
  async def test_interleaved_args_keep_arrival_order(self):
    """Args keep the order their paths first appeared in the stream."""
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      responses = [
          _streaming_fc_response(
              [types.PartialArg(json_path="$.a", string_value="hello")],
              name="my_tool",
          ),
          _streaming_fc_response(
              [types.PartialArg(json_path="$.b", number_value=7)]
          ),
          _streaming_fc_response(
              [types.PartialArg(json_path="$.a", string_value=" world")],
              will_continue=False,
          ),
      ]
      for response in responses:
        async for _ in aggregator.process_response(response):
          pass

      closed_response = aggregator.close()

    assert closed_response is not None
    args = closed_response.content.parts[0].function_call.args
    assert args == {"a": "hello world", "b": 7}
    assert list(args.keys()) == ["a", "b"]


class TestJsonPathHelpers:

  def test_parse_json_path(self):
    assert streaming_utils._parse_json_path("$.foo") == ["foo"]
    assert streaming_utils._parse_json_path("$foo") == ["foo"]
    assert streaming_utils._parse_json_path("foo") == ["foo"]
    assert streaming_utils._parse_json_path("$.foo.bar") == ["foo", "bar"]
    assert streaming_utils._parse_json_path("$.foo[0]") == ["foo", 0]
    assert streaming_utils._parse_json_path("$.foo[10].bar") == [
        "foo",
        10,
        "bar",
    ]
    assert streaming_utils._parse_json_path("$.foo[0][1]") == ["foo", 0, 1]
    assert streaming_utils._parse_json_path("$.items[20000]") == [
        "items",
        20000,
    ]
    assert streaming_utils._parse_json_path("$.items[999999].bar") == [
        "items",
        999999,
        "bar",
    ]
    assert streaming_utils._parse_json_path("$['a.b']") == ["a.b"]
    assert streaming_utils._parse_json_path('$["a.b"]') == ["a.b"]
    assert streaming_utils._parse_json_path("$.foo['bar.baz']") == [
        "foo",
        "bar.baz",
    ]
    assert streaming_utils._parse_json_path("$['foo']['bar']") == ["foo", "bar"]
    assert streaming_utils._parse_json_path("$['a\\'b']") == ["a'b"]
    assert streaming_utils._parse_json_path('$["a\\"b"]') == ['a"b']

  def test_get_value_by_json_path(self):
    target = {"foo": "bar"}
    val, found = streaming_utils._get_value_by_json_path(target, ["foo"])
    assert found is True
    assert val == "bar"

    val, found = streaming_utils._get_value_by_json_path(target, ["baz"])
    assert found is False

    target = {"foo": {"bar": "baz"}}
    val, found = streaming_utils._get_value_by_json_path(target, ["foo", "bar"])
    assert found is True
    assert val == "baz"

    target = {"foo": ["bar", "baz"]}
    val, found = streaming_utils._get_value_by_json_path(target, ["foo", 1])
    assert found is True
    assert val == "baz"

    val, found = streaming_utils._get_value_by_json_path(target, ["foo", 2])
    assert found is False

    val, found = streaming_utils._get_value_by_json_path(target, ["foo", "bar"])
    assert found is False

  def test_set_value_by_json_path(self):
    target = {}
    streaming_utils._set_value_by_json_path(target, ["foo"], 1)
    assert target == {"foo": 1}

    target = {"foo": {}}
    streaming_utils._set_value_by_json_path(target, ["foo", "bar"], 2)
    assert target == {"foo": {"bar": 2}}

    target = {}
    streaming_utils._set_value_by_json_path(target, ["foo", "bar"], 2)
    assert target == {"foo": {"bar": 2}}

    target = {}
    streaming_utils._set_value_by_json_path(target, ["foo", 0], 3)
    assert target == {"foo": [3]}

    target = {}
    streaming_utils._set_value_by_json_path(target, ["foo", 1], 4)
    assert target == {"foo": [None, 4]}

    target = {"foo": [1]}
    streaming_utils._set_value_by_json_path(target, ["foo", 0], 5)
    assert target == {"foo": [5]}

    target = {"foo": [1]}
    streaming_utils._set_value_by_json_path(target, ["foo", 1], 6)
    assert target == {"foo": [1, 6]}

    target = {"foo": "bar"}
    streaming_utils._set_value_by_json_path(target, ["foo", "baz"], 1)
    assert target == {"foo": "bar"}

    target = {"foo": 42}
    streaming_utils._set_value_by_json_path(target, ["foo", 0], 1)
    assert target == {"foo": 42}

    target = {"items": ["first"]}
    streaming_utils._set_value_by_json_path(
        target, streaming_utils._parse_json_path("$.items[20000]"), "scalar"
    )
    assert len(target["items"]) == 1
    assert target["items"] == ["first"]

    streaming_utils._set_value_by_json_path(
        target, streaming_utils._parse_json_path("$.items[2]"), "third"
    )
    assert len(target["items"]) == 3
    assert target["items"] == ["first", None, "third"]


class TestJsonPathTracker:

  def test_tracker_flat_object(self):
    tracker = streaming_utils._JsonPathTracker()

    # Chunk 1: incomplete
    diffs = tracker.handle_chunk('{"foo":')
    assert diffs == []

    # Chunk 2: completes value
    diffs = tracker.handle_chunk('"bar"')
    assert len(diffs) == 1
    assert diffs[0].json_path == "$.foo"
    assert diffs[0].string_value == "bar"

    # Chunk 3: closing
    diffs = tracker.handle_chunk("}")
    assert diffs == []

  def test_tracker_nested_object(self):
    tracker = streaming_utils._JsonPathTracker()

    # Chunk 1: start nested
    diffs = tracker.handle_chunk('{"foo": {')
    assert diffs == []

    # Chunk 2: incomplete key
    diffs = tracker.handle_chunk('"bar":')
    assert diffs == []

    # Chunk 3: value
    diffs = tracker.handle_chunk('"baz"')
    assert len(diffs) == 1
    assert diffs[0].json_path == "$.foo.bar"
    assert diffs[0].string_value == "baz"

    # Chunk 4: closing nested
    diffs = tracker.handle_chunk("}")
    assert diffs == []

    # Chunk 5: closing outer
    diffs = tracker.handle_chunk("}")
    assert diffs == []

  def test_tracker_bracket_array_paths(self):
    tracker = streaming_utils._JsonPathTracker()

    # Chunk 1: start array
    diffs = tracker.handle_chunk('{"tags": [')
    assert diffs == []

    # Chunk 2: first element
    diffs = tracker.handle_chunk('"a"')
    assert len(diffs) == 1
    assert diffs[0].json_path == "$.tags[0]"
    assert diffs[0].string_value == "a"

    # Chunk 3: comma
    diffs = tracker.handle_chunk(",")
    assert diffs == []

    # Chunk 4: second element
    diffs = tracker.handle_chunk('"b"')
    assert len(diffs) == 1
    assert diffs[0].json_path == "$.tags[1]"
    assert diffs[0].string_value == "b"

    # Chunk 5: close array and object
    diffs = tracker.handle_chunk("]}")
    assert diffs == []

  def test_tracker_complete_json_per_delta_reset(self):
    tracker = streaming_utils._JsonPathTracker()

    # Chunk 1: complete JSON
    diffs = tracker.handle_chunk('{"foo": "bar"}')
    assert len(diffs) == 1
    assert diffs[0].json_path == "$.foo"
    assert diffs[0].string_value == "bar"

    # Chunk 2: complete JSON in subsequent payload emits all keys
    diffs = tracker.handle_chunk('{"foo": "bar", "baz": "qux"}')
    assert len(diffs) == 2
    assert diffs[0].json_path == "$.foo"
    assert diffs[0].string_value == "bar"
    assert diffs[1].json_path == "$.baz"
    assert diffs[1].string_value == "qux"

  def test_tracker_empty_dict_and_list(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs = tracker.handle_chunk('{"empty_dict": {}, "empty_list": []}')
    assert len(diffs) == 2
    paths = {d.json_path for d in diffs}
    assert "$.empty_dict" in paths
    assert "$.empty_list" in paths

  def test_tracker_mixed_types(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs = tracker.handle_chunk(
        '{"int": 1, "float": 1.5, "bool": true, "null": null}'
    )

    assert len(diffs) == 4

    diff_dict = {d.json_path: d for d in diffs}

    assert "$.int" in diff_dict
    assert diff_dict["$.int"].number_value == 1

    assert "$.float" in diff_dict
    assert diff_dict["$.float"].number_value == 1.5

    assert "$.bool" in diff_dict
    assert diff_dict["$.bool"].bool_value is True

    assert "$.null" in diff_dict
    assert diff_dict["$.null"].null_value == "NULL_VALUE"

  def test_tracker_fast_path_gated_by_pending_buffer(self):
    tracker = streaming_utils._JsonPathTracker()

    # Chunk 1: start nested object (incomplete)
    diffs = tracker.handle_chunk('{"foo": ')
    assert diffs == []

    # Chunk 2: nested object content, which is a valid dict on its own.
    # It should not trigger the fast path because Chunk 1 is pending.
    diffs = tracker.handle_chunk('{"bar": "baz"}')
    assert len(diffs) == 1
    assert diffs[0].json_path == "$.foo.bar"
    assert diffs[0].string_value == "baz"

    # Chunk 3: close the outer object
    diffs = tracker.handle_chunk("}")
    assert diffs == []

  def test_tracker_buffers_hold_fragments_instead_of_growing_copies(self):
    tracker = streaming_utils._JsonPathTracker()
    fragments = ['{"a": ', '"hello ', 'world", ', '"b": 42}']
    for frag in fragments:
      tracker.handle_chunk(frag)
    assert tracker.accumulated_parts == fragments

  def test_tracker_large_payload_streaming_no_quadratic_slowdown(self):
    tracker = streaming_utils._JsonPathTracker()
    payload_body = "x" * 5000
    diffs = tracker.handle_chunk('{"code":')
    assert diffs == []
    all_diffs = []
    # Start the string value
    diffs = tracker.handle_chunk(' "')
    all_diffs.extend(diffs)
    for i in range(0, len(payload_body), 5):
      chunk = payload_body[i : i + 5]
      diffs = tracker.handle_chunk(chunk)
      all_diffs.extend(diffs)
    closing_diffs = tracker.handle_chunk('"}')
    all_diffs.extend(closing_diffs)
    reconstructed = "".join(d.string_value for d in all_diffs if d.string_value)
    assert reconstructed == payload_body

  def test_tracker_escaped_characters_in_string(self):
    tracker = streaming_utils._JsonPathTracker()
    tracker.handle_chunk('{"code": "hello \\"world\\" and ')
    diffs = tracker.handle_chunk('backslash \\\\ and brace {}}"}')
    assert len(diffs) >= 1
    assert tracker.previous_dict == {
        "code": 'hello "world" and backslash \\ and brace {}}'
    }

  def test_tracker_closing_brace_in_own_delta_resets_properly(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"a": "x"')
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$.a"
    assert diffs_1[0].string_value == "x"

    diffs_2 = tracker.handle_chunk("}")
    assert diffs_2 == []

    diffs_3 = tracker.handle_chunk('{"b": "y"}')
    assert len(diffs_3) == 1
    assert diffs_3[0].json_path == "$.b"
    assert diffs_3[0].string_value == "y"

  def test_tracker_dangling_numeric_literal_not_prematurely_emitted(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"n": 1')
    assert diffs_1 == []

    diffs_2 = tracker.handle_chunk("2")
    assert diffs_2 == []

    diffs_3 = tracker.handle_chunk("3}")
    assert len(diffs_3) == 1
    assert diffs_3[0].json_path == "$.n"
    assert diffs_3[0].number_value == 123
    assert diffs_3[0].will_continue is False

  def test_tracker_deep_recursion_does_not_raise(self):
    tracker = streaming_utils._JsonPathTracker()
    deep_brackets = "[" * 20000
    diffs = tracker.handle_chunk(deep_brackets)
    assert diffs == []

  def test_tracker_deeply_nested_arguments(self):
    tracker = streaming_utils._JsonPathTracker()
    depth = 120
    prefix = '{"a": ' * depth + '{"val": "hello'
    diffs_1 = tracker.handle_chunk(prefix)
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$.a" + ".a" * (depth - 1) + ".val"
    assert diffs_1[0].string_value == "hello"
    assert diffs_1[0].will_continue is True

    suffix = ' world"}' + "}" * depth
    diffs_2 = tracker.handle_chunk(suffix)
    assert len(diffs_2) == 1
    assert diffs_2[0].json_path == "$.a" + ".a" * (depth - 1) + ".val"
    assert diffs_2[0].string_value == " world"
    assert diffs_2[0].will_continue is False

    diffs_3 = tracker.handle_chunk('{"next": "value"}')
    assert len(diffs_3) == 1
    assert diffs_3[0].json_path == "$.next"
    assert diffs_3[0].string_value == "value"

  def test_tracker_will_continue_values(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"text": "hello ')
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$.text"
    assert diffs_1[0].string_value == "hello "
    assert diffs_1[0].will_continue is True

    diffs_2 = tracker.handle_chunk('world"}')
    assert len(diffs_2) == 1
    assert diffs_2[0].json_path == "$.text"
    assert diffs_2[0].string_value == "world"
    assert diffs_2[0].will_continue is False

  def test_tracker_empty_chunk_after_numeric_does_not_raise(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"n": 1')
    assert diffs_1 == []
    diffs_2 = tracker.handle_chunk("")
    assert diffs_2 == []

  def test_tracker_escape_at_chunk_boundary_does_not_corrupt_fast_path(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"text": "line1\\')
    assert diffs_1 == []
    diffs_2 = tracker.handle_chunk("nline2")
    assert len(diffs_2) == 1
    assert diffs_2[0].json_path == "$.text"
    assert diffs_2[0].string_value == "line1\nline2"
    assert diffs_2[0].will_continue is True

    diffs_3 = tracker.handle_chunk('"}')
    assert len(diffs_3) == 1
    assert diffs_3[0].json_path == "$.text"
    assert diffs_3[0].string_value == ""
    assert diffs_3[0].will_continue is False
    assert tracker._in_string is False
    assert tracker.previous_dict == {"text": "line1\nline2"}

  def test_tracker_fast_path_emits_terminal_will_continue_false_on_separate_closing_chunk(
      self,
  ):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"text": "hello ')
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$.text"
    assert diffs_1[0].string_value == "hello "
    assert diffs_1[0].will_continue is True

    diffs_2 = tracker.handle_chunk("world")
    assert len(diffs_2) == 1
    assert diffs_2[0].json_path == "$.text"
    assert diffs_2[0].string_value == "world"
    assert diffs_2[0].will_continue is True

    diffs_3 = tracker.handle_chunk('"}')
    assert len(diffs_3) == 1
    assert diffs_3[0].json_path == "$.text"
    assert diffs_3[0].string_value == ""
    assert diffs_3[0].will_continue is False

  def test_tracker_empty_chunk_in_string_fast_path_returns_empty(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"text": "hello ')
    assert len(diffs_1) == 1
    assert diffs_1[0].string_value == "hello "
    assert diffs_1[0].will_continue is True

    diffs_2 = tracker.handle_chunk("")
    assert diffs_2 == []

    diffs_3 = tracker.handle_chunk("world")
    assert len(diffs_3) == 1
    assert diffs_3[0].string_value == "world"
    assert diffs_3[0].will_continue is True

  def test_tracker_closed_string_in_same_chunk_will_continue_false(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"a": "first", "b": "second ')
    assert len(diffs_1) == 2
    assert diffs_1[0].json_path == "$.a"
    assert diffs_1[0].string_value == "first"
    assert diffs_1[0].will_continue is False

    assert diffs_1[1].json_path == "$.b"
    assert diffs_1[1].string_value == "second "
    assert diffs_1[1].will_continue is True

    diffs_2 = tracker.handle_chunk('word", "c": "third"}')
    assert len(diffs_2) == 2
    assert diffs_2[0].json_path == "$.b"
    assert diffs_2[0].string_value == "word"
    assert diffs_2[0].will_continue is False

    assert diffs_2[1].json_path == "$.c"
    assert diffs_2[1].string_value == "third"
    assert diffs_2[1].will_continue is False

  def test_tracker_closing_string_and_opening_key_does_not_corrupt_fast_path(
      self,
  ):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"key1": "value1')
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$.key1"
    assert diffs_1[0].string_value == "value1"
    assert diffs_1[0].will_continue is True

    # Chunk 2 closes key1 string and opens key2 name
    diffs_2 = tracker.handle_chunk('", "key2')
    assert diffs_2 == []

    # Chunk 3 appends characters to key2 name without quotes
    diffs_3 = tracker.handle_chunk("_name")
    assert diffs_3 == []

    # Chunk 4 completes key2 and its value
    diffs_4 = tracker.handle_chunk('": "value2"}')
    assert len(diffs_4) == 2
    assert diffs_4[0].json_path == "$.key1"
    assert diffs_4[0].string_value == ""
    assert diffs_4[0].will_continue is False
    assert diffs_4[1].json_path == "$.key2_name"
    assert diffs_4[1].string_value == "value2"
    assert diffs_4[1].will_continue is False

    assert tracker.previous_dict == {
        "key1": "value1",
        "key2_name": "value2",
    }

  def test_tracker_empty_containers_do_not_emit_null_value(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs = tracker.handle_chunk('{"empty_dict": {}, "empty_list": []}')
    for diff in diffs:
      assert diff.null_value is None

  def test_tracker_key_with_dots_does_not_duplicate_value(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"a.b": "hel')
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$['a.b']"
    assert diffs_1[0].string_value == "hel"
    assert diffs_1[0].will_continue is True

    diffs_2 = tracker.handle_chunk("lo")
    assert len(diffs_2) == 1
    assert diffs_2[0].json_path == "$['a.b']"
    assert diffs_2[0].string_value == "lo"
    assert diffs_2[0].will_continue is True

    diffs_3 = tracker.handle_chunk('"}')
    assert len(diffs_3) == 1
    assert diffs_3[0].json_path == "$['a.b']"
    assert diffs_3[0].string_value == ""
    assert diffs_3[0].will_continue is False

    assert tracker.previous_dict == {"a.b": "hello"}

  @pytest.mark.asyncio
  async def test_empty_containers_dropped_in_streaming_aggregator(self):
    """Verifies that empty containers are dropped during streaming aggregation.

    Because PartialArg only supports scalar values, empty containers ({}, [])
    cannot be represented with a value field and are omitted by the aggregator.
    """
    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, True
    ):
      aggregator = streaming_utils.StreamingResponseAggregator()
      tracker = streaming_utils._JsonPathTracker()
      diffs = tracker.handle_chunk('{"empty_dict": {}, "empty_list": []}')
      response = _streaming_fc_response(
          diffs, name="test_tool", will_continue=False
      )
      async for _ in aggregator.process_response(response):
        pass
      closed = aggregator.close()
      assert closed is not None
      fc = closed.content.parts[0].function_call
      assert fc.args == {}

  def test_tracker_resets_previous_dict_between_payloads(self):
    tracker = streaming_utils._JsonPathTracker()
    diffs_1 = tracker.handle_chunk('{"val": "hello"}')
    assert len(diffs_1) == 1
    assert diffs_1[0].json_path == "$.val"
    assert diffs_1[0].string_value == "hello"
    assert diffs_1[0].will_continue is False

    diffs_2 = tracker.handle_chunk('{"val": "hello"}')
    assert len(diffs_2) == 1
    assert diffs_2[0].json_path == "$.val"
    assert diffs_2[0].string_value == "hello"
    assert diffs_2[0].will_continue is False
