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
  async def test_process_response_with_none_content(self):
    """Test that StreamingResponseAggregator handles content=None."""
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
    assert len(results) == 1
    assert results[0].content is not None

    closed_response = aggregator.close()
    assert closed_response is None
