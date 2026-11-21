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

"""MAX_TOKENS with content must not set error_code in streaming aggregation."""

import asyncio

from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.genai import types


def _make_chunk(
    text: str | None = None,
    finish_reason: types.FinishReason | None = None,
):
  candidates = [
      types.Candidate(
          content=(
              types.Content(
                  role="model", parts=[types.Part.from_text(text=text)]
              )
              if text
              else None
          ),
          finish_reason=finish_reason,
      )
  ]
  return types.GenerateContentResponse(candidates=candidates)


async def _aggregate(chunks):
  aggregator = StreamingResponseAggregator()
  for chunk in chunks:
    async for _ in aggregator.process_response(chunk):
      pass
  return aggregator.close()


def test_max_tokens_with_content_has_no_error_code():
  """MAX_TOKENS with text must report finish_reason but not error_code.

  The streaming aggregator used to set error_code for any non-STOP finish
  reason regardless of content, diverging from the non-streaming path
  (LlmResponse.create) which only sets error_code when there is no content.
  """
  result = asyncio.run(
      _aggregate([
          _make_chunk(
              text="partial answer", finish_reason=types.FinishReason.MAX_TOKENS
          )
      ])
  )
  assert result.error_code is None
  assert result.finish_reason == types.FinishReason.MAX_TOKENS
  assert result.content is not None


def test_no_content_with_error_finish_reason_keeps_error_code():
  """A SAFETY block with no content still carries error_code."""
  result = asyncio.run(
      _aggregate([_make_chunk(finish_reason=types.FinishReason.SAFETY)])
  )
  assert result.error_code == types.FinishReason.SAFETY


def test_stop_with_content_has_no_error_code():
  """A normal STOP response has no error_code (control)."""
  result = asyncio.run(
      _aggregate(
          [_make_chunk(text="done", finish_reason=types.FinishReason.STOP)]
      )
  )
  assert result.error_code is None
  assert result.content is not None
