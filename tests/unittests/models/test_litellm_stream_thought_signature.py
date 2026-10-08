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

"""Streaming must keep the thought_signature Gemini attaches to a tool call.

Gemini thinking models attach a thought_signature to each function call. On the
OpenAI-compatible route it rides in extra_content.google.thought_signature (Vertex
also uses provider_specific_fields). If the streaming reassembly drops it, the tool
call ADK stores carries no signature and Gemini rejects the follow-up request:

    Function call is missing a thought_signature in functionCall parts.
"""

from litellm.types.utils import ChatCompletionDeltaToolCall
from litellm.types.utils import Delta
from litellm.types.utils import ModelResponseStream
from litellm.types.utils import StreamingChoices

from google.adk.models.lite_llm import _model_response_to_chunk
from google.adk.models.lite_llm import FunctionChunk

_SIGNATURE = "b3BhcXVlLXNpZ25hdHVyZQ=="


def _streamed_tool_call_response(**tool_call_extras):
  """One streamed chunk carrying a complete tool call."""
  tool_call = ChatCompletionDeltaToolCall(
      index=0,
      id="call_1",
      type="function",
      function={"name": "get_weather", "arguments": '{"city": "Paris"}'},
      **tool_call_extras,
  )
  return ModelResponseStream(
      model="test_model",
      choices=[
          StreamingChoices(
              finish_reason=None,
              delta=Delta(role="assistant", tool_calls=[tool_call]),
          )
      ],
  )


def _function_chunks(response):
  return [
      chunk
      for chunk, _ in _model_response_to_chunk(response)
      if isinstance(chunk, FunctionChunk)
  ]


def test_streaming_keeps_extra_content_thought_signature():
  """The OpenAI-compatible route carries the signature in extra_content.google."""
  response = _streamed_tool_call_response(
      extra_content={"google": {"thought_signature": _SIGNATURE}}
  )

  chunks = _function_chunks(response)

  assert len(chunks) == 1
  assert chunks[0].name == "get_weather"
  # The regression: this was discarded while the streamed call was reassembled.
  assert chunks[0].extra_content == {"google": {"thought_signature": _SIGNATURE}}


def test_streaming_keeps_provider_specific_fields_signature():
  """The Vertex route carries it in provider_specific_fields instead."""
  response = _streamed_tool_call_response(
      provider_specific_fields={"thought_signature": _SIGNATURE}
  )

  chunks = _function_chunks(response)

  assert len(chunks) == 1
  assert chunks[0].provider_specific_fields == {
      "thought_signature": _SIGNATURE
  }


def test_streaming_without_signature_still_works():
  """A plain tool call keeps working; the new fields stay None."""
  response = _streamed_tool_call_response()

  chunks = _function_chunks(response)

  assert len(chunks) == 1
  assert chunks[0].extra_content is None
  assert chunks[0].provider_specific_fields is None
