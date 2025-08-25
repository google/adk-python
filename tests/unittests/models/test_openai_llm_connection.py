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

from __future__ import annotations

import asyncio
import base64
import json
from unittest import mock

from google.adk.models.openai_llm_connection import OpenAILlmConnection
from google.genai import types
import pytest


class _FakeWS:

  def __init__(self, messages: list[str] | None = None):
    self._messages = messages or []
    self.sent: list[str] = []

  def __aiter__(self):
    self._iter = iter(self._messages)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration

  async def send(self, data: str):
    self.sent.append(data)

  async def close(self):
    return


def _collect(async_gen):
  items = []

  async def _run():
    async for it in async_gen:
      items.append(it)

  asyncio.run(_run())
  return items


def _json(obj) -> str:
  return json.dumps(obj, separators=(",", ":"))


@pytest.mark.asyncio
async def test_send_history_triggers_response_on_user_last():
  ws = _FakeWS()
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  history = [
      types.Content(role="user", parts=[types.Part.from_text(text="Hello")]),
  ]
  await conn.send_history(history)
  payloads = [json.loads(p) for p in ws.sent]
  # Expect conversation.item.create then response.create
  assert payloads[-2]["type"] == "conversation.item.create"
  assert payloads[-2]["item"]["role"] == "user"
  assert payloads[-1] == {"type": "response.create"}


@pytest.mark.asyncio
async def test_send_content_text_creates_and_triggers():
  ws = _FakeWS()
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  content = types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
  await conn.send_content(content)
  payloads = [json.loads(p) for p in ws.sent]
  assert payloads[0]["type"] == "conversation.item.create"
  assert payloads[0]["item"]["role"] == "user"
  assert payloads[1] == {"type": "response.create"}


@pytest.mark.asyncio
async def test_send_realtime_appends_audio_buffer():
  ws = _FakeWS()
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  blob = types.Blob(data=b"\x00\xFF", mime_type="audio/pcm")
  await conn.send_realtime(blob)
  sent = json.loads(ws.sent[-1])
  assert sent["type"] == "input_audio_buffer.append"
  assert base64.b64decode(sent["audio"]) == b"\x00\xFF"


def test_receive_text_delta_and_done():
  ws = _FakeWS([
      _json({"type": "response.text.delta", "delta": "Hi"}),
      _json({
          "type": "response.done",
          "response": {"usage": {"input_tokens": 1, "output_tokens": 2}},
      }),
  ])
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  outs = _collect(conn.receive())
  # One final response with text and turn_complete
  assert len(outs) == 1
  assert outs[0].content.parts[0].text == "Hi"
  assert outs[0].turn_complete is True
  assert outs[0].usage_metadata.total_token_count == 3


def test_receive_output_transcript_and_done():
  ws = _FakeWS([
      _json({"type": "response.audio_transcript.delta", "delta": "Bonjour"}),
      _json({
          "type": "response.done",
          "response": {"usage": {"input_tokens": 0, "output_tokens": 7}},
      }),
  ])
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  outs = _collect(conn.receive())
  # We expect one partial delta, then a final transcript flush on done
  assert len(outs) == 2
  assert (
      outs[0].partial is True and outs[0].output_transcription.text == "Bonjour"
  )
  assert outs[1].content.parts[0].text == "Bonjour"
  assert outs[1].turn_complete is True


def test_receive_input_transcript_delta_and_completed():
  ws = _FakeWS([
      _json({
          "type": "conversation.item.input_audio_transcription.delta",
          "delta": "salut",
      }),
      _json({
          "type": "conversation.item.input_audio_transcription.completed",
          "transcript": "salut",
      }),
  ])
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  outs = _collect(conn.receive())
  # First partial delta, then final user content
  assert outs[0].partial is True and outs[0].input_transcription.text == "salut"
  assert outs[1].content.role == "user"
  assert outs[1].content.parts[0].text == "salut"


def test_meta_events_marked_partial():
  ws = _FakeWS([
      _json({"type": "input_audio_buffer.speech_started"}),
      _json({"type": "conversation.item.truncated"}),
  ])
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  outs = _collect(conn.receive())
  assert all(o.partial for o in outs)
  assert all(o.interrupted for o in outs)


def test_function_call_arguments_stream():
  ws = _FakeWS([
      _json({
          "type": "response.output_item.added",
          "item": {
              "type": "function_call",
              "name": "get_time",
              "arguments": '{"city":"P"}',
              "id": "it_1",
              "call_id": "call_1",
          },
      }),
      _json({
          "type": "response.function_call_arguments.delta",
          "delta": "aris",
          "item_id": "it_1",
      }),
      _json({
          "type": "response.function_call_arguments.done",
          "arguments": '{"city":"Paris"}',
          "item_id": "it_1",
      }),
      _json({
          "type": "response.output_item.done",
          "item": {"type": "function_call", "id": "it_1", "call_id": "call_1"},
      }),
  ])
  conn = OpenAILlmConnection(websocket=ws, model_name="gpt-4o-realtime-preview")
  outs = _collect(conn.receive())
  assert len(outs) == 1
  fc = outs[0].content.parts[0].function_call
  assert fc.name == "get_time"
  assert fc.args == {"city": "Paris"}
