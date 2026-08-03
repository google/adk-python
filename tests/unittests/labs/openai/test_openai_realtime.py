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

"""Hermetic tests for the OpenAI Realtime live-model connection."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest import mock

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.agents import RunConfig
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import StreamingMode
from google.adk.apps import App
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.labs.openai._openai_realtime import _OpenAIRealtimeLlmConnection
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions import InMemorySessionService
from google.genai import types
from openai import AsyncOpenAI
import pytest


class _AsyncCall:
  """Records an SDK resource method call."""

  def __init__(self, session: _FakeRealtimeSession, name: str) -> None:
    self._session = session
    self._name = name

  async def __call__(self, **kwargs: Any) -> None:
    await self._session.record_call(self._name, kwargs)


class _FakeRealtimeSession:
  """Small in-memory double for ``AsyncRealtimeConnection``."""

  def __init__(self, events: list[object] | None = None) -> None:
    self.calls: list[tuple[str, dict[str, Any]]] = []
    self.close_count = 0
    self._events: deque[object] = deque(events or [])
    self.session = SimpleNamespace(update=_AsyncCall(self, "session.update"))
    self.conversation = SimpleNamespace(
        item=SimpleNamespace(
            create=_AsyncCall(self, "conversation.item.create")
        )
    )
    self.input_audio_buffer = SimpleNamespace(
        append=_AsyncCall(self, "input_audio_buffer.append"),
        commit=_AsyncCall(self, "input_audio_buffer.commit"),
        clear=_AsyncCall(self, "input_audio_buffer.clear"),
    )
    self.response = SimpleNamespace(
        create=_AsyncCall(self, "response.create"),
        cancel=_AsyncCall(self, "response.cancel"),
    )

  def __aiter__(self) -> AsyncIterator[object]:
    async def iterate() -> AsyncIterator[object]:
      while self._events:
        yield self._events.popleft()

    return iterate()

  async def close(self) -> None:
    self.close_count += 1

  async def record_call(self, name: str, kwargs: dict[str, Any]) -> None:
    self.calls.append((name, kwargs))

  def calls_named(self, name: str) -> list[dict[str, Any]]:
    return [kwargs for call_name, kwargs in self.calls if call_name == name]


class _RoundTripRealtimeSession(_FakeRealtimeSession):
  """Provider double that waits for a complete parallel-tool round trip."""

  def __init__(
      self,
      first_turn: list[object],
      second_turn: list[object],
      *,
      expected_tool_outputs: int,
  ) -> None:
    super().__init__()
    self._first_turn = first_turn
    self._second_turn = second_turn
    self._expected_tool_outputs = expected_tool_outputs
    self._tool_outputs_ready = asyncio.Event()
    self._phase = 0
    self.response_created_before_all_tool_outputs = False

  def __aiter__(self) -> AsyncIterator[object]:
    async def iterate() -> AsyncIterator[object]:
      if self._phase == 0:
        self._phase = 1
        for event in self._first_turn:
          yield event
      if self._phase == 1:
        await self._tool_outputs_ready.wait()
        self._phase = 2
        for event in self._second_turn:
          yield event
      await asyncio.Event().wait()

    return iterate()

  async def record_call(self, name: str, kwargs: dict[str, Any]) -> None:
    await super().record_call(name, kwargs)
    outputs = [
        call
        for call in self.calls_named("conversation.item.create")
        if call["item"]["type"] == "function_call_output"
    ]
    if name == "response.create" and len(outputs) < self._expected_tool_outputs:
      self.response_created_before_all_tool_outputs = True
    if (
        name == "response.create"
        and len(outputs) == self._expected_tool_outputs
    ):
      self._tool_outputs_ready.set()


class _FakeConnectionManager:

  def __init__(self, realtime: _FakeRealtime, session: _FakeRealtimeSession):
    self._realtime = realtime
    self._session = session

  async def __aenter__(self) -> _FakeRealtimeSession:
    self._realtime.enter_count += 1
    return self._session

  async def __aexit__(self, *unused_args: object) -> None:
    self._realtime.exit_count += 1


class _FakeRealtime:

  def __init__(self, session: _FakeRealtimeSession) -> None:
    self._session = session
    self.models: list[str] = []
    self.enter_count = 0
    self.exit_count = 0

  def connect(self, *, model: str) -> _FakeConnectionManager:
    self.models.append(model)
    return _FakeConnectionManager(self, self._session)


def _openai_llm(
    session: _FakeRealtimeSession,
) -> tuple[OpenAILlm, _FakeRealtime]:
  realtime = _FakeRealtime(session)
  client = mock.Mock(spec=AsyncOpenAI)
  client.realtime = realtime
  return OpenAILlm(model="gpt-realtime", client=client), realtime


def _live_config(
    *,
    automatic_detection: bool = True,
    modalities: list[types.Modality] | None = None,
) -> types.LiveConnectConfig:
  return types.LiveConnectConfig(
      response_modalities=modalities or [types.Modality.AUDIO],
      max_output_tokens=321,
      speech_config=types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(
                  voice_name="marin"
              )
          )
      ),
      input_audio_transcription=types.AudioTranscriptionConfig(
          language_hints=types.LanguageHints(language_codes=["it", "it"]),
          adaptation_phrases=["Agent Development Kit"],
      ),
      realtime_input_config=types.RealtimeInputConfig(
          automatic_activity_detection=types.AutomaticActivityDetection(
              disabled=not automatic_detection,
              prefix_padding_ms=240,
              silence_duration_ms=480,
          )
      ),
  )


def _request(
    *,
    automatic_detection: bool = True,
    modalities: list[types.Modality] | None = None,
) -> LlmRequest:
  tool = types.Tool(
      function_declarations=[
          types.FunctionDeclaration(
              name="get_weather",
              description="Read the weather",
              parameters=types.Schema(
                  type=types.Type.OBJECT,
                  properties={
                      "city": types.Schema(type=types.Type.STRING),
                  },
                  required=["city"],
              ),
          )
      ]
  )
  return LlmRequest(
      model="gpt-realtime-test",
      config=types.GenerateContentConfig(
          system_instruction="Be concise.", tools=[tool]
      ),
      live_connect_config=_live_config(
          automatic_detection=automatic_detection,
          modalities=modalities,
      ),
  )


def _connection(
    session: _FakeRealtimeSession,
) -> _OpenAIRealtimeLlmConnection:
  return _OpenAIRealtimeLlmConnection(
      session,  # type: ignore[arg-type]
      model_version="gpt-realtime-test",
  )


async def _responses(
    session: _FakeRealtimeSession,
) -> list[Any]:
  responses = []
  async for response in _connection(session).receive():
    responses.append(response)
    if response.turn_complete:
      break
  return responses


def _done(
    *,
    status: str = "completed",
    **response: Any,
) -> dict[str, Any]:
  return {"type": "response.done", "response": {"status": status, **response}}


async def test_configure_sends_openai_realtime_session_payload() -> None:
  session = _FakeRealtimeSession()

  await _connection(session).configure(_request())

  payload = session.calls_named("session.update")[0]["session"]
  assert payload["type"] == "realtime"
  assert payload["instructions"] == "Be concise."
  assert payload["output_modalities"] == ["audio"]
  assert payload["max_output_tokens"] == 321
  assert payload["audio"]["input"] == {
      "format": {"type": "audio/pcm", "rate": 24000},
      "turn_detection": {
          "type": "server_vad",
          "create_response": True,
          "interrupt_response": True,
          "prefix_padding_ms": 240,
          "silence_duration_ms": 480,
      },
      "transcription": {
          "model": "gpt-live-transcribe",
          "languages": ["it"],
          "keywords": ["Agent Development Kit"],
      },
  }
  assert payload["audio"]["output"]["voice"] == "marin"
  assert payload["tools"] == [{
      "type": "function",
      "name": "get_weather",
      "description": "Read the weather",
      "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
      },
  }]
  assert payload["tool_choice"] == "auto"


async def test_configure_supports_text_output_and_manual_vad() -> None:
  session = _FakeRealtimeSession()

  await _connection(session).configure(
      _request(
          automatic_detection=False,
          modalities=[types.Modality.TEXT],
      )
  )

  payload = session.calls_named("session.update")[0]["session"]
  assert payload["output_modalities"] == ["text"]
  assert payload["audio"]["input"]["turn_detection"] is None


async def test_configure_defaults_empty_modalities_to_audio() -> None:
  session = _FakeRealtimeSession()
  request = _request()
  request.live_connect_config.response_modalities = []

  await _connection(session).configure(request)

  payload = session.calls_named("session.update")[0]["session"]
  assert payload["output_modalities"] == ["audio"]


async def test_configure_rejects_multiple_or_unsupported_modalities() -> None:
  for modalities in (
      [types.Modality.AUDIO, types.Modality.TEXT],
      [types.Modality.IMAGE],
  ):
    request = _request()
    request.live_connect_config.response_modalities = modalities
    with pytest.raises(ValueError, match="exactly one output modality"):
      await _connection(_FakeRealtimeSession()).configure(request)


async def test_no_interruption_activity_handling_configures_server_vad() -> (
    None
):
  session = _FakeRealtimeSession()
  request = _request()
  request.live_connect_config.realtime_input_config.activity_handling = (
      types.ActivityHandling.NO_INTERRUPTION
  )

  await _connection(session).configure(request)

  payload = session.calls_named("session.update")[0]["session"]
  detection = payload["audio"]["input"]["turn_detection"]
  assert detection["interrupt_response"] is False


def test_async_api_key_provider_is_delegated_to_openai_sdk() -> None:
  called = False

  async def provider() -> str:
    nonlocal called
    called = True
    return "dynamic-key"

  with mock.patch(
      "google.adk.labs.openai._openai_llm.AsyncOpenAI"
  ) as client_class:
    llm = OpenAILlm(model="gpt-realtime", api_key=provider)
    _ = llm._openai_client

  assert called is False
  client_class.assert_called_once_with(api_key=provider)


def test_credentials_and_injected_client_are_not_serialized_or_repr() -> None:
  client = mock.Mock(spec=AsyncOpenAI)
  llm = OpenAILlm(
      model="gpt-realtime",
      api_key="sk-test-secret",
      client=client,
  )

  dumped = llm.model_dump()
  rendered = repr(llm)

  assert "api_key" not in dumped
  assert "client" not in dumped
  assert "sk-test-secret" not in rendered
  assert repr(client) not in rendered


async def test_openai_llm_connect_uses_request_model_and_exits_context() -> (
    None
):
  for raise_inside in (False, True):
    session = _FakeRealtimeSession()
    llm, realtime = _openai_llm(session)

    async def use_connection() -> None:
      async with llm.connect(_request()):
        if raise_inside:
          raise RuntimeError("caller failed")

    if raise_inside:
      with pytest.raises(RuntimeError, match="caller failed"):
        await use_connection()
    else:
      await use_connection()

    assert realtime.models == ["gpt-realtime-test"]
    assert realtime.enter_count == 1
    assert realtime.exit_count == 1
    assert len(session.calls_named("session.update")) == 1


async def test_send_history_replays_items_and_answers_last_user() -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)
  call = types.Part.from_function_call(
      name="get_weather", args={"city": "Rome"}
  )
  call.function_call.id = "call-1"
  history = [
      types.Content(
          role="model",
          parts=[types.Part.from_text(text="Hello"), call],
      ),
      types.Content(role="user", parts=[types.Part.from_text(text="Weather?")]),
  ]

  await connection.send_history(history)

  items = [
      call["item"] for call in session.calls_named("conversation.item.create")
  ]
  assert items == [
      {
          "type": "message",
          "role": "assistant",
          "content": [{"type": "output_text", "text": "Hello"}],
      },
      {
          "type": "function_call",
          "call_id": "call-1",
          "name": "get_weather",
          "arguments": '{"city": "Rome"}',
      },
      {
          "type": "message",
          "role": "user",
          "content": [{"type": "input_text", "text": "Weather?"}],
      },
  ]
  assert len(session.calls_named("response.create")) == 1


async def test_send_history_does_not_answer_last_assistant() -> None:
  session = _FakeRealtimeSession()

  await _connection(session).send_history(
      [types.Content(role="model", parts=[types.Part.from_text(text="Done")])]
  )

  assert not session.calls_named("response.create")


async def test_partial_text_is_coalesced_before_sending() -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)

  await connection._send_content(
      types.Content(parts=[types.Part.from_text(text="hel")]), partial=True
  )
  await connection._send_content(
      types.Content(parts=[types.Part.from_text(text="lo")]), partial=True
  )
  assert not session.calls

  await connection._send_content(
      types.Content(parts=[types.Part.from_text(text="!")]), partial=False
  )

  item = session.calls_named("conversation.item.create")[0]["item"]
  assert item["content"] == [{"type": "input_text", "text": "hello!"}]
  assert len(session.calls_named("response.create")) == 1


async def test_function_output_is_sent_as_conversation_item() -> None:
  session = _FakeRealtimeSession()
  response = types.Part.from_function_response(
      name="get_weather", response={"temperature": 25}
  )
  response.function_response.id = "call-1"

  await _connection(session).send_content(
      types.Content(role="tool", parts=[response])
  )

  item = session.calls_named("conversation.item.create")[0]["item"]
  assert item == {
      "type": "function_call_output",
      "call_id": "call-1",
      "output": '{"temperature": 25}',
  }
  assert len(session.calls_named("response.create")) == 1


async def test_partial_text_does_not_replace_following_function_output() -> (
    None
):
  session = _FakeRealtimeSession()
  connection = _connection(session)
  response = types.Part.from_function_response(
      name="get_weather", response={"temperature": 25}
  )
  response.function_response.id = "call-1"

  await connection._send_content(
      types.Content(parts=[types.Part.from_text(text="context")]),
      partial=True,
  )
  await connection._send_content(
      types.Content(role="tool", parts=[response]), partial=False
  )

  items = [
      call["item"] for call in session.calls_named("conversation.item.create")
  ]
  assert items == [
      {
          "type": "message",
          "role": "user",
          "content": [{"type": "input_text", "text": "context"}],
      },
      {
          "type": "function_call_output",
          "call_id": "call-1",
          "output": '{"temperature": 25}',
      },
  ]
  assert len(session.calls_named("response.create")) == 1


async def test_pcm16_audio_is_base64_encoded_for_openai() -> None:
  session = _FakeRealtimeSession()

  await _connection(session).send_realtime(
      types.Blob(data=b"\x00\x01\x02\x03", mime_type="audio/pcm; rate=24000")
  )

  assert session.calls_named("input_audio_buffer.append") == [
      {"audio": base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")}
  ]


@pytest.mark.parametrize(
    ("blob", "message"),
    [
        (
            types.Blob(data=b"\x00\x00", mime_type="audio/wav;rate=24000"),
            "only supports PCM16",
        ),
        (
            types.Blob(data=b"\x00\x00", mime_type="audio/pcm;rate=16000"),
            "24000 Hz",
        ),
        (
            types.Blob(data=b"\x00", mime_type="audio/pcm;rate=24000"),
            "complete 16-bit samples",
        ),
    ],
)
async def test_invalid_pcm_audio_is_rejected(
    blob: types.Blob,
    message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    await _connection(_FakeRealtimeSession()).send_realtime(blob)


@pytest.mark.parametrize("response_active", [False, True])
async def test_activity_start_cancels_only_an_active_response(
    response_active: bool,
) -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)
  connection._response_active = response_active

  await connection.send_realtime(types.ActivityStart())

  assert bool(session.calls_named("response.cancel")) is response_active
  assert not session.calls_named("input_audio_buffer.clear")


async def test_activity_start_clears_audio_buffer_with_manual_vad() -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)
  connection._automatic_detection = False

  await connection.send_realtime(types.ActivityStart())

  assert len(session.calls_named("input_audio_buffer.clear")) == 1


async def test_no_interruption_prevents_manual_activity_start_cancel() -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)
  request = _request(automatic_detection=False)
  request.live_connect_config.realtime_input_config.activity_handling = (
      types.ActivityHandling.NO_INTERRUPTION
  )
  await connection.configure(request)
  connection._response_active = True

  await connection.send_realtime(types.ActivityStart())

  assert not session.calls_named("response.cancel")
  assert len(session.calls_named("input_audio_buffer.clear")) == 1


@pytest.mark.parametrize("automatic_detection", [False, True])
async def test_activity_end_commits_only_with_manual_vad(
    automatic_detection: bool,
) -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)
  connection._automatic_detection = automatic_detection

  await connection.send_realtime(types.ActivityEnd())

  expected = not automatic_detection
  assert bool(session.calls_named("input_audio_buffer.commit")) is expected
  assert bool(session.calls_named("response.create")) is expected


async def test_audio_stream_end_commits_and_requests_response() -> None:
  session = _FakeRealtimeSession()

  await _connection(session).send_realtime(
      types.LiveClientRealtimeInput(audio_stream_end=True)
  )

  assert len(session.calls_named("input_audio_buffer.commit")) == 1
  assert len(session.calls_named("response.create")) == 1


async def test_receive_maps_audio_and_streaming_text() -> None:
  session = _FakeRealtimeSession([
      {"type": "session.created", "session": {"id": "session-1"}},
      {"type": "response.created"},
      {
          "type": "response.output_audio.delta",
          "delta": base64.b64encode(b"\x01\x02").decode("ascii"),
      },
      {"type": "response.output_text.delta", "delta": "hel"},
      {"type": "response.output_text.delta", "delta": "lo"},
      _done(),
  ])

  responses = await _responses(session)

  audio = responses[0]
  assert audio.content.parts[0].inline_data == types.Blob(  # type: ignore[index,union-attr]
      data=b"\x01\x02", mime_type="audio/pcm;rate=24000"
  )
  assert [response.content.parts[0].text for response in responses[1:3]] == [  # type: ignore[index,union-attr]
      "hel",
      "lo",
  ]
  assert responses[3].partial is False
  assert responses[3].content.parts[0].text == "hello"  # type: ignore[index,union-attr]
  assert responses[4].turn_complete is True
  assert all(response.live_session_id == "session-1" for response in responses)


async def test_receive_discards_invalid_audio_delta() -> None:
  session = _FakeRealtimeSession([
      {"type": "response.output_audio.delta", "delta": "not base64!"},
      _done(),
  ])

  responses = await _responses(session)

  assert len(responses) == 1
  assert responses[0].turn_complete is True


async def test_receive_maps_input_and_output_transcriptions() -> None:
  session = _FakeRealtimeSession([
      {
          "type": "conversation.item.input_audio_transcription.delta",
          "delta": "ciao ",
      },
      {
          "type": "conversation.item.input_audio_transcription.completed",
          "transcript": "ciao mondo",
      },
      {"type": "response.output_audio_transcript.delta", "delta": "salve "},
      {
          "type": "response.output_audio_transcript.done",
          "transcript": "salve a te",
      },
      _done(),
  ])

  responses = await _responses(session)

  assert (
      responses[0].input_transcription.text,
      responses[0].input_transcription.finished,
      responses[0].partial,
  ) == ("ciao ", False, True)
  assert (
      responses[1].input_transcription.text,
      responses[1].input_transcription.finished,
      responses[1].partial,
  ) == ("ciao mondo", True, False)
  assert (
      responses[2].output_transcription.text,
      responses[2].output_transcription.finished,
      responses[2].partial,
  ) == ("salve ", False, True)
  assert (
      responses[3].output_transcription.text,
      responses[3].output_transcription.finished,
      responses[3].partial,
  ) == ("salve a te", True, False)


async def test_tool_call_is_deduplicated_between_delta_done_and_response() -> (
    None
):
  function_call = {
      "type": "function_call",
      "call_id": "call-1",
      "name": "get_weather",
      "arguments": "not-json",
  }
  session = _FakeRealtimeSession([
      {
          **function_call,
          "type": "response.function_call_arguments.done",
      },
      _done(output=[function_call]),
  ])

  responses = await _responses(session)

  calls = [
      response.content.parts[0].function_call
      for response in responses
      if response.content and response.content.parts[0].function_call
  ]
  assert len(calls) == 1
  assert calls[0].id == "call-1"
  assert calls[0].name == "get_weather"
  assert calls[0].args == {}


async def test_completed_response_aggregates_parallel_calls_in_provider_order() -> (
    None
):
  first_call = {
      "type": "function_call",
      "call_id": "call-1",
      "name": "get_weather",
      "arguments": '{"city": "Rome"}',
  }
  second_call = {
      "type": "function_call",
      "call_id": "call-2",
      "name": "get_weather",
      "arguments": '{"city": "Milan"}',
  }
  session = _FakeRealtimeSession([
      {"type": "response.output_text.delta", "delta": "Checking both."},
      {
          **second_call,
          "type": "response.function_call_arguments.done",
      },
      {
          **first_call,
          "type": "response.function_call_arguments.done",
      },
      _done(
          output=[
              {
                  "type": "message",
                  "role": "assistant",
                  "content": [
                      {"type": "output_text", "text": "Checking both."}
                  ],
              },
              first_call,
              second_call,
          ]
      ),
  ])

  responses = await _responses(session)

  final_content = [
      response.content
      for response in responses
      if response.content and response.partial is False
  ]
  assert len(final_content) == 1
  parts = final_content[0].parts
  assert [
      part.text or (part.function_call.id if part.function_call else None)
      for part in parts
  ] == ["Checking both.", "call-1", "call-2"]
  assert [part.function_call.args for part in parts if part.function_call] == [
      {"city": "Rome"},
      {"city": "Milan"},
  ]


async def test_completed_response_maps_usage_including_cached_tokens() -> None:
  session = _FakeRealtimeSession([
      _done(
          usage={
              "input_tokens": 12,
              "output_tokens": 7,
              "total_tokens": 19,
              "input_token_details": {"cached_tokens": 5},
          }
      )
  ])

  response = (await _responses(session))[0]

  assert response.finish_reason == types.FinishReason.STOP
  assert response.usage_metadata.prompt_token_count == 12
  assert response.usage_metadata.candidates_token_count == 7
  assert response.usage_metadata.total_token_count == 19
  assert response.usage_metadata.cached_content_token_count == 5


async def test_response_done_statuses_are_mapped() -> None:
  cases = [
      (_done(status="incomplete"), types.FinishReason.MAX_TOKENS, False, None),
      (_done(status="cancelled"), types.FinishReason.STOP, True, None),
      (
          _done(
              status="failed",
              status_details={
                  "error": {"code": "server_error", "message": "Try again"}
              },
          ),
          types.FinishReason.OTHER,
          False,
          "server_error",
      ),
  ]
  for event, finish_reason, interrupted, error_code in cases:
    response = (await _responses(_FakeRealtimeSession([event])))[0]
    assert response.turn_complete is True
    assert response.finish_reason == finish_reason
    assert response.interrupted is interrupted
    assert response.error_code == error_code


async def test_cancelled_response_consolidates_text_and_interrupts_once() -> (
    None
):
  session = _FakeRealtimeSession([
      {"type": "response.created"},
      {"type": "response.output_text.delta", "delta": "cut off"},
      _done(status="cancelled"),
  ])

  responses = await _responses(session)

  content_responses = [response for response in responses if response.content]
  assert len(content_responses) == 2
  assert content_responses[0].partial is True
  assert content_responses[0].content.parts[0].text == "cut off"
  assert content_responses[1].partial is False
  assert content_responses[1].content.parts[0].text == "cut off"
  assert content_responses[1].interrupted is True
  terminal = responses[-1]
  assert terminal.interrupted is False
  assert terminal.turn_complete is True
  assert terminal.content is None
  assert sum(response.interrupted is True for response in responses) == 1


async def test_text_interruption_waits_for_late_delta_and_emits_once() -> None:
  session = _FakeRealtimeSession([
      {"type": "response.created"},
      {"type": "input_audio_buffer.speech_started", "audio_start_ms": 125},
      {"type": "response.output_text.delta", "delta": "late delta"},
      _done(status="cancelled"),
  ])
  connection = _connection(session)
  await connection.configure(_request(modalities=[types.Modality.TEXT]))

  responses = [response async for response in connection.receive()]

  assert responses[0].voice_activity.voice_activity_type == (
      types.VoiceActivityType.ACTIVITY_START
  )
  assert sum(response.interrupted is True for response in responses) == 1
  final_content = [
      response
      for response in responses
      if response.content and response.partial is False
  ]
  assert len(final_content) == 1
  assert final_content[0].content.parts[0].text == "late delta"
  assert final_content[0].interrupted is True


async def test_speech_started_interrupts_only_during_response() -> None:
  for response_active in (False, True):
    events: list[object] = []
    if response_active:
      events.append({"type": "response.created"})
    events.extend([
        {"type": "input_audio_buffer.speech_started", "audio_start_ms": 125},
        _done(status="cancelled" if response_active else "completed"),
    ])

    responses = await _responses(_FakeRealtimeSession(events))
    response = responses[0]
    assert response.voice_activity.voice_activity_type == (
        types.VoiceActivityType.ACTIVITY_START
    )
    assert response.voice_activity.audio_offset == "125ms"
    if response_active:
      assert responses[1].interrupted is True
    else:
      assert response.interrupted is None


async def test_speech_stopped_reports_activity_end() -> None:
  session = _FakeRealtimeSession([
      {"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 925},
      _done(),
  ])

  response = (await _responses(session))[0]

  assert response.voice_activity.voice_activity_type == (
      types.VoiceActivityType.ACTIVITY_END
  )
  assert response.voice_activity.audio_offset == "925ms"


async def test_no_interruption_speech_started_only_reports_activity() -> None:
  session = _FakeRealtimeSession([
      {"type": "response.created"},
      {"type": "input_audio_buffer.speech_started", "audio_start_ms": 125},
      _done(),
  ])
  connection = _connection(session)
  request = _request()
  request.live_connect_config.realtime_input_config.activity_handling = (
      types.ActivityHandling.NO_INTERRUPTION
  )
  await connection.configure(request)

  responses = []
  async for response in connection.receive():
    responses.append(response)
    if response.turn_complete:
      break

  activity = responses[0]
  assert activity.voice_activity.voice_activity_type == (
      types.VoiceActivityType.ACTIVITY_START
  )
  assert all(response.interrupted is not True for response in responses)


async def test_provider_error_is_recoverable_within_session() -> None:
  session = _FakeRealtimeSession([
      {
          "type": "error",
          "error": {"code": "invalid_value", "message": "Bad request"},
      },
      {"type": "response.output_text.delta", "delta": "recovered"},
      _done(),
  ])

  responses = await _responses(session)

  assert responses[0].error_code == "invalid_value"
  assert responses[0].error_message == "OpenAI Realtime error (invalid_value)"
  assert responses[0].finish_reason == types.FinishReason.OTHER
  assert responses[1].content.parts[0].text == "recovered"  # type: ignore[index,union-attr]
  assert responses[-1].turn_complete is True


async def test_close_is_idempotent() -> None:
  session = _FakeRealtimeSession()
  connection = _connection(session)

  await connection.close()
  await connection.close()

  assert session.close_count == 1


async def test_clean_provider_eof_emits_one_go_away() -> None:
  responses = [
      response
      async for response in _connection(_FakeRealtimeSession()).receive()
  ]

  assert len(responses) == 1
  assert responses[0].go_away == types.LiveServerGoAway(time_left="0s")


async def test_runner_run_live_consumes_openai_realtime_connection() -> None:
  session = _FakeRealtimeSession([
      {"type": "session.created", "session": {"id": "session-runner"}},
      {
          "type": "conversation.item.input_audio_transcription.completed",
          "transcript": "hello",
      },
      {"type": "response.output_text.delta", "delta": "Hi"},
      _done(),
  ])
  llm, realtime = _openai_llm(session)
  agent = Agent(name="assistant", model=llm)
  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="realtime-test", user_id="user", session_id="session"
  )
  runner = Runner(
      app=App(name="realtime-test", root_agent=agent),
      session_service=session_service,
  )
  queue = LiveRequestQueue()
  stream = runner.run_live(
      user_id="user",
      session_id="session",
      live_request_queue=queue,
      run_config=RunConfig(
          streaming_mode=StreamingMode.BIDI,
          response_modalities=[types.Modality.TEXT],
          input_audio_transcription=types.AudioTranscriptionConfig(),
      ),
  )
  events = []

  async def consume_turn() -> None:
    async for event in stream:
      events.append(event)
      if event.turn_complete:
        break

  try:
    await asyncio.wait_for(consume_turn(), timeout=2)
  finally:
    queue.close()
    await stream.aclose()

  assert realtime.enter_count == 1
  assert realtime.exit_count == 1
  assert any(event.author == "user" for event in events)
  assert any(
      event.content
      and event.content.parts
      and event.content.parts[0].text == "Hi"
      for event in events
  )
  assert any(event.turn_complete for event in events)


async def test_runner_round_trips_parallel_tools_before_next_response() -> None:
  first_call = {
      "type": "function_call",
      "call_id": "call-rome",
      "name": "get_weather",
      "arguments": '{"city": "Rome"}',
  }
  second_call = {
      "type": "function_call",
      "call_id": "call-milan",
      "name": "get_weather",
      "arguments": '{"city": "Milan"}',
  }
  session = _RoundTripRealtimeSession(
      [
          {"type": "response.created"},
          _done(output=[first_call, second_call]),
      ],
      [
          {"type": "response.created"},
          {"type": "response.output_text.delta", "delta": "Both checked."},
          _done(
              output=[{
                  "type": "message",
                  "role": "assistant",
                  "content": [{"type": "output_text", "text": "Both checked."}],
              }]
          ),
      ],
      expected_tool_outputs=2,
  )
  tool_invocations: list[str] = []

  def get_weather(city: str) -> dict[str, str]:
    """Returns deterministic weather for a city."""
    tool_invocations.append(city)
    return {"city": city, "condition": "sunny"}

  llm, _ = _openai_llm(session)
  agent = Agent(name="assistant", model=llm, tools=[get_weather])
  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="realtime-tools", user_id="user", session_id="session"
  )
  runner = Runner(
      app=App(name="realtime-tools", root_agent=agent),
      session_service=session_service,
  )
  queue = LiveRequestQueue()
  stream = runner.run_live(
      user_id="user",
      session_id="session",
      live_request_queue=queue,
      run_config=RunConfig(
          streaming_mode=StreamingMode.BIDI,
          response_modalities=[types.Modality.TEXT],
      ),
  )
  events = []
  final_text_seen = False

  async def consume_tool_round_trip() -> None:
    nonlocal final_text_seen
    async for event in stream:
      events.append(event)
      if event.content and any(
          part.text == "Both checked." for part in event.content.parts or []
      ):
        final_text_seen = True
      if final_text_seen and event.turn_complete:
        break

  try:
    await asyncio.wait_for(consume_tool_round_trip(), timeout=2)
  finally:
    queue.close()
    await stream.aclose()

  call_events = [event for event in events if event.get_function_calls()]
  response_events = [
      event for event in events if event.get_function_responses()
  ]
  assert len(call_events) == 1
  assert [call.id for call in call_events[0].get_function_calls()] == [
      "call-rome",
      "call-milan",
  ]
  assert len(response_events) == 1
  assert [
      response.id for response in response_events[0].get_function_responses()
  ] == ["call-rome", "call-milan"]
  assert sorted(tool_invocations) == ["Milan", "Rome"]
  assert session.response_created_before_all_tool_outputs is False
  assert len(session.calls_named("response.create")) == 1
  output_items = [
      call["item"]
      for call in session.calls_named("conversation.item.create")
      if call["item"]["type"] == "function_call_output"
  ]
  assert [item["call_id"] for item in output_items] == [
      "call-rome",
      "call-milan",
  ]
