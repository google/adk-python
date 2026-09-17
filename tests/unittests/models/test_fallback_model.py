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

"""Tests for FallbackModel."""

from __future__ import annotations

import contextlib
import gc
import sys
import threading
from typing import AsyncGenerator

from google.adk.models import FallbackModel
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models._fallback_model import _SNAPSHOT_PRIVATE
from google.adk.models._fallback_model import _status_code
from google.adk.models._fallback_model import _UNSNAPSHOTTED_PRIVATE
from google.adk.models.base_llm import BaseLlm
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.tools.base_tool import BaseTool
from google.genai import errors as genai_errors
from google.genai import types
import httpx
import litellm
import pydantic
import pytest


class _FakeLlm(BaseLlm):
  """A model that yields canned text, and optionally fails."""

  error: Exception | None = None
  """The error to raise, if any."""

  error_after: int = 0
  """How many responses to yield before raising :attr:`error`."""

  seen_request_models: list[str] = pydantic.Field(default_factory=list)
  """The model name carried by each request this model was called with."""

  seen_contents: list[str] = pydantic.Field(default_factory=list)
  """The text of every part in the last request this model was called with."""

  @property
  def capabilities(self) -> LlmCapabilities:
    return LlmCapabilities(output_schema_and_tools=True)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.seen_request_models.append(llm_request.model)
    self.seen_contents = [
        part.text for content in llm_request.contents for part in content.parts
    ]
    for _ in range(self.error_after):
      yield self._response()
    if self.error is not None:
      raise self.error
    yield self._response()

  def _response(self) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text=self.model)])
    )


def _failing(model: str, error: Exception, error_after: int = 0) -> _FakeLlm:
  return _FakeLlm(model=model, error=error, error_after=error_after)


def _rate_limited() -> genai_errors.ClientError:
  return genai_errors.ClientError(429, {'error': {'message': 'slow down'}})


def _request() -> LlmRequest:
  return LlmRequest(
      model='unset',
      contents=[types.Content(role='user', parts=[types.Part(text='hi')])],
      config=types.GenerateContentConfig(),
  )


async def _collect(model: BaseLlm, request: LlmRequest) -> list[LlmResponse]:
  return [response async for response in model.generate_content_async(request)]


def test_model_name_defaults_to_primary():
  fallback = FallbackModel(
      models=['gemini-3.1-pro-preview', 'gemini-3.5-flash']
  )

  assert fallback.model == 'gemini-3.1-pro-preview'


def test_model_name_defaults_to_primary_instance():
  fallback = FallbackModel(
      models=[_FakeLlm(model='primary'), 'gemini-3.5-flash']
  )

  assert fallback.model == 'primary'


def test_setting_model_directly_is_rejected():
  with pytest.raises(pydantic.ValidationError, match='derived from'):
    FallbackModel(models=['gemini-3.1-pro-preview'], model='my-fallback')


def test_model_dump_round_trips():
  fallback = FallbackModel(
      models=['gemini-3.1-pro-preview', 'gemini-3.5-flash']
  )

  restored = FallbackModel.model_validate(fallback.model_dump())

  assert restored.model == 'gemini-3.1-pro-preview'
  assert restored.models == ['gemini-3.1-pro-preview', 'gemini-3.5-flash']


def test_empty_models_is_rejected():
  with pytest.raises(pydantic.ValidationError):
    FallbackModel(models=[])


def test_names_are_not_resolved_at_construction():
  # Names stay lazy, as they are on LlmAgent.model: constructing must not
  # import a provider, so an unknown name surfaces when it is first used.
  fallback = FallbackModel(models=['gemini-3.1-pro-preview', 'gemni-2.5-flash'])

  assert fallback.model == 'gemini-3.1-pro-preview'


@pytest.mark.asyncio
async def test_primary_success_does_not_touch_backup():
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(models=[_FakeLlm(model='primary'), backup])

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['primary']
  assert not backup.seen_request_models


@pytest.mark.asyncio
async def test_falls_back_on_retriable_status():
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(
      models=[_failing('primary', _rate_limited()), backup]
  )

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['backup']
  assert backup.seen_request_models == ['backup']


@pytest.mark.asyncio
async def test_falls_back_through_several_models():
  fallback = FallbackModel(
      models=[
          _failing('first', _rate_limited()),
          _failing('second', genai_errors.ServerError(503, {})),
          _FakeLlm(model='third'),
      ]
  )

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['third']


@pytest.mark.asyncio
async def test_non_retriable_status_propagates():
  backup = _FakeLlm(model='backup')
  invalid_argument = genai_errors.ClientError(400, {})
  fallback = FallbackModel(
      models=[_failing('primary', invalid_argument), backup]
  )

  with pytest.raises(genai_errors.ClientError) as caught:
    await _collect(fallback, _request())

  assert caught.value.code == 400
  assert not backup.seen_request_models


@pytest.mark.asyncio
async def test_error_without_status_propagates():
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(
      models=[_failing('primary', ValueError('bad request')), backup]
  )

  with pytest.raises(ValueError):
    await _collect(fallback, _request())

  assert not backup.seen_request_models


@pytest.mark.asyncio
async def test_streaming_failure_after_first_chunk_does_not_fall_back():
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(
      models=[_failing('primary', _rate_limited(), error_after=2), backup]
  )

  collected = []
  with pytest.raises(genai_errors.ClientError):
    async for response in fallback.generate_content_async(
        _request(), stream=True
    ):
      collected.append(response)

  # The chunks the primary already emitted stay with the caller, and the
  # backup is never asked to finish a turn the primary began.
  assert [r.content.parts[0].text for r in collected] == ['primary', 'primary']
  assert not backup.seen_request_models


@pytest.mark.asyncio
async def test_all_models_failing_raises_last_error():
  fallback = FallbackModel(
      models=[
          _failing('first', _rate_limited()),
          _failing('second', genai_errors.ServerError(503, {})),
      ]
  )

  with pytest.raises(genai_errors.ServerError) as caught:
    await _collect(fallback, _request())

  assert caught.value.code == 503


@pytest.mark.asyncio
async def test_request_model_points_at_the_delegate():
  backup = _FakeLlm(model='backup')
  primary = _failing('primary', _rate_limited())
  fallback = FallbackModel(models=[primary, backup])
  request = _request()

  await _collect(fallback, request)

  assert primary.seen_request_models == ['primary']
  assert backup.seen_request_models == ['backup']
  # The request is left naming the model that actually served it.
  assert request.model == 'backup'


@pytest.mark.asyncio
async def test_retriable_status_codes_are_configurable():
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(
      models=[_failing('primary', genai_errors.ClientError(400, {})), backup],
      retriable_status_codes=frozenset({400}),
  )

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['backup']


@pytest.mark.asyncio
async def test_falls_back_on_httpx_status_error():
  backup = _FakeLlm(model='backup')
  httpx_error = httpx.HTTPStatusError(
      'unavailable',
      request=httpx.Request('POST', 'https://example.test'),
      response=httpx.Response(503),
  )
  fallback = FallbackModel(models=[_failing('primary', httpx_error), backup])

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['backup']


@pytest.mark.asyncio
async def test_falls_back_on_status_code_attribute():
  class _RateLimitError(Exception):
    status_code = 429

  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(
      models=[_failing('primary', _RateLimitError()), backup]
  )

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['backup']


def test_capabilities_come_from_the_primary():
  fallback = FallbackModel(
      models=[_FakeLlm(model='primary'), 'gemini-3.5-flash']
  )

  assert fallback.capabilities.output_schema_and_tools


@pytest.mark.asyncio
async def test_string_entries_are_resolved_once(
    monkeypatch: pytest.MonkeyPatch,
):
  created: list[str] = []

  def _new_llm(model: str) -> BaseLlm:
    created.append(model)
    return _FakeLlm(model=model)

  monkeypatch.setattr(LLMRegistry, 'new_llm', staticmethod(_new_llm))
  fallback = FallbackModel(models=['named-model'])

  responses = await _collect(fallback, _request())
  await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['named-model']
  assert created == ['named-model']


def test_default_status_codes_is_reachable_from_the_class():
  # The guide tells users to widen the defaults, so they need a public handle
  # on them without reaching into the private module.
  widened = FallbackModel.DEFAULT_STATUS_CODES | {529}
  fallback = FallbackModel(
      models=['gemini-3.5-flash'], retriable_status_codes=widened
  )

  assert 529 in fallback.retriable_status_codes
  assert 429 in fallback.retriable_status_codes


def test_status_code_returns_none_without_one():
  assert _status_code(ValueError('nope')) is None


class _FakeConnection(BaseLlmConnection):
  """A live connection that only records that it was opened and closed."""

  def __init__(self, model: str, log: list[str]):
    self.model = model
    self._log = log

  async def send_history(self, history):
    raise NotImplementedError()

  async def send_content(self, content):
    raise NotImplementedError()

  async def send_realtime(self, blob):
    raise NotImplementedError()

  async def receive(self):
    raise NotImplementedError()
    yield

  async def close(self):
    self._log.append(f'closed {self.model}')


class _LiveLlm(_FakeLlm):
  """A model whose live connection can be made to fail on connect."""

  connect_error: Exception | None = None
  log: list[str] = pydantic.Field(default_factory=list)

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest):
    self.log.append(f'connecting {self.model} as {llm_request.model}')
    if self.connect_error is not None:
      raise self.connect_error
    connection = _FakeConnection(self.model, self.log)
    try:
      yield connection
    finally:
      await connection.close()


@pytest.mark.asyncio
async def test_connect_falls_back_when_the_primary_cannot_connect():
  primary = _LiveLlm(model='primary', connect_error=_rate_limited())
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])
  request = _request()

  async with fallback.connect(request) as connection:
    assert connection.model == 'backup'

  assert primary.log == ['connecting primary as primary']
  assert backup.log == ['connecting backup as backup', 'closed backup']


@pytest.mark.asyncio
async def test_connect_does_not_fall_back_on_a_non_retriable_error():
  primary = _LiveLlm(
      model='primary', connect_error=genai_errors.ClientError(400, {})
  )
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])

  with pytest.raises(genai_errors.ClientError):
    async with fallback.connect(_request()):
      pass

  assert not backup.log


@pytest.mark.asyncio
async def test_connect_raises_the_last_error_when_none_connect():
  fallback = FallbackModel(
      models=[
          _LiveLlm(model='first', connect_error=_rate_limited()),
          _LiveLlm(
              model='second', connect_error=genai_errors.ServerError(503, {})
          ),
      ]
  )

  with pytest.raises(genai_errors.ServerError) as caught:
    async with fallback.connect(_request()):
      pass

  assert caught.value.code == 503


@pytest.mark.asyncio
async def test_connect_closes_the_connection_when_the_body_raises():
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(
      models=[_LiveLlm(model='primary', connect_error=_rate_limited()), backup]
  )

  with pytest.raises(RuntimeError):
    async with fallback.connect(_request()):
      raise RuntimeError('caller blew up')

  # The fallback connection is still torn down on the exception path.
  assert backup.log[-1] == 'closed backup'


def _resuming_request(model: str, handle: str) -> LlmRequest:
  request = _request()
  request.model = model
  request.live_connect_config = types.LiveConnectConfig(
      session_resumption=types.SessionResumptionConfig(handle=handle)
  )
  return request


@pytest.mark.asyncio
async def test_a_cross_run_handle_pins_by_name():
  # The primary was down when the session opened, so the backup holds it. The
  # live flow reconnects with the backup's handle; retrying the primary would
  # replay a handle it never issued.
  primary = _LiveLlm(model='primary')
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])

  async with fallback.connect(_resuming_request('backup', 'h-1')) as conn:
    assert conn.model == 'backup'

  assert not primary.log


@pytest.mark.asyncio
async def test_reconnect_does_not_fall_back_to_another_model():
  primary = _LiveLlm(model='primary')
  backup = _LiveLlm(model='backup', connect_error=_rate_limited())
  fallback = FallbackModel(models=[primary, backup])

  # A retriable failure would normally move on; while resuming it must not,
  # because no other model can honour this handle.
  with pytest.raises(genai_errors.ClientError):
    async with fallback.connect(_resuming_request('backup', 'h-1')):
      pass

  assert not primary.log


@pytest.mark.asyncio
async def test_fresh_connection_still_falls_back_with_no_handle():
  primary = _LiveLlm(model='primary', connect_error=_rate_limited())
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])

  async with fallback.connect(_request()) as conn:
    assert conn.model == 'backup'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error_name', ['APIConnectionError', 'APIResponseValidationError']
)
async def test_litellm_misreported_500_does_not_fall_back(error_name: str):
  # litellm hard-codes status 500 on these even though neither is a server
  # error, so the status must not be taken at face value.
  error = getattr(litellm.exceptions, error_name)(
      message='not a server error', llm_provider='openai', model='gpt-4o'
  )
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(models=[_failing('primary', error), backup])

  assert error.status_code == 500
  with pytest.raises(type(error)):
    await _collect(fallback, _request())
  assert not backup.seen_request_models


@pytest.mark.asyncio
async def test_litellm_rate_limit_error_still_falls_back():
  error = litellm.exceptions.RateLimitError(
      message='slow down', llm_provider='openai', model='gpt-4o'
  )
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(models=[_failing('primary', error), backup])

  responses = await _collect(fallback, _request())

  assert [r.content.parts[0].text for r in responses] == ['backup']


def test_misreported_500_check_is_inert_without_litellm(
    monkeypatch: pytest.MonkeyPatch,
):
  # The guard consults litellm only if the process already imported it, so
  # with litellm unloaded the status is read at face value again. Pinning that
  # is what shows the guard, not something else, is doing the suppressing.
  error = litellm.exceptions.APIConnectionError(
      message='connection refused', llm_provider='openai', model='gpt-4o'
  )
  monkeypatch.delitem(sys.modules, 'litellm', raising=False)

  assert _status_code(error) == 500


def test_reconnect_pins_a_prefixed_entry_to_its_delegate():
  # 'gemini:...' loses its prefix when the registry builds it, so the entry
  # string and the delegate's name differ. Matching on the entry string alone
  # would find no owner and offer the handle to every model.
  primary = _LiveLlm(model='primary')
  fallback = FallbackModel(models=[primary, 'gemini:gemini-3.5-flash'])
  built = fallback._delegate('gemini:gemini-3.5-flash')

  assert built.model == 'gemini-3.5-flash'
  assert fallback._candidate_indexes(
      _resuming_request('gemini-3.5-flash', 'h-1')
  ) == [1]
  assert not primary.log


@pytest.mark.asyncio
async def test_reconnect_without_an_identifiable_owner_is_rejected():
  # Offering one model's handle to the others is the outcome the pin exists
  # to prevent, so an unidentifiable owner fails loudly instead. Going through
  # `connect` pins the error the caller actually sees: an empty candidate list
  # would surface as "generator didn't yield" from the context manager.
  primary = _LiveLlm(model='a')
  fallback = FallbackModel(models=[primary, _LiveLlm(model='b')])

  with pytest.raises(ValueError, match='Cannot resume a live session'):
    async with fallback.connect(_resuming_request('someone-else', 'h-1')):
      pass

  assert not primary.log


def test_reconnect_with_no_model_named_is_rejected():
  fallback = FallbackModel(models=[_FakeLlm(model='a'), _FakeLlm(model='b')])
  request = _resuming_request('a', 'h-1')
  request.model = None

  with pytest.raises(ValueError, match='Cannot resume a live session'):
    fallback._candidate_indexes(request)


def test_duplicate_names_are_rejected_only_without_a_remembered_session():
  # Within a run the session is remembered, so same-named entries are fine
  # (see test_reconnect_works_for_two_entries_with_the_same_name). This is the
  # cross-run case, where the name is all there is and it identifies neither.
  fallback = FallbackModel(
      models=[_FakeLlm(model='same'), _FakeLlm(model='same')]
  )

  with pytest.raises(ValueError, match='matches 2'):
    fallback._candidate_indexes(_resuming_request('same', 'h-1'))


@pytest.mark.asyncio
async def test_first_connection_with_resumption_enabled_still_falls_back():
  # A live run that asks for session resumption carries the config from the
  # start, with the handle still empty. That is a fresh connection, not a
  # reconnect, so it must keep its failover.
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(
      models=[_LiveLlm(model='primary', connect_error=_rate_limited()), backup]
  )
  request = _request()
  request.live_connect_config = types.LiveConnectConfig(
      session_resumption=types.SessionResumptionConfig()
  )

  async with fallback.connect(request) as connection:
    assert connection.model == 'backup'


class _MutatingLlm(_FakeLlm):
  """A model that edits the request in place, as the real ones do."""

  note: str = 'edited'

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    llm_request.contents.append(
        types.Content(role='user', parts=[types.Part(text=self.note)])
    )
    llm_request.config.temperature = 0.99
    async for response in super().generate_content_async(llm_request, stream):
      yield response


@pytest.mark.asyncio
async def test_a_failed_model_does_not_leak_its_edits_to_the_next():
  # Real models append a user turn and preprocess tools before sending. A
  # model that then fails must not hand those edits to the backup as if the
  # caller had written them.
  backup = _FakeLlm(model='backup')
  primary = _MutatingLlm(
      model='primary', note='injected by primary', error=_rate_limited()
  )
  fallback = FallbackModel(models=[primary, backup])
  request = _request()
  original_turns = len(request.contents)

  await _collect(fallback, request)

  assert backup.seen_contents == [c.text for c in _request().contents[0].parts]
  assert len(request.contents) == original_turns
  assert request.config.temperature is None


@pytest.mark.asyncio
async def test_the_model_that_succeeds_keeps_its_edits():
  # Only failed attempts are rolled back; what the winner actually sent is
  # what traces should show.
  fallback = FallbackModel(models=[_MutatingLlm(model='primary')])
  request = _request()

  await _collect(fallback, request)

  assert request.contents[-1].parts[0].text == 'edited'
  assert request.config.temperature == 0.99


class _LockHoldingTool(BaseTool):
  """A tool carrying an uncopyable handle, as an MCP tool does."""

  def __init__(self):
    super().__init__(name='locked', description='holds a lock')
    self._lock = threading.Lock()


@pytest.mark.asyncio
async def test_falls_back_with_a_tool_that_cannot_be_copied():
  # tools_dict holds live tool objects, and an MCP tool reaches a
  # threading.Lock that deep copy refuses. Rolling the request back must not
  # try to copy them, or every call would fail once a backup is configured.
  backup = _FakeLlm(model='backup')
  fallback = FallbackModel(
      models=[_failing('primary', _rate_limited()), backup]
  )
  request = _request()
  tool = _LockHoldingTool()
  request.tools_dict['locked'] = tool

  responses = await _collect(fallback, request)

  assert [r.content.parts[0].text for r in responses] == ['backup']
  # The registry is shared, not copied: the delegates see the same instance.
  assert request.tools_dict['locked'] is tool


class _VoiceLlm(_LiveLlm):
  """A live model that configures a voice, as Gemini.connect does."""

  voice: str | None = None

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest):
    if self.voice is not None:
      llm_request.live_connect_config.speech_config = types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(
                  voice_name=self.voice
              )
          )
      )
    async with super().connect(llm_request) as connection:
      yield connection


@pytest.mark.asyncio
async def test_a_failed_connect_does_not_leak_its_edits_to_the_next():
  # Gemini.connect writes speech_config only when the model has one, so
  # without a rollback a backup with no voice would speak in the primary's.
  primary = _VoiceLlm(
      model='primary', voice='Charon', connect_error=_rate_limited()
  )
  backup = _VoiceLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])
  request = _request()

  async with fallback.connect(request) as connection:
    assert connection.model == 'backup'

  assert request.live_connect_config.speech_config is None


@pytest.mark.asyncio
async def test_a_handle_carried_into_a_new_run_pins_to_the_primary():
  # base_llm_flow sets llm_request.model from the agent at the start of every
  # run, so a handle passed in through RunConfig.session_resumption arrives
  # naming the primary whichever model actually opened the session. Pinning
  # can only follow the name, so it lands on the primary.
  primary = _LiveLlm(model='primary')
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])

  # What the flow hands us at the top of a fresh run.
  request = _resuming_request(fallback.model, 'handle-from-a-previous-run')

  async with fallback.connect(request) as connection:
    assert connection.model == 'primary'

  assert not backup.log


@pytest.mark.asyncio
async def test_reconnect_works_for_two_entries_with_the_same_name():
  # One model behind two keys or regions is a reason to reach for this class,
  # and both entries report the same name. Reconnecting has to follow the
  # session, which the name cannot identify.
  region_a = _LiveLlm(model='gemini-3.5-flash', connect_error=_rate_limited())
  region_b = _LiveLlm(model='gemini-3.5-flash')
  fallback = FallbackModel(models=[region_a, region_b])
  request = _request()

  async with fallback.connect(request) as connection:
    assert connection is not None
  # Region A was down, so region B owns the session.
  assert len(region_b.log) == 2  # connected, then closed

  request.live_connect_config = types.LiveConnectConfig(
      session_resumption=types.SessionResumptionConfig(handle='h-1')
  )
  region_a.connect_error = None
  async with fallback.connect(request) as connection:
    assert connection is not None

  # The reconnect went back to B, not to the now-healthy A.
  assert len(region_a.log) == 1  # only the failed first attempt
  assert len(region_b.log) == 4


@pytest.mark.asyncio
async def test_reconnect_follows_the_session_not_the_name():
  # Even with distinct names, the remembered owner is what decides.
  primary = _LiveLlm(model='primary', connect_error=_rate_limited())
  backup = _LiveLlm(model='backup')
  fallback = FallbackModel(models=[primary, backup])
  request = _request()

  async with fallback.connect(request):
    pass

  # Point the name at the primary, as a stale reader might; the session still
  # belongs to the backup.
  request.model = 'primary'
  assert fallback._candidate_indexes(_resuming_request('primary', 'h-1')) == [0]
  request.live_connect_config = types.LiveConnectConfig(
      session_resumption=types.SessionResumptionConfig(handle='h-1')
  )
  assert fallback._candidate_indexes(request) == [1]


def test_a_finished_session_is_forgotten():
  fallback = FallbackModel(models=[_LiveLlm(model='a'), _LiveLlm(model='b')])
  request = _request()
  fallback._remember_live_owner(request, 1)

  assert fallback._recall_live_owner(request) == 1

  del request
  gc.collect()
  fallback._remember_live_owner(_request(), 0)

  # The dead entry is pruned rather than accumulating per session.
  assert len(fallback._live_owner) == 1


def test_two_sessions_with_equal_requests_keep_separate_owners():
  # LlmRequest compares by value, so two concurrent live sessions can hold
  # requests that are equal without being the same session. Matching them by
  # equality would hand one session's server the other's handle.
  fallback = FallbackModel(models=[_LiveLlm(model='a'), _LiveLlm(model='b')])
  first, second = _request(), _request()

  assert first == second and first is not second

  fallback._remember_live_owner(first, 0)
  fallback._remember_live_owner(second, 1)

  assert fallback._recall_live_owner(first) == 0
  assert fallback._recall_live_owner(second) == 1


class _ClosingLlm(_FakeLlm):
  """A model whose stream records that it was closed."""

  closed: list[str] = pydantic.Field(default_factory=list)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    try:
      for index in range(100):
        yield LlmResponse(
            content=types.Content(
                role='model', parts=[types.Part(text=f'chunk{index}')]
            )
        )
    finally:
      self.closed.append(self.model)


@pytest.mark.asyncio
async def test_abandoning_the_stream_closes_the_delegate():
  # The flow consumes a model through Aclosing and can stop early, when a
  # callback raises or the client goes away. Wrapping has to pass that on, or
  # the delegate's stream — and the provider connection under it — is left
  # open until the loop finalises it.
  primary = _ClosingLlm(model='primary')
  fallback = FallbackModel(models=[primary, _FakeLlm(model='backup')])

  async with contextlib.aclosing(
      fallback.generate_content_async(_request(), stream=True)
  ) as agen:
    async for _ in agen:
      break

  assert primary.closed == ['primary']


def test_default_status_codes_membership():
  # 408 is left out on purpose: a timeout does not say whether the request was
  # processed, and litellm reports client-side timeouts as 408. Pinning the
  # set so that reasoning cannot be undone by an unrelated edit.
  assert FallbackModel.DEFAULT_STATUS_CODES == frozenset(
      {429, 500, 502, 503, 504}
  )


@pytest.mark.asyncio
async def test_a_timeout_does_not_fall_back_by_default():
  backup = _FakeLlm(model='backup')
  timed_out = genai_errors.ClientError(408, {})
  fallback = FallbackModel(models=[_failing('primary', timed_out), backup])

  with pytest.raises(genai_errors.ClientError):
    await _collect(fallback, _request())

  assert not backup.seen_request_models


def test_remembering_a_session_twice_keeps_one_entry():
  # A long live session reconnects many times; each one records the owner
  # again. Without the de-duplication the table would grow per reconnect.
  fallback = FallbackModel(models=[_LiveLlm(model='a'), _LiveLlm(model='b')])
  request = _request()

  for _ in range(50):
    fallback._remember_live_owner(request, 1)

  assert len(fallback._live_owner) == 1
  assert fallback._recall_live_owner(request) == 1


def test_remembering_a_session_again_replaces_the_owner():
  fallback = FallbackModel(models=[_LiveLlm(model='a'), _LiveLlm(model='b')])
  request = _request()

  fallback._remember_live_owner(request, 0)
  fallback._remember_live_owner(request, 1)

  assert fallback._recall_live_owner(request) == 1


def test_every_private_attribute_is_accounted_for():
  # Copying whatever LlmRequest happens to hold is what made tools_dict crash
  # the wrapper, so the snapshot names what it restores. An attribute added
  # later has to be sorted into one tuple or the other — restored, or
  # deliberately left alone because it cannot be copied or is never edited.
  snapshotted = set(_SNAPSHOT_PRIVATE)
  skipped = set(_UNSNAPSHOTTED_PRIVATE)

  assert not snapshotted & skipped, (
      'A private attribute is in both _SNAPSHOT_PRIVATE and'
      ' _UNSNAPSHOTTED_PRIVATE; put it in exactly one.'
  )
  assert snapshotted | skipped == set(LlmRequest.__private_attributes__), (
      'LlmRequest has a private attribute not accounted for here. A rollback'
      ' between fallback attempts has to decide what happens to it: add it to'
      ' _SNAPSHOT_PRIVATE to restore it (it must be deep-copyable), or to'
      ' _UNSNAPSHOTTED_PRIVATE to leave it (say why). Both are in'
      ' _fallback_model.py.'
  )


def test_every_public_field_is_accounted_for():
  # A model edits the request in place, so a new public field a model touches
  # would leak across fallback attempts, the trap tools_dict was. This fails
  # until a field added to LlmRequest is sorted: restored by _RequestSnapshot,
  # or deliberately left. The lists live here, not in the module, because the
  # snapshot copies these fields by hand and reads neither at runtime;
  # test_a_failed_model_does_not_leak_its_edits_to_the_next is what proves a
  # restored field is actually restored.
  restored = {'contents', 'config', 'live_connect_config'}
  left = {
      'model',  # Reset to the delegate on every attempt.
      'tools_dict',  # Live tool objects, shared not copied.
      'cache_config',  # Written by a request processor before the call.
      'cache_metadata',  # Written onto the response.
      'cacheable_contents_token_count',  # Response-side too.
      'previous_interaction_id',  # Set by the interactions processor pre-call.
  }

  assert (
      not restored & left
  ), 'A field is in both `restored` and `left`; put it in exactly one.'
  assert restored | left == set(LlmRequest.model_fields), (
      'LlmRequest has a public field not accounted for here. If a model edits'
      ' it mid-call, add it to `restored` AND to _RequestSnapshot (the'
      ' NamedTuple field, of(), and restore()), plus a leak test like'
      ' test_a_failed_model_does_not_leak_its_edits_to_the_next. If a model'
      ' never edits it, add it to `left`.'
  )
