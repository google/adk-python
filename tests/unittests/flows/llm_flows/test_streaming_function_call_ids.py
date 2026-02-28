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

"""Tests that function call IDs stay stable across streaming events."""

from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _finalize_model_response_event
from google.adk.flows.llm_flows.functions import populate_client_function_call_id
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest


def _make_fc_response(name: str, args: dict | None = None, partial: bool = False) -> LlmResponse:
  """Create an LlmResponse containing a single function call."""
  fc = types.FunctionCall(name=name, args=args or {})
  return LlmResponse(
      content=types.Content(role='model', parts=[types.Part(function_call=fc)]),
      partial=partial,
  )


def _make_multi_fc_response(calls: list[tuple[str, dict]], partial: bool = False) -> LlmResponse:
  """Create an LlmResponse containing multiple function calls."""
  parts = [
      types.Part(function_call=types.FunctionCall(name=name, args=args))
      for name, args in calls
  ]
  return LlmResponse(
      content=types.Content(role='model', parts=parts),
      partial=partial,
  )


class TestPopulateClientFunctionCallIdWithCache:
  """Tests for populate_client_function_call_id with ID caching."""

  def test_generates_id_and_stores_in_cache(self):
    event = Event(author='agent')
    event.content = types.Content(
        role='model',
        parts=[types.Part(function_call=types.FunctionCall(name='get_weather', args={}))],
    )
    cache: dict[str, str] = {}
    populate_client_function_call_id(event, cache)
    fc = event.get_function_calls()[0]
    assert fc.id.startswith('adk-')
    assert 'get_weather:0' in cache
    assert cache['get_weather:0'] == fc.id

  def test_reuses_cached_id(self):
    cache: dict[str, str] = {'get_weather:0': 'adk-cached-id-123'}

    event = Event(author='agent')
    event.content = types.Content(
        role='model',
        parts=[types.Part(function_call=types.FunctionCall(name='get_weather', args={}))],
    )
    populate_client_function_call_id(event, cache)
    assert event.get_function_calls()[0].id == 'adk-cached-id-123'

  def test_no_cache_generates_new_id_each_time(self):
    event1 = Event(author='agent')
    event1.content = types.Content(
        role='model',
        parts=[types.Part(function_call=types.FunctionCall(name='get_weather', args={}))],
    )
    event2 = Event(author='agent')
    event2.content = types.Content(
        role='model',
        parts=[types.Part(function_call=types.FunctionCall(name='get_weather', args={}))],
    )
    populate_client_function_call_id(event1)
    populate_client_function_call_id(event2)
    assert event1.get_function_calls()[0].id != event2.get_function_calls()[0].id

  def test_multiple_calls_same_name_get_separate_ids(self):
    event = Event(author='agent')
    event.content = types.Content(
        role='model',
        parts=[
            types.Part(function_call=types.FunctionCall(name='search', args={'q': 'a'})),
            types.Part(function_call=types.FunctionCall(name='search', args={'q': 'b'})),
        ],
    )
    cache: dict[str, str] = {}
    populate_client_function_call_id(event, cache)
    fcs = event.get_function_calls()
    assert fcs[0].id != fcs[1].id
    assert cache['search:0'] == fcs[0].id
    assert cache['search:1'] == fcs[1].id

  def test_skips_function_calls_that_already_have_ids(self):
    event = Event(author='agent')
    event.content = types.Content(
        role='model',
        parts=[types.Part(function_call=types.FunctionCall(
            name='get_weather', args={}, id='server-provided-id'))],
    )
    cache: dict[str, str] = {}
    populate_client_function_call_id(event, cache)
    assert event.get_function_calls()[0].id == 'server-provided-id'
    assert len(cache) == 0


class TestFinalizeModelResponseEventWithCache:
  """Tests that _finalize_model_response_event preserves IDs via cache."""

  def test_partial_and_final_share_same_function_call_id(self):
    model_response_event = Event(
        author='agent',
        invocation_id='inv-1',
    )
    llm_request = LlmRequest(model='mock', contents=[])
    cache: dict[str, str] = {}

    # Partial event
    partial_response = _make_fc_response('get_weather', partial=True)
    partial_event = _finalize_model_response_event(
        llm_request, partial_response, model_response_event, cache,
    )
    partial_id = partial_event.get_function_calls()[0].id
    assert partial_id.startswith('adk-')

    # Final event — same function call must get the same ID
    final_response = _make_fc_response('get_weather', partial=False)
    final_event = _finalize_model_response_event(
        llm_request, final_response, model_response_event, cache,
    )
    final_id = final_event.get_function_calls()[0].id
    assert final_id == partial_id

  def test_without_cache_ids_differ(self):
    model_response_event = Event(
        author='agent',
        invocation_id='inv-1',
    )
    llm_request = LlmRequest(model='mock', contents=[])

    partial_response = _make_fc_response('get_weather', partial=True)
    partial_event = _finalize_model_response_event(
        llm_request, partial_response, model_response_event,
    )
    partial_id = partial_event.get_function_calls()[0].id

    final_response = _make_fc_response('get_weather', partial=False)
    final_event = _finalize_model_response_event(
        llm_request, final_response, model_response_event,
    )
    final_id = final_event.get_function_calls()[0].id

    # Without cache, IDs are different (this is the bug scenario)
    assert final_id != partial_id

  def test_multi_function_call_streaming_preserves_all_ids(self):
    model_response_event = Event(
        author='agent',
        invocation_id='inv-1',
    )
    llm_request = LlmRequest(model='mock', contents=[])
    cache: dict[str, str] = {}

    # Partial with two function calls
    partial_response = _make_multi_fc_response(
        [('search', {'q': 'weather'}), ('lookup', {'id': '42'})],
        partial=True,
    )
    partial_event = _finalize_model_response_event(
        llm_request, partial_response, model_response_event, cache,
    )
    partial_ids = [fc.id for fc in partial_event.get_function_calls()]

    # Final with same two function calls
    final_response = _make_multi_fc_response(
        [('search', {'q': 'weather'}), ('lookup', {'id': '42'})],
        partial=False,
    )
    final_event = _finalize_model_response_event(
        llm_request, final_response, model_response_event, cache,
    )
    final_ids = [fc.id for fc in final_event.get_function_calls()]

    assert partial_ids == final_ids
    assert partial_ids[0] != partial_ids[1]  # different calls have different IDs
