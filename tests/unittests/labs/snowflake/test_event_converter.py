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

"""Tests for the Cortex event converter.

Verifies that Cortex Agents Run API events map onto ADK events as the design
prescribes: deltas only in SSE mode and without duplicates, server-side tool
use paired with its result, persisted payloads bounded, and one final event
rebuilt from the authoritative ``response``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.labs.snowflake._event_converter import CortexEventConverter
from google.adk.labs.snowflake._event_converter import METADATA_KEY
from google.adk.labs.snowflake._event_converter import UnsupportedCortexEventError
from google.adk.labs.snowflake._sse_parser import SseEvent
from google.adk.labs.snowflake._sse_parser import SseParseError
import pytest

_TOOL_USE_ID = 'tool-use-1'


def _sse(name: str, payload: Any) -> SseEvent:
  return SseEvent(event=name, data=json.dumps(payload))


def _make_converter(
    *,
    streaming: bool = True,
    max_tool_result_bytes: int = 32 * 1024,
    include_thinking: bool = False,
    thread_id: str | None = None,
) -> CortexEventConverter:
  ctx = MagicMock()
  ctx.invocation_id = 'inv_1'
  ctx.branch = 'main'
  return CortexEventConverter(
      ctx=ctx,
      author='cortex',
      streaming=streaming,
      max_tool_result_bytes=max_tool_result_bytes,
      include_thinking_in_final_event=include_thinking,
      thread_id=thread_id,
  )


def _tool_use(**overrides: Any) -> SseEvent:
  payload = {
      'client_side_execute': False,
      'content_index': 20,
      'input': {'semantic_model': 'SV', 'sql': 'SELECT 1'},
      'name': 'system_execute_sql',
      'sequence_number': 946,
      'tool_use_id': _TOOL_USE_ID,
      'type': 'system_execute_sql',
  }
  payload.update(overrides)
  return _sse('response.tool_use', payload)


def _tool_result(
    *, status: str = 'success', content: list[Any] | None = None
) -> SseEvent:
  return _sse(
      'response.tool_result',
      {
          'content': (
              [{'json': {'query_id': 'q1', 'result_set': {}}, 'type': 'json'}]
              if content is None
              else content
          ),
          'content_index': 21,
          'name': 'system_execute_sql',
          'sequence_number': 952,
          'status': status,
          'tool_use_id': _TOOL_USE_ID,
          'type': 'system_execute_sql',
      },
  )


def _final_response(**overrides: Any) -> SseEvent:
  payload = {
      'content': [
          {'thinking': {'text': 'reasoning'}, 'type': 'thinking'},
          {'text': 'The answer.', 'type': 'text'},
      ],
      'metadata': {
          'assistant_message_id': 456,
          'run_id': 'run-1',
          'thread_id': 123,
          'usage': {'tokens_consumed': []},
          'user_message_id': 455,
      },
      'role': 'assistant',
      'status': 'completed',
  }
  payload.update(overrides)
  return _sse('response', payload)


def _cortex(event: Event) -> dict[str, Any]:
  assert event.custom_metadata is not None
  return event.custom_metadata[METADATA_KEY]


# --- ids and terminators -----------------------------------------------------


def test_metadata_records_message_ids_as_strings_without_events():
  """Thread message ids are kept per role as decimal strings."""
  converter = _make_converter()

  events = converter.convert(
      _sse('metadata', {'metadata': {'role': 'user', 'message_id': 123}})
  ) + converter.convert(
      _sse('metadata', {'metadata': {'role': 'assistant', 'message_id': 456}})
  )

  assert events == []
  assert converter.user_message_id == '123'
  assert converter.assistant_message_id == '456'


@pytest.mark.parametrize(
    'terminator',
    [SseEvent(event='done', data='[DONE]'), SseEvent(data='[DONE]')],
)
def test_done_marks_the_run_complete_without_an_event(terminator: SseEvent):
  """The transport terminator only flips `is_done`."""
  converter = _make_converter()

  assert converter.convert(terminator) == []
  assert converter.is_done


def test_non_object_json_is_rejected():
  """A payload that is JSON but not an object is a protocol error."""
  converter = _make_converter()

  with pytest.raises(SseParseError, match='not an object'):
    converter.convert(SseEvent(event='response.status', data='[1, 2]'))


def test_invalid_json_is_rejected():
  """Data that is not JSON at all is a protocol error."""
  converter = _make_converter()

  with pytest.raises(SseParseError, match='not JSON'):
    converter.convert(SseEvent(event='response.status', data='nope'))


# --- progress and unknown events ---------------------------------------------


def test_status_streams_as_partial_metadata_in_sse_mode():
  """Progress is forwarded as a partial, non-persisted metadata event."""
  converter = _make_converter(streaming=True)
  payload = {'message': 'Planning', 'sequence_number': 1, 'status': 'planning'}

  (event,) = converter.convert(_sse('response.status', payload))

  assert event.partial is True
  assert event.content is None
  assert _cortex(event) == {'event': 'response.status', 'data': payload}


def test_status_is_silent_without_streaming():
  """A consumer that did not ask for SSE gets no partial events."""
  converter = _make_converter(streaming=False)

  events = converter.convert(
      _sse('response.status', {'status': 'planning', 'sequence_number': 1})
  )

  assert events == []


@pytest.mark.parametrize(
    'name',
    ['response.tool_result.status', 'response.tool_result.analyst.delta'],
)
def test_tool_progress_streams_as_partial_metadata(name: str):
  """Tool execution progress is forwarded like any other progress."""
  converter = _make_converter(streaming=True)

  (event,) = converter.convert(_sse(name, {'tool_use_id': _TOOL_USE_ID}))

  assert event.partial is True
  assert _cortex(event)['event'] == name


def test_unknown_event_is_forwarded_and_does_not_stop_the_run():
  """An event the converter does not know is passed through as `unknown`."""
  converter = _make_converter(streaming=True)

  (event,) = converter.convert(_sse('response.new_thing', {'x': 1}))
  later = converter.convert(
      _sse('response.text.delta', {'content_index': 0, 'text': 'ok'})
  )

  assert _cortex(event) == {
      'unknown': {'event': 'response.new_thing', 'data': {'x': 1}}
  }
  assert len(later) == 1


def test_unknown_event_is_silent_without_streaming():
  """Unknown events are progress-only and never persisted."""
  converter = _make_converter(streaming=False)

  assert converter.convert(_sse('response.new_thing', {'x': 1})) == []


# --- deltas and completed blocks ---------------------------------------------


def test_text_deltas_stream_in_order_as_partial_text_parts():
  """Each text delta becomes one partial model text event."""
  converter = _make_converter(streaming=True)

  events = converter.convert(
      _sse(
          'response.text.delta',
          {'content_index': 1, 'sequence_number': 13, 'text': '17과'},
      )
  ) + converter.convert(
      _sse(
          'response.text.delta',
          {'content_index': 1, 'sequence_number': 14, 'text': ' 23'},
      )
  )

  assert [e.content.parts[0].text for e in events] == ['17과', ' 23']
  assert all(e.partial for e in events)
  assert all(e.content.role == 'model' for e in events)


def test_thinking_deltas_are_thought_parts():
  """Reasoning deltas are marked as thoughts so UIs can fold them."""
  converter = _make_converter(streaming=True)

  (event,) = converter.convert(
      _sse(
          'response.thinking.delta',
          {'content_index': 0, 'sequence_number': 2, 'text': 'hmm'},
      )
  )

  assert event.partial is True
  assert event.content.parts[0].thought is True
  assert event.content.parts[0].text == 'hmm'


def test_duplicate_sequence_number_is_dropped():
  """A resent delta neither streams again nor doubles the buffered text."""
  converter = _make_converter(streaming=True)
  delta = _sse(
      'response.text.delta',
      {'content_index': 1, 'sequence_number': 13, 'text': 'once'},
  )

  events = converter.convert(delta) + converter.convert(delta)
  converter.convert(_final_response(content=[]))

  assert len(events) == 1
  assert converter.final_event().content.parts[0].text == 'once'


def test_deltas_without_streaming_still_feed_the_final_answer():
  """Without SSE nothing streams, but the buffer backs the final event."""
  converter = _make_converter(streaming=False)

  events = converter.convert(
      _sse('response.text.delta', {'content_index': 1, 'text': 'buffered'})
  )
  converter.convert(_final_response(content=[]))

  assert events == []
  assert converter.final_event().content.parts[0].text == 'buffered'


def test_completed_block_replaces_its_deltas():
  """`response.text` is authoritative for its content index."""
  converter = _make_converter(streaming=False)
  converter.convert(
      _sse('response.text.delta', {'content_index': 1, 'text': 'drafty'})
  )

  converter.convert(
      _sse('response.text', {'content_index': 1, 'text': 'final'})
  )
  converter.convert(_final_response(content=[]))

  assert converter.final_event().content.parts[0].text == 'final'


# --- server-side tools --------------------------------------------------------


def test_tool_use_becomes_a_model_function_call():
  """A server-side tool call is recorded as a real `FunctionCall`."""
  converter = _make_converter()

  (event,) = converter.convert(_tool_use())

  call = event.content.parts[0].function_call
  assert event.partial is None
  assert event.author == 'cortex'
  assert event.content.role == 'model'
  assert (call.id, call.name) == (_TOOL_USE_ID, 'system_execute_sql')
  assert call.args == {'semantic_model': 'SV', 'sql': 'SELECT 1'}
  assert not event.is_final_response()


def test_duplicate_tool_use_is_dropped():
  """The same `tool_use_id` is recorded once."""
  converter = _make_converter()

  events = converter.convert(_tool_use()) + converter.convert(_tool_use())

  assert len(events) == 1


def test_non_dict_tool_input_is_wrapped():
  """`FunctionCall.args` must be a dict, so a scalar input is wrapped."""
  converter = _make_converter()

  (event,) = converter.convert(_tool_use(input='raw'))

  assert event.content.parts[0].function_call.args == {'input': 'raw'}


@pytest.mark.parametrize(
    'overrides',
    [{'client_side_execute': True}, {'permission': {'kind': 'ask'}}],
)
def test_client_side_tools_and_permissions_are_unsupported(overrides: dict):
  """Client-side execution and permission prompts end the turn explicitly."""
  converter = _make_converter()

  with pytest.raises(UnsupportedCortexEventError, match='not support'):
    converter.convert(_tool_use(**overrides))


def test_tool_result_becomes_a_function_response_from_the_tool():
  """The result is authored by the tool and paired by `tool_use_id`."""
  converter = _make_converter()
  converter.convert(_tool_use())

  (event,) = converter.convert(_tool_result())

  response = event.content.parts[0].function_response
  assert event.author == 'system_execute_sql'
  assert event.content.role == 'user'
  assert (response.id, response.name) == (_TOOL_USE_ID, 'system_execute_sql')
  assert response.response == {
      'status': 'success',
      'content': [
          {'json': {'query_id': 'q1', 'result_set': {}}, 'type': 'json'}
      ],
  }
  assert not event.is_final_response()


def test_duplicate_tool_result_is_dropped():
  """The same `tool_use_id` is answered once."""
  converter = _make_converter()

  events = converter.convert(_tool_result()) + converter.convert(_tool_result())

  assert len(events) == 1


def test_failed_tool_result_carries_the_error():
  """A tool error surfaces under `error` next to the raw content."""
  converter = _make_converter()
  error = {'error_code': 'SQL_COMPILATION_ERROR', 'message': 'bad sql'}

  (event,) = converter.convert(
      _tool_result(
          status='error',
          content=[{'json': {'error': error, 'sql': 'SELECT'}, 'type': 'json'}],
      )
  )

  response = event.content.parts[0].function_response.response
  assert response['status'] == 'error'
  assert response['error'] == error


def test_oversized_result_rows_are_dropped_but_shape_is_kept():
  """Past the size bound only the query id and column metadata survive."""
  converter = _make_converter(max_tool_result_bytes=400)
  rows = [[str(i), 'Country A', 'Air Purifier'] for i in range(50)]
  result_set = {'data': rows, 'resultSetMetaData': {'numRows': 50}}

  (event,) = converter.convert(
      _tool_result(
          content=[{
              'json': {'query_id': 'q1', 'result_set': result_set},
              'type': 'json',
          }]
      )
  )

  response = event.content.parts[0].function_response.response
  assert response['truncated'] is True
  assert response['original_bytes'] > 400
  assert response['content'][0]['json']['result_set'] == {
      'resultSetMetaData': {'numRows': 50}
  }
  assert response['content'][0]['json']['query_id'] == 'q1'


def test_result_still_too_large_after_dropping_rows_loses_its_content():
  """When even the shape does not fit, the content is emptied, not stored."""
  converter = _make_converter(max_tool_result_bytes=64)

  (event,) = converter.convert(
      _tool_result(content=[{'text': 'x' * 500, 'type': 'text'}])
  )

  response = event.content.parts[0].function_response.response
  assert response['truncated'] is True
  assert response['content'] == []
  assert response['status'] == 'success'


# --- final-event metadata ----------------------------------------------------


def test_warning_streams_and_is_kept_for_the_final_event():
  """Warnings are both forwarded live and recorded on the final event."""
  converter = _make_converter(streaming=True)
  warning = {'message': 'Semantic view is stale'}

  (partial,) = converter.convert(_sse('response.warning', warning))
  converter.convert(_final_response())

  assert partial.partial is True
  assert _cortex(converter.final_event())['warnings'] == [warning]


def test_annotations_and_suggested_queries_land_on_the_final_event():
  """Citations and follow-up questions are final-event metadata only."""
  converter = _make_converter(streaming=True)
  annotation = {'type': 'cortex_search_citation', 'index': 0}

  events = converter.convert(_sse('response.text.annotation', annotation))
  events += converter.convert(
      _sse(
          'response.suggested_queries',
          {'suggested_queries': [{'query': 'And next year?'}]},
      )
  )
  converter.convert(_final_response())

  metadata = _cortex(converter.final_event())
  assert events == []
  assert metadata['annotations'] == [annotation]
  assert metadata['suggested_queries'] == [{'query': 'And next year?'}]


def test_suggested_queries_fall_back_to_the_final_response():
  """Without a dedicated event, the final payload's block is used."""
  converter = _make_converter()
  converter.convert(
      _final_response(
          content=[
              {'text': 'A', 'type': 'text'},
              {
                  'suggested_queries': [{'query': 'B?'}],
                  'type': 'suggested_queries',
              },
          ]
      )
  )

  assert _cortex(converter.final_event())['suggested_queries'] == [
      {'query': 'B?'}
  ]


def test_oversized_table_keeps_metadata_only():
  """A big table is reduced to its query id and column metadata."""
  converter = _make_converter(max_tool_result_bytes=200)
  table = {
      'content_index': 2,
      'query_id': 'q1',
      'result_set': {
          'data': [['x'] * 10] * 30,
          'resultSetMetaData': {'numRows': 30},
      },
      'title': 'By country',
  }

  converter.convert(_sse('response.table', table))
  converter.convert(_final_response())

  (bounded,) = _cortex(converter.final_event())['tables']
  assert bounded['truncated'] is True
  assert bounded['result_set'] == {'resultSetMetaData': {'numRows': 30}}
  assert bounded['title'] == 'By country'


def test_oversized_chart_drops_its_spec():
  """A big chart keeps everything but the serialized spec."""
  converter = _make_converter(max_tool_result_bytes=100)

  converter.convert(
      _sse('response.chart', {'content_index': 3, 'chart_spec': 'x' * 500})
  )
  converter.convert(_final_response())

  (bounded,) = _cortex(converter.final_event())['charts']
  assert bounded == {'content_index': 3, 'truncated': True}


def test_small_table_and_chart_are_kept_whole():
  """Under the bound, presentations are recorded as sent."""
  converter = _make_converter()
  table = {'content_index': 2, 'result_set': {'data': [['1']]}}
  chart = {'content_index': 3, 'chart_spec': '{}'}

  converter.convert(_sse('response.table', table))
  converter.convert(_sse('response.chart', chart))
  converter.convert(_final_response())

  metadata = _cortex(converter.final_event())
  assert metadata['tables'] == [table]
  assert metadata['charts'] == [chart]


# --- final event --------------------------------------------------------------


def test_final_event_is_rebuilt_from_the_final_response():
  """The answer comes from `response.content`, not from the deltas."""
  converter = _make_converter(streaming=True)
  converter.convert(
      _sse('response.text.delta', {'content_index': 1, 'text': 'drafty'})
  )
  converter.convert(_final_response())

  event = converter.final_event()

  assert event.partial is None
  assert event.author == 'cortex'
  assert event.content.role == 'model'
  assert [p.text for p in event.content.parts] == ['The answer.']
  assert event.is_final_response()
  assert _cortex(event)['run_id'] == 'run-1'
  assert _cortex(event)['status'] == 'completed'
  assert _cortex(event)['usage'] == {'tokens_consumed': []}


def test_final_event_omits_thinking_by_default():
  """Reasoning is not persisted with the final event unless asked for."""
  converter = _make_converter(include_thinking=False)
  converter.convert(_final_response())

  parts = converter.final_event().content.parts

  assert [p.thought for p in parts] == [None]


def test_final_event_includes_thinking_when_enabled():
  """With the option on, completed reasoning precedes the answer."""
  converter = _make_converter(include_thinking=True)
  converter.convert(_final_response())

  parts = converter.final_event().content.parts

  assert [(p.thought, p.text) for p in parts] == [
      (True, 'reasoning'),
      (None, 'The answer.'),
  ]


def test_final_event_carries_the_state_delta():
  """The cursor handed in is committed through `EventActions`."""
  converter = _make_converter()
  converter.convert(_final_response())

  event = converter.final_event(state_delta={'_snowflake_cortex_x': {'a': 1}})

  assert event.actions.state_delta == {'_snowflake_cortex_x': {'a': 1}}


def test_run_id_is_derived_when_the_payload_lacks_one():
  """`{thread_id}-{user_message_id}` is the documented run id shape."""
  converter = _make_converter(thread_id='123')
  converter.convert(
      _sse('metadata', {'metadata': {'role': 'user', 'message_id': 455}})
  )
  converter.convert(_final_response(metadata={}))

  assert _cortex(converter.final_event())['run_id'] == '123-455'


def test_final_response_backfills_ids():
  """Ids missing from `metadata` events are taken from the final payload."""
  converter = _make_converter()

  converter.convert(_final_response())

  assert converter.thread_id == '123'
  assert converter.assistant_message_id == '456'


def test_final_event_requires_a_final_response():
  """Without the authoritative payload there is nothing to record."""
  converter = _make_converter()

  assert not converter.has_final_response
  with pytest.raises(ValueError, match='without a final response'):
    converter.final_event()


# --- terminal error -----------------------------------------------------------


def test_error_becomes_a_terminal_error_event():
  """A stream-level `error` ends the run with code and message."""
  converter = _make_converter()
  payload = {'code': 'STREAM_TIMEOUT', 'message': 'took too long'}

  (event,) = converter.convert(_sse('error', payload))

  assert event.partial is None
  assert (event.error_code, event.error_message) == (
      'STREAM_TIMEOUT',
      'took too long',
  )
  assert _cortex(event) == {'event': 'error', 'data': payload}
  assert event.is_final_response()
  assert converter.failed


def test_error_without_fields_uses_defaults():
  """An empty error payload still produces a usable error event."""
  converter = _make_converter()

  (event,) = converter.convert(_sse('error', {}))

  assert event.error_code == 'SNOWFLAKE_CORTEX_ERROR'
  assert event.error_message
