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

"""Tests for the Snowflake SSE parser.

Verifies that ``text/event-stream`` bytes become events regardless of how the
network splits them, and that the Cortex terminal markers are recognized.
"""

from __future__ import annotations

from typing import AsyncIterator
from typing import Sequence

from google.adk.labs.snowflake._sse_parser import iter_sse_events
from google.adk.labs.snowflake._sse_parser import SseEvent
from google.adk.labs.snowflake._sse_parser import SseParseError
from google.adk.labs.snowflake._sse_parser import SseParser
import pytest

_TWO_EVENTS = (
    b'event: metadata\n'
    b'data: {"metadata":{"role":"user","message_id":123}}\n'
    b'\n'
    b'event: response.text.delta\n'
    b'data: {"content_index":1,"sequence_number":13,"text":"17\xea\xb3\xbc"}\n'
    b'\n'
)


def _parse(stream: bytes, *, chunk_size: int | None = None) -> list[SseEvent]:
  """Feeds `stream` in `chunk_size` pieces and closes the parser."""
  parser = SseParser()
  events: list[SseEvent] = []
  size = chunk_size or len(stream)
  for start in range(0, len(stream), size):
    events.extend(parser.feed(stream[start : start + size]))
  events.extend(parser.close())
  return events


async def _chunks(pieces: Sequence[bytes]) -> AsyncIterator[bytes]:
  for piece in pieces:
    yield piece


def test_single_event_carries_its_name_and_data():
  """An `event:` and `data:` pair becomes one event."""
  events = _parse(b'event: response.status\ndata: {"status":"planning"}\n\n')

  assert events == [
      SseEvent(event='response.status', data='{"status":"planning"}')
  ]


def test_event_name_defaults_to_message():
  """A block with only `data:` uses the specification's default name."""
  events = _parse(b'data: hello\n\n')

  assert events == [SseEvent(event='message', data='hello')]


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 7, 16, None])
def test_chunk_boundaries_do_not_change_the_events(chunk_size: int | None):
  """Any split of the byte stream yields the same events in the same order."""
  events = _parse(_TWO_EVENTS, chunk_size=chunk_size)

  assert [e.event for e in events] == ['metadata', 'response.text.delta']
  assert events[1].json_data()['text'] == '17과'


def test_multibyte_character_split_across_chunks_is_decoded():
  """A UTF-8 sequence cut between chunks is reassembled, not replaced."""
  parser = SseParser()
  head, tail = b'data: 17\xea', b'\xb3\xbc\n\n'

  events = parser.feed(head) + parser.feed(tail) + parser.close()

  assert events == [SseEvent(data='17과')]


def test_invalid_utf8_is_replaced_rather_than_fatal():
  """A bad byte becomes U+FFFD so the rest of the stream still parses."""
  events = _parse(b'data: a\xffb\n\n')

  assert events == [SseEvent(data='a\ufffdb')]


@pytest.mark.parametrize('newline', [b'\n', b'\r\n', b'\r'])
def test_every_line_ending_frames_events(newline: bytes):
  """LF, CRLF and bare CR all end lines and blank lines."""
  stream = newline.join([b'event: a', b'data: 1', b'', b'data: 2', b'', b''])

  events = _parse(stream)

  assert events == [SseEvent(event='a', data='1'), SseEvent(data='2')]


def test_crlf_split_across_chunks_is_one_line_ending():
  """A CR ending one chunk and an LF starting the next do not add a line."""
  parser = SseParser()

  events = (
      parser.feed(b'event: a\r')
      + parser.feed(b'\ndata: 1\r')
      + parser.feed(b'\n\r')
      + parser.feed(b'\n')
      + parser.close()
  )

  assert events == [SseEvent(event='a', data='1')]


def test_multi_line_data_is_joined_with_lf():
  """Several `data:` lines form one payload separated by LF."""
  events = _parse(b'data: {\ndata:   "a": 1\ndata: }\n\n')

  assert events == [SseEvent(data='{\n  "a": 1\n}')]
  assert events[0].json_data() == {'a': 1}


def test_one_leading_space_after_the_colon_is_dropped():
  """`data:x`, `data: x` and `data:  x` differ only by the extra spaces."""
  events = _parse(b'data:x\n\ndata: x\n\ndata:  x\n\n')

  assert [e.data for e in events] == ['x', 'x', ' x']


def test_comments_retry_and_unknown_fields_are_ignored():
  """Only `event`, `data` and `id` shape the event."""
  events = _parse(
      b': keep-alive\nretry: 3000\nfoo: bar\nid: 7\nevent: a\ndata: 1\n\n'
  )

  assert events == [SseEvent(event='a', data='1', id='7')]


def test_block_without_data_is_dropped_with_its_name():
  """An `event:` line followed by a blank line is not an event."""
  events = _parse(b'event: orphan\n\ndata: 1\n\n')

  assert events == [SseEvent(event='message', data='1')]


def test_leading_byte_order_mark_is_ignored():
  """A BOM at the start of the stream does not become part of a field name."""
  events = _parse(b'\xef\xbb\xbfevent: a\ndata: 1\n\n')

  assert events == [SseEvent(event='a', data='1')]


def test_unterminated_final_event_is_discarded():
  """An event cut off before its blank line is not dispatched at close."""
  events = _parse(b'event: a\ndata: 1\n\nevent: response\ndata: {"x":1}')

  assert events == [SseEvent(event='a', data='1')]


def test_unknown_event_names_pass_through_in_order():
  """The parser preserves names it does not know about."""
  events = _parse(
      b'event: response.new_thing\ndata: 1\n\nevent: z\ndata: 2\n\n'
  )

  assert [e.event for e in events] == ['response.new_thing', 'z']


def test_done_event_with_done_data_is_terminal():
  """`event: done` plus `data: [DONE]` is how Cortex ends a run."""
  (event,) = _parse(b'event: done\ndata: [DONE]\n\n')

  assert event.is_done


@pytest.mark.parametrize(
    'stream',
    [b'event: done\ndata: {}\n\n', b'data: [DONE]\n\n', b'data:  [DONE] \n\n'],
)
def test_either_done_marker_alone_is_terminal(stream: bytes):
  """The name or the data alone marks the end, for servers that send one."""
  (event,) = _parse(stream)

  assert event.is_done


def test_ordinary_events_are_not_terminal():
  """A `response` event with a completed status still is not the end."""
  (event,) = _parse(b'event: response\ndata: {"status":"completed"}\n\n')

  assert not event.is_done


def test_error_event_is_parsed_like_any_other():
  """A terminal `error` event keeps its name and JSON body."""
  (event,) = _parse(
      b'event: error\ndata: {"code":"STREAM_TIMEOUT","message":"m"}\n\n'
  )

  assert event.event == 'error'
  assert event.json_data() == {'code': 'STREAM_TIMEOUT', 'message': 'm'}


def test_non_json_data_raises_without_quoting_the_payload():
  """`json_data` fails clearly and keeps the payload out of the message."""
  (event,) = _parse(b'event: response.text\ndata: SELECT secret FROM t\n\n')

  with pytest.raises(SseParseError, match='not JSON') as info:
    event.json_data()

  assert 'secret' not in str(info.value)


def test_oversized_event_raises():
  """An event past `max_event_bytes` fails instead of growing memory."""
  parser = SseParser(max_event_bytes=32)

  with pytest.raises(SseParseError, match='maximum size'):
    parser.feed(b'data: ' + b'x' * 64)


def test_size_limit_applies_per_event_not_per_stream():
  """Many small events do not add up against the limit."""
  parser = SseParser(max_event_bytes=32)

  events = parser.feed(b'data: 1\n\n' * 20)

  assert len(events) == 20


async def test_async_iteration_yields_events_across_chunks():
  """`iter_sse_events` drives the parser over an async byte source."""
  pieces = [
      _TWO_EVENTS[:20],
      _TWO_EVENTS[20:],
      b'event: done\ndata: [DONE]\n\n',
  ]

  events = [e async for e in iter_sse_events(_chunks(pieces))]

  assert [e.event for e in events] == [
      'metadata',
      'response.text.delta',
      'done',
  ]
  assert events[-1].is_done
