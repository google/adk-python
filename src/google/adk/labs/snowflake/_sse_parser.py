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

"""Incremental parser for the Server-Sent Events wire format.

The Cortex Agents Run API streams ``text/event-stream``. Network chunks line up
neither with event boundaries nor with UTF-8 code points, so the parser is fed
raw bytes and emits each event once its terminating blank line has arrived.
Framing follows the SSE specification: ``data:`` lines join with LF, a line
ends at LF, CR or CRLF, a leading byte-order mark is dropped, comment lines are
skipped, and an event cut off by the end of the stream is discarded. The parser
knows nothing about the JSON inside ``data``.
"""

from __future__ import annotations

import codecs
import dataclasses
import json
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterable

_DEFAULT_MAX_EVENT_BYTES = 16 * 1024 * 1024
_DEFAULT_EVENT_NAME = 'message'
_DONE_EVENT_NAME = 'done'
_DONE_DATA = '[DONE]'
_BOM = '\ufeff'


class SseParseError(ValueError):
  """The byte stream is not a usable event stream.

  Raised for an event larger than the parser's limit, and for ``data`` that is
  not JSON when JSON was asked for. The message never quotes the payload, which
  can hold a user's query or generated SQL.
  """


@dataclasses.dataclass(frozen=True)
class SseEvent:
  """One event from a ``text/event-stream`` response."""

  data: str
  """The ``data`` lines of the event joined with LF."""

  event: str = _DEFAULT_EVENT_NAME
  """The ``event`` field, or ``'message'`` when the server sent none."""

  id: str | None = None
  """The most recent ``id`` field seen on the stream, if any."""

  @property
  def is_done(self) -> bool:
    """Whether this event ends the stream normally.

    Cortex sends ``event: done`` together with ``data: [DONE]``. Either half on
    its own counts, so a server that drops one of them still terminates
    cleanly.
    """
    return self.event == _DONE_EVENT_NAME or self.data.strip() == _DONE_DATA

  def json_data(self) -> Any:
    """Parses ``data`` as JSON.

    Returns:
      The decoded JSON value.

    Raises:
      SseParseError: ``data`` is not valid JSON.
    """
    try:
      return json.loads(self.data)
    except json.JSONDecodeError as e:
      raise SseParseError(
          f'SSE event {self.event!r} carries data that is not JSON (error at'
          f' position {e.pos}).'
      ) from e


class SseParser:
  """Turns byte chunks into ``SseEvent``s, holding partial input in between.

  Feed every chunk to ``feed`` and the events it completes come back in order;
  call ``close`` at the end of the stream to flush the decoder and drop any
  unterminated event.
  """

  def __init__(self, *, max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES):
    """Initializes the parser.

    Args:
      max_event_bytes: Upper bound on the bytes buffered for one event. A
        stream that exceeds it raises ``SseParseError`` from ``feed``, so a
        single oversized event cannot grow memory without limit.
    """
    self._max_event_bytes = max_event_bytes
    # 'replace' rather than 'strict': one bad byte inside a multi-megabyte
    # result set should not fail the whole turn, and the payload is JSON text
    # whose structure survives a replacement character.
    self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    self._pending = ''
    self._at_start = True
    self._buffered_bytes = 0
    self._event_name: str | None = None
    self._data_lines: list[str] = []
    self._last_id: str | None = None

  def feed(self, chunk: bytes) -> list[SseEvent]:
    """Consumes one chunk and returns the events it completed, in order.

    Args:
      chunk: The next bytes of the response body.

    Returns:
      The events whose terminating blank line arrived in this chunk.

    Raises:
      SseParseError: The event being buffered exceeds ``max_event_bytes``.
    """
    self._buffered_bytes += len(chunk)
    events = self._consume(self._decoder.decode(chunk), final=False)
    if self._buffered_bytes > self._max_event_bytes:
      raise SseParseError(
          'SSE event exceeds the maximum size of'
          f' {self._max_event_bytes} bytes.'
      )
    return events

  def close(self) -> list[SseEvent]:
    """Ends the stream and returns any events completed by the final bytes.

    An event the stream ended in the middle of, before its blank line, is
    discarded as the SSE specification requires: a truncated final event is
    not evidence that the run finished.
    """
    events = self._consume(self._decoder.decode(b'', final=True), final=True)
    self._pending = ''
    self._reset_event()
    return events

  def _consume(self, text: str, *, final: bool) -> list[SseEvent]:
    if self._at_start and text:
      text = text.removeprefix(_BOM)
      self._at_start = False
    self._pending += text
    events: list[SseEvent] = []
    while (line := self._pop_line(final=final)) is not None:
      event = self._process_line(line)
      if event is not None:
        events.append(event)
    return events

  def _pop_line(self, *, final: bool) -> str | None:
    pending = self._pending
    cr = pending.find('\r')
    lf = pending.find('\n')
    if cr == -1 and lf == -1:
      return None
    if cr != -1 and (lf == -1 or cr < lf):
      end = cr
      if end + 1 == len(pending) and not final:
        # A CR at the very end may be the first half of a CRLF split across
        # chunks; wait for the next chunk rather than dispatch on it now and
        # read the LF as an extra blank line later.
        return None
      skip = 2 if pending[end + 1 : end + 2] == '\n' else 1
    else:
      end = lf
      skip = 1
    self._pending = pending[end + skip :]
    return pending[:end]

  def _process_line(self, line: str) -> SseEvent | None:
    if not line:
      return self._dispatch()
    if line.startswith(':'):
      return None
    field, sep, value = line.partition(':')
    if sep and value.startswith(' '):
      value = value[1:]
    if field == 'event':
      self._event_name = value
    elif field == 'data':
      self._data_lines.append(value)
    elif field == 'id' and '\x00' not in value:
      self._last_id = value
    # `retry` and unknown fields are ignored, as the specification requires.
    return None

  def _dispatch(self) -> SseEvent | None:
    if not self._data_lines:
      # A block without data is not an event; its event name is dropped too.
      self._reset_event()
      return None
    event = SseEvent(
        data='\n'.join(self._data_lines),
        event=self._event_name or _DEFAULT_EVENT_NAME,
        id=self._last_id,
    )
    self._reset_event()
    return event

  def _reset_event(self) -> None:
    self._event_name = None
    self._data_lines = []
    # Whatever follows the blank line already belongs to the next event.
    self._buffered_bytes = len(self._pending.encode('utf-8'))


async def iter_sse_events(
    chunks: AsyncIterable[bytes],
    *,
    max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES,
) -> AsyncGenerator[SseEvent, None]:
  """Yields the events of a byte stream as they complete.

  Args:
    chunks: The response body, such as ``httpx.Response.aiter_bytes()``.
    max_event_bytes: See ``SseParser``.

  Yields:
    Each complete event, in stream order. Reading stops when ``chunks`` does;
    deciding that a ``done`` event ends the stream is the caller's job.

  Raises:
    SseParseError: An event exceeds ``max_event_bytes``.
  """
  parser = SseParser(max_event_bytes=max_event_bytes)
  async for chunk in chunks:
    for event in parser.feed(chunk):
      yield event
  for event in parser.close():
    yield event
