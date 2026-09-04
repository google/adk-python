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

"""Translates Cortex Agents Run API events into ADK events.

Kept separate from the agent so the mapping rules stay readable and testable
without a network or a session. One ``CortexEventConverter`` accumulates a
single run: it deduplicates repeated deltas, pairs server-side tool use with
tool results, bounds what gets persisted, and builds the one final event from
the run's authoritative ``response`` payload.

Scope: progress and unknown events as partial metadata events, thinking and
text deltas as partial parts (both only in SSE streaming mode), server-side
tool use and results as ``FunctionCall`` / ``FunctionResponse`` events,
annotations, warnings, tables, charts and suggested queries as final-event
metadata, ``error`` as a terminal error event, and ``done`` as the transport
terminator.
"""

from __future__ import annotations

import json
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from google.genai import types as genai_types
from pydantic import JsonValue

from ...events.event import Event
from ...events.event_actions import EventActions
from ._sse_parser import SseEvent
from ._sse_parser import SseParseError

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext

METADATA_KEY = 'snowflake_cortex'
"""The ``custom_metadata`` key under which every Cortex detail is namespaced."""

_DEFAULT_ERROR_CODE = 'SNOWFLAKE_CORTEX_ERROR'
_DEFAULT_ERROR_MESSAGE = 'The Snowflake Cortex Agent run failed.'
_DEFAULT_TOOL_ERROR = {'message': 'The Snowflake tool call failed.'}

_Handler = Callable[[str, dict[str, Any]], list[Event]]


class UnsupportedCortexEventError(RuntimeError):
  """The run asked for a capability this integration does not support yet.

  Raised when Cortex requests client-side tool execution or a permission
  decision, neither of which this agent can answer. The run is left to time
  out on the Snowflake side; the caller sees the turn fail.
  """


def _json_size(value: Any) -> int:
  return len(
      json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode(
          'utf-8'
      )
  )


def _as_dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
  if value is None:
    return []
  return value if isinstance(value, list) else [value]


def _content_index(payload: dict[str, Any]) -> int:
  value = payload.get('content_index')
  return value if isinstance(value, int) else 0


def _without_result_rows(item: Any) -> Any:
  """Drops ``result_set.data`` from one tool-result content block."""
  if not isinstance(item, dict):
    return item
  body = item.get('json')
  if not isinstance(body, dict) or not isinstance(body.get('result_set'), dict):
    return item
  result_set = {k: v for k, v in body['result_set'].items() if k != 'data'}
  return {**item, 'json': {**body, 'result_set': result_set}}


def _tool_error(content: list[Any]) -> JsonValue:
  for item in content:
    body = _as_dict(item).get('json')
    if isinstance(body, dict) and 'error' in body:
      error: JsonValue = body['error']
      return error
  return dict(_DEFAULT_TOOL_ERROR)


class CortexEventConverter:
  """Accumulates one Cortex run and maps its events onto ADK events.

  Create one per run, pass every parsed ``SseEvent`` to ``convert`` in stream
  order, and once ``is_done`` and ``has_final_response`` hold, ask
  ``final_event`` for the single non-partial event that carries the answer.
  """

  def __init__(
      self,
      *,
      ctx: InvocationContext,
      author: str,
      streaming: bool,
      max_tool_result_bytes: int,
      include_thinking_in_final_event: bool,
      thread_id: str | None = None,
  ):
    """Initializes the converter for one run.

    Args:
      ctx: The invocation context whose id and branch every event carries.
      author: The ADK agent name to author model events with.
      streaming: Whether the consumer asked for SSE streaming. Partial events
        are produced only when this is true.
      max_tool_result_bytes: Size bound for one recorded tool result, table or
        chart; larger payloads are reduced to their metadata.
      include_thinking_in_final_event: Whether the final event also carries the
        completed reasoning as a ``thought`` part.
      thread_id: The Snowflake thread the run belongs to, when known, so a
        ``run_id`` can be derived if the final payload does not name one.
    """
    self._ctx = ctx
    self._author = author
    self._streaming = streaming
    self._max_bytes = max_tool_result_bytes
    self._include_thinking = include_thinking_in_final_event
    self._thread_id = thread_id
    self._user_message_id: str | None = None
    self._assistant_message_id: str | None = None
    self._seen_deltas: set[tuple[str, int, int]] = set()
    self._seen_tool_calls: set[str] = set()
    self._seen_tool_results: set[str] = set()
    self._text_blocks: dict[int, str] = {}
    self._thinking_blocks: dict[int, str] = {}
    self._annotations: list[Any] = []
    self._warnings: list[Any] = []
    self._tables: list[Any] = []
    self._charts: list[Any] = []
    self._suggested_queries: list[Any] = []
    self._final_response: dict[str, Any] | None = None
    self._error: dict[str, Any] | None = None
    self._done = False
    self._handlers: dict[str, _Handler] = {
        'metadata': self._on_metadata,
        'response.status': self._on_progress,
        'response.thinking.delta': self._on_thinking_delta,
        'response.thinking': self._on_thinking,
        'response.text.delta': self._on_text_delta,
        'response.text': self._on_text,
        'response.tool_use': self._on_tool_use,
        'response.tool_result.status': self._on_progress,
        'response.tool_result.analyst.delta': self._on_progress,
        'response.tool_result': self._on_tool_result,
        'response.text.annotation': self._on_annotation,
        'response.warning': self._on_warning,
        'response.table': self._on_table,
        'response.chart': self._on_chart,
        'response.suggested_queries': self._on_suggested_queries,
        'response': self._on_response,
        'error': self._on_error,
    }

  @property
  def thread_id(self) -> str | None:
    """The Snowflake thread id, from the caller or the final payload."""
    return self._thread_id

  @property
  def user_message_id(self) -> str | None:
    """The id Snowflake gave the user message of this run, if seen."""
    return self._user_message_id

  @property
  def assistant_message_id(self) -> str | None:
    """The id Snowflake gave the assistant message of this run, if seen."""
    return self._assistant_message_id

  @property
  def is_done(self) -> bool:
    """Whether the transport terminator has been seen."""
    return self._done

  @property
  def has_final_response(self) -> bool:
    """Whether the authoritative final ``response`` has been seen."""
    return self._final_response is not None

  @property
  def failed(self) -> bool:
    """Whether a terminal ``error`` event has been seen."""
    return self._error is not None

  def convert(self, sse_event: SseEvent) -> list[Event]:
    """Maps one Cortex event onto zero or more ADK events.

    Args:
      sse_event: The next event of the run, in stream order.

    Returns:
      The ADK events to yield for it, possibly none.

    Raises:
      SseParseError: The event's data is not a JSON object.
      UnsupportedCortexEventError: The run needs client-side tool execution or
        a permission decision.
    """
    if sse_event.is_done:
      self._done = True
      return []
    payload = sse_event.json_data()
    if not isinstance(payload, dict):
      raise SseParseError(
          f'SSE event {sse_event.event!r} carries JSON that is not an object.'
      )
    handler = self._handlers.get(sse_event.event)
    if handler is None:
      return self._partial_metadata(
          {'unknown': {'event': sse_event.event, 'data': payload}}
      )
    return handler(sse_event.event, payload)

  def final_event(self, *, state_delta: dict[str, Any] | None = None) -> Event:
    """Builds the single non-partial event that closes a successful run.

    Args:
      state_delta: Session state to commit with the event, typically the
        thread cursor. Applied by the ADK ``Runner``.

    Returns:
      A model event carrying the answer text, the run's metadata under
      ``custom_metadata['snowflake_cortex']``, and ``state_delta``.

    Raises:
      ValueError: No final ``response`` was received.
    """
    if self._final_response is None:
      raise ValueError(
          'The Cortex run ended without a final response event, so there is'
          ' no answer to record.'
      )
    content = _as_list(self._final_response.get('content'))
    metadata = _as_dict(self._final_response.get('metadata'))

    parts: list[genai_types.Part] = []
    if self._include_thinking:
      thinking = self._final_text(content, 'thinking', self._thinking_blocks)
      if thinking:
        parts.append(genai_types.Part(text=thinking, thought=True))
    text = self._final_text(content, 'text', self._text_blocks)
    if text:
      parts.append(genai_types.Part.from_text(text=text))

    suggested = self._suggested_queries or [
        query
        for block in content
        if _as_dict(block).get('type') == 'suggested_queries'
        for query in _as_list(block.get('suggested_queries'))
    ]
    run_id = metadata.get('run_id')
    if run_id is None and self._thread_id and self._user_message_id:
      run_id = f'{self._thread_id}-{self._user_message_id}'

    return Event(
        invocation_id=self._ctx.invocation_id,
        author=self._author,
        branch=self._ctx.branch,
        content=genai_types.Content(role='model', parts=parts),
        custom_metadata={
            METADATA_KEY: {
                'run_id': run_id,
                'status': self._final_response.get('status'),
                'annotations': list(self._annotations),
                'warnings': list(self._warnings),
                'suggested_queries': suggested,
                'usage': metadata.get('usage'),
                'tables': list(self._tables),
                'charts': list(self._charts),
            }
        },
        actions=EventActions(state_delta=state_delta or {}),
    )

  def _final_text(
      self, content: list[Any], block_type: str, buffered: dict[int, str]
  ) -> str:
    # The final `response` aggregates the run, so its blocks win over the
    # deltas; the buffer only covers a payload that omits them.
    texts = []
    for block in content:
      block = _as_dict(block)
      if block.get('type') != block_type:
        continue
      value = block.get(block_type)
      if isinstance(value, dict):
        value = value.get('text')
      if isinstance(value, str):
        texts.append(value)
    if texts:
      return ''.join(texts)
    return ''.join(buffered[index] for index in sorted(buffered))

  def _event(self, **kwargs: Any) -> Event:
    return Event(
        invocation_id=self._ctx.invocation_id,
        author=self._author,
        branch=self._ctx.branch,
        **kwargs,
    )

  def _partial_metadata(self, body: dict[str, Any]) -> list[Event]:
    if not self._streaming:
      return []
    return [self._event(partial=True, custom_metadata={METADATA_KEY: body})]

  def _on_metadata(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    metadata = _as_dict(payload.get('metadata')) or payload
    message_id = metadata.get('message_id')
    if message_id is not None:
      # Decimal strings: Snowflake ids are NUMBER(38,0) and would lose
      # precision in JSON consumers that read them as doubles.
      if metadata.get('role') == 'user':
        self._user_message_id = str(message_id)
      elif metadata.get('role') == 'assistant':
        self._assistant_message_id = str(message_id)
    if metadata.get('thread_id') is not None:
      self._thread_id = str(metadata['thread_id'])
    return []

  def _on_progress(self, name: str, payload: dict[str, Any]) -> list[Event]:
    return self._partial_metadata({'event': name, 'data': payload})

  def _on_thinking_delta(
      self, name: str, payload: dict[str, Any]
  ) -> list[Event]:
    return self._on_delta(name, payload, self._thinking_blocks, thought=True)

  def _on_text_delta(self, name: str, payload: dict[str, Any]) -> list[Event]:
    return self._on_delta(name, payload, self._text_blocks, thought=False)

  def _on_delta(
      self,
      name: str,
      payload: dict[str, Any],
      blocks: dict[int, str],
      *,
      thought: bool,
  ) -> list[Event]:
    index = _content_index(payload)
    sequence = payload.get('sequence_number')
    if isinstance(sequence, int):
      # Snowflake may resend a delta with the same sequence number; appending
      # it twice would duplicate text in the buffer and on screen.
      key = (name, index, sequence)
      if key in self._seen_deltas:
        return []
      self._seen_deltas.add(key)
    text = payload.get('text')
    if not isinstance(text, str) or not text:
      return []
    blocks[index] = blocks.get(index, '') + text
    if not self._streaming:
      return []
    part = (
        genai_types.Part(text=text, thought=True)
        if thought
        else genai_types.Part.from_text(text=text)
    )
    return [
        self._event(
            partial=True,
            content=genai_types.Content(role='model', parts=[part]),
        )
    ]

  def _on_thinking(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    return self._on_block(payload, self._thinking_blocks)

  def _on_text(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    return self._on_block(payload, self._text_blocks)

  def _on_block(
      self, payload: dict[str, Any], blocks: dict[int, str]
  ) -> list[Event]:
    # The completed block is authoritative for its index; the deltas that
    # built it up may have been reordered or duplicated on the wire.
    text = payload.get('text')
    if isinstance(text, str):
      blocks[_content_index(payload)] = text
    return []

  def _tool_identity(self, payload: dict[str, Any]) -> tuple[str, str]:
    tool_use_id = payload.get('tool_use_id')
    if tool_use_id is None:
      tool_use_id = f"{payload.get('type')}-{payload.get('sequence_number')}"
    name = payload.get('name') or payload.get('type') or 'unknown_tool'
    return str(tool_use_id), str(name)

  def _on_tool_use(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    if payload.get('client_side_execute') or 'permission' in payload:
      raise UnsupportedCortexEventError(
          'The Cortex Agent asked for a client-side tool execution or a'
          ' permission decision, which SnowflakeCortexAgent does not support'
          ' yet. Configure the Cortex Agent with server-side tools only.'
      )
    tool_use_id, tool_name = self._tool_identity(payload)
    if tool_use_id in self._seen_tool_calls:
      return []
    self._seen_tool_calls.add(tool_use_id)
    args = payload.get('input')
    if not isinstance(args, dict):
      args = {} if args is None else {'input': args}
    return [
        self._event(
            content=genai_types.Content(
                role='model',
                parts=[
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=tool_name, args=args, id=tool_use_id
                        )
                    )
                ],
            )
        )
    ]

  def _on_tool_result(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    tool_use_id, tool_name = self._tool_identity(payload)
    if tool_use_id in self._seen_tool_results:
      return []
    self._seen_tool_results.add(tool_use_id)
    status = str(payload.get('status') or 'unknown')
    content = _as_list(payload.get('content'))
    response: dict[str, JsonValue] = {'status': status, 'content': content}
    if status == 'error':
      response['error'] = _tool_error(content)
    return [
        Event(
            invocation_id=self._ctx.invocation_id,
            # Authored by the tool so session history attributes the response
            # to it, mirroring ADK's own function-response events.
            author=tool_name,
            branch=self._ctx.branch,
            content=genai_types.Content(
                role='user',
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            id=tool_use_id,
                            response=self._bound_tool_result(response),
                        )
                    )
                ],
            ),
        )
    ]

  def _bound_tool_result(
      self, response: dict[str, JsonValue]
  ) -> dict[str, JsonValue]:
    # The response is persisted with the session, so a SQL result set is cut
    # down to its shape (query id, column metadata) rather than stored whole.
    size = _json_size(response)
    if size <= self._max_bytes:
      return response
    content = [
        _without_result_rows(item) for item in _as_list(response['content'])
    ]
    bounded: dict[str, JsonValue] = {
        **response,
        'content': content,
        'truncated': True,
        'original_bytes': size,
    }
    if _json_size(bounded) <= self._max_bytes:
      return bounded
    return {**bounded, 'content': []}

  def _on_annotation(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    self._annotations.append(payload)
    return []

  def _on_warning(self, name: str, payload: dict[str, Any]) -> list[Event]:
    self._warnings.append(payload)
    return self._partial_metadata({'event': name, 'data': payload})

  def _on_table(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    table = payload
    if _json_size(table) > self._max_bytes:
      result_set = {
          k: v
          for k, v in _as_dict(table.get('result_set')).items()
          if k != 'data'
      }
      table = {**table, 'result_set': result_set, 'truncated': True}
    self._tables.append(table)
    return []

  def _on_chart(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    chart = payload
    if _json_size(chart) > self._max_bytes:
      chart = {k: v for k, v in chart.items() if k != 'chart_spec'}
      chart['truncated'] = True
    self._charts.append(chart)
    return []

  def _on_suggested_queries(
      self, name: str, payload: dict[str, Any]
  ) -> list[Event]:
    del name
    self._suggested_queries.extend(_as_list(payload.get('suggested_queries')))
    return []

  def _on_response(self, name: str, payload: dict[str, Any]) -> list[Event]:
    del name
    self._final_response = payload
    metadata = _as_dict(payload.get('metadata'))
    if (
        self._assistant_message_id is None
        and metadata.get('assistant_message_id') is not None
    ):
      self._assistant_message_id = str(metadata['assistant_message_id'])
    if self._thread_id is None and metadata.get('thread_id') is not None:
      self._thread_id = str(metadata['thread_id'])
    return []

  def _on_error(self, name: str, payload: dict[str, Any]) -> list[Event]:
    self._error = payload
    code = payload.get('code') or payload.get('error_code')
    message = payload.get('message') or payload.get('error_message')
    return [
        self._event(
            error_code=str(code) if code else _DEFAULT_ERROR_CODE,
            error_message=str(message) if message else _DEFAULT_ERROR_MESSAGE,
            custom_metadata={METADATA_KEY: {'event': name, 'data': payload}},
        )
    ]
