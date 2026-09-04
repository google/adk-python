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

"""Runs a Snowflake Cortex Agent as an ADK agent.

Wraps an existing, named Snowflake Cortex Agent object as a native ADK
``BaseAgent`` node. Snowflake runs the agent loop and owns the conversation
thread; this node sends each ADK turn to the Cortex Agents Run API and
projects the resulting SSE stream onto ADK events.

Because the loop and the thread live in Snowflake, a ``SnowflakeCortexAgent``
must run as an ADK root agent: it accepts no ``sub_agents`` and refuses to be
adopted by a parent agent.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Awaitable
from typing import Callable

import httpx
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ...agents._streaming_mode import StreamingMode
from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ._client import CortexTransportError
from ._client import SnowflakeCortexClient
from ._event_converter import CortexEventConverter

logger = logging.getLogger('google_adk.' + __name__)

_STATE_KEY_PREFIX = '_snowflake_cortex_'
_CURSOR_SCHEMA_VERSION = 1

_SUB_AGENTS_NOT_SUPPORTED_MESSAGE = (
    'SnowflakeCortexAgent does not support sub_agents: the agent loop runs'
    ' inside Snowflake, where an ADK sub-agent cannot be reached.'
)

_PARENT_NOT_SUPPORTED_MESSAGE = (
    'SnowflakeCortexAgent must run as an ADK root agent and cannot be a'
    ' sub-agent: Snowflake runs the agent loop and owns the conversation'
    ' thread, so it cannot take part in the turn of an ADK parent.'
)


@dataclasses.dataclass(frozen=True)
class _Cursor:
  """Where the next turn continues in the Snowflake thread."""

  thread_id: str
  parent_message_id: str

  def to_state(self, fingerprint: str) -> dict[str, Any]:
    return {
        'schema_version': _CURSOR_SCHEMA_VERSION,
        'resource_fingerprint': fingerprint,
        'thread_id': self.thread_id,
        'parent_message_id': self.parent_message_id,
    }


class SnowflakeCortexAgent(BaseAgent):
  """Runs a Snowflake Cortex Agent as an ADK agent node.

  Each ADK turn sends the user's message to an existing Cortex Agent object
  through the Cortex Agents Run API and streams the run back as ADK events:
  partial text and reasoning deltas in SSE streaming mode, server-side tool
  calls and results as ``FunctionCall`` / ``FunctionResponse`` events, and one
  final event carrying the answer with citations, warnings, tables, charts and
  suggested queries under ``custom_metadata['snowflake_cortex']``.

  The Snowflake thread and the last assistant message id are kept in ADK
  session state under a key scoped to this agent's ``name``, so a conversation
  continues across turns and survives a restart. Persisting that cursor needs
  the ADK ``Runner``, which is what applies a yielded event's ``state_delta``.
  The cursor also records which account and Cortex Agent object it belongs
  to; pointing an existing session at a different one fails loudly rather
  than mixing two conversations.

  Credentials are supplied per request by ``header_provider`` rather than
  stored on the agent, and the provider is excluded from ``repr`` and
  serialization.

  Must be an ADK root agent: ``sub_agents`` are rejected and a parent cannot
  adopt it.

  Example:
    ```python
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.labs.snowflake import SnowflakeCortexAgent

    def bearer_headers(ctx: ReadonlyContext) -> dict[str, str]:
      return {
          'Authorization': f'Bearer {load_snowflake_token()}',
          'X-Snowflake-Authorization-Token-Type': 'PROGRAMMATIC_ACCESS_TOKEN',
      }

    root_agent = SnowflakeCortexAgent(
        name='sales_analyst',
        account_url='https://<account>.snowflakecomputing.com',
        database='SALES_DB',
        schema_name='ANALYTICS',
        cortex_agent_name='SALES_AGENT',
        header_provider=bearer_headers,
    )
    ```
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      use_attribute_docstrings=True,
      extra='forbid',
  )

  account_url: str
  """Base URL of the Snowflake account.

  For example ``https://<account>.snowflakecomputing.com``, without a trailing
  slash.
  """

  database: str
  """Database that holds the Cortex Agent object."""

  schema_name: str
  """Schema that holds the Cortex Agent object.

  Named ``schema_name`` because ``schema`` is reserved by pydantic.
  """

  cortex_agent_name: str
  """Name of the Cortex Agent object in Snowflake.

  Distinct from ``name``, which identifies this node within ADK; two ADK
  agents may point at the same Snowflake object.
  """

  header_provider: Callable[
      [ReadonlyContext], dict[str, str] | Awaitable[dict[str, str]]
  ] = Field(exclude=True, repr=False)
  """Supplies the HTTP headers for each Snowflake request.

  Typically ``Authorization`` and, for tokens other than OAuth, the matching
  ``X-Snowflake-Authorization-Token-Type``. Called with the
  ``ReadonlyContext`` of the current invocation and may be sync or async.
  Excluded from serialization and ``repr`` so that a token never reaches the
  ``adk web`` agent graph, logs, or a session store.
  """

  http_client: httpx.AsyncClient | None = Field(
      default=None, exclude=True, repr=False
  )
  """An ``httpx.AsyncClient`` to send Snowflake requests through.

  Share one to pool connections across agents or to configure proxies and
  certificates. When omitted the agent creates its own and closes it in
  ``cleanup``. Excluded from serialization: it is runtime wiring.
  """

  timeout: float = Field(default=900.0, gt=0)
  """Seconds to wait on Snowflake before the turn fails with a timeout.

  Cortex Agent runs that plan, execute SQL and summarize can take minutes, so
  the default is deliberately long.
  """

  cancel_on_disconnect: bool = True
  """Whether to cancel the Snowflake run when the ADK consumer stops reading.

  Best effort: the cancel is attempted, not guaranteed, and Snowflake keeps
  whatever partial output it already produced in the thread either way.
  """

  max_tool_result_bytes: int = Field(default=32 * 1024, gt=0)
  """Upper bound on the serialized size of one recorded tool result.

  A server-side tool result larger than this is cut down before it is
  recorded in a ``FunctionResponse`` event: SQL rows go first, then each
  block is reduced to its type and key sizes. Tool results are persisted
  with the session, so this bounds how much a single result can grow it.
  """

  include_thinking_in_final_event: bool = False
  """Whether the final event also carries the completed reasoning text.

  Off by default so that reasoning is not written to the session store with
  the final event. Reasoning deltas are still streamed as partial events in
  SSE mode.
  """

  _cortex_client: SnowflakeCortexClient | None = PrivateAttr(default=None)

  @override
  def model_post_init(self, __context: Any) -> None:
    super().model_post_init(__context)
    self._validate_no_sub_agents()

  def _validate_no_sub_agents(self) -> None:
    # Called again on entry to `_run_async_impl` because `sub_agents` can be
    # mutated or `model_copy`-ed after construction, bypassing
    # `model_post_init`.
    if self.sub_agents:
      raise ValueError(_SUB_AGENTS_NOT_SUPPORTED_MESSAGE)

  def __setattr__(self, name: str, value: Any) -> None:
    # `BaseAgent` adopts a child by assigning `parent_agent` from the parent's
    # `model_post_init`, so refusing the assignment fails the parent's
    # construction at its `sub_agents=[...]` declaration rather than a turn.
    if name == 'parent_agent' and value is not None:
      raise ValueError(_PARENT_NOT_SUPPORTED_MESSAGE)
    super().__setattr__(name, value)

  def _state_key(self) -> str:
    # Scoped by agent name so two `SnowflakeCortexAgent`s in one ADK session
    # do not continue each other's Snowflake thread.
    return _STATE_KEY_PREFIX + self.name

  def _resource_fingerprint(self) -> str:
    # The account is part of it: a thread id only means something within the
    # account that issued it, whatever the object is called.
    material = '|'.join([
        self.account_url.rstrip('/'),
        self.database,
        self.schema_name,
        self.cortex_agent_name,
    ])
    return 'sha256:' + hashlib.sha256(material.encode('utf-8')).hexdigest()

  def _read_cursor(self, stored: Any) -> _Cursor | None:
    """Validates the stored cursor, or returns None when there is none yet."""
    if stored is None:
      return None
    key = self._state_key()
    remedy = f' Remove session state key {key!r} or start a new session.'
    if (
        not isinstance(stored, dict)
        or stored.get('schema_version') != _CURSOR_SCHEMA_VERSION
    ):
      raise ValueError(
          f'Session state key {key!r} does not hold a SnowflakeCortexAgent'
          ' cursor this version understands.'
          + remedy
      )
    if stored.get('resource_fingerprint') != self._resource_fingerprint():
      raise ValueError(
          f'Session state key {key!r} holds a Snowflake thread that belongs'
          ' to a different account, database, schema or Cortex Agent than'
          ' this agent is configured with; continuing it would mix two'
          ' conversations.'
          + remedy
      )
    thread_id = stored.get('thread_id')
    parent_message_id = stored.get('parent_message_id')
    if not isinstance(thread_id, str) or not isinstance(parent_message_id, str):
      raise ValueError(
          f'Session state key {key!r} holds a cursor without string'
          ' thread_id and parent_message_id.'
          + remedy
      )
    return _Cursor(thread_id=thread_id, parent_message_id=parent_message_id)

  def _user_text(self, ctx: InvocationContext) -> str:
    parts = (
        ctx.user_content.parts
        if ctx.user_content and ctx.user_content.parts
        else []
    )
    text = '\n'.join(part.text for part in parts if part.text)
    if not text.strip():
      raise ValueError(
          'SnowflakeCortexAgent needs a text message: this version sends only'
          ' text to the Cortex Agents Run API, and the current user content'
          ' has none.'
      )
    return text

  def _get_client(self) -> SnowflakeCortexClient:
    if self._cortex_client is None:
      self._cortex_client = SnowflakeCortexClient(
          account_url=self.account_url,
          database=self.database,
          schema_name=self.schema_name,
          cortex_agent_name=self.cortex_agent_name,
          header_provider=self.header_provider,
          timeout=self.timeout,
          http_client=self.http_client,
      )
    return self._cortex_client

  async def cleanup(self) -> None:
    """Closes the HTTP client this agent created.

    A shared ``http_client`` is left open for its owner to close.
    """
    if self._cortex_client is not None:
      await self._cortex_client.aclose()
      self._cortex_client = None

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    self._validate_no_sub_agents()
    text = self._user_text(ctx)
    cursor = self._read_cursor(ctx.session.state.get(self._state_key()))
    readonly_ctx = ReadonlyContext(ctx)
    client = self._get_client()
    if cursor is None:
      cursor = _Cursor(
          thread_id=await client.create_thread(readonly_ctx),
          parent_message_id='0',
      )

    streaming = bool(
        ctx.run_config and ctx.run_config.streaming_mode == StreamingMode.SSE
    )
    converter = CortexEventConverter(
        ctx=ctx,
        author=self.name,
        streaming=streaming,
        max_tool_result_bytes=self.max_tool_result_bytes,
        include_thinking_in_final_event=self.include_thinking_in_final_event,
        thread_id=cursor.thread_id,
    )

    try:
      async with client.run(
          readonly_ctx,
          thread_id=cursor.thread_id,
          parent_message_id=cursor.parent_message_id,
          text=text,
      ) as events:
        async for sse_event in events:
          for event in converter.convert(sse_event):
            yield event
          if converter.is_done or converter.failed:
            # Nothing useful follows the terminator; stop reading rather than
            # wait for Snowflake to close the connection.
            break
    except (GeneratorExit, asyncio.CancelledError):
      # The consumer is gone. `run()` has already closed the upstream
      # response; awaiting here is fine, yielding would not be.
      await self._cancel_abandoned_run(readonly_ctx, client, converter)
      raise

    if converter.failed:
      # The terminal error event has been yielded; the cursor stays where it
      # was so the next turn continues from the last good message.
      return
    if not (converter.is_done and converter.has_final_response):
      raise CortexTransportError(
          'The Snowflake stream ended before the run finished, so the answer'
          ' is incomplete. The conversation cursor was left unchanged.',
          timed_out=False,
      )

    state_delta: dict[str, Any] | None = None
    if converter.assistant_message_id is not None:
      # Only the assistant id may become the parent of the next turn; the
      # user id would fork the thread.
      state_delta = {
          self._state_key(): (
              _Cursor(
                  thread_id=cursor.thread_id,
                  parent_message_id=converter.assistant_message_id,
              ).to_state(self._resource_fingerprint())
          )
      }
    yield converter.final_event(state_delta=state_delta)

  async def _cancel_abandoned_run(
      self,
      readonly_ctx: ReadonlyContext,
      client: SnowflakeCortexClient,
      converter: CortexEventConverter,
  ) -> None:
    if not self.cancel_on_disconnect or converter.is_done or converter.failed:
      return
    if converter.thread_id is None or converter.user_message_id is None:
      # Snowflake names a run `{thread_id}-{user_message_id}`, so until the
      # user message is acknowledged there is nothing to cancel by.
      return
    run_id = f'{converter.thread_id}-{converter.user_message_id}'
    try:
      await client.cancel_run(readonly_ctx, run_id)
    except Exception:  # pylint: disable=broad-exception-caught
      # Best effort only: the consumer that would care has already gone.
      logger.debug('Cancelling an abandoned Snowflake Cortex run failed.')
