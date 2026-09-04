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

from typing import Any
from typing import AsyncGenerator
from typing import Awaitable
from typing import Callable

from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event

_STATE_KEY_PREFIX = '_snowflake_cortex_'

_SUB_AGENTS_NOT_SUPPORTED_MESSAGE = (
    'SnowflakeCortexAgent does not support sub_agents: the agent loop runs'
    ' inside Snowflake, where an ADK sub-agent cannot be reached.'
)

_PARENT_NOT_SUPPORTED_MESSAGE = (
    'SnowflakeCortexAgent must run as an ADK root agent and cannot be a'
    ' sub-agent: Snowflake runs the agent loop and owns the conversation'
    ' thread, so it cannot take part in the turn of an ADK parent.'
)


class SnowflakeCortexAgent(BaseAgent):
  """Runs a Snowflake Cortex Agent as an ADK agent node.

  Each ADK turn sends the user's message to an existing Cortex Agent object
  through the Cortex Agents Run API and streams the run back as ADK events.
  The Snowflake thread and the last message id are kept in ADK session state
  under a key scoped to this agent's ``name``, so a conversation continues
  across turns and survives a restart. Persisting that cursor needs the ADK
  ``Runner``, which is what applies a yielded event's ``state_delta``.

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
      return {'Authorization': f'Bearer {load_snowflake_token()}'}

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

  Typically an ``Authorization`` header. Called with the ``ReadonlyContext``
  of the current invocation and may be sync or async. Excluded from
  serialization and ``repr`` so that a token never reaches the ``adk web``
  agent graph, logs, or a session store.
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

  A server-side tool result larger than this is truncated to its metadata
  before it is recorded in a ``FunctionResponse`` event. Tool results are
  persisted with the session, so this bounds how much a single SQL result set
  can grow it.
  """

  include_thinking_in_final_event: bool = False
  """Whether the final event also carries the completed reasoning text.

  Off by default so that reasoning is not written to the session store with
  the final event. Reasoning deltas are still streamed as partial events in
  SSE mode.
  """

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

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    self._validate_no_sub_agents()
    raise NotImplementedError(
        'SnowflakeCortexAgent cannot run yet: the Cortex Agents Run API'
        ' client is not implemented.'
    )
    yield  # AsyncGenerator requires having at least one yield statement
