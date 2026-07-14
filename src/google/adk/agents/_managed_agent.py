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

from __future__ import annotations

import inspect
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from google.genai.interactions import CreateAgentInteractionAgentConfigParam
from google.genai.interactions import CreateAgentInteractionEnvironmentParam
from google.genai.interactions import ToolParam
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..events.event import Event
from ..flows.llm_flows.interactions_processor import _find_previous_interaction_state
from ..models.interactions_utils import _build_mcp_server_param
from ..models.interactions_utils import _convert_content_to_step
from ..models.interactions_utils import _create_interactions
from ..models.interactions_utils import build_interactions_request_log
from ..models.interactions_utils import convert_tools_config_to_interactions_format
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..telemetry import tracer
from ..tools._remote_mcp_server import RemoteMcpServer
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ..utils._google_client_headers import get_tracking_http_options
from ..utils._google_client_headers import merge_tracking_headers
from ..utils.content_utils import to_user_content
from ..utils.context_utils import Aclosing
from ..utils.env_utils import is_enterprise_mode_enabled
from .base_agent import BaseAgent
from .context import Context
from .invocation_context import InvocationContext
from .readonly_context import ReadonlyContext
from .run_config import StreamingMode

if TYPE_CHECKING:
  from google.genai import Client

logger = logging.getLogger('google_adk.' + __name__)

# The Managed Agents / Interactions API is only served from the `global`
# location; regional endpoints reject these calls (e.g. "Resource setup has
# just started"). We pin it here so the agent works regardless of
# GOOGLE_CLOUD_LOCATION in the caller's environment. The project is still
# resolved from the environment / ADC as usual.
_MANAGED_AGENT_LOCATION = 'global'


def _resolve_client_location(api_client: Client) -> Optional[str]:
  """Return the client's resolved location, or ``None`` if unavailable.

  google-genai 2.9.0 exposes no public accessor for a ``Client``'s location, so
  we read the genai-internal ``client._api_client.location``. This is the single
  remaining private dependency; the enterprise backend flag uses the public
  ``Client.vertexai`` property. A missing value (e.g. test doubles) yields
  ``None`` and is treated as acceptable.
  """
  try:
    # google-genai 2.9.0 has no public accessor for a Client's location.
    return api_client._api_client.location  # pylint: disable=protected-access
  except AttributeError:
    return None


def _validate_client_location(api_client: Client) -> None:
  """Reject an injected enterprise client not targeting the `global` location.

  The Managed Agents API is only served from `global`. This check applies only
  to enterprise (Vertex) clients: the Gemini Developer API has no location
  concept, yet google-genai still stamps `GOOGLE_CLOUD_LOCATION` onto every
  client's `_api_client.location`, so a Developer-API client must not be
  rejected for it. We do not override a caller-supplied client, but a
  non-`global` enterprise client cannot work, so we reject it loudly. The
  backend is read from the public `Client.vertexai` property; the resolved
  location has no public accessor in google-genai 2.9.0, so it is read from the
  genai-internal `client._api_client.location` via `_resolve_client_location`
  (an unresolvable location is treated as acceptable).
  """
  # `Client.vertexai` is the public accessor (it returns False for the Gemini
  # Developer API, which has no location concept); only enterprise (Vertex)
  # clients have a meaningful location.
  if not api_client.vertexai:
    return
  location = _resolve_client_location(api_client)
  if isinstance(location, str) and location != _MANAGED_AGENT_LOCATION:
    raise ValueError(
        'ManagedAgent requires an enterprise client configured for the'
        f" '{_MANAGED_AGENT_LOCATION}' location; got location='{location}'."
        ' The Managed Agents API is only served from'
        f" '{_MANAGED_AGENT_LOCATION}'."
    )


class ManagedAgent(BaseAgent):
  """An agent backed by the Managed Agents API (interactions.create).

  This agent calls the Managed Agents API directly from its execution loop.
  Only server-side tools are supported: ADK built-in tools, raw
  ``google.genai.types.Tool`` configs (the kinds the interactions converter
  understands), and server-side remote MCP servers declared as
  ``RemoteMcpServer`` specs (forwarded to the backend as an ``MCPServerParam``).
  Client-executed tools (FunctionTool/callables) and raw
  ``types.Tool.mcp_servers`` configs are not supported and are rejected.

  ManagedAgent supports streaming interactions only. Interactions are always
  created with ``background=True`` (required by the Managed Agents workflow) and
  consumed over the streaming connection; non-streaming / background-polling
  execution is not yet supported.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

  agent_id: str
  """The Managed Agent id (e.g. 'antigravity-preview-05-2026' or 'agents/ID')."""

  environment: Optional[CreateAgentInteractionEnvironmentParam] = None
  """A sandbox environment spec (e.g. ``{'type': 'remote'}``) or an existing
  environment id string to reuse across turns."""

  agent_config: Optional[CreateAgentInteractionAgentConfigParam] = None
  """Runtime configuration passed to interactions.create."""

  tools: list[
      Union[types.Tool, BaseTool, Callable[..., Any], RemoteMcpServer]
  ] = Field(default_factory=list)
  """Server-side tools: ADK built-in tools, raw types.Tool configs, or
  RemoteMcpServer specs for server-side remote MCP."""

  mode: Literal['single_turn'] | None = None
  """Composition mode.

  Only ``single_turn`` is supported: the agent runs as an inline single-turn
  tool of a parent ``LlmAgent`` (the recommended replacement for ``AgentTool``),
  preserving its internal events in the shared session. ``None`` (default)
  leaves the agent usable as an LLM-transfer target.
  """

  _api_client: Optional[Client] = PrivateAttr(default=None)

  def __init__(
      self, *, api_client: Optional[Client] = None, **kwargs: Any
  ) -> None:
    super().__init__(**kwargs)
    if api_client is not None:
      _validate_client_location(api_client)
    self._api_client = api_client

  @property
  def api_client(self) -> Client:
    """The genai client, lazily created if none was injected.

    The backend is resolved from the environment
    (``GOOGLE_GENAI_USE_ENTERPRISE`` or the legacy
    ``GOOGLE_GENAI_USE_VERTEXAI``), matching google-genai semantics; the
    no-env default is the Gemini Developer API. The enterprise backend is
    pinned to the ``global`` location (the Managed Agents API is only served
    from ``global``); the Developer API takes no ``location`` (it is
    meaningless there).
    """
    if self._api_client is None:
      from google.genai import Client

      if is_enterprise_mode_enabled():
        self._api_client = Client(
            enterprise=True,
            location=_MANAGED_AGENT_LOCATION,
            http_options=get_tracking_http_options(),
        )
      else:
        self._api_client = Client(
            enterprise=False,
            http_options=get_tracking_http_options(),
        )
    return self._api_client

  async def _resolve_backend_tools(
      self, ctx: InvocationContext
  ) -> list[ToolParam]:
    """Resolve self.tools into interaction ToolParams (server-side only).

    Raw types.Tool configs are passed through; ADK built-in tools are processed
    into native tool configs. ``RemoteMcpServer`` specs are resolved to an
    ``MCPServerParam`` (headers minted at request time via ``header_provider``).
    Client-executed tools (FunctionTool/callables) and raw
    ``types.Tool.mcp_servers`` configs are rejected.
    """
    # Built-in tools are resolved in "managed agent" mode: the request carries
    # the internal _is_managed_agent flag (and no model), so tools that normally
    # gate on a Gemini model still resolve. Nothing here is sent to the API; the
    # real call uses ``agent=self.agent_id``.
    llm_request = LlmRequest(config=types.GenerateContentConfig())
    llm_request._is_managed_agent = True
    tool_context = ToolContext(ctx)
    mcp_params: list[ToolParam] = []

    for tool in self.tools:
      if isinstance(tool, RemoteMcpServer):
        resolved_headers = dict(tool.headers or {})
        if tool.header_provider is not None:
          dynamic = tool.header_provider(ReadonlyContext(ctx))
          if inspect.isawaitable(dynamic):
            dynamic = await dynamic
          if dynamic:
            resolved_headers.update(dynamic)  # dynamic wins on key conflict
        mcp_params.append(_build_mcp_server_param(tool, resolved_headers))
        continue

      if isinstance(tool, types.Tool):
        if tool.mcp_servers:
          raise NotImplementedError(
              'Raw mcp_servers tools are not yet supported by ManagedAgent '
              '(MCP is deferred).'
          )
        if tool.function_declarations:
          raise NotImplementedError(
              'client-executed tools are not yet supported by ManagedAgent: '
              f'{tool!r}'
          )
        if not (
            tool.google_search
            or tool.code_execution
            or tool.url_context
            or tool.computer_use
        ):
          raise NotImplementedError(
              'Unsupported raw types.Tool for ManagedAgent; supported '
              'server-side fields are google_search, code_execution, '
              f'url_context, computer_use: {tool!r}'
          )
        llm_request.config.tools = (llm_request.config.tools or []) + [tool]
        continue

      if not isinstance(tool, BaseTool):
        raise NotImplementedError(
            'client-executed tools are not yet supported by ManagedAgent: '
            f'{tool!r}'
        )

      # Built-in (server-side) tools mutate config.tools directly; tools that
      # register a function declaration via append_tools grow tools_dict and are
      # therefore client-executed.
      before = len(llm_request.tools_dict)
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )
      if len(llm_request.tools_dict) > before:
        # The tool registered a function declaration -> client-executed.
        raise NotImplementedError(
            'client-executed tools are not yet supported by ManagedAgent: '
            f'{tool.name}'
        )

    return (
        convert_tools_config_to_interactions_format(llm_request.config)
        + mcp_params
    )

  def _response_to_event(
      self, ctx: InvocationContext, llm_response: LlmResponse
  ) -> Event:
    """Map a streamed LlmResponse to an ADK Event authored by this agent."""
    base_event = Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
    )
    return Event.model_validate({
        **base_event.model_dump(exclude_none=True),
        **llm_response.model_dump(exclude_none=True),
    })

  def _error_event(
      self,
      ctx: InvocationContext,
      *,
      error_code: str,
      error_message: str,
  ) -> Event:
    """Build a terminal error event authored by this agent.

    Always sets ``turn_complete=True`` so the Runner receives a terminal event
    even when the interactions call/stream fails.
    """
    return Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        error_code=error_code,
        error_message=error_message,
        turn_complete=True,
    )

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Event, None]:
    """Runs the ManagedAgent as a node, threading node_input into user_content.

    When invoked as a single-turn tool (``mode='single_turn'``), the parent's
    tool-call argument arrives as ``node_input``; surface it as the agent's
    ``user_content`` so ``_run_async_impl`` sends it to the interactions API.
    When ``node_input`` is ``None`` (classic agent-tree run), behavior is
    identical to ``BaseAgent._run_impl``.
    """
    parent_context = ctx.get_invocation_context()
    if node_input is not None:
      parent_context = parent_context.model_copy(
          update={'user_content': to_user_content(node_input)}
      )
    async for event in self.run_async(parent_context=parent_context):
      if event.author:
        ctx.event_author = event.author
      if not event.node_info.path and event.author == self.name:
        event.node_info.path = ctx.node_path
      yield event

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Lazy import: google.genai is heavy, so only `types` is imported at module
    # level (see CheckGoogleGenaiLazyImport / base_llm_flow.run_live).
    from google.genai import errors

    # Recovery and tool resolution run outside the try so config errors (e.g.
    # unsupported tools) surface loudly rather than becoming an error event.
    prev_interaction_id, prev_environment_id = _find_previous_interaction_state(
        ctx.session.events,
        agent_name=self.name,
        current_branch=ctx.branch,
    )

    environment = prev_environment_id or self.environment

    input_steps = (
        _convert_content_to_step(ctx.user_content) if ctx.user_content else []
    )
    interaction_tools = await self._resolve_backend_tools(ctx)

    create_kwargs: dict[str, Any] = {
        'agent': self.agent_id,
        'input': input_steps,
        # The Managed Agents interactions workflow (server-side tools + remote
        # environment) requires background execution. ManagedAgent supports
        # streaming only, so the background result is consumed via the open SSE
        # stream (stream=True at the _create_interactions call site below).
        'background': True,
    }
    if interaction_tools:
      create_kwargs['tools'] = interaction_tools
    if environment is not None:
      create_kwargs['environment'] = environment
    if self.agent_config is not None:
      create_kwargs['agent_config'] = self.agent_config
    if prev_interaction_id:
      create_kwargs['previous_interaction_id'] = prev_interaction_id

    # Request-time header merge, parity with google_llm.generate_content_async:
    # combine any RunConfig headers with ADK tracking headers, non-destructively.
    run_config = ctx.run_config
    run_config_headers = (
        run_config.http_options.headers
        if run_config is not None and run_config.http_options is not None
        else None
    )
    extra_headers = merge_tracking_headers(
        run_config_headers, framework_label='managed_agent'
    )

    logger.info(
        'Sending request via interactions API, agent: %s, stream: %s, '
        'previous_interaction_id: %s, environment: %s',
        self.agent_id,
        True,
        prev_interaction_id,
        environment,
    )
    logger.debug(
        build_interactions_request_log(
            model=self.agent_id,
            input_steps=input_steps,
            system_instruction=None,
            tools=interaction_tools if interaction_tools else None,
            generation_config=None,
            previous_interaction_id=prev_interaction_id,
            stream=True,
        )
    )

    try:
      with tracer.start_as_current_span('managed_agent_interaction'):
        async with Aclosing(
            _create_interactions(
                self.api_client,
                create_kwargs=create_kwargs,
                stream=True,
                extra_headers=extra_headers,
            )
        ) as agen:
          async for llm_response in agen:
            # ManagedAgent always streams from the server, but only surface
            # intermediate partials to the caller in SSE mode. In non-streaming
            # mode (the default) emit just the non-partial events (the
            # aggregated final event, plus any error event), mirroring
            # base_llm_flow's behavior for LlmAgent.
            if (
                ctx.run_config is not None
                and ctx.run_config.streaming_mode == StreamingMode.SSE
            ) or not llm_response.partial:
              yield self._response_to_event(ctx, llm_response)
    except errors.APIError as e:
      # Surface the backend's real status/code (e.g. RESOURCE_EXHAUSTED) instead
      # of a blanket UNKNOWN_ERROR, mirroring the status=='failed' interaction
      # path and base_llm_flow's APIError handling.
      logger.exception('ManagedAgent interaction failed with backend API error')
      yield self._error_event(
          ctx,
          error_code=e.status or 'UNKNOWN_ERROR',
          error_message=e.message or str(e),
      )
    except Exception as e:  # pylint: disable=broad-except
      # Top-level safety net: any other failure still becomes a terminal error
      # event so the Runner never hangs.
      logger.exception('ManagedAgent interaction failed')
      yield self._error_event(
          ctx, error_code='UNKNOWN_ERROR', error_message=str(e)
      )
