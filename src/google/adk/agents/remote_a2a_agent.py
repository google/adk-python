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

import ipaddress
import json
import logging
from pathlib import Path
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Union
from urllib.parse import urlparse

from a2a.client import Client as A2AClient
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.types import AgentCard
from a2a.types import Message as A2AMessage
from a2a.types import Part as A2APart
from a2a.types import TaskArtifactUpdateEvent as A2ATaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatusUpdateEvent as A2ATaskStatusUpdateEvent
from google.adk.platform import uuid as platform_uuid
from google.genai import types as genai_types
import httpx

from ..a2a import _compat

try:
  from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
except ImportError:
  # Fallback for older versions of a2a-sdk.
  AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent.json"


from ..a2a.agent.config import A2aRemoteAgentConfig
from ..a2a.agent.interceptors.new_integration_extension import _NEW_A2A_ADK_INTEGRATION_EXTENSION
from ..a2a.agent.interceptors.new_integration_extension import _new_integration_extension_interceptor
from ..a2a.agent.utils import execute_after_request_interceptors
from ..a2a.agent.utils import execute_before_card_request_interceptors
from ..a2a.agent.utils import execute_before_request_interceptors
from ..a2a.converters.event_converter import convert_a2a_message_to_event
from ..a2a.converters.event_converter import convert_a2a_task_to_event
from ..a2a.converters.event_converter import convert_event_to_a2a_message
from ..a2a.converters.part_converter import A2APartToGenAIPartConverter
from ..a2a.converters.part_converter import convert_a2a_part_to_genai_part
from ..a2a.converters.part_converter import convert_genai_part_to_a2a_part
from ..a2a.converters.part_converter import GenAIPartToA2APartConverter
from ..a2a.converters.to_adk_event import _create_mock_function_call_for_required_user_input
from ..a2a.converters.to_adk_event import MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_AUTH
from ..a2a.converters.to_adk_event import MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_INPUT
from ..a2a.experimental import a2a_experimental
from ..a2a.logs.log_utils import build_a2a_request_log
from ..a2a.logs.log_utils import build_a2a_response_log
from ..agents.invocation_context import InvocationContext
from ..agents.llm.task._finish_task_tool import FINISH_TASK_ERROR_RESULT
from ..agents.llm.task._finish_task_tool import FINISH_TASK_SUCCESS_RESULT
from ..agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from ..agents.llm.task._finish_task_tool import get_output_wrapper_key
from ..agents.llm.task._finish_task_tool import is_finish_task_terminal_fr
from ..events.event import Event
from ..flows.llm_flows.contents import _is_other_agent_reply
from ..flows.llm_flows.contents import _present_other_agent_message
from ..flows.llm_flows.functions import find_matching_function_call
from ..flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from ..flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..flows.llm_flows.functions import REQUEST_INPUT_FUNCTION_CALL_NAME
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from .base_agent import BaseAgent

__all__ = [
    "A2AClientError",
    "AGENT_CARD_WELL_KNOWN_PATH",
    "AgentCardResolutionError",
    "RemoteA2aAgent",
]


# Constants
A2A_METADATA_PREFIX = "a2a:"
DEFAULT_TIMEOUT = 600.0

_DEFAULT_PORTS = {"http": 80, "https": 443}

logger = logging.getLogger("google_adk." + __name__)


# Function call names whose pause is resolved locally (ADK request-* tools or a
# workflow HITL node); their response is flattened to text before forwarding.
_HUMAN_INPUT_FUNCTION_CALL_NAMES = frozenset({
    MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_INPUT,
    MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_AUTH,
    REQUEST_INPUT_FUNCTION_CALL_NAME,
    REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
    REQUEST_EUC_FUNCTION_CALL_NAME,
})

_CREDENTIAL_FUNCTION_CALL_NAMES = frozenset({
    MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_AUTH,
    REQUEST_EUC_FUNCTION_CALL_NAME,
})

_RESULT_KEY = "result"

# Top-level keys of a serialized AuthConfig, the shape an adk_request_credential
# response carries (see auth.auth_preprocessor); snake_case and camelCase forms.
_CREDENTIAL_PAYLOAD_KEYS = frozenset({
    "auth_scheme",
    "authScheme",
    "exchanged_auth_credential",
    "exchangedAuthCredential",
    "raw_auth_credential",
    "rawAuthCredential",
})


def _payload_is_auth_config(payload: Any) -> bool:
  """Whether a payload looks like a serialized AuthConfig (fail closed)."""
  candidate = payload
  if isinstance(payload, dict) and len(payload) == 1 and _RESULT_KEY in payload:
    candidate = payload[_RESULT_KEY]
  return isinstance(candidate, dict) and any(
      key in candidate for key in _CREDENTIAL_PAYLOAD_KEYS
  )


def _is_credential_function_response(
    function_response: genai_types.FunctionResponse,
    matched_call_names: Optional[set[str]] = None,
) -> bool:
  """Whether a function_response carries credential material (fail closed)."""
  if matched_call_names and not matched_call_names.isdisjoint(
      _CREDENTIAL_FUNCTION_CALL_NAMES
  ):
    return True
  if function_response.name in _CREDENTIAL_FUNCTION_CALL_NAMES:
    return True
  return _payload_is_auth_config(function_response.response)


def _render_user_function_response(
    response: Optional[dict[str, Any]],
) -> Optional[str]:
  """Renders a human-input response payload as text, or None if empty."""
  # ADK's mock path wraps the answer as {"result": <text>}; workflow producers
  # send the resolved parameters directly (e.g. {"company_name": "Okta"}).
  if not response:
    return None
  if (
      isinstance(response, dict)
      and len(response) == 1
      and _RESULT_KEY in response
  ):
    value = response[_RESULT_KEY]
    return None if value is None else str(value)
  return json.dumps(response, default=str)


def _sanitize_user_function_response_event(
    event: Event,
    trusted_call_names_by_id: dict[Optional[str], set[str]],
    id_less_call_is_ambiguous: bool,
) -> Event:
  """Returns a copy of ``event`` with its parts sanitized for forwarding."""
  if event.content is None:
    return event
  new_event = event.model_copy(deep=True)
  # ``event.content`` is non-None (checked above) and ``model_copy`` preserves
  # it; bind a local so the checker keeps it narrowed after ``.parts`` is set.
  new_content = new_event.content
  assert new_content is not None
  parts = new_content.parts or []

  def _is_human_input(fr: genai_types.FunctionResponse) -> bool:
    names = trusted_call_names_by_id.get(fr.id)
    if not names or names.isdisjoint(_HUMAN_INPUT_FUNCTION_CALL_NAMES):
      return False
    # An id-less response with an unknown name can't be classified by id when a
    # call the rewrite must not flatten shares the id-less bucket.
    if (
        id_less_call_is_ambiguous
        and fr.id is None
        and fr.name not in _HUMAN_INPUT_FUNCTION_CALL_NAMES
    ):
      return False
    return True

  def _is_credential(fr: genai_types.FunctionResponse) -> bool:
    return _is_credential_function_response(
        fr, trusted_call_names_by_id.get(fr.id)
    )

  # If any function_response is kept as data, the message must stay a resume: no
  # text (including a flattened answer) can ride alongside it.
  preserve_as_resume = any(
      p.function_response is not None
      and not _is_credential(p.function_response)
      and not _is_human_input(p.function_response)
      for p in parts
  )

  new_parts: list[genai_types.Part] = []
  for part in parts:
    fr = part.function_response
    if fr is None:
      if preserve_as_resume and part.text is not None:
        continue
      new_parts.append(part)
      continue
    if _is_credential(fr):
      continue
    if not _is_human_input(fr) or preserve_as_resume:
      new_parts.append(part)
      continue
    text_value = _render_user_function_response(fr.response)
    if text_value is not None:
      new_parts.append(genai_types.Part(text=text_value))

  new_content.parts = new_parts
  return new_event


def _is_loopback_host(hostname: Optional[str]) -> bool:
  """Returns whether a hostname names the local machine.

  Covers ``localhost`` and the reserved ``*.localhost`` names as well as any
  literal loopback address, so the local-development pattern the A2A helpers
  emit -- a plain-http card served from ``localhost`` -- keeps working.
  """
  if not hostname:
    return False
  host = hostname.strip("[]").lower()
  if host == "localhost" or host.endswith(".localhost"):
    return True
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return False


def _url_origin(url: str) -> tuple[str, str, Optional[int]]:
  """Returns the ``(scheme, host, port)`` origin triple for a URL.

  Raises:
    ValueError: If the URL carries a malformed port.
  """
  parsed = urlparse(url)
  scheme = parsed.scheme.lower()
  return (
      scheme,
      (parsed.hostname or "").lower(),
      (parsed.port or _DEFAULT_PORTS.get(scheme)),
  )


@a2a_experimental
class AgentCardResolutionError(Exception):
  """Raised when agent card resolution fails."""

  pass


@a2a_experimental
class A2AClientError(Exception):
  """Raised when A2A client operations fail."""

  pass


def _text_from_content(content: Optional[genai_types.Content]) -> Optional[str]:
  """Joins the text parts of a content, or None when there is no text."""
  if content is None or not content.parts:
    return None
  texts = [part.text for part in content.parts if part.text]
  return "\n".join(texts) if texts else None


def _create_finish_task_event(
    ctx: InvocationContext,
    agent_name: str,
    *,
    output: Any = None,
    error_message: Optional[str] = None,
    is_error: bool = False,
) -> Event:
  """Creates a finish_task Event."""
  return Event(
      author=agent_name,
      invocation_id=ctx.invocation_id,
      branch=ctx.branch,
      isolation_scope=ctx.isolation_scope,
      error_message=error_message,
      content=genai_types.Content(
          role="user",
          parts=[
              genai_types.Part(
                  function_response=genai_types.FunctionResponse(
                      name=FINISH_TASK_TOOL_NAME,
                      response={
                          "result": (
                              FINISH_TASK_ERROR_RESULT
                              if is_error
                              else FINISH_TASK_SUCCESS_RESULT
                          )
                      },
                  )
              )
          ],
      ),
      output=output,
  )


def _create_task_failure_events(
    error_text: str,
    ctx: InvocationContext,
    agent_name: str,
    task_id: str,
    a2a_request: Any = None,
) -> tuple[Event, Event]:
  """Creates events for a failed remote task."""
  error_message = f"Remote A2A task failed: {error_text}"
  error_event_metadata: dict[str, Any] = {
      A2A_METADATA_PREFIX + "error": error_message,
      A2A_METADATA_PREFIX + "task_id": task_id,
  }
  if a2a_request is not None:
    error_event_metadata[A2A_METADATA_PREFIX + "request"] = _compat.a2a_to_dict(
        a2a_request
    )
  error_event = Event(
      author=agent_name,
      invocation_id=ctx.invocation_id,
      branch=ctx.branch,
      isolation_scope=ctx.isolation_scope,
      error_message=error_message,
      custom_metadata=error_event_metadata,
  )
  finish_event = _create_finish_task_event(
      ctx=ctx,
      agent_name=agent_name,
      error_message=error_message,
      is_error=True,
  )
  return error_event, finish_event


def _add_mock_function_call(event: Event, state: TaskState) -> None:
  """Generates a mock function call for input-required events if applicable."""
  if event.content is None:
    return

  output_parts, long_running_tool_ids = (
      _create_mock_function_call_for_required_user_input(
          state,
          event.content.parts or [],
          event.long_running_tool_ids or set(),
      )
  )
  event.content.parts = output_parts
  event.long_running_tool_ids = long_running_tool_ids


def _find_finish_task_args_from_history(
    session: Session,
    isolation_scope: Optional[str] = None,
    completed_fr_event: Optional[Event] = None,
) -> Optional[dict[str, Any]]:
  """Search session events for the latest finish_task FC and return args."""
  matching_fc_id = None
  if completed_fr_event:
    for fr in completed_fr_event.get_function_responses():
      if fr.name == FINISH_TASK_TOOL_NAME:
        matching_fc_id = fr.id
        break

  for event in reversed(session.events):
    if isolation_scope and event.isolation_scope != isolation_scope:
      continue
    calls = event.get_function_calls()
    for fc in calls:
      if fc.name == FINISH_TASK_TOOL_NAME:
        if matching_fc_id is not None:
          if fc.id == matching_fc_id:
            return dict(fc.args or {})
        else:
          return dict(fc.args or {})
  return None


@a2a_experimental
class RemoteA2aAgent(BaseAgent):
  """Agent that communicates with a remote A2A agent via A2A client.

  This agent supports multiple ways to specify the remote agent:
  1. Direct AgentCard object
  2. URL to agent card JSON
  3. File path to agent card JSON

  The agent handles:
  - Agent card resolution and validation
  - HTTP client management with proper resource cleanup
  - A2A message conversion and error handling
  - Session state management across requests
  """

  mode: Literal["task"] | None = None
  """Delegation mode.

  Only ``task`` is supported: the agent runs as a task sub-agent of a parent
  ``LlmAgent`` that owns the conversation across multiple turns, then hands
  control back to the parent when the remote A2A task reaches a terminal
  completed state. Note: this requires the remote agent to invoke the
  ``finish_task`` tool to signal completion (natively supported by ADK
  task-mode agents, or must be manually implemented on custom A2A servers
  by returning a FunctionResponse named ``finish_task`` with a response
  containing a ``result`` key matching ``"Task completed."`` for success, or
  ``"Task failed."`` for failure). Additionally, the client's ``output_schema``
  must be set to mirror the remote agent's output schema to ensure correct
  output unwrapping. ``None`` (default) leaves the agent as a plain
  ``transfer_to_agent`` target.
  """

  def __init__(
      self,
      name: str,
      agent_card: Union[AgentCard, str],
      *,
      description: str = "",
      httpx_client: Optional[httpx.AsyncClient] = None,
      timeout: float = DEFAULT_TIMEOUT,
      genai_part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part,
      a2a_part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
      a2a_client_factory: Optional[A2AClientFactory] = None,
      a2a_request_meta_provider: Optional[
          Callable[[InvocationContext, A2AMessage], dict[str, Any]]
      ] = None,
      full_history_when_stateless: bool = False,
      config: Optional[A2aRemoteAgentConfig] = None,
      use_legacy: bool = True,
      **kwargs: Any,
  ) -> None:
    """Initialize RemoteA2aAgent.

    Args:
      name: Agent name (must be unique identifier)
      agent_card: AgentCard object, URL string, or file path string
      description: Agent description (autopopulated from card if empty)
      httpx_client: Optional shared HTTP client (will create own if not
        provided) [deprecated] Use a2a_client_factory instead.
      timeout: HTTP timeout in seconds
      a2a_client_factory: Optional A2AClientFactory object (will create own if
        not provided)
      a2a_request_meta_provider: Optional callable that takes InvocationContext
        and A2AMessage and returns a metadata object to attach to the A2A
        request.
      full_history_when_stateless: If True, stateless agents (those that do not
        return Tasks or context IDs) will receive all session events on every
        request. If False (default), the behavior depends on the agent's
        delegation mode: True in "task" mode, False otherwise.
      config: Optional configuration object.
      use_legacy: If false, send request to the server including the extension
        indicating that the server should use the new implementation.
      **kwargs: Additional arguments passed to BaseAgent

    Raises:
      ValueError: If name is invalid or agent_card is None
      TypeError: If agent_card is not a supported type
    """
    super().__init__(name=name, description=description, **kwargs)

    if agent_card is None:
      raise ValueError("agent_card cannot be None")

    self._agent_card: Optional[AgentCard] = None
    self._agent_card_source: Optional[str] = None
    self._a2a_client: Optional[A2AClient] = None
    # This is stored to support backward compatible usage of class.
    # In future, the client is expected to be present in the factory.
    self._httpx_client = httpx_client
    if a2a_client_factory and a2a_client_factory._config.httpx_client:
      self._httpx_client = a2a_client_factory._config.httpx_client
    self._httpx_client_needs_cleanup = self._httpx_client is None
    self._timeout = timeout
    self._is_resolved = False
    self._genai_part_converter = genai_part_converter
    self._a2a_part_converter = a2a_part_converter
    self._a2a_client_factory: Optional[A2AClientFactory] = a2a_client_factory
    self._a2a_request_meta_provider = a2a_request_meta_provider
    self._full_history_when_stateless_param = full_history_when_stateless
    self._config = config or A2aRemoteAgentConfig()

    if not use_legacy:
      if self._config.request_interceptors is None:
        self._config.request_interceptors = []
      self._config.request_interceptors.append(
          _new_integration_extension_interceptor
      )

    # Validate and store agent card reference
    if isinstance(agent_card, AgentCard):
      self._agent_card = agent_card
      # Update description if empty. A card supplied directly never goes
      # through the resolution path, so adopt it here instead; a parent agent
      # reads the description to build its transfer instruction, which happens
      # before this agent ever runs.
      if not self.description and agent_card.description:
        self.description = agent_card.description
    elif isinstance(agent_card, str):
      if not agent_card.strip():
        raise ValueError("agent_card string cannot be empty")
      self._agent_card_source = agent_card.strip()
    else:
      raise TypeError(
          "agent_card must be AgentCard, URL string, or file path string, "
          f"got {type(agent_card)}"
      )

  @property
  def _full_history_when_stateless(self) -> bool:
    return self._full_history_when_stateless_param or self.mode == "task"

  @_full_history_when_stateless.setter
  def _full_history_when_stateless(self, value: bool) -> None:
    self._full_history_when_stateless_param = value

  async def _ensure_httpx_client(self) -> httpx.AsyncClient:
    """Ensure HTTP client is available and properly configured."""
    if not self._httpx_client:
      self._httpx_client = httpx.AsyncClient(
          timeout=httpx.Timeout(timeout=self._timeout)
      )
      self._httpx_client_needs_cleanup = True
      if self._a2a_client_factory:
        self._a2a_client_factory = _compat.rebind_client_factory_httpx(
            self._a2a_client_factory, self._httpx_client
        )
    if not self._a2a_client_factory:
      self._a2a_client_factory = A2AClientFactory(
          config=_compat.make_client_config(httpx_client=self._httpx_client)
      )
    return self._httpx_client

  async def _resolve_agent_card_from_url(
      self, url: str, ctx: Optional[InvocationContext] = None
  ) -> AgentCard:
    """Resolve agent card from URL."""
    try:
      parsed_url = urlparse(url)
      if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL format: {url}")

      base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
      relative_card_path = parsed_url.path

      httpx_client = await self._ensure_httpx_client()
      resolver = A2ACardResolver(
          httpx_client=httpx_client,
          base_url=base_url,
      )
      http_kwargs = await execute_before_card_request_interceptors(
          self._config.card_request_interceptors, ctx
      )
      return await resolver.get_agent_card(
          relative_card_path=relative_card_path,
          http_kwargs=http_kwargs,
      )
    except Exception as e:
      raise AgentCardResolutionError(
          f"Failed to resolve AgentCard from URL {url}: {e}"
      ) from e

  async def _resolve_agent_card_from_file(self, file_path: str) -> AgentCard:
    """Resolve agent card from file path."""
    try:
      path = Path(file_path)
      if not path.exists():
        raise FileNotFoundError(f"Agent card file not found: {file_path}")
      if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

      with path.open("r", encoding="utf-8") as f:
        agent_json_data = json.load(f)
        return _compat.parse_agent_card(agent_json_data)
    except json.JSONDecodeError as e:
      raise AgentCardResolutionError(
          f"Invalid JSON in agent card file {file_path}: {e}"
      ) from e
    except Exception as e:
      raise AgentCardResolutionError(
          f"Failed to resolve AgentCard from file {file_path}: {e}"
      ) from e

  async def _resolve_agent_card(
      self, ctx: Optional[InvocationContext] = None
  ) -> AgentCard:
    """Resolve agent card from source."""
    agent_card_source = self._agent_card_source
    if agent_card_source is None:
      raise AgentCardResolutionError("No agent card source was configured.")

    # Determine if source is URL or file path
    if agent_card_source.startswith(("http://", "https://")):
      return await self._resolve_agent_card_from_url(agent_card_source, ctx)
    else:
      return await self._resolve_agent_card_from_file(agent_card_source)

  async def _validate_agent_card(self, agent_card: AgentCard) -> None:
    """Validate resolved agent card."""
    card_url = _compat.agent_card_url(agent_card)
    if not card_url:
      raise AgentCardResolutionError(
          "Agent card must have a valid URL for RPC communication"
      )

    # Additional validation can be added here
    try:
      parsed_url = urlparse(str(card_url))
      if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Invalid RPC URL format")
    except Exception as e:
      raise AgentCardResolutionError(
          f"Invalid RPC URL in agent card: {card_url}, error: {e}"
      ) from e

    self._validate_card_rpc_targets(agent_card)

  def _validate_card_rpc_targets(self, agent_card: AgentCard) -> None:
    """Constrains where a card fetched over the network may aim RPC traffic.

    Every URL the card offers is checked, not only the one this ADK version
    would select, because the client factory negotiates the endpoint across
    the card's whole interface list. Each must be https and share the origin
    the card was fetched from; plain http stays allowed on a loopback host,
    the local-development shape the A2A helpers emit.

    A card passed in directly or read from a local file did not come off the
    network here, so its target is left to the caller.
    """
    source = self._agent_card_source
    if not source or not source.startswith(("http://", "https://")):
      return

    try:
      source_origin = _url_origin(source)
    except ValueError as e:
      raise AgentCardResolutionError(
          f"Invalid agent card source URL: {source}, error: {e}"
      ) from e

    for card_url in _compat.agent_card_rpc_urls(agent_card):
      parsed_card = urlparse(card_url)
      if parsed_card.scheme.lower() != "https" and not _is_loopback_host(
          parsed_card.hostname
      ):
        raise AgentCardResolutionError(
            "Agent card RPC URL must use https, or http on a loopback host:"
            f" {card_url}"
        )

      try:
        card_origin = _url_origin(card_url)
      except ValueError as e:
        raise AgentCardResolutionError(
            f"Invalid RPC URL in agent card: {card_url}, error: {e}"
        ) from e

      if card_origin != source_origin:
        raise AgentCardResolutionError(
            "Agent card RPC URL must have the same origin as the location the"
            f" card was fetched from ({source}): {card_url}"
        )

  async def _ensure_resolved(
      self, ctx: Optional[InvocationContext] = None
  ) -> A2AClient:
    """Resolves the agent card and returns the A2A client for this invocation."""
    # Per the A2A spec, the authenticated (extended) agent card is scoped to a
    # single authenticated session: "Clients retrieving this extended card
    # SHOULD replace their cached public Agent Card ... for the duration of
    # their authenticated session"
    # (https://a2a-protocol.org/latest/specification/#3111-get-extended-agent-card).
    # So when card request interceptors are configured for a URL-based card,
    # resolve the card (and build the client) per invocation using the current
    # ctx, and keep them local rather than caching on shared instance state.
    # This prevents one session's authenticated card from leaking into other
    # sessions. A None ctx means we cannot derive per-session auth, so fall
    # back to the shared cached path.
    per_invocation_card = bool(
        self._config.card_request_interceptors
        and self._agent_card_source
        and ctx is not None
    )

    if not per_invocation_card and self._is_resolved and self._a2a_client:
      return self._a2a_client

    try:
      if per_invocation_card:
        # Build a per-invocation client; never cached on shared state.
        agent_card = await self._resolve_agent_card(ctx)
        await self._validate_agent_card(agent_card)
        await self._ensure_httpx_client()
        if not self._a2a_client_factory:
          raise ValueError("A2A client factory is not available")
        client = self._a2a_client_factory.create(agent_card)
        logger.info("Resolved remote A2A agent per invocation: %s", self.name)
        return client

      # Shared (cached) resolution path.
      if not self._agent_card:

        # Resolve agent card if needed
        self._agent_card = await self._resolve_agent_card(ctx)

        # Validate agent card
        await self._validate_agent_card(self._agent_card)

        # Update description if empty
        if not self.description and self._agent_card.description:
          self.description = self._agent_card.description

      # Initialize A2A client
      if not self._a2a_client:
        await self._ensure_httpx_client()
        # This should be assured via ensure_httpx_client
        if self._a2a_client_factory:
          self._a2a_client = self._a2a_client_factory.create(self._agent_card)

      self._is_resolved = True
      logger.info("Successfully resolved remote A2A agent: %s", self.name)
      return self._a2a_client

    except Exception as e:
      logger.error("Failed to resolve remote A2A agent %s: %s", self.name, e)
      raise AgentCardResolutionError(
          f"Failed to initialize remote A2A agent {self.name}: {e}"
      ) from e

  def _create_a2a_request_for_user_function_response(
      self, ctx: InvocationContext
  ) -> Optional[A2AMessage]:
    """Create A2A request for user function response if applicable.

    Args:
      ctx: The invocation context

    Returns:
      SendMessageRequest if function response found, None otherwise
    """
    if not ctx.session.events or ctx.session.events[-1].author != "user":
      return None
    function_call_event = find_matching_function_call(ctx.session.events)
    if not function_call_event:
      return None

    event = ctx.session.events[-1]

    # Map every pending call to its id by the trusted function CALL name so
    # credential matching does not depend on the human-input set.
    trusted_call_names_by_id: dict[Optional[str], set[str]] = {}
    id_less_call_is_ambiguous = False
    for fc in function_call_event.get_function_calls():
      if fc.name is None:
        continue
      trusted_call_names_by_id.setdefault(fc.id, set()).add(fc.name)
      if fc.id is None and fc.name not in _HUMAN_INPUT_FUNCTION_CALL_NAMES:
        id_less_call_is_ambiguous = True

    event = _sanitize_user_function_response_event(
        event, trusted_call_names_by_id, id_less_call_is_ambiguous
    )

    a2a_message = convert_event_to_a2a_message(
        event, ctx, _compat.ROLE_USER, self._genai_part_converter
    )
    # All parts dropped (e.g. a credential-only resume): the caller rebuilds
    # from history (also dropping credentials); None avoids a task_id crash.
    if a2a_message is None:
      return None

    if function_call_event.custom_metadata:
      metadata = function_call_event.custom_metadata
      task_id = metadata.get(A2A_METADATA_PREFIX + "task_id")
      if isinstance(task_id, str):
        a2a_message.task_id = task_id
      context_id = metadata.get(A2A_METADATA_PREFIX + "context_id")
      if isinstance(context_id, str):
        a2a_message.context_id = context_id

    return a2a_message

  def _is_remote_response(self, event: Event) -> bool:
    is_a2a_resp = bool(
        event.author == self.name
        and event.custom_metadata
        and event.custom_metadata.get(A2A_METADATA_PREFIX + "response", False)
    )
    if is_a2a_resp:
      return True

    # Also stop on synthesized FR events for this agent (meaning the previous
    # delegation to this agent has completed).
    if self.mode == "task":
      for fr in event.get_function_responses():
        if fr.name == self.name:
          return True

    return False

  def _construct_message_parts_from_session(
      self, ctx: InvocationContext
  ) -> tuple[list[A2APart], Optional[str]]:
    """Construct A2A message parts from session events.

    Args:
      ctx: The invocation context

    Returns:
      List of A2A parts extracted from session events, context ID,
      request metadata
    """
    message_parts: list[A2APart] = []
    context_id = None

    events_to_process = []
    task_scope = ctx.isolation_scope if self.mode == "task" else None
    broke_loop = False

    for event in reversed(ctx.session.events):
      if task_scope:
        # In task mode, we restrict the history to the current task scope
        # (isolation scope) to prevent cross-task data leakage and minimize
        # context size.
        if event.isolation_scope == task_scope:
          # Stop walking backward if we hit a previous response from this
          # remote agent. Stateful remote servers already have this history
          # in their session, so we don't need to resend it.
          if self._is_remote_response(event):
            if event.custom_metadata:
              metadata = event.custom_metadata
              context_id = metadata.get(A2A_METADATA_PREFIX + "context_id")
            if not self._full_history_when_stateless or context_id:
              broke_loop = True
              break
          events_to_process.append(event)
          continue
        # We must also include the coordinator's FunctionCall event that
        # triggered this task (its ID matches the task_scope). This provides the
        # remote task agent with the initial task parameters (inputs). Once we
        # find it, we stop because anything older is outside this task's
        # lifetime.
        has_trigger_fc = False
        calls = event.get_function_calls()
        for fc in calls:
          if fc.id == task_scope:
            has_trigger_fc = True
            break
        if has_trigger_fc:
          events_to_process.append(event)
          broke_loop = True
          break
        # Ignore events belonging to other tasks (different isolation scopes)
        # or coordinator events outside the remote task agent execution.
        continue

      if self._is_remote_response(event):
        # stop on content generated by current a2a agent given it should already
        # be in remote session
        if event.custom_metadata:
          metadata = event.custom_metadata
          context_id = metadata.get(A2A_METADATA_PREFIX + "context_id")
        # Historical note: this behavior originally always applied, regardless
        # of whether the agent was stateful or stateless. However, only stateful
        # agents can be expected to have previous events in the remote session.
        # For backwards compatibility, we maintain this behavior when
        # _full_history_when_stateless is false (the default) or if the agent
        # is stateful (i.e. returned a context ID).
        if not self._full_history_when_stateless or context_id:
          broke_loop = True
          break
      events_to_process.append(event)

    # In task mode, an FC-delegation task must be bounded by a triggering
    # FunctionCall from the coordinator. If the history walk completes to the
    # root without finding the matching FC (and did not stop at a prior
    # stateful turn), the isolation scope is invalid (e.g. a workflow graph
    # node).
    if self.mode == "task" and task_scope and not broke_loop:
      raise ValueError(
          f"RemoteA2aAgent '{self.name}' in task mode could not find the"
          f" triggering FunctionCall for isolation scope '{task_scope}' in"
          " session history. Workflow path scopes are not supported."
      )

    # Collect all FC IDs emitted by this remote agent in the task scope.
    remote_fc_ids = set()
    if self.mode == "task":
      for event in ctx.session.events:
        if (
            not task_scope or event.isolation_scope == task_scope
        ) and event.author == self.name:
          calls = event.get_function_calls()
          for fc in calls:
            if fc.id is not None:
              remote_fc_ids.add(fc.id)

    for event in reversed(events_to_process):
      processed_event: Optional[Event] = event
      if _is_other_agent_reply(self.name, event):
        processed_event = _present_other_agent_message(event)

      if (
          not processed_event
          or not processed_event.content
          or not processed_event.content.parts
      ):
        continue

      for part in processed_event.content.parts:
        if (
            part.function_response is not None
            and _is_credential_function_response(part.function_response)
        ):
          # Never forward credential material (an AuthConfig envelope with
          # access tokens / client secrets) to the remote peer, even when
          # reconstructing the request from raw session history. This closes the
          # path where a dropped credential resume falls back to here and the
          # untouched function_response would otherwise be re-serialized.
          continue

        if (
            self.mode == "task"
            and task_scope
            and part.function_call
            and isinstance(part.function_call, genai_types.FunctionCall)
            and part.function_call.id is not None
            and part.function_call.id != task_scope
            and part.function_call.id not in remote_fc_ids
        ):
          # Skip sibling function calls from the coordinator intended for other tools/agents.
          continue

        if (
            self.mode == "task"
            and part.function_response
            and isinstance(part.function_response, genai_types.FunctionResponse)
            and part.function_response.id not in remote_fc_ids
        ):
          # Convert non-agent function response to text to prevent A2A server
          # validation errors.
          text_content = (
              f"Tool {part.function_response.name} returned:"
              f" {json.dumps(part.function_response.response)}"
          )
          converted_parts = [_compat.make_text_part(text_content)]
        else:
          raw_parts = self._genai_part_converter(part)
          if isinstance(raw_parts, list):
            converted_parts = raw_parts
          elif raw_parts is not None:
            converted_parts = [raw_parts]
          else:
            converted_parts = []

        if processed_event.author == "user":
          for a2a_part in converted_parts:
            meta = _compat.part_metadata(a2a_part) or {}
            meta["is_user_input"] = True
            _compat.set_part_metadata(a2a_part, meta)

        if converted_parts:
          message_parts.extend(converted_parts)
        else:
          logger.warning("Failed to convert part to A2A format: %s", part)

    return message_parts, context_id

  async def _handle_a2a_response(
      self,
      a2a_response: _compat.A2AClientEvent | A2AMessage,
      ctx: InvocationContext,
  ) -> Optional[Event]:
    """Handle A2A response and convert to Event.

    Args:
      a2a_response: The A2A response object
      ctx: The invocation context

    Returns:
      Event object representing the response, or None if no event should be
      emitted.
    """
    try:
      if isinstance(a2a_response, tuple):
        task, update = a2a_response
        if update is None:
          # This is the initial response for a streaming task or the complete
          # response for a non-streaming task, which is the full task state.
          # We process this to get the initial message.
          event = convert_a2a_task_to_event(
              task, self.name, ctx, self._a2a_part_converter
          )
          if not event:
            return None
          # for streaming task, we update the event with the task status.
          # We update the event as Thought updates.
          if (
              task
              and task.status
              and task.status.state
              in (
                  _compat.TS_SUBMITTED,
                  _compat.TS_WORKING,
              )
              and event.content is not None
              and event.content.parts
          ):
            for part in event.content.parts or []:
              part.thought = True
          _add_mock_function_call(event, task.status.state)
        elif isinstance(update, A2ATaskStatusUpdateEvent) and (
            _status_message := (
                _compat.normalize_message(update.status.message)
                if update.status
                else None
            )
        ):
          # This is a streaming task status update with a message.
          # ``normalize_message`` collapses the always-present empty proto
          # ``Message`` (1.x) to ``None`` so this branch only fires when a real
          # message is attached, matching 0.3.x where the field is ``None``.
          event = convert_a2a_message_to_event(
              _status_message, self.name, ctx, self._a2a_part_converter
          )
          if not event:
            return None
          if event.content is not None and update.status.state in (
              _compat.TS_SUBMITTED,
              _compat.TS_WORKING,
          ):
            for part in event.content.parts or []:
              part.thought = True
          _add_mock_function_call(event, update.status.state)
        elif isinstance(update, A2ATaskArtifactUpdateEvent):
          # This is a streaming task artifact update.
          # Convert only the parts carried by this update. Converting the
          # accumulated task here would re-emit earlier chunks of the same
          # artifact, duplicating already-streamed content.
          if not update.artifact.parts:
            return None
          event = convert_a2a_message_to_event(
              _compat.make_message(
                  message_id="",
                  role="agent",
                  parts=update.artifact.parts,
              ),
              self.name,
              ctx,
              self._a2a_part_converter,
          )
          if not event:
            return None
          event.partial = not update.last_chunk
        else:
          # This is a streaming update without a message (e.g. status change)
          # or a partial artifact update. We don't emit an event for these
          # for now.
          return None

        if not event:
          return None
        event.custom_metadata = event.custom_metadata or {}
        event.custom_metadata[A2A_METADATA_PREFIX + "task_id"] = task.id
        if task.context_id:
          event.custom_metadata[A2A_METADATA_PREFIX + "context_id"] = (
              task.context_id
          )

      # Otherwise, it's a regular A2AMessage for non-streaming responses.
      elif isinstance(a2a_response, A2AMessage):
        event = convert_a2a_message_to_event(
            a2a_response, self.name, ctx, self._a2a_part_converter
        )
        if not event:
          return None
        event.custom_metadata = event.custom_metadata or {}

        if a2a_response.context_id:
          event.custom_metadata[A2A_METADATA_PREFIX + "context_id"] = (
              a2a_response.context_id
          )
      else:
        event = Event(
            author=self.name,
            error_message="Unknown A2A response type",
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
        )
      # Filter out thought parts from user-facing response content.
      # Intermediate (submitted/working) events have all parts marked as
      # thought, so non_thought_parts will be empty and we preserve them.
      if event.content is not None and event.content.parts:
        non_thought_parts = [p for p in event.content.parts if not p.thought]
        if non_thought_parts:
          event.content.parts = non_thought_parts

      return event
    except A2AClientError as e:
      logger.error("Failed to handle A2A response: %s", e)
      return Event(
          author=self.name,
          error_message=f"Failed to process A2A response: {e}",
          invocation_id=ctx.invocation_id,
          branch=ctx.branch,
      )

  async def _handle_a2a_response_v2(
      self,
      a2a_response: _compat.A2AClientEvent | A2AMessage,
      ctx: InvocationContext,
  ) -> Optional[Event]:
    """Handle A2A response and convert to Event.

    Args:
      a2a_response: The A2A response object
      ctx: The invocation context

    Returns:
      Event object representing the response, or None if no event should be
      emitted.
    """
    try:
      if isinstance(a2a_response, tuple):
        task, update = a2a_response
        event = None
        if update is None:
          # This is the initial response for a streaming task or the complete
          # response for a non-streaming task.
          event = self._config.a2a_task_converter(
              task, self.name, ctx, self._config.a2a_part_converter
          )
        elif isinstance(update, A2ATaskStatusUpdateEvent):
          # This is a streaming task status update.
          event = self._config.a2a_status_update_converter(
              update, self.name, ctx, self._config.a2a_part_converter
          )
        elif isinstance(update, A2ATaskArtifactUpdateEvent):
          # This is a streaming task artifact update.
          event = self._config.a2a_artifact_update_converter(
              update, self.name, ctx, self._config.a2a_part_converter
          )
        if not event:
          return None
        event.custom_metadata = event.custom_metadata or {}
        event.custom_metadata[A2A_METADATA_PREFIX + "task_id"] = task.id
        if task.context_id:
          event.custom_metadata[A2A_METADATA_PREFIX + "context_id"] = (
              task.context_id
          )

      # Otherwise, it's a regular A2AMessage.
      elif isinstance(a2a_response, A2AMessage):
        event = self._config.a2a_message_converter(
            a2a_response, self.name, ctx, self._config.a2a_part_converter
        )
        if not event:
          return None
        event.custom_metadata = event.custom_metadata or {}

        if a2a_response.context_id:
          event.custom_metadata[A2A_METADATA_PREFIX + "context_id"] = (
              a2a_response.context_id
          )
      else:
        event = Event(
            author=self.name,
            error_message="Unknown A2A response type",
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
        )
      return event
    except A2AClientError as e:
      logger.error("Failed to handle A2A response: %s", e)
      return Event(
          author=self.name,
          error_message=f"Failed to process A2A response: {e}",
          invocation_id=ctx.invocation_id,
          branch=ctx.branch,
      )

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core implementation for async agent execution."""
    # Tracks whether task control should be released back to the parent
    # coordinator and any error output to emit on early termination.
    should_release_task_control = False
    task_error_message: Optional[str] = None
    a2a_request = None

    try:
      try:
        a2a_client = await self._ensure_resolved(ctx)
      except Exception as e:
        task_error_message = f"Failed to initialize remote A2A agent: {e}"
        should_release_task_control = True
        yield Event(
            author=self.name,
            error_message=task_error_message,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
        )
        return

      # Create A2A request for function response or regular message
      a2a_request = self._create_a2a_request_for_user_function_response(ctx)
      if not a2a_request:
        message_parts, context_id = self._construct_message_parts_from_session(
            ctx
        )

        if not message_parts:
          logger.warning(
              "No parts to send to remote A2A agent. Emitting empty event."
          )
          task_error_message = "No parts to send to remote A2A agent."
          should_release_task_control = True
          yield Event(
              author=self.name,
              content=genai_types.Content(),
              invocation_id=ctx.invocation_id,
              branch=ctx.branch,
          )
          return

        a2a_request = A2AMessage(
            message_id=platform_uuid.new_uuid(),
            parts=message_parts,
            role=_compat.ROLE_USER,
            context_id=context_id,
        )

      logger.debug(build_a2a_request_log(a2a_request))

      try:
        intercepted_request, parameters = (
            await execute_before_request_interceptors(
                self._config.request_interceptors, ctx, a2a_request
            )
        )

        if isinstance(intercepted_request, Event):
          task_error_message = "Request intercepted"
          should_release_task_control = True
          yield intercepted_request
          return
        a2a_request = intercepted_request

        # Backward compatibility
        if self._a2a_request_meta_provider:
          parameters.request_metadata = self._a2a_request_meta_provider(
              ctx, a2a_request
          )

        # TODO: Add support for requested_extension and
        # message_send_configuration once they are supported by the A2A client.
        # A single stateful normalizer per stream so incremental
        # status/artifact updates are aggregated into a running task (matching the
        # 0.3.x client behavior).
        normalize_stream_item = _compat.make_stream_normalizer()
        async with Aclosing(
            _compat.send_message(
                a2a_client,
                request=a2a_request,
                request_metadata=parameters.request_metadata,
                context=parameters.client_call_context,
            )
        ) as agen:
          async for raw_a2a_response in agen:
            a2a_response = normalize_stream_item(raw_a2a_response)
            logger.debug(build_a2a_response_log(a2a_response))

            task = None
            metadata = None
            if isinstance(a2a_response, tuple):
              task = a2a_response[0]
              if task:
                metadata = task.metadata
            else:
              metadata = a2a_response.metadata

            if metadata and _compat.metadata_get(
                metadata, _NEW_A2A_ADK_INTEGRATION_EXTENSION
            ):
              event = await self._handle_a2a_response_v2(a2a_response, ctx)
            else:
              event = await self._handle_a2a_response(a2a_response, ctx)
            if not event:
              continue

            event = await execute_after_request_interceptors(
                self._config.request_interceptors, ctx, a2a_response, event
            )
            if not event:
              continue

            # Add metadata about the request and response
            event.custom_metadata = event.custom_metadata or {}
            if a2a_request:
              event.custom_metadata[A2A_METADATA_PREFIX + "request"] = (
                  _compat.a2a_to_dict(a2a_request)
              )
            # If the response is a ClientEvent, record the task state; otherwise,
            # record the message object.
            if isinstance(a2a_response, tuple):
              event.custom_metadata[A2A_METADATA_PREFIX + "response"] = (
                  _compat.a2a_to_dict(a2a_response[0])
              )
            else:
              event.custom_metadata[A2A_METADATA_PREFIX + "response"] = (
                  _compat.a2a_to_dict(a2a_response)
              )

            if self.mode == "task" and is_finish_task_terminal_fr(event):
              args = _find_finish_task_args_from_history(
                  ctx.session, ctx.isolation_scope, completed_fr_event=event
              )
              if args is not None:
                wrapper_key = get_output_wrapper_key(self.output_schema)
                if wrapper_key and wrapper_key in args:
                  event.output = args[wrapper_key]
                else:
                  event.output = args
              else:
                logger.warning(
                    "Could not find finish_task arguments in session history"
                    " for isolation scope '%s'. Task output will not be set.",
                    ctx.isolation_scope,
                )
              # Yield the semantic output event so the parent runner can capture
              # the final task output and record the tool response in history.
              yield event
              # Mark the agent as finished so parent coordinator regains control.
              # Returning early terminates the stream reader, ignoring any legacy
              # duplicate FRs sent by the server at the end of the run.
              should_release_task_control = True
              return

            yield event

            if self.mode == "task" and task:
              if task.status and task.status.state in (
                  _compat.TS_FAILED,
                  _compat.TS_CANCELED,
              ):
                is_cancel = task.status.state == _compat.TS_CANCELED
                logger.warning(
                    "Remote task reported %s state. Yielding error event and "
                    "releasing control.",
                    "canceled" if is_cancel else "failure",
                )
                error_text = "Unknown error"
                if is_cancel:
                  error_text = "Task canceled"
                elif event:
                  error_text = (
                      _text_from_content(event.content) or "Unknown error"
                  )

                error_event, failure_event = _create_task_failure_events(
                    error_text=error_text,
                    ctx=ctx,
                    agent_name=self.name,
                    task_id=task.id,
                    a2a_request=a2a_request,
                )
                yield error_event
                yield failure_event
                should_release_task_control = True
                return

      except _compat.A2A_HTTP_ERRORS as e:
        error_message = f"A2A request failed: {e}"
        task_error_message = error_message
        should_release_task_control = True
        logger.error(error_message)
        status_code: object = getattr(e, "status_code", None)
        custom_metadata: dict[str, Any] = {
            A2A_METADATA_PREFIX + "error": error_message,
            A2A_METADATA_PREFIX + "status_code": str(status_code),
        }
        if a2a_request:
          custom_metadata[A2A_METADATA_PREFIX + "request"] = (
              _compat.a2a_to_dict(a2a_request)
          )
        yield Event(
            author=self.name,
            error_message=error_message,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
            custom_metadata=custom_metadata,
        )

      except Exception as e:
        error_message = f"A2A request failed: {e}"
        task_error_message = error_message
        should_release_task_control = True
        logger.error(error_message)
        custom_metadata = {
            A2A_METADATA_PREFIX + "error": error_message,
        }
        if a2a_request:
          custom_metadata[A2A_METADATA_PREFIX + "request"] = (
              _compat.a2a_to_dict(a2a_request)
          )
        yield Event(
            author=self.name,
            error_message=error_message,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
            custom_metadata=custom_metadata,
        )

    finally:
      if self.mode == "task" and should_release_task_control:
        if task_error_message is not None:
          yield _create_finish_task_event(
              ctx=ctx,
              agent_name=self.name,
              error_message=task_error_message,
              is_error=True,
          )
        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)

  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core implementation for live agent execution (not implemented)."""
    raise NotImplementedError(
        f"_run_live_impl for {type(self)} via A2A is not implemented."
    )
    # This makes the function into an async generator but the yield is still unreachable
    yield

  # Task states that represent in-progress or input-awaiting work.
  # Events stamped with one of these states carry intermediate or
  # waiting-for-input content, never the final answer, so they must
  # not be promoted to the workflow node's output.
  _NON_FINAL_TASK_STATES = frozenset(
      {"submitted", "working", "input-required", "auth-required", "unknown"}
  )

  async def _run_impl(
      self,
      *,
      ctx: Any,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Runs the agent as a workflow node.

    Promotes textual response content to ``event.output`` so the
    workflow scheduler propagates it downstream. Without this, a
    ``JoinNode`` that aggregates parallel ``RemoteA2aAgent`` predecessors
    sees ``None`` for each predecessor because ``BaseAgent._run_impl``
    never sets ``event.output`` and ``RemoteA2aAgent`` carries its
    response only in ``event.content``.

    A node may produce at most one output (``Context.output`` raises
    ``ValueError`` on a second assignment), so promotion is gated to
    the first terminal A2A event of the run. Non-final task states and
    later events are passed through untouched.
    """
    promoted = False
    async for event in super()._run_impl(ctx=ctx, node_input=node_input):
      if (
          self.mode != "task"
          and not promoted
          and self._promote_response_to_output(event, ctx.node_path)
      ):
        promoted = True
      yield event

  def _promote_response_to_output(self, event: Event, node_path: str) -> bool:
    """Sets ``event.output`` from non-thought text parts, if eligible.

    Returns True iff this call assigned ``event.output``. Skips:

    * partial events and events whose ``event.output`` is already set;
    * events that do not belong to this node. ``BaseAgent._run_impl``
      stamps ``event.node_info.path`` with this node's path only for the
      agent's own events, so matching on the path uniquely identifies the
      node in the workflow hierarchy even when agent names collide across
      branches;
    * events whose content carries only thoughts, function calls, or
      function responses (e.g. ``input_required`` mock function calls);
    * events whose A2A task state is non-final (``submitted``,
      ``working``, ``input-required``, ``auth-required``, ``unknown``).
      Streaming converters do not always mark ``working`` text as
      ``thought=True``, so the task-state check guards against
      promoting an intermediate streaming chunk and then raising on the
      true final event.
    """
    if event.partial or event.output is not None:
      return False
    if event.node_info.path != node_path:
      return False
    if not event.content or not event.content.parts:
      return False

    response_meta = (event.custom_metadata or {}).get(
        A2A_METADATA_PREFIX + "response"
    )
    if isinstance(response_meta, dict):
      status = response_meta.get("status")
      if (
          isinstance(status, dict)
          and status.get("state") in self._NON_FINAL_TASK_STATES
      ):
        return False

    text_chunks = [
        part.text
        for part in event.content.parts
        if part.text
        and not part.thought
        and not part.function_call
        and not part.function_response
    ]
    if not text_chunks:
      return False
    event.output = "".join(text_chunks)
    event.node_info.message_as_output = True
    return True

  async def cleanup(self) -> None:
    """Clean up resources, especially the HTTP client if owned by this agent."""
    if self._httpx_client_needs_cleanup and self._httpx_client:
      try:
        await self._httpx_client.aclose()
        logger.debug("Closed HTTP client for agent %s", self.name)
      except Exception as e:
        logger.warning(
            "Failed to close HTTP client for agent %s: %s",
            self.name,
            e,
        )
      finally:
        self._httpx_client = None
