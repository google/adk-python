# Copyright 2025 Google LLC
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

import json
import logging
from pathlib import Path
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import Union
from urllib.parse import urlparse
import uuid

try:
  from a2a.client import A2AClient
  from a2a.client.card_resolver import A2ACardResolver
  from a2a.types import AgentCard
  from a2a.types import Message as A2AMessage
  from a2a.types import MessageSendParams as A2AMessageSendParams
  from a2a.types import Part as A2APart
  from a2a.types import Role
  from a2a.types import SendMessageRequest
  from a2a.types import SendMessageSuccessResponse
  from a2a.types import Task as A2ATask
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "A2A requires Python 3.10 or above. Please upgrade your Python version."
    ) from e
  else:
    raise e

try:
  from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
except ImportError:
  # Fallback for older versions of a2a-sdk.
  AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent.json"

from google.genai import types as genai_types
import httpx

from ..a2a.converters.event_converter import convert_a2a_message_to_event
from ..a2a.converters.event_converter import convert_a2a_task_to_event
from ..a2a.converters.event_converter import convert_event_to_a2a_message
from ..a2a.converters.part_converter import convert_genai_part_to_a2a_part
from ..a2a.experimental import a2a_experimental
from ..a2a.logs.log_utils import build_a2a_request_log
from ..a2a.logs.log_utils import build_a2a_response_log
from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from ..flows.llm_flows.contents import _convert_foreign_event
from ..flows.llm_flows.contents import _is_other_agent_reply
from ..flows.llm_flows.functions import find_matching_function_call
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

logger = logging.getLogger("google_adk." + __name__)


@a2a_experimental
class AgentCardResolutionError(Exception):
  """Raised when agent card resolution fails."""

  pass


@a2a_experimental
class A2AClientError(Exception):
  """Raised when A2A client operations fail."""

  pass


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

  def __init__(
      self,
      name: str,
      agent_card: Union[AgentCard, str],
      description: str = "",
      httpx_client: Optional[httpx.AsyncClient] = None,
      timeout: float = DEFAULT_TIMEOUT,
      **kwargs: Any,
  ) -> None:
    """Initialize RemoteA2aAgent.

    Args:
      name: Agent name (must be unique identifier)
      agent_card: AgentCard object, URL string, or file path string
      description: Agent description (auto-populated from card if empty)
      httpx_client: Optional shared HTTP client (will create own if not provided)
      timeout: HTTP timeout in seconds
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
    self._rpc_url: Optional[str] = None
    self._a2a_client: Optional[A2AClient] = None
    self._httpx_client = httpx_client
    self._httpx_client_needs_cleanup = httpx_client is None
    self._timeout = timeout
    self._is_resolved = False

    # Validate and store agent card reference
    if isinstance(agent_card, AgentCard):
      self._agent_card = agent_card
    elif isinstance(agent_card, str):
      if not agent_card.strip():
        raise ValueError("agent_card string cannot be empty")
      self._agent_card_source = agent_card.strip()
    else:
      raise TypeError(
          "agent_card must be AgentCard, URL string, or file path string, "
          f"got {type(agent_card)}"
      )

  async def _ensure_httpx_client(self) -> httpx.AsyncClient:
    """Ensure HTTP client is available and properly configured."""
    if not self._httpx_client:
      self._httpx_client = httpx.AsyncClient(
          timeout=httpx.Timeout(timeout=self._timeout)
      )
      self._httpx_client_needs_cleanup = True
    return self._httpx_client

  async def _resolve_agent_card_from_url(self, url: str) -> AgentCard:
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
      return await resolver.get_agent_card(
          relative_card_path=relative_card_path
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
        return AgentCard(**agent_json_data)
    except json.JSONDecodeError as e:
      raise AgentCardResolutionError(
          f"Invalid JSON in agent card file {file_path}: {e}"
      ) from e
    except Exception as e:
      raise AgentCardResolutionError(
          f"Failed to resolve AgentCard from file {file_path}: {e}"
      ) from e

  async def _resolve_agent_card(self) -> AgentCard:
    """Resolve agent card from source."""

    # Determine if source is URL or file path
    if self._agent_card_source.startswith(("http://", "https://")):
      return await self._resolve_agent_card_from_url(self._agent_card_source)
    else:
      return await self._resolve_agent_card_from_file(self._agent_card_source)

  async def _validate_agent_card(self, agent_card: AgentCard) -> None:
    """Validate resolved agent card."""
    if not agent_card.url:
      raise AgentCardResolutionError(
          "Agent card must have a valid URL for RPC communication"
      )

    # Additional validation can be added here
    try:
      parsed_url = urlparse(str(agent_card.url))
      if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Invalid RPC URL format")
    except Exception as e:
      raise AgentCardResolutionError(
          f"Invalid RPC URL in agent card: {agent_card.url}, error: {e}"
      ) from e

  async def _ensure_resolved(self) -> None:
    """Ensures agent card is resolved, RPC URL is determined, and A2A client is initialized."""
    if self._is_resolved:
      return

    try:
      # Resolve agent card if needed
      if not self._agent_card:
        self._agent_card = await self._resolve_agent_card()

      # Validate agent card
      await self._validate_agent_card(self._agent_card)

      # Set RPC URL
      self._rpc_url = str(self._agent_card.url)

      # Update description if empty
      if not self.description and self._agent_card.description:
        self.description = self._agent_card.description

      # Initialize A2A client
      if not self._a2a_client:
        httpx_client = await self._ensure_httpx_client()
        self._a2a_client = A2AClient(
            httpx_client=httpx_client,
            agent_card=self._agent_card,
            url=self._rpc_url,
        )

      self._is_resolved = True
      logger.info("Successfully resolved remote A2A agent: %s", self.name)

    except Exception as e:
      logger.error("Failed to resolve remote A2A agent %s: %s", self.name, e)
      raise AgentCardResolutionError(
          f"Failed to initialize remote A2A agent {self.name}: {e}"
      ) from e

  def _create_a2a_request_for_user_function_response(
      self, ctx: InvocationContext
  ) -> Optional[SendMessageRequest]:
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

    a2a_message = convert_event_to_a2a_message(
        ctx.session.events[-1], ctx, Role.user
    )
    if function_call_event.custom_metadata:
      a2a_message.task_id = (
          function_call_event.custom_metadata.get(
              A2A_METADATA_PREFIX + "task_id"
          )
          if function_call_event.custom_metadata
          else None
      )
      a2a_message.context_id = (
          function_call_event.custom_metadata.get(
              A2A_METADATA_PREFIX + "context_id"
          )
          if function_call_event.custom_metadata
          else None
      )

    return SendMessageRequest(
        id=str(uuid.uuid4()),
        params=A2AMessageSendParams(
            message=a2a_message,
        ),
    )

  def _construct_message_parts_from_session(
      self, ctx: InvocationContext
  ) -> tuple[list[A2APart], dict[str, Any], str]:
    """Construct A2A message parts from session events.

    Args:
      ctx: The invocation context

    Returns:
      List of A2A parts extracted from session events, context ID
    """
    message_parts: list[A2APart] = []
    context_id = None
    for event in reversed(ctx.session.events):
      if _is_other_agent_reply(self.name, event):
        event = _convert_foreign_event(event)
      elif event.author == self.name:
        # stop on content generated by current a2a agent given it should already
        # be in remote session
        if event.custom_metadata:
          context_id = (
              event.custom_metadata.get(A2A_METADATA_PREFIX + "context_id")
              if event.custom_metadata
              else None
          )
        break

      if not event.content or not event.content.parts:
        continue

      for part in event.content.parts:

        converted_part = convert_genai_part_to_a2a_part(part)
        if converted_part:
          message_parts.append(converted_part)
        else:
          logger.warning("Failed to convert part to A2A format: %s", part)

    return message_parts[::-1], context_id

  async def _handle_a2a_response(
      self, a2a_response: Any, ctx: InvocationContext
  ) -> Event:
    """Handle A2A response and convert to Event.

    Args:
      a2a_response: The A2A response object
      ctx: The invocation context

    Returns:
      Event object representing the response
    """
    try:
      if isinstance(a2a_response.root, SendMessageSuccessResponse):
        if a2a_response.root.result:
          if isinstance(a2a_response.root.result, A2ATask):
            event = convert_a2a_task_to_event(
                a2a_response.root.result, self.name, ctx
            )
            event.custom_metadata = event.custom_metadata or {}
            event.custom_metadata[A2A_METADATA_PREFIX + "task_id"] = (
                a2a_response.root.result.id
            )

          else:
            event = convert_a2a_message_to_event(
                a2a_response.root.result, self.name, ctx
            )
            event.custom_metadata = event.custom_metadata or {}
            if a2a_response.root.result.task_id:
              event.custom_metadata[A2A_METADATA_PREFIX + "task_id"] = (
                  a2a_response.root.result.task_id
              )

          if a2a_response.root.result.context_id:
            event.custom_metadata[A2A_METADATA_PREFIX + "context_id"] = (
                a2a_response.root.result.context_id
            )

        else:
          logger.warning("A2A response has no result: %s", a2a_response.root)
          event = Event(
              author=self.name,
              invocation_id=ctx.invocation_id,
              branch=ctx.branch,
          )
      else:
        # Handle error response
        error_response = a2a_response.root
        logger.error(
            "A2A request failed with error: %s, data: %s",
            error_response.error.message,
            error_response.error.data,
        )
        event = Event(
            author=self.name,
            error_message=error_response.error.message,
            error_code=str(error_response.error.code),
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
        )

      return event
    except Exception as e:
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
    try:
      await self._ensure_resolved()
    except Exception as e:
      yield Event(
          author=self.name,
          error_message=f"Failed to initialize remote A2A agent: {e}",
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
        yield Event(
            author=self.name,
            content=genai_types.Content(),
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
        )
        return

      a2a_request = SendMessageRequest(
          id=str(uuid.uuid4()),
          params=A2AMessageSendParams(
              message=A2AMessage(
                  message_id=str(uuid.uuid4()),
                  parts=message_parts,
                  role="user",
                  context_id=context_id,
              )
          ),
      )

    logger.debug(build_a2a_request_log(a2a_request))

    try:
      a2a_response = await self._a2a_client.send_message(request=a2a_request)
      logger.debug(build_a2a_response_log(a2a_response))

      event = await self._handle_a2a_response(a2a_response, ctx)

      # Add metadata about the request and response
      event.custom_metadata = event.custom_metadata or {}
      event.custom_metadata[A2A_METADATA_PREFIX + "request"] = (
          a2a_request.model_dump(exclude_none=True, by_alias=True)
      )
      event.custom_metadata[A2A_METADATA_PREFIX + "response"] = (
          a2a_response.root.model_dump(exclude_none=True, by_alias=True)
      )

      yield event

    except Exception as e:
      error_message = f"A2A request failed: {e}"
      logger.error(error_message)

      yield Event(
          author=self.name,
          error_message=error_message,
          invocation_id=ctx.invocation_id,
          branch=ctx.branch,
          custom_metadata={
              A2A_METADATA_PREFIX
              + "request": a2a_request.model_dump(
                  exclude_none=True, by_alias=True
              ),
              A2A_METADATA_PREFIX + "error": error_message,
          },
      )

  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core implementation for live agent execution (not implemented)."""
    raise NotImplementedError(
        f"_run_live_impl for {type(self)} via A2A is not implemented."
    )
    # This makes the function an async generator but the yield is still unreachable
    yield

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
