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

"""Toolset that dispatches runtime agents with persistent child sessions.

Addresses the gap where static ``sub_agents`` must be known at construct time
and ``AgentTool`` creates a fresh session per call (no follow-ups). See
https://github.com/google/adk-python/issues/4759.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
import re
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Union
import uuid

from google.genai import types
from typing_extensions import override

from ...agents.llm_agent import LlmAgent
from ...agents.readonly_context import ReadonlyContext
from ...memory.in_memory_memory_service import InMemoryMemoryService
from ...runners import Runner
from ...sessions.in_memory_session_service import InMemorySessionService
from ...utils.context_utils import Aclosing
from .._forwarding_artifact_service import ForwardingArtifactService
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..function_tool import FunctionTool
from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)

_STATE_KEY = '_adk_agent_dispatcher'
_AGENT_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

ToolFactory = Union[BaseTool, Callable[..., Any]]
AgentMode = Literal['chat', 'task', 'single_turn']


def _part_to_text(part: types.Part) -> str:
  """Returns user-visible text from a Part."""
  if part.text:
    return part.text
  if part.code_execution_result and part.code_execution_result.output:
    return part.code_execution_result.output.rstrip('\n')
  if part.executable_code and part.executable_code.code:
    return part.executable_code.code
  return ''


def _normalize_agent_name(name: str) -> str:
  """Returns a valid agent name or raises ValueError."""
  cleaned = name.strip().replace(' ', '_').replace('-', '_')
  if not _AGENT_NAME_RE.match(cleaned):
    raise ValueError(
        'Agent name must be a valid identifier '
        f'(letters, digits, underscore); got {name!r}.'
    )
  return cleaned


@dataclass
class _DispatchEntry:
  """Process-local handle for a dispatched agent session."""

  dispatch_id: str
  agent_name: str
  session_id: str
  user_id: str
  app_name: str
  runner: Runner
  status: str = 'completed'
  result: str = ''
  metadata: dict[str, Any] = field(default_factory=dict)


class AgentDispatcherToolset(BaseToolset):
  """Exposes tools to spawn and follow up with runtime agents.

  Sync-first MVP: ``dispatch_agent`` awaits the child run to completion and
  keeps the child session alive for ``message_agent`` follow-ups.
  """

  def __init__(
      self,
      *,
      model: Union[str, Any],
      tool_allowlist: Optional[dict[str, ToolFactory]] = None,
      include_plugins: bool = True,
      default_mode: AgentMode = 'chat',
  ):
    """Initializes the dispatcher toolset.

    Args:
      model: Model used for dispatched agents (string id or BaseLlm).
      tool_allowlist: Optional map of tool name -> tool/callable that
        ``dispatch_agent`` may attach by name. Unknown names are rejected.
      include_plugins: Whether child Runners inherit parent plugins.
      default_mode: Default ``LlmAgent.mode`` for dispatched agents.
    """
    super().__init__()
    self._model = model
    self._tool_allowlist = dict(tool_allowlist or {})
    self._include_plugins = include_plugins
    self._default_mode = default_mode
    self._entries: dict[str, _DispatchEntry] = {}

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    del readonly_context  # Tools are always available.
    return [
        FunctionTool(self.dispatch_agent),
        FunctionTool(self.get_agent_result),
        FunctionTool(self.message_agent),
    ]

  @override
  async def close(self) -> None:
    """Closes child runners held by this toolset."""
    for entry in list(self._entries.values()):
      try:
        await entry.runner.close()
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            'Failed to close runner for dispatch_id=%s', entry.dispatch_id
        )
    self._entries.clear()

  async def dispatch_agent(
      self,
      name: str,
      instruction: str,
      user_message: str,
      tool_context: ToolContext,
      tool_names: Optional[list[str]] = None,
      mode: Optional[str] = None,
  ) -> dict[str, Any]:
    """Dispatch a new agent with a persistent session and run it once.

    Args:
      name: Agent name (identifier). Must be unique enough for your workflow.
      instruction: System instruction for the dispatched agent.
      user_message: Initial user message for the child agent.
      tool_context: ADK tool context (injected).
      tool_names: Optional allowlisted tool names to attach.
      mode: Optional agent mode: ``chat``, ``task``, or ``single_turn``.

    Returns:
      Dict with ``dispatch_id``, ``status``, ``result``, and ``agent_name``.
      On validation errors, returns ``status='failed'`` with an error message
      in ``result`` (no live entry is stored).
    """
    try:
      agent_name = _normalize_agent_name(name)
      agent_mode = self._resolve_mode(mode)
      tools = self._resolve_tools(tool_names)
    except ValueError as e:
      return {
          'dispatch_id': '',
          'agent_name': name,
          'status': 'failed',
          'result': str(e),
      }

    agent = LlmAgent(
        name=agent_name,
        model=self._model,
        instruction=instruction,
        mode=agent_mode,
        tools=tools,
    )

    invocation_context = tool_context._invocation_context
    parent_app_name = (
        invocation_context.app_name if invocation_context else None
    )
    child_app_name = parent_app_name or agent_name
    plugins = (
        invocation_context.plugin_manager.plugins
        if self._include_plugins and invocation_context
        else None
    )
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=child_app_name,
        agent=agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=session_service,
        memory_service=InMemoryMemoryService(),
        credential_service=(
            invocation_context.credential_service
            if invocation_context
            else None
        ),
        plugins=plugins,
    )
    if self._include_plugins and plugins is not None:
      runner.plugin_manager.set_skip_closing_plugins(True)

    state_dict = {
        k: v
        for k, v in tool_context.state.to_dict().items()
        if not k.startswith('_adk')
    }
    user_id = invocation_context.user_id if invocation_context else 'dispatcher'
    session = await session_service.create_session(
        app_name=child_app_name,
        user_id=user_id,
        state=state_dict,
    )

    dispatch_id = uuid.uuid4().hex
    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=user_message)],
    )
    result_text, error_message = await self._run_child(
        runner=runner,
        user_id=session.user_id,
        session_id=session.id,
        content=content,
        tool_context=tool_context,
    )

    status = 'failed' if error_message and not result_text else 'completed'
    result = result_text or error_message or ''
    entry = _DispatchEntry(
        dispatch_id=dispatch_id,
        agent_name=agent_name,
        session_id=session.id,
        user_id=session.user_id,
        app_name=child_app_name,
        runner=runner,
        status=status,
        result=result,
        metadata={
            'mode': agent_mode,
            'tool_names': list(tool_names or []),
        },
    )
    self._entries[dispatch_id] = entry
    self._write_state(tool_context, entry)
    return self._public_payload(entry)

  async def get_agent_result(
      self,
      dispatch_id: str,
      tool_context: ToolContext,
  ) -> dict[str, Any]:
    """Return the latest status/result for a previously dispatched agent.

    Args:
      dispatch_id: Id returned by ``dispatch_agent``.
      tool_context: ADK tool context (injected).

    Returns:
      Dict with ``dispatch_id``, ``status``, ``result``, and ``agent_name``.
    """
    entry = self._require_entry(dispatch_id, tool_context)
    return self._public_payload(entry)

  async def message_agent(
      self,
      dispatch_id: str,
      user_message: str,
      tool_context: ToolContext,
  ) -> dict[str, Any]:
    """Send a follow-up message to a dispatched agent on the same session.

    Args:
      dispatch_id: Id returned by ``dispatch_agent``.
      user_message: Follow-up user message.
      tool_context: ADK tool context (injected).

    Returns:
      Updated dict with ``dispatch_id``, ``status``, ``result``, ``agent_name``.
    """
    entry = self._require_entry(dispatch_id, tool_context)
    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=user_message)],
    )
    result_text, error_message = await self._run_child(
        runner=entry.runner,
        user_id=entry.user_id,
        session_id=entry.session_id,
        content=content,
        tool_context=tool_context,
    )
    entry.status = (
        'failed' if error_message and not result_text else 'completed'
    )
    entry.result = result_text or error_message or ''
    self._write_state(tool_context, entry)
    return self._public_payload(entry)

  def _resolve_mode(self, mode: Optional[str]) -> AgentMode:
    resolved = mode or self._default_mode
    if resolved not in ('chat', 'task', 'single_turn'):
      raise ValueError(
          f'Unsupported agent mode {resolved!r}; '
          "expected 'chat', 'task', or 'single_turn'."
      )
    return resolved  # type: ignore[return-value]

  def _resolve_tools(
      self, tool_names: Optional[list[str]]
  ) -> list[ToolFactory]:
    if not tool_names:
      return []
    missing = [n for n in tool_names if n not in self._tool_allowlist]
    if missing:
      allowed = sorted(self._tool_allowlist)
      raise ValueError(
          f'Unknown tool_names {missing}; allowlist has {allowed}.'
      )
    return [self._tool_allowlist[n] for n in tool_names]

  def _require_entry(
      self, dispatch_id: str, tool_context: ToolContext
  ) -> _DispatchEntry:
    entry = self._entries.get(dispatch_id)
    if entry is not None:
      return entry
    # Recover metadata-only view is not enough to run follow-ups; require live
    # entry from this toolset instance.
    state_registry = tool_context.state.get(_STATE_KEY) or {}
    if dispatch_id in state_registry:
      raise ValueError(
          f'dispatch_id {dispatch_id!r} exists in session state but the live '
          'runner is not available on this AgentDispatcherToolset instance. '
          'Reuse the same toolset instance for follow-ups.'
      )
    raise ValueError(f'Unknown dispatch_id: {dispatch_id!r}')

  async def _run_child(
      self,
      *,
      runner: Runner,
      user_id: str,
      session_id: str,
      content: types.Content,
      tool_context: ToolContext,
  ) -> tuple[str, Optional[str]]:
    last_content = None
    last_error_message = None
    async with Aclosing(
        runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        )
    ) as agen:
      async for event in agen:
        if event.actions.state_delta:
          # Do not forward child `_adk*` keys into the parent dispatcher map.
          for key, value in event.actions.state_delta.items():
            if key.startswith('_adk'):
              continue
            tool_context.state[key] = value
        if event.error_message:
          last_error_message = event.error_message
        if event.content:
          last_content = event.content

    if last_content is None or last_content.parts is None:
      return '', last_error_message
    parts_text = (_part_to_text(p) for p in last_content.parts if not p.thought)
    merged_text = '\n'.join(t for t in parts_text if t)
    return merged_text, last_error_message

  def _write_state(
      self, tool_context: ToolContext, entry: _DispatchEntry
  ) -> None:
    registry = dict(tool_context.state.get(_STATE_KEY) or {})
    registry[entry.dispatch_id] = {
        'dispatch_id': entry.dispatch_id,
        'agent_name': entry.agent_name,
        'session_id': entry.session_id,
        'status': entry.status,
        'result': entry.result,
        'metadata': entry.metadata,
    }
    tool_context.state[_STATE_KEY] = registry

  def _public_payload(self, entry: _DispatchEntry) -> dict[str, Any]:
    return {
        'dispatch_id': entry.dispatch_id,
        'agent_name': entry.agent_name,
        'status': entry.status,
        'result': entry.result,
    }
