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

Supports background / parallel dispatch, completion callbacks, allowlisted
skills, and durable sessions via a shared ``BaseSessionService`` so follow-ups
survive toolset rebuilds when session state is restored.

See https://github.com/google/adk-python/issues/4759.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import inspect
import logging
import re
from typing import Any
from typing import Literal
from typing import Optional
from typing import Union
import uuid

from google.genai import types
from typing_extensions import override

from ...agents.llm_agent import LlmAgent
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...memory.in_memory_memory_service import InMemoryMemoryService
from ...runners import Runner
from ...sessions.base_session_service import BaseSessionService
from ...sessions.in_memory_session_service import InMemorySessionService
from ...skills.models import Skill
from ...skills.skill_registry import SkillRegistry
from ...utils.context_utils import Aclosing
from .._forwarding_artifact_service import ForwardingArtifactService
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..function_tool import FunctionTool
from ..skill_toolset import SkillToolset
from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)

_STATE_KEY = '_adk_agent_dispatcher'
_AGENT_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

ToolFactory = Union[BaseTool, Callable[..., Any]]
AgentMode = Literal['chat', 'task', 'single_turn']
CompletionCallback = Callable[[dict[str, Any]], Union[None, Awaitable[None]]]


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
  """Handle for a dispatched agent session."""

  dispatch_id: str
  agent_name: str
  session_id: str
  user_id: str
  app_name: str
  instruction: str
  mode: AgentMode
  model_spec: Any
  tool_names: list[str] = field(default_factory=list)
  skill_names: list[str] = field(default_factory=list)
  runner: Optional[Runner] = None
  task: Optional[asyncio.Task[None]] = None
  status: str = 'running'
  result: str = ''
  parent_app_name: str = ''
  parent_user_id: str = ''
  parent_session_id: str = ''
  session_service: Optional[BaseSessionService] = None
  done_event: asyncio.Event = field(default_factory=asyncio.Event)

  def to_state_dict(self) -> dict[str, Any]:
    model_spec = self.model_spec if isinstance(self.model_spec, str) else None
    return {
        'dispatch_id': self.dispatch_id,
        'agent_name': self.agent_name,
        'session_id': self.session_id,
        'user_id': self.user_id,
        'app_name': self.app_name,
        'instruction': self.instruction,
        'mode': self.mode,
        'model': model_spec,
        'tool_names': list(self.tool_names),
        'skill_names': list(self.skill_names),
        'status': self.status,
        'result': self.result,
        'parent_app_name': self.parent_app_name,
        'parent_user_id': self.parent_user_id,
        'parent_session_id': self.parent_session_id,
    }

  @classmethod
  def from_state_dict(cls, data: dict[str, Any]) -> _DispatchEntry:
    return cls(
        dispatch_id=data['dispatch_id'],
        agent_name=data['agent_name'],
        session_id=data['session_id'],
        user_id=data['user_id'],
        app_name=data['app_name'],
        instruction=data.get('instruction', ''),
        mode=data.get('mode', 'chat'),  # type: ignore[arg-type]
        model_spec=data.get('model'),
        tool_names=list(data.get('tool_names') or []),
        skill_names=list(data.get('skill_names') or []),
        status=data.get('status', 'completed'),
        result=data.get('result', ''),
        parent_app_name=data.get('parent_app_name', ''),
        parent_user_id=data.get('parent_user_id', ''),
        parent_session_id=data.get('parent_session_id', ''),
    )


class AgentDispatcherToolset(BaseToolset):
  """Exposes tools to spawn, poll, await, and follow up with runtime agents.

  By default ``dispatch_agent`` runs children in the background so the
  orchestrator can continue (including parallel multi-dispatch). Use
  ``wait=True`` or ``await_agent`` to block on completion. Completion
  callbacks fire when a background (or waited) run finishes.
  """

  def __init__(
      self,
      *,
      model: Union[str, Any],
      tool_allowlist: Optional[dict[str, ToolFactory]] = None,
      skill_allowlist: Optional[dict[str, Skill]] = None,
      skill_registry: Optional[SkillRegistry] = None,
      session_service: Optional[BaseSessionService] = None,
      on_complete: Optional[CompletionCallback] = None,
      include_plugins: bool = True,
      default_mode: AgentMode = 'chat',
      default_wait: bool = False,
  ):
    """Initializes the dispatcher toolset.

    Args:
      model: Default model for dispatched agents (string id or BaseLlm).
      tool_allowlist: Optional map of tool name -> tool/callable attachable by
        ``tool_names``.
      skill_allowlist: Optional map of skill name -> ``Skill`` attachable by
        ``skill_names``.
      skill_registry: Optional registry for resolving skill names not in the
        allowlist (still named lookup — not free-form path execution).
      session_service: Optional shared session service for child sessions. When
        omitted, uses the parent invocation's session service, falling back to
        a process-local ``InMemorySessionService``.
      on_complete: Optional sync/async callback invoked with the public payload
        when a dispatch finishes.
      include_plugins: Whether child Runners inherit parent plugins.
      default_mode: Default ``LlmAgent.mode`` for dispatched agents.
      default_wait: Default for ``dispatch_agent(wait=...)``. ``False`` means
        background so the orchestrator can continue.
    """
    super().__init__()
    self._model = model
    self._tool_allowlist = dict(tool_allowlist or {})
    self._skill_allowlist = dict(skill_allowlist or {})
    self._skill_registry = skill_registry
    self._session_service_override = session_service
    self._fallback_session_service = InMemorySessionService()
    self._on_complete = on_complete
    self._include_plugins = include_plugins
    self._default_mode = default_mode
    self._default_wait = default_wait
    self._entries: dict[str, _DispatchEntry] = {}
    self._lock = asyncio.Lock()

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    del readonly_context
    return [
        FunctionTool(self.dispatch_agent),
        FunctionTool(self.get_agent_result),
        FunctionTool(self.message_agent),
        FunctionTool(self.await_agent),
    ]

  @override
  async def close(self) -> None:
    """Cancels background tasks and closes child runners."""
    for entry in list(self._entries.values()):
      if entry.task and not entry.task.done():
        entry.task.cancel()
        try:
          await entry.task
        except asyncio.CancelledError:
          pass
      if entry.runner is not None:
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
      skill_names: Optional[list[str]] = None,
      mode: Optional[str] = None,
      wait: Optional[bool] = None,
  ) -> dict[str, Any]:
    """Dispatch a runtime agent on a persistent child session.

    Args:
      name: Agent name (identifier).
      instruction: System instruction for the dispatched agent.
      user_message: Initial user message for the child agent.
      tool_context: ADK tool context (injected).
      tool_names: Optional allowlisted tool names to attach.
      skill_names: Optional skill names from allowlist/registry to attach.
      mode: Optional agent mode: ``chat``, ``task``, or ``single_turn``.
      wait: If true, await completion before returning. If false/omitted,
        runs in the background (default) so the orchestrator can continue and
        multiple dispatches can run in parallel.

    Returns:
      Dict with ``dispatch_id``, ``status``, ``result``, ``agent_name``.
      Background dispatches return ``status='running'`` immediately.
    """
    should_wait = self._default_wait if wait is None else wait
    try:
      agent_name = _normalize_agent_name(name)
      agent_mode = self._resolve_mode(mode)
      tools = self._resolve_tools(tool_names)
      skills = await self._resolve_skills(skill_names)
    except ValueError as e:
      return {
          'dispatch_id': '',
          'agent_name': name,
          'status': 'failed',
          'result': str(e),
      }

    agent_tools: list[Any] = list(tools)
    if skills:
      agent_tools.append(SkillToolset(skills=skills))

    agent = LlmAgent(
        name=agent_name,
        model=self._model,
        instruction=instruction,
        mode=agent_mode,
        tools=agent_tools,
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
    session_service = self._resolve_session_service(tool_context)
    runner = Runner(
        app_name=child_app_name,
        agent=agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=session_service,
        memory_service=(
            invocation_context.memory_service
            if invocation_context and invocation_context.memory_service
            else InMemoryMemoryService()
        ),
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

    parent_session = invocation_context.session if invocation_context else None
    dispatch_id = uuid.uuid4().hex
    entry = _DispatchEntry(
        dispatch_id=dispatch_id,
        agent_name=agent_name,
        session_id=session.id,
        user_id=session.user_id,
        app_name=child_app_name,
        instruction=instruction,
        mode=agent_mode,
        model_spec=self._model if isinstance(self._model, str) else None,
        tool_names=list(tool_names or []),
        skill_names=list(skill_names or []),
        runner=runner,
        status='running',
        parent_app_name=parent_app_name or '',
        parent_user_id=user_id,
        parent_session_id=parent_session.id if parent_session else '',
        session_service=session_service,
    )
    self._entries[dispatch_id] = entry
    await self._write_state(tool_context, entry, persist_parent=True)

    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=user_message)],
    )
    if should_wait:
      await self._execute_and_finalize(
          entry=entry,
          content=content,
          tool_context=tool_context,
          forward_child_state=True,
      )
      return self._public_payload(entry)

    entry.task = asyncio.create_task(
        self._background_run(entry=entry, content=content)
    )
    return self._public_payload(entry)

  async def get_agent_result(
      self,
      dispatch_id: str,
      tool_context: ToolContext,
  ) -> dict[str, Any]:
    """Return the latest status/result for a previously dispatched agent."""
    entry = await self._require_entry(dispatch_id, tool_context)
    return self._public_payload(entry)

  async def message_agent(
      self,
      dispatch_id: str,
      user_message: str,
      tool_context: ToolContext,
      wait: bool = True,
  ) -> dict[str, Any]:
    """Send a follow-up message on the same persistent child session.

    Args:
      dispatch_id: Id returned by ``dispatch_agent``.
      user_message: Follow-up user message.
      tool_context: ADK tool context (injected).
      wait: Await completion (default True). If False, runs in background.
    """
    entry = await self._require_entry(dispatch_id, tool_context)
    if entry.task and not entry.task.done():
      await entry.task
    await self._ensure_runner(entry, tool_context)

    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=user_message)],
    )
    entry.status = 'running'
    entry.result = ''
    entry.done_event = asyncio.Event()
    await self._write_state(tool_context, entry, persist_parent=True)

    if wait:
      await self._execute_and_finalize(
          entry=entry,
          content=content,
          tool_context=tool_context,
          forward_child_state=True,
      )
      return self._public_payload(entry)

    entry.task = asyncio.create_task(
        self._background_run(entry=entry, content=content)
    )
    return self._public_payload(entry)

  async def await_agent(
      self,
      dispatch_id: str,
      tool_context: ToolContext,
      timeout_seconds: Optional[float] = None,
  ) -> dict[str, Any]:
    """Wait until a background dispatch completes (or timeout).

    Args:
      dispatch_id: Id returned by ``dispatch_agent``.
      tool_context: ADK tool context (injected).
      timeout_seconds: Optional timeout; on timeout returns current status.
    """
    entry = await self._require_entry(dispatch_id, tool_context)
    if entry.status == 'running':
      try:
        if timeout_seconds is None:
          await entry.done_event.wait()
        else:
          await asyncio.wait_for(
              entry.done_event.wait(), timeout=timeout_seconds
          )
      except asyncio.TimeoutError:
        return self._public_payload(entry)
      # Refresh from live entry / state after wait.
      entry = await self._require_entry(dispatch_id, tool_context)
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

  async def _resolve_skills(
      self, skill_names: Optional[list[str]]
  ) -> list[Skill]:
    if not skill_names:
      return []
    skills: list[Skill] = []
    missing: list[str] = []
    for name in skill_names:
      if name in self._skill_allowlist:
        skills.append(self._skill_allowlist[name])
        continue
      if self._skill_registry is not None:
        try:
          skills.append(await self._skill_registry.get_skill(name=name))
          continue
        except Exception:  # pylint: disable=broad-exception-caught
          logger.debug('Skill registry miss for %s', name, exc_info=True)
      missing.append(name)
    if missing:
      allowed = sorted(self._skill_allowlist)
      raise ValueError(
          f'Unknown skill_names {missing}; allowlist has {allowed}.'
      )
    return skills

  def _resolve_session_service(
      self,
      tool_context: ToolContext,
      *,
      entry: Optional[_DispatchEntry] = None,
  ) -> BaseSessionService:
    if entry is not None and entry.session_service is not None:
      return entry.session_service
    if self._session_service_override is not None:
      return self._session_service_override
    invocation_context = tool_context._invocation_context
    if invocation_context is not None:
      return invocation_context.session_service
    return self._fallback_session_service

  async def _require_entry(
      self, dispatch_id: str, tool_context: ToolContext
  ) -> _DispatchEntry:
    entry = self._entries.get(dispatch_id)
    if entry is not None:
      return entry

    state_registry = tool_context.state.get(_STATE_KEY) or {}
    data = state_registry.get(dispatch_id)
    if not data:
      # Also try parent session service persistence.
      data = await self._load_state_record(dispatch_id, tool_context)
    if not data:
      raise ValueError(f'Unknown dispatch_id: {dispatch_id!r}')

    entry = _DispatchEntry.from_state_dict(data)
    if entry.model_spec is None:
      entry.model_spec = self._model
    entry.session_service = self._resolve_session_service(tool_context)
    if entry.status != 'running':
      entry.done_event.set()
    self._entries[dispatch_id] = entry
    await self._ensure_runner(entry, tool_context)
    return entry

  async def _load_state_record(
      self, dispatch_id: str, tool_context: ToolContext
  ) -> Optional[dict[str, Any]]:
    invocation_context = tool_context._invocation_context
    if invocation_context is None or invocation_context.session is None:
      return None
    session = await invocation_context.session_service.get_session(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
    )
    if session is None:
      return None
    registry = session.state.get(_STATE_KEY) or {}
    record = registry.get(dispatch_id)
    return dict(record) if record else None

  async def _ensure_runner(
      self, entry: _DispatchEntry, tool_context: ToolContext
  ) -> None:
    if entry.runner is not None:
      return
    tools = self._resolve_tools(entry.tool_names)
    skills = await self._resolve_skills(entry.skill_names)
    agent_tools: list[Any] = list(tools)
    if skills:
      agent_tools.append(SkillToolset(skills=skills))
    model = entry.model_spec or self._model
    agent = LlmAgent(
        name=entry.agent_name,
        model=model,
        instruction=entry.instruction,
        mode=entry.mode,
        tools=agent_tools,
    )
    invocation_context = tool_context._invocation_context
    plugins = (
        invocation_context.plugin_manager.plugins
        if self._include_plugins and invocation_context
        else None
    )
    session_service = self._resolve_session_service(tool_context, entry=entry)
    entry.session_service = session_service
    runner = Runner(
        app_name=entry.app_name,
        agent=agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=session_service,
        memory_service=(
            invocation_context.memory_service
            if invocation_context and invocation_context.memory_service
            else InMemoryMemoryService()
        ),
        credential_service=(
            invocation_context.credential_service
            if invocation_context
            else None
        ),
        plugins=plugins,
    )
    if self._include_plugins and plugins is not None:
      runner.plugin_manager.set_skip_closing_plugins(True)
    entry.runner = runner

  async def _background_run(
      self, *, entry: _DispatchEntry, content: types.Content
  ) -> None:
    try:
      await self._execute_and_finalize(
          entry=entry,
          content=content,
          tool_context=None,
          forward_child_state=False,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.exception('Background dispatch failed: %s', entry.dispatch_id)
      entry.status = 'failed'
      entry.result = str(e)
      entry.done_event.set()
      await self._persist_entry_to_parent_session(entry)
      await self._fire_on_complete(entry)

  async def _execute_and_finalize(
      self,
      *,
      entry: _DispatchEntry,
      content: types.Content,
      tool_context: Optional[ToolContext],
      forward_child_state: bool,
  ) -> None:
    if entry.runner is None:
      raise RuntimeError(f'No runner for dispatch_id={entry.dispatch_id!r}')
    result_text, error_message = await self._run_child(
        runner=entry.runner,
        user_id=entry.user_id,
        session_id=entry.session_id,
        content=content,
        tool_context=tool_context if forward_child_state else None,
    )
    entry.status = (
        'failed' if error_message and not result_text else 'completed'
    )
    entry.result = result_text or error_message or ''
    entry.done_event.set()
    if tool_context is not None:
      await self._write_state(tool_context, entry, persist_parent=True)
    else:
      await self._persist_entry_to_parent_session(entry)
    await self._fire_on_complete(entry)

  async def _run_child(
      self,
      *,
      runner: Runner,
      user_id: str,
      session_id: str,
      content: types.Content,
      tool_context: Optional[ToolContext],
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
        if tool_context is not None and event.actions.state_delta:
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

  async def _write_state(
      self,
      tool_context: ToolContext,
      entry: _DispatchEntry,
      *,
      persist_parent: bool,
  ) -> None:
    async with self._lock:
      registry = dict(tool_context.state.get(_STATE_KEY) or {})
      registry[entry.dispatch_id] = entry.to_state_dict()
      tool_context.state[_STATE_KEY] = registry
    if persist_parent:
      await self._persist_entry_to_parent_session(entry)

  async def _persist_entry_to_parent_session(
      self, entry: _DispatchEntry
  ) -> None:
    if (
        not entry.parent_session_id
        or not entry.parent_app_name
        or not entry.parent_user_id
    ):
      return
    service = entry.session_service
    if service is None:
      service = self._session_service_override or self._fallback_session_service
    try:
      session = await service.get_session(
          app_name=entry.parent_app_name,
          user_id=entry.parent_user_id,
          session_id=entry.parent_session_id,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          'Failed to load parent session for dispatch_id=%s', entry.dispatch_id
      )
      return
    if session is None:
      return
    registry = dict(session.state.get(_STATE_KEY) or {})
    registry[entry.dispatch_id] = entry.to_state_dict()
    event = Event(
        author='agent_dispatcher',
        state={_STATE_KEY: registry},
    )
    try:
      await service.append_event(session, event)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          'Failed to persist dispatcher state for %s', entry.dispatch_id
      )

  async def _fire_on_complete(self, entry: _DispatchEntry) -> None:
    if self._on_complete is None:
      return
    payload = self._public_payload(entry)
    try:
      result = self._on_complete(payload)
      if inspect.isawaitable(result):
        await result
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          'on_complete callback failed for dispatch_id=%s', entry.dispatch_id
      )

  def _public_payload(self, entry: _DispatchEntry) -> dict[str, Any]:
    return {
        'dispatch_id': entry.dispatch_id,
        'agent_name': entry.agent_name,
        'status': entry.status,
        'result': entry.result,
    }
