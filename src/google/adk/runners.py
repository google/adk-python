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

import asyncio
from contextlib import aclosing
import inspect
import logging
from pathlib import Path
import queue
import sys
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
import warnings

from google.genai import types

from .agents.base_agent import BaseAgent
from .agents.context_cache_config import ContextCacheConfig
from .agents.invocation_context import InvocationContext
from .agents.invocation_context import new_invocation_context_id
from .agents.live_request_queue import LiveRequestQueue
from .agents.llm.task._finish_task_tool import FINISH_TASK_SUCCESS_RESULT
from .agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from .agents.run_config import RunConfig
from .artifacts.base_artifact_service import BaseArtifactService
from .auth.credential_service.base_credential_service import BaseCredentialService
from .code_executors.built_in_code_executor import BuiltInCodeExecutor
from .errors.session_not_found_error import SessionNotFoundError
from .events.event import Event
from .events.event import EventActions
from .flows.llm_flows import contents
from .flows.llm_flows.functions import find_event_by_function_call_id
from .flows.llm_flows.functions import find_matching_function_call
from .memory.base_memory_service import BaseMemoryService
from .platform.thread import create_thread
from .plugins.base_plugin import BasePlugin
from .plugins.plugin_manager import PluginManager
from .sessions.base_session_service import BaseSessionService
from .sessions.base_session_service import GetSessionConfig
from .sessions.session import Session
from .telemetry import _instrumentation
from .telemetry.tracing import tracer
from .tools.base_toolset import BaseToolset
from .utils._debug_output import print_event

if TYPE_CHECKING:
  from .apps.app import App
  from .apps.app import ResumabilityConfig

logger = logging.getLogger('google_adk.' + __name__)

# Silence unused warning.
# tracer is imported for backwards compatibility, to avoid breaking change in the API.
_ = tracer


def _find_active_task_isolation_scope(session) -> Optional[str]:
  """Walk session backwards; find the active paused task agent's scope.

  Two flavors of task scope:
    * FC delegation (chat coordinator → task agent via function call):
      scope = ``fc.id``, opened by an unresolved task FC.
    * Workflow node (task-mode LlmAgent dispatched as a graph node):
      scope = ``<node_name>@<run_id>``, stamped on every event the
      task agent emits.

  Both close on a SUCCESSFUL ``finish_task`` FunctionResponse —
  i.e., one whose response is ``FINISH_TASK_SUCCESS_RESULT``.  An
  error FR (validation failure) does NOT close the scope: the task
  agent is still active, will see the error, and retry.  Walking
  backward, the first non-empty scope we encounter that hasn't been
  closed by a later successful ``finish_task`` is the paused task
  awaiting the user's next reply.

  Used by ``Runner._append_user_event`` to scope the new user message
  to that task agent's view.
  """
  finished_scopes: set[str] = set()
  for event in reversed(session.events):
    scope = event.isolation_scope
    if not scope:
      continue
    if event.content and event.content.parts:
      for part in event.content.parts:
        fr = part.function_response
        if fr and fr.name == FINISH_TASK_TOOL_NAME:
          response = fr.response or {}
          if response.get('result') == FINISH_TASK_SUCCESS_RESULT:
            finished_scopes.add(scope)
          break
    if scope not in finished_scopes:
      return scope
  return None


def _get_function_responses_from_content(
    content: types.Content,
) -> list[types.FunctionResponse]:
  if not content:
    return []
  return [
      part.function_response for part in content.parts if part.function_response
  ]


def _apply_run_config_custom_metadata(
    event: Event, run_config: RunConfig | None
) -> None:
  """Merges run-level custom metadata into the event, if present."""
  if not run_config or not run_config.custom_metadata:
    return

  event.custom_metadata = {
      **run_config.custom_metadata,
      **(event.custom_metadata or {}),
  }


class Runner:
  """The Runner class is used to run agents.

  It manages the execution of an agent within a session, handling message
  processing, event generation, and interaction with various services like
  artifact storage, session management, and memory.

  Attributes:
      app_name: The application name of the runner.
      agent: The root agent to run.
      artifact_service: The artifact service for the runner.
      plugin_manager: The plugin manager for the runner.
      session_service: The session service for the runner.
      memory_service: The memory service for the runner.
      credential_service: The credential service for the runner.
      context_cache_config: The context cache config for the runner.
      resumability_config: The resumability config for the application.
  """

  app_name: str
  """The app name of the runner."""
  agent: Optional[BaseAgent | 'BaseNode'] = None
  """The root agent or node to run."""
  artifact_service: Optional[BaseArtifactService] = None
  """The artifact service for the runner."""
  plugin_manager: PluginManager
  """The plugin manager for the runner."""
  session_service: BaseSessionService
  """The session service for the runner."""
  memory_service: Optional[BaseMemoryService] = None
  """The memory service for the runner."""
  credential_service: Optional[BaseCredentialService] = None
  """The credential service for the runner."""
  context_cache_config: Optional[ContextCacheConfig] = None
  """The context cache config for the runner."""
  resumability_config: Optional[ResumabilityConfig] = None
  """The resumability config for the application."""

  def __init__(
      self,
      *,
      app: Optional[App] = None,
      app_name: Optional[str] = None,
      agent: Optional[BaseAgent] = None,
      node: Any = None,
      plugins: Optional[List[BasePlugin]] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      session_service: BaseSessionService,
      memory_service: Optional[BaseMemoryService] = None,
      credential_service: Optional[BaseCredentialService] = None,
      plugin_close_timeout: float = 5.0,
      auto_create_session: bool = False,
  ):
    """Initializes the Runner.

    Exactly one of `app`, `agent`, or `node` must be provided. When `agent`
    or `node` is provided, the Runner wraps it into an `App` internally.
    Providing `app` is the recommended way to create a runner. When `app` is
    provided, `app_name` can optionally override the app's name.

    Args:
        app: An `App` instance. Mutually exclusive with `agent` and `node`.
        app_name: The application name. Required when `agent` is provided.
          Optional override for `app.name` when `app` is provided. Defaults to
          `node.name` when only `node` is provided.
        agent: The root agent to run. Mutually exclusive with `app` and `node`.
        node: The root node to run. Mutually exclusive with `app` and `agent`.
        plugins: Deprecated. A list of plugins for the runner. Please use the
          `app` argument to provide plugins instead.
        artifact_service: The artifact service for the runner.
        session_service: The session service for the runner.
        memory_service: The memory service for the runner.
        credential_service: The credential service for the runner.
        plugin_close_timeout: The timeout in seconds for plugin close methods.
        auto_create_session: Whether to automatically create a session when not
          found. Defaults to False. If False, a missing session raises
          ValueError with a helpful message.

    Raises:
        ValueError: If more than one of `app`, `agent`, or `node` is provided,
          or if none is provided, or if `agent` is provided without `app_name`.
    """
    app = self._resolve_app(app, app_name, agent, node, plugins)

    # Extract from App — single code path.
    self.app = app
    self.app_name = app_name or app.name
    self.agent = app.root_agent
    self.context_cache_config = app.context_cache_config
    self.resumability_config = app.resumability_config
    self.artifact_service = artifact_service
    self.session_service = session_service
    self.memory_service = memory_service
    self.credential_service = credential_service
    self.plugin_manager = PluginManager(
        plugins=app.plugins, close_timeout=plugin_close_timeout
    )
    self.auto_create_session = auto_create_session
    if self.agent is not None:
      (
          self._agent_origin_app_name,
          self._agent_origin_dir,
      ) = self._infer_agent_origin(self.agent)
    else:
      self._agent_origin_app_name = None
      self._agent_origin_dir = None
    self._app_name_alignment_hint: Optional[str] = None
    self._enforce_app_name_alignment()

  @staticmethod
  def _resolve_app(
      app: Optional[App],
      app_name: Optional[str],
      agent: Optional[BaseAgent],
      node: Any,
      plugins: Optional[List[BasePlugin]],
  ) -> App:
    """Validates inputs and normalizes to an App instance.

    Exactly one of ``app``, ``agent``, or ``node`` must be provided.
    When ``agent`` or ``node`` is given, it is wrapped in a new ``App``.

    Returns:
      The resolved ``App`` instance.

    Raises:
      ValueError: If the combination of arguments is invalid.
    """
    # Validate mutual exclusivity.
    provided = sum(x is not None for x in (app, agent, node))
    if provided > 1:
      raise ValueError('Only one of app, agent, or node may be provided.')
    if provided == 0:
      raise ValueError('One of app, agent, or node must be provided.')

    # Handle deprecated plugins argument.
    if plugins is not None:
      if app is not None:
        raise ValueError(
            'When app is provided, plugins should not be provided and should'
            ' be provided in the app instead.'
        )
      warnings.warn(
          'The `plugins` argument is deprecated. Please use the `app` argument'
          ' to provide plugins instead.',
          DeprecationWarning,
      )

    # Lazy import keeps apps.app off the `import google.adk` cold-start path.
    from .apps.app import App

    # Normalize to App — wrap bare agent or node. Uses model_construct to
    # bypass App._validate for the legacy (app_name, agent) API, which v1
    # accepted with arbitrary names and root_agent types. Direct App(name=...)
    # construction still validates strictly.
    if agent is not None:
      if not app_name:
        raise ValueError(
            'app_name is required when agent is provided without app.'
        )
      return App.model_construct(
          name=app_name, root_agent=agent, plugins=plugins or []
      )
    if node is not None:
      return App.model_construct(
          name=app_name or getattr(node, 'name', 'default'),
          root_agent=node,
          plugins=plugins or [],
      )
    return app

  @staticmethod
  def _validate_runner_params(
      app: Optional[App],
      app_name: Optional[str],
      agent: Optional[BaseAgent],
      plugins: Optional[List[BasePlugin]],
  ) -> tuple[
      str,
      BaseAgent,
      Optional[ContextCacheConfig],
      Optional[ResumabilityConfig],
      Optional[List[BasePlugin]],
  ]:
    """Deprecated: use _resolve_app instead."""
    resolved = Runner._resolve_app(app, app_name, agent, None, plugins)
    return (
        app_name or resolved.name,
        resolved.root_agent,
        resolved.context_cache_config,
        resolved.resumability_config,
        plugins if app is None else resolved.plugins,
    )

  def _infer_agent_origin(
      self, agent: BaseAgent
  ) -> tuple[Optional[str], Optional[Path]]:
    """Infer the origin app name and directory from an agent's module location.

    Returns:
      A tuple of (origin_app_name, origin_path):
        - origin_app_name: The inferred app name (directory name containing the
          agent), or None if inference is not possible/applicable.
        - origin_path: The directory path where the agent is defined, or None
          if the path cannot be determined.

      Both values are None when:
        - The agent has no associated module
        - The agent is defined in google.adk.* (ADK internal modules)
        - The module has no __file__ attribute
    """
    # First, check for metadata set by AgentLoader (most reliable source).
    # AgentLoader sets these attributes when loading agents.
    origin_app_name = getattr(agent, '_adk_origin_app_name', None)
    origin_path = getattr(agent, '_adk_origin_path', None)
    if origin_app_name is not None and origin_path is not None:
      return origin_app_name, origin_path

    # Fall back to heuristic inference for programmatic usage.
    module = inspect.getmodule(agent.__class__)
    if not module:
      return None, None

    # Skip ADK internal modules. When users instantiate LlmAgent directly
    # (not subclassed), inspect.getmodule() returns the ADK module. This
    # could falsely match 'agents' in 'google/adk/agents/' path.
    if module.__name__.startswith('google.adk.'):
      return None, None

    module_file = getattr(module, '__file__', None)
    if not module_file:
      return None, None
    module_path = Path(module_file).resolve()
    project_root = Path.cwd()
    try:
      relative_path = module_path.relative_to(project_root)
    except ValueError:
      return None, module_path.parent
    origin_dir = module_path.parent
    if 'agents' not in relative_path.parts:
      return None, origin_dir
    origin_name = origin_dir.name
    if origin_name.startswith('.'):
      return None, origin_dir
    return origin_name, origin_dir

  def _enforce_app_name_alignment(self) -> None:
    origin_name = self._agent_origin_app_name
    origin_dir = self._agent_origin_dir
    if not origin_name or origin_name.startswith('__'):
      self._app_name_alignment_hint = None
      return
    if origin_name == self.app_name:
      self._app_name_alignment_hint = None
      return
    origin_location = str(origin_dir) if origin_dir else origin_name
    mismatch_details = (
        'The runner is configured with app name '
        f'"{self.app_name}", but the root agent was loaded from '
        f'"{origin_location}", which implies app name "{origin_name}".'
    )
    resolution = (
        'Ensure the runner app_name matches that directory or pass app_name '
        'explicitly when constructing the runner.'
    )
    self._app_name_alignment_hint = f'{mismatch_details} {resolution}'
    logger.warning('App name mismatch detected. %s', mismatch_details)

  def _resolve_invocation_id(
      self,
      session: Session,
      new_message: Optional[types.Content],
      invocation_id: Optional[str],
  ) -> Optional[str]:
    """Infers invocation_id from new_message if it is a function response."""
    function_responses = _get_function_responses_from_content(new_message)
    if not function_responses:
      return invocation_id

    fc_event = find_event_by_function_call_id(
        session.events, function_responses[0].id
    )
    if not fc_event:
      raise ValueError(
          'Function call event not found for function response id:'
          f' {function_responses[0].id}'
      )

    if invocation_id and invocation_id != fc_event.invocation_id:
      logger.warning(
          'Provided invocation_id %s is ignored because new_message has a '
          'function response with invocation_id %s.',
          invocation_id,
          fc_event.invocation_id,
      )
    return fc_event.invocation_id

  def _format_session_not_found_message(self, session_id: str) -> str:
    message = f'Session not found: {session_id}'
    if not self._app_name_alignment_hint:
      return message
    return (
        f'{message}. {self._app_name_alignment_hint} '
        'The mismatch prevents the runner from locating the session. '
        'To automatically create a session when missing, set '
        'auto_create_session=True when constructing the runner.'
    )

  async def _run_node_async(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
      yield_user_message: bool = False,
      node: Optional['BaseNode'] = None,
  ) -> AsyncGenerator[Event, None]:
    """Run a BaseNode through NodeRunner.

    Events flow through ic._event_queue via NodeRunner.
    """

    with _instrumentation.record_invocation(
        entrypoint_node=node or self.agent, conversation_id=session_id
    ):
      # 1. Setup
      session = await self._get_or_create_session(
          user_id=user_id, session_id=session_id
      )

      # Validate and resolve resume inputs
      resume_inputs = self._extract_resume_inputs(new_message)
      self._validate_new_message(new_message, resume_inputs)

      if not invocation_id and new_message:
        invocation_id = self._resolve_invocation_id_from_fr(
            session, new_message
        )

      ic = self._new_invocation_context(
          session,
          new_message=new_message,
          run_config=run_config or RunConfig(),
          invocation_id=invocation_id,
      )
      ic._event_queue = asyncio.Queue()

      # 2. Append user message to session and resolve node_input
      node_input = None
      if resume_inputs or invocation_id:
        # Resume: recover the original user content. new_message here is a
        # function response (or None), so it can't populate user_content.
        node_input = self._find_original_user_content(
            ic.session, ic.invocation_id
        )
        if node_input:
          ic.user_content = node_input
      if not node_input:
        # Fresh: use user message as node_input
        node_input = new_message

      # Run callbacks on user message
      if new_message:
        modified_user_message = (
            await ic.plugin_manager.run_on_user_message_callback(
                invocation_context=ic, user_message=new_message
            )
        )
        if modified_user_message is not None:
          new_message = modified_user_message
          ic.user_content = new_message

      # Append user message to session for history
      if new_message:
        user_event = await self._append_user_event(
            ic, new_message, state_delta=state_delta
        )
        if yield_user_message and user_event:
          yield user_event

      # Run before_run callbacks
      await ic.plugin_manager.run_before_run_callback(invocation_context=ic)

      # 3. Start root node in background
      from .agents.base_agent import BaseAgent
      from .agents.context import Context
      from .workflow._dynamic_node_scheduler import DynamicNodeScheduler
      from .workflow._errors import DynamicNodeFailError
      from .workflow._errors import NodeInterruptedError
      from .workflow._workflow import _LoopState

      root_ctx = Context(ic)
      root_agent = node or self.agent
      is_agent = isinstance(self.agent, BaseAgent)
      has_sub_agents = is_agent and bool(
          getattr(self.agent, 'sub_agents', None)
      )
      use_scheduler = is_agent and has_sub_agents

      # The root chat coordinator's isolation_scope stays None: its own
      # events (FCs, text, synthesized FRs from completed task
      # delegations) are also unscoped, so the content-builder's
      # isolation_scope filter lets the coordinator see all of them
      # across user turns. Task sub-agents are scoped under their
      # originating function-call id and so remain invisible to the
      # coordinator's view.

      done_sentinel = object()

      async def _drive_root_node():
        try:
          if use_scheduler:
            # Rehydration warning: DynamicNodeScheduler relies on session.events scanning.
            # Stateful live EUC/LRO streams may rehydrate freshly if not yet persisted.
            scheduler = DynamicNodeScheduler(state=_LoopState())
            root_ctx._workflow_scheduler = scheduler

          try:
            await root_ctx._run_node_internal(
                root_agent,
                node_input=node_input,
                resume_inputs=resume_inputs,
            )
          except NodeInterruptedError:
            # The node was interrupted (e.g. for HITL).
            pass
          except DynamicNodeFailError as e:
            raise e.error
        finally:
          await ic._event_queue.put((done_sentinel, None))

      task = asyncio.create_task(_drive_root_node())

      # 4. Main loop: consume events, persist, yield
      try:
        async with aclosing(
            self._consume_event_queue(ic, done_sentinel)
        ) as agen:
          async for event in agen:
            yield event
      finally:
        await self._cleanup_root_task(task, self.agent.name)
        await ic.plugin_manager.run_after_run_callback(invocation_context=ic)
        if self.app and self.app.events_compaction_config:
          logger.debug('Running event compactor.')
          from google.adk.apps.compaction import _run_compaction_for_sliding_window

          await _run_compaction_for_sliding_window(
              self.app,
              session,
              self.session_service,
              skip_token_compaction=ic.token_compaction_checked,
          )

  async def _run_node_live(
      self,
      *,
      session: Session,
      live_request_queue: LiveRequestQueue,
      run_config: Optional[RunConfig] = None,
  ) -> AsyncGenerator[Event, None]:
    """Run a non-agent BaseNode in live mode."""
    from .agents.context import Context
    from .workflow._dynamic_node_scheduler import DynamicNodeScheduler
    from .workflow._errors import DynamicNodeFailError
    from .workflow._errors import NodeInterruptedError
    from .workflow._workflow import _LoopState
    from .workflow._workflow import Workflow

    ic = self._new_invocation_context_for_live(
        session,
        live_request_queue=live_request_queue,
        run_config=run_config or RunConfig(),
    )
    ic._event_queue = asyncio.Queue()

    root_ctx = Context(ic)
    root_agent = self.agent
    is_workflow = isinstance(root_agent, Workflow)

    done_sentinel = object()

    async def _drive_root_node():
      try:
        if is_workflow:
          scheduler = DynamicNodeScheduler(state=_LoopState())
          root_ctx._workflow_scheduler = scheduler

        try:
          await root_ctx.run_node(
              root_agent,
              node_input=None,
          )
        except NodeInterruptedError:
          pass
        except DynamicNodeFailError as e:
          raise e.error
      finally:
        await ic._event_queue.put((done_sentinel, None))

    task = asyncio.create_task(_drive_root_node())

    try:
      async with aclosing(self._consume_event_queue(ic, done_sentinel)) as agen:
        async for event in agen:
          yield event
    finally:
      await self._cleanup_root_task(task, self.agent.name)

  def _extract_resume_inputs(
      self, message: Optional[types.Content]
  ) -> dict[str, Any] | None:
    """Extract function response payloads from a message as resume_inputs."""
    if not message or not message.parts:
      return None
    inputs = {}
    for part in message.parts:
      if part.function_response and part.function_response.id:
        inputs[part.function_response.id] = part.function_response.response
    return inputs or None

  def _validate_new_message(
      self,
      message: Optional[types.Content],
      resume_inputs: dict[str, Any] | None,
  ) -> None:
    """Validate that new_message doesn't mix FR and text parts."""
    if not resume_inputs or not message or not message.parts:
      return
    if any(p.text for p in message.parts):
      raise ValueError(
          'Message cannot contain both function responses and text.'
          ' Function responses resume an existing invocation while'
          ' text starts a new one.'
      )

  def _resolve_invocation_id_from_fr(
      self,
      session: Session,
      new_message: types.Content,
  ) -> Optional[str]:
    """Infer invocation_id by matching function responses to FC events.

    Raises ValueError if responses resolve to different invocations.
    """
    fr_ids = {
        p.function_response.id
        for p in new_message.parts or []
        if p.function_response and p.function_response.id
    }
    if not fr_ids:
      return None

    # Find invocation_id for each FR by matching its FC in session
    invocation_ids = set()
    for event in reversed(session.events):
      for fc in event.get_function_calls():
        if fc.id in fr_ids:
          invocation_ids.add(event.invocation_id)
          fr_ids.discard(fc.id)
      if not fr_ids:
        break

    if fr_ids:
      raise ValueError(
          f'Function call not found for function response ids: {fr_ids}.'
      )
    if len(invocation_ids) > 1:
      raise ValueError(
          'Function responses resolve to multiple'
          f' invocations: {invocation_ids}.'
      )
    return invocation_ids.pop()

  async def _append_user_event(
      self,
      ic: InvocationContext,
      content: types.Content,
      *,
      state_delta: Optional[dict[str, Any]] = None,
  ) -> Event:
    """Append a user message event to the session and return it."""
    if state_delta:
      event = Event(
          invocation_id=ic.invocation_id,
          author='user',
          actions=EventActions(state_delta=state_delta),
          content=content,
      )
    else:
      event = Event(
          invocation_id=ic.invocation_id,
          author='user',
          content=content,
      )
    # when a paused task delegation is in flight, stamp
    # the new user message with that task's isolation_scope so the
    # task agent's content-build (scoped to <fc_id>) sees it.
    if event.isolation_scope is None:
      iso = _find_active_task_isolation_scope(ic.session)
      if iso is not None:
        event.isolation_scope = iso
    _apply_run_config_custom_metadata(event, ic.run_config)
    ic.stamp_event_branch_context(event)
    return await self.session_service.append_event(
        session=ic.session, event=event
    )

  def _find_original_user_content(
      self, session: Session, invocation_id: str
  ) -> types.Content | None:
    """Find the original user text message for a given invocation_id."""
    for event in session.events:
      if (
          event.invocation_id == invocation_id
          and event.author == 'user'
          and event.content
          and event.content.parts
          and any(p.text for p in event.content.parts)
      ):
        return event.content
    return None

  async def _consume_event_queue(
      self, ic: InvocationContext, done_sentinel: object
  ) -> AsyncGenerator[Event, None]:
    """Consume events from ic._event_queue until done_sentinel."""
    while True:
      event_or_done, processed_signal = await ic._event_queue.get()
      if event_or_done is done_sentinel:
        break
      event: Event = event_or_done
      # When an LlmAgent node uses ``message_as_output`` (no
      # ``output_schema``), the wrapper sets both ``event.content``
      # (the model's text) AND ``event.output`` (the same text) to
      # signal that the message IS the node's output.  Clear
      # ``event.output`` on a copy here so downstream renderers don't
      # surface the same text twice.  Task-mode agents set
      # ``event.output`` from the ``finish_task`` FC args without
      # ``message_as_output``, so this clearing doesn't affect them.
      if not event.partial:
        if event.node_info.message_as_output and event.content is not None:
          event = event.model_copy()
          event.output = None

      _apply_run_config_custom_metadata(event, ic.run_config)
      modified_event = await ic.plugin_manager.run_on_event_callback(
          invocation_context=ic, event=event
      )
      output_event = self._get_output_event(
          original_event=event,
          modified_event=modified_event,
          run_config=ic.run_config,
      )

      if not event.partial:
        await self.session_service.append_event(
            session=ic.session, event=output_event
        )
      yield output_event

      if isinstance(processed_signal, asyncio.Event):
        processed_signal.set()

  async def _cleanup_root_task(
      self, task: asyncio.Task, node_name: str
  ) -> None:
    """Cancel the root task if still running, then await it.

    The task may still be running if the caller stopped iterating
    early (e.g., break in async for). In that case we must cancel
    to avoid a leaked task.
    """
    if not task.done():
      logger.debug(
          'Cancelling root node %s (caller stopped early).',
          node_name,
      )
      task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      logger.warning('Root node %s was cancelled.', node_name)
    except Exception:
      logger.error('Root node %s failed.', node_name, exc_info=True)
      raise

  async def _get_or_create_session(
      self,
      *,
      user_id: str,
      session_id: str,
      get_session_config: Optional[GetSessionConfig] = None,
  ) -> Session:
    """Gets the session or creates it if auto-creation is enabled.

    This helper first attempts to retrieve the session. If not found and
    auto_create_session is True, it creates a new session with the provided
    identifiers. Otherwise, it raises a SessionNotFoundError.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      get_session_config: Optional configuration for controlling which events
        are fetched from session storage.

    Returns:
      The existing or newly created `Session`.

    Raises:
      SessionNotFoundError: If the session is not found and
        auto_create_session is False.
    """
    session = await self.session_service.get_session(
        app_name=self.app_name,
        user_id=user_id,
        session_id=session_id,
        config=get_session_config,
    )
    if not session:
      if self.auto_create_session:
        session = await self.session_service.create_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )
      else:
        message = self._format_session_not_found_message(session_id)
        raise SessionNotFoundError(message)
    return session

  def run(
      self,
      *,
      user_id: str,
      session_id: str,
      new_message: types.Content,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ) -> Generator[Event, None, None]:
    """Runs the agent.

    NOTE:
      This sync interface is only for local testing and convenience purpose.
      Consider using `run_async` for production usage.

    If event compaction is enabled in the App configuration, it will be
    performed after all agent events for the current invocation have been
    yielded. The generator will only finish iterating after event
    compaction is complete.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      new_message: A new message to append to the session.
      state_delta: Optional state changes to apply to the session.
      run_config: The run config for the agent.

    Yields:
      The events generated by the agent.
    """
    run_config = run_config or RunConfig()
    event_queue = queue.Queue()

    async def _invoke_run_async():
      try:
        async with aclosing(
            self.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
                state_delta=state_delta,
                run_config=run_config,
            )
        ) as agen:
          async for event in agen:
            event_queue.put(event)
      finally:
        event_queue.put(None)

    def _asyncio_thread_main():
      try:
        asyncio.run(_invoke_run_async())
      finally:
        event_queue.put(None)

    thread = create_thread(target=_asyncio_thread_main)
    thread.start()

    # consumes and re-yield the events from background thread.
    while True:
      event = event_queue.get()
      if event is None:
        break
      else:
        yield event

    thread.join()

  async def run_async(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
      yield_user_message: bool = False,
  ) -> AsyncGenerator[Event, None]:
    """Main entry method to run the agent in this runner.

    If event compaction is enabled in the App configuration, it will be
    performed after all agent events for the current invocation have been
    yielded. The async generator will only finish iterating after event
    compaction is complete. However, this does not block new `run_async`
    calls for subsequent user queries, which can be started concurrently.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      invocation_id: The invocation ID of the session, set this to resume an
        interrupted invocation.
      new_message: A new message to append to the session.
      state_delta: Optional state changes to apply to the session.
      run_config: The run config for the agent.
      yield_user_message: If True, yield the user message event before
        agent/node events.

    Yields:
      The events generated by the agent.

    Raises:
      ValueError: If the session is not found; If both invocation_id and
        new_message are None.
    """
    run_config = run_config or RunConfig()

    if new_message and not new_message.role:
      new_message.role = 'user'

    from .agents.llm_agent import LlmAgent
    from .workflow._base_node import BaseNode

    if isinstance(self.agent, LlmAgent):
      if self.agent.mode is None:
        # LlmAgent as root agent must have chat mode.
        self.agent.mode = 'chat'

      if self.agent.mode == 'chat':
        session = await self._get_or_create_session(
            user_id=user_id, session_id=session_id
        )
        # when the chat coordinator has task-mode sub-agents,
        # the wrapper handles delegation via ctx.run_node. Don't let
        # the legacy sub-agent picker bypass the coordinator on resume.
        has_task_subagent = any(
            isinstance(sa, LlmAgent) and getattr(sa, 'mode', None) == 'task'
            for sa in self.agent.sub_agents or []
        )
        if has_task_subagent:
          agent_to_run = self.agent
        else:
          agent_to_run = self._find_agent_to_run(session, self.agent)

        # The agent_to_run will be built/cloned inside Context.run_node,
        # so we don't call build_node here to avoid double cloning.
      else:
        raise ValueError(
            "LlmAgent as root agent must have mode='chat', but got"
            f" mode='{self.agent.mode}'."
        )
      async with aclosing(
          self._run_node_async(
              user_id=user_id,
              session_id=session_id,
              invocation_id=invocation_id,
              new_message=new_message,
              state_delta=state_delta,
              run_config=run_config,
              yield_user_message=yield_user_message,
              node=agent_to_run,
          )
      ) as agen:
        async for event in agen:
          yield event
      return

    # TODO: remove `not isinstance(self.agent, BaseAgent)` after all agents are
    # refactored to use the node runtime path (requires adding tracing and plugins to it).
    if isinstance(self.agent, BaseNode) and not isinstance(
        self.agent, BaseAgent
    ):
      async with aclosing(
          self._run_node_async(
              user_id=user_id,
              session_id=session_id,
              invocation_id=invocation_id,
              new_message=new_message,
              state_delta=state_delta,
              run_config=run_config,
              yield_user_message=yield_user_message,
          )
      ) as agen:
        async for event in agen:
          yield event
      return

    async def _run_with_trace(
        new_message: Optional[types.Content] = None,
        invocation_id: Optional[str] = None,
    ) -> AsyncGenerator[Event, None]:
      with _instrumentation.record_invocation(
          entrypoint_node=self.agent, conversation_id=session_id
      ):
        session = await self._get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            get_session_config=run_config.get_session_config,
        )

        if not invocation_id and not new_message:
          raise ValueError(
              'Running an agent requires either a new_message or an '
              'invocation_id to resume a previous invocation. '
              f'Session: {session_id}, User: {user_id}'
          )

        is_resumable = (
            self.resumability_config and self.resumability_config.is_resumable
        )
        if not is_resumable and not new_message:
          raise ValueError(
              'Running an agent requires a new_message or a resumable app. '
              f'Session: {session_id}, User: {user_id}'
          )

        if not is_resumable:
          invocation_context = await self._setup_context_for_new_invocation(
              session=session,
              new_message=new_message,
              run_config=run_config,
              state_delta=state_delta,
              invocation_id=invocation_id,
          )
        else:
          invocation_id = self._resolve_invocation_id(
              session, new_message, invocation_id
          )
          if not invocation_id:
            invocation_context = await self._setup_context_for_new_invocation(
                session=session,
                new_message=new_message,
                run_config=run_config,
                state_delta=state_delta,
            )
          else:
            invocation_context = (
                await self._setup_context_for_resumed_invocation(
                    session=session,
                    new_message=new_message,
                    invocation_id=invocation_id,
                    run_config=run_config,
                    state_delta=state_delta,
                )
            )
            if invocation_context.end_of_agents.get(
                invocation_context.agent.name
            ):
              # Directly return if the current agent in invocation context is
              # already final.
              return

        async def execute(ctx: InvocationContext) -> AsyncGenerator[Event]:
          async with aclosing(ctx.agent.run_async(ctx)) as agen:
            async for event in agen:
              yield event

        async with aclosing(
            self._exec_with_plugin(
                invocation_context=invocation_context,
                session=invocation_context.session,
                execute_fn=execute,
                is_live_call=False,
            )
        ) as agen:
          async for event in agen:
            yield event
        # Run compaction after all events are yielded from the agent.
        # (We don't compact in the middle of an invocation, we only compact at
        # the end of an invocation.)
        if self.app and self.app.events_compaction_config:
          logger.debug('Running event compactor.')
          from google.adk.apps.compaction import _run_compaction_for_sliding_window

          await _run_compaction_for_sliding_window(
              self.app,
              invocation_context.session,
              self.session_service,
              skip_token_compaction=invocation_context.token_compaction_checked,
          )

    async with aclosing(_run_with_trace(new_message, invocation_id)) as agen:
      async for event in agen:
        yield event

  async def rewind_async(
      self,
      *,
      user_id: str,
      session_id: str,
      rewind_before_invocation_id: str,
      run_config: Optional[RunConfig] = None,
  ) -> None:
    """Rewinds the session to before the specified invocation."""
    run_config = run_config or RunConfig()
    session = await self._get_or_create_session(
        user_id=user_id,
        session_id=session_id,
        get_session_config=run_config.get_session_config,
    )
    rewind_event_index = -1
    for i, event in enumerate(session.events):
      if event.invocation_id == rewind_before_invocation_id:
        rewind_event_index = i
        break

    if rewind_event_index == -1:
      raise ValueError(
          f'Invocation ID not found: {rewind_before_invocation_id}'
      )

    # Compute state delta to reverse changes
    state_delta = await self._compute_state_delta_for_rewind(
        session, rewind_event_index
    )

    # Compute artifact delta to reverse changes
    artifact_delta = await self._compute_artifact_delta_for_rewind(
        session, rewind_event_index
    )

    # Create rewind event
    rewind_event = Event(
        invocation_id=new_invocation_context_id(),
        author='user',
        actions=EventActions(
            rewind_before_invocation_id=rewind_before_invocation_id,
            state_delta=state_delta,
            artifact_delta=artifact_delta,
        ),
    )

    logger.info('Rewinding session to invocation: %s', rewind_event)

    await self.session_service.append_event(session=session, event=rewind_event)

  async def _compute_state_delta_for_rewind(
      self, session: Session, rewind_event_index: int
  ) -> dict[str, Any]:
    """Computes the state delta to reverse changes."""
    state_at_rewind_point: dict[str, Any] = {}
    for i in range(rewind_event_index):
      if session.events[i].actions.state_delta:
        for k, v in session.events[i].actions.state_delta.items():
          if k.startswith('app:') or k.startswith('user:'):
            continue
          if v is None:
            state_at_rewind_point.pop(k, None)
          else:
            state_at_rewind_point[k] = v

    current_state = session.state
    rewind_state_delta = {}

    # 1. Add/update keys in rewind_state_delta to match state_at_rewind_point.
    for key, value_at_rewind in state_at_rewind_point.items():
      if key not in current_state or current_state[key] != value_at_rewind:
        rewind_state_delta[key] = value_at_rewind

    # 2. Set keys to None in rewind_state_delta if they are in current_state
    #    but not in state_at_rewind_point. These keys were added after the
    #    rewind point and need to be removed.
    for key in current_state:
      if key.startswith('app:') or key.startswith('user:'):
        continue
      if key not in state_at_rewind_point:
        rewind_state_delta[key] = None

    return rewind_state_delta

  async def _compute_artifact_delta_for_rewind(
      self, session: Session, rewind_event_index: int
  ) -> dict[str, int]:
    """Computes the artifact delta to reverse changes."""
    if not self.artifact_service:
      return {}

    versions_at_rewind_point: dict[str, int] = {}
    for i in range(rewind_event_index):
      event = session.events[i]
      if event.actions.artifact_delta:
        versions_at_rewind_point.update(event.actions.artifact_delta)

    current_versions: dict[str, int] = {}
    for event in session.events:
      if event.actions.artifact_delta:
        current_versions.update(event.actions.artifact_delta)

    rewind_artifact_delta = {}
    for filename, vn in current_versions.items():
      if filename.startswith('user:'):
        # User artifacts are not restored on rewind.
        continue
      vt = versions_at_rewind_point.get(filename)
      if vt == vn:
        continue

      rewind_artifact_delta[filename] = vn + 1
      if vt is None:
        # Artifact did not exist at rewind point. Mark it as inaccessible.
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type='application/octet-stream', data=b''
            )
        )
      else:
        # Artifact version changed after rewind point. Restore to version at
        # rewind point by loading the actual data via the artifact service.
        artifact = await self.artifact_service.load_artifact(
            app_name=self.app_name,
            user_id=session.user_id,
            session_id=session.id,
            filename=filename,
            version=vt,
        )
        if artifact is None:
          logger.warning(
              'Artifact %s version %d not found during rewind for'
              ' session %s. Replacing with empty data.',
              filename,
              vt,
              session.id,
          )
          artifact = types.Part(
              inline_data=types.Blob(
                  mime_type='application/octet-stream', data=b''
              )
          )
      await self.artifact_service.save_artifact(
          app_name=self.app_name,
          user_id=session.user_id,
          session_id=session.id,
          filename=filename,
          artifact=artifact,
      )

    return rewind_artifact_delta

  def _should_append_event(self, event: Event, is_live_call: bool) -> bool:
    """Checks if an event should be appended to the session."""
    # Don't append media (audio/video/image) response from model in live mode to session.
    # The data is appended to artifacts with a reference in file_data in the
    # event if save_live_blob is True.
    # We should append non-partial events only.For example, non-finished(partial)
    # transcription events should not be appended.
    # Function call and function response events should be appended.
    # Other control events should be appended.
    if is_live_call and contents._is_live_model_media_event_with_inline_data(
        event
    ):
      # We don't append live model media events with inline data to avoid
      # storing large blobs in the session. However, events with file_data
      # (references to artifacts) should be appended.
      return False
    return True

  def _get_output_event(
      self,
      *,
      original_event: Event,
      modified_event: Event | None,
      run_config: RunConfig | None,
  ) -> Event:
    """Returns the event that should be persisted and yielded.

    Plugins may return a replacement event that only overrides a subset of
    fields. Merge those changes onto the original event so the streamed event
    and the persisted event stay aligned without losing the original event
    identity.
    """
    if modified_event is None:
      return original_event

    _apply_run_config_custom_metadata(modified_event, run_config)
    update = {}
    for field_name in modified_event.model_fields_set:
      if field_name in {'id', 'invocation_id', 'timestamp'}:
        continue
      update[field_name] = modified_event.__dict__[field_name]
    output_event = original_event.model_copy(update=update)
    if not output_event.author:
      output_event.author = original_event.author
    return output_event

  async def _exec_with_plugin(
      self,
      invocation_context: InvocationContext,
      session: Session,
      execute_fn: Callable[[InvocationContext], AsyncGenerator[Event, None]],
      is_live_call: bool = False,
  ) -> AsyncGenerator[Event, None]:
    """Wraps execution with plugin callbacks.

    Args:
      invocation_context: The invocation context
      session: The current session (ignored, kept for backward compatibility)
      execute_fn: A callable that returns an AsyncGenerator of Events
      is_live_call: Whether this is a live call

    Yields:
      Events from the execution, including any generated by plugins
    """

    plugin_manager = invocation_context.plugin_manager

    # Step 1: Run the before_run callbacks to see if we should early exit.
    early_exit_result = await plugin_manager.run_before_run_callback(
        invocation_context=invocation_context
    )
    if isinstance(early_exit_result, types.Content):
      early_exit_event = Event(
          invocation_id=invocation_context.invocation_id,
          author='model',
          content=early_exit_result,
      )
      _apply_run_config_custom_metadata(
          early_exit_event, invocation_context.run_config
      )
      if self._should_append_event(early_exit_event, is_live_call):
        await self.session_service.append_event(
            session=invocation_context.session,
            event=early_exit_event,
        )
      yield early_exit_event
    else:
      # Step 2: Otherwise continue with normal execution
      async with aclosing(execute_fn(invocation_context)) as agen:
        async for event in agen:
          _apply_run_config_custom_metadata(
              event, invocation_context.run_config
          )
          # Step 3: Run the on_event callbacks before persisting so callback
          # changes are stored in the session and match the streamed event.
          modified_event = await plugin_manager.run_on_event_callback(
              invocation_context=invocation_context, event=event
          )
          output_event = self._get_output_event(
              original_event=event,
              modified_event=modified_event,
              run_config=invocation_context.run_config,
          )

          if is_live_call:
            # Skip partial transcriptions for Live
            if event.partial is not True and self._should_append_event(
                event, is_live_call
            ):
              logger.debug('Appending live event: %s', output_event)
              await self.session_service.append_event(
                  session=invocation_context.session, event=output_event
              )
          else:
            if event.partial is not True:
              await self.session_service.append_event(
                  session=invocation_context.session, event=output_event
              )

          yield output_event

    # Step 4: Run the after_run callbacks to perform global cleanup tasks or
    # finalizing logs and metrics data.
    # This does NOT emit any event.
    await plugin_manager.run_after_run_callback(
        invocation_context=invocation_context
    )

  async def _append_new_message_to_session(
      self,
      *,
      session: Session,
      new_message: types.Content,
      invocation_context: InvocationContext,
      save_input_blobs_as_artifacts: bool = False,
      state_delta: Optional[dict[str, Any]] = None,
  ):
    """Appends a new message to the session.

    Args:
        session: The session to append the message to.
        new_message: The new message to append.
        invocation_context: The invocation context for the message.
        save_input_blobs_as_artifacts: Whether to save input blobs as artifacts.
        state_delta: Optional state changes to apply to the session.
    """
    if not new_message.parts:
      raise ValueError('No parts in the new_message.')

    if self.artifact_service and save_input_blobs_as_artifacts:
      # Issue deprecation warning
      warnings.warn(
          "The 'save_input_blobs_as_artifacts' parameter is deprecated. Use"
          ' SaveFilesAsArtifactsPlugin instead for better control and'
          ' flexibility. See google.adk.plugins.SaveFilesAsArtifactsPlugin for'
          ' migration guidance.',
          DeprecationWarning,
          stacklevel=3,
      )
      # The runner directly saves the artifacts (if applicable) in the
      # user message and replaces the artifact data with a file name
      # placeholder.
      for i, part in enumerate(new_message.parts):
        if part.inline_data is None:
          continue
        file_name = f'artifact_{invocation_context.invocation_id}_{i}'
        await self.artifact_service.save_artifact(
            app_name=self.app_name,
            user_id=invocation_context.session.user_id,
            session_id=invocation_context.session.id,
            filename=file_name,
            artifact=part,
        )
        new_message.parts[i] = types.Part(
            text=f'Uploaded file: {file_name}. It is saved into artifacts'
        )
    # Appends only. We do not yield the event because it's not from the model.
    if state_delta:
      event = Event(
          invocation_id=invocation_context.invocation_id,
          author='user',
          actions=EventActions(state_delta=state_delta),
          content=new_message,
      )
    else:
      event = Event(
          invocation_id=invocation_context.invocation_id,
          author='user',
          content=new_message,
      )
    _apply_run_config_custom_metadata(event, invocation_context.run_config)
    invocation_context.stamp_event_branch_context(event)

    await self.session_service.append_event(
        session=invocation_context.session, event=event
    )

  async def run_live(
      self,
      *,
      user_id: Optional[str] = None,
      session_id: Optional[str] = None,
      live_request_queue: LiveRequestQueue,
      run_config: Optional[RunConfig] = None,
      session: Optional[Session] = None,
  ) -> AsyncGenerator[Event, None]:
    """Runs the agent in live mode (experimental feature).

    The `run_live` method yields a stream of `Event` objects, but not all
    yielded events are saved to the session. Here's a breakdown:

    **Events Yielded to Callers:**
    *   **Live Model Audio Events with Inline Data:** Events containing raw
        audio `Blob` data(`inline_data`).
    *   **Live Model Audio Events with File Data:** Both input and output audio
        data are aggregated into an audio file saved into artifacts. The
        reference to the file is saved in the event as `file_data`.
    *   **Usage Metadata:** Events containing token usage.
    *   **Transcription Events:** Both partial and non-partial transcription
        events are yielded.
    *   **Function Call and Response Events:** Always saved.
    *   **Other Control Events:** Most control events are saved.

    **Events Saved to the Session:**
    *   **Live Model Audio Events with File Data:** Both input and ouput audio
        data are aggregated into an audio file saved into artifacts. The
        reference to the file is saved as event in the `file_data` to session
        if RunConfig.save_live_model_audio_to_session is True.
    *   **Usage Metadata Events:** Saved to the session.
    *   **Non-Partial Transcription Events:** Non-partial transcription events
        are saved.
    *   **Function Call and Response Events:** Always saved.
    *   **Other Control Events:** Most control events are saved.

    **Events Not Saved to the Session:**
    *   **Live Model Audio Events with Inline Data:** Events containing raw
        audio `Blob` data are **not** saved to the session.

    Args:
        user_id: The user ID for the session. Required if `session` is None.
        session_id: The session ID for the session. Required if `session` is
          None.
        live_request_queue: The queue for live requests.
        run_config: The run config for the agent.
        session: The session to use. This parameter is deprecated, please use
          `user_id` and `session_id` instead.

    Yields:
        AsyncGenerator[Event, None]: An asynchronous generator that yields
        `Event`
        objects as they are produced by the agent during its live execution.

    .. warning::
        This feature is **experimental** and its API or behavior may change
        in future releases.

    .. NOTE::
        Either `session` or both `user_id` and `session_id` must be provided.
    """
    run_config = run_config or RunConfig()
    # Some native audio models requires the modality to be set. So we set it to
    # AUDIO by default.
    if run_config.response_modalities is None:
      run_config.response_modalities = [types.Modality.AUDIO]
    if session is None and (user_id is None or session_id is None):
      raise ValueError(
          'Either session or user_id and session_id must be provided.'
      )
    if live_request_queue is None:
      raise ValueError('live_request_queue is required for run_live.')
    if session is not None:
      warnings.warn(
          'The `session` parameter is deprecated. Please use `user_id` and'
          ' `session_id` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
    if not session:
      session = await self._get_or_create_session(
          user_id=user_id,
          session_id=session_id,
          get_session_config=run_config.get_session_config,
      )

    from .agents.base_agent import BaseAgent
    from .workflow._base_node import BaseNode

    if isinstance(self.agent, BaseNode) and not isinstance(
        self.agent, BaseAgent
    ):
      async with aclosing(
          self._run_node_live(
              session=session,
              live_request_queue=live_request_queue,
              run_config=run_config,
          )
      ) as agen:
        async for event in agen:
          yield event
      return
    invocation_context = self._new_invocation_context_for_live(
        session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

    root_agent = self.agent
    invocation_context.agent = self._find_agent_to_run(
        invocation_context.session, root_agent
    )

    async def execute(ctx: InvocationContext) -> AsyncGenerator[Event]:
      async with aclosing(ctx.agent.run_live(ctx)) as agen:
        async for event in agen:
          yield event

    async with aclosing(
        self._exec_with_plugin(
            invocation_context=invocation_context,
            session=invocation_context.session,
            execute_fn=execute,
            is_live_call=True,
        )
    ) as agen:
      async for event in agen:
        yield event

  def _find_agent_to_run(
      self, session: Session, root_agent: BaseAgent
  ) -> BaseAgent:
    """Finds the agent to run to continue the session.

    A qualified agent must be either of:

    - The agent that returned a function call and the last user message is a
      function response to this function call.
    - The root agent.
    - An LlmAgent who replied last and is capable to transfer to any other agent
      in the agent hierarchy.

    TODO: use wait_for_output to decide the agent to run

    Args:
        session: The session to find the agent for.
        root_agent: The root agent of the runner.

    Returns:
      The agent to run. (the active agent that should reply to the latest user
      message)
    """
    # Mesh and Workflow Agents handle their own internal routing.
    # Workflow will figure which node is interrupted and should be resumed.
    from .workflow._workflow import Workflow

    if isinstance(root_agent, Workflow):
      return root_agent

    # If the last event is a function response, should send this response to
    # the agent that returned the corresponding function call regardless the
    # type of the agent. e.g. a remote a2a agent may surface a credential
    # request as a special long-running function tool call.
    event = find_matching_function_call(session.events)
    is_resumable = (
        self.resumability_config and self.resumability_config.is_resumable
    )
    # Only route based on a past function response if resumability is enabled.
    # In non-resumable scenarios, a turn ending with function call response
    # shouldn't trap the next turn on that same agent if it's not transferable.
    # Falling through allows it to return to root.
    if event and event.author and is_resumable:
      # `find_agent` returns None when the author does not correspond to any
      # agent in the current hierarchy (e.g. the author is "user" or a stale or
      # foreign agent name carried over from a previous turn/session). Returning
      # None here would propagate to `build_node`, raising a confusing
      # "Invalid node type: <class 'NoneType'>" error. Fall through to the
      # event-scan logic below (which ultimately falls back to the root agent)
      # whenever the author cannot be resolved.
      if (resumed_agent := root_agent.find_agent(event.author)) is not None:
        return resumed_agent

    def _event_filter(event: Event) -> bool:
      """Filters out user-authored events and agent state change events."""
      if event.author == 'user':
        return False
      if event.actions.agent_state is not None or event.actions.end_of_agent:
        return False
      return True

    for event in filter(_event_filter, reversed(session.events)):
      if event.author == root_agent.name:
        # Found root agent.
        return root_agent
      if not (agent := root_agent.find_sub_agent(event.author)):
        # Agent not found, continue looking.
        logger.warning(
            'Event from an unknown agent: %s, event id: %s',
            event.author,
            event.id,
        )
        continue
      transferable = self._is_transferable_across_agent_tree(agent)
      if transferable:
        return agent
    # Falls back to root agent if no suitable agents are found in the session.
    return root_agent

  def _is_transferable_across_agent_tree(self, agent_to_run: BaseAgent) -> bool:
    """Whether the agent to run can transfer to any other agent in the agent tree.

    This typically means all agent_to_run's ancestor can transfer to their
    parent_agent all the way to the root_agent.

    Args:
        agent_to_run: The agent to check for transferability.

    Returns:
        True if the agent can transfer, False otherwise.
    """
    agent = agent_to_run
    while agent:
      if not hasattr(agent, 'disallow_transfer_to_parent'):
        # Only agents with transfer capability can transfer.
        return False
      if agent.disallow_transfer_to_parent:
        return False
      agent = agent.parent_agent
    return True

  async def run_debug(
      self,
      user_messages: str | list[str],
      *,
      user_id: str = 'debug_user_id',
      session_id: str = 'debug_session_id',
      run_config: RunConfig | None = None,
      quiet: bool = False,
      verbose: bool = False,
  ) -> list[Event]:
    """Debug helper for quick agent experimentation and testing.

    This convenience method is designed for developers getting started with ADK
    who want to quickly test agents without dealing with session management,
    content formatting, or event streaming. It automatically handles common
    boilerplate while hiding complexity.

    IMPORTANT: This is for debugging and experimentation only. For production
    use, please use the standard run_async() method which provides full control
    over session management, event streaming, and error handling.

    Args:
        user_messages: Message(s) to send to the agent. Can be: - Single string:
          "What is 2+2?" - List of strings: ["Hello!", "What's my name?"]
        user_id: User identifier. Defaults to "debug_user_id".
        session_id: Session identifier for conversation persistence. Defaults to
          "debug_session_id". Reuse the same ID to continue a conversation.
        run_config: Optional configuration for the agent execution.
        quiet: If True, suppresses console output. Defaults to False (output
          shown).
        verbose: If True, shows detailed tool calls and responses. Defaults to
          False for cleaner output showing only final agent responses.

    Returns:
        list[Event]: All events from all messages.

    Raises:
        ValueError: If session creation/retrieval fails.

    Examples:
        Quick debugging:
        >>> runner = InMemoryRunner(agent=my_agent)
        >>> await runner.run_debug("What is 2+2?")

        Multiple queries in conversation:
        >>> await runner.run_debug(["Hello!", "What's my name?"])

        Continue a debug session:
        >>> await runner.run_debug("What did we discuss?")  # Continues default
        session

        Separate debug sessions:
        >>> await runner.run_debug("Hi", user_id="alice", session_id="debug1")
        >>> await runner.run_debug("Hi", user_id="bob", session_id="debug2")

        Capture events for inspection:
        >>> events = await runner.run_debug("Analyze this")
        >>> for event in events:
        ...     inspect_event(event)

    Note:
        For production applications requiring:
        - Custom session/memory services (Spanner, Cloud SQL, etc.)
        - Fine-grained event processing and streaming
        - Error recovery and resumability
        - Performance optimization
        Please use run_async() with proper configuration.
    """
    run_config = run_config or RunConfig()
    session = await self.session_service.get_session(
        app_name=self.app_name,
        user_id=user_id,
        session_id=session_id,
        config=run_config.get_session_config,
    )
    if not session:
      session = await self.session_service.create_session(
          app_name=self.app_name, user_id=user_id, session_id=session_id
      )
      if not quiet:
        logger.info('Created new session: %s', session_id)
    elif not quiet:
      logger.info('Continue session: %s', session_id)

    collected_events: list[Event] = []

    if isinstance(user_messages, str):
      user_messages = [user_messages]

    for message in user_messages:
      if not quiet:
        logger.info('User > %s', message)

      async with aclosing(
          self.run_async(
              user_id=user_id,
              session_id=session.id,
              new_message=types.UserContent(parts=[types.Part(text=message)]),
              run_config=run_config,
          )
      ) as agen:
        async for event in agen:
          if not quiet:
            print_event(event, verbose=verbose)

          collected_events.append(event)

    return collected_events

  async def _setup_context_for_new_invocation(
      self,
      *,
      session: Session,
      new_message: types.Content,
      run_config: RunConfig,
      state_delta: Optional[dict[str, Any]],
      invocation_id: Optional[str] = None,
  ) -> InvocationContext:
    """Sets up the context for a new invocation.

    Args:
      session: The session to set up the invocation context for.
      new_message: The new message to process and append to the session.
      run_config: The run config of the agent.
      state_delta: Optional state changes to apply to the session.
      invocation_id: Optional invocation identifier.

    Returns:
      The invocation context for the new invocation.
    """
    # Step 1: Create invocation context in memory.
    invocation_context = self._new_invocation_context(
        session,
        new_message=new_message,
        run_config=run_config,
        invocation_id=invocation_id,
    )
    # Step 2: Handle new message, by running callbacks and appending to
    # session.
    await self._handle_new_message(
        session=invocation_context.session,
        new_message=new_message,
        invocation_context=invocation_context,
        run_config=run_config,
        state_delta=state_delta,
    )
    # Step 3: Set agent to run for the invocation.
    invocation_context.agent = self._find_agent_to_run(
        invocation_context.session, self.agent
    )
    return invocation_context

  async def _setup_context_for_resumed_invocation(
      self,
      *,
      session: Session,
      new_message: Optional[types.Content],
      invocation_id: Optional[str],
      run_config: RunConfig,
      state_delta: Optional[dict[str, Any]],
  ) -> InvocationContext:
    """Sets up the context for a resumed invocation.

    Args:
      session: The session to set up the invocation context for.
      new_message: The new message to process and append to the session.
      invocation_id: The invocation id to resume.
      run_config: The run config of the agent.
      state_delta: Optional state changes to apply to the session.

    Returns:
      The invocation context for the resumed invocation.

    Raises:
      ValueError: If the session has no events to resume; If no user message is
        available for resuming the invocation; Or if the app is not resumable.
    """
    if not session.events:
      raise ValueError(f'Session {session.id} has no events to resume.')

    # Step 1: Maybe retrieve a previous user message for the invocation.
    user_message = new_message or self._find_user_message_for_invocation(
        session.events, invocation_id
    )
    if not user_message:
      raise ValueError(
          f'No user message available for resuming invocation: {invocation_id}'
      )
    # Step 2: Create invocation context.
    invocation_context = self._new_invocation_context(
        session,
        new_message=user_message,
        run_config=run_config,
        invocation_id=invocation_id,
    )
    # Step 3: Maybe handle new message.
    if new_message:
      await self._handle_new_message(
          session=invocation_context.session,
          new_message=user_message,
          invocation_context=invocation_context,
          run_config=run_config,
          state_delta=state_delta,
      )
    # Step 4: Populate agent states for the current invocation.
    invocation_context.populate_invocation_agent_states()
    # Step 5: Set agent to run for the invocation.
    #
    # If the root agent is not found in end_of_agents, it means the invocation
    # started from a sub-agent and paused on a sub-agent.
    # We should find the appropriate agent to run to continue the invocation.
    if self.agent.name not in invocation_context.end_of_agents:
      invocation_context.agent = self._find_agent_to_run(
          invocation_context.session, self.agent
      )
    return invocation_context

  def _find_user_message_for_invocation(
      self, events: list[Event], invocation_id: str
  ) -> Optional[types.Content]:
    """Finds the user message that started a specific invocation."""
    for event in events:
      if (
          event.invocation_id == invocation_id
          and event.author == 'user'
          and event.content
          and event.content.parts
          and event.content.parts[0].text
      ):
        return event.content
    return None

  def _create_invocation_context(self, **kwargs) -> InvocationContext:
    """Creates an InvocationContext instance."""
    return InvocationContext(**kwargs)

  def _new_invocation_context(
      self,
      session: Session,
      *,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      live_request_queue: Optional[LiveRequestQueue] = None,
      run_config: Optional[RunConfig] = None,
  ) -> InvocationContext:
    """Creates a new invocation context.

    Args:
        session: The session for the context.
        invocation_id: The invocation id for the context.
        new_message: The new message for the context.
        live_request_queue: The live request queue for the context.
        run_config: The run config for the context.

    Returns:
        The new invocation context.
    """
    run_config = run_config or RunConfig()
    invocation_id = invocation_id or new_invocation_context_id()

    if run_config.support_cfc and hasattr(self.agent, 'canonical_model'):
      model_name = self.agent.canonical_model.model
      if not model_name.startswith('gemini-2'):
        raise ValueError(
            f'CFC is not supported for model: {model_name} in agent:'
            f' {self.agent.name}'
        )
      if not isinstance(self.agent.code_executor, BuiltInCodeExecutor):
        self.agent.code_executor = BuiltInCodeExecutor()

    return self._create_invocation_context(
        artifact_service=self.artifact_service,
        session_service=self.session_service,
        memory_service=self.memory_service,
        credential_service=self.credential_service,
        plugin_manager=self.plugin_manager,
        context_cache_config=self.context_cache_config,
        events_compaction_config=(
            self.app.events_compaction_config if self.app else None
        ),
        invocation_id=invocation_id,
        agent=self.agent if isinstance(self.agent, BaseAgent) else None,
        session=session,
        user_content=new_message,
        live_request_queue=live_request_queue,
        run_config=run_config,
        resumability_config=self.resumability_config,
    )

  def _new_invocation_context_for_live(
      self,
      session: Session,
      *,
      live_request_queue: LiveRequestQueue,
      run_config: Optional[RunConfig] = None,
  ) -> InvocationContext:
    """Creates a new invocation context for live multi-agent."""
    run_config = run_config or RunConfig()

    # For live multi-agents system, we need model's text transcription as
    # context for the transferred agent.
    if hasattr(self.agent, 'sub_agents') and self.agent.sub_agents:
      if types.Modality.AUDIO in run_config.response_modalities:
        if not run_config.output_audio_transcription:
          run_config.output_audio_transcription = (
              types.AudioTranscriptionConfig()
          )
      if not run_config.input_audio_transcription:
        run_config.input_audio_transcription = types.AudioTranscriptionConfig()
    return self._new_invocation_context(
        session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

  async def _handle_new_message(
      self,
      *,
      session: Session,
      new_message: types.Content,
      invocation_context: InvocationContext,
      run_config: RunConfig,
      state_delta: Optional[dict[str, Any]],
  ) -> None:
    """Handles a new message by running callbacks and appending to session.

    Args:
      session: The session of the new message.
      new_message: The new message to process and append to the session.
      invocation_context: The invocation context to use for the message
        handling.
      run_config: The run config of the agent.
      state_delta: Optional state changes to apply to the session.
    """
    modified_user_message = (
        await invocation_context.plugin_manager.run_on_user_message_callback(
            invocation_context=invocation_context, user_message=new_message
        )
    )
    if modified_user_message is not None:
      new_message = modified_user_message
      invocation_context.user_content = new_message

    if new_message:
      deprecated_save_blobs = False
      if 'save_input_blobs_as_artifacts' in run_config.model_fields_set:
        deprecated_save_blobs = run_config.save_input_blobs_as_artifacts
      await self._append_new_message_to_session(
          session=invocation_context.session,
          new_message=new_message,
          invocation_context=invocation_context,
          save_input_blobs_as_artifacts=deprecated_save_blobs,
          state_delta=state_delta,
      )

  def _collect_toolset(self, agent: BaseAgent) -> set[BaseToolset]:
    toolsets = set()
    if hasattr(agent, 'tools'):
      for tool_union in agent.tools:
        if isinstance(tool_union, BaseToolset):
          toolsets.add(tool_union)
    if hasattr(agent, 'sub_agents'):
      for sub_agent in agent.sub_agents:
        toolsets.update(self._collect_toolset(sub_agent))
    return toolsets

  async def _cleanup_toolsets(self, toolsets_to_close: set[BaseToolset]):
    """Clean up toolsets with proper task context management."""
    if not toolsets_to_close:
      return

    # This maintains the same task context throughout cleanup
    for toolset in toolsets_to_close:
      try:
        logger.info('Closing toolset: %s', type(toolset).__name__)
        # Use asyncio.wait_for to add timeout protection
        await asyncio.wait_for(toolset.close(), timeout=10.0)
        logger.info('Successfully closed toolset: %s', type(toolset).__name__)
      except asyncio.TimeoutError:
        logger.warning('Toolset %s cleanup timed out', type(toolset).__name__)
      except asyncio.CancelledError as e:
        # Handle cancel scope issues in Python 3.10 and 3.11 with anyio
        #
        # Root cause: MCP library uses anyio.CancelScope() in RequestResponder.__enter__()
        # and __exit__() methods. When asyncio.wait_for() creates a new task for cleanup,
        # the cancel scope is entered in one task context but exited in another.
        #
        # Python 3.12+ fixes: Enhanced task context management (Task.get_context()),
        # improved context propagation across task boundaries, and better cancellation
        # handling prevent the cross-task cancel scope violation.
        logger.warning(
            'Toolset %s cleanup cancelled: %s', type(toolset).__name__, e
        )
      except Exception as e:
        logger.error('Error closing toolset %s: %s', type(toolset).__name__, e)

  async def close(self):
    """Closes the runner."""
    logger.info('Closing runner...')
    # Close Toolsets
    if self.agent is not None:
      await self._cleanup_toolsets(self._collect_toolset(self.agent))

    # Close Plugins
    if self.plugin_manager:
      await self.plugin_manager.close()

    # Close Session Service
    if self.session_service:
      await self.session_service.flush()

    logger.info('Runner closed.')

  if sys.version_info < (3, 11):
    Self = 'Runner'  # pylint: disable=invalid-name
  else:
    from typing import Self  # pylint: disable=g-import-not-at-top

  async def __aenter__(self) -> Self:
    """Async context manager entry."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit."""
    await self.close()
    return False  # Don't suppress exceptions from the async with block


class InMemoryRunner(Runner):
  """An in-memory Runner for testing and development.

  This runner uses in-memory implementations for artifact, session, and memory
  services, providing a lightweight and self-contained environment for agent
  execution.

  Attributes:
      agent: The root agent to run.
      app_name: The application name of the runner. Defaults to
        'InMemoryRunner'.
  """

  def __init__(
      self,
      agent: Optional[BaseAgent] = None,
      *,
      node: Any = None,
      app_name: Optional[str] = None,
      plugins: Optional[list[BasePlugin]] = None,
      app: Optional[App] = None,
      plugin_close_timeout: float = 5.0,
  ):
    """Initializes the InMemoryRunner.

    Args:
        agent: The root agent to run.
        node: The root node to run.
        app_name: The application name of the runner. Defaults to
          'InMemoryRunner'.
        plugins: Optional list of plugins for the runner.
        app: Optional App instance.
        plugin_close_timeout: The timeout in seconds for plugin close methods.
    """
    from .artifacts.in_memory_artifact_service import InMemoryArtifactService
    from .memory.in_memory_memory_service import InMemoryMemoryService
    from .sessions.in_memory_session_service import InMemorySessionService

    if app is None and app_name is None:
      app_name = 'InMemoryRunner'
    super().__init__(
        app_name=app_name,
        agent=agent,
        node=node,
        artifact_service=InMemoryArtifactService(),
        plugins=plugins,
        app=app,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        plugin_close_timeout=plugin_close_timeout,
    )
