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

import asyncio
import inspect
import logging
from pathlib import Path
import queue
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
import warnings

from google.adk.apps.compaction import _run_compaction_for_sliding_window
from google.adk.artifacts import artifact_util
from google.genai import types

from .agents.active_streaming_tool import ActiveStreamingTool
from .agents.base_agent import BaseAgent
from .agents.base_agent import BaseAgentState
from .agents.context_cache_config import ContextCacheConfig
from .agents.invocation_context import InvocationContext
from .agents.invocation_context import new_invocation_context_id
from .agents.live_request_queue import LiveRequestQueue
from .agents.llm_agent import LlmAgent
from .agents.run_config import RunConfig
from .apps.app import App
from .apps.app import ResumabilityConfig
from .artifacts.base_artifact_service import BaseArtifactService
from .artifacts.in_memory_artifact_service import InMemoryArtifactService
from .auth.credential_service.base_credential_service import BaseCredentialService
from .code_executors.built_in_code_executor import BuiltInCodeExecutor
from .events.event import Event
from .events.event import EventActions
from .flows.llm_flows import contents
from .flows.llm_flows.functions import find_matching_function_call
from .memory.base_memory_service import BaseMemoryService
from .memory.in_memory_memory_service import InMemoryMemoryService
from .platform.thread import create_thread
from .plugins.base_plugin import BasePlugin
from .plugins.plugin_manager import PluginManager
from .sessions.base_session_service import BaseSessionService
from .sessions.in_memory_session_service import InMemorySessionService
from .sessions.session import Session
from .telemetry.tracing import tracer
from .tools.base_toolset import BaseToolset
from .utils._debug_output import print_event
from .utils.context_utils import Aclosing

logger = logging.getLogger('google_adk.' + __name__)


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
  agent: BaseAgent
  """The root agent to run."""
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
      plugins: Optional[List[BasePlugin]] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      session_service: BaseSessionService,
      memory_service: Optional[BaseMemoryService] = None,
      credential_service: Optional[BaseCredentialService] = None,
  ):
    """Initializes the Runner.

    Developers should provide either an `app` instance or both `app_name` and
    `agent`. Providing a mix of `app` and `app_name`/`agent` will result in a
    `ValueError`. Providing `app` is the recommended way to create a runner.

    Args:
        app: An optional `App` instance. If provided, `app_name` and `agent`
          should not be specified.
        app_name: The application name of the runner. Required if `app` is not
          provided.
        agent: The root agent to run. Required if `app` is not provided.
        plugins: Deprecated. A list of plugins for the runner. Please use the
          `app` argument to provide plugins instead.
        artifact_service: The artifact service for the runner.
        session_service: The session service for the runner.
        memory_service: The memory service for the runner.
        credential_service: The credential service for the runner.

    Raises:
        ValueError: If `app` is provided along with `app_name` or `plugins`, or
          if `app` is not provided but either `app_name` or `agent` is missing.
    """
    self.app = app
    (
        self.app_name,
        self.agent,
        self.context_cache_config,
        self.resumability_config,
        plugins,
    ) = self._validate_runner_params(app, app_name, agent, plugins)
    self.artifact_service = artifact_service
    self.session_service = session_service
    self.memory_service = memory_service
    self.credential_service = credential_service
    self.plugin_manager = PluginManager(plugins=plugins)
    (
        self._agent_origin_app_name,
        self._agent_origin_dir,
    ) = self._infer_agent_origin(self.agent)
    self._app_name_alignment_hint: Optional[str] = None
    self._enforce_app_name_alignment()

  def _validate_runner_params(
      self,
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
    """Validates and extracts runner parameters.

    Args:
        app: An optional `App` instance.
        app_name: The application name of the runner.
        agent: The root agent to run.
        plugins: A list of plugins for the runner.

    Returns:
        A tuple containing (app_name, agent, context_cache_config,
        resumability_config, plugins).

    Raises:
        ValueError: If parameters are invalid.
    """
    if app:
      if app_name:
        raise ValueError(
            'When app is provided, app_name should not be provided.'
        )
      if agent:
        raise ValueError('When app is provided, agent should not be provided.')
      if plugins:
        raise ValueError(
            'When app is provided, plugins should not be provided and should be'
            ' provided in the app instead.'
        )
      app_name = app.name
      agent = app.root_agent
      plugins = app.plugins
      context_cache_config = app.context_cache_config
      resumability_config = app.resumability_config
    elif not app_name or not agent:
      raise ValueError(
          'Either app or both app_name and agent must be provided.'
      )
    else:
      context_cache_config = None
      resumability_config = None

    if plugins:
      warnings.warn(
          'The `plugins` argument is deprecated. Please use the `app` argument'
          ' to provide plugins instead.',
          DeprecationWarning,
      )
    return app_name, agent, context_cache_config, resumability_config, plugins

  def _infer_agent_origin(
      self, agent: BaseAgent
  ) -> tuple[Optional[str], Optional[Path]]:
    module = inspect.getmodule(agent.__class__)
    if not module:
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

  def _format_session_not_found_message(self, session_id: str) -> str:
    message = f'Session not found: {session_id}'
    if not self._app_name_alignment_hint:
      return message
    return (
        f'{message}. {self._app_name_alignment_hint} '
        'The mismatch prevents the runner from locating the session.'
    )

  def run(
      self,
      *,
      user_id: str,
      session_id: str,
      new_message: types.Content,
      run_config: Optional[RunConfig] = None,
  ) -> Generator[Event, None, None]:
    """Runs the agent.

    NOTE:
      This sync interface is only for local testing and convenience purpose.
      Consider using `run_async` for production usage.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      new_message: A new message to append to the session.
      run_config: The run config for the agent.

    Yields:
      The events generated by the agent.
    """
    run_config = run_config or RunConfig()
    event_queue = queue.Queue()

    async def _invoke_run_async():
      try:
        async with Aclosing(
            self.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
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
  ) -> AsyncGenerator[Event, None]:
    """Main entry method to run the agent in this runner.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      invocation_id: The invocation ID of the session, set this to resume an
        interrupted invocation.
      new_message: A new message to append to the session.
      state_delta: Optional state changes to apply to the session.
      run_config: The run config for the agent.

    Yields:
      The events generated by the agent.

    Raises:
      ValueError: If the session is not found; If both invocation_id and
        new_message are None.
    """
    run_config = run_config or RunConfig()

    if new_message and not new_message.role:
      new_message.role = 'user'

    async def _run_with_trace(
        new_message: Optional[types.Content] = None,
        invocation_id: Optional[str] = None,
    ) -> AsyncGenerator[Event, None]:
      with tracer.start_as_current_span('invocation'):
        session = await self.session_service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )
        if not session:
          message = self._format_session_not_found_message(session_id)
          raise ValueError(message)
        if not invocation_id and not new_message:
          raise ValueError('Both invocation_id and new_message are None.')

        if invocation_id:
          if (
              not self.resumability_config
              or not self.resumability_config.is_resumable
          ):
            raise ValueError(
                f'invocation_id: {invocation_id} is provided but the app is not'
                ' resumable.'
            )
          invocation_context = await self._setup_context_for_resumed_invocation(
              session=session,
              new_message=new_message,
              invocation_id=invocation_id,
              run_config=run_config,
              state_delta=state_delta,
          )
          if invocation_context.end_of_agents.get(
              invocation_context.agent.name
          ):
            # Directly return if the current agent in invocation context is
            # already final.
            return
        else:
          invocation_context = await self._setup_context_for_new_invocation(
              session=session,
              new_message=new_message,  # new_message is not None.
              run_config=run_config,
              state_delta=state_delta,
          )

        async def execute(ctx: InvocationContext) -> AsyncGenerator[Event]:
          async with Aclosing(ctx.agent.run_async(ctx)) as agen:
            async for event in agen:
              yield event

        async with Aclosing(
            self._exec_with_plugin(
                invocation_context=invocation_context,
                session=session,
                execute_fn=execute,
                is_live_call=False,
            )
        ) as agen:
          async for event in agen:
            yield event
        # Run compaction after all events are yielded from the agent.
        # (We don't compact in the middle of an invocation, we only compact at the end of an invocation.)
        if self.app and self.app.events_compaction_config:
          logger.info('Running event compactor.')
          # Run compaction in a separate task to avoid blocking the main thread.
          # So the users can still finish the event loop from the agent while the
          # compaction is running.
          asyncio.create_task(
              _run_compaction_for_sliding_window(
                  self.app, session, self.session_service
              )
          )

    async with Aclosing(_run_with_trace(new_message, invocation_id)) as agen:
      async for event in agen:
        yield event

  async def rewind_async(
      self,
      *,
      user_id: str,
      session_id: str,
      rewind_before_invocation_id: str,
  ) -> None:
    """Rewinds the session to before the specified invocation."""
    session = await self.session_service.get_session(
        app_name=self.app_name, user_id=user_id, session_id=session_id
    )
    if not session:
      raise ValueError(f'Session not found: {session_id}')

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
        # rewind point.
        artifact_uri = artifact_util.get_artifact_uri(
            app_name=self.app_name,
            user_id=session.user_id,
            session_id=session.id,
            filename=filename,
            version=vt,
        )
        artifact = types.Part(file_data=types.FileData(file_uri=artifact_uri))
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
    # Don't append audio response from model in live mode to session.
    # The data is appended to artifacts with a reference in file_data in the
    # event.
    # We should append non-partial events only.For example, non-finished(partial)
    # transcription events should not be appended.
    # Function call and function response events should be appended.
    # Other control events should be appended.
    if is_live_call and contents._is_live_model_audio_event(event):
      return False
    return True

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
      session: The current session
      execute_fn: A callable that returns an AsyncGenerator of Events

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
      if self._should_append_event(early_exit_event, is_live_call):
        await self.session_service.append_event(
            session=session,
            event=early_exit_event,
        )
      yield early_exit_event
    else:
      # Step 2: Otherwise continue with normal execution
      async with Aclosing(execute_fn(invocation_context)) as agen:
        async for event in agen:
          if not event.partial:
            if self._should_append_event(event, is_live_call):
              await self.session_service.append_event(
                  session=session, event=event
              )
          # Step 3: Run the on_event callbacks to optionally modify the event.
          modified_event = await plugin_manager.run_on_event_callback(
              invocation_context=invocation_context, event=event
          )
          yield (modified_event if modified_event else event)

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
            user_id=session.user_id,
            session_id=session.id,
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
    # If new_message is a function response, find the matching function call
    # and use its branch as the new event's branch.
    if function_call := invocation_context._find_matching_function_call(event):
      event.branch = function_call.branch

    await self.session_service.append_event(session=session, event=event)

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
      run_config.response_modalities = ['AUDIO']
    if session is None and (user_id is None or session_id is None):
      raise ValueError(
          'Either session or user_id and session_id must be provided.'
      )
    if session is not None:
      warnings.warn(
          'The `session` parameter is deprecated. Please use `user_id` and'
          ' `session_id` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
    if not session:
      session = await self.session_service.get_session(
          app_name=self.app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        raise ValueError(f'Session not found: {session_id}')
    invocation_context = self._new_invocation_context_for_live(
        session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

    root_agent = self.agent
    invocation_context.agent = self._find_agent_to_run(session, root_agent)

    # Pre-processing for live streaming tools
    # Inspect the tool's parameters to find if it uses LiveRequestQueue
    invocation_context.active_streaming_tools = {}
    # TODO(hangfei): switch to use canonical_tools.
    # for shell agents, there is no tools associated with it so we should skip.
    if hasattr(invocation_context.agent, 'tools'):
      import inspect

      for tool in invocation_context.agent.tools:
        # We use `inspect.signature()` to examine the tool's underlying function (`tool.func`).
        # This approach is deliberately chosen over `typing.get_type_hints()` for robustness.
        #
        # The Problem with `get_type_hints()`:
        # `get_type_hints()` attempts to resolve forward-referenced (string-based) type
        # annotations. This resolution can easily fail with a `NameError` (e.g., "Union not found")
        # if the type isn't available in the scope where `get_type_hints()` is called.
        # This is a common and brittle issue in framework code that inspects functions
        # defined in separate user modules.
        #
        # Why `inspect.signature()` is Better Here:
        # `inspect.signature()` does NOT resolve the annotations; it retrieves the raw
        # annotation object as it was defined on the function. This allows us to
        # perform a direct and reliable identity check (`param.annotation is LiveRequestQueue`)
        # without risking a `NameError`.
        callable_to_inspect = tool.func if hasattr(tool, 'func') else tool
        # Ensure the target is actually callable before inspecting to avoid errors.
        if not callable(callable_to_inspect):
          continue
        for param in inspect.signature(callable_to_inspect).parameters.values():
          if param.annotation is LiveRequestQueue:
            if not invocation_context.active_streaming_tools:
              invocation_context.active_streaming_tools = {}
            active_streaming_tool = ActiveStreamingTool(
                stream=LiveRequestQueue()
            )
            invocation_context.active_streaming_tools[tool.__name__] = (
                active_streaming_tool
            )

    async def execute(ctx: InvocationContext) -> AsyncGenerator[Event]:
      async with Aclosing(ctx.agent.run_live(ctx)) as agen:
        async for event in agen:
          yield event

    async with Aclosing(
        self._exec_with_plugin(
            invocation_context=invocation_context,
            session=session,
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

    Args:
        session: The session to find the agent for.
        root_agent: The root agent of the runner.

    Returns:
      The agent to run. (the active agent that should reply to the latest user
      message)
    """
    # If the last event is a function response, should send this response to
    # the agent that returned the corresponding function call regardless the
    # type of the agent. e.g. a remote a2a agent may surface a credential
    # request as a special long running function tool call.
    event = find_matching_function_call(session.events)
    if event and event.author:
      return root_agent.find_agent(event.author)

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
      if self._is_transferable_across_agent_tree(agent):
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
      if not isinstance(agent, LlmAgent):
        # Only LLM-based Agent can provide agent transfer capability.
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
        user_messages: Message(s) to send to the agent. Can be:
            - Single string: "What is 2+2?"
            - List of strings: ["Hello!", "What's my name?"]
        user_id: User identifier. Defaults to "debug_user_id".
        session_id: Session identifier for conversation persistence.
            Defaults to "debug_session_id". Reuse the same ID to continue a conversation.
        run_config: Optional configuration for the agent execution.
        quiet: If True, suppresses console output. Defaults to False (output shown).
        verbose: If True, shows detailed tool calls and responses. Defaults to False
            for cleaner output showing only final agent responses.

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
        >>> await runner.run_debug("What did we discuss?")  # Continues default session

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
    session = await self.session_service.get_session(
        app_name=self.app_name, user_id=user_id, session_id=session_id
    )
    if not session:
      session = await self.session_service.create_session(
          app_name=self.app_name, user_id=user_id, session_id=session_id
      )
      if not quiet:
        print(f'\n ### Created new session: {session_id}')
    elif not quiet:
      print(f'\n ### Continue session: {session_id}')

    collected_events: list[Event] = []

    if isinstance(user_messages, str):
      user_messages = [user_messages]

    for message in user_messages:
      if not quiet:
        print(f'\nUser > {message}')

      async for event in self.run_async(
          user_id=user_id,
          session_id=session.id,
          new_message=types.UserContent(parts=[types.Part(text=message)]),
          run_config=run_config,
      ):
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
  ) -> InvocationContext:
    """Sets up the context for a new invocation.

    Args:
      session: The session to set up the invocation context for.
      new_message: The new message to process and append to the session.
      run_config: The run config of the agent.
      state_delta: Optional state changes to apply to the session.

    Returns:
      The invocation context for the new invocation.
    """
    # Step 1: Create invocation context in memory.
    invocation_context = self._new_invocation_context(
        session,
        new_message=new_message,
        run_config=run_config,
    )
    # Step 2: Handle new message, by running callbacks and appending to
    # session.
    await self._handle_new_message(
        session=session,
        new_message=new_message,
        invocation_context=invocation_context,
        run_config=run_config,
        state_delta=state_delta,
    )
    # Step 3: Set agent to run for the invocation.
    invocation_context.agent = self._find_agent_to_run(session, self.agent)
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
          session=session,
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
      invocation_context.agent = self._find_agent_to_run(session, self.agent)
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

    if run_config.support_cfc and isinstance(self.agent, LlmAgent):
      model_name = self.agent.canonical_model.model
      if not model_name.startswith('gemini-2'):
        raise ValueError(
            f'CFC is not supported for model: {model_name} in agent:'
            f' {self.agent.name}'
        )
      if not isinstance(self.agent.code_executor, BuiltInCodeExecutor):
        self.agent.code_executor = BuiltInCodeExecutor()

    return InvocationContext(
        artifact_service=self.artifact_service,
        session_service=self.session_service,
        memory_service=self.memory_service,
        credential_service=self.credential_service,
        plugin_manager=self.plugin_manager,
        context_cache_config=self.context_cache_config,
        invocation_id=invocation_id,
        agent=self.agent,
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
      live_request_queue: Optional[LiveRequestQueue] = None,
      run_config: Optional[RunConfig] = None,
  ) -> InvocationContext:
    """Creates a new invocation context for live multi-agent."""
    run_config = run_config or RunConfig()

    # For live multi-agent, we need model's text transcription as context for
    # next agent.
    if self.agent.sub_agents and live_request_queue:
      if not run_config.response_modalities:
        # default
        run_config.response_modalities = ['AUDIO']
        if not run_config.output_audio_transcription:
          run_config.output_audio_transcription = (
              types.AudioTranscriptionConfig()
          )
      elif 'TEXT' not in run_config.response_modalities:
        if not run_config.output_audio_transcription:
          run_config.output_audio_transcription = (
              types.AudioTranscriptionConfig()
          )
      if not run_config.input_audio_transcription:
        # need this input transcription for agent transferring in live mode.
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
      await self._append_new_message_to_session(
          session=session,
          new_message=new_message,
          invocation_context=invocation_context,
          save_input_blobs_as_artifacts=run_config.save_input_blobs_as_artifacts,
          state_delta=state_delta,
      )

  def _collect_toolset(self, agent: BaseAgent) -> set[BaseToolset]:
    toolsets = set()
    if isinstance(agent, LlmAgent):
      for tool_union in agent.tools:
        if isinstance(tool_union, BaseToolset):
          toolsets.add(tool_union)
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
    await self._cleanup_toolsets(self._collect_toolset(self.agent))

  async def __aenter__(self):
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
      app_name: Optional[str] = None,
      plugins: Optional[list[BasePlugin]] = None,
      app: Optional[App] = None,
  ):
    """Initializes the InMemoryRunner.

    Args:
        agent: The root agent to run.
        app_name: The application name of the runner. Defaults to
          'InMemoryRunner'.
    """
    if app is None and app_name is None:
      app_name = 'InMemoryRunner'
    super().__init__(
        app_name=app_name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        plugins=plugins,
        app=app,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
