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
import logging
import queue
import time
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
import uuid
import warnings

from google.genai import types

from .agents.active_streaming_tool import ActiveStreamingTool
from .agents.base_agent import BaseAgent
from .agents.invocation_context import InvocationContext
from .agents.invocation_context import new_invocation_context_id
from .agents.live_request_queue import LiveRequestQueue
from .agents.llm_agent import LlmAgent
from .agents.run_config import RunConfig
from .artifacts.base_artifact_service import BaseArtifactService
from .artifacts.in_memory_artifact_service import InMemoryArtifactService
from .auth.credential_service.base_credential_service import BaseCredentialService
from .code_executors.built_in_code_executor import BuiltInCodeExecutor
from .events.event import Event
from .events.event import EventActions
from .flows.llm_flows.functions import find_matching_function_call
from .memory.base_memory_service import BaseMemoryService
from .memory.in_memory_memory_service import InMemoryMemoryService
from .platform.thread import create_thread
from .plugins.base_plugin import BasePlugin
from .plugins.plugin_manager import PluginManager
from .sessions.base_session_service import BaseSessionService
from .sessions.in_memory_session_service import InMemorySessionService
from .sessions.session import Session
from .telemetry import tracer
from .tools.base_toolset import BaseToolset
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

  def __init__(
      self,
      *,
      app_name: str,
      agent: BaseAgent,
      plugins: Optional[List[BasePlugin]] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      session_service: BaseSessionService,
      memory_service: Optional[BaseMemoryService] = None,
      credential_service: Optional[BaseCredentialService] = None,
  ):
    """Initializes the Runner.

    Args:
        app_name: The application name of the runner.
        agent: The root agent to run.
        plugins: A list of plugins for the runner.
        artifact_service: The artifact service for the runner.
        session_service: The session service for the runner.
        memory_service: The memory service for the runner.
        credential_service: The credential service for the runner.
    """
    self.app_name = app_name
    self.agent = agent
    self.artifact_service = artifact_service
    self.session_service = session_service
    self.memory_service = memory_service
    self.credential_service = credential_service
    self.plugin_manager = PluginManager(plugins=plugins)

  def run(
      self,
      *,
      user_id: str,
      session_id: str,
      new_message: types.Content,
      run_config: RunConfig = RunConfig(),
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
      new_message: types.Content,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: RunConfig = RunConfig(),
  ) -> AsyncGenerator[Event, None]:
    """Main entry method to run the agent in this runner.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      new_message: A new message to append to the session.
      run_config: The run config for the agent.

    Yields:
      The events generated by the agent.
    """

    async def _run_with_trace(
        new_message: types.Content,
    ) -> AsyncGenerator[Event, None]:
      with tracer.start_as_current_span('invocation'):
        session = await self.session_service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )
        if not session:
          raise ValueError(f'Session not found: {session_id}')

        invocation_context = self._new_invocation_context(
            session,
            new_message=new_message,
            run_config=run_config,
        )
        root_agent = self.agent

        # Modify user message before execution.
        modified_user_message = await invocation_context.plugin_manager.run_on_user_message_callback(
            invocation_context=invocation_context, user_message=new_message
        )
        if modified_user_message is not None:
          new_message = modified_user_message

        if new_message:
          await self._append_new_message_to_session(
              session,
              new_message,
              invocation_context,
              run_config.save_input_blobs_as_artifacts,
              state_delta,
          )

        invocation_context.agent = self._find_agent_to_run(session, root_agent)

        async def execute(ctx: InvocationContext) -> AsyncGenerator[Event]:
          async with Aclosing(ctx.agent.run_async(ctx)) as agen:
            async for event in agen:
              yield event

        async with Aclosing(
            self._exec_with_plugin(invocation_context, session, execute)
        ) as agen:
          async for event in agen:
            yield event

    async with Aclosing(_run_with_trace(new_message)) as agen:
      async for event in agen:
        yield event

  async def _exec_with_plugin(
      self,
      invocation_context: InvocationContext,
      session: Session,
      execute_fn: Callable[[InvocationContext], AsyncGenerator[Event, None]],
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
            await self.session_service.append_event(
                session=session, event=event
            )
          # Step 3: Run the on_event callbacks to optionally modify the event.
          modified_event = await plugin_manager.run_on_event_callback(
              invocation_context=invocation_context, event=event
          )
          yield (modified_event if modified_event else event)

    # Step 4: Run the after_run callbacks to optionally modify the context.
    await plugin_manager.run_after_run_callback(
        invocation_context=invocation_context
    )

  async def _append_new_message_to_session(
      self,
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
    """
    if not new_message.parts:
      raise ValueError('No parts in the new_message.')

    if self.artifact_service and save_input_blobs_as_artifacts:
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
    await self.session_service.append_event(session=session, event=event)

  async def run_live(
      self,
      *,
      user_id: Optional[str] = None,
      session_id: Optional[str] = None,
      live_request_queue: LiveRequestQueue,
      run_config: RunConfig = RunConfig(),
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
        self._exec_with_plugin(invocation_context, session, execute)
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
    # the agent that returned the corressponding function call regardless the
    # type of the agent. e.g. a remote a2a agent may surface a credential
    # request as a special long running function tool call.
    event = find_matching_function_call(session.events)
    if event and event.author:
      return root_agent.find_agent(event.author)
    for event in filter(lambda e: e.author != 'user', reversed(session.events)):
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

  def _new_invocation_context(
      self,
      session: Session,
      *,
      new_message: Optional[types.Content] = None,
      live_request_queue: Optional[LiveRequestQueue] = None,
      run_config: RunConfig = RunConfig(),
  ) -> InvocationContext:
    """Creates a new invocation context.

    Args:
        session: The session for the context.
        new_message: The new message for the context.
        live_request_queue: The live request queue for the context.
        run_config: The run config for the context.

    Returns:
        The new invocation context.
    """
    invocation_id = new_invocation_context_id()

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
        invocation_id=invocation_id,
        agent=self.agent,
        session=session,
        user_content=new_message,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

  def _new_invocation_context_for_live(
      self,
      session: Session,
      *,
      live_request_queue: Optional[LiveRequestQueue] = None,
      run_config: RunConfig = RunConfig(),
  ) -> InvocationContext:
    """Creates a new invocation context for live multi-agent."""

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
      except Exception as e:
        logger.error('Error closing toolset %s: %s', type(toolset).__name__, e)

  async def close(self):
    """Closes the runner."""
    await self._cleanup_toolsets(self._collect_toolset(self.agent))


class InMemoryRunner(Runner):
  """An in-memory Runner for testing and development.

  This runner uses in-memory implementations for artifact, session, and memory
  services, providing a lightweight and self-contained environment for agent
  execution.

  Attributes:
      agent: The root agent to run.
      app_name: The application name of the runner. Defaults to
        'InMemoryRunner'.
      _in_memory_session_service: Deprecated. Please don't use. The in-memory
        session service for the runner.
  """

  def __init__(
      self,
      agent: BaseAgent,
      *,
      app_name: str = 'InMemoryRunner',
      plugins: Optional[list[BasePlugin]] = None,
  ):
    """Initializes the InMemoryRunner.

    Args:
        agent: The root agent to run.
        app_name: The application name of the runner. Defaults to
          'InMemoryRunner'.
    """
    self._in_memory_session_service = InMemorySessionService()
    super().__init__(
        app_name=app_name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        plugins=plugins,
        session_service=self._in_memory_session_service,
        memory_service=InMemoryMemoryService(),
    )
