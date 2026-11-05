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
import inspect
import logging
from typing import Awaitable
from typing import Callable

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatusUpdateEvent
from google.adk.platform import uuid as platform_uuid
from google.adk.runners import Runner
from google.adk.sessions.session import Session
from typing_extensions import override

from .. import _compat
from ...events.event import Event
from ..agent.interceptors.new_integration_extension import _NEW_A2A_ADK_INTEGRATION_EXTENSION
from ..converters.request_converter import AgentRunRequest
from ..converters.utils import _get_adk_metadata_key
from .a2a_agent_executor_impl import _A2aAgentExecutor as ExecutorImpl
from .config import A2aAgentExecutorConfig
from .executor_context import ExecutorContext
from .task_result_aggregator import TaskResultAggregator
from .utils import _ActiveTaskRegistry
from .utils import _consume_runner_with_abort
from .utils import _ErrorRecoveryTracker
from .utils import _is_abort_adk_event
from .utils import _is_error_event
from .utils import _require_request_context
from .utils import _require_user_id
from .utils import execute_after_agent_interceptors
from .utils import execute_after_event_interceptors
from .utils import execute_before_agent_interceptors

logger = logging.getLogger('google_adk.' + __name__)


class A2aAgentExecutor(AgentExecutor):
  """An AgentExecutor that runs an ADK Agent against an A2A request and

  publishes updates to an event queue.

  Args:
    runner: The runner to use for the agent.
    config: The config to use for the executor.
    use_legacy: If true, force the legacy implementation.
    force_new_version: If true, force the new implementation regardless of the
      extension.
  """

  def __init__(
      self,
      *,
      runner: Runner | Callable[..., Runner | Awaitable[Runner]],
      config: A2aAgentExecutorConfig | None = None,
      use_legacy: bool = False,
      force_new_version: bool = False,
  ):
    super().__init__()
    self._runner = runner
    self._config = config or A2aAgentExecutorConfig()
    self._use_legacy = use_legacy
    self._force_new_version = force_new_version
    self._executor_impl: ExecutorImpl | None = None
    self._active_tasks = _ActiveTaskRegistry()

  async def _resolve_runner(self) -> Runner:
    """Resolve the runner, handling cases where it's a callable that returns a Runner."""
    # If already resolved and cached, return it
    if isinstance(self._runner, Runner):
      return self._runner
    if callable(self._runner):
      # Call the function to get the runner
      result = self._runner()

      # Handle async callables
      if inspect.isawaitable(result):
        resolved_runner = await result
      else:
        resolved_runner = result

      # Cache the resolved runner for future calls
      self._runner = resolved_runner
      return self._runner

    raise TypeError(
        'Runner must be a Runner instance or a callable that returns a'
        f' Runner, got {type(self._runner)}'
    )

  @override
  async def cancel(
      self, context: RequestContext, event_queue: EventQueue
  ) -> None:
    """Cancel the execution."""
    await self._active_tasks.cancel(context, event_queue)

  @override
  async def execute(
      self,
      context: RequestContext,
      event_queue: EventQueue,
  ) -> None:
    """Executes an A2A request and publishes updates to the event queue

    specified. It runs as following:
    * Takes the input from the A2A request
    * Convert the input to ADK input content, and runs the ADK agent
    * Collects output events of the underlying ADK Agent
    * Converts the ADK output events into A2A task updates
    * Publishes the updates back to A2A server via event queue
    """
    should_use_new_impl = not self._use_legacy and (
        self._force_new_version or self._check_new_version_extension(context)
    )

    if should_use_new_impl:
      if self._executor_impl is None:
        self._executor_impl = ExecutorImpl(
            runner=self._runner,
            config=self._config,
            active_tasks=self._active_tasks,
        )
      await self._executor_impl.execute(context, event_queue)
      return

    if not context.message:
      raise ValueError('A2A request must have a message')

    context = await execute_before_agent_interceptors(
        context, self._config.execute_interceptors
    )

    # For a new task, publish the initial "submitted" signal. The leading-Task
    # (1.x) vs submitted-event (0.3.x) divergence is handled by ``_compat``.
    await _compat.enqueue_submitted_signal(event_queue, context=context)

    active_run = self._active_tasks.start(context.task_id, event_queue)
    # Handle the request and publish updates to the event queue
    try:
      await self._handle_request(context, event_queue)
    except Exception as e:
      logger.error('Error handling A2A request: %s', e, exc_info=True)
      # Publish failure event
      try:
        await event_queue.enqueue_event(
            _compat.make_task_status_update_event(
                task_id=context.task_id,
                context_id=context.context_id,
                status=_compat.make_task_status(
                    _compat.TS_FAILED,
                    message=Message(
                        message_id=platform_uuid.new_uuid(),
                        role=_compat.ROLE_AGENT,
                        parts=[_compat.make_text_part(str(e))],
                    ),
                ),
                final=True,
            )
        )
      except Exception as enqueue_error:
        logger.error(
            'Failed to publish failure event: %s', enqueue_error, exc_info=True
        )
    finally:
      self._active_tasks.finish(active_run)

  async def _handle_request(
      self,
      context: RequestContext,
      event_queue: EventQueue,
  ) -> None:
    _, task_id, context_id = _require_request_context(context)
    active_run = self._active_tasks.get(task_id)
    # Resolve the runner instance
    runner = await self._resolve_runner()

    # Convert the a2a request to AgentRunRequest
    run_request = self._config.request_converter(
        context,
        self._config.a2a_part_converter,
    )

    # ensure the session exists
    session = await self._prepare_session(context, run_request, runner)
    if not run_request.session_id:
      # A _prepare_session override may not write the id back to the request.
      run_request.session_id = session.id

    # create invocation context
    invocation_context = runner._new_invocation_context(
        session=session,
        new_message=run_request.new_message,
        run_config=run_request.run_config,
    )

    executor_context = ExecutorContext(
        app_name=runner.app_name,
        user_id=_require_user_id(run_request),
        session_id=run_request.session_id,
        runner=runner,
    )

    # publish the task working event
    await event_queue.enqueue_event(
        _compat.make_task_status_update_event(
            task_id=task_id,
            context_id=context_id,
            status=_compat.make_task_status(_compat.TS_WORKING),
            final=False,
            metadata={
                _get_adk_metadata_key('app_name'): runner.app_name,
                _get_adk_metadata_key('user_id'): run_request.user_id,
                _get_adk_metadata_key('session_id'): run_request.session_id,
            },
        )
    )

    task_result_aggregator = TaskResultAggregator()
    error_tracker = _ErrorRecoveryTracker()
    # Each failure reported, in emission order, with the ADK error event it
    # reports. A failure reported for a non-error event (source None) cannot be
    # recovered from.
    failures: list[tuple[Event | None, Message | None]] = []
    last_adk_event: Event | None = None
    abort_signal = (
        active_run.abort_signal if active_run is not None else asyncio.Event()
    )
    aborted = False

    async def _on_event(adk_event: Event) -> None:
      nonlocal aborted, last_adk_event
      last_adk_event = adk_event
      if adk_event and _is_abort_adk_event(adk_event):
        aborted = True
      error_tracker.observe(adk_event)
      is_error_event = _is_error_event(adk_event)
      for a2a_event in self._config.event_converter(
          adk_event,
          invocation_context,
          task_id,
          context_id,
          self._config.gen_ai_part_converter,
      ):
        a2a_events = await execute_after_event_interceptors(
            a2a_event,
            executor_context,
            adk_event,
            self._config.execute_interceptors,
        )
        for e in a2a_events:
          if (
              isinstance(e, TaskStatusUpdateEvent)
              and e.status.state == _compat.TS_FAILED
          ):
            failures.append(
                (adk_event if is_error_event else None, e.status.message)
            )
          task_result_aggregator.process_event(e)
          await event_queue.enqueue_event(e)

    await _consume_runner_with_abort(
        runner.run_async, run_request, abort_signal, _on_event
    )
    if abort_signal.is_set():
      aborted = True

    # An error the run recovered from (e.g. a failed sub-agent whose parent
    # went on to answer) neither fails the task nor supplies its message.
    if task_result_aggregator.task_state == _compat.TS_FAILED:
      # pylint: disable-next=protected-access
      task_result_aggregator._keep_failures([
          message
          for source, message in failures
          if source is None or error_tracker.is_unresolved(source)
      ])

    # Build metadata for final event to preserve invocation_id and event_id.
    final_metadata = {
        _get_adk_metadata_key('app_name'): runner.app_name,
        _get_adk_metadata_key('user_id'): run_request.user_id,
        _get_adk_metadata_key('session_id'): run_request.session_id,
    }
    if last_adk_event:
      for key, attr in [
          ('invocation_id', 'invocation_id'),
          ('author', 'author'),
          ('event_id', 'id'),
      ]:
        val = getattr(last_adk_event, attr, None)
        if val is not None:
          final_metadata[_get_adk_metadata_key(key)] = val

    # publish the task result event - this is final
    if (
        not aborted
        and task_result_aggregator.task_state == _compat.TS_WORKING
        and task_result_aggregator.task_status_message is not None
        and task_result_aggregator.task_status_message.parts
    ):
      # if task is still working properly, publish the artifact update event as
      # the final result according to a2a protocol.
      await event_queue.enqueue_event(
          TaskArtifactUpdateEvent(
              task_id=task_id,
              last_chunk=True,
              context_id=context_id,
              artifact=Artifact(
                  artifact_id=platform_uuid.new_uuid(),
                  parts=task_result_aggregator.task_status_message.parts,
              ),
              metadata=final_metadata,
          )
      )
      # publish the final status update event
      final_event = _compat.make_task_status_update_event(
          task_id=task_id,
          context_id=context_id,
          status=_compat.make_task_status(_compat.TS_COMPLETED),
          final=True,
          metadata=final_metadata,
      )
    else:
      final_state = (
          _compat.TS_CANCELED if aborted else task_result_aggregator.task_state
      )
      # A run that ends without failing or pausing has completed.
      if final_state == _compat.TS_WORKING:
        final_state = _compat.TS_COMPLETED
      status_message = task_result_aggregator.task_status_message
      if aborted and task_result_aggregator.task_state != _compat.TS_CANCELED:
        status_message = None
      final_event = _compat.make_task_status_update_event(
          task_id=task_id,
          context_id=context_id,
          status=_compat.make_task_status(
              final_state,
              message=status_message,
          ),
          final=True,
          metadata=final_metadata,
      )

    final_event = await execute_after_agent_interceptors(
        executor_context,
        final_event,
        self._config.execute_interceptors,
    )
    await event_queue.enqueue_event(final_event)
    if active_run and final_event.status.state == _compat.TS_CANCELED:
      active_run.terminal_cancel_emitted = True

  async def _prepare_session(
      self,
      context: RequestContext,
      run_request: AgentRunRequest,
      runner: Runner,
  ) -> Session:

    session_id = run_request.session_id
    # create a new session if not exists
    user_id = _require_user_id(run_request)
    session = None
    if session_id:
      session = await runner.session_service.get_session(
          app_name=runner.app_name,
          user_id=user_id,
          session_id=session_id,
      )
    if session is None:
      session = await runner.session_service.create_session(
          app_name=runner.app_name,
          user_id=user_id,
          state={},
          session_id=session_id,
      )
      # Update run_request with the new session_id
      run_request.session_id = session.id

    return session

  def _check_new_version_extension(self, context: RequestContext) -> bool:
    """Check if the extension for the new version is requested and activate it."""
    if _NEW_A2A_ADK_INTEGRATION_EXTENSION in context.requested_extensions:
      _compat.add_activated_extension(
          context, _NEW_A2A_ADK_INTEGRATION_EXTENSION
      )
      return True
    return False
