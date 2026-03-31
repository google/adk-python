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

from datetime import datetime
from datetime import timezone
import inspect
import logging
from typing import Awaitable
from typing import Callable
from typing import Optional

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Role
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from google.adk.platform import time as platform_time
from google.adk.platform import uuid as platform_uuid
from google.adk.runners import Runner
from typing_extensions import override

from ...utils.context_utils import Aclosing
from ..agent.interceptors.new_integration_extension import _NEW_A2A_ADK_INTEGRATION_EXTENSION
from ..converters.request_converter import AgentRunRequest
from ..converters.utils import _get_adk_metadata_key
from ..experimental import a2a_experimental
from .a2a_agent_executor_impl import _A2aAgentExecutor as ExecutorImpl
from .config import A2aAgentExecutorConfig
from .executor_context import ExecutorContext
from .task_result_aggregator import TaskResultAggregator
from .utils import execute_after_agent_interceptors
from .utils import execute_after_event_interceptors
from .utils import execute_before_agent_interceptors

logger = logging.getLogger('google_adk.' + __name__)


@a2a_experimental
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
      config: Optional[A2aAgentExecutorConfig] = None,
      use_legacy: bool = False,
      force_new_version: bool = False,
  ):
    super().__init__()
    self._runner = runner
    self._config = config or A2aAgentExecutorConfig()
    self._use_legacy = use_legacy
    self._force_new_version = force_new_version
    self._executor_impl = None

  async def _resolve_runner(self) -> Runner:
    """Resolve the runner, handling cases where it's a callable that returns a Runner."""
    # If already resolved and cached, return it
    if isinstance(self._runner, Runner):
      return self._runner
    if callable(self._runner):
      # Call the function to get the runner
      result = self._runner()

      # Handle async callables
      if inspect.iscoroutine(result):
        resolved_runner = await result
      else:
        resolved_runner = result

      # Cache the resolved runner for future calls
      self._runner = resolved_runner
      return resolved_runner

    raise TypeError(
        'Runner must be a Runner instance or a callable that returns a'
        f' Runner, got {type(self._runner)}'
    )

  @override
  async def cancel(self, context: RequestContext, event_queue: EventQueue):
    """Cancel the execution."""
    if self._executor_impl:
      await self._executor_impl.cancel(context, event_queue)
      return

    # TODO: Implement proper cancellation logic if needed
    raise NotImplementedError('Cancellation is not supported')

  @override
  async def execute(
      self,
      context: RequestContext,
      event_queue: EventQueue,
  ):
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
        )
      await self._executor_impl.execute(context, event_queue)
      return

    if not context.message:
      raise ValueError('A2A request must have a message')

    context = await execute_before_agent_interceptors(
        context, self._config.execute_interceptors
    )

    # for new task, create a task submitted event
    if not context.current_task:
      await event_queue.enqueue_event(
          TaskStatusUpdateEvent(
              task_id=context.task_id,
              status=TaskStatus(
                  state=TaskState.submitted,
                  message=context.message,
                  timestamp=datetime.fromtimestamp(
                      platform_time.get_time(), tz=timezone.utc
                  ).isoformat(),
              ),
              context_id=context.context_id,
              final=False,
          )
      )

    # Handle the request and publish updates to the event queue
    try:
      await self._handle_request(context, event_queue)
    except Exception as e:
      logger.error('Error handling A2A request: %s', e, exc_info=True)
      # Publish failure event
      try:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    timestamp=datetime.fromtimestamp(
                        platform_time.get_time(), tz=timezone.utc
                    ).isoformat(),
                    message=Message(
                        message_id=platform_uuid.new_uuid(),
                        role=Role.agent,
                        parts=[TextPart(text=str(e))],
                    ),
                ),
                context_id=context.context_id,
                final=True,
            )
        )
      except Exception as enqueue_error:
        logger.error(
            'Failed to publish failure event: %s', enqueue_error, exc_info=True
        )

  async def _handle_request(
      self,
      context: RequestContext,
      event_queue: EventQueue,
  ):
    # Resolve the runner instance
    runner = await self._resolve_runner()

    # Convert the a2a request to AgentRunRequest
    run_request = self._config.request_converter(
        context,
        self._config.a2a_part_converter,
    )

    # ensure the session exists
    session = await self._prepare_session(context, run_request, runner)

    # create invocation context
    invocation_context = runner._new_invocation_context(
        session=session,
        new_message=run_request.new_message,
        run_config=run_request.run_config,
    )

    executor_context = ExecutorContext(
        app_name=runner.app_name,
        user_id=run_request.user_id,
        session_id=run_request.session_id,
        runner=runner,
    )

    # publish the task working event
    await event_queue.enqueue_event(
        TaskStatusUpdateEvent(
            task_id=context.task_id,
            status=TaskStatus(
                state=TaskState.working,
                timestamp=datetime.fromtimestamp(
                    platform_time.get_time(), tz=timezone.utc
                ).isoformat(),
            ),
            context_id=context.context_id,
            final=False,
            metadata={
                _get_adk_metadata_key('app_name'): runner.app_name,
                _get_adk_metadata_key('user_id'): run_request.user_id,
                _get_adk_metadata_key('session_id'): run_request.session_id,
            },
        )
    )

    task_result_aggregator = TaskResultAggregator()
    async with Aclosing(runner.run_async(**vars(run_request))) as agen:
      async for adk_event in agen:
        for a2a_event in self._config.event_converter(
            adk_event,
            invocation_context,
            context.task_id,
            context.context_id,
            self._config.gen_ai_part_converter,
        ):
          a2a_event = await execute_after_event_interceptors(
              a2a_event,
              executor_context,
              adk_event,
              self._config.execute_interceptors,
          )
          if a2a_event is None:
            continue

          task_result_aggregator.process_event(a2a_event)
          await event_queue.enqueue_event(a2a_event)

    # publish the task result event - this is final
    if (
        task_result_aggregator.task_state == TaskState.working
        and task_result_aggregator.task_status_message is not None
        and task_result_aggregator.task_status_message.parts
    ):
      # if task is still working properly, publish the artifact update event as
      # the final result according to a2a protocol.
      await event_queue.enqueue_event(
          TaskArtifactUpdateEvent(
              task_id=context.task_id,
              last_chunk=True,
              context_id=context.context_id,
              artifact=Artifact(
                  artifact_id=platform_uuid.new_uuid(),
                  parts=task_result_aggregator.task_status_message.parts,
              ),
          )
      )
      # public the final status update event
      final_event = TaskStatusUpdateEvent(
          task_id=context.task_id,
          status=TaskStatus(
              state=TaskState.completed,
              timestamp=datetime.fromtimestamp(
                  platform_time.get_time(), tz=timezone.utc
              ).isoformat(),
          ),
          context_id=context.context_id,
          final=True,
      )
    else:
      final_event = TaskStatusUpdateEvent(
          task_id=context.task_id,
          status=TaskStatus(
              state=task_result_aggregator.task_state,
              timestamp=datetime.fromtimestamp(
                  platform_time.get_time(), tz=timezone.utc
              ).isoformat(),
              message=task_result_aggregator.task_status_message,
          ),
          context_id=context.context_id,
          final=True,
      )

    final_event = await execute_after_agent_interceptors(
        executor_context,
        final_event,
        self._config.execute_interceptors,
    )
    await event_queue.enqueue_event(final_event)

  async def _prepare_session(
      self,
      context: RequestContext,
      run_request: AgentRunRequest,
      runner: Runner,
  ):

    session_id = run_request.session_id
    # create a new session if not exists
    user_id = run_request.user_id
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

  def _check_new_version_extension(self, context: RequestContext):
    """Check if the extension for the new version is requested and activate it."""
    if _NEW_A2A_ADK_INTEGRATION_EXTENSION in context.requested_extensions:
      context.add_activated_extension(_NEW_A2A_ADK_INTEGRATION_EXTENSION)
      return True
    return False
