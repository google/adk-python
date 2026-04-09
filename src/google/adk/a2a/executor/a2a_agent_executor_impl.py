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
import uuid

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from typing_extensions import override

from ...runners import Runner
from ...sessions import base_session_service
from ...utils.context_utils import Aclosing
from ..agent.interceptors.new_integration_extension import _NEW_A2A_ADK_INTEGRATION_EXTENSION
from ..converters.from_adk_event import create_error_status_event
from ..converters.long_running_functions import handle_user_input
from ..converters.long_running_functions import LongRunningFunctions
from ..converters.request_converter import AgentRunRequest
from ..converters.utils import _get_adk_metadata_key
from ..experimental import a2a_experimental
from .config import A2aAgentExecutorConfig
from .executor_context import ExecutorContext
from .utils import execute_after_agent_interceptors
from .utils import execute_after_event_interceptors
from .utils import execute_before_agent_interceptors

logger = logging.getLogger('google_adk.' + __name__)


@a2a_experimental
class _A2aAgentExecutor(AgentExecutor):
  """An AgentExecutor that runs an ADK Agent against an A2A request and

  publishes updates to an event queue.
  """

  def __init__(
      self,
      *,
      runner: Runner | Callable[..., Runner | Awaitable[Runner]],
      config: Optional[A2aAgentExecutorConfig] = None,
  ):
    super().__init__()
    self._runner = runner
    self._config = config or A2aAgentExecutorConfig()

  @override
  async def cancel(self, context: RequestContext, event_queue: EventQueue):
    """Cancel the execution."""
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
    if not context.message:
      raise ValueError('A2A request must have a message')

    context = await execute_before_agent_interceptors(
        context, self._config.execute_interceptors
    )

    runner = await self._resolve_runner()
    try:
      run_request = self._config.request_converter(
          context,
          self._config.a2a_part_converter,
      )
      await self._resolve_session(run_request, runner)

      executor_context = ExecutorContext(
          app_name=runner.app_name,
          user_id=run_request.user_id,
          session_id=run_request.session_id,
          runner=runner,
      )

      # for new task, create a task submitted event
      if not context.current_task:
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                status=TaskStatus(
                    state=TaskState.submitted,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                context_id=context.context_id,
                history=[context.message],
                metadata=self._get_invocation_metadata(executor_context),
            )
        )
      else:
        # Check if the user input is responding to the agent's
        # request for input.
        missing_user_input_event = handle_user_input(context)
        if missing_user_input_event:
          missing_user_input_event.metadata = self._get_invocation_metadata(
              executor_context
          )
          await event_queue.enqueue_event(missing_user_input_event)
          return

      await event_queue.enqueue_event(
          TaskStatusUpdateEvent(
              task_id=context.task_id,
              status=TaskStatus(
                  state=TaskState.working,
                  timestamp=datetime.now(timezone.utc).isoformat(),
              ),
              context_id=context.context_id,
              final=False,
              metadata=self._get_invocation_metadata(executor_context),
          )
      )

      # Handle the request and publish updates to the event queue
      await self._handle_request(
          context,
          executor_context,
          event_queue,
          runner,
          run_request,
      )
    except Exception as e:
      logger.error('Error handling A2A request: %s', e, exc_info=True)
      # Publish failure event
      try:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    message=Message(
                        message_id=str(uuid.uuid4()),
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
      executor_context: ExecutorContext,
      event_queue: EventQueue,
      runner: Runner,
      run_request: AgentRunRequest,
  ):
    agents_artifact: dict[str, str] = {}
    error_event = None
    long_running_functions = LongRunningFunctions(
        self._config.gen_ai_part_converter
    )
    async with Aclosing(runner.run_async(**vars(run_request))) as agen:
      async for adk_event in agen:
        # Handle error scenarios
        if adk_event and (adk_event.error_code or adk_event.error_message):
          error_event = create_error_status_event(
              adk_event,
              context.task_id,
              context.context_id,
          )

        # Handle long running function calls
        adk_event = long_running_functions.process_event(adk_event)

        for a2a_event in self._config.adk_event_converter(
            adk_event,
            agents_artifact,
            context.task_id,
            context.context_id,
            self._config.gen_ai_part_converter,
        ):
          a2a_event.metadata = self._get_invocation_metadata(executor_context)
          a2a_event = await execute_after_event_interceptors(
              a2a_event,
              executor_context,
              adk_event,
              self._config.execute_interceptors,
          )
          if not a2a_event:
            continue
          await event_queue.enqueue_event(a2a_event)

    if error_event:
      final_event = error_event
    elif long_running_functions.has_long_running_function_calls():
      final_event = (
          long_running_functions.create_long_running_function_call_event(
              context.task_id, context.context_id
          )
      )
    else:
      final_event = TaskStatusUpdateEvent(
          task_id=context.task_id,
          status=TaskStatus(
              state=TaskState.completed,
              timestamp=datetime.now(timezone.utc).isoformat(),
          ),
          context_id=context.context_id,
          final=True,
      )

    final_event.metadata = self._get_invocation_metadata(executor_context)
    final_event = await execute_after_agent_interceptors(
        executor_context, final_event, self._config.execute_interceptors
    )
    await event_queue.enqueue_event(final_event)

  async def _resolve_runner(self) -> Runner:
    """Resolve the runner, handling cases where it's a callable that returns a Runner."""
    if isinstance(self._runner, Runner):
      return self._runner
    if callable(self._runner):
      result = self._runner()

      if inspect.iscoroutine(result):
        resolved_runner = await result
      else:
        resolved_runner = result

      self._runner = resolved_runner
      return resolved_runner

    raise TypeError(
        'Runner must be a Runner instance or a callable that returns a'
        f' Runner, got {type(self._runner)}'
    )

  async def _resolve_session(
      self,
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
        # Checking existence doesn't require event history.
        config=base_session_service.GetSessionConfig(num_recent_events=0),
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

  def _get_invocation_metadata(
      self, executor_context: ExecutorContext
  ) -> dict[str, str]:
    return {
        _get_adk_metadata_key('app_name'): executor_context.app_name,
        _get_adk_metadata_key('user_id'): executor_context.user_id,
        _get_adk_metadata_key('session_id'): executor_context.session_id,
        # TODO: Remove this metadata once the new agent executor
        # is fully adopted.
        _NEW_A2A_ADK_INTEGRATION_EXTENSION: {'adk_agent_executor_v2': True},
    }
