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

from datetime import datetime
from datetime import timezone
import inspect
import logging
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Optional
import uuid

from ...utils.context_utils import Aclosing

try:
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

except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'A2A requires Python 3.10 or above. Please upgrade your Python version.'
    ) from e
  else:
    raise e
from google.adk.runners import Runner
from pydantic import BaseModel
from typing_extensions import override

from ..converters.event_converter import convert_event_to_a2a_events
from ..converters.part_converter import A2APartToGenAIPartConverter
from ..converters.part_converter import convert_a2a_part_to_genai_part
from ..converters.part_converter import convert_genai_part_to_a2a_part
from ..converters.part_converter import GenAIPartToA2APartConverter
from ..converters.request_converter import convert_a2a_request_to_adk_run_args
from ..converters.utils import _get_adk_metadata_key
from ..experimental import a2a_experimental
from .task_result_aggregator import TaskResultAggregator

logger = logging.getLogger('google_adk.' + __name__)


@a2a_experimental
class A2aAgentExecutorConfig(BaseModel):
  """Configuration for the A2aAgentExecutor."""

  a2a_part_converter: A2APartToGenAIPartConverter = (
      convert_a2a_part_to_genai_part
  )
  gen_ai_part_converter: GenAIPartToA2APartConverter = (
      convert_genai_part_to_a2a_part
  )


@a2a_experimental
class A2aAgentExecutor(AgentExecutor):
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

    # for new task, create a task submitted event
    if not context.current_task:
      await event_queue.enqueue_event(
          TaskStatusUpdateEvent(
              task_id=context.task_id,
              status=TaskStatus(
                  state=TaskState.submitted,
                  message=context.message,
                  timestamp=datetime.now(timezone.utc).isoformat(),
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
      event_queue: EventQueue,
  ):
    # Resolve the runner instance
    runner = await self._resolve_runner()

    # Convert the a2a request to ADK run args
    run_args = convert_a2a_request_to_adk_run_args(
        context, self._config.a2a_part_converter
    )

    # ensure the session exists
    session = await self._prepare_session(context, run_args, runner)

    # create invocation context
    invocation_context = runner._new_invocation_context(
        session=session,
        new_message=run_args['new_message'],
        run_config=run_args['run_config'],
    )

    # publish the task working event
    await event_queue.enqueue_event(
        TaskStatusUpdateEvent(
            task_id=context.task_id,
            status=TaskStatus(
                state=TaskState.working,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            context_id=context.context_id,
            final=False,
            metadata={
                _get_adk_metadata_key('app_name'): runner.app_name,
                _get_adk_metadata_key('user_id'): run_args['user_id'],
                _get_adk_metadata_key('session_id'): run_args['session_id'],
            },
        )
    )

    task_result_aggregator = TaskResultAggregator()
    async with Aclosing(runner.run_async(**run_args)) as agen:
      async for adk_event in agen:
        for a2a_event in convert_event_to_a2a_events(
            adk_event,
            invocation_context,
            context.task_id,
            context.context_id,
            self._config.gen_ai_part_converter,
        ):
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
                  artifact_id=str(uuid.uuid4()),
                  parts=task_result_aggregator.task_status_message.parts,
              ),
          )
      )
      # public the final status update event
      await event_queue.enqueue_event(
          TaskStatusUpdateEvent(
              task_id=context.task_id,
              status=TaskStatus(
                  state=TaskState.completed,
                  timestamp=datetime.now(timezone.utc).isoformat(),
              ),
              context_id=context.context_id,
              final=True,
          )
      )
    else:
      await event_queue.enqueue_event(
          TaskStatusUpdateEvent(
              task_id=context.task_id,
              status=TaskStatus(
                  state=task_result_aggregator.task_state,
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  message=task_result_aggregator.task_status_message,
              ),
              context_id=context.context_id,
              final=True,
          )
      )

  async def _prepare_session(
      self, context: RequestContext, run_args: dict[str, Any], runner: Runner
  ):

    session_id = run_args['session_id']
    # create a new session if not exists
    user_id = run_args['user_id']
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
      # Update run_args with the new session_id
      run_args['session_id'] = session.id

    return session
