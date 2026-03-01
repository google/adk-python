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

import dataclasses
from typing import Awaitable
from typing import Callable
from typing import Optional
from typing import Union

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import Event as A2AEvent
from a2a.types import TaskStatusUpdateEvent

from ...events.event import Event
from ..converters.utils import _get_adk_metadata_key
from .executor_context import ExecutorContext


@dataclasses.dataclass
class ExecuteInterceptor:
  """Interceptor for the A2aAgentExecutor."""

  before_agent: Optional[
      Callable[[RequestContext], Awaitable[RequestContext]]
  ] = None
  """Hook executed before the agent starts processing the request.

    Allows inspection or modification of the incoming request context.
    Must return a valid `RequestContext` to continue execution.
  """

  after_event: Optional[
      Callable[
          [ExecutorContext, A2AEvent, Event],
          Awaitable[Union[A2AEvent, None]],
      ]
  ] = None
  """Hook executed after an ADK event is converted to an A2A event.

    Allows mutating the outgoing event before it is enqueued.
    Return `None` to filter out and drop the event entirely,
    which also halts any subsequent interceptors in the chain.
    """

  after_agent: Optional[
      Callable[
          [ExecutorContext, TaskStatusUpdateEvent],
          Awaitable[TaskStatusUpdateEvent],
      ]
  ] = None
  """Hook executed after the agent finishes and the final event is prepared.

    Allows inspection or modification of the terminal status event (e.g.,
    completed or failed) before it is enqueued. Must return a valid
    `TaskStatusUpdateEvent`.
  """
