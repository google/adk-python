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
from pydantic import BaseModel

from ...events.event import Event
from ..converters.event_converter import AdkEventToA2AEventsConverter
from ..converters.event_converter import convert_event_to_a2a_events as legacy_convert_event_to_a2a_events
from ..converters.from_adk_event import AdkEventToA2AEventsConverter as AdkEventToA2AEventsConverterImpl
from ..converters.from_adk_event import convert_event_to_a2a_events as convert_event_to_a2a_events_impl
from ..converters.part_converter import A2APartToGenAIPartConverter
from ..converters.part_converter import convert_a2a_part_to_genai_part
from ..converters.part_converter import convert_genai_part_to_a2a_part
from ..converters.part_converter import GenAIPartToA2APartConverter
from ..converters.request_converter import A2ARequestToAgentRunRequestConverter
from ..converters.request_converter import convert_a2a_request_to_agent_run_request
from ..converters.utils import _get_adk_metadata_key
from ..experimental import a2a_experimental
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


@a2a_experimental
class A2aAgentExecutorConfig(BaseModel):
  """Configuration for the A2aAgentExecutor."""

  a2a_part_converter: A2APartToGenAIPartConverter = (
      convert_a2a_part_to_genai_part
  )
  gen_ai_part_converter: GenAIPartToA2APartConverter = (
      convert_genai_part_to_a2a_part
  )
  request_converter: A2ARequestToAgentRunRequestConverter = (
      convert_a2a_request_to_agent_run_request
  )
  event_converter: AdkEventToA2AEventsConverter = (
      legacy_convert_event_to_a2a_events
  )
  """Set up the default event converter implementation to be used by the legacy agent executor implementation."""

  adk_event_converter: AdkEventToA2AEventsConverterImpl = (
      convert_event_to_a2a_events_impl
  )
  """Set up the imlp event converter implementation to be used by the new agent executor implementation."""

  execute_interceptors: Optional[list[ExecuteInterceptor]] = None
