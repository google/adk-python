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

from typing import Optional

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import Event as A2AEvent
from a2a.types import TaskStatusUpdateEvent

from ...events.event import Event
from ..converters.utils import _get_adk_metadata_key
from .config import ExecuteInterceptor
from .executor_context import ExecutorContext


async def execute_before_agent_interceptors(
    context: RequestContext,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> RequestContext:
  if execute_interceptors:
    for interceptor in execute_interceptors:
      if interceptor.before_agent:
        context = await interceptor.before_agent(context)
  return context


async def execute_after_event_interceptors(
    a2a_event: A2AEvent,
    executor_context: ExecutorContext,
    adk_event: Event,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> Optional[A2AEvent]:
  if execute_interceptors:
    for interceptor in execute_interceptors:
      if interceptor.after_event:
        a2a_event = await interceptor.after_event(
            executor_context, a2a_event, adk_event
        )
        if a2a_event is None:
          return None
  return a2a_event


async def execute_after_agent_interceptors(
    executor_context: ExecutorContext,
    final_event: TaskStatusUpdateEvent,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> TaskStatusUpdateEvent:
  if execute_interceptors:
    for interceptor in reversed(execute_interceptors):
      if interceptor.after_agent:
        final_event = await interceptor.after_agent(
            executor_context, final_event
        )
  return final_event
