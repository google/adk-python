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

"""Utilities for A2A agents."""

from __future__ import annotations

from typing import Optional
from typing import Union

from a2a.client import ClientEvent as A2AClientEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import Message as A2AMessage

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from .config import ParametersConfig
from .config import RequestInterceptor


async def execute_before_request_interceptors(
    request_interceptors: Optional[list[RequestInterceptor]],
    ctx: InvocationContext,
    a2a_request: A2AMessage,
) -> tuple[Union[A2AMessage, Event], ParametersConfig]:
  """Executes registered before_request interceptors."""

  params = ParametersConfig(
      client_call_context=ClientCallContext(state=ctx.session.state)
  )
  if request_interceptors:
    for interceptor in request_interceptors:
      if not interceptor.before_request:
        continue

      result, params = await interceptor.before_request(
          ctx, a2a_request, params
      )
      if isinstance(result, Event):
        return result, params
      a2a_request = result

  return a2a_request, params


async def execute_after_request_interceptors(
    request_interceptors: Optional[list[RequestInterceptor]],
    ctx: InvocationContext,
    a2a_response: A2AMessage | A2AClientEvent,
    event: Event,
) -> Optional[Event]:
  """Executes registered after_request interceptors."""
  if request_interceptors:
    for interceptor in reversed(request_interceptors):
      if interceptor.after_request:
        event = await interceptor.after_request(ctx, a2a_response, event)
        if not event:
          return None
  return event
