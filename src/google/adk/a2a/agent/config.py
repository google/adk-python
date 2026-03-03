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

"""Configuration for A2A agents."""

from __future__ import annotations

from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Optional
from typing import Union

from a2a.client.middleware import ClientCallContext
from a2a.server.events import Event as A2AEvent
from a2a.types import Message as A2AMessage
from a2a.types import MessageSendConfiguration
from pydantic import BaseModel

from ...agents.invocation_context import InvocationContext
from ...events.event import Event


class ParametersConfig(BaseModel):
  """Configuration for the parameters passed to the A2A send_message request."""

  request_metadata: Optional[dict[str, Any]] = None
  client_call_context: Optional[ClientCallContext] = None
  # TODO: Add support for requested_extension and
  # message_send_configuration once they are supported by the A2A client.
  #
  # requested_extension: Optional[list[str]] = None
  # message_send_configuration: Optional[MessageSendConfiguration] = None


class RequestInterceptor(BaseModel):
  """Interceptor for A2A requests."""

  before_request: Optional[
      Callable[
          [InvocationContext, A2AMessage, ParametersConfig],
          Awaitable[tuple[Union[A2AMessage, Event], ParametersConfig]],
      ]
  ] = None
  """Hook executed before the agent starts processing the request.

    Returns an Event if the request should be aborted and the Event
    returned to the caller.
  """

  after_request: Optional[
      Callable[
          [InvocationContext, A2AEvent, Event], Awaitable[Union[Event, None]]
      ]
  ] = None
  """Hook executed after the agent has processed the request.

    Returns None if the event should not be sent to the caller.
  """


class A2aRemoteAgentConfig(BaseModel):
  """Configuration for the RemoteA2aAgent."""

  request_interceptors: Optional[list[RequestInterceptor]] = None
