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
"""Interceptor that injects the new agent version extension."""

from __future__ import annotations

from typing import Union

from a2a.client.middleware import ClientCallContext
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.types import Message as A2AMessage
from google.adk.a2a.agent.config import ParametersConfig
from google.adk.a2a.agent.config import RequestInterceptor
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event

_NEW_A2A_ADK_INTEGRATION_EXTENSION = (
    'https://google.github.io/adk-docs/a2a/a2a-extension/'
)


async def _before_request(
    _: InvocationContext,
    a2a_request: A2AMessage,
    params: ParametersConfig,
) -> tuple[Union[A2AMessage, Event], ParametersConfig]:
  """Adds A2A_new_agent_version to client_call_context."""
  if params.client_call_context is None:
    params.client_call_context = ClientCallContext()

  http_kwargs = params.client_call_context.state.get('http_kwargs', {})
  headers = http_kwargs.get('headers', {})
  a2a_extensions = headers.get(HTTP_EXTENSION_HEADER, '').split(',')
  a2a_extensions = [ext for ext in a2a_extensions if ext]
  if _NEW_A2A_ADK_INTEGRATION_EXTENSION not in a2a_extensions:
    a2a_extensions.append(_NEW_A2A_ADK_INTEGRATION_EXTENSION)
  headers[HTTP_EXTENSION_HEADER] = ','.join(a2a_extensions)
  http_kwargs['headers'] = headers
  params.client_call_context.state['http_kwargs'] = http_kwargs
  return a2a_request, params


_new_integration_extension_interceptor = RequestInterceptor(
    before_request=_before_request
)
