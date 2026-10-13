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

from collections.abc import Callable
import functools
import logging
from typing import Any
from typing import Optional

from a2a.server.agent_execution import RequestContext
from google.genai import types as genai_types
from pydantic import BaseModel

from .. import _compat
from ...agents.run_config import RunConfig
from ..experimental import a2a_experimental
from .part_converter import A2APartToGenAIPartConverter
from .part_converter import convert_a2a_part_to_genai_part

logger = logging.getLogger('google_adk.' + __name__)

A2A_METADATA_KEY = 'a2a_metadata'


@a2a_experimental
class AgentRunRequest(BaseModel):
  """Data model for arguments passed to the ADK runner."""

  user_id: Optional[str] = None
  session_id: Optional[str] = None
  invocation_id: Optional[str] = None
  new_message: Optional[genai_types.Content] = None
  state_delta: Optional[dict[str, Any]] = None
  run_config: Optional[RunConfig] = None


A2ARequestToAgentRunRequestConverter = Callable[
    [
        RequestContext,
        A2APartToGenAIPartConverter,
    ],
    AgentRunRequest,
]
"""A callable that converts an A2A RequestContext to RunnerRequest for ADK runner.

This interface allows for custom logic to map an incoming A2A RequestContext to the
structured arguments expected by the ADK runner's `run_async` method.

Args:
    request: The incoming request context from the A2A server.
    part_converter: A function to convert A2A content parts to GenAI parts.

Returns:
    An RunnerRequest object containing the keyword arguments for ADK runner's run_async method.
"""


def _get_user_id(request: RequestContext) -> str:
  """Returns the ADK user id to run this request as.

  The user id scopes the session, ``user:``-prefixed state, artifacts and
  memory, so it is taken only from a principal the A2A server authenticated.
  The name on an unauthenticated principal is a claim the caller made rather
  than one the server checked, so it is ignored and the conversation is used as
  an anonymous identity instead.
  """
  user = request.call_context.user if request.call_context else None
  if user and user.user_name:
    if user.is_authenticated:
      return user.user_name
    _warn_unauthenticated_user_name_once()

  return f'A2A_USER_{request.context_id}'


@functools.lru_cache(maxsize=1)
def _warn_unauthenticated_user_name_once() -> None:
  logger.warning(
      'Ignoring the user name of an unauthenticated A2A caller. Requests'
      ' without an authenticated user run as A2A_USER_<context_id>, so'
      ' sessions, state, artifacts and memory stored under an unverified name'
      ' are no longer used.'
  )


@a2a_experimental
def convert_a2a_request_to_agent_run_request(
    request: RequestContext,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> AgentRunRequest:
  """Converts an A2A RequestContext to an AgentRunRequest model.

  Args:
    request: The incoming request context from the A2A server.
    part_converter: A function to convert A2A content parts to GenAI parts.

  Returns:
    A AgentRunRequest object ready to be used as arguments for the ADK runner.

  Raises:
    ValueError: If the request message is None.
  """

  if not request.message:
    raise ValueError('Request message cannot be None')

  custom_metadata = {}
  request_metadata = _compat.meta_to_dict(request.metadata)
  if request_metadata:
    custom_metadata[A2A_METADATA_KEY] = request_metadata

  output_parts = []
  for a2a_part in request.message.parts:
    genai_parts = part_converter(a2a_part)
    if not isinstance(genai_parts, list):
      genai_parts = [genai_parts] if genai_parts else []
    output_parts.extend(genai_parts)

  return AgentRunRequest(
      user_id=_get_user_id(request),
      session_id=request.context_id,
      new_message=genai_types.Content(
          role='user',
          parts=output_parts,
      ),
      run_config=RunConfig(custom_metadata=custom_metadata),
  )
