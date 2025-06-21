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

import sys
from typing import Any

try:
  from a2a.server.agent_execution import RequestContext
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        'A2A Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

from google.genai import types as genai_types

from ...runners import RunConfig
from ...utils.feature_decorator import working_in_progress
from .part_converter import convert_a2a_part_to_genai_part
from .utils import _from_a2a_context_id
from .utils import _get_adk_metadata_key


def _get_user_id(request: RequestContext, user_id_from_context: str) -> str:
  # Get user from call context if available (auth is enabled on a2a server)
  if request.call_context and request.call_context.user:
    return request.call_context.user.user_name

  # Get user from context id if available
  if user_id_from_context:
    return user_id_from_context

  # Get user from message metadata if available (client is an ADK agent)
  if request.message.metadata:
    user_id = request.message.metadata.get(_get_adk_metadata_key('user_id'))
    if user_id:
      return f'ADK_USER_{user_id}'

  # Get user from task if available (client is a an ADK agent)
  if request.current_task:
    user_id = request.current_task.metadata.get(
        _get_adk_metadata_key('user_id')
    )
    if user_id:
      return f'ADK_USER_{user_id}'
  return (
      f'temp_user_{request.task_id}'
      if request.task_id
      else f'TEMP_USER_{request.message.messageId}'
  )


@working_in_progress
def convert_a2a_request_to_adk_run_args(
    request: RequestContext,
) -> dict[str, Any]:

  if not request.message:
    raise ValueError('Request message cannot be None')

  _, user_id, session_id = _from_a2a_context_id(request.context_id)

  return {
      'user_id': _get_user_id(request, user_id),
      'session_id': session_id,
      'new_message': genai_types.Content(
          role='user',
          parts=[
              convert_a2a_part_to_genai_part(part)
              for part in request.message.parts
          ],
      ),
      'run_config': RunConfig(),
  }
