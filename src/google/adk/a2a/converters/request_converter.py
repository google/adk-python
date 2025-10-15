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
        'A2A requires Python 3.10 or above. Please upgrade your Python version.'
    ) from e
  else:
    raise e

from google.genai import types as genai_types

from ...runners import RunConfig
from ..experimental import a2a_experimental
from .part_converter import A2APartToGenAIPartConverter
from .part_converter import convert_a2a_part_to_genai_part


def _get_user_id(request: RequestContext) -> str:
  # Get user from call context if available (auth is enabled on a2a server)
  if (
      request.call_context
      and request.call_context.user
      and request.call_context.user.user_name
  ):
    return request.call_context.user.user_name

  # Get user from context id
  return f'A2A_USER_{request.context_id}'


@a2a_experimental
def convert_a2a_request_to_adk_run_args(
    request: RequestContext,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> dict[str, Any]:

  if not request.message:
    raise ValueError('Request message cannot be None')

  return {
      'user_id': _get_user_id(request),
      'session_id': request.context_id,
      'new_message': genai_types.Content(
          role='user',
          parts=[part_converter(part) for part in request.message.parts],
      ),
      'run_config': RunConfig(),
  }
