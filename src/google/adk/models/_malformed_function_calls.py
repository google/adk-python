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

"""Reports function calls whose arguments are not a JSON object."""

from __future__ import annotations

from collections.abc import Sequence
import logging

from google.genai import types

from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

_FINISH_REASONS_THAT_EXPLAIN_NOTHING = (
    None,
    types.FinishReason.STOP,
    types.FinishReason.OTHER,
    types.FinishReason.FINISH_REASON_UNSPECIFIED,
)


def report_malformed_function_calls(
    response: LlmResponse, names: Sequence[object]
) -> LlmResponse:
  """Marks a response that had function calls with malformed arguments.

  A function call is malformed when its arguments are not a JSON object. The
  caller leaves those calls out of the content and keeps the others. The
  response reports MALFORMED_FUNCTION_CALL, or the finish reason it already has
  when that says the model was cut short.

  Args:
    response: The response without the malformed calls, changed in place.
    names: Names of the malformed function calls, as the provider sent them.
      The response is left as it is when there are none.

  Returns:
    The same response.
  """
  if not names:
    return response
  listed = ', '.join(
      name if isinstance(name, str) and name else '<unnamed>' for name in names
  )
  logger.warning(
      'Discarded function calls whose arguments are not a JSON object: %s',
      listed,
  )
  if response.content and not response.content.parts:
    response.content = None
  if response.finish_reason in _FINISH_REASONS_THAT_EXPLAIN_NOTHING:
    response.finish_reason = types.FinishReason.MALFORMED_FUNCTION_CALL
  response.error_code = response.finish_reason
  message = (
      'Arguments for the following function calls were not a valid JSON'
      f' object: {listed}'
  )
  response.error_message = (
      f'{response.error_message}. {message}'
      if response.error_message
      else message
  )
  return response
