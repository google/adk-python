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

import logging
from typing import Any
from typing import Optional

from google.adk.flows.llm_flows.functions import REQUEST_INPUT_FUNCTION_CALL_NAME

from .long_running_tool import LongRunningFunctionTool

logger = logging.getLogger('google_adk.' + __name__)


def _request_input_func(
    message: str,
    response_schema: Optional[dict[str, Any]] = None,
) -> None:
  """Ask the user a question and wait for their response.

  Use this when you need clarification or additional information before
  proceeding.

  Args:
    message: The question or prompt to display to the user.
    response_schema: JSON Schema describing the expected response format. Use
      {"type": "string"} for free-text, {"type": "boolean"} for
      yes/no, or a structured object schema for complex input.

  Returns:
    None. Long-running tools return None to signal that the execution should
    pause and wait for user input.
  """
  logger.info('request_input called with message: %s', message)
  # Returning None triggers the long-running tool interruption mechanism.
  return None


# Dynamically rename the function to match the workflow interrupt naming space.
# This allows direct instantiation of LongRunningFunctionTool without subclassing,
# keeping RequestInputTool out of the public API.
_request_input_func.__name__ = REQUEST_INPUT_FUNCTION_CALL_NAME

request_input = LongRunningFunctionTool(_request_input_func)
