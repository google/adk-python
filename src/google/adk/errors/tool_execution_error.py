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

import enum


class ToolErrorType(str, enum.Enum):
  """HTTP error types conforming to OpenTelemetry semantics."""

  BAD_REQUEST = 'BAD_REQUEST'
  UNAUTHORIZED = 'UNAUTHORIZED'
  FORBIDDEN = 'FORBIDDEN'
  NOT_FOUND = 'NOT_FOUND'
  REQUEST_TIMEOUT = 'REQUEST_TIMEOUT'
  INTERNAL_SERVER_ERROR = 'INTERNAL_SERVER_ERROR'
  BAD_GATEWAY = 'BAD_GATEWAY'
  SERVICE_UNAVAILABLE = 'SERVICE_UNAVAILABLE'
  GATEWAY_TIMEOUT = 'GATEWAY_TIMEOUT'


class ToolExecutionError(Exception):
  """Represents an error that occurs during the execution of a tool."""

  def __init__(
      self, message: str, error_type: ToolErrorType | str | None = None
  ):
    """Initializes the ToolExecutionError exception.

    Args:
      message (str): A message describing the error.
      error_type (ToolErrorType | str | None): The semantic error type (e.g.,
        ToolErrorType.REQUEST_TIMEOUT or '500'). Used to populate the
        `error.type` span attribute in OpenTelemetry traces.
    """
    self.message = message
    if isinstance(error_type, ToolErrorType):
      self.error_type = error_type.value
    else:
      self.error_type = error_type
    super().__init__(self.message)
