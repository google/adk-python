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

from .not_found_error import NotFoundError


class SessionNotFoundError(ValueError, NotFoundError):
  """Raised when a session cannot be found.

  Inherits from both ValueError (for backward compatibility) and NotFoundError
  (for semantic consistency with the project's error hierarchy).
  """

  def __init__(self, message="Session not found."):
    super().__init__(message)
