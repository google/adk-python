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


class InputValidationError(ValueError):
  """Represents an error raised when user input fails validation."""

  def __init__(self, message="Invalid input."):
    """Initializes the InputValidationError exception.

    Args:
        message (str): A message describing why the input is invalid.
    """
    self.message = message
    super().__init__(self.message)
