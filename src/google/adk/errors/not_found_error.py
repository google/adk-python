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


class NotFoundError(Exception):
  """Represents an error that occurs when an entity is not found."""

  def __init__(self, message="The requested item was not found."):
    """Initializes the NotFoundError exception.

    Args:
        message (str): An optional custom message to describe the error.
    """
    self.message = message
    super().__init__(self.message)
