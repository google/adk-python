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


class VersionMismatchError(Exception):
  """Represents an error that occurs when database schema version is incompatible.

  This error is raised when the database schema version stored in the database
  does not match the expected version by the current code. This typically
  indicates that a migration is needed.
  """

  def __init__(
      self,
      message: str = "Database schema version is incompatible.",
      expected_version: int | None = None,
      actual_version: int | None = None,
  ):
    """Initializes the VersionMismatchError exception.

    Args:
        message: An optional custom message to describe the error.
        expected_version: The schema version expected by the current code.
        actual_version: The actual schema version found in the database.
    """
    self.expected_version = expected_version
    self.actual_version = actual_version

    if expected_version is not None and actual_version is not None:
      self.message = (
          f"Database schema version mismatch: expected version {expected_version}, "
          f"but found version {actual_version}. "
          "Please run the migration script to update the database schema."
      )
    else:
      self.message = message

    super().__init__(self.message)
