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

from enum import Enum

from pydantic import BaseModel

from ...features import experimental
from ...features import FeatureName


class Capabilities(Enum):
  """Capabilities indicating what type of operations are allowed for GCS tools."""

  READ_ONLY = "read_only"
  """Only read operations are allowed."""

  READ_WRITE = "read_write"
  """Both read and write operations are allowed."""


@experimental(FeatureName.GCS_TOOL_SETTINGS)
class GCSToolSettings(BaseModel):
  """Settings for GCS tools."""

  capabilities: list[Capabilities] = [
      Capabilities.READ_ONLY,
  ]
  """Allowed capabilities for GCS tools.

  By default, tools allow only read operations. This behaviour may change in
  future versions.
  """
