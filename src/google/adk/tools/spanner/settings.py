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

from enum import Enum
from typing import List

from pydantic import BaseModel

from ...utils.feature_decorator import experimental


class Capabilities(Enum):
  """Capabilities indicating what type of operation tools are allowed to be performed on Spanner."""

  DATA_READ = 'data_read'
  """Read only data operations tools are allowed."""


@experimental('Tool settings defaults may have breaking change in the future.')
class SpannerToolSettings(BaseModel):
  """Settings for Spanner tools."""

  capabilities: List[Capabilities] = [
      Capabilities.DATA_READ,
  ]
  """Allowed capabilities for Spanner tools.

  By default, the tool will allow only read operations. This behaviour may
  change in future versions.
  """

  max_executed_query_result_rows: int = 50
  """Maximum number of rows to return from a query result."""
