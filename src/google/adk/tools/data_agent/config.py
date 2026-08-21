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

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class DataAgentToolConfig(BaseModel):
  """Configuration for Data Agent tools."""

  # Forbid any fields not defined in the model
  model_config = ConfigDict(extra="forbid")

  max_query_result_rows: int = 50
  """Maximum number of rows to return from a query.

  By default, the query result will be limited to 50 rows.
  """

  location: str | None = None
  """The Google Cloud location of the Data Agent (e.g., 'eu', 'us', 'global').

  If not specified, the location will be parsed automatically from the
  Data Agent resource name when possible, or default to 'global'.
  """

  api_endpoint: str | None = None
  """Optional custom API endpoint for Gemini Data Analytics requests.

  If provided, this overrides the default or location-derived API endpoint.
  """

  data_agent_modification_timeout_seconds: int = Field(default=60, gt=0)
  """Total timeout in seconds when waiting for data agent mutation.

  Defaults to 60 seconds.
  """

  data_agent_modification_poll_interval_seconds: int = Field(default=2, gt=0)
  """Poll interval in seconds when waiting for data agent mutation.

  Defaults to 2 seconds.
  """

  enable_data_agent_modification: bool = False
  """Whether the toolset is allowed to mutate data agent resources (i.e., create, update, and delete).

  Defaults to False so a read-only toolset stays read-only.
  """
