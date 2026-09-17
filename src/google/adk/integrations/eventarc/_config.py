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

"""Configuration models for the Eventarc toolset."""

from __future__ import annotations

import pydantic

from ...features import experimental
from ...features import FeatureName
from ...tools._google_credentials import BaseGoogleCredentialsConfig


@experimental(FeatureName.EVENTARC_TOOL_CONFIG)
class EventarcToolConfig(pydantic.BaseModel):
  """Configuration for the Eventarc tool."""

  model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

  project_id: str | None = None
  """Optional project ID for telemetry and API calls."""

  publish_timeout: float = 15.0
  """Timeout in seconds for publishing messages. Defaults to 15.0."""


class EventarcCredentialsConfig(BaseGoogleCredentialsConfig):
  """Configuration for Google Cloud credentials."""

  pass
