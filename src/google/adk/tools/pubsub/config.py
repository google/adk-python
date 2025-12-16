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

from pydantic import BaseModel
from pydantic import ConfigDict

from ...utils.feature_decorator import experimental


@experimental('Config defaults may have breaking change in the future.')
class PubSubToolConfig(BaseModel):
  """Configuration for Pub/Sub tools."""

  # Forbid any fields not defined in the model
  model_config = ConfigDict(extra='forbid')

  project_id: str | None = None
  """GCP project ID to use for the Pub/Sub operations.

  If not set, the project ID will be inferred from the environment or
  credentials.
  """
