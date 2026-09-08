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

"""Configuration for AgentCore Memory session storage."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class AgentCoreSessionServiceConfig(BaseModel):
  """Configuration for AgentCoreSessionService."""

  memory_id: str = Field(
      description='AgentCore Memory resource id (memoryId).',
  )
  region_name: Optional[str] = Field(
      default=None,
      description=(
          'AWS region for the bedrock-agentcore client. If unset, boto3 uses'
          ' the default credential/region chain.'
      ),
  )
