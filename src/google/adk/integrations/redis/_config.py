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

"""Configuration for Redis integrations."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class RedisSessionServiceConfig(BaseModel):
  """Configuration for RedisSessionService."""

  uri: Optional[str] = Field(
      default=None,
      description=(
          "Redis connection URI (e.g. redis://[:password@]host:port/db)."
      ),
  )
  host: Optional[str] = Field(
      default="localhost",
      description="Redis server hostname.",
  )
  port: Optional[int] = Field(
      default=6379,
      description="Redis server port.",
  )
  password: Optional[str] = Field(
      default=None,
      description="Password for Redis authentication.",
  )
  ssl: bool = Field(
      default=False,
      description="Whether to use SSL for Redis connections.",
  )
  db: int = Field(
      default=0,
      description="Redis database index.",
  )
  ttl_seconds: int = Field(
      default=604800,  # 7 days
      description="TTL for session keys in seconds.",
  )
  key_prefix: str = Field(
      default="adk:session:",
      description="Key prefix used for session keys in Redis.",
  )
