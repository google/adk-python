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

"""Configuration for retrying a workflow node."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class RetryConfig(BaseModel):
  """Configuration for retrying a node."""

  max_attempts: int | None = Field(
      default=None,
      description="""Maximum number of attempts, including the original request.
      If 0 or 1, it means no retries. If not specified, default to 5.""",
  )
  initial_delay: float | None = Field(
      default=None,
      description="""Initial delay before the first retry, in fractions of a second. If not specified, default to 1.0 second.""",
  )
  max_delay: float | None = Field(
      default=None,
      description="""Maximum delay between retries, in fractions of a second. If not specified, default to 60.0 seconds.""",
  )
  backoff_factor: float | None = Field(
      default=None,
      description="""Multiplier by which the delay increases after each attempt. If not specified, default to 2.0.""",
  )
  jitter: float | None = Field(
      default=None,
      description="""Randomness factor for the delay. If not specified, default to 1.0. Otherwise use 0.0 to remove randomness.""",
  )

  exceptions: list[str | type[BaseException]] | None = Field(
      default=None,
      description="""Exceptions to retry on. Accepts exception class names as
      strings (e.g. ``['ValueError']``) or exception classes directly (e.g.
      ``[ValueError]``). ``None`` means retry on all exceptions.""",
  )

  @field_validator('exceptions', mode='before')
  @classmethod
  def _normalize_exceptions(cls, v: list[Any] | None) -> list[str] | None:
    """Converts exception classes to their class names for uniform handling."""
    if v is None:
      return None
    normalized = []
    for item in v:
      if isinstance(item, str):
        normalized.append(item)
      elif isinstance(item, type) and issubclass(item, BaseException):
        normalized.append(item.__name__)
      else:
        raise ValueError(
            'exceptions must contain exception class names (str) or'
            f' exception classes, got {type(item).__name__}: {item!r}'
        )
    return normalized
