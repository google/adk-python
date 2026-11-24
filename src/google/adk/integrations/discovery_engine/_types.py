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

"""Data types shared by the CheckGrounding integration."""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class FactChunk(BaseModel):
  """A piece of evidence that a grounding check verifies an answer against."""

  model_config = ConfigDict(extra='forbid')

  text: str
  """The evidence text."""

  title: str | None = None
  """Optional title, used for citation attribution."""

  link: str | None = None
  """Optional source URI, used for citation attribution."""

  attributes: dict[str, str] = Field(default_factory=dict)
  """Optional key-value metadata, returned with citations."""


def _content_id(data: dict[str, Any]) -> str:
  key = '\0'.join((
      data.get('title') or '',
      data.get('link') or '',
      json.dumps(data.get('attributes') or {}, sort_keys=True),
      (data.get('text') or '').strip(),
  ))
  return 'fact_' + hashlib.sha256(key.encode()).hexdigest()[:16]


class Fact(BaseModel):
  """A piece of evidence that responses are checked against.

  The plugin splits ``text`` into chunks that fit the backend's limits.
  """

  model_config = ConfigDict(extra='forbid')

  text: str
  """The fact text, of any length."""

  title: str | None = None
  """Optional title, used for citation attribution."""

  link: str | None = None
  """Optional source URI, used for citation attribution."""

  attributes: dict[str, str] = Field(default_factory=dict)
  """Optional key-value metadata, returned with citations."""

  id: str = Field(default_factory=_content_id)
  """Identifier. Defaults to a hash of the text, title, link and attributes,
  so the same content always gets the same ID."""

  @field_validator('text')
  @classmethod
  def _text_not_blank(cls, text: str) -> str:
    if not text.strip():
      raise ValueError('Fact text must not be empty.')
    return text


class GroundingLimits(BaseModel):
  """Capacity limits that a grounding backend enforces.

  The defaults match the Discovery Engine CheckGrounding API.
  """

  model_config = ConfigDict(frozen=True, extra='forbid')

  max_facts: int = Field(default=200, gt=0)
  """Maximum number of fact chunks evaluated per check."""

  max_fact_chars: int = Field(default=10_000, gt=0)
  """Maximum character length of a single fact chunk."""


class GroundingStatus(enum.StrEnum):
  """Outcome of a single grounding check."""

  OK = 'ok'
  """The check ran and returned grounding metadata."""

  SKIPPED_NO_FACTS = 'skipped_no_facts'
  """Skipped because there were no fact chunks to ground against."""

  SKIPPED_EMPTY = 'skipped_empty'
  """Skipped because the answer had no non-whitespace text."""

  REJECTED_TOO_LONG = 'rejected_too_long'
  """The backend rejected the answer for exceeding its token limit."""

  API_ERROR = 'api_error'
  """The backend call failed for any other reason."""


class GroundingResult(BaseModel):
  """Outcome of a grounding check and the signals it produced."""

  model_config = ConfigDict(extra='forbid')

  status: GroundingStatus
  """Outcome of the check."""

  num_facts_evaluated: int
  """Number of fact chunks sent to the backend."""

  metadata: types.GroundingMetadata | None = None
  """Citation metadata. Set only when ``status`` is ``OK``."""

  support_score: float | None = None
  """Answer-level support score, when the backend returned one."""

  num_claims: int | None = None
  """Number of claims in the answer that required a grounding check."""

  error: str | None = None
  """Error message for ``REJECTED_TOO_LONG`` and ``API_ERROR``."""
