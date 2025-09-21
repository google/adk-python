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

import enum
from typing import Optional

from google.genai import types as genai_types

from .evaluator import EvalStatus


@enum.unique
class Label(enum.Enum):
  """Labels for auto rater response."""

  TRUE = "true"
  INVALID = "invalid"
  VALID = "valid"
  PARTIALLY_VALID = "partially_valid", "partially valid", "partially"
  ALMOST = "almost"
  FALSE = "false"
  NOT_FOUND = "label field not found"


def get_text_from_content(
    content: Optional[genai_types.Content],
) -> Optional[str]:
  if content and content.parts:
    return "\n".join([p.text for p in content.parts if p.text])


def get_eval_status(score: Optional[float], threshold: float) -> EvalStatus:
  if score is None:
    return EvalStatus.NOT_EVALUATED
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED
