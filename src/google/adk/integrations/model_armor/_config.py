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

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import model_validator

_DEFAULT_BLOCKED_MESSAGE = "I'm sorry, but I can't help with that request."


class ModelArmorConfig(BaseModel):
  """Configuration for the Model Armor guardrail plugin."""

  model_config = ConfigDict(extra='forbid')

  prompt_template_name: Optional[str] = None
  """The Model Armor template used to screen user input (prompts).

  Should use the fully-qualified resource name:
  ``projects/{project}/locations/{location}/templates/{template}``.
  If unset, input screening is skipped.
  """

  response_template_name: Optional[str] = None
  """The Model Armor template used to screen model output (responses).

  Should use the fully-qualified resource name:
  ``projects/{project}/locations/{location}/templates/{template}``.
  If unset, output screening is skipped.
  """

  input_blocked_message: str = _DEFAULT_BLOCKED_MESSAGE
  """The safe replacement text returned to the user when user content is blocked."""

  output_blocked_message: str = _DEFAULT_BLOCKED_MESSAGE
  """The safe replacement text returned to the user when model output is blocked."""

  block_on_screening_failure: bool = True
  """Whether to block when Model Armor screening fails."""

  @model_validator(mode='after')
  def _validate_templates(self) -> ModelArmorConfig:
    """Ensure at least one template is configured."""
    if not self.prompt_template_name and not self.response_template_name:
      raise ValueError(
          'At least one of prompt_template_name or response_template_name'
          ' must be set for ModelArmorConfig.'
      )
    return self
