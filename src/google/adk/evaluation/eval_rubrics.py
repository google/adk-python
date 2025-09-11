# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

from pydantic import Field

from .common import EvalBaseModel


class RubricContent(EvalBaseModel):
  """The content of a rubric."""

  text_property: Optional[str] = Field(
      description=(
          "The property being evaluated. Example: \"The agent's response is"
          ' grammatically correct." '
      )
  )


class Rubric(EvalBaseModel):
  """This class represents a single Rubric."""

  rubric_id: str = Field(
      description="Unique identifier for the rubric.",
  )

  rubric_content: RubricContent = Field(
      description="The actual testable criterion for the rubric."
  )

  description: Optional[str] = Field(
      default=None,
      description=(
          "A description of the rubric that provide details on how the results"
          " of the rubric assessment be interpreted."
      ),
  )

  type: Optional[str] = Field(
      default=None,
      description="""Optional. A type designator for the rubric, which can
      inform how it's evaluated or interpreted by systems or users.

      It's recommended to use consistent, well-defined, upper snake_case
      strings.

      Examples: "TOOL_USE_QUALITY", "FINAL_RESPONSE_QUALITY",
      "INSTRUCTION_ADHERENCE".""",
  )


class RubricScore(EvalBaseModel):
  """The score obtained after applying the rubric to the Agent's response."""

  rubric_id: str = Field(description="The id of the rubric that was assessed.")

  rationale: Optional[str] = Field(
      default=None, description="Reasoning/rationale for the score."
  )

  score: Optional[float] = Field(
      default=None,
      description=(
          "Score obtained after assessing the rubric. Optional, as assessment"
          " might not have happened."
      ),
  )
