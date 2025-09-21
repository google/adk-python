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

from typing import Union

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .eval_metrics import BaseCriterion
from .eval_metrics import Threshold


class EvalConfig(BaseModel):
  """Configurations needed to run an Eval.

  Allows users to specify metrics, their thresholds and other properties.
  """

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  criteria: dict[str, Union[Threshold, BaseCriterion]] = Field(
      default_factory=dict,
      description="""A dictionary that maps criterion to be used for a metric.

The key of the dictionary is the name of the eval metric and the value is the
criterion to be used.

In the sample below, `tool_trajectory_avg_score`, `response_match_score` and
`final_response_match_v2` are the standard eval metric names, represented as
keys in the dictionary. The values in the dictionary are the corresponding
criterions. For the first two metrics, we use simple threshold as the criterion,
the third one uses `LlmAsAJudgeCriterion`.
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.5,
    "final_response_match_v2": {
      "threshold": 0.5,
      "judge_model_options": {
            "judge_model": "my favorite LLM",
            "num_samples": 5
          }
        }
    },
  }
}
""",
  )
