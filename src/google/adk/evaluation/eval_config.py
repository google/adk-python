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

import logging
import os
from typing import Optional
from typing import Union

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from ..agents.common_configs import CodeConfig
from ..evaluation.eval_metrics import EvalMetric
from .eval_metrics import BaseCriterion
from .eval_metrics import Threshold
from .simulation.user_simulator import BaseUserSimulatorConfig

logger = logging.getLogger("google_adk." + __name__)


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

  custom_metrics: Optional[dict[str, CodeConfig]] = Field(
      default=None,
      description="""A dictionary mapping custom metric names to CodeConfig
objects, which specify the path to the function for each custom metric.

If a metric name in `criteria` is also present in `custom_metrics`, the
corresponding `CodeConfig`'s `name` field will be used to locate the custom
metric implementation. The `name` field should contain the fully qualified
path to the custom metric function, e.g., `my.custom.metrics.metric_function`.

Example:
{
  "criteria": {
    "my_custom_metric": 0.5
  },
  "custom_metrics": {
    "my_custom_metric": {
      "name": "path.to.my.custom.metric.function"
    }
  }
}
""",
  )

  user_simulator_config: Optional[BaseUserSimulatorConfig] = Field(
      default=None,
      description="Config to be used by the user simulator.",
  )

  @model_validator(mode="after")
  def check_custom_metrics_code_config_args(self) -> "EvalConfig":
    if self.custom_metrics:
      for metric_name, metric_config in self.custom_metrics.items():
        if metric_config.args:
          raise ValueError(
              f"args field in CodeConfig for custom metric '{metric_name}' is"
              " not supported."
          )
    return self


_DEFAULT_EVAL_CONFIG = EvalConfig(
    criteria={"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}
)


def get_evaluation_criteria_or_default(
    eval_config_file_path: Optional[str],
) -> EvalConfig:
  """Returns EvalConfig read from the config file, if present.

  Otherwise a default one is returned.
  """
  if eval_config_file_path and os.path.exists(eval_config_file_path):
    with open(eval_config_file_path, "r", encoding="utf-8") as f:
      content = f.read()
      return EvalConfig.model_validate_json(content)

  logger.info(
      "No config file supplied or file not found. Using default criteria."
  )
  return _DEFAULT_EVAL_CONFIG


def get_eval_metrics_from_config(eval_config: EvalConfig) -> list[EvalMetric]:
  """Returns a list of EvalMetrics mapped from the EvalConfig."""
  eval_metric_list = []
  if eval_config.criteria:
    for metric_name, criterion in eval_config.criteria.items():
      custom_function_path = None
      if (
          eval_config.custom_metrics
          and metric_name in eval_config.custom_metrics
      ):
        custom_function_path = eval_config.custom_metrics[metric_name].name

      if isinstance(criterion, float):
        eval_metric_list.append(
            EvalMetric(
                metric_name=metric_name,
                threshold=criterion,
                criterion=BaseCriterion(threshold=criterion),
                custom_function_path=custom_function_path,
            )
        )
      elif isinstance(criterion, BaseCriterion):
        eval_metric_list.append(
            EvalMetric(
                metric_name=metric_name,
                threshold=criterion.threshold,
                criterion=criterion,
                custom_function_path=custom_function_path,
            )
        )
      else:
        raise ValueError(
            f"Unexpected criterion type. {type(criterion).__name__} not"
            " supported."
        )

  return eval_metric_list
