# Copyright 2026 Google LLC
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
from typing import Any
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
from .eval_metrics import MetricInfo
from .eval_metrics import Threshold
from .simulation._llm_audio_user_simulator import LlmAudioUserSimulatorConfig
from .simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig

logger = logging.getLogger("google_adk." + __name__)

# The set of user-simulator config subclasses that `EvalConfig` can
# deserialize into via the `type` discriminator. Add any new subclass to
# this Union (each with a unique `Literal[...]` for its `type` field).
_UserSimulatorConfig = Union[
    LlmBackedUserSimulatorConfig, LlmAudioUserSimulatorConfig
]

# Legacy default preserved for backward compatibility with eval configs authored
# before the `type` discriminator existed. See
# `EvalConfig._inject_default_user_simulator_type` below.
_LEGACY_DEFAULT_USER_SIMULATOR_TYPE = "llm_backed"


class CustomMetricConfig(BaseModel):
  """Configuration for a custom metric."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  code_config: CodeConfig = Field(
      description=(
          "Code config for the custom metric, used to locate the custom metric"
          " function."
      )
  )
  metric_info: Optional[MetricInfo] = Field(
      default=None,
      description="Metric info for the custom metric.",
  )
  description: str = Field(
      default="",
      description="Description for the custom metric info.",
  )


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
criteria. For the first two metrics, we use simple threshold as the criterion,
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

  custom_metrics: Optional[dict[str, CustomMetricConfig]] = Field(
      default=None,
      description="""A dictionary mapping custom metric names to
a CustomMetricConfig object.

If a metric name in `criteria` is also present in `custom_metrics`, the
`code_config` in `CustomMetricConfig` will be used to locate the custom metric
implementation.

The `metric` field in `CustomMetricConfig` can be used to provide metric
information like `min_value`, `max_value`, and `description`. If `metric`
is not provided, a default `MetricInfo` will be created, using
`description` from `CustomMetricConfig` if provided, and default values
for `min_value` (0.0) and `max_value` (1.0).

Example:
{
  "criteria": {
    "my_custom_metric": 0.5,
    "my_simple_metric": 0.8
  },
  "custom_metrics": {
    "my_simple_metric": {
      "code_config": {
        "name": "path.to.my.simple.metric.function"
      }
    },
    "my_custom_metric": {
      "code_config": {
        "name": "path.to.my.custom.metric.function"
      },
      "metric": {
        "metric_name": "my_custom_metric",
        "min_value": -10.0,
        "max_value": 10.0,
        "description": "My custom metric."
      }
    }
  }
}
""",
  )

  user_simulator_config: Optional[_UserSimulatorConfig] = Field(
      default=None,
      discriminator="type",
      description=(
          "Config to be used by the user simulator. When authored as JSON,"
          " the concrete subclass is selected via the `type` discriminator"
          ' field (e.g. `{"type": "llm_backed", ...}`). Configs that'
          " predate the `type` field are treated as"
          f' `type="{_LEGACY_DEFAULT_USER_SIMULATOR_TYPE}"` for backward'
          " compatibility."
      ),
  )

  @model_validator(mode="before")
  @classmethod
  def _inject_default_user_simulator_type(cls, values: Any) -> Any:
    """Inject the legacy default `type` when a JSON config predates the

    discriminator field.

    Without this validator, existing configs that never carried a `type`
    key would fail validation with `union_tag_not_found`. Here we silently
    treat a missing `type` as the legacy default so existing files keep
    working. Configs that DO carry `type` are left untouched.
    """
    if not isinstance(values, dict):
      return values
    # Handle both snake_case and camelCase spellings (this model uses
    # `alias_generator=to_camel`).
    for key in ("user_simulator_config", "userSimulatorConfig"):
      inner = values.get(key)
      # Treat a missing key AND an explicit `type=None` (e.g. from a
      # `BaseUserSimulatorConfig().model_dump()`) both as "no discriminator
      # supplied" so backward-compat is preserved either way.
      if isinstance(inner, dict) and inner.get("type") is None:
        logger.info(
            "eval_config.%s has no `type` discriminator; defaulting to"
            ' \'%s\'. Add `"type": "%s"` to your config to make this'
            " explicit.",
            key,
            _LEGACY_DEFAULT_USER_SIMULATOR_TYPE,
            _LEGACY_DEFAULT_USER_SIMULATOR_TYPE,
        )
        values = {
            **values,
            key: {**inner, "type": _LEGACY_DEFAULT_USER_SIMULATOR_TYPE},
        }
    return values


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
      if eval_config.custom_metrics and (
          config := eval_config.custom_metrics.get(metric_name)
      ):
        custom_function_path = config.code_config.name

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
