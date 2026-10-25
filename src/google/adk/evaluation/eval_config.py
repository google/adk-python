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
from typing import Annotated
from typing import Any
from typing import Optional
from typing import Union

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from pydantic import SerializeAsAny

from ..agents.common_configs import CodeConfig
from ..evaluation.eval_metrics import EvalMetric
from ._multi_run_aggregation_methods import get_built_in_aggregation_methods
from .constants import DEFAULT_LIVE_TIMEOUT_SECONDS
from .eval_metrics import BaseCriterion
from .eval_metrics import MetricInfo
from .eval_metrics import PrebuiltMetrics
from .eval_metrics import Threshold
from .simulation._llm_audio_user_simulator import LlmAudioUserSimulatorConfig
from .simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig

logger = logging.getLogger("google_adk." + __name__)

# The set of user-simulator config subclasses that `EvalConfig` can
# deserialize into via the `type` discriminator. Add any new subclass to
# this Union (each with a unique `Literal[...]` for its `type` field).
_UserSimulatorConfig = Annotated[
    Union[LlmBackedUserSimulatorConfig, LlmAudioUserSimulatorConfig],
    Field(discriminator="type"),
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


class LiveModelConfig(BaseModel):
  """Configuration for evaluating models in Live (bidirectional streaming) mode."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  timeout_seconds: int = Field(
      default=DEFAULT_LIVE_TIMEOUT_SECONDS,
      description=(
          "Timeout in seconds for waiting for model turn completion in"
          " live mode."
      ),
  )


class MultiRunCriterion(BaseModel):
  """A rule that counts the runs that passed, with a threshold of its own.

  Each run is first judged on its own, as a single run would be. The method
  turns those verdicts into one number, such as the fraction of runs that
  passed, and the rule passes when that number is at least `threshold`.
  """

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      # A misspelled key would otherwise be dropped, silently changing the
      # rule.
      extra="forbid",
      use_attribute_docstrings=True,
  )

  method: str
  """The name of a method that counts passing runs, such as "pass_rate".

  It can also name a method declared in `custom_methods`.
  """

  threshold: float = Field(ge=0.0, le=1.0)
  """The lowest value of `method` that passes, from 0.0 to 1.0."""


class CustomCrossRunMethodConfig(BaseModel):
  """A cross-run method of your own, written as a Python function.

  Like a built-in method, it only turns an eval case's runs into one number.
  Comparing that number with a threshold, and deciding the verdict, are done
  as for a built-in method of the same kind.
  """

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      # A misspelled key would otherwise be dropped, silently changing the
      # method.
      extra="forbid",
      use_attribute_docstrings=True,
  )

  code_config: CodeConfig
  """The fully qualified name of the function, e.g. "my_project.methods.p25".

  The function receives a sequence of `RunResult`, one target's result in each
  run of the eval case, where the target is a metric or the whole eval case.
  It returns a number, or None when the runs allow no value.
  """

  uses_scores: bool
  """Whether the method combines scores, rather than counting passing runs.

  True: it combines scores, so it is written as a plain name in
  `cross_run_scoring` or `cross_run_scoring_by_metric`, and its value is judged
  against the metric's threshold in `criteria`. False: it counts passing runs,
  so it is written as an object with a threshold of its own, and can also be
  used in `cross_run_criteria`.
  """


class MultiRunConfig(BaseModel):
  """Configuration for running each eval case more than once.

  An agent can answer the same eval case differently from one run to the next,
  so a single run can pass or fail by chance. Running each case several times
  and judging it on all of its runs gives a steadier verdict.
  """

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      # A misspelled key would otherwise be dropped, silently running each
      # eval case once.
      extra="forbid",
  )

  num_runs: Optional[int] = Field(
      default=None,
      ge=1,
      description=(
          "Number of times to run each eval case, each in a fresh session. If"
          " unset, `adk eval` uses its `--num_runs` flag, or 1."
      ),
  )

  max_crashed_runs: int = Field(
      default=0,
      ge=0,
      description=(
          "How many runs of an eval case may crash, failing without producing"
          " any metric results, before the case fails. A crashed run has no"
          " verdict, so it is left out of every count of passing runs, and a"
          " case needs at least one run that did not crash to pass. The"
          " default, 0, fails a case on any crash. A case run once with no"
          " rule that counts passing runs keeps that run's own verdict, so a"
          " crash fails it whatever this is."
      ),
  )

  cross_run_scoring: Union[str, MultiRunCriterion] = Field(
      default="mean_over_invocations",
      description=(
          "How each metric's run scores become its case score, unless"
          " `cross_run_scoring_by_metric` names the metric. A string names a"
          " method that combines scores, and the metric passes when the result"
          ' meets its threshold in `criteria`: "mean_over_invocations"'
          " averages every invocation of every run, a run scored only as a"
          ' whole counting once with its overall score; "mean_of_runs"'
          ' averages the overall scores of the runs; and "min_over_runs"'
          " takes the worst run's score: the lowest, or the highest for a"
          " metric where lower is better. An object, such as"
          ' {"method": "pass_rate", "threshold": 0.8}, names a method that'
          " counts the runs in which the metric passed, and carries its own"
          " threshold."
      ),
  )

  cross_run_scoring_by_metric: Optional[
      dict[str, Union[str, MultiRunCriterion]]
  ] = Field(
      default=None,
      description=(
          "Like `cross_run_scoring`, for individual metrics, keyed by metric"
          " name. Each value takes either form of `cross_run_scoring` and"
          " replaces it for that metric. Every key must be a metric in"
          " `criteria`."
      ),
  )

  cross_run_criteria: Optional[list[MultiRunCriterion]] = Field(
      default=None,
      description=(
          "Rules that judge the eval case on its runs as a whole, on top of"
          " the per-metric rules. A run counts as passed when it passed and"
          " every one of its metrics was evaluated, and as failed when a"
          " metric failed. Any other run is left out, as is a run that"
          " crashed without producing any metric results. Each entry names a"
          ' method that counts passing runs, such as "pass_rate", and its'
          " threshold. The case fails if any entry fails."
      ),
  )

  custom_methods: Optional[dict[str, CustomCrossRunMethodConfig]] = Field(
      default=None,
      description=(
          "Methods of your own, keyed by name. `cross_run_scoring`,"
          " `cross_run_scoring_by_metric` and `cross_run_criteria` refer to a"
          " custom method by this name, as they do to a built-in one, and the"
          " output shows it under this name. A name cannot be that of a"
          " built-in method. The functions are imported when the eval runs,"
          " not when the config is read."
      ),
  )

  @model_validator(mode="after")
  def _validate_methods(self) -> MultiRunConfig:
    """Rejects a rule whose method is unknown or written in the wrong form.

    A method that combines scores is written as a plain name and judged
    against the metric's threshold in `criteria`. A method that counts passing
    runs is written as an object with a threshold of its own, and only such a
    method can judge the whole eval case. Every rule is checked here, not on
    its own field, because a rule can name a custom method. A custom method's
    kind is read from its `uses_scores`, so its function is not imported.
    """
    custom_methods = self.custom_methods or {}
    built_in_methods = get_built_in_aggregation_methods()
    for name in custom_methods:
      if name in built_in_methods:
        raise ValueError(
            f"custom_methods[{name!r}]: {name!r} is the name of a"
            " built-in method. Give the custom method another name."
        )
    uses_scores = {
        name: method.uses_scores for name, method in built_in_methods.items()
    }
    uses_scores.update(
        (name, config.uses_scores) for name, config in custom_methods.items()
    )

    # Each rule as where it is written, the method it names, and whether it is
    # written as an object, with a threshold of its own.
    rules: list[tuple[str, str, bool]] = []
    metric_rules = [
        ("cross_run_scoring", self.cross_run_scoring),
        *(
            (f"cross_run_scoring_by_metric[{metric_name!r}]", rule)
            for metric_name, rule in (
                self.cross_run_scoring_by_metric or {}
            ).items()
        ),
    ]
    for where, rule in metric_rules:
      if isinstance(rule, str):
        rules.append((where, rule, False))
      elif isinstance(rule, MultiRunCriterion):
        rules.append((where, rule.method, True))
      else:
        raise TypeError(f"Unsupported aggregation rule: {rule!r}.")
    rules.extend(
        (f"cross_run_criteria[{index}]", criterion.method, True)
        for index, criterion in enumerate(self.cross_run_criteria or [])
    )

    for where, method, has_threshold in rules:
      if method not in uses_scores:
        message = (
            f"{where}: Unknown method: {method!r}. Built-in"
            f" methods: {sorted(built_in_methods)}."
        )
        if custom_methods:
          message += f" Custom methods: {sorted(custom_methods)}."
        raise ValueError(message)
      if uses_scores[method] and has_threshold:
        raise ValueError(
            f"{where}: {method!r} combines scores, so it is judged against the"
            " metric's threshold in `criteria` and cannot have a threshold of"
            " its own. Write it as a plain name in `cross_run_scoring` or"
            " `cross_run_scoring_by_metric`, or use a method that counts"
            " passing runs, such as 'pass_rate'."
        )
      if not uses_scores[method] and not has_threshold:
        raise ValueError(
            f"{where}: {method!r} counts passing runs, so it needs a threshold"
            f' of its own: write {{"method": "{method}", "threshold": ...}}'
            " instead of the plain name."
        )
    return self


class EvalConfig(BaseModel):
  """Configurations needed to run an Eval.

  Allows users to specify metrics, their thresholds and other properties.
  """

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  criteria: dict[str, Union[Threshold, SerializeAsAny[BaseCriterion]]] = Field(
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

The `metric_info` field in `CustomMetricConfig` can be used to provide metric
information like `description` and the value range in
`metric_value_info.interval`. If `metric_info` is not provided, a default
`MetricInfo` will be created, using
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
      "metric_info": {
        "metric_name": "my_custom_metric",
        "metric_value_info": {
          "interval": {
            "min_value": -10.0,
            "max_value": 10.0
          }
        },
        "description": "My custom metric."
      }
    }
  }
}
""",
  )

  user_simulator_config: Optional[_UserSimulatorConfig] = Field(
      default=None,
      description=(
          "Config to be used by the user simulator. When authored as JSON,"
          " the concrete subclass is selected via the `type` discriminator"
          ' field (e.g. `{"type": "llm_backed", ...}`). Configs that'
          " predate the `type` field are treated as"
          f' `type="{_LEGACY_DEFAULT_USER_SIMULATOR_TYPE}"` for backward'
          " compatibility."
      ),
  )

  live_model_config: Optional[LiveModelConfig] = Field(
      default=None,
      description=(
          "Config for evaluating in live (bidirectional streaming) mode."
          " Required for Live API models (e.g. `gemini-*-live-*`)."
      ),
  )

  multi_run_config: Optional[MultiRunConfig] = Field(
      default=None,
      description="""Config for running each eval case more than once.

When `num_runs` is greater than 1, each eval case is run that many times, each
time in a fresh session. The result of every run is saved, and the case is
judged on all of its runs.

How the runs are combined can be configured in three ways, which can be used
together:
- `cross_run_scoring` combines each metric's runs: by default the mean of its
  per-invocation scores over every run, a run scored only as a whole counting
  once with its overall score, compared with its threshold in `criteria`. A
  string names a method that combines scores; an object names a method that
  counts the runs in which the metric passed, with its own threshold. The
  metrics then decide the case as they do for a single run.
- `cross_run_scoring_by_metric` replaces `cross_run_scoring` for the metrics
  it names.
- `cross_run_criteria` adds rules that count the runs that passed as a whole.
  The case fails if any of them fails.
Without these fields, each metric's runs are combined by the default above.
Besides the built-in methods, these fields can name methods of your own:
Python functions declared in `custom_methods`.

A run that crashed, failing without producing any metric results, has no
verdict, so all of these leave it out. Instead the case fails if more of its
runs crashed than `max_crashed_runs` allows, by default any, and it needs at
least one run that did not crash to pass.

In the sample below, each eval case is run 3 times. `response_match_score`
passes if no run scored below 0.8, `tool_trajectory_avg_score` passes if it
passed in at least 90% of the runs, and at least 80% of the runs must pass as a
whole.
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.8
  },
  "multi_run_config": {
    "num_runs": 3,
    "cross_run_scoring_by_metric": {
      "response_match_score": "min_over_runs",
      "tool_trajectory_avg_score": {"method": "pass_rate", "threshold": 0.9}
    },
    "cross_run_criteria": [{"method": "pass_rate", "threshold": 0.8}]
  }
}

`adk eval` reads this field; `--num_runs` overrides `num_runs`, and if neither
is set each case runs once. `AgentEvaluator` and `adk web` do not read it yet;
`AgentEvaluator` takes the number of runs as its own `num_runs` argument.
""",
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

  @model_validator(mode="after")
  def _validate_cross_run_scoring_by_metric(self) -> EvalConfig:
    """Rejects a multi-run rule for a metric that is not in `criteria`.

    Only the metrics in `criteria` can be configured, so a rule for any other
    name, such as a misspelled one, would silently never apply.
    """
    if self.multi_run_config is None:
      return self
    rules = self.multi_run_config.cross_run_scoring_by_metric or {}
    unknown_metrics = sorted(set(rules) - set(self.criteria))
    if unknown_metrics:
      raise ValueError(
          "multi_run_config.cross_run_scoring_by_metric has rules for metrics"
          f" that are not in criteria: {unknown_metrics}."
      )
    return self


_DEFAULT_EVAL_CONFIG = EvalConfig(
    criteria={"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}
)

# Informational efficiency metrics that are reported automatically for every
# eval, without the user having to enable them in the config. They are
# reference-free and never pass or fail (their status is always
# INFORMATIONAL); they simply report a value for the user to track. Because
# they require no configuration, they are always on and cannot be turned off.
_DEFAULT_EFFICIENCY_METRICS: tuple[str, ...] = (
    PrebuiltMetrics.TOOL_CALL_COUNT_V1.value,
    PrebuiltMetrics.INFERENCE_CALL_COUNT_V1.value,
    PrebuiltMetrics.TOKEN_USAGE_V1.value,
    PrebuiltMetrics.INVOCATION_DURATION_V1.value,
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


def append_default_efficiency_metrics(
    eval_metrics: list[EvalMetric],
) -> list[EvalMetric]:
  """Returns `eval_metrics` with the informational efficiency metrics added.

  Efficiency is reported for every eval without the caller asking for it, so
  every entry point that assembles a metric list runs the list through here.
  A metric the caller already named is left as it is rather than replaced, so
  the caller's own entry -- and the error it earns for carrying a threshold --
  survives, and no duplicate is added.

  Args:
    eval_metrics: The metrics the caller asked for. Not modified.

  Returns:
    A new list: the caller's metrics, then the efficiency metrics they did not
    already name.
  """
  requested = {eval_metric.metric_name for eval_metric in eval_metrics}
  return list(eval_metrics) + [
      EvalMetric(metric_name=metric_name)
      for metric_name in _DEFAULT_EFFICIENCY_METRICS
      if metric_name not in requested
  ]


def get_eval_metrics_from_config(eval_config: EvalConfig) -> list[EvalMetric]:
  """Returns a list of EvalMetrics mapped from the EvalConfig.

  In addition to the metrics explicitly configured in `eval_config.criteria`,
  the informational efficiency metrics in `_DEFAULT_EFFICIENCY_METRICS` are
  always appended, so that efficiency is reported for every eval without any
  configuration. These metrics never pass or fail.

  They cannot be configured, though: naming one in `criteria` means giving it a
  threshold, and these metrics reject a threshold, so the eval fails with a
  `ValueError`. The entry is still not duplicated here, so the failure names the
  metric once.
  """
  eval_metric_list = []
  if eval_config.criteria:
    for metric_name, criterion in eval_config.criteria.items():
      custom_function_path = None
      if eval_config.custom_metrics and (
          config := eval_config.custom_metrics.get(metric_name)
      ):
        custom_function_path = config.code_config.name

      if isinstance(criterion, float):
        eval_metric = EvalMetric(
            metric_name=metric_name,
            threshold=criterion,
            criterion=BaseCriterion(threshold=criterion),
            custom_function_path=custom_function_path,
        )
      elif isinstance(criterion, BaseCriterion):
        eval_metric = EvalMetric(
            metric_name=metric_name,
            threshold=criterion.threshold,
            criterion=criterion,
            custom_function_path=custom_function_path,
        )
      else:
        raise ValueError(
            f"Unexpected criterion type. {type(criterion).__name__} not"
            " supported."
        )

      # The config is written by the developer running the eval, so the path it
      # declares is the one honoured when the metric runs. It travels with the
      # metric rather than in a registry keyed by metric name, so two apps in
      # one process can declare the same metric name and each still gets its
      # own function.
      eval_metric._config_custom_function_path = custom_function_path  # pylint: disable=protected-access
      eval_metric_list.append(eval_metric)

  return append_default_efficiency_metrics(eval_metric_list)
