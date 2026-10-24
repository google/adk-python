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

import json
import re

from google.adk.agents.common_configs import CodeConfig
from google.adk.evaluation.eval_config import _DEFAULT_EFFICIENCY_METRICS
from google.adk.evaluation.eval_config import _DEFAULT_EVAL_CONFIG
from google.adk.evaluation.eval_config import append_default_efficiency_metrics
from google.adk.evaluation.eval_config import CustomCrossRunMethodConfig
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_config import get_evaluation_criteria_or_default
from google.adk.evaluation.eval_config import MultiRunConfig
from google.adk.evaluation.eval_config import MultiRunCriterion
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import LlmAsAJudgeCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.simulation._llm_audio_user_simulator import LlmAudioUserSimulatorConfig
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from pydantic import ValidationError
import pytest


def test_get_evaluation_criteria_or_default_returns_default():
  assert get_evaluation_criteria_or_default("") == _DEFAULT_EVAL_CONFIG


def test_get_evaluation_criteria_or_default_reads_from_file(mocker):
  mocker.patch("os.path.exists", return_value=True)
  eval_config = EvalConfig(
      criteria={"tool_trajectory_avg_score": 0.5, "response_match_score": 0.5}
  )
  mocker.patch(
      "builtins.open", mocker.mock_open(read_data=eval_config.model_dump_json())
  )
  assert get_evaluation_criteria_or_default("dummy_path") == eval_config


def test_get_evaluation_criteria_or_default_returns_default_if_file_not_found(
    mocker,
):
  mocker.patch("os.path.exists", return_value=False)
  assert (
      get_evaluation_criteria_or_default("dummy_path") == _DEFAULT_EVAL_CONFIG
  )


def test_get_eval_metrics_from_config():
  rubric_1 = Rubric(
      rubric_id="test-rubric",
      rubric_content=RubricContent(text_property="test"),
  )
  eval_config = EvalConfig(
      criteria={
          "tool_trajectory_avg_score": 1.0,
          "response_match_score": 0.8,
          "final_response_match_v2": {
              "threshold": 0.5,
              "judge_model_options": {
                  "judge_model": "gemini-pro",
                  "num_samples": 1,
              },
          },
          "rubric_based_final_response_quality_v1": {
              "threshold": 0.9,
              "judge_model_options": {
                  "judge_model": "gemini-ultra",
                  "num_samples": 1,
              },
              "rubrics": [rubric_1],
          },
      }
  )
  eval_metrics = get_eval_metrics_from_config(eval_config)

  # The configured metrics come first, in declaration order, followed by the
  # auto-appended informational efficiency metrics.
  assert len(eval_metrics) == 4 + len(_DEFAULT_EFFICIENCY_METRICS)
  assert eval_metrics[0].metric_name == "tool_trajectory_avg_score"
  assert eval_metrics[0].threshold == 1.0
  assert eval_metrics[0].criterion.threshold == 1.0
  assert eval_metrics[1].metric_name == "response_match_score"
  assert eval_metrics[1].threshold == 0.8
  assert eval_metrics[1].criterion.threshold == 0.8
  assert eval_metrics[2].metric_name == "final_response_match_v2"
  assert eval_metrics[2].threshold == 0.5
  assert eval_metrics[2].criterion.threshold == 0.5
  assert (
      eval_metrics[2].criterion.judge_model_options["judge_model"]
      == "gemini-pro"
  )
  assert eval_metrics[3].metric_name == "rubric_based_final_response_quality_v1"
  assert eval_metrics[3].threshold == 0.9
  assert eval_metrics[3].criterion.threshold == 0.9
  assert (
      eval_metrics[3].criterion.judge_model_options["judge_model"]
      == "gemini-ultra"
  )
  assert len(eval_metrics[3].criterion.rubrics) == 1
  assert eval_metrics[3].criterion.rubrics[0] == rubric_1
  assert [m.metric_name for m in eval_metrics[4:]] == list(
      _DEFAULT_EFFICIENCY_METRICS
  )


def test_get_eval_metrics_from_config_with_custom_metrics():
  eval_config = EvalConfig(
      criteria={
          "custom_metric_1": 1.0,
          "custom_metric_2": {
              "threshold": 0.5,
          },
      },
      custom_metrics={
          "custom_metric_1": {
              "code_config": {"name": "path/to/custom/metric_1"},
          },
          "custom_metric_2": {
              "code_config": {"name": "path/to/custom/metric_2"},
          },
      },
  )
  eval_metrics = get_eval_metrics_from_config(eval_config)

  # The 2 configured custom metrics come first, then the auto-appended
  # informational efficiency metrics (which carry no custom function path).
  assert len(eval_metrics) == 2 + len(_DEFAULT_EFFICIENCY_METRICS)
  assert eval_metrics[0].metric_name == "custom_metric_1"
  assert eval_metrics[0].threshold == 1.0
  assert eval_metrics[0].criterion.threshold == 1.0
  assert eval_metrics[0].custom_function_path == "path/to/custom/metric_1"
  assert eval_metrics[1].metric_name == "custom_metric_2"
  assert eval_metrics[1].threshold == 0.5
  assert eval_metrics[1].criterion.threshold == 0.5
  assert eval_metrics[1].custom_function_path == "path/to/custom/metric_2"
  assert [m.metric_name for m in eval_metrics[2:]] == list(
      _DEFAULT_EFFICIENCY_METRICS
  )
  assert all(m.custom_function_path is None for m in eval_metrics[2:])


def test_get_eval_metrics_from_config_empty_criteria():
  eval_config = EvalConfig(criteria={})
  eval_metrics = get_eval_metrics_from_config(eval_config)
  # Even with no criteria, the informational efficiency metrics are reported.
  assert [m.metric_name for m in eval_metrics] == list(
      _DEFAULT_EFFICIENCY_METRICS
  )


def test_efficiency_metrics_are_appended_and_informational():
  """The efficiency metrics are always added, with no threshold or criterion."""
  eval_metrics = get_eval_metrics_from_config(
      EvalConfig(criteria={"response_match_score": 0.8})
  )

  names = [m.metric_name for m in eval_metrics]
  for metric_name in _DEFAULT_EFFICIENCY_METRICS:
    assert metric_name in names

  efficiency_metrics = [
      m for m in eval_metrics if m.metric_name in _DEFAULT_EFFICIENCY_METRICS
  ]
  # Informational: no threshold or criterion is attached, so they never gate
  # pass/fail.
  assert all(m.threshold is None for m in efficiency_metrics)
  assert all(m.criterion is None for m in efficiency_metrics)


def test_named_efficiency_metric_is_not_duplicated():
  """Naming an always-on metric in the config does not add a second copy.

  The threshold is what makes the entry parse; the metric rejects one later,
  at evaluator construction, which is not what this test covers.
  """
  eval_config = EvalConfig(criteria={"token_usage_v1": 1.0})

  eval_metrics = get_eval_metrics_from_config(eval_config)

  token_usage_metrics = [
      m for m in eval_metrics if m.metric_name == "token_usage_v1"
  ]
  assert len(token_usage_metrics) == 1
  # The other efficiency metrics are still auto-appended.
  names = [m.metric_name for m in eval_metrics]
  for metric_name in _DEFAULT_EFFICIENCY_METRICS:
    assert metric_name in names


def test_append_default_efficiency_metrics_adds_them_to_a_plain_list():
  """The Dev UI hands over a list, not a config, and must get them too."""
  requested = [
      EvalMetric(metric_name="tool_trajectory_avg_score", threshold=1.0)
  ]

  eval_metrics = append_default_efficiency_metrics(requested)

  assert [m.metric_name for m in eval_metrics] == [
      "tool_trajectory_avg_score",
      *_DEFAULT_EFFICIENCY_METRICS,
  ]
  # Informational: appended without a threshold, so they never gate pass/fail.
  appended = eval_metrics[1:]
  assert all(m.threshold is None for m in appended)
  assert all(m.criterion is None for m in appended)


def test_append_default_efficiency_metrics_keeps_the_caller_entry():
  """A metric the caller named is left alone, not replaced or duplicated.

  The caller's entry is what carries the threshold that the metric later
  rejects, so replacing it here would swallow the error.
  """
  caller_entry = EvalMetric(metric_name="token_usage_v1", threshold=1.0)

  eval_metrics = append_default_efficiency_metrics([caller_entry])

  assert [m.metric_name for m in eval_metrics].count("token_usage_v1") == 1
  assert eval_metrics[0] is caller_entry


def test_append_default_efficiency_metrics_does_not_mutate_its_argument():
  requested = [EvalMetric(metric_name="safety_v1", threshold=0.5)]

  append_default_efficiency_metrics(requested)

  assert [m.metric_name for m in requested] == ["safety_v1"]


def test_eval_metric_dump_preserves_concrete_criterion_fields():
  """Serializing a metric must not degrade its criterion to the base class."""
  eval_metric = EvalMetric(
      metric_name="final_response_match_v2",
      criterion=LlmAsAJudgeCriterion(
          threshold=0.8,
          judge_model_options=JudgeModelOptions(
              judge_model="my-judge", num_samples=3
          ),
      ),
  )

  dumped = eval_metric.model_dump()

  assert dumped["criterion"]["judge_model_options"]["judge_model"] == "my-judge"
  assert dumped["criterion"]["judge_model_options"]["num_samples"] == 3


def test_eval_metric_criterion_survives_json_round_trip():
  """A serialized metric still yields its concrete criterion when reloaded."""
  eval_metric = EvalMetric(
      metric_name="final_response_match_v2",
      criterion=LlmAsAJudgeCriterion(
          threshold=0.8,
          judge_model_options=JudgeModelOptions(judge_model="my-judge"),
      ),
  )

  restored = EvalMetric.model_validate_json(eval_metric.model_dump_json())
  criterion = LlmAsAJudgeCriterion.model_validate(
      restored.criterion.model_dump()
  )

  assert criterion.judge_model_options.judge_model == "my-judge"


@pytest.mark.parametrize("num_samples", [0, -1])
def test_judge_model_options_rejects_non_positive_num_samples(num_samples):
  """num_samples <= 0 must be rejected, matching parallelism_limit's own ge=1.

  A non-positive judge sample count is never a legitimate value -- it causes
  LlmAsJudge.evaluate_invocations to silently drop the invocation from the
  aggregated result with no error and no NOT_EVALUATED marker.
  """
  with pytest.raises(ValidationError, match="greater than or equal to 1"):
    JudgeModelOptions(num_samples=num_samples)


def test_eval_config_dump_preserves_concrete_criterion_fields():
  """Criteria values keep their subclass fields, and plain thresholds survive."""
  eval_config = EvalConfig(
      criteria={
          "tool_trajectory_avg_score": 1.0,
          "final_response_match_v2": LlmAsAJudgeCriterion(
              threshold=0.8,
              judge_model_options=JudgeModelOptions(judge_model="my-judge"),
          ),
      }
  )

  dumped = eval_config.model_dump()

  assert dumped["criteria"]["tool_trajectory_avg_score"] == 1.0
  assert (
      dumped["criteria"]["final_response_match_v2"]["judge_model_options"][
          "judge_model"
      ]
      == "my-judge"
  )


# -----------------------------------------------------------------------------
# `user_simulator_config` discriminator + backward-compat coverage
# -----------------------------------------------------------------------------


def test_user_simulator_config_default_is_none():
  """A brand-new EvalConfig has no user simulator config by default."""
  eval_config = EvalConfig()
  assert eval_config.user_simulator_config is None


def test_user_simulator_config_json_with_explicit_type():
  """A JSON config that carries `type=llm_backed` should deserialize to the

  concrete subclass, not just the base.
  """
  payload = (
      '{"criteria": {"tool_trajectory_avg_score": 1.0},'
      ' "userSimulatorConfig": {"type": "llm_backed",'
      ' "model": "my-model", "maxAllowedInvocations": 5}}'
  )
  eval_config = EvalConfig.model_validate_json(payload)

  assert isinstance(
      eval_config.user_simulator_config, LlmBackedUserSimulatorConfig
  )
  assert eval_config.user_simulator_config.type == "llm_backed"
  assert eval_config.user_simulator_config.model == "my-model"
  assert eval_config.user_simulator_config.max_allowed_invocations == 5


def test_user_simulator_config_json_with_llm_audio_type():
  """A JSON config that carries `type=llm_audio` should deserialize to the

  `LlmAudioUserSimulatorConfig` subclass via the `type` discriminator.
  """
  payload = (
      '{"criteria": {"tool_trajectory_avg_score": 1.0},'
      ' "userSimulatorConfig": {"type": "llm_audio",'
      ' "model": "my-model", "maxAllowedInvocations": 5}}'
  )
  eval_config = EvalConfig.model_validate_json(payload)

  assert isinstance(
      eval_config.user_simulator_config, LlmAudioUserSimulatorConfig
  )
  assert eval_config.user_simulator_config.type == "llm_audio"
  assert eval_config.user_simulator_config.model == "my-model"
  assert eval_config.user_simulator_config.max_allowed_invocations == 5


def test_user_simulator_config_json_without_type_backward_compat():
  """Pre-discriminator JSON (no `type` field) must still deserialize into

  `LlmBackedUserSimulatorConfig` -- this is the backward-compat contract.
  """
  # Note the ABSENCE of `type`: this shape is what existing configs on disk
  # look like today.
  payload = (
      '{"criteria": {"tool_trajectory_avg_score": 1.0},'
      ' "userSimulatorConfig": {"model": "legacy-model"}}'
  )
  eval_config = EvalConfig.model_validate_json(payload)

  assert isinstance(
      eval_config.user_simulator_config, LlmBackedUserSimulatorConfig
  )
  assert eval_config.user_simulator_config.type == "llm_backed"
  assert eval_config.user_simulator_config.model == "legacy-model"


def test_user_simulator_config_json_without_type_snake_case():
  """The default-type injector must handle snake_case JSON keys too, since

  users may serialize with `by_alias=False`.
  """
  payload = (
      '{"criteria": {"tool_trajectory_avg_score": 1.0},'
      ' "user_simulator_config": {"model": "legacy-model-snake"}}'
  )
  eval_config = EvalConfig.model_validate_json(payload)

  assert isinstance(
      eval_config.user_simulator_config, LlmBackedUserSimulatorConfig
  )
  assert eval_config.user_simulator_config.model == "legacy-model-snake"


def test_user_simulator_config_json_with_explicit_null_type():
  """`type: null` in JSON (the shape produced by a `BaseUserSimulatorConfig`

  whose default `type=None` gets serialized) must be treated the same as a
  missing `type` key: default to the legacy subclass.
  """
  payload = (
      '{"criteria": {},'
      ' "userSimulatorConfig": {"type": null, "model": "explicit-null"}}'
  )
  eval_config = EvalConfig.model_validate_json(payload)

  assert isinstance(
      eval_config.user_simulator_config, LlmBackedUserSimulatorConfig
  )
  assert eval_config.user_simulator_config.type == "llm_backed"
  assert eval_config.user_simulator_config.model == "explicit-null"


def test_user_simulator_config_json_with_unknown_type_raises():
  """An unknown discriminator value must fail validation loudly."""
  payload = (
      '{"criteria": {}, "userSimulatorConfig": {"type": "typo_type_name"}}'
  )
  with pytest.raises(ValidationError):
    EvalConfig.model_validate_json(payload)


def test_user_simulator_config_round_trip_via_model_dump_json():
  """Serialize -> deserialize preserves the concrete subclass (and the

  `type` tag survives the round-trip).
  """
  original = EvalConfig(
      user_simulator_config=LlmBackedUserSimulatorConfig(
          model="round-trip-model"
      )
  )
  restored = EvalConfig.model_validate_json(original.model_dump_json())
  assert isinstance(
      restored.user_simulator_config, LlmBackedUserSimulatorConfig
  )
  assert restored.user_simulator_config.model == "round-trip-model"
  assert restored.user_simulator_config.type == "llm_backed"


def test_user_simulator_config_python_construction():
  """Direct Python construction with a concrete subclass instance also

  works -- the discriminator on `Field` doesn't interfere with that path.
  """
  eval_config = EvalConfig(
      user_simulator_config=LlmBackedUserSimulatorConfig(model="py-model"),
  )
  assert isinstance(
      eval_config.user_simulator_config, LlmBackedUserSimulatorConfig
  )
  assert eval_config.user_simulator_config.model == "py-model"


from google.adk.evaluation.eval_config import LiveModelConfig


def test_live_model_config_defaults_to_none():
  eval_config = EvalConfig(criteria={})
  assert eval_config.live_model_config is None


def test_live_model_config_from_json():
  eval_config = EvalConfig.model_validate({
      "criteria": {},
      "liveModelConfig": {"timeoutSeconds": 600},
  })
  assert isinstance(eval_config.live_model_config, LiveModelConfig)
  assert eval_config.live_model_config.timeout_seconds == 600


def test_multi_run_config_defaults_to_none():
  """A config that says nothing about runs has no multi-run config."""
  eval_config = EvalConfig(criteria={})
  assert eval_config.multi_run_config is None


@pytest.mark.parametrize(
    "config",
    [
        {"multi_run_config": {"num_runs": 3}},
        {"multiRunConfig": {"numRuns": 3}},
    ],
    ids=["snake_case", "camel_case"],
)
def test_multi_run_config_from_json(config):
  """The number of runs is read from either spelling of the keys."""
  eval_config = EvalConfig.model_validate_json(json.dumps(config))
  assert eval_config.multi_run_config == MultiRunConfig(num_runs=3)


def test_multi_run_config_leaves_num_runs_unset_by_default():
  """Without `num_runs`, the number of runs is left to the entry point."""
  assert MultiRunConfig().num_runs is None


def test_multi_run_config_rejects_num_runs_below_one():
  """Running an eval case zero times is a config error."""
  with pytest.raises(ValidationError, match="num_runs"):
    EvalConfig.model_validate({"multi_run_config": {"num_runs": 0}})


def test_multi_run_config_tolerates_no_crashed_run_by_default():
  """By default a single crashed run fails the case."""
  assert MultiRunConfig().max_crashed_runs == 0


def test_multi_run_config_reads_max_crashed_runs_in_camel_case():
  """The number of crashed runs to tolerate can be spelled maxCrashedRuns."""
  multi_run_config = MultiRunConfig.model_validate({"maxCrashedRuns": 2})

  assert multi_run_config.max_crashed_runs == 2


def test_multi_run_config_rejects_negative_max_crashed_runs():
  """Tolerating fewer than zero crashed runs is a config error."""
  with pytest.raises(ValidationError, match="max_crashed_runs"):
    EvalConfig.model_validate({"multi_run_config": {"max_crashed_runs": -1}})


def test_multi_run_config_rejects_unknown_keys():
  """A misspelled key raises instead of silently running each case once."""
  with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
    EvalConfig.model_validate({"multi_run_config": {"num_run": 3}})


def test_multi_run_config_pools_scores_by_default():
  """Without a rule, each metric's runs are pooled over every invocation."""
  multi_run_config = MultiRunConfig()

  assert (
      multi_run_config.cross_run_scoring,
      multi_run_config.cross_run_scoring_by_metric,
      multi_run_config.cross_run_criteria,
  ) == ("mean_over_invocations", None, None)


def test_multi_run_config_reads_rules_for_metrics_and_the_case():
  """Rules are read as method names, or as objects with their own threshold."""
  config = {
      "criteria": {
          "tool_trajectory_avg_score": 1.0,
          "response_match_score": 0.6,
      },
      "multi_run_config": {
          "num_runs": 3,
          "cross_run_scoring": {"method": "pass_rate", "threshold": 0.5},
          "cross_run_scoring_by_metric": {
              "response_match_score": "min_over_runs",
              "tool_trajectory_avg_score": {
                  "method": "pass_rate",
                  "threshold": 0.9,
              },
          },
          "cross_run_criteria": [{"method": "pass_rate", "threshold": 0.8}],
      },
  }

  eval_config = EvalConfig.model_validate_json(json.dumps(config))

  assert eval_config.multi_run_config == MultiRunConfig(
      num_runs=3,
      cross_run_scoring=MultiRunCriterion(method="pass_rate", threshold=0.5),
      cross_run_scoring_by_metric={
          "response_match_score": "min_over_runs",
          "tool_trajectory_avg_score": MultiRunCriterion(
              method="pass_rate", threshold=0.9
          ),
      },
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)],
  )


@pytest.mark.parametrize(
    "multi_run_config",
    [
        {"cross_run_scoring": "mean"},
        {"cross_run_scoring": {"method": "pass_at_k", "threshold": 0.5}},
    ],
    ids=["name", "object"],
)
def test_multi_run_config_rejects_unknown_method(multi_run_config):
  """A misspelled method name is rejected, not silently defaulted."""
  with pytest.raises(ValidationError, match="Unknown method"):
    EvalConfig.model_validate({"multi_run_config": multi_run_config})


@pytest.mark.parametrize(
    "multi_run_config",
    [
        {"cross_run_scoring": "pass_rate"},
        {"cross_run_scoring_by_metric": {"response_match_score": "pass_rate"}},
    ],
    ids=["default_rule", "metric_rule"],
)
def test_multi_run_config_rejects_pass_count_method_without_threshold(
    multi_run_config,
):
  """A method that counts passing runs, given by name alone, has no threshold.

  The metric's threshold in `criteria` is on the scale of its scores, not of a
  fraction of runs, so such a method must come with a threshold of its own.
  """
  with pytest.raises(ValidationError, match="needs a threshold of its own"):
    EvalConfig.model_validate({
        "criteria": {"response_match_score": 0.8},
        "multi_run_config": multi_run_config,
    })


@pytest.mark.parametrize(
    "multi_run_config",
    [
        {"cross_run_scoring": {"method": "mean_of_runs", "threshold": 0.5}},
        {
            "cross_run_scoring_by_metric": {
                "response_match_score": {
                    "method": "min_over_runs",
                    "threshold": 0.5,
                }
            }
        },
        {"cross_run_criteria": [{"method": "mean_of_runs", "threshold": 0.5}]},
    ],
    ids=["default_rule", "metric_rule", "cross_run_criterion"],
)
def test_multi_run_config_rejects_score_method_with_its_own_threshold(
    multi_run_config,
):
  """A method that combines scores is judged against the metric's threshold.

  So it takes no threshold of its own, and cannot judge a whole eval case,
  which has no score.
  """
  with pytest.raises(ValidationError, match="combines scores"):
    EvalConfig.model_validate({
        "criteria": {"response_match_score": 0.8},
        "multi_run_config": multi_run_config,
    })


def test_multi_run_config_rejects_rule_for_metric_not_in_criteria():
  """A rule for a metric that is not scored, e.g. a typo, would never apply."""
  with pytest.raises(
      ValidationError, match=r"not in criteria: \['response_match'\]"
  ):
    EvalConfig.model_validate({
        "criteria": {"response_match_score": 0.8},
        "multi_run_config": {
            "cross_run_scoring_by_metric": {"response_match": "mean_of_runs"}
        },
    })


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_multi_run_criterion_rejects_threshold_outside_zero_to_one(threshold):
  """A fraction of runs is between 0 and 1, so its threshold must be too."""
  with pytest.raises(ValidationError, match="threshold"):
    MultiRunCriterion(method="pass_rate", threshold=threshold)


def test_multi_run_criterion_rejects_unknown_keys():
  """A misspelled key raises instead of being silently dropped from a rule."""
  with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
    MultiRunCriterion.model_validate(
        {"method": "pass_rate", "threshold": 0.5, "k": 3}
    )


def _custom_method(*, uses_scores: bool) -> dict[str, object]:
  """Returns a custom method's config, as written in JSON."""
  return {
      "code_config": {"name": "my_project.eval.methods.my_method"},
      "uses_scores": uses_scores,
  }


def test_multi_run_config_reads_custom_methods():
  """Custom methods are declared by name, and the rules then name them."""
  config = {
      "criteria": {"response_match_score": 0.6},
      "multi_run_config": {
          "num_runs": 5,
          "custom_methods": {
              "passed_at_least_twice": {
                  "code_config": {
                      "name": "my_project.eval.methods.passed_at_least_twice"
                  },
                  "uses_scores": False,
              },
              "median_of_runs": {
                  "code_config": {
                      "name": "my_project.eval.methods.median_of_runs"
                  },
                  "uses_scores": True,
              },
          },
          "cross_run_scoring": "median_of_runs",
          "cross_run_criteria": [
              {"method": "passed_at_least_twice", "threshold": 1.0}
          ],
      },
  }

  eval_config = EvalConfig.model_validate_json(json.dumps(config))

  assert eval_config.multi_run_config == MultiRunConfig(
      num_runs=5,
      custom_methods={
          "passed_at_least_twice": CustomCrossRunMethodConfig(
              code_config=CodeConfig(
                  name="my_project.eval.methods.passed_at_least_twice"
              ),
              uses_scores=False,
          ),
          "median_of_runs": CustomCrossRunMethodConfig(
              code_config=CodeConfig(
                  name="my_project.eval.methods.median_of_runs"
              ),
              uses_scores=True,
          ),
      },
      cross_run_scoring="median_of_runs",
      cross_run_criteria=[
          MultiRunCriterion(method="passed_at_least_twice", threshold=1.0)
      ],
  )


def test_custom_method_reads_camel_case_keys():
  """A custom method's keys can be written in camelCase as well."""
  multi_run_config = MultiRunConfig.model_validate({
      "customMethods": {
          "median_of_runs": {
              "codeConfig": {"name": "my_project.eval.methods.median_of_runs"},
              "usesScores": True,
          }
      },
      "crossRunScoring": "median_of_runs",
  })

  assert multi_run_config.custom_methods == {
      "median_of_runs": CustomCrossRunMethodConfig(
          code_config=CodeConfig(name="my_project.eval.methods.median_of_runs"),
          uses_scores=True,
      )
  }


def test_custom_method_is_not_imported_when_the_config_is_read():
  """Reading a config does not import its custom methods' functions."""
  multi_run_config = MultiRunConfig.model_validate({
      "custom_methods": {
          "median_of_runs": {
              "code_config": {"name": "no_such_module_for_adk_tests.median"},
              "uses_scores": True,
          }
      },
      "cross_run_scoring": "median_of_runs",
  })

  assert multi_run_config.cross_run_scoring == "median_of_runs"


def test_custom_method_cannot_take_a_built_in_name():
  """A custom method named like a built-in one would make the name ambiguous."""
  with pytest.raises(
      ValidationError, match="'pass_rate' is the name of a built-in method"
  ):
    MultiRunConfig.model_validate(
        {"custom_methods": {"pass_rate": _custom_method(uses_scores=False)}}
    )


def test_unknown_method_error_lists_the_custom_methods():
  """An unknown name is reported with the custom methods that do exist."""
  with pytest.raises(
      ValidationError, match=re.escape("Custom methods: ['median_of_runs']")
  ):
    MultiRunConfig.model_validate({
        "custom_methods": {"median_of_runs": _custom_method(uses_scores=True)},
        "cross_run_scoring": "median",
    })


@pytest.mark.parametrize(
    ("multi_run_config", "message"),
    [
        (
            {
                "custom_methods": {"twice": _custom_method(uses_scores=False)},
                "cross_run_scoring": "twice",
            },
            "needs a threshold of its own",
        ),
        (
            {
                "custom_methods": {"median": _custom_method(uses_scores=True)},
                "cross_run_scoring_by_metric": {
                    "response_match_score": {
                        "method": "median",
                        "threshold": 0.5,
                    }
                },
            },
            "combines scores",
        ),
        (
            {
                "custom_methods": {"median": _custom_method(uses_scores=True)},
                "cross_run_criteria": [{"method": "median", "threshold": 0.5}],
            },
            "combines scores",
        ),
    ],
    ids=[
        "pass_count_method_as_name",
        "score_method_as_object",
        "score_method_in_cross_run_criteria",
    ],
)
def test_custom_method_is_written_in_the_form_its_kind_takes(
    multi_run_config, message
):
  """`uses_scores` decides whether a custom method has a threshold of its own."""
  with pytest.raises(ValidationError, match=message):
    MultiRunConfig.model_validate(multi_run_config)


@pytest.mark.parametrize(
    ("multi_run_config", "where"),
    [
        ({"cross_run_scoring": "pass_rate"}, "cross_run_scoring"),
        (
            {"cross_run_scoring_by_metric": {"response_match_score": "mean"}},
            "cross_run_scoring_by_metric['response_match_score']",
        ),
        (
            {
                "cross_run_criteria": [
                    {"method": "pass_rate", "threshold": 0.5},
                    {"method": "mean_of_runs", "threshold": 0.5},
                ]
            },
            "cross_run_criteria[1]",
        ),
        (
            {
                "custom_methods": {
                    "pass_rate": _custom_method(uses_scores=False)
                }
            },
            "custom_methods['pass_rate']",
        ),
    ],
    ids=[
        "default_rule",
        "metric_rule",
        "cross_run_criterion",
        "custom_method",
    ],
)
def test_multi_run_config_error_starts_with_where_the_mistake_is(
    multi_run_config, where
):
  """The message names the field, and key or index, holding the mistake."""
  with pytest.raises(
      ValidationError, match=re.escape(f"Value error, {where}:")
  ):
    MultiRunConfig.model_validate(multi_run_config)


def test_custom_method_requires_uses_scores():
  """Whether a custom method combines scores has no default; it must be said."""
  with pytest.raises(ValidationError, match=r"usesScores\s+Field required"):
    CustomCrossRunMethodConfig.model_validate(
        {"code_config": {"name": "my_project.eval.methods.median_of_runs"}}
    )


def test_custom_method_rejects_unknown_keys():
  """A misspelled key raises instead of being silently dropped from a method."""
  with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
    CustomCrossRunMethodConfig.model_validate({
        "code_config": {"name": "my_project.eval.methods.median_of_runs"},
        "uses_scores": True,
        "threshold": 0.5,
    })
