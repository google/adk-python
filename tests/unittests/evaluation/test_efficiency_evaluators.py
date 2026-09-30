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

"""Tests for the efficiency evaluators.

Efficiency metrics are informational: they report a value but never pass or
fail, so every result has an `EvalStatus` of `INFORMATIONAL`.
"""

from typing import Optional

from google.adk.evaluation._efficiency_evaluators import _InferenceCallCountV1Evaluator
from google.adk.evaluation._efficiency_evaluators import _InvocationDurationV1Evaluator
from google.adk.evaluation._efficiency_evaluators import _TokenUsageV1Evaluator
from google.adk.evaluation._efficiency_evaluators import _ToolCallCountV1Evaluator
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import TokenUsageDetails
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.telemetry._token_usage import TokenUsage
from google.genai import types as genai_types
import pytest

_USER_CONTENT = genai_types.Content(
    parts=[genai_types.Part(text="User input here.")]
)


def _usage(
    *,
    prompt: int = 0,
    candidates: int = 0,
    cached: int = 0,
    thoughts: int = 0,
    tool_use_prompt: int = 0,
    backend_total: Optional[int] = None,
) -> genai_types.GenerateContentResponseUsageMetadata:
  """Returns usage metadata populated with the given token counts.

  There is no `total` parameter because the reported total is derived from the
  parts rather than read from the backend. `backend_total` sets the total the
  backend claims, which only the test that pins that behaviour needs.
  """
  return genai_types.GenerateContentResponseUsageMetadata(
      prompt_token_count=prompt,
      candidates_token_count=candidates,
      total_token_count=backend_total,
      cached_content_token_count=cached,
      thoughts_token_count=thoughts,
      tool_use_prompt_token_count=tool_use_prompt,
  )


def _invocation_with(
    *usages: genai_types.GenerateContentResponseUsageMetadata,
) -> Invocation:
  """Returns an invocation whose events report the given usage metadata."""
  return Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(author="agent", usage_metadata=usage)
              for usage in usages
          ]
      ),
  )


def _tool_call(name: str) -> genai_types.FunctionCall:
  """Returns a function call with the given name and no args."""
  return genai_types.FunctionCall(name=name, args={})


# ---------------------------------------------------------------------------
# _ToolCallCountV1Evaluator
# ---------------------------------------------------------------------------


def test_tool_call_count_counts_calls_and_averages():
  """Tool call count reports the average per-invocation count, no pass/fail."""
  evaluator = _ToolCallCountV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.TOOL_CALL_COUNT_V1.value
      )
  )
  inv1 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[_tool_call("a"), _tool_call("b")]
      ),
  )
  inv2 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[_tool_call("a")]),
  )

  result = evaluator.evaluate_invocations([inv1, inv2])

  assert result.overall_score == 1.5
  assert result.overall_eval_status == EvalStatus.INFORMATIONAL
  assert result.per_invocation_results[0].score == 2.0
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.INFORMATIONAL
  )
  assert result.per_invocation_results[1].score == 1.0
  assert (
      result.per_invocation_results[1].eval_status == EvalStatus.INFORMATIONAL
  )


def test_tool_call_count_never_fails_even_for_high_counts():
  """A high count is still reported as INFORMATIONAL (informational only)."""
  evaluator = _ToolCallCountV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.TOOL_CALL_COUNT_V1.value
      )
  )
  inv = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[_tool_call("a"), _tool_call("b"), _tool_call("c")]
      ),
  )

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score == 3.0
  assert result.overall_eval_status == EvalStatus.INFORMATIONAL


@pytest.mark.parametrize(
    "evaluator_type,metric_name",
    [
        (_ToolCallCountV1Evaluator, PrebuiltMetrics.TOOL_CALL_COUNT_V1.value),
        (
            _InferenceCallCountV1Evaluator,
            PrebuiltMetrics.INFERENCE_CALL_COUNT_V1.value,
        ),
        (_TokenUsageV1Evaluator, PrebuiltMetrics.TOKEN_USAGE_V1.value),
        (
            _InvocationDurationV1Evaluator,
            PrebuiltMetrics.INVOCATION_DURATION_V1.value,
        ),
    ],
)
def test_efficiency_evaluator_rejects_a_configured_threshold(
    evaluator_type, metric_name
):
  """A threshold is rejected, not ignored, so configs never carry a dead one."""
  with pytest.raises(ValueError, match="does not support a threshold"):
    evaluator_type(
        eval_metric=EvalMetric(metric_name=metric_name, threshold=1.0)
    )


def test_efficiency_evaluator_rejects_a_threshold_on_the_criterion():
  """A threshold reached through the criterion is rejected the same way."""
  with pytest.raises(ValueError, match="does not support a threshold"):
    _TokenUsageV1Evaluator(
        eval_metric=EvalMetric(
            metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value,
            criterion=BaseCriterion(threshold=0.5),
        )
    )


def test_efficiency_evaluator_no_invocations():
  """An empty invocation list yields an empty result."""
  evaluator = _ToolCallCountV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.TOOL_CALL_COUNT_V1.value
      )
  )

  result = evaluator.evaluate_invocations([])

  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.INFORMATIONAL
  assert not result.per_invocation_results


# ---------------------------------------------------------------------------
# _InferenceCallCountV1Evaluator
# ---------------------------------------------------------------------------


def test_inference_call_count_counts_model_calls():
  """LLM call count reports the number of recorded model calls."""
  evaluator = _InferenceCallCountV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INFERENCE_CALL_COUNT_V1.value
      )
  )
  inv = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(author="agent", usage_metadata=_usage()),
              InvocationEvent(author="agent", usage_metadata=_usage()),
          ]
      ),
  )

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score == 2.0
  assert result.overall_eval_status == EvalStatus.INFORMATIONAL


def test_inference_call_count_none_when_not_captured():
  """LLM call count reports no value when no intermediate data was captured."""
  evaluator = _InferenceCallCountV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INFERENCE_CALL_COUNT_V1.value
      )
  )
  inv = Invocation(user_content=_USER_CONTENT)

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score is None
  assert result.per_invocation_results[0].score is None
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.INFORMATIONAL
  )


def test_inference_call_count_empty_list_is_zero():
  """An empty invocation events list counts as zero calls."""
  evaluator = _InferenceCallCountV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INFERENCE_CALL_COUNT_V1.value
      )
  )
  inv = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=InvocationEvents(invocation_events=[]),
  )

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score == 0.0


# ---------------------------------------------------------------------------
# _TokenUsageV1Evaluator
# ---------------------------------------------------------------------------


def test_token_usage_sums_total_tokens_across_model_calls():
  """Token usage sums total tokens over all model calls."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = _invocation_with(
      _usage(prompt=60, candidates=40),
      _usage(prompt=200, candidates=50),
  )

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score == 350.0
  assert result.overall_eval_status == EvalStatus.INFORMATIONAL


def test_token_usage_total_is_derived_not_the_backend_reported_total():
  """The total sums input and output instead of trusting the backend's own."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  # The backend's total disagrees with the parts it reported; the parts win, so
  # the total always adds up to the breakdown printed underneath it.
  inv = _invocation_with(
      _usage(prompt=100, candidates=20, thoughts=5, backend_total=9999)
  )

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score == 125.0


def test_token_usage_input_and_output_group_their_parts():
  """`input` and `output` sum the counts nested under each of them."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = _invocation_with(
      _usage(
          prompt=700,
          cached=100,
          tool_use_prompt=20,
          candidates=200,
          thoughts=80,
      )
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  # `cached` is a portion of `prompt`, so it is not added on top of it.
  assert details.input_tokens == 720.0
  assert details.output_tokens == 280.0
  assert details.total_tokens == 1000.0


def test_every_token_count_is_reported_in_the_breakdown():
  """Every field gets filled when the backend reports every count.

  Guards against a count being dropped from `_add_call`: the field would keep
  its None default and read as n/a forever, with nothing else failing.
  """
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = _invocation_with(
      _usage(
          prompt=700,
          cached=100,
          tool_use_prompt=20,
          candidates=200,
          thoughts=80,
      )
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  unfilled = [
      field
      for field in TokenUsageDetails.model_fields
      if getattr(details, field) is None
  ]
  assert not unfilled


def test_token_usage_missing_count_on_one_call_counts_as_zero():
  """A call reporting no count contributes zero rather than voiding the sum.

  Gemini leaves `thoughts_token_count` out of a call the model did not think
  through -- the second call here, which only relays a tool result.
  """
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = _invocation_with(
      _usage(prompt=368, candidates=28, thoughts=80),
      genai_types.GenerateContentResponseUsageMetadata(
          prompt_token_count=483, candidates_token_count=8
      ),
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  assert details.reasoning_tokens == 80.0
  assert details.candidates_tokens == 36.0
  assert details.output_tokens == 116.0


def test_token_usage_count_no_invocation_reported_stays_na():
  """A count the backend never reports stays None rather than averaging to 0."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  # Raw metadata rather than `_usage`, which defaults every count to zero.
  inv = _invocation_with(
      genai_types.GenerateContentResponseUsageMetadata(
          prompt_token_count=100, candidates_token_count=20
      )
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  assert details.cached_tokens is None
  assert details.tool_use_tokens is None


def test_token_usage_counts_agree_with_telemetry():
  """The breakdown reports what telemetry reports for the same model call."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  usage_metadata = _usage(
      prompt=700, cached=100, tool_use_prompt=20, candidates=200, thoughts=80
  )
  telemetry_usage = TokenUsage.from_usage_metadata(usage_metadata)

  result = evaluator.evaluate_invocations([_invocation_with(usage_metadata)])

  details = result.overall_token_usage_details
  assert details.input_tokens == telemetry_usage.input_tokens
  assert details.output_tokens == telemetry_usage.output_tokens
  assert details.prompt_tokens == telemetry_usage.prompt_input_tokens
  assert details.cached_tokens == telemetry_usage.cache_read_input_tokens
  assert details.tool_use_tokens == telemetry_usage.tool_input_tokens
  assert details.candidates_tokens == telemetry_usage.candidate_output_tokens
  assert details.reasoning_tokens == telemetry_usage.reasoning_output_tokens
  assert details.total_tokens == telemetry_usage.total_tokens


def test_token_usage_none_when_no_usage_present():
  """Token usage reports no value when no event reported usage."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(author="agent", usage_metadata=None)
          ]
      ),
  )

  result = evaluator.evaluate_invocations([inv])

  assert result.overall_score is None
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.INFORMATIONAL
  )


def test_token_usage_reports_every_type_without_configuration():
  """With no criterion at all, the full per-type breakdown is still reported."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = _invocation_with(
      _usage(
          prompt=700,
          candidates=200,
          cached=100,
          thoughts=80,
          tool_use_prompt=20,
      )
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  assert details is not None
  assert details.total_tokens == 1000.0
  assert details.input_tokens == 720.0
  assert details.prompt_tokens == 700.0
  assert details.cached_tokens == 100.0
  assert details.tool_use_tokens == 20.0
  assert details.output_tokens == 280.0
  assert details.candidates_tokens == 200.0
  assert details.reasoning_tokens == 80.0
  # The score stays the headline count, unchanged by the breakdown.
  assert result.overall_score == 1000.0


def test_token_usage_details_are_reported_per_invocation():
  """Each per-invocation result carries that invocation's own breakdown."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  first = _invocation_with(_usage(prompt=60, candidates=40))
  second = _invocation_with(_usage(prompt=200, candidates=100))

  result = evaluator.evaluate_invocations([first, second])

  prompts = [
      r.token_usage_details.prompt_tokens for r in result.per_invocation_results
  ]
  assert prompts == [60.0, 200.0]


def test_token_usage_details_are_na_not_zero_when_unreported():
  """A count the backend never reported stays None rather than becoming 0."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  # Only the prompt is reported; every other count is absent.
  inv = _invocation_with(
      genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=500)
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  assert details.prompt_tokens == 500.0
  assert details.input_tokens == 500.0
  assert details.total_tokens == 500.0
  assert details.output_tokens is None
  assert details.candidates_tokens is None
  assert details.reasoning_tokens is None
  assert details.cached_tokens is None
  assert details.tool_use_tokens is None


def test_token_usage_averages_an_unreported_count_as_zero_for_that_turn():
  """A count an invocation did not report is a zero in the average, not a gap.

  A backend omits a count for a turn that did not spend it, so dropping that
  turn from the denominator would report a per-turn average higher than what
  the run actually spent per turn.
  """
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  reports_thoughts = _invocation_with(
      _usage(prompt=50, candidates=10, thoughts=40)
  )
  # No thoughts count at all on the second invocation.
  omits_thoughts = _invocation_with(
      genai_types.GenerateContentResponseUsageMetadata(
          prompt_token_count=200, candidates_token_count=100
      )
  )

  result = evaluator.evaluate_invocations([reports_thoughts, omits_thoughts])

  details = result.overall_token_usage_details
  assert details.total_tokens == 200.0  # (100 + 300) / 2
  # Averaged over both turns; the second spent no reasoning tokens.
  assert details.reasoning_tokens == 20.0  # 40 / 2


def test_token_usage_counts_tool_use_tokens():
  """Server-side tool result tokens are summed and reported like any other."""
  evaluator = _TokenUsageV1Evaluator(
      eval_metric=EvalMetric(metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value)
  )
  inv = _invocation_with(
      _usage(prompt=470, tool_use_prompt=30),
      _usage(prompt=488, tool_use_prompt=12),
  )

  result = evaluator.evaluate_invocations([inv])

  details = result.overall_token_usage_details
  assert details.tool_use_tokens == 42.0
  # They are part of the input, not an addition on top of it.
  assert details.input_tokens == 1000.0


# ---------------------------------------------------------------------------
# _InvocationDurationV1Evaluator
# ---------------------------------------------------------------------------


def _invocation_lasting(duration: Optional[float]) -> Invocation:
  """Returns an invocation that recorded the given wall-clock duration."""
  return Invocation(user_content=_USER_CONTENT, duration=duration)


def test_duration_reports_the_recorded_wall_clock_time():
  """The value is the duration measured while the invocation ran."""
  evaluator = _InvocationDurationV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value
      )
  )

  result = evaluator.evaluate_invocations([_invocation_lasting(2.844)])

  assert result.overall_score == 2.844
  assert result.overall_eval_status == EvalStatus.INFORMATIONAL


def test_duration_averages_across_invocations():
  """The eval case value is the mean of the per-turn durations."""
  evaluator = _InvocationDurationV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value
      )
  )

  result = evaluator.evaluate_invocations(
      [_invocation_lasting(2.0), _invocation_lasting(4.0)]
  )

  assert result.overall_score == 3.0


def test_duration_is_na_when_the_run_recorded_none():
  """An invocation the eval did not time reports n/a rather than zero.

  Invocations read back from a stored session carry no timing, and it cannot be
  recovered from event timestamps afterwards.
  """
  evaluator = _InvocationDurationV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value
      )
  )

  result = evaluator.evaluate_invocations([_invocation_lasting(None)])

  assert result.overall_score is None
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.INFORMATIONAL
  )


def test_duration_averages_only_the_timed_invocations():
  """An untimed turn is left out of the average rather than counted as zero."""
  evaluator = _InvocationDurationV1Evaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value
      )
  )

  result = evaluator.evaluate_invocations(
      [_invocation_lasting(3.0), _invocation_lasting(None)]
  )

  # 3.0 / 1, not 3.0 / 2: the second turn has no measurement, which is not the
  # same as having taken no time.
  assert result.overall_score == 3.0
