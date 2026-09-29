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

"""Informational efficiency metrics for agent evaluation.

These metrics quantify how "expensive" an agent was in producing its responses,
rather than how good the responses were. They cover, with the unit each one
reports in:

  - `tool_call_count_v1`: tool (function) calls made -- a count.
  - `inference_call_count_v1`: calls made to the model -- a count.
  - `token_usage_v1`: tokens consumed -- a token count. The score is the total,
    and every token type is reported alongside it as a breakdown.
  - `invocation_duration_v1`: wall-clock time the turn took -- seconds, not
    milliseconds.

Lower is better for all four. Every value is reported **per invocation** (per
conversation turn): each per-invocation result holds that turn's own value, and
the overall value for the eval case is their average (see Aggregation below).
So the overall number reads as "tool calls per turn", "tokens per turn" or
"seconds per turn" -- never a per-case total. A value of `None` means the
metric is not available (n/a) for that invocation, e.g. the model backend
reported no usage metadata; it never means zero.

These are reference-free, informational metrics: they compute and report a value
for the user to track their agent's efficiency, but they do NOT pass or fail an
eval case. Their status is always `INFORMATIONAL`, and any threshold configured
for them is ignored.

Aggregation: a metric is computed **per invocation** (i.e. per conversation
turn) first, and those per-invocation values are all reported. The single
overall value for the eval case is the **average** of the per-invocation values.
Averaging (rather than summing) is deliberate:

  - It keeps the overall number comparable across eval cases that have different
    numbers of turns -- e.g. "average tokens per turn" is meaningful whether a
    case has 1 turn or 10, whereas a raw total would just grow with turn count.
  - It matches how every other ADK metric aggregates per-invocation results
    (e.g. `response_match_score`, `tool_trajectory_avg_score`), so efficiency
    metrics behave consistently with the rest of the framework.

If you want a per-case total instead (e.g. total tokens or total tool calls for
the whole conversation), sum the reported per-invocation values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from typing_extensions import override

from ..telemetry._token_usage import TokenUsage
from .eval_case import ConversationScenario
from .eval_case import get_all_tool_calls
from .eval_case import get_all_usage_metadata
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .eval_case import InvocationEvents
from .eval_metrics import EvalMetric
from .eval_metrics import TokenUsageDetails
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult


def _reject_threshold(eval_metric: Optional[EvalMetric]) -> None:
  """Rejects a threshold configured on an informational efficiency metric.

  These metrics report a value and never pass or fail, so a threshold has no
  effect today. Rejecting one rather than ignoring it keeps thresholds out of
  user configs entirely, so making them gate later -- when a configured
  threshold starts deciding pass or fail -- cannot silently begin failing an
  eval that passes today.
  """
  if eval_metric is None:
    return
  # A criterion carries a threshold unconditionally, so its mere presence means
  # one was configured; reaching into it would only cost the static check that
  # a rename of that field has to break something.
  if eval_metric.threshold is None and eval_metric.criterion is None:
    return
  raise ValueError(
      f"`{eval_metric.metric_name}` does not support a threshold: it reports a"
      " value and never passes or fails. Remove it from `criteria` entirely;"
      " it is reported automatically on every eval."
  )


def _is_model_call_event(event: InvocationEvent) -> bool:
  """Returns whether an invocation event represents a model (LLM) call."""
  return event.usage_metadata is not None or event.model_version is not None


def _evaluate_efficiency_metric(
    actual_invocations: list[Invocation],
    compute_value_fn: Callable[[Invocation], Optional[float]],
) -> EvaluationResult:
  """Evaluates an efficiency metric across invocations using the provided computation function."""
  per_invocation_results = []
  values = []
  for actual in actual_invocations:
    value = compute_value_fn(actual)
    per_invocation_results.append(
        PerInvocationResult(
            actual_invocation=actual,
            score=value,
            # Informational: a value is reported but never passes or fails.
            # The status is INFORMATIONAL even when the value is None (n/a).
            eval_status=EvalStatus.INFORMATIONAL,
        )
    )
    if value is not None:
      values.append(value)

  if not per_invocation_results:
    return EvaluationResult(overall_eval_status=EvalStatus.INFORMATIONAL)

  # The per-invocation (per-turn) values are reported above. The single
  # overall value is their AVERAGE, not their sum: averaging keeps the number
  # comparable across eval cases with different turn counts (e.g. average
  # tokens/tool-calls per turn) and matches how every other ADK metric
  # aggregates per-invocation results. Callers who want a per-case total can
  # sum the per-invocation values. `None` values (n/a) are excluded from the
  # average; if no invocation produced a value, the overall stays None.
  overall_score = sum(values) / len(values) if values else None
  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=EvalStatus.INFORMATIONAL,
      per_invocation_results=per_invocation_results,
  )


class _ToolCallCountV1Evaluator(Evaluator):
  """Counts the number of tool (function) calls made in an invocation.

  Unit: tool calls per invocation.

  Informational only: reports the count, never passes or fails.
  """

  def __init__(
      self,
      eval_metric: Optional[EvalMetric] = None,
  ):
    _reject_threshold(eval_metric)
    self._eval_metric = eval_metric

  def _compute_value(self, invocation: Invocation) -> Optional[float]:
    return float(len(get_all_tool_calls(invocation.intermediate_data)))

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    # Efficiency metrics are reference-free, so expected invocations and
    # conversation scenarios are not used.
    del expected_invocations, conversation_scenario
    return _evaluate_efficiency_metric(actual_invocations, self._compute_value)


class _InferenceCallCountV1Evaluator(Evaluator):
  """Counts the number of inference (model) calls made in an invocation.

  Unit: inference calls per invocation.

  Named after the count ADK telemetry publishes, so the same quantity carries
  the same name on both surfaces. An eval invocation spans a whole turn (every
  sub-agent shares the turn's invocation id), so this counts what telemetry
  reports per turn as `adk.experimental.invoke_workflow.inference_calls`, not
  the per-agent `gen_ai.invoke_agent.inference_calls`.

  This is a proxy for the number of attempts or reasoning steps the agent took.
  Read alongside `token_usage_v1` it separates the two ways a turn gets
  expensive: more calls, or a larger context per call. Informational only.
  Returns None when model-call data was not captured for the invocation.
  """

  def __init__(
      self,
      eval_metric: Optional[EvalMetric] = None,
  ):
    _reject_threshold(eval_metric)
    self._eval_metric = eval_metric

  def _compute_value(self, invocation: Invocation) -> Optional[float]:
    if not isinstance(invocation.intermediate_data, InvocationEvents):
      return None
    model_events = [
        e
        for e in invocation.intermediate_data.invocation_events
        if _is_model_call_event(e)
    ]
    return float(len(model_events))

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    # Efficiency metrics are reference-free, so expected invocations and
    # conversation scenarios are not used.
    del expected_invocations, conversation_scenario
    return _evaluate_efficiency_metric(actual_invocations, self._compute_value)


class _InvocationDurationV1Evaluator(Evaluator):
  """Reports how long an invocation took, in seconds.

  Unit: seconds per invocation.

  The value is measured while the agent runs and carried on the invocation; it
  is never reconstructed from event timestamps, which mark when an event object
  was built rather than when the model answered. An invocation that was not
  produced by this eval run -- read back from a stored session, say -- carries
  no timing and reports n/a.

  An eval invocation spans a whole turn (every sub-agent shares the turn's
  invocation id), so this is the quantity telemetry publishes per turn as
  `gen_ai.invoke_workflow.duration`, not the per-agent
  `gen_ai.invoke_agent.duration`.

  Informational only. Wall-clock time is far noisier than the token and call
  counts -- it moves with model-server load and network, not just with the
  agent -- so read it as an indication and use the counts to judge a
  regression.
  """

  def __init__(
      self,
      eval_metric: Optional[EvalMetric] = None,
  ):
    _reject_threshold(eval_metric)
    self._eval_metric = eval_metric

  def _compute_value(self, invocation: Invocation) -> Optional[float]:
    # None here means the run did not record a duration, not that the turn took
    # no time, so `_evaluate_efficiency_metric` leaves it out of the average
    # rather than counting it as zero.
    return invocation.duration

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    # Efficiency metrics are reference-free, so expected invocations and
    # conversation scenarios are not used.
    del expected_invocations, conversation_scenario
    return _evaluate_efficiency_metric(actual_invocations, self._compute_value)


class _TokenUsageV1Evaluator(Evaluator):
  """Sums the token usage reported across all model calls in an invocation.

  Unit: tokens per invocation.

  The score is the total. Every token type is reported alongside it as
  `TokenUsageDetails`: the counts nest (`total` is `input` plus `output`, each
  with subsets underneath) and take their meanings from
  `google.adk.telemetry._token_usage`, so they match the token metrics ADK
  publishes for the same run.

  Informational only. A count is None -- unavailable (n/a) -- rather than 0
  when no model call reported it.
  """

  def __init__(
      self,
      eval_metric: Optional[EvalMetric] = None,
  ):
    _reject_threshold(eval_metric)
    self._eval_metric = eval_metric

  @staticmethod
  def _add(total: Optional[float], count: Optional[int]) -> Optional[float]:
    """Adds one call's count to a running total, keeping n/a distinct from zero.

    A count the call did not report leaves the total untouched, including
    leaving it None when nothing has reported that count yet.
    """
    if count is None:
      return total
    return (total or 0) + count

  def _add_call(self, totals: TokenUsageDetails, usage: TokenUsage) -> None:
    """Folds one model call's usage into the running invocation totals.

    Every count is read through `TokenUsage`, which owns what these counts
    mean, so an eval breakdown and the telemetry metric of the same name agree
    by construction rather than by two copies of the same arithmetic staying in
    step.

    Mirrors `telemetry._token_usage.TokenUsage.add`, differing in one deliberate
    respect: there a missing count sums as zero, because it feeds a histogram;
    here it stays None, because an eval reports the figure to a person, for whom
    "not reported" and "zero" are different answers.
    """
    totals.input_tokens = self._add(totals.input_tokens, usage.input_tokens)
    totals.prompt_tokens = self._add(
        totals.prompt_tokens, usage.prompt_input_tokens
    )
    totals.cached_tokens = self._add(
        totals.cached_tokens, usage.cache_read_input_tokens
    )
    totals.tool_use_tokens = self._add(
        totals.tool_use_tokens, usage.tool_input_tokens
    )
    totals.output_tokens = self._add(totals.output_tokens, usage.output_tokens)
    totals.candidates_tokens = self._add(
        totals.candidates_tokens, usage.candidate_output_tokens
    )
    totals.reasoning_tokens = self._add(
        totals.reasoning_tokens, usage.reasoning_output_tokens
    )

  def _token_usage_details(self, invocation: Invocation) -> TokenUsageDetails:
    """Returns every token count for a single invocation."""
    totals = TokenUsageDetails()
    for usage_metadata in get_all_usage_metadata(invocation):
      self._add_call(totals, TokenUsage.from_usage_metadata(usage_metadata))
    # Derived from the two directions rather than read back from the backend,
    # so the headline count always agrees with the breakdown under it. Set in
    # this one place, once the parts are final.
    if totals.input_tokens is not None or totals.output_tokens is not None:
      totals.total_tokens = (totals.input_tokens or 0) + (
          totals.output_tokens or 0
      )
    return totals

  def _average(self, all_details: list[TokenUsageDetails]) -> TokenUsageDetails:
    """Averages each count over every invocation, not just those reporting it.

    A backend omits a count for a call that did not spend it -- Gemini leaves
    `thoughts_token_count` out of a call the model did not think through -- so a
    missing count means zero for that invocation and belongs in the denominator.
    A count no invocation reported at all is a different thing: the backend does
    not report it, and it stays None rather than becoming 0.

    One shared denominator also keeps `total` equal to `input` plus `output`
    after averaging.
    """
    averaged = {}
    for field in TokenUsageDetails.model_fields:
      values = [getattr(details, field) for details in all_details]
      present = [value for value in values if value is not None]
      averaged[field] = sum(present) / len(values) if present else None
    return TokenUsageDetails(**averaged)

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    # Efficiency metrics are reference-free, so expected invocations and
    # conversation scenarios are not used.
    del expected_invocations, conversation_scenario

    per_invocation_results = []
    all_details = []
    for actual in actual_invocations:
      details = self._token_usage_details(actual)
      all_details.append(details)
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              score=details.total_tokens,
              # Informational: a value is reported but never passes or fails.
              # The status is INFORMATIONAL even when the value is None (n/a).
              eval_status=EvalStatus.INFORMATIONAL,
              token_usage_details=details,
          )
      )

    if not per_invocation_results:
      return EvaluationResult(overall_eval_status=EvalStatus.INFORMATIONAL)

    overall_details = self._average(all_details)
    return EvaluationResult(
        overall_score=overall_details.total_tokens,
        overall_eval_status=EvalStatus.INFORMATIONAL,
        per_invocation_results=per_invocation_results,
        overall_token_usage_details=overall_details,
    )
