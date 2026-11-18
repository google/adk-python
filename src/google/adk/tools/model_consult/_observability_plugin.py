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

"""Escalation, token, latency, and cost tracking for `ModelConsultTool`.

`ModelConsultObservabilityPlugin` measures what consulting an advisor costs an
agent: how often the executor escalates through `ModelConsultTool`, the tokens
the advisor and the executor each spend, the latency the consultations add, and
an estimated dollar cost when a price table is supplied. It reports every
invocation through an `on_report` hook and keeps cumulative totals for the
lifetime of the plugin.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
import copy
import dataclasses
import inspect
import logging
import sys
import threading
import time
from typing import Any
from typing import TYPE_CHECKING

from typing_extensions import override

from ...plugins.base_plugin import BasePlugin
from ._advisor import AdvisorUsage
from ._force_escalation_plugin import CONSULT_REQUIRED_STATUS
from ._model_consult_tool import ModelConsultTool

if TYPE_CHECKING:
  from ...agents.callback_context import CallbackContext
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse
  from ..base_tool import BaseTool
  from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)

_DEFAULT_PLUGIN_NAME = 'model_consult_observability'
_TOKENS_PER_PRICE_UNIT = 1_000_000
_UNKNOWN_MODEL = 'unknown'
# Most invocations tracked at once. A run that ends through cancellation never
# reaches an end-of-run callback, so without a bound the counters of such runs
# would accumulate for the life of the process.
_MAX_IN_FLIGHT_INVOCATIONS = 10_000
# Providers report token counts as int64 values. Larger values are read as
# malformed, which also keeps the cost arithmetic within the float range.
_MAX_TOKEN_COUNT = 2**63 - 1


@dataclasses.dataclass(kw_only=True)
class ModelConsultUsage:
  """Token usage of one model, summed over the calls made to it.

  Attributes:
    calls: Number of model calls counted.
    prompt_tokens: Input tokens, including tool-use prompt tokens when the
      provider reports them.
    output_tokens: Generated output tokens, excluding reasoning tokens.
    thoughts_tokens: Reasoning tokens.
    cached_tokens: Prompt tokens served from a context cache. They are also
      counted in `prompt_tokens`.
    total_tokens: Total tokens as the provider reports them, or the sum of
      prompt, output, and reasoning tokens when the provider omits the total.
  """

  calls: int = 0
  prompt_tokens: int = 0
  output_tokens: int = 0
  thoughts_tokens: int = 0
  cached_tokens: int = 0
  total_tokens: int = 0

  def to_dict(self) -> dict[str, int]:
    """Returns the counts as a JSON-serializable dictionary."""
    return {
        'calls': self.calls,
        'prompt_tokens': self.prompt_tokens,
        'output_tokens': self.output_tokens,
        'thoughts_tokens': self.thoughts_tokens,
        'cached_tokens': self.cached_tokens,
        'total_tokens': self.total_tokens,
    }


@dataclasses.dataclass(kw_only=True)
class ModelConsultStats:
  """Escalation, token, and latency counters for one or more invocations.

  The same type holds the counters of a single invocation, in
  `ModelConsultReport.stats`, and the cumulative totals of a plugin, in
  `ModelConsultObservabilityPlugin.stats`.

  Attributes:
    invocations: Runner invocations counted.
    failed_invocations: Invocations that ended with an unhandled exception.
    tool_calls: Calls to a `ModelConsultTool`, whatever their outcome.
    successful_consults: Calls that returned advisor guidance, with status
      `'ok'`.
    limited_consults: Calls refused because the turn or session budget was
      exhausted, with status `'limit_reached'`.
    failed_consults: Calls whose result has status `'error'` or a status the
      plugin does not recognize, and calls that raised an exception.
    invalid_consults: Calls rejected before reaching the advisor because the
      question was empty, with status `'invalid_request'`.
    advisor_latency_ms: Advisor latency summed over successful consultations, in
      milliseconds.
    wall_clock_ms: End-to-end latency summed over invocations, in milliseconds.
    advisor_usage: Advisor token usage keyed by advisor model.
    executor_usage: Token usage of every other model call the agents made, keyed
      by model.
  """

  invocations: int = 0
  failed_invocations: int = 0
  tool_calls: int = 0
  successful_consults: int = 0
  limited_consults: int = 0
  failed_consults: int = 0
  invalid_consults: int = 0
  advisor_latency_ms: float = 0.0
  wall_clock_ms: float = 0.0
  advisor_usage: dict[str, ModelConsultUsage] = dataclasses.field(
      default_factory=dict
  )
  executor_usage: dict[str, ModelConsultUsage] = dataclasses.field(
      default_factory=dict
  )

  @property
  def advisor_tokens(self) -> int:
    """Advisor tokens summed across advisor models."""
    return sum(usage.total_tokens for usage in self.advisor_usage.values())

  @property
  def executor_tokens(self) -> int:
    """Executor tokens summed across executor models."""
    return sum(usage.total_tokens for usage in self.executor_usage.values())

  @property
  def total_tokens(self) -> int:
    """Advisor and executor tokens combined."""
    return self.advisor_tokens + self.executor_tokens

  @property
  def executor_calls(self) -> int:
    """Executor model calls summed across executor models."""
    return sum(usage.calls for usage in self.executor_usage.values())

  @property
  def escalation_rate(self) -> float:
    """Successful consultations per executor model call.

    The rate is 0.0 when no executor call was counted, and it can exceed 1.0
    when a single model response requests several consultations.
    """
    executor_calls = self.executor_calls
    if not executor_calls:
      return 0.0
    return self.successful_consults / executor_calls

  def to_dict(self) -> dict[str, Any]:
    """Returns the counters and derived totals as a JSON-serializable dict."""
    return {
        'invocations': self.invocations,
        'failed_invocations': self.failed_invocations,
        'tool_calls': self.tool_calls,
        'successful_consults': self.successful_consults,
        'limited_consults': self.limited_consults,
        'failed_consults': self.failed_consults,
        'invalid_consults': self.invalid_consults,
        'advisor_latency_ms': self.advisor_latency_ms,
        'wall_clock_ms': self.wall_clock_ms,
        'executor_calls': self.executor_calls,
        'advisor_tokens': self.advisor_tokens,
        'executor_tokens': self.executor_tokens,
        'total_tokens': self.total_tokens,
        'escalation_rate': self.escalation_rate,
        'advisor_usage': {
            model: usage.to_dict()
            for model, usage in self.advisor_usage.items()
        },
        'executor_usage': {
            model: usage.to_dict()
            for model, usage in self.executor_usage.items()
        },
    }


@dataclasses.dataclass(kw_only=True)
class ModelConsultReport:
  """Summary of one invocation, passed to the `on_report` hook.

  Attributes:
    invocation_id: ID of the invocation.
    app_name: Name of the app that ran the invocation.
    user_id: ID of the user who owns the session.
    session_id: ID of the session the invocation ran in.
    agent_name: Name of the agent the invocation started with, or an empty
      string when the invocation started from a workflow node that is not an
      agent.
    stats: Counters for this invocation alone.
    estimated_cost_usd: Estimated cost of the invocation in US dollars, or
      `None` when the plugin has no price for any model the invocation used.
    error_type: Class name of the exception that ended the invocation, or `None`
      when the invocation completed.
  """

  invocation_id: str
  app_name: str
  user_id: str
  session_id: str
  agent_name: str
  stats: ModelConsultStats
  estimated_cost_usd: float | None = None
  error_type: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Returns the report as a JSON-serializable dictionary."""
    return {
        'invocation_id': self.invocation_id,
        'app_name': self.app_name,
        'user_id': self.user_id,
        'session_id': self.session_id,
        'agent_name': self.agent_name,
        'estimated_cost_usd': self.estimated_cost_usd,
        'error_type': self.error_type,
        'stats': self.stats.to_dict(),
    }


@dataclasses.dataclass(kw_only=True)
class _ExecutorCall:
  """Usage reported by one executor model call."""

  # The call's `EventActions`, whose identity keys the call. Holding the
  # reference keeps its `id()` from being reused while the run is in flight.
  actions: object
  requested_model: str | None = None
  model_version: str | None = None
  # Whether a final, non-partial response arrived. Until one does, the usage
  # of partial chunks stands in for a stream that the caller or an abort cut
  # off; afterwards, partial chunks no longer change the usage.
  completed: bool = False
  # A call can yield several final responses, each carrying the usage so far,
  # so the newest report replaces the previous one instead of adding to it.
  usage: AdvisorUsage | None = None


def _new_invocation_stats() -> ModelConsultStats:
  return ModelConsultStats(invocations=1)


@dataclasses.dataclass(kw_only=True)
class _InvocationState:
  """Counters for one in-flight invocation."""

  started_at: float = dataclasses.field(default_factory=time.perf_counter)
  stats: ModelConsultStats = dataclasses.field(
      default_factory=_new_invocation_stats
  )
  # Only calls that produced a response get an entry, so a live session,
  # which runs the model callbacks to screen every input and transcription
  # chunk, cannot grow this table.
  executor_calls: dict[int, _ExecutorCall] = dataclasses.field(
      default_factory=dict
  )
  # The model each agent last requested, keyed by agent name, for responses
  # that omit their model version.
  requested_models: dict[str, str] = dataclasses.field(default_factory=dict)
  # Tool contexts of consultations that raised and were already counted as
  # failed, so that a replacement result supplied by another plugin's
  # `on_tool_error_callback` is not counted again in `after_tool_callback`.
  errored_tool_contexts: dict[int, object] = dataclasses.field(
      default_factory=dict
  )

  def executor_call(self, actions: object, *, agent_name: str) -> _ExecutorCall:
    call = self.executor_calls.get(id(actions))
    if call is None:
      call = _ExecutorCall(
          actions=actions,
          requested_model=self.requested_models.get(agent_name),
      )
      self.executor_calls[id(actions)] = call
    return call


def _is_price(value: object) -> bool:
  """Returns whether `value` is a finite, non-negative number."""
  # The chained comparison also rejects NaN and infinities, and unlike
  # `math.isfinite()` it does not overflow on an int beyond the float range.
  return (
      not isinstance(value, bool)
      and isinstance(value, (int, float))
      and 0 <= value <= sys.float_info.max
  )


def _validate_prices(
    prices: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
  """Returns a copy of `prices` after checking every entry."""
  validated: dict[str, tuple[float, float]] = {}
  for key, value in (prices or {}).items():
    if not isinstance(key, str) or not key.strip():
      raise ValueError(f'prices keys must be non-empty strings, got {key!r}')
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or not all(_is_price(price) for price in value)
    ):
      raise ValueError(
          f'prices[{key!r}] must be a pair of non-negative numbers'
          f' (input_usd_per_1m, output_usd_per_1m), got {value!r}'
      )
    validated[key] = (float(value[0]), float(value[1]))
  return validated


def _non_negative_int(value: object) -> int:
  """Returns `value` as a token count, or 0 when it is not a valid count."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return 0
  if not 0 <= value <= _MAX_TOKEN_COUNT:
    return 0
  return int(value)


def _non_negative_float(value: object) -> float:
  """Returns `value` as a non-negative float, or 0.0 when it is not a number."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return 0.0
  if not 0 <= value <= sys.float_info.max:
    return 0.0
  return float(value)


def _advisor_usage_from_payload(raw: object) -> AdvisorUsage:
  """Reads the `usage` dictionary of a successful consultation result."""
  fields = raw if isinstance(raw, Mapping) else {}
  prompt = _non_negative_int(fields.get('prompt_tokens'))
  output = _non_negative_int(fields.get('output_tokens'))
  thoughts = _non_negative_int(fields.get('thoughts_tokens'))
  total = _non_negative_int(fields.get('total_tokens'))
  return AdvisorUsage(
      prompt_tokens=prompt,
      output_tokens=output,
      thoughts_tokens=thoughts,
      cached_tokens=_non_negative_int(fields.get('cached_tokens')),
      total_tokens=total or prompt + output + thoughts,
  )


def _add_tokens(
    *, target: ModelConsultUsage, source: AdvisorUsage | ModelConsultUsage
) -> None:
  """Adds the token counts of `source` to `target`."""
  target.prompt_tokens += source.prompt_tokens
  target.output_tokens += source.output_tokens
  target.thoughts_tokens += source.thoughts_tokens
  target.cached_tokens += source.cached_tokens
  target.total_tokens += source.total_tokens


def _add_call(
    table: dict[str, ModelConsultUsage], model: str, usage: AdvisorUsage
) -> None:
  """Adds one call and its usage to the bucket for `model`."""
  bucket = table.setdefault(model, ModelConsultUsage())
  bucket.calls += 1
  _add_tokens(target=bucket, source=usage)


def _merge_usage(
    *,
    target: dict[str, ModelConsultUsage],
    source: Mapping[str, ModelConsultUsage],
) -> None:
  """Adds every bucket in `source` to the matching bucket in `target`."""
  for model, usage in source.items():
    bucket = target.setdefault(model, ModelConsultUsage())
    bucket.calls += usage.calls
    _add_tokens(target=bucket, source=usage)


def _merge_stats(
    *, target: ModelConsultStats, source: ModelConsultStats
) -> None:
  """Adds every counter in `source` to `target`."""
  target.invocations += source.invocations
  target.failed_invocations += source.failed_invocations
  target.tool_calls += source.tool_calls
  target.successful_consults += source.successful_consults
  target.limited_consults += source.limited_consults
  target.failed_consults += source.failed_consults
  target.invalid_consults += source.invalid_consults
  target.advisor_latency_ms += source.advisor_latency_ms
  target.wall_clock_ms += source.wall_clock_ms
  _merge_usage(target=target.advisor_usage, source=source.advisor_usage)
  _merge_usage(target=target.executor_usage, source=source.executor_usage)


class ModelConsultObservabilityPlugin(BasePlugin):
  """Tracks `ModelConsultTool` escalations, token usage, latency, and cost.

  For every invocation, the plugin counts `ModelConsultTool` calls by outcome,
  the advisor tokens and latency that each successful consultation reports,
  the tokens of every other model call the agents make, and the invocation's
  wall-clock time. Advisor requests bypass the model callbacks, so advisor and
  executor tokens never mix. When an invocation ends, the plugin adds its
  counters to the cumulative `stats` and passes a `ModelConsultReport` to
  `on_report`.

  Example:
    ```python
    plugin = ModelConsultObservabilityPlugin(
        prices={
            'gemini-3.5-flash': (0.30, 2.50),
            'gemini-3.1-pro': (2.00, 12.00),
        },
        on_report=lambda report: metrics.emit(report.stats.to_dict()),
    )
    app = App(name='demo', root_agent=root_agent, plugins=[plugin])
    runner = Runner(app=app, session_service=session_service)
    ...
    print(plugin.summary())
    ```

  Register the plugin ahead of other plugins. The runner stops calling the
  remaining plugins for a callback once one of them returns a value or raises,
  so an earlier plugin can hide model responses, tool results, or the end of a
  run from this one.

  Attributes:
    name: Unique name of this plugin instance within a runner.
  """

  def __init__(
      self,
      *,
      name: str = _DEFAULT_PLUGIN_NAME,
      prices: Mapping[str, tuple[float, float]] | None = None,
      on_report: Callable[[ModelConsultReport], object] | None = None,
  ) -> None:
    """Initializes the plugin.

    Args:
      name: Plugin name, unique within a runner.
      prices: Optional price table that maps a model-name substring to
        `(input_usd_per_1m_tokens, output_usd_per_1m_tokens)`. A model takes the
        price of the longest key its name contains, so a broad key also prices
        other models whose names contain it. Without a table, the plugin does
        not estimate cost.
      on_report: Optional callable that receives a `ModelConsultReport` when
        each invocation ends. It can be a regular function or a coroutine
        function; the plugin awaits an awaitable result. Exceptions it raises
        are logged and do not affect the run.

    Raises:
      ValueError: If `name` is empty, a `prices` key is empty, or a `prices`
        value is not a pair of finite, non-negative numbers.
    """
    if not name or not name.strip():
      raise ValueError('name must be a non-empty string')
    super().__init__(name=name)
    self._prices = _validate_prices(prices)
    self._on_report = on_report
    self._states: dict[str, _InvocationState] = {}
    # Guards the cumulative totals and changes to the set of tracked
    # invocations. The synchronous `Runner.run` drives each run on its own
    # thread, and one instance can serve several runners, so runs sharing this
    # plugin can start and finish concurrently.
    self._lock = threading.Lock()
    self._totals = ModelConsultStats()
    self._warned_unpriced_models: set[str] = set()

  @property
  def stats(self) -> ModelConsultStats:
    """A snapshot of the cumulative counters over finished invocations.

    The snapshot is a copy, so changing it does not affect the plugin.
    """
    with self._lock:
      return copy.deepcopy(self._totals)

  def reset(self) -> ModelConsultStats:
    """Clears the cumulative counters and returns the counters it cleared.

    The plugin takes the counters and clears them in one step, so an
    invocation that finishes on another thread at the same moment lands in
    exactly one of the returned counters and the new totals. Calling `reset()`
    on a schedule therefore exports consecutive reporting windows without gaps
    or overlaps.

    Invocations still in flight are unaffected: the plugin still reports them
    when they end, and their counters land in the new totals.

    Returns:
      The cumulative counters as they were before the reset.
    """
    with self._lock:
      cleared, self._totals = self._totals, ModelConsultStats()
    return cleared

  def estimate_cost(
      self, stats: ModelConsultStats | None = None
  ) -> float | None:
    """Estimates a cost in US dollars from the `prices` table.

    Prompt tokens are priced at the input rate, and output and reasoning tokens
    at the output rate. Models without a price add nothing to the estimate.

    Args:
      stats: The counters to price. Defaults to the cumulative counters.

    Returns:
      The estimated cost, or `None` when no price matches any model in `stats`.
    """
    cost, _ = self._cost(self.stats if stats is None else stats)
    return cost

  def summary(self) -> str:
    """Returns a human-readable summary of the cumulative counters."""
    stats = self.stats
    lines = [
        'Model consult summary',
        (
            f'  invocations          : {stats.invocations}'
            f' (failed: {stats.failed_invocations})'
        ),
        f'  consult tool calls   : {stats.tool_calls}',
        (
            f'  successful consults  : {stats.successful_consults}'
            f' (limited: {stats.limited_consults},'
            f' failed: {stats.failed_consults},'
            f' invalid: {stats.invalid_consults})'
        ),
        (
            f'  escalation rate      : {stats.escalation_rate:.2%} of'
            f' {stats.executor_calls} executor calls'
        ),
        f'  advisor tokens       : {stats.advisor_tokens}',
        f'  executor tokens      : {stats.executor_tokens}',
        f'  total tokens         : {stats.total_tokens}',
        f'  total advisor ms     : {stats.advisor_latency_ms:.0f}',
        f'  total e2e ms         : {stats.wall_clock_ms:.0f}',
    ]
    cost = self.estimate_cost(stats)
    if cost is not None:
      lines.append(f'  estimated cost (USD) : {cost:.6f}')
    for role, table in (
        ('advisor', stats.advisor_usage),
        ('executor', stats.executor_usage),
    ):
      for model, usage in sorted(table.items()):
        lines.append(
            f'  {role:<8} {model}: calls={usage.calls}'
            f' prompt={usage.prompt_tokens} output={usage.output_tokens}'
            f' thoughts={usage.thoughts_tokens} cached={usage.cached_tokens}'
            f' total={usage.total_tokens}'
        )
    return '\n'.join(lines)

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Starts tracking an invocation and its wall-clock time.

    Args:
      invocation_context: The context of the invocation that is starting.
    """
    invocation_id = invocation_context.invocation_id
    evicted: list[str] = []
    with self._lock:
      # Re-inserting moves a resumed invocation behind the older entries.
      self._states.pop(invocation_id, None)
      self._states[invocation_id] = _InvocationState()
      while len(self._states) > _MAX_IN_FLIGHT_INVOCATIONS:
        oldest_id = next(iter(self._states))
        self._states.pop(oldest_id)
        evicted.append(oldest_id)
    for oldest_id in evicted:
      logger.warning(
          'More than %d invocations are in flight; dropping the counters of'
          ' invocation %s, which started first and has not finished.',
          _MAX_IN_FLIGHT_INVOCATIONS,
          oldest_id,
      )

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    """Records the model an agent requests, for responses without a version.

    Args:
      callback_context: The callback context of the model call.
      llm_request: The request about to be sent to the executor model.
    """
    state = self._states.get(callback_context.invocation_id)
    if state is None or not llm_request.model:
      return
    state.requested_models[callback_context.agent_name] = llm_request.model

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> None:
    """Records the token usage of an executor model call.

    A call counts once however many responses it yields, and the newest usage
    report wins. Once a final response arrives, later partial chunks are
    ignored; until then, the usage of partial chunks stands in for a stream
    that ends early. Output transcriptions are ignored, because a live session
    passes them to plugins to screen them, not as the response of a model
    call.

    Args:
      callback_context: The callback context of the model call.
      llm_response: A response received from the executor model.
    """
    if llm_response.output_transcription is not None:
      return
    state = self._states.get(callback_context.invocation_id)
    if state is None:
      return
    call = state.executor_call(
        callback_context.actions, agent_name=callback_context.agent_name
    )
    if llm_response.partial and call.completed:
      return
    if not llm_response.partial:
      call.completed = True
    if llm_response.model_version:
      call.model_version = llm_response.model_version
    if llm_response.usage_metadata is not None:
      call.usage = AdvisorUsage.from_metadata(llm_response.usage_metadata)

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> None:
    """Counts a `ModelConsultTool` call by the status of its result.

    A call that `ForceEscalationPlugin` refuses before the tool runs is not
    counted.

    Args:
      tool: The tool that ran. Tools other than `ModelConsultTool` are ignored.
      tool_args: The arguments the tool ran with.
      tool_context: The context of the tool call.
      result: The dictionary the tool returned.
    """
    del tool_args
    if not isinstance(tool, ModelConsultTool):
      return
    state = self._states.get(tool_context.invocation_id)
    if state is None:
      return
    if state.errored_tool_contexts.pop(id(tool_context), None) is not None:
      # Counted as failed when the tool raised; `result` is a replacement.
      return
    payload: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
    status = payload.get('status')
    if status == CONSULT_REQUIRED_STATUS:
      return
    stats = state.stats
    stats.tool_calls += 1
    if status == 'ok':
      stats.successful_consults += 1
      stats.advisor_latency_ms += _non_negative_float(payload.get('latency_ms'))
      model = payload.get('advisor_model')
      if not isinstance(model, str) or not model:
        model = tool.advisor_model.model or _UNKNOWN_MODEL
      _add_call(
          stats.advisor_usage,
          model,
          _advisor_usage_from_payload(payload.get('usage')),
      )
    elif status == 'limit_reached':
      stats.limited_consults += 1
    elif status == 'invalid_request':
      stats.invalid_consults += 1
    else:
      stats.failed_consults += 1

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> None:
    """Counts a `ModelConsultTool` call that raised as a failed consultation.

    Args:
      tool: The tool that raised. Tools other than `ModelConsultTool` are
        ignored.
      tool_args: The arguments the tool ran with.
      tool_context: The context of the tool call.
      error: The exception the tool raised.
    """
    del tool_args, error
    if not isinstance(tool, ModelConsultTool):
      return
    state = self._states.get(tool_context.invocation_id)
    if state is None:
      return
    state.stats.tool_calls += 1
    state.stats.failed_consults += 1
    state.errored_tool_contexts[id(tool_context)] = tool_context

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Reports a completed invocation and adds it to the cumulative totals.

    Args:
      invocation_context: The context of the invocation that completed.
    """
    await self._finish(invocation_context, error=None)

  @override
  async def on_run_error_callback(
      self,
      *,
      invocation_context: InvocationContext,
      error: Exception,
  ) -> None:
    """Reports a failed invocation and adds it to the cumulative totals.

    Args:
      invocation_context: The context of the invocation that failed.
      error: The unhandled exception that ended the invocation.
    """
    await self._finish(invocation_context, error=error)

  def _price_for(self, model: str) -> tuple[float, float] | None:
    best_key = max(
        (key for key in self._prices if key in model), key=len, default=None
    )
    return None if best_key is None else self._prices[best_key]

  def _cost(self, stats: ModelConsultStats) -> tuple[float | None, list[str]]:
    """Returns the estimated cost of `stats` and the models it left unpriced."""
    if not self._prices:
      return None, []
    total = 0.0
    priced = False
    unpriced: list[str] = []
    for table in (stats.advisor_usage, stats.executor_usage):
      for model, usage in table.items():
        price = self._price_for(model)
        if price is None:
          unpriced.append(model)
          continue
        priced = True
        input_price, output_price = price
        billed_output = usage.output_tokens + usage.thoughts_tokens
        total += (
            usage.prompt_tokens * input_price + billed_output * output_price
        ) / _TOKENS_PER_PRICE_UNIT
    return (total if priced else None), unpriced

  async def _finish(
      self, invocation_context: InvocationContext, *, error: Exception | None
  ) -> None:
    with self._lock:
      state = self._states.pop(invocation_context.invocation_id, None)
    if state is None:
      return
    stats = state.stats
    stats.wall_clock_ms = (time.perf_counter() - state.started_at) * 1000
    if error is not None:
      stats.failed_invocations = 1
    for call in state.executor_calls.values():
      model = call.model_version or call.requested_model or _UNKNOWN_MODEL
      _add_call(stats.executor_usage, model, call.usage or AdvisorUsage())

    cost, unpriced = self._cost(stats)
    with self._lock:
      _merge_stats(target=self._totals, source=stats)
      newly_unpriced = sorted(set(unpriced) - self._warned_unpriced_models)
      self._warned_unpriced_models.update(newly_unpriced)
    for model in newly_unpriced:
      logger.warning(
          'No price in the price table matches model %r; its tokens are left'
          ' out of estimated costs.',
          model,
      )

    report = ModelConsultReport(
        invocation_id=invocation_context.invocation_id,
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        agent_name=getattr(invocation_context.agent, 'name', None) or '',
        stats=stats,
        estimated_cost_usd=cost,
        error_type=None if error is None else type(error).__name__,
    )
    logger.debug(
        'Model consult invocation %s: %d of %d consults succeeded, advisor'
        ' tokens=%d, executor tokens=%d, wall clock=%.0f ms',
        report.invocation_id,
        stats.successful_consults,
        stats.tool_calls,
        stats.advisor_tokens,
        stats.executor_tokens,
        stats.wall_clock_ms,
    )
    await self._emit(report)

  async def _emit(self, report: ModelConsultReport) -> None:
    if self._on_report is None:
      return
    try:
      outcome = self._on_report(report)
      if inspect.isawaitable(outcome):
        await outcome
    except Exception:  # pylint: disable=broad-exception-caught
      # Exporting metrics must never fail the run being measured.
      logger.warning(
          'on_report failed for invocation %s.',
          report.invocation_id,
          exc_info=True,
      )
