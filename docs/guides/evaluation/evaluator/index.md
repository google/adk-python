# Evaluator

`Evaluator` is the interface behind all thirteen built-in evaluation metrics.
Implementing it yourself, either as a plain function or as a class, is how you
score a rule that is specific to your agent and that no general-purpose metric
can express.

## Introduction

The built-in metrics answer general questions: did the agent call the tools the
recording expected, does its answer resemble the golden one, would a judge model
call it grounded. Real agents also have rules that are specific to them. A
thermostat agent must never set a temperature outside a safe band. A support
agent must never quote a price it did not look up. A booking agent must not
confirm before it has checked availability.

Those are cheap to check in Python and impossible to express as a threshold on
someone else's metric, so evaluation lets you supply your own. There are two
ways in, and they differ in how much machinery you take on.

A **custom metric function** is the light one. You write a function with a fixed
four-argument signature, name it by dotted path in the eval config, and the
framework wraps it for you. Nothing needs registering and nothing needs
importing at the right moment, so the metric lives entirely in the config file
and the module it names.

An **`Evaluator` subclass** is the heavier one. It exists for a metric that needs
per-run construction, such as a client to build, an expensive model to load
once, or a criterion type of its own with extra config keys. A config file
cannot name a class, only a function, so a subclass has to be registered from
Python before the run starts.

Take the function unless one of those three needs applies, because a function
costs you nothing beyond the function, while a subclass adds a registration step
that has to run in the same process as the evaluation.

Either way, the object the run actually calls is an `Evaluator`, and the thing it
must produce is an `EvaluationResult`.

## Get started

A custom metric function takes four arguments and returns an `EvaluationResult`.
This one fails any invocation where the agent set a temperature outside a safe
range:

```python
from typing import Optional

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult

_SAFE_MIN = 18
_SAFE_MAX = 30


def _is_safe(invocation: Invocation) -> bool:
  for call in get_all_tool_calls(invocation.intermediate_data):
    if call.name != "set_temperature":
      continue
    temperature = (call.args or {}).get("temperature")
    if temperature is not None and not (_SAFE_MIN <= temperature <= _SAFE_MAX):
      return False
  return True


def temperature_safety_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario],
) -> EvaluationResult:
  """Scores 1.0 unless a set_temperature call left the safe range."""
  per_invocation_results = []
  for invocation in actual_invocations:
    safe = _is_safe(invocation)
    per_invocation_results.append(
        PerInvocationResult(
            actual_invocation=invocation,
            score=1.0 if safe else 0.0,
            eval_status=EvalStatus.PASSED if safe else EvalStatus.FAILED,
        )
    )

  if not per_invocation_results:
    return EvaluationResult()

  overall_score = sum(r.score for r in per_invocation_results) / len(
      per_invocation_results
  )
  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=(
          EvalStatus.PASSED if overall_score == 1.0 else EvalStatus.FAILED
      ),
      per_invocation_results=per_invocation_results,
  )
```

Wire it up in the eval config by naming it in `criteria` and pointing
`custom_metrics` at its dotted path:

```json
{
  "criteria": {
    "temperature_safety_score": 1.0
  },
  "custom_metrics": {
    "temperature_safety_score": {
      "code_config": {"name": "temperature_safety.temperature_safety_score"},
      "description": "Fails if any set_temperature call is outside 18-30 Celsius."
    }
  }
}
```

The module part of that path is imported with `importlib`, so it has to be
importable from wherever the evaluation runs. In this example,
`temperature_safety.py` sits beside the config and the eval data.

## How it works

When a run starts, the eval config is walked and every entry under
`custom_metrics` is registered into a `MetricEvaluatorRegistry` against its
dotted path. `AgentEvaluator` registers them into a **fork** of the
process-wide default, so the metrics one test declared do not leak into the
next one. `adk eval` registers them into the default registry itself, which is
fine for a single-shot command.

Scoring a metric then resolves that dotted path: it is split at its last dot,
the module is imported, and the attribute is fetched. A module that will not
import, a name that is not there, and a path with no dot in it all surface the
same way, as `ImportError: Could not import custom metric function from
<path>`. An attribute that exists but is not callable raises `TypeError`
instead.

Your function is then called **positionally with exactly four arguments**, so its
arity matters and its parameter names do not. If the return value is awaitable
it is awaited, which is why an `async def` metric works with no extra
declaration.

Three things about that call catch people out.

*   **`eval_metric.threshold` is deliberately `None`.** The metric you receive
    has that field blanked, because comparing a score to a threshold is the
    framework's job, not the metric's. Read the metric's name and its criterion
    if you need them; do not read `threshold`.
*   **The contract on the returned result is stricter than it looks.** Set
    `overall_eval_status`, and return exactly one `PerInvocationResult` for
    every entry in `actual_invocations`, in the same order. Returning a
    different number raises
    `ValueError: Eval metric should return results for each invocation.` and
    stops the run. Leaving `overall_eval_status` at its `NOT_EVALUATED` default
    is worse than an error, because it is quiet: the per-invocation results you
    computed are thrown away and replaced with empty ones, so the metric reports
    no score at all.
*   **An exception inside your metric does not fail loudly.** The eval service
    catches it, logs the traceback, and substitutes an empty result with status
    `NOT_EVALUATED` so that one broken metric cannot take down the others. Under
    `AgentEvaluator` that still fails the test, but the message you get is
    `Expected 1.0, but got None.`, which tells you nothing. When you see that,
    go and look in the log for the real error.

## The result objects

`EvaluationResult` is what a metric returns.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `overall_score` | `float \| None` | `None` | Aggregate score across invocations. |
| `overall_eval_status` | `EvalStatus` | `NOT_EVALUATED` | Verdict for the metric. Must be set. |
| `per_invocation_results` | `list[PerInvocationResult]` | `[]` | One entry per actual invocation, in order. |
| `overall_rubric_scores` | `list[RubricScore] \| None` | `None` | Only for rubric-based metrics. |

`PerInvocationResult` is one row of that list.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `actual_invocation` | `Invocation` | required | The invocation this row scores. |
| `expected_invocation` | `Invocation \| None` | `None` | Its recorded counterpart, when the metric used one. |
| `score` | `float \| None` | `None` | Score for this invocation. |
| `eval_status` | `EvalStatus` | `NOT_EVALUATED` | Verdict for this invocation. |
| `rubric_scores` | `list[RubricScore] \| None` | `None` | Per-rubric detail, for rubric-based metrics. |

`EvalStatus` has three members: `PASSED`, `FAILED`, and `NOT_EVALUATED`.

How those two levels combine depends on who is driving. `LocalEvalService`, and
so `adk eval`, reads the statuses: a case is `FAILED` as soon as any metric
reports `FAILED`, `PASSED` if at least one passed and none failed, and
`NOT_EVALUATED` otherwise. `AgentEvaluator` additionally averages the
per-invocation `score` values across every run and compares that mean against
the configured threshold, which is where its failure messages come from. Setting
both the scores and the statuses, as the example above does, satisfies both.

## Advanced applications

The class-based route is what makes per-run construction and extra configuration
keys available, and those two capabilities are worth taking separately.

### Write an `Evaluator` subclass

Subclass `Evaluator` when the metric needs setup that should happen once per
run rather than once per call. The constructor is invoked with a single keyword
argument, `eval_metric=`, and `evaluate_invocations` may be sync or async.

```python
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import Evaluator
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult


class ResponseLengthEvaluator(Evaluator):
  """Scores 1.0 when the final response stays under a character budget."""

  criterion_type = BaseCriterion

  def __init__(self, eval_metric: EvalMetric):
    self._threshold = eval_metric.criterion.threshold

  def evaluate_invocations(
      self,
      actual_invocations,
      expected_invocations=None,
      conversation_scenario=None,
  ) -> EvaluationResult:
    results = []
    for invocation in actual_invocations:
      response = invocation.final_response
      parts = (response.parts or []) if response else []
      length = len("".join(part.text or "" for part in parts))
      score = 1.0 if length <= 200 else 0.0
      results.append(
          PerInvocationResult(
              actual_invocation=invocation,
              score=score,
              eval_status=(
                  EvalStatus.PASSED if score else EvalStatus.FAILED
              ),
          )
      )

    overall = sum(r.score for r in results) / len(results)
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=(
            EvalStatus.PASSED
            if overall >= self._threshold
            else EvalStatus.FAILED
        ),
        per_invocation_results=results,
    )
```

Register it before the run, on the process-wide default registry:

```python
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY

DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
    metric_info=MetricInfo(
        metric_name="response_length",
        description="Penalizes over-long final responses.",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    ),
    evaluator=ResponseLengthEvaluator,
)
```

It has to be the *default* registry specifically. An `AgentEvaluator` run forks
that registry rather than building a fresh one, precisely so that classes
registered here stay resolvable; the fork is what keeps an eval config's
function-based metrics local to one run while your class-based ones remain
available to every run. The metric then needs an ordinary entry in `criteria`
and no `custom_metrics` entry, since the registry already knows the name.

The registration has to happen in the same process as the evaluation, which in
practice means a programmatic run: a `conftest.py` for a pytest suite, or the
script that calls the eval service. There is no hook that would let `adk eval`
pick up a class from a config file.

### Declare your own criterion type

`criterion_type` is a `ClassVar` naming the criterion class your evaluator
expects. The built-in evaluators set it and then re-validate the incoming
criterion into that type, which is how `match_type` reaches
`TrajectoryEvaluator` and `judge_model_options` reaches the judge-based ones.
Subclass `BaseCriterion` with your own fields, point `criterion_type` at it, and
validate in your constructor to get the same behavior. The extra keys survive
config parsing because `BaseCriterion` allows extras, so they are already there
waiting for you. See the [eval config guide](../eval_config/index.md) for how
that two-stage validation works.

## Limitations

*   **A config can only name a function.** Class-based metrics cannot be
    declared in `eval_config.json` at all; they require Python that runs before
    the evaluation does.
*   **Registration is process-global.** `DEFAULT_METRIC_EVALUATOR_REGISTRY` is a
    module-level singleton, so registering the same metric name twice replaces
    the first registration and logs it. Constructing any registry also emits an
    experimental-feature warning, `MetricEvaluatorRegistry` being marked
    experimental.
*   **Nothing is re-exported at package level.** Import from
    `google.adk.evaluation.evaluator` and
    `google.adk.evaluation.metric_evaluator_registry` directly; the package
    `__init__` exports only `AgentEvaluator`.
*   **Errors are swallowed.** A metric that raises degrades to `NOT_EVALUATED`
    with a log line rather than surfacing the exception to the caller.

## Related samples

*   [Custom metric](../../../../contributing/samples/evaluation/custom_metric/temperature_safety.py)
    is the worked function the example above is adapted from, alongside the
    [eval config](../../../../contributing/samples/evaluation/custom_metric/eval_config.json)
    that wires it in.
*   [Evaluation samples](../../../../contributing/samples/evaluation) holds the
    shared agent and the other five evaluation techniques.

## Related guides

*   [EvalConfig and the eval config file](../eval_config/index.md) covers how a
    metric name and its criterion reach the registry.
*   [BaseEvalService and LocalEvalService](../eval_service/index.md) is the
    service that constructs your evaluator and calls it, and where a metric
    that raises is turned into an empty `NOT_EVALUATED` result.
