# BaseEvalService and LocalEvalService

The eval service returns evaluation results as data rather than as a test that
passed or failed: a score per metric per case, which a build job can post to a
dashboard and compare against last week. `BaseEvalService` splits evaluation
into two phases you call separately, where
`perform_inference` runs the agent and hands back what it produced, and
`evaluate` scores those results. `LocalEvalService` is the implementation that
runs both in your process, and it is what
[`AgentEvaluator`](../agent_evaluator/index.md) and `adk eval` sit on top of.

## Introduction

`AgentEvaluator` is one call that ends in an `assert`, which is exactly right
inside a test and awkward everywhere else. A continuous-integration job usually
wants more than pass or fail: the score per metric per case, so it can post a
summary; the results persisted, so it can compare against last week; and often
the two phases pulled apart, so that an expensive inference run can be scored
twice against different thresholds without paying for the model again.

That separation is the whole reason the service interface exists.
`perform_inference` is where the money goes, because it executes your agent
against every user turn in an eval set. `evaluate` takes the results of that
and is usually cheap, unless you configured a judge-model metric. Because
inference results are ordinary Pydantic models, you can keep them, and
rescoring later costs nothing.

Both methods are async generators that yield each result as it becomes
available, rather than returning a list at the end. A long eval set reports its
first case in seconds.

## Get started

Score an agent and fail the job if any case failed. The eval set here is built
in memory so that the snippet stands alone; a real job would load it from disk
with `LocalEvalSetsManager`.

```python
from google.adk.agents import LlmAgent
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.genai import types

APP_NAME = "home_automation"
EVAL_SET_ID = "smoke"

expected = Invocation(
    user_content=types.Content(
        role="user", parts=[types.Part(text="Turn off the bedroom light.")]
    ),
    final_response=types.Content(
        role="model", parts=[types.Part(text="The bedroom light is off.")]
    ),
    intermediate_data=IntermediateData(
        tool_uses=[
            types.FunctionCall(
                name="set_light", args={"room": "bedroom", "on": False}
            )
        ]
    ),
)

eval_sets_manager = InMemoryEvalSetsManager()
eval_sets_manager.create_eval_set(app_name=APP_NAME, eval_set_id=EVAL_SET_ID)
eval_sets_manager.add_eval_case(
    app_name=APP_NAME,
    eval_set_id=EVAL_SET_ID,
    eval_case=EvalCase(eval_id="turn_off_bedroom", conversation=[expected]),
)

eval_service = LocalEvalService(
    root_agent=root_agent, eval_sets_manager=eval_sets_manager
)
metrics = get_eval_metrics_from_config(
    EvalConfig(criteria={"tool_trajectory_avg_score": 1.0})
)


async def run_eval() -> bool:
  inference_results = []
  async for result in eval_service.perform_inference(
      InferenceRequest(
          app_name=APP_NAME,
          eval_set_id=EVAL_SET_ID,
          inference_config=InferenceConfig(parallelism=4),
      )
  ):
    inference_results.append(result)

  all_passed = True
  async for case_result in eval_service.evaluate(
      EvaluateRequest(
          inference_results=inference_results,
          evaluate_config=EvaluateConfig(eval_metrics=metrics),
      )
  ):
    for metric in case_result.overall_eval_metric_results:
      print(
          f"{case_result.eval_id} {metric.metric_name}:"
          f" {metric.score} (threshold {metric.threshold}) {metric.eval_status.name}"
      )
    all_passed &= case_result.final_eval_status == EvalStatus.PASSED
  return all_passed
```

On a passing run that prints:

```text
turn_off_bedroom tool_trajectory_avg_score: 1.0 (threshold 1.0) PASSED
```

and when the agent calls `set_light` for the kitchen instead, the same line
reads `0.0 (threshold 1.0) FAILED`.

`root_agent` is your own agent. `get_eval_metrics_from_config` is the bridge
from the [eval config file format](../eval_config/index.md) to the
`list[EvalMetric]` that `EvaluateConfig` wants. If you would rather not go
through a config file at all, construct `EvalMetric` objects directly.

## How it works

A run has two phases, and you call them separately. The sections below take them
in the order they happen.

### The two phases, and what each one costs

**`perform_inference`** loads the eval set by id, selects the cases named in
`eval_case_ids`, or all of them when that list is empty, and runs the agent
against each one, at most
`InferenceConfig.parallelism` of them at a time. Each result is yielded as soon
as that case finishes, so they arrive out of order.

This phase makes real model calls. Credentials and quota are required even when
every metric you plan to use is deterministic, because producing the output to
score is itself a model call.

**A failed inference is a result, not an exception.** Any exception from the
agent run is caught, logged, and turned into an `InferenceResult` with
`status=InferenceStatus.FAILURE` and the message in `error_message`. The
generator keeps going, so one broken case cannot abort the run. Check `status`
before you treat `inferences` as data. A build job that only looks at eval
scores reports a clean run for an agent that crashed on every case, because
a failed inference produces no metric results to fail on.

**`evaluate`** takes those results back, again at most
`EvaluateConfig.parallelism` at a time, and scores each one against every metric.
It re-reads the eval case from the eval sets manager to get the expected
conversation, which is why the manager is a constructor argument rather than
something you pass per request. An `InferenceResult` naming a case the manager
does not have raises `NotFoundError`.

### Statuses

Each metric produces an `EvalMetricResult` with a score and an `EvalStatus`.
Those combine into the case's `final_eval_status`: `FAILED` as soon as any
metric failed, `PASSED` if at least one passed and none failed, and
`NOT_EVALUATED` when every metric declined to score.

`NOT_EVALUATED` is the one to watch. It is what you get when a metric raised,
because the exception is caught and logged and the metric contributes an empty
result. A case in that state is neither a pass nor a failure, so a job that
tests `!= FAILED` treats it as success. Test `== PASSED` instead.

### Persist results

Pass an `eval_set_results_manager` and `evaluate` saves the results grouped by
eval set, once the last case for that set has been yielded rather than case by
case. `LocalEvalSetResultsManager(agents_dir=...)` writes one
`*.evalset_result.json` per eval set under
`<agents_dir>/<app_name>/.adk/eval_history/`. Without a manager, nothing is
written and the yielded objects are all you get.

### Where the eval cases come from

`LocalEvalService` never reads a file itself. It takes an `EvalSetsManager` and
asks it for eval sets and eval cases by id. Three implementations ship:

*   `InMemoryEvalSetsManager` holds cases you build in Python, which suits
    generated or parameterized suites, and tests.
*   `LocalEvalSetsManager(agents_dir=...)` reads `*.evalset.json` files on disk
    under the agents directory, and is what `adk eval` uses.
*   `GcsEvalSetsManager` reads the same layout from a Cloud Storage bucket.

Implement `EvalSetsManager` yourself for cases held anywhere else. It is seven
methods, and `LocalEvalService` only calls `get_eval_set` and `get_eval_case`.

## Configuration options

Settings live on the service itself, which is where the agent and its backing
services are wired in, and on one config object per phase.

### LocalEvalService

The constructor takes the agent, the source of eval cases, and the services the
inference runs use.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root_agent` | `BaseAgent` | required | The agent to evaluate. |
| `eval_sets_manager` | `EvalSetsManager` | required | Where eval sets and cases are read from. |
| `metric_evaluator_registry` | `MetricEvaluatorRegistry \| None` | `None` | Resolves metric names to evaluators. Defaults to the process-wide registry. |
| `session_service` | `BaseSessionService \| None` | `None` | Sessions for the inference runs. Defaults to in-memory. |
| `artifact_service` | `BaseArtifactService \| None` | `None` | Artifacts for the inference runs. Defaults to in-memory. |
| `eval_set_results_manager` | `EvalSetResultsManager \| None` | `None` | Persists results. Nothing is written when omitted. |
| `session_id_supplier` | `Callable[[], str]` | random `___eval___session___*` | Generates the session id for a case that does not pin one. |
| `user_simulator_provider` | `UserSimulatorProvider` | `UserSimulatorProvider()` | Builds the simulated user for a case that has a `conversation_scenario`. Pass one built from your `EvalConfig.user_simulator_config` to use those settings. |
| `memory_service` | `BaseMemoryService \| None` | `None` | Memory service for the inference runs. |
| `app` | `App \| None` | `None` | Keyword-only. Run inference through an `App` rather than a bare agent. |

**`app`** is the one to reach for when your agent is not the whole story. Pass
the `App` and inference runs through a runner built from it, so `app.plugins`,
`app.context_cache_config` and `app.resumability_config` are in force during
the eval. Pass only `root_agent` and none of them are, which means an eval can
pass while production, running the same agent under a plugin that rewrites tool
results, behaves differently.

**`artifact_service`** matters when a case depends on a file that must already
exist. Pre-load it, pass the service here, and pin the case to a session id
through `SessionInput.session_id` so the lookup resolves.

**`metric_evaluator_registry`** lets you scope custom metric registrations to
one service instead of the process-wide default. See
[Evaluator](../evaluator/index.md).

### InferenceConfig

`InferenceConfig` travels on an `InferenceRequest` and governs the phase that
runs the agent.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `parallelism` | `int` | `4` | Eval cases inferred concurrently. |
| `labels` | `dict[str, str] \| None` | `None` | Metadata attached for billing breakdown. |
| `use_live` | `bool` | `False` | Use bidirectional streaming. Required for Live API models. |
| `live_timeout_seconds` | `int` | `300` | How long to wait for a model turn in live mode. |

`parallelism` is bounded by your model quota, not your CPU. Models enforce
per-minute limits, so raising it on a large eval set is a reliable way to start
collecting rate-limit errors, and those arrive as failed inferences rather than
as an exception you would notice.

### EvaluateConfig

`EvaluateConfig` travels on an `EvaluateRequest` and governs the scoring phase.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `eval_metrics` | `list[EvalMetric]` | required | The metrics to score with. |
| `parallelism` | `int` | `4` | Eval cases scored concurrently. |

Its `parallelism` is a separate number from the inference one and matters only
for judge-model metrics, which are themselves model calls. Deterministic
metrics are fast enough that the value makes no difference.

## Advanced applications

Separating the phases is what makes the rest of this possible. Each section
below is something you can do only because inference and scoring are two calls.

### Score a run again without rerunning it

Rescoring is the reason to hold the two phases apart. `InferenceResult` is a
Pydantic model, so you can keep a run and score it again later against a
stricter threshold, an extra metric, or a metric that did not exist when the
run happened.

```python
# After an expensive inference run:
saved = [r.model_dump_json() for r in inference_results]

# Later, in a different process:
restored = [InferenceResult.model_validate_json(s) for s in saved]
async for case_result in eval_service.evaluate(
    EvaluateRequest(
        inference_results=restored,
        evaluate_config=EvaluateConfig(eval_metrics=stricter_metrics),
    )
):
  ...
```

The eval sets manager must still hold the same cases under the same ids, since
`evaluate` re-reads the expected conversation from it.

### Evaluate a subset

`InferenceRequest.eval_case_ids` narrows a run to named cases, which is how you
re-run the three that failed rather than all two hundred.

```python
InferenceRequest(
    app_name=APP_NAME,
    eval_set_id=EVAL_SET_ID,
    eval_case_ids=["turn_off_bedroom", "set_thermostat"],
    inference_config=InferenceConfig(),
)
```

A name that matches nothing is silently skipped rather than raising, so a typo
quietly shrinks the run.

### Implement your own eval service

There are two abstract methods to fill in, both async generators. The usual
reason to write your own is a remote evaluation backend, with inference in your
process and scoring by a hosted service, or the other way round. Keep the
streaming contract when you do, yielding each result as it is ready rather than
collecting a list, because callers rely on that to report progress.

## Limitations

*   **`LocalEvalService` is experimental.** It carries the `@experimental`
    decorator and warns on construction; treat the constructor signature as
    subject to change.
*   **It needs the evaluation extra.** `base_eval_service` imports on a base
    install, but `local_eval_service` pulls in `vertexai` through
    `google-cloud-aiplatform[evaluation]`. Install `google-adk[eval]`.
*   **Nothing is re-exported at package level.** Import from
    `google.adk.evaluation.base_eval_service` and
    `google.adk.evaluation.local_eval_service`; `google.adk.evaluation` exports
    only `AgentEvaluator`.
*   **Failures are absorbed at both layers.** A failed inference becomes a
    `FAILURE` result, and a metric that raises becomes `NOT_EVALUATED`. Neither
    reaches the caller as an exception, so a job that does not check statuses
    reports a green run over an agent that never worked.
*   **A static eval case must match invocation for invocation.** Unless the
    case uses a `conversation_scenario`, an inference result with a different
    number of invocations than the recorded conversation raises `ValueError`
    from `evaluate`.
*   **There is no cancellation.** The generators run to completion; stopping a
    long inference run means canceling the surrounding task, and the in-flight
    model calls are not cleaned up.
*   **Results are persisted per eval set, at the end.** A run interrupted part
    way writes nothing.

## Related samples

*   [Evaluation samples](../../../../contributing/samples/evaluation): six
    variations on evaluating one shared agent, with a README comparing them.
*   [Shared home-automation agent](../../../../contributing/samples/evaluation/home_automation_agent/agent.py):
    the deterministic agent every evaluation sample scores.

## Related guides

*   [AgentEvaluator](../agent_evaluator/index.md) is the one-call, test-shaped
    front door to this service.
*   [EvalConfig and the eval config file](../eval_config/index.md) covers where
    `eval_metrics` normally comes from.
*   [Evaluator](../evaluator/index.md) covers writing the metrics this service
    runs.
