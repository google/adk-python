# EvalConfig and the eval config file

`EvalConfig` is the schema of the small JSON file that says which metrics score
an evaluation run and how strict each one is. Write one as soon as the built-in
scoring stops matching what you care about, whether that is a strict
tool-trajectory match failing on a harmless extra lookup, or a metric of your
own that ADK knows nothing about.

## Introduction

Everything else in ADK evaluation is fixed by the eval data, which holds the
user turns, the expected answers and the expected tool calls. The eval config
is the one part you tune. It answers "score this run how, and how close is
close enough", and because the answer differs per metric, the shape of each
entry differs too. A tool-trajectory entry can say how strictly the order must
match; a judge-based entry can name the judge model and how many times to
sample it; a rubric-based entry must carry the rubrics themselves.

The file is read by [`AgentEvaluator`](../agent_evaluator/index.md) when you
evaluate from a test, and by `adk eval` when you evaluate from the command line.
Both parse the JSON into an `EvalConfig` and then turn it into a list of
`EvalMetric` objects, one per entry in `criteria`.

## Where the file is found

There are two names, and which one applies depends on how you start the run.

`test_config.json` is the name that gets discovered automatically. Put it in the
same folder as your eval data and both entry points find it:
`AgentEvaluator.find_config_for_test_file` looks for exactly that filename beside
each `*.test.json` it is about to run, and `adk eval` falls back to
`<eval-data-dir>/test_config.json` when `--config_file_path` is not given and
the eval data came from a single file.

Any other path works only if you pass it explicitly, which is what
`adk eval ... --config_file_path path/to/eval_config.json` does. That is why the
samples in this repository call their files `eval_config.json`: they are all
driven from the command line.

**If no file is found, evaluation does not fail. It falls back to a built-in
default of `tool_trajectory_avg_score` at `1.0` and `response_match_score` at
`0.8`, with only an informational log line to say so.** Those are strict
numbers, and a first run that fails on them is often failing on the default
rather than on your agent.

## Get started

The smallest useful file scores two things: whether the agent called the right
tools, and whether its final answer resembled the recorded one.

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.6
  }
}
```

A bare number is shorthand for a threshold. Write an object instead when the
metric has options you want to set:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": {
      "threshold": 1.0,
      "match_type": "IN_ORDER"
    },
    "final_response_match_v2": {
      "threshold": 0.8,
      "judge_model_options": {
        "num_samples": 5
      }
    }
  }
}
```

`threshold` is required in the object form. The metric passes when the mean of
its per-invocation scores, across every run, is greater than or equal to it,
unless `multi_run_config` combines the runs another way.

Keys may be written in `snake_case` or `camelCase`, so `match_type` and
`matchType` both parse. That holds for every key in the file, at every level.

## The metrics, and the criterion each one accepts

Every metric name is one of the thirteen `PrebuiltMetrics` values, or a name you
declared yourself under `custom_metrics`. The **criterion type** column says
which set of keys that metric's evaluator accepts in the object form.

Most of them group into four families, and the family a metric belongs to
decides what it costs and what its score is worth.

*   **The deterministic pair**, `tool_trajectory_avg_score` and
    `response_match_score`, compares against the recording with no model
    involved, so those two are free, repeatable, and blind to meaning.
*   **The judge-based metrics**, among them `final_response_match_v2`,
    `safety_v1` and `hallucinations_v1`, ask a model for a verdict, so they read
    meaning and cost one model call per sample per invocation.
*   **The rubric-based metrics** score against sentences you write yourself,
    which is the route to a rule no general metric knows about, such as "the
    response names the device it changed".
*   **The multi-turn metrics** score a whole conversation rather than a single
    turn, which is what you need when the failure you are hunting only appears
    over several exchanges.

Scoring every metric makes a suite slow and its output hard to read. Start with
the deterministic pair, and add one judge-based or rubric-based metric for the
specific quality a deterministic score cannot see.

| Metric | Criterion type | Range | What it scores |
| :--- | :--- | :--- | :--- |
| `tool_trajectory_avg_score` | `ToolTrajectoryCriterion` | 0 to 1 | Exact match of tool name and arguments against the expected trajectory. |
| `response_match_score` | `BaseCriterion` | 0 to 1 | ROUGE-1 word overlap between the final response and the expected one. |
| `response_evaluation_score` | `BaseCriterion` | **1 to 5** | How coherent the response was. |
| `final_response_match_v2` | `LlmAsAJudgeCriterion` | 0 to 1 | Whether the response matches the expected one, judged by a model. |
| `safety_v1` | `BaseCriterion` | 0 to 1 | Harmlessness of the response. |
| `hallucinations_v1` | `HallucinationsCriterion` | 0 to 1 | Whether the response contains unsupported or contradictory claims. |
| `rubric_based_final_response_quality_v1` | `RubricsBasedCriterion` | 0 to 1 | The response against rubrics you write. |
| `rubric_based_tool_use_quality_v1` | `RubricsBasedCriterion` | 0 to 1 | The tool usage against rubrics you write. |
| `rubric_based_multi_turn_trajectory_quality_v1` | `RubricsBasedCriterion` | 0 to 1 | A whole conversation's trajectory against rubrics you write. |
| `multi_turn_task_success_v1` | `BaseCriterion` | 0 to 1 | Whether the conversation achieved its goal. |
| `multi_turn_trajectory_quality_v1` | `BaseCriterion` | 0 to 1 | How the conversation got there, not only whether it did. |
| `multi_turn_tool_use_quality_v1` | `BaseCriterion` | 0 to 1 | The function calls made across a conversation, without a reference. |
| `per_turn_user_simulator_quality_v1` | `LlmBackedUserSimulatorCriterion` | 0 to 1 | Whether the simulated user's messages followed the scenario. |

`response_evaluation_score` is the one whose range is not 0 to 1. A threshold of
`0.8` on it, copied from a neighboring entry, is a threshold every possible
score clears.

### Keys by criterion type

**`BaseCriterion`** is the base that every other criterion extends, so these two
keys are available everywhere.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `threshold` | `float` | required | Minimum mean score for the metric to pass. |
| `include_intermediate_responses_in_final` | `bool` | `false` | Concatenate text the agent emitted before its tool calls onto the final response before scoring. |

`include_intermediate_responses_in_final` matters for an agent that narrates:
"Let me check that for you" before the tool call, the real answer after it. By
default only the final response text reaches the judge. Turn this on and the
text from every intermediate event is concatenated onto it first.

**`ToolTrajectoryCriterion`** adds `match_type`, which takes one of three
values.

*   **`EXACT`**, the default, demands the actual tool calls be exactly the
    expected ones.
*   **`IN_ORDER`** requires every expected call to appear, in the expected
    order, but tolerates extra calls in between.
*   **`ANY_ORDER`** drops the ordering requirement as well.

Reach for `IN_ORDER` when your agent legitimately makes an extra lookup that you
do not want to pin down, and for `ANY_ORDER` when the calls are independent of
each other, such as three separate reads whose results are combined at the end.
Each step away from `EXACT` buys tolerance for a variation you consider harmless
and gives up the ability to catch that variation when it is not.

**`LlmAsAJudgeCriterion`** adds `judge_model_options`, an object with four keys
of its own:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `judge_model` | `str` | `"gemini-2.5-flash"` | Model that grades the response. |
| `num_samples` | `int` | `5` | How many times the judge is asked about each invocation before the answers are aggregated. |
| `parallelism_limit` | `int` | `1` | Maximum concurrent judge calls. |
| `judge_model_config` | object | `null` | A `GenerateContentConfig` for the judge. Accepted from JSON, but left out of the generated JSON schema, so no tooling will suggest it. |

`num_samples` defaults to 5 because a single judge call is unreliable; the
documented experience is that five samples aggregate into a stable score. It is
also the direct multiplier on what a judged metric costs, so lowering it is the
first lever when an eval suite is too expensive. `parallelism_limit` trades
quota pressure for wall-clock time and does not change the score.

**`RubricsBasedCriterion`** adds `judge_model_options` and a `rubrics` list.
Each rubric carries a `rubric_id` and a `rubric_content` object whose
`text_property` states, in one plain sentence, the thing the judge should check
for. Metrics that do not use rubrics ignore the key if you set it anyway.

`rubrics` is required in practice but not by the model, which defaults it to an
empty list. The evaluator merges the criterion's rubrics with any carried by the
eval case itself, and only raises `ValueError: Rubrics are required.` if both
are empty. That happens during the run rather than when the config is read, so
a config that omits `rubrics` parses cleanly and fails later.

A rubric-based entry therefore looks like this, with each check written as one
sentence the judge can answer yes or no to:

```json
{
  "criteria": {
    "rubric_based_final_response_quality_v1": {
      "threshold": 0.8,
      "rubrics": [
        {
          "rubric_id": "reports_device_state",
          "rubric_content": {
            "text_property": "The response names the device and states whether it is on or off."
          }
        },
        {
          "rubric_id": "concise",
          "rubric_content": {
            "text_property": "The response is concise and free of filler."
          }
        }
      ]
    }
  }
}
```

**`HallucinationsCriterion`** adds `judge_model_options` and
`evaluate_intermediate_nl_responses`, which defaults to `false`. Turn it on and
the agent's running commentary is checked for invented facts too, not only its
final answer.

**`LlmBackedUserSimulatorCriterion`** extends `LlmAsAJudgeCriterion` with
`stop_signal`, default `"</finished>"`. It should match the stop signal the user
simulator itself uses.

## Custom metrics

`custom_metrics` maps a metric name to the Python function that computes it. The
name must also appear in `criteria` with its threshold, because `criteria`
decides what runs and `custom_metrics` decides how the name is resolved. A
custom metric therefore takes two entries, one in each place:

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

`code_config.name` is a dotted path that is split at the last dot and imported,
so the module part must be importable from wherever the eval runs. `code_config`
takes no other keys and rejects any you add. Writing the function itself is
covered in the [`Evaluator`](../evaluator/index.md) guide.

`description` is a one-liner that becomes the metric's description. For finer
control, `metric_info` replaces it with a full `MetricInfo`, whose main use is
declaring a value range other than the default 0 to 1:

```json
"custom_metrics": {
  "my_metric": {
    "code_config": {"name": "my_pkg.metrics.my_metric"},
    "metric_info": {
      "metric_name": "my_metric",
      "description": "My metric.",
      "metric_value_info": {"interval": {"min_value": -10.0, "max_value": 10.0}}
    }
  }
}
```

Two parts of that shape catch people out. `min_value` and `max_value`
sit inside `metric_value_info.interval`, not at the top of `metric_info`, and
because `MetricInfo` rejects keys it does not recognize, unlike a criterion,
getting that wrong fails loudly. The other is that the key really is
`metric_info`, or `metricInfo`. A `metric` key is not a field, so it is dropped
and the metric silently gets default info instead.

## The other top-level keys

`EvalConfig` accepts five keys in all. Two of them, `criteria` and
`custom_metrics`, are covered above. The remaining three are
`user_simulator_config`, `live_model_config` and `multi_run_config`.

`user_simulator_config` replaces the recorded user turns with a model that plays
the user against a scenario. Which simulator you get is chosen by a `type`
field, whose two values are `llm_backed` and the audio variant `llm_audio`.
Leave `type` out and you get `llm_backed`, for backward compatibility, with an
informational log line. Both variants take the keys below, and `llm_audio` adds
audio settings of its own on top of them.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `type` | `str` | `"llm_backed"` | Which simulator implementation to use. |
| `model` | `str` | `"gemini-2.5-flash"` | Model that plays the user. |
| `model_configuration` | `GenerateContentConfig` | thinking on, budget 10240 | Generation config for the simulator's model. The default asks for thoughts, so overriding it with a bare `{}` turns thinking off. |
| `max_allowed_invocations` | `int` | `20` | Hard cap on conversation length. `-1` removes the cap, which is how a run-off conversation becomes a bill. |
| `custom_instructions` | `str` | `null` | Replacement instructions for the simulator. It must contain the Jinja placeholders `{{ stop_signal }}`, `{{ conversation_plan }}` and `{{ conversation_history }}`, plus `{{ persona }}` if the scenario sets one; a validator rejects it otherwise. |
| `include_function_calls` | `bool` | `false` | Whether the simulated user sees the agent's tool calls. |

`live_model_config` switches inference to bidirectional streaming mode, which is
required for Live API models. Its only key is `timeout_seconds`, default `300`,
the time to wait for a model turn to complete.

`multi_run_config` runs each eval case more than once. An agent can answer the
same case differently from one run to the next, so a single run can pass or
fail by chance. `num_runs` says how many times:

```json
{
  "criteria": {
    "response_match_score": 0.8
  },
  "multi_run_config": {
    "num_runs": 3
  }
}
```

With `num_runs` above 1, three things change.

*   **Each run starts from a fresh session.** A `session_id` pinned in the eval
    case's `session_input` is ignored, with a warning in the log, so every run
    starts from the case's initial state instead of from the session an earlier
    run left behind.
*   **Every run is kept.** Each run produces its own `EvalCaseResult`, and
    `adk eval` saves all of them to the eval history. Each result carries the
    0-based `repetition_index` of its run, which is always 0 when a case runs
    once.
*   **The case is judged on all of its runs.** By default each metric's
    per-invocation scores from every run are averaged together, and the metric
    passes when that mean meets its threshold. A case can therefore pass with a
    failing run among its runs, if the others scored high enough to carry the
    mean. A run that crashed before it could be scored is left out, and by
    default fails the case outright; see [Crashed runs](#crashed-runs).

`adk eval --num_runs N` overrides `num_runs` for one command, and if neither is
set each case runs once. With `--print_detailed_results`, each case's verdict
across its runs is printed first, followed by the results of each of its runs.
`AgentEvaluator` does not read `multi_run_config`; it takes the number of runs
as its own `num_runs` argument, which defaults to `2`.

Unlike a criterion, `multi_run_config` rejects keys it does not recognize, so a
misspelled `num_run` fails when the file is read instead of silently running
each case once. The top-level key itself gets no such check: misspell
`multi_run_config` and the whole entry is dropped.

### How the runs are combined

Three more keys of `multi_run_config` decide how the runs make the verdict.
Each is optional, and they can be combined.

| Key | Judges | Value |
| :--- | :--- | :--- |
| `cross_run_scoring` | Every metric not named in `cross_run_scoring_by_metric`. | A method name, or a rule object. Default `"mean_over_invocations"`. |
| `cross_run_scoring_by_metric` | The metrics it names. Each must be in `criteria`. | A map from metric name to a method name or a rule object. |
| `cross_run_criteria` | The eval case as a whole. | A list of rule objects. |

A **method name** combines scores, and the result is compared with the
metric's threshold in `criteria`:

*   `mean_over_invocations`, the default, averages every invocation of every
    run. A run with no per-invocation score, as for a metric scored only for
    the whole conversation, counts once, with its overall score.
*   `mean_of_runs` averages the overall scores of the runs, so each run weighs
    the same however many invocations it has.
*   `min_over_runs` takes the worst run's score: the lowest, or the highest for
    a metric where lower is better. The metric therefore passes only if every
    run met its threshold.

As in `AgentEvaluator`, the comparison goes the way the metric's evaluator
judged its scores: if each score it passed is below the threshold and each one
it failed is above, lower is better, so the result passes at or below the
threshold and the output adds `lower is better`; otherwise it passes at or
above the threshold.

A **rule object**, `{"method": ..., "threshold": ...}`, counts the runs that
passed and compares the result with its own `threshold`, from 0 to 1. Each run
is first judged on its own, as a single run would be. The one such method so
far is `pass_rate`: the runs that passed, divided by the runs that passed or
failed. pass@k and pass^k are planned as further methods of this kind.

The case passes when its metrics pass, combined as for a single run, and every
cross-run criterion passes, unless more of its runs crashed than
`max_crashed_runs` allows.

Take one case run three times, scored with `tool_trajectory_avg_score` at a
threshold of 1.0 and `response_match_score` at 0.6:

| Run | `tool_trajectory_avg_score` | `response_match_score` | Whole run |
| :--- | :--- | :--- | :--- |
| 1 | 1.0, passed | 0.875, passed | passed |
| 2 | 1.0, passed | 0.182, failed | failed |
| 3 | 1.0, passed | 0.875, passed | passed |

By default the case passes: `response_match_score` averages 0.644. Each way of
combining the runs can turn that around.

*   **By score**, with `{"response_match_score": "min_over_runs"}` in
    `cross_run_scoring_by_metric`: the lowest run, 0.182, is below 0.6, so the
    metric fails, and with it the case.
*   **By how often a metric passed**, with
    `{"response_match_score": {"method": "pass_rate", "threshold": 0.9}}` in
    `cross_run_scoring_by_metric`: the metric passed in 2 of 3 runs, and 0.67 is
    short of 0.9, so the metric fails, and with it the case.
*   **By how often the whole case passed**, with
    `[{"method": "pass_rate", "threshold": 0.9}]` as `cross_run_criteria`:
    2 of 3 runs passed as a whole, so the criterion fails the case although
    every metric passes on its mean.

A rule that cannot produce a value, because no run produced a score or no run
passed or failed, is `NOT_EVALUATED`. The output says why, and the rule is
skipped when the verdict is decided, as a metric that was not evaluated is. For
`cross_run_criteria`, a run in which any metric was not evaluated counts as
neither passed nor failed, since the missing result could have been a failure.

The file fails to load when a method name is unknown, when `pass_rate` is given
as a bare name without a threshold, when a method that combines scores is given
as a rule object or in `cross_run_criteria`, when a key of
`cross_run_scoring_by_metric` is not in `criteria`, when a `threshold` is
outside 0 to 1, when `max_crashed_runs` is negative, or when a rule object has
a key it does not recognize.

A rule object judges a case run once as well, so with one configured,
`adk eval --print_detailed_results` prints every case the way it prints a case
run several times: each metric with its method and value, then each cross-run
criterion, then the runs. `AgentEvaluator`, which pytest suites call, and
`adk web` ignore these keys for now.

### Crashed runs

A run crashes when it fails without producing any metric results, which is
what happens when inference raises. Such a run has no verdict, so it is left
out of every count of passing runs, by a metric's rule and by
`cross_run_criteria` alike. `max_crashed_runs`, default `0`, says how many
crashed runs a case tolerates: the case fails if more of its runs crashed than
that. A case also needs at least one run that did not crash to pass, so a case
whose runs all crashed never passes.

```json
{
  "criteria": {
    "response_match_score": 0.8
  },
  "multi_run_config": {
    "num_runs": 4,
    "max_crashed_runs": 1,
    "cross_run_criteria": [{"method": "pass_rate", "threshold": 0.75}]
  }
}
```

If 3 runs pass and 1 crashes, `pass_rate` is 3 of 3, which is 1.0, and the case
passes. `--print_detailed_results` counts the runs as
`3 of 4 runs passed, 1 crashed`, in the case's header and in the note of every
rule that counts passing runs, and adds
`1 of 4 runs failed without producing any metric results (up to 1 allowed).`
With the default `max_crashed_runs` of 0 the same runs fail the case, and so
would a second crash here, since 2 is more than the 1 allowed. A case run once
without a rule object keeps that run's own verdict, so a crash fails it
whatever `max_crashed_runs` is. `adk eval` does not retry a crashed run.

### Custom methods

When no built-in method fits, write one in Python and declare it under
`custom_methods`. Its key is the name the rules use and the name the output
shows:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.6
  },
  "multi_run_config": {
    "num_runs": 5,
    "custom_methods": {
      "passed_at_least_twice": {
        "code_config": {"name": "my_project.eval.methods.passed_at_least_twice"},
        "uses_scores": false
      },
      "median_of_runs": {
        "code_config": {"name": "my_project.eval.methods.median_of_runs"},
        "uses_scores": true
      }
    },
    "cross_run_scoring": "median_of_runs",
    "cross_run_criteria": [
      {"method": "passed_at_least_twice", "threshold": 1.0}
    ]
  }
}
```

A method has the signature of a built-in one: it takes one `RunResult` per
run, for one metric or for the whole case, and returns a number, or `None`
when the runs allow no value.

```python
# my_project/eval/methods.py
from collections.abc import Sequence
import statistics

from google.adk.evaluation import RunResult
from google.adk.evaluation.evaluator import EvalStatus


def median_of_runs(runs: Sequence[RunResult]) -> float | None:
  scores = [run.score for run in runs if run.score is not None]
  return statistics.median(scores) if scores else None


def passed_at_least_twice(runs: Sequence[RunResult]) -> float | None:
  passed = sum(1 for run in runs if run.status == EvalStatus.PASSED)
  return 1.0 if passed >= 2 else 0.0
```

`uses_scores` is required, and says which kind of method it is:

*   **`true`**, a method that combines scores: it reads `score`, the run's
    overall score, or `invocation_scores`. It is written as a method name in
    `cross_run_scoring` or `cross_run_scoring_by_metric`, and its value is
    compared with the metric's threshold in `criteria`. It receives the raw
    scores, so a method that picks a run, such as the worst, must handle the
    metric's direction itself.
*   **`false`**, a method that counts passing runs: it reads `status`. It is
    written as a rule object with a threshold of its own, and can also be used
    in `cross_run_criteria`, where each `RunResult` is a whole run, with no
    score, and a run that crashed is `NOT_EVALUATED`.

The method only computes the number; the threshold and the verdict are applied
as for a built-in method. A method that returns `None` leaves its rule
`NOT_EVALUATED`, with a note, and the rule is skipped when the verdict is
decided. Returning anything else that is not a number, even a `bool`, stops the
eval with a `TypeError` naming the method, and an exception the method raises
is not caught. As with a built-in method, a case run once keeps the run's own
score instead of being re-scored by a method that combines scores.

The file fails to load when a custom method has the name of a built-in one,
when a rule names a method that is neither built in nor declared, or when a
method is written in the form of the other kind. Reading the file does not
import the functions: `adk eval` imports them before it runs any eval case, so
a path that does not import, or names something that is not callable, stops
it at once with an error naming the method. As with the rest of
`multi_run_config`, only `adk eval` reads them.

`code_config` takes no arguments, so a method with parameters is made in Python
and referenced by its full name: with
`pass_at_3 = functools.partial(pass_at_k, k=3)` in `my_project/eval/methods.py`,
the config names `my_project.eval.methods.pass_at_3`.

## How validation actually works

Parsing is deliberately permissive. Every object under `criteria` is validated
as a plain `BaseCriterion`, which requires `threshold` and **allows any other
key through as an extra**. `match_type`, `judge_model_options` and `rubrics` are
all extras at this stage. Only later, when the run resolves a metric name to its
evaluator, does that evaluator re-validate the criterion into its own
`criterion_type`. `TrajectoryEvaluator` re-validates into
`ToolTrajectoryCriterion`, `FinalResponseMatchV2Evaluator` into
`LlmAsAJudgeCriterion`, and so on.

Two consequences follow.

**A misspelled key is silently ignored.** Write `judge_model_option` without the
`s` and the config parses, the evaluator constructs, and the run scores against
the default judge model as though you had said nothing. Nothing warns.

**A wrongly-shaped key fails late, with a message about types rather than
keys.** The re-validation raises this when the evaluator is built, not when the
file is read:

```text
ValueError: `final_response_match_v2` metric expects a criterion of type `<class '...LlmAsAJudgeCriterion'>`.
```

The underlying pydantic error, which is the one that says which key is wrong, is
on that exception's `__cause__`.

Missing `threshold` is the one mistake caught immediately, because the value
then matches neither a bare float nor a `BaseCriterion` and the whole file fails
to parse.

## Limitations

*   **`EvalConfig` is not re-exported.** Import it as
    `from google.adk.evaluation.eval_config import EvalConfig`; the package
    `__init__` exports only `AgentEvaluator` and `RunResult`.
*   **Parsing a config works on a base install; running one does not.**
    `eval_config` and `eval_metrics` import with no extras, so you can validate
    a file anywhere. The metric registry that turns a metric name into an
    evaluator cannot: it pulls in `vertexai` through
    `google-cloud-aiplatform[evaluation]`. Install `google-adk[eval]` before you
    try to run an evaluation.
*   **No validation that a metric name exists.** A typo in a metric name is not
    caught by the config; it surfaces later as a `NotFoundError` from the metric
    registry.
*   **Thresholds are not range-checked** against the metric's declared value
    interval, which is why a 0-to-1 threshold on `response_evaluation_score`
    passes silently.

## Related samples

*   [Basic criteria](../../../../contributing/samples/evaluation/basic_criteria/eval_config.json):
    deterministic tool-trajectory and reference-match scoring.
*   [LLM-judged match](../../../../contributing/samples/evaluation/llm_judge_match/eval_config.json):
    `final_response_match_v2` with judge model options.
*   [Rubric criteria](../../../../contributing/samples/evaluation/rubric_criteria/eval_config.json):
    two rubric-based metrics with their rubrics written out.
*   [User simulation](../../../../contributing/samples/evaluation/user_simulation/eval_config.json):
    `user_simulator_config` alongside hallucination and simulator-quality metrics.
*   [Custom metric](../../../../contributing/samples/evaluation/custom_metric/eval_config.json):
    wiring a metric name to a Python function.
