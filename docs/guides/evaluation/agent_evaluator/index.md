# AgentEvaluator

`AgentEvaluator` measures agent quality from inside a `pytest` suite. It replays
a file of recorded conversations through your agent, scores each answer and each
tool call against criteria you set, and fails the test when a score drops below
its threshold. It is the only name `google.adk.evaluation` exports, and the
supported way to check agent quality from a test.

## Introduction

Agent quality is awkward to assert with an ordinary unit test. The model
rephrases its answer between runs, so `assert response == "..."` fails on a
harmless wording change, and it says nothing at all about whether the agent
called the right tools with the right arguments to get there.

`AgentEvaluator` replaces that assertion with a scored comparison. You record a
conversation once, covering the user turns, the expected final responses, and
the expected tool calls. The evaluator then replays it, runs your real agent
against each user turn, and scores what came back. Tool calls are compared
structurally, while the response text is compared with a ROUGE-1 word-overlap
score that tolerates rephrasing. Each metric has a threshold, and the method
ends in an `assert`, so the whole thing is one `await` inside a test function.

The threshold is what turns a score into a regression test. A single run tells
you a number, and a number on its own does not say whether an agent is good. A
number compared against a threshold you set from a run you trusted says
something you can act on, which is that your latest change moved quality in one
direction or the other. Choosing metrics whose scores mean what you think
they mean is therefore most of the work, and the two defaults are explained
under [Get started](#get-started).

It drives the same components as `adk eval`.
[`LocalEvalService`](../eval_service/index.md) runs the agent and scores the
output, the criteria come from an [`EvalConfig`](../eval_config/index.md), and
each metric name resolves to an [`Evaluator`](../evaluator/index.md) through a
`MetricEvaluatorRegistry`. `AgentEvaluator` is the thin, test-shaped front door
to all of it; reach for the service directly when you want the scores as data
rather than as an assertion.

## Get started

The evaluator needs three things on disk: an agent package it can import, a file
of recorded conversations named `*.test.json`, and a `test_config.json` next to
that file saying what to score.

```
my_agents/
  home_automation/
    __init__.py        # from . import agent
    agent.py           # defines root_agent
tests/
  eval/
    home_automation/
      simple.test.json
      test_config.json
  test_home_automation.py
```

`test_config.json` names each metric and its threshold. These two are where most
suites start, because between them they score the two halves of a turn, what the
agent did and what it said:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.6
  }
}
```

`tool_trajectory_avg_score` scores what the agent did. For each user turn it
compares the tool calls the agent made against the ones the recording expects,
tool name and arguments both, and awards 1.0 for a turn that matches and 0.0 for
a turn that does not. The average in the name is the average over turns, not
partial credit inside one, so a score of 0.5 means half the turns were exactly
right rather than that every turn was half right. That is why `1.0` is the usual
threshold: any lower number is a statement about how many turns you are willing
to let go wrong. Where the metric misleads is on an agent that reaches the same
answer by more than one route, because a harmless extra lookup scores the turn
0.0. The fix for that is `match_type`, covered in the
[eval config guide](../eval_config/index.md), rather than a lower threshold.

`response_match_score` scores what the agent said. It is a ROUGE-1 comparison,
which reduces both texts to word stems, counts the single words they have in
common, and returns the balance of precision and recall. Word order therefore
does not matter, while padding the answer and leaving something out both lower
the score. A rephrased answer still scores well, which is the whole reason to
prefer the metric over an equality assertion. Where it misleads is on
meaning, because "the light is on" and "the light is off" share almost every
word. Read it as a check that the agent talked about the right thing, and use
`final_response_match_v2`, which asks a judge model whether the two answers
agree, when correctness itself is what you want to assert.

The `0.6` above is a starting point, not a recommendation. Run the eval against
an agent you already trust, look at the scores it produces, and set each
threshold a little below the lowest of them, so that the test fails on a
regression rather than on normal variation.

The test itself is one call:

```python
import pytest

from google.adk.evaluation import AgentEvaluator


@pytest.mark.asyncio
async def test_home_automation_agent():
  await AgentEvaluator.evaluate(
      agent_module="my_agents.home_automation",
      eval_dataset_file_path_or_dir="tests/eval/home_automation/simple.test.json",
      num_runs=2,
  )
```

Two details in that call decide whether it works at all.

`agent_module` is an **importable dotted module path**, not a filesystem path.
Name the package, and the loader imports it, looks for a member called `agent`
on it, and reads `root_agent` off that inner module. That is why the
conventional package carries `from . import agent` in its `__init__.py`. Name
the inner module yourself, as in `"my_agents.home_automation.agent"`, and that
works too. If your module exposes an async `get_agent_async()` in place of
`root_agent`, the loader awaits it and takes its first return value.

`eval_dataset_file_path_or_dir` is resolved against the process working
directory, so what you write is relative to wherever you launched `pytest`, not
to the test file. If you point it at a directory rather than a single file,
every file below it ending in `.test.json` is run, each with whatever
`test_config.json` sits beside it.

## How it works

A call to `evaluate` walks through five stages, in this order.

1.  **Collect the eval data.** A directory argument is walked recursively for
    `*.test.json` files; a file argument is used as-is. Each file is parsed as
    an `EvalSet`. A file in the pre-`EvalSet` schema still loads, with a warning
    pointing at `AgentEvaluator.migrate_eval_data_to_new_schema`.
2.  **Find the config.** For every test file, `find_config_for_test_file` looks
    for `test_config.json` in the same folder. If you leave that file out,
    evaluation does not complain: it falls back to a built-in default of
    `tool_trajectory_avg_score` at `1.0` and `response_match_score` at `0.8`.
    Those are strict criteria, and a first run often fails them for reasons that
    have nothing to do with the agent.
3.  **Resolve the agent.** The module is imported and `root_agent` located as
    described above. If the module also exposes an `App` instance named `app`,
    that App is picked up too, so its plugins and context-cache configuration
    take part in the run.
4.  **Run the agent, for real.** Phase one is live inference: `LocalEvalService`
    actually executes your agent against every user turn in the eval set,
    `num_runs` times over. A model credential is required even when every
    criterion you configured is deterministic, because producing the output to
    score is itself a model call. The runs are sequential, so wall-clock time
    scales with `num_runs`.
5.  **Score and assert.** Phase two scores the recorded output. For each metric,
    the per-invocation scores from every run are averaged into one number, and
    the metric passes when that mean is greater than or equal to its threshold.

Every failing metric contributes one line:

```text
response_match_score for my_agents.home_automation Failed. Expected 0.6, but got 0.41.
```

The method finishes with `assert not failures`, so those lines become the pytest
failure message. If a run produced no metric results at all, which happens when
inference itself raised, that is reported separately so a crash cannot pass as
a clean run.

With `print_detailed_results` left at its default of `True`, a failing metric
also prints a grid comparing the expected and actual response and tool calls for
each invocation. That table needs `pandas` and `tabulate`; without them the
print raises `ModuleNotFoundError` with an install hint.

## Configuration options

These are the arguments of `evaluate`. `evaluate_eval_set` takes the same set
apart from `eval_dataset_file_path_or_dir` and `initial_session_file`, which it
replaces with `eval_set` and `eval_config`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `agent_module` | `str` | required | Dotted module path to the agent package. |
| `eval_dataset_file_path_or_dir` | `str` | required | One eval file, or a directory searched recursively for `*.test.json`. |
| `num_runs` | `int` | `2` | How many times the whole eval set is run before scores are averaged. |
| `agent_name` | `str \| None` | `None` | Evaluate a named sub-agent instead of the root agent. |
| `initial_session_file` | `str \| None` | `None` | Initial session state, for pre-`EvalSet` data only. |
| `print_detailed_results` | `bool` | `True` | Print the expected-vs-actual table for failing metrics. |
| `artifact_service` | `BaseArtifactService \| None` | `None` | Artifact service the run reads from. Defaults to an in-memory one. |
| `output_file` | `str \| None` | `None` | Write per-invocation results to this path as CSV. |
| `app_name` | `str \| None` | `None` | App name used when persisting results. |
| `eval_set_results_manager` | `EvalSetResultsManager \| None` | `None` | Persists results as `*.evalset_result.json`. |

`num_runs` exists because a single run of a live model is noisy. Averaging over
two runs is the default; the ADK integration tests use four for cases known to
vary. Raising it is the first thing to try when a test is flaky rather than
wrong, and the cost is linear in model calls.

`agent_name` selects a sub-agent by name from the loaded root agent's tree, so
you can score one specialist in isolation. A name that does not match anything
raises `ValueError`.

`initial_session_file` only applies to data in the older, pre-`EvalSet` schema.
Pass it together with a modern `EvalSet` file and the load fails with a message
telling you the initial session belongs inside the eval set file.

`artifact_service` matters when a case depends on an artifact that has to exist
before the run starts, such as a PDF the agent is meant to summarize. Pre-load
it into a service, pass that service here, and pin the eval case to a session
id through `SessionInput.session_id` so the lookup resolves.

`eval_set_results_manager` persists what a run produced, which is what turns a
CI job into a history you can compare against. It requires `app_name`; passing
the manager without it raises `ValueError` before anything runs.
`LocalEvalSetResultsManager(agents_dir=...)` writes a single
`*.evalset_result.json` per evaluated eval set under
`<agents_dir>/<app_name>/.adk/eval_history/`, holding one `EvalCaseResult` for
each of the `num_runs` runs.

`output_file` is the flat alternative: one CSV row per metric per invocation,
carrying the threshold, the score, the status, the prompt, and both the expected
and actual responses and tool calls. Rows are appended, so several test files
can accumulate into one sheet.

## Advanced applications

Two situations take you off the file-driven path: eval cases that are produced
by code rather than checked in, and eval data written before the `EvalSet`
schema existed.

### Evaluate without files

`evaluate_eval_set` takes an `EvalSet` object and an `EvalConfig` object
directly, which is what you want when the cases are generated rather than
checked in. That covers cases parameterized over a matrix of inputs, and cases
read out of a database.

```python
from google.adk.evaluation import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types


eval_set = EvalSet(
    eval_set_id="generated",
    eval_cases=[
        EvalCase(
            eval_id="turn_off_bedroom_light",
            conversation=[
                Invocation(
                    user_content=types.Content(
                        role="user",
                        parts=[types.Part(text="Turn off the bedroom light.")],
                    ),
                    final_response=types.Content(
                        role="model",
                        parts=[types.Part(text="The bedroom light is off.")],
                    ),
                )
            ],
        )
    ],
)

await AgentEvaluator.evaluate_eval_set(
    agent_module="my_agents.home_automation",
    eval_set=eval_set,
    eval_config=EvalConfig(criteria={"response_match_score": 0.6}),
    num_runs=1,
)
```

`evaluate_eval_set` also accepts a `criteria` dictionary, which is deprecated.
It is worse than merely old: when `criteria` is non-empty it *replaces* whatever
you passed as `eval_config`, so a call that supplies both silently loses the
config. Use `eval_config` alone.

### Migrate old eval data

`AgentEvaluator.migrate_eval_data_to_new_schema(old_file, new_file)` rewrites a
pre-`EvalSet` JSON file into the current schema, taking the criteria from the
`test_config.json` beside the old file. It is a one-shot utility, not something
to call from a test.

## Limitations

*   **It needs the evaluation extra, and the failure tells you almost
    nothing.** Install the extra with `pip install "google-adk[eval]"`. Without
    it, `from google.adk.evaluation import AgentEvaluator` fails with
    `ImportError: cannot import name 'AgentEvaluator' from
    'google.adk.evaluation'` and nothing more, which does not name the
    dependency that is actually missing. To find out which one it is, import
    `google.adk.evaluation.agent_evaluator` directly. On a base install that
    reports `No module named 'vertexai'`, which comes in through
    `google-cloud-aiplatform[evaluation]`.
*   **Evaluation is not offline.** Every run executes the agent, so credentials
    and model quota are required even for purely deterministic criteria.
*   **A failure is an `AssertionError`.** There is no typed exception and no
    result object returned to the caller; to inspect scores programmatically,
    pass an `eval_set_results_manager` or an `output_file`.
*   **Missing config is silent.** No `test_config.json` beside the eval file
    means the strict built-in defaults apply, with only an informational log
    line to say so.

## Related samples

*   [Evaluation samples](../../../../contributing/samples/evaluation): six
    variations on evaluating one shared agent, with a README comparing the
    techniques.
*   [Test file vs. eval set](../../../../contributing/samples/evaluation/test_file_vs_evalset):
    what `.test.json` and `.evalset.json` mean, and why both load the same way.
*   [Shared home-automation agent](../../../../contributing/samples/evaluation/home_automation_agent/agent.py):
    the deterministic agent every evaluation sample scores.
