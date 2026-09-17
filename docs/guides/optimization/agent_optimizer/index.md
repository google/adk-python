# AgentOptimizer and Sampler

`google.adk.optimization` rewrites an agent's instruction automatically, scoring
candidate prompts against an evaluation set and keeping whichever one scores
better. You supply a `Sampler` that knows how to evaluate your agent, and ADK
supplies the optimizers that drive the search.

## Introduction

An optimizer proposes a new instruction, an evaluator scores it over a fixed set
of examples, and the higher score wins. That replaces changing a sentence by
hand, running the agent over a handful of cases, and deciding it looks better.

The package splits into two halves that meet at one interface.

*   **`AgentOptimizer`** is the search. It decides which prompts to try, in what
    order, and when to stop. Two implementations ship:
    `SimplePromptOptimizer`, a hill-climb over one prompt at a time, and
    `GEPARootAgentOptimizer`, which wraps the
    [GEPA](https://arxiv.org/abs/2507.19457) algorithm and returns a Pareto
    front of prompts rather than one winner.
*   **`Sampler`** is the scoring. It answers "which examples exist" and "how
    well did this candidate agent do on them". `LocalEvalSampler` implements it
    on top of ADK's own eval sets, and you can implement it against whatever
    scoring you already trust.

Nothing here runs at request time. Optimization is an offline batch job you run
against an eval set, and its output is an `Agent` object carrying a better
instruction, which you then copy into your source.

`google/adk/optimization/__init__.py` re-exports nothing, so every import is by
module path:

```python
from google.adk.optimization.agent_optimizer import AgentOptimizer
from google.adk.optimization.data_types import AgentWithScores
from google.adk.optimization.data_types import OptimizerResult
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizer
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizerConfig
```

## Get started

Implement `Sampler` over whatever your scoring already is, then hand it to an
optimizer. This sampler pretends to score; a real one runs the agent.

```python
from google.adk.agents import Agent


class MySampler(Sampler[UnstructuredSamplingResult]):

  def get_train_example_ids(self) -> list[str]:
    return ["case-1", "case-2", "case-3"]

  def get_validation_example_ids(self) -> list[str]:
    return ["holdout-1", "holdout-2"]

  async def sample_and_score(
      self,
      candidate,
      example_set=Sampler.VALIDATION_SET,
      batch=None,
      capture_full_eval_data=False,
  ) -> UnstructuredSamplingResult:
    if batch is None:
      batch = (
          self.get_train_example_ids()
          if example_set == Sampler.TRAIN_SET
          else self.get_validation_example_ids()
      )
    scores = {example_id: await my_score(candidate, example_id) for example_id in batch}
    return UnstructuredSamplingResult(scores=scores)


agent = Agent(
    name="support_agent",
    instruction="Help the user with their order.",
    tools=[check_order_status, issue_refund],
)

optimizer = SimplePromptOptimizer(
    SimplePromptOptimizerConfig(num_iterations=5, batch_size=3)
)
result = await optimizer.optimize(agent, MySampler())

best = result.optimized_agents[0]
print(best.overall_score)
print(best.optimized_agent.instruction)
```

`optimize` never mutates the agent you pass in. Every candidate is built with
`clone(update={"instruction": ...})`, so the object you constructed still
carries your original instruction when the run finishes.

Scores are floats where higher is better. That is the only contract, so the
range is yours to choose, and the optimizer only ever compares two of your own
numbers to each other.

## How it works

The interface between optimizer and sampler is small enough to state in a
paragraph, and the two shipped optimizers differ mainly in how much they ask of
it. `SimplePromptOptimizer` needs only a score; `GEPARootAgentOptimizer` needs to
see why a candidate failed.

### The contract between the two halves

An optimizer calls the sampler and nothing else. It asks for the example IDs
once, then repeatedly calls `sample_and_score(candidate, example_set, batch)`
and reads `result.scores`, a `dict` keyed by example ID. When an optimizer needs
to see *why* a candidate failed, it passes `capture_full_eval_data=True`, and
the sampler is expected to populate `UnstructuredSamplingResult.data`. That is a
second dict, keyed by the same example IDs, holding whatever JSON-serializable
material helps a model reason about the failure: the input, the response, the
tool calls, the metric verdicts. `SimplePromptOptimizer` never asks for it.
`GEPARootAgentOptimizer` depends on it, and reflects over exactly what you put
there.

`Sampler.TRAIN_SET` and `Sampler.VALIDATION_SET` are class constants holding the
strings `"train"` and `"validation"`. They are the only two values
`example_set` takes.

### What `SimplePromptOptimizer` does

The loop is a hill climb with no memory:

1.  Score the initial agent on a random batch of training examples. That is the
    score to beat.
2.  For each iteration, ask the optimizer model to rewrite the current best
    instruction, given only that instruction and its score. Clone the best agent
    with the rewrite, score the clone on a fresh random training batch, and keep
    it only if it scored higher.
3.  After the last iteration, score the surviving agent once over the **whole**
    validation set. That number becomes `overall_score`.

So a run with `num_iterations=n` makes `n + 2` calls to `sample_and_score`: one
baseline, `n` candidates, one final validation. Selection happens entirely on
training scores; validation is measured once, at the end, and never influences
which prompt is chosen.

Two consequences shape how you read a result. Each comparison
uses a *different* random batch of size `batch_size`, so a candidate can win on
batch noise rather than merit. The optimizer model also sees only the previous
prompt and a single number, never a failing case, so it is guessing at what to
improve.

`SimplePromptOptimizer` resolves its optimizer model in `__init__`, not in
`optimize`, so a bad model name fails as soon as you construct it.

### What `GEPARootAgentOptimizer` does

GEPA reflects on failures rather than guessing. It runs candidates, feeds the
captured trajectories of low-scoring runs to a reflection model, and uses that
model's diagnosis to propose the next prompt. It keeps a Pareto front rather
than a single champion, so `optimized_agents` comes back with several agents
that are each best at something, and `GEPARootAgentOptimizerResult.gepa_result`
carries the raw algorithm output as a dict.

Two things it optimizes that the simple optimizer does not: the root agent's
instruction, and the `instructions` text of every `Skill` reachable through a
`SkillToolset` in the agent's tools. Each
becomes a separately evolved component. Sub-agent prompts are **not** optimized;
if `initial_agent.sub_agents` is non-empty the optimizer logs a warning and
proceeds on the root only.

The module list also contains `GEPARootAgentPromptOptimizer`, which is the
earlier, narrower version of the same thing: it evolves the root instruction
only and ignores skills entirely. Its configuration is otherwise identical apart
from two defaults, `gemini-2.5-flash` and a thinking *budget* rather than a
thinking *level*. Prefer `GEPARootAgentOptimizer`, which does everything it does.

`GEPARootAgentOptimizer` is decorated `@experimental`, so constructing it emits
a `UserWarning` reading `[EXPERIMENTAL] GEPARootAgentOptimizer: ...`. The
algorithm itself lives in the third-party `gepa` package, which ADK imports
lazily inside `optimize`. Without it the call raises ``ImportError: Eval module
is not installed, please install via `pip install "google-adk[eval]"`.``

It also requires `initial_agent.instruction` to be a plain string. An agent
whose instruction is a callable provider raises `ValueError` before any
evaluation runs, because a request-scoped provider cannot be resolved without a
live invocation to resolve it against. That check sits behind the `gepa` import,
so install the extra first or you will see the `ImportError` instead.

### `LocalEvalSampler`

`LocalEvalSampler` is the ready-made bridge from the optimizers to ADK's
[evaluation](../../evaluation/eval_service/index.md) machinery. Give it an
`EvalConfig`, an app name, and the ID of an eval set, and it runs each candidate
agent through `LocalEvalService`, doing inference and then metrics, for every
eval case you name.

It scores by status, not by metric value. A case that passes gives the optimizer
`1.0` and every other outcome gives it `0.0`, so if one case scored 0.94 against
a 0.95 threshold and another errored out, the optimizer cannot tell them apart.

Importing `local_eval_sampler` pulls in the eval stack, which needs the `eval`
extra, installed with `pip install "google-adk[eval]"`. Without it the import
fails with `ModuleNotFoundError: No module named 'vertexai'` rather than ADK's
own install hint.

## Configuration options

Each optimizer takes its own config object, and the two differ in how they bound
a run: the simple optimizer counts iterations, GEPA counts scored evaluations.

### `SimplePromptOptimizerConfig`

The four settings cover which model rewrites the prompt, how that model
generates, how many rewrites to try, and how many examples each rewrite is
judged on.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `optimizer_model` | `str` | `"gemini-2.5-flash"` | Model that rewrites the prompt. Not the agent's own model. |
| `model_configuration` | `GenerateContentConfig` | thinking on, budget 10240 | Generation config for the optimizer model. |
| `num_iterations` | `int` | `10` | Candidate prompts to try. |
| `batch_size` | `int` | `5` | Training examples scored per candidate. |

`optimizer_model` names the model doing the rewriting, and it is independent of
whatever model the agent under optimization runs on. The default enables
thinking with a budget of 10240 tokens, because the rewrite is a reasoning task.

`batch_size` trades cost against signal. Every candidate costs `batch_size`
agent runs, so a run costs roughly `(num_iterations + 1) * batch_size` training
runs plus one full validation pass. Small batches make the comparison noisy;
that noise is the main reason a run can end with a prompt that is not actually
better.

`optimize` clamps `batch_size` down to the number of training examples when it
exceeds them, and it does so by **writing to the config object you passed in**.
Read `config.batch_size` after a run and you may find a different number than
you set.

### `GEPARootAgentOptimizerConfig`

GEPA bounds a run by total evaluation budget rather than by iteration count, so
the settings are about how much you are willing to spend and how much the
reflection model sees each time.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `optimizer_model` | `str` | `"gemini-3.5-flash"` | Model used for reflection and proposing new prompts. |
| `model_configuration` | `GenerateContentConfig` | thinking level `HIGH` | Generation config for the reflection model. |
| `max_metric_calls` | `int` | `100` | Total evaluation budget for the whole run. |
| `reflection_minibatch_size` | `int` | `3` | Examples shown to the reflection model per step. |
| `run_dir` | `str \| None` | `None` | Directory for checkpoints. Set it to make the run resumable. |

`max_metric_calls` is the one knob that decides what a run costs. It caps the
number of scored evaluations across the entire search, so raising it buys more
exploration linearly. The GEPA sample suggests starting at the default of 100
and raising it past 500 for a serious run.

`run_dir` is worth setting on any run long enough to be interrupted. With it,
GEPA writes intermediate state and picks up from the last checkpoint; without
it, an interrupted run starts over.

## Advanced applications

Almost all of the work in optimization is the sampler, so the questions that
matter are where your scoring comes from and whether the number it reports can
be trusted.

### Optimize against an existing eval set

Eval cases and metrics you already have are scoring code you do not need to
write twice. Use `LocalEvalSampler` instead of your own `Sampler`, and point it
at a train set and, separately, a validation set. Omitting `validation_eval_set`
makes it reuse the training set for validation, which reports a score the
optimizer already fitted to and therefore tells you nothing about a new case:

```python
sampler = LocalEvalSampler(
    LocalEvalSamplerConfig(
        eval_config=eval_config,
        app_name="my_app",
        train_eval_set="train_set",
        validation_eval_set="holdout_set",
    ),
    eval_sets_manager=eval_sets_manager,
)
result = await GEPARootAgentOptimizer(
    GEPARootAgentOptimizerConfig(max_metric_calls=200, run_dir="/tmp/gepa_run")
).optimize(agent, sampler)
```

The `EvalConfig` is the same object the eval config file produces, so the
metrics you already tuned there decide what "better" means here. See
[EvalConfig](../../evaluation/eval_config/index.md).

### Keep training and validation genuinely separate

An optimizer that selects on the same examples it reports on always looks
successful, because the score it reports is the score it was tuned against.
Return disjoint ID lists from `get_train_example_ids` and
`get_validation_example_ids`. `GEPARootAgentOptimizer` checks for you and logs a
warning when the two sets intersect, because a shared UID meaning two different
examples in the two sets silently aliases them. The other optimizers do not
check, so on those the separation is yours to enforce.

### Write a sampler over your own scoring

A quality signal that is a human rating, a production metric, or a rubric-based
rater does not fit an ADK eval set, and it does not have to. Implement the three
`Sampler` methods and return your numbers in `scores`. If you intend to use
GEPA, also honor `capture_full_eval_data=True` by filling `data` with the
trajectory material the reflection model should read, since GEPA raises rather
than proceeding when trajectories are missing. Everything else, batching and
caching and concurrency included, is yours to decide, because the optimizer only
awaits the coroutine.

## Limitations

*   **Nothing is written back.** `optimize` returns an in-memory `Agent`. Copying
    the winning instruction into your source, and re-running your normal tests
    against it, is manual.
*   **`SimplePromptOptimizer` has a customer-support prompt baked in.** Its
    optimizer template states that "the agent needs to solve customer support
    tasks by using tools correctly and following policies", and there is no
    option to replace it. For any other domain that sentence is misdirection
    sent to the rewriting model on every iteration.
*   **`SimplePromptOptimizer` returns exactly one agent** even though
    `OptimizerResult.optimized_agents` is a list documented as a Pareto front.
    Only GEPA populates it with more than one.
*   **Selection is on training scores only.** A `SimplePromptOptimizer` run
    reports a validation score it never optimized against, and can report a
    validation score *worse* than the initial agent's while still returning the
    rewritten prompt.
*   **Cost is real and unbounded by the package.** Every score is a full agent
    run against a live model. `max_metric_calls` bounds GEPA; nothing bounds a
    custom sampler.
*   **`LocalEvalSampler` throws away metric resolution.** Pass or fail only, so
    the optimizer cannot see a candidate that improved from 0.4 to 0.9 without
    crossing the threshold.
*   **No sample in this repository uses these classes.** The GEPA sample linked
    below calls the third-party `gepa` package directly instead.
*   **Importing `data_types` emits Pydantic deprecation warnings.** The models
    pass a `required=True` keyword that Pydantic v2 no longer recognizes. The
    fields are required regardless; the warning is noise.

## Related samples

*   [GEPA integration](../../../../contributing/samples/integrations/gepa/README.md)
    optimizes an ADK agent's prompt with GEPA on the Tau-bench retail
    benchmark, including an LLM rater for when no reward signal exists. It
    predates `google.adk.optimization` and writes its own GEPA adapter against
    the `gepa` package, so read it for the shape of the evaluation work a
    `Sampler` encapsulates rather than as an API example.

## Related guides

*   [EvalConfig and the eval config file](../../evaluation/eval_config/index.md)
    covers the metrics that decide what `LocalEvalSampler` calls a pass.
*   [Evaluator](../../evaluation/evaluator/index.md) covers writing the metric a
    sampler scores against.
*   `SkillToolset` covers the skill
    instructions `GEPARootAgentOptimizer` evolves alongside the root prompt.
