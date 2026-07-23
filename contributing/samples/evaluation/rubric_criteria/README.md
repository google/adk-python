# Rubric-based evaluation

## Overview

Score response quality and tool-use quality against custom yes/no rubrics judged
by an LLM, with no reference answer required. This sample evaluates the shared
home-automation agent with two reference-free, LLM-judged criteria:

- `rubric_based_final_response_quality_v1`: judges the agent's final response
  against rubrics about *how good the answer is* (does it name the device(s) in
  the requested location and report each one's on/off status? is it concise?).
- `rubric_based_tool_use_quality_v1`: judges the agent's tool calls against
  rubrics about *how it used its tools* (does it filter `list_devices` by the
  room the user named? does it avoid changing the temperature when only asked to
  inspect devices?).

Each criterion scores the agent against the custom yes/no rubrics you provide, so
you evaluate quality directly instead of matching a golden answer. Because the
judge is an LLM, this sample needs a model credential for the judge (a Gemini API
key or Vertex), in addition to the credential used for the agent's own inference.

## Sample Inputs

The eval set (`home_automation.evalset.json`) contains one single-turn case:

- `What devices are in the Bedroom?`

## How To

Run the sample from the workspace root:

```bash
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/rubric_criteria/home_automation.evalset.json \
    --config_file_path contributing/samples/evaluation/rubric_criteria/eval_config.json \
    --print_detailed_results
```

`adk eval` takes the agent folder and the eval-set file as two separate
arguments, so this folder holds only eval data (`home_automation.evalset.json`),
the criteria config (`eval_config.json`), and this README, with no agent code.

### The `rubrics` list

Both criteria are configured in `eval_config.json` through a `rubrics` list. Each
rubric is a single yes/no property the judge decides against the agent's
behavior:

- `rubric_id`: a stable, unique identifier for the rubric (e.g.
  `reports_device_state`). It labels the rubric in the scored output and must be
  unique within the criterion.
- `rubric_content.text_property`: the natural-language property being judged,
  phrased so the answer is a clean "yes" or "no" (e.g. "The response is concise
  and free of filler."). Write each property as one fair, achievable behavior;
  avoid bundling several requirements into one rubric.

The `rubrics` list must be non-empty: `RubricBasedEvaluator` asserts this at
init, so a rubric-based criterion with no rubrics fails immediately.

For each invocation the judge is sampled `num_samples` times (here `5`); the
per-rubric verdicts are combined by majority vote, and the criterion score is the
fraction of rubrics that pass. The `threshold` then decides the case:
`rubric_based_final_response_quality_v1` uses `0.8` (a strong majority of its
rubrics must hold), and `rubric_based_tool_use_quality_v1` uses `1.0` (every
tool-use rubric must hold).

### Criterion-level vs. per-case rubrics

The rubrics in `eval_config.json` are criterion-level: they apply to every
eval case scored by that criterion. You can also attach rubrics to a single case
via `EvalCase.rubrics` in the eval set. Per-case rubrics are filtered by their
`type` field before they are handed to a criterion:

- `rubric_based_final_response_quality_v1` only consumes rubrics of type
  `FINAL_RESPONSE_QUALITY`.
- `rubric_based_tool_use_quality_v1` only consumes rubrics of type
  `TOOL_USE_QUALITY`.

The filtered per-case rubrics are then added to the criterion-level list to
form the effective rubric list for the case. Rubric IDs must be unique across the
two scopes: a `rubric_id` that appears in both the criterion-level list and a
case's `EvalCase.rubrics` raises an error: duplicates are not silently
deduplicated or overridden. Use criterion-level rubrics for expectations shared
across the whole eval set and per-case rubrics for expectations unique to one
scenario.

### When quality rubrics beat reference matching

Reach for rubric-based criteria when "correct" isn't a single golden answer or
trajectory. Reference-based criteria like `response_match_score` (ROUGE-1) or
`tool_trajectory_avg_score` require you to write the expected answer or the exact
sequence of tool calls, and they penalize any legitimate variation: a
reworded-but-correct answer, or a harmless extra tool call. Rubrics instead let
you state the *qualities* that matter ("confirms the device and its state", "uses
a location filter") and let the judge decide whether the agent exhibited them,
regardless of exact wording or an extra step. That makes them a good fit for
open-ended responses and flexible trajectories where you care about quality, not
byte-for-byte equality. The trade-off is the usual LLM-judge cost: a model call
per sample, plus some run-to-run variability that `num_samples` and majority vote
are there to smooth out. Keep the deterministic reference-based criteria when the
answer or trajectory really is fixed.

## Related Guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
