# Efficiency metrics

## Overview

Shows the **informational efficiency metrics** on the shared home-automation
agent, and the per-type token breakdown `token_usage_v1` reports underneath its
score.

Efficiency metrics are **informational**: they report a value and never pass or
fail an eval case (their status is `INFORMATIONAL`).

> The three efficiency metrics — `tool_call_count_v1`,
> `inference_call_count_v1`, and `token_usage_v1` — are reported automatically
> on **every** eval run, with no configuration. Note that `eval_config.json`
> below names **no** efficiency metric at all, and all three still show up:
> there is nothing to turn on, because nothing can turn them off.

## Sample inputs

`home_automation.evalset.json` has two single-turn cases (a device action and a
temperature lookup). They exist only to drive real inference so the efficiency
metrics have something to measure.

## Run it

`eval_config.json` configures one quality metric and nothing else:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0
  }
}
```

From the workspace root:

```bash
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/efficiency_metrics/home_automation.evalset.json \
    --config_file_path contributing/samples/evaluation/efficiency_metrics/eval_config.json \
    --print_detailed_results
```

Alongside the one metric that was asked for, all three efficiency metrics
appear on their own:

```
Metric: tool_trajectory_avg_score, Status: PASSED, Score: 1.0, Threshold: 1.0
---------------------------------------------------------------------
Metric: tool_call_count_v1, Status: INFORMATIONAL, Score: 1.0, Threshold: None
---------------------------------------------------------------------
Metric: inference_call_count_v1, Status: INFORMATIONAL, Score: 2.0, Threshold: None
---------------------------------------------------------------------
Metric: token_usage_v1, Status: INFORMATIONAL, Score: 1218.0, Threshold: None
Token breakdown:
  total:            1218
    input:          921
      prompt:       921
        cached:     n/a
      tool use:     n/a
    output:         297
      candidates:   30
      reasoning:    267
```

## Reading the breakdown

Indentation is containment: each count is a part of the one above it that is
indented less. `total` is `input` plus `output`, and it is derived from those
two rather than taken from the backend's own reported total, so the headline
number always agrees with the lines beneath it.

Two counts read `n/a` above, and both are real rather than missing data: this
agent uses no context cache, and its tools are ordinary client-side functions
whose results bill as plain prompt tokens rather than as server-side tool
tokens. `n/a` never means zero.

To track how much a change moves the model's reasoning, read `reasoning_tokens`
from the breakdown — in the CLI table, or from
`token_usage_details` in the saved result JSON.

## How the value is reported

Each metric is computed **per invocation (per turn)** and those per-turn values
are all reported; the single overall value is their **average** (so it stays
comparable across cases with different turn counts). For a per-case total (e.g.
total tokens for the whole conversation), sum the per-invocation values in the
saved result JSON.

Results are written automatically to
`home_automation_agent/.adk/eval_history/<name>.evalset_result.json`.

## Caveats

- **`token_usage_v1` needs backend usage metadata.** Vertex / AI Studio Gemini
  report it; some backends don't (then the metric is n/a).
- **Reasoning tokens need a thinking-capable model** that reports
  `thoughts_token_count`. Models without it report `n/a` for that count rather
  than zero. Simple deterministic turns may spend few reasoning tokens.
- **These metrics never pass or fail.** They report a value with status
  `INFORMATIONAL`; your real quality metrics still decide pass/fail.

## Related guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
