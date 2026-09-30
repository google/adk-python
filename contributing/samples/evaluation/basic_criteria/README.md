# Basic evaluation criteria

## Overview

Evaluates the shared home-automation agent with the two deterministic,
reference-based criteria:

- `tool_trajectory_avg_score`: does the agent call the right tools with the
  right args? Each expected tool call (name + args) is compared against what the
  agent actually did.
- `response_match_score`: ROUGE-1 word overlap between the agent's final
  response and a reference answer.

Both criteria are computed locally with no judge model, so this sample needs only
a model credential for the agent's own inference (a Gemini API key or Vertex).

## Sample Inputs

The eval set (`home_automation.evalset.json`) contains two single-turn cases:

- `Turn off device_2.`
- `What is the temperature in the Living Room?`

## How To

Run the sample from the workspace root:

```bash
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/basic_criteria/home_automation.evalset.json \
    --config_file_path contributing/samples/evaluation/basic_criteria/eval_config.json \
    --print_detailed_results
```

`adk eval` takes the agent folder and the eval-set file as two separate
arguments, so this folder holds only eval data (`home_automation.evalset.json`),
the criteria config (`eval_config.json`), and this README, with no agent code.

### `match_type` for tool trajectory

`tool_trajectory_avg_score` has a `match_type` (set to `EXACT` here in
`eval_config.json`) that controls how the expected and actual tool calls are
compared:

- `EXACT`: the actual tool calls must match the expected calls one-for-one, in
  the same order, with identical args. Use this when the trajectory is fully
  deterministic (as in this sample).
- `IN_ORDER`: the expected calls must appear in the given order, but extra
  actual calls in between are tolerated. Useful when the agent may take
  additional, harmless steps.
- `ANY_ORDER`: the expected calls must all appear, but order does not matter.
  Useful when the agent may reorder independent tool calls.

The `threshold` is `1.0`, so every expected call must match for the case to pass.

### Why `response_match_score` uses a `0.6` threshold

`adk eval` runs live inference, so the exact wording of the agent's final
response varies from run to run (for example, "I have turned off device_2." vs
"device_2 has been switched off."). `response_match_score` is a ROUGE-1 score,
which measures word overlap rather than exact-string equality, so it tolerates
this phrasing variation. The `0.6` threshold requires the response to share most
of its wording with the reference while still allowing some rewording. Raise it
toward `1.0` for stricter wording, lower it to tolerate more paraphrasing.

### Expectations captured from a real run

The expected `tool_uses` (tool names and args) in `home_automation.evalset.json`
were captured from an actual `adk eval` run of the agent: run with
`--print_detailed_results`, read the printed Actual-vs-Expected, then set the
expected values to match what the agent really produced. The reference
`final_response` for each case is an independently written natural answer (not a
copy of the model output), which is exactly what ROUGE-1 is designed to tolerate.

## Related Guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
