# Test file vs. eval set

## Overview

A `.test.json` file and a `.evalset.json` file are the same `EvalSet` Pydantic
schema. `adk eval` loads either one with `load_eval_set_from_file`, which
validates by schema, not by file extension, so both run with the exact same
`adk eval` command. The two extensions are only a naming *convention*:

- A `.test.json` is the "unit test" convention: one simple session, kept small and
  focused, like a single unit test.
- A `.evalset.json` is the "integration test" convention: multiple, longer,
  multi-turn sessions grouped together, like an integration test.

This sample ships one of each against the shared `home_automation_agent`:

- `single_turn.test.json`: a single one-turn session.
- `multi_session.evalset.json`: two sessions, one of which is a two-turn
  conversation.

## Sample Inputs

`single_turn.test.json` (one session):

- `What's the temperature in the Kitchen?`

`multi_session.evalset.json` (two sessions):

- `list_then_turn_off` (two turns): `Which devices are on?` then
  `Turn that one off.`
- `set_bedroom_temperature` (one turn): `Set the Bedroom to 21 degrees.`

## How To

Both files run with the same `adk eval` command; only the eval-data path changes.
Run from the workspace root.

Run the `.test.json`:

```bash
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/test_file_vs_evalset/single_turn.test.json \
    --config_file_path contributing/samples/evaluation/test_file_vs_evalset/eval_config.json \
    --print_detailed_results
```

Run the `.evalset.json`:

```bash
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/test_file_vs_evalset/multi_session.evalset.json \
    --config_file_path contributing/samples/evaluation/test_file_vs_evalset/eval_config.json \
    --print_detailed_results
```

`--print_detailed_results` prints an Actual-vs-Expected table so you can compare
the agent's real tool calls and responses against the expected values in each
file.

The `.test.json` name is also the format that `pytest` + `AgentEvaluator.evaluate`
auto-discovers, so the same file can be driven from a Python test without change
(not shown here, since this sample uses `adk eval` only).

### `match_type: IN_ORDER`

`eval_config.json` scores the tool trajectory with `match_type: "IN_ORDER"`: the
expected tool calls must appear in the given order, but any extra actual tool
calls in between are tolerated. The `threshold` is `1.0`, so every expected call
(name + args) must still match a real call. `response_match_score` uses a `0.6`
threshold, a ROUGE-1 word-overlap score that tolerates the phrasing variation of
live inference.

## Related Guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
