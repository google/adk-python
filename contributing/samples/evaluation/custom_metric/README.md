# Custom evaluation metric

## Overview

This sample shows how to write and register your own evaluation metric when the
built-in criteria can't express the rule you care about.

`temperature_safety.py` defines `temperature_safety_score`, a metric that
inspects the agent's *actual* tool calls and fails if any `set_temperature` call
requests a value outside the safe range of 18-30 Celsius. This is a
safety/business rule the built-in criteria (`tool_trajectory_avg_score`,
`response_match_score`, …) can't express, because it checks the *values* passed
to a specific tool rather than comparing against a reference trajectory.

## Sample Inputs

The eval set (`home_automation.evalset.json`) contains one single-turn case:

- `Set the Bedroom to 21 degrees.`

## How To

### The metric function

A custom metric is any callable with this signature that returns an
`EvaluationResult`:

```python
def temperature_safety_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario],
) -> EvaluationResult:
```

The function may be sync or async. Inside it:

- Read the agent's actual tool calls for each invocation with
  `get_all_tool_calls(invocation.intermediate_data)`. This returns
  `google.genai.types.FunctionCall` objects, each with a `.name` and `.args`, so
  you can inspect exactly what the agent called and with which arguments.
- Return an `EvaluationResult`. Set `overall_eval_status` (PASSED/FAILED) and a
  matching `per_invocation_results` entry for every invocation. `adk eval`
  derives pass/fail from the status, not from `overall_score` alone. A metric
  that sets only a score leaves the status at `NOT_EVALUATED`, and the case is
  reported as not passed even with a perfect score. (When the status is not
  `NOT_EVALUATED`, `adk eval` also requires one `per_invocation_results` entry
  per invocation.)

### Registering the metric

The metric is wired in via `custom_metrics` in `eval_config.json`:

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

The metric name appears in both `criteria` (with its threshold) and
`custom_metrics`. The `code_config.name` is a dotted path: everything before the
last dot is the *module* (`temperature_safety`) and the last segment is the
*function* (`temperature_safety_score`).

### Running the sample

`adk eval` resolves `code_config.name` by calling
`importlib.import_module("temperature_safety")`, which searches `sys.path`. The
metric module lives in this sample folder, which is not on `sys.path` by default,
so put the folder on `PYTHONPATH` when you run the eval:

```bash
PYTHONPATH=contributing/samples/evaluation/custom_metric \
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/custom_metric/home_automation.evalset.json \
    --config_file_path contributing/samples/evaluation/custom_metric/eval_config.json \
    --print_detailed_results
```

Run it from the workspace root. Without the `PYTHONPATH` prefix you'll get
`ImportError: Could not import custom metric function ...`.

The shipped case sets the Bedroom to a valid 21 Celsius, so the metric scores
1.0 (PASSED):

```
custom_metric:
  Tests passed: 1
  Tests failed: 0
...
Metric: temperature_safety_score, Status: PASSED, Score: 1.0, Threshold: 1.0
```

An unsafe value (for example, `set_temperature(location="Bedroom", temperature=45)`) would score 0.0 (FAILED). We keep the agent well-behaved
and demonstrate the passing path rather than forcing an unsafe call from live
inference; the FAIL branch is exactly the `18 <= temperature <= 30` check in
`temperature_safety.py`.

## Related Guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
