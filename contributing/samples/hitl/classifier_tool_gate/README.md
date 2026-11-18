# Classifier Tool Gate Sample

## Overview

This sample demonstrates how to gate tool calls with a classifier from `google.adk.labs.classifiers`.

The agent manages a small customer database through two tools: `run_sql` and `send_email`. Before each tool call, a `before_tool_callback` asks a classifier a `Binary` question about the call: "Would this tool call delete or change data, or contact a customer?" Calls the classifier rates below the threshold run right away. The rest pause until the user approves them through ADK's tool confirmation flow.

`FunctionTool(require_confirmation=...)` decides per tool, or with code you write for each tool. A classifier decides from the arguments, so one question covers every tool: a `SELECT` through `run_sql` runs, and a `DELETE` through the same tool asks first.

This sample uses `LlmClassifier`, which answers with any `BaseLlm`. A dedicated decision model answers faster, which matters for a check that runs before every tool call, and can be swapped in without other changes.

## Sample Inputs

- `Which customers are on the pro plan?`

- `Move Alan Turing to the pro plan.`

  *The gate asks for confirmation before the `UPDATE` runs.*

- `Delete every customer on the free plan.`

- `Email Grace Hopper to thank her for renewing.`

## How To

1. Create the callback with `create_tool_gate_callback` from [`tool_gate.py`](tool_gate.py) and pass it as the agent's `before_tool_callback`:

   ```python
   tool_gate = create_tool_gate_callback(
       classifier=LlmClassifier(model="gemini-3.5-flash"),
       instructions="Would this tool call delete or change data, or contact a customer?",
       on_threshold_exceeded="confirm",
   )

   root_agent = Agent(..., before_tool_callback=tool_gate)
   ```

1. Choose what happens when a call's probability is at or above `threshold` (default `0.5`):

   - `on_threshold_exceeded="block"` (default) skips the call and returns an error to the model.
   - `on_threshold_exceeded="confirm"` asks the user with `tool_context.request_confirmation()` and runs the call only if they approve. When the user answers, the call comes back through the callback with `tool_context.tool_confirmation` set, and the callback runs or rejects it without asking the classifier again.

1. By default the classifier sees the tool's name, description and arguments. To classify something else, such as recent conversation or session state, pass a `state_extractor(tool, args, tool_context)` that returns the classifier input.

If the classifier raises, the gate treats the call as risky and blocks it or asks the user, so an outage can't let a risky call through.

`LlmClassifier` probabilities are the model's stated values, not measured frequencies, and the right threshold differs between classifiers. Pick `threshold` from an evaluation set of calls you expect to pass and to stop.
