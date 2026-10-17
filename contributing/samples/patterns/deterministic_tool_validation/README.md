# Deterministic Pre-Execution Tool Validation

This recipe demonstrates a deterministic, spec-bound policy check using ADK's `before_tool_callback` (and `BasePlugin`).

## Background

While `ToolCallIntegrityPlugin` provides cryptographic HMAC integrity verification (attesting that a tool call was emitted by a trusted model/key and not tampered with in session storage), deterministic policy validation ensures that tool calls adhere to fixed physical bounds and operator-declared plan envelopes.

This pattern enforces:
- **Operation allowlists**: Verifies that the tool being invoked is part of the operator-declared plan.
- **Resource target allowlists with URI normalization**: Validates that external resources (e.g. API endpoints, database targets) match permitted canonical URIs, preventing bypasses via un-normalized URI variations or traversal tricks.
- **Spec-bound argument limits**: Enforces fixed numeric boundaries (e.g., minimum and maximum operating temperatures) and allowed discrete values without placing an LLM in the validation path.
- **Short-circuiting**: Returning a non-`None` dictionary from `before_tool_callback` short-circuits execution, preventing the tool body from running while emitting a structured rejection event.
- **Decision ledger auditing**: Appends a structured audit record (`verdict`, `rule`, `authority`, `timestamp`) into `tool_context.state['decision_ledger']` (addressing Issue #6099).

## Running the Sample

Run the agent with the ADK CLI:

```shell
adk run contributing/samples/patterns/deterministic_tool_validation
```
