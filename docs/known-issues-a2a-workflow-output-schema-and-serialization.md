# Known Issues: A2A Workflow Output Schema and Serialization

## Discarded Workflow `output_schema` in `to_a2a()`

When a `Workflow` with a configured `output_schema` is exposed over the Agent-to-Agent (A2A) protocol via `to_a2a()`:
- The final structured object validated against `output_schema` may not be placed into the A2A response artifact.
- Instead, the artifact may only contain the raw text output of the last executed agent node.

## Schema Validation Serialization Mode

In `BaseNode._validate_schema`, validated Pydantic `BaseModel` instances are converted using `model_dump()` in Pydantic Python mode rather than JSON mode. Fields containing non-primitive types (such as `Decimal`, `datetime`, or `UUID`) are preserved as Python objects and can cause serialization errors when downstream components attempt standard JSON serialization.

## Relayed Human-Input Responses

When a remote agent pauses for user confirmation or input (e.g., `adk_request_confirmation`) and relays the pause to the caller:
- Responding to the pause in `RemoteA2aAgent` may flatten the human response into a plain text message rather than forwarding a `FunctionResponse` matching the tool call name.
- This can prevent the remote paused agent from recognizing completion of the requested confirmation.

*Note: Maintainer verification is required regarding planned upstream fixes for A2A artifact output mapping and relayed function response preservation.*
