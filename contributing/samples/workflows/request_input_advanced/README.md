# ADK Workflow Request Input Advanced Sample

## Overview

This sample demonstrates advanced features for requesting Human-in-the-Loop (HITL) input dynamically during an **ADK Workflow** execution.

Specifically, it highlights how to pass structured data to the client UI using the `payload` parameter, and how to mandate a structured response type using the `response_schema` parameter on the yielded `RequestInput` event.

In this scenario, an employee requests time off by providing a natural language description of their request (e.g., "I need next Monday off to go to the dentist").

- An LLM agent (`process_request`) parses the natural language into a structured Pydantic model containing the number of `days` and a `reason`.
- A python node (`evaluate_request`) evaluates the parsed request:
  - If `days <= 1`, it yields a `TimeOffDecision` approving the request.
  - If `days > 1`, it yields a `RequestInput` to a manager. It attaches the request details to the `payload` so the client UI can render it. It enforces that the manager must respond with a JSON object containing an `approved` boolean and an optional `approved_days` integer by specifying `response_schema` with a valid Pydantic JSON schema.

## Sample Inputs

Start the workflow by providing the initial time off request in natural language:

- `I'm feeling under the weather and need to take today off.`

  *Parses as 1 day, auto-approves.*

- `Taking my family to Disney World, I'll be out for 5 days next week.`

  *Parses as 5 days, routes to manager review.*

When the terminal prompts you as the manager, provide valid JSON matching the schema:

- `{"approved": true, "approved_days": 5}`

- `{"approved": false, "approved_days": 0}`

## Graph

```mermaid
graph TD
    START --> process_request[process_request <br/>LLM Agent]
    process_request --> evaluate_request
    evaluate_request -->|Yields TimeOffDecision OR RequestInput| process_decision
    process_decision --> END[END]
```

## How To

1. **Define the Response Schema:** Use a Pydantic model's `model_json_schema()` to get a standard layout of what the human should return.

   ```python
   from typing import Optional
   from pydantic import BaseModel, Field

   class TimeOffDecision(BaseModel):
       approved: bool = Field(...)
       approved_days: Optional[int] = Field(None)
   ```

1. **Yield a RequestInput:** Pass the schema and optionally a `payload` for the client to display.

   ```python
   def evaluate_request(request: TimeOffRequest):
       # ... logic to check if manager review is needed ...
       yield RequestInput(
           interrupt_id="manager_approval",
           message="Please review this time off request.",
           payload=request,
           response_schema=TimeOffDecision.model_json_schema()
       )
   ```

1. **Parse the Resumed Input:** When the workflow resumes, the `node_input` to the next node will be the parsed Pydantic model implicitly (if type-hinted).

   ```python
   def process_decision(request: TimeOffRequest, node_input: TimeOffDecision):
       if node_input.approved:
           # ...
   ```
