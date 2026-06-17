# Human-in-the-Loop (HITL) Reference

Pause workflow execution to request user input and resume with their response.

## 📋 Agent Verification Checklist (HITL)
Use this checklist when implementing human-in-the-loop logic:
- [ ] **Unique ID**: Is the `interrupt_id` unique per iteration in loops? (Critical to prevent infinite loops)
- [ ] **Resumability**: For multi-step HITL, did you export an `App` with `is_resumable=True`?
- [ ] **Resume Inputs**: If `rerun_on_resume=True` (default for LLM nodes), does the node handle `ctx.resume_inputs`?

## 💡 Quick Reference
- **Request Input**: `yield RequestInput(message="Question", response_schema=Schema)`
- **Resumable Config**: `ResumabilityConfig(is_resumable=True)`

HITL works in two modes:

### Resumable mode (recommended for multi-step HITL)

Export an `App` with resumability. The workflow checkpoints state and resumes at the interrupted node:

```python
from google.adk.apps.app import App, ResumabilityConfig

app = App(
    name="my_app",
    root_agent=workflow_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

The agent loader checks for `app` before `root_agent`, so export both from `agent.py`.

### Non-resumable mode (simpler, no App needed)

The workflow replays from START on each user response, reconstructing state from session events. No `App` or `ResumabilityConfig` needed — just define `root_agent`. This works for simple single-interrupt HITL but replays all nodes up to the interrupt point on each resume.

## Imports

```python
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.workflow import Workflow
from google.adk.apps.app import App, ResumabilityConfig
```

## Basic Request Input

Yield or return a `RequestInput` to pause execution and ask the user for input:

```python
# Yield from a generator
async def approval_gate(ctx: Context, node_input: str):
  yield RequestInput(
      message="Please approve this action:",
      response_schema={"type": "string"},
  )

# Or return directly from a regular function (no generator needed)
def evaluate_request(request: TimeOffRequest):
  if request.days <= 1:
    return TimeOffDecision(approved=True)  # Auto-approve
  return RequestInput(
      interrupt_id="manager_approval",
      message="Please review this time off request.",
      payload=request,
      response_schema=TimeOffDecision,
  )
```

The workflow pauses and emits a function call event to the user. When the user responds, the workflow resumes.

## RequestInput Fields

```python
from pydantic import BaseModel

class ApprovalResponse(BaseModel):
  approved: bool
  comment: str

RequestInput(
    interrupt_id="custom_id",         # Auto-generated UUID if omitted
    message="Question for user",      # Display message
    payload={"key": "value"},         # Custom data to include
    response_schema=ApprovalResponse, # Pydantic class, Python type, or JSON schema dict
)
```

| Field | Type | Description |
|-------|------|-------------|
| `interrupt_id` | `str` | Unique ID for this interrupt (auto-generated UUID) |
| `message` | `str` | Message shown to the user |
| `payload` | `Any` | Custom payload sent with the request |
| `response_schema` | `type \| dict` | Expected response format (Pydantic BaseModel class, Python type, or JSON schema dict) |

## Resume Behavior: rerun_on_resume

When a node is interrupted and the user responds, the `rerun_on_resume` flag controls what happens:

### rerun_on_resume=False (default for FunctionNode)

The user's response becomes the node's output. The node is NOT re-executed:

```python
from google.adk.workflow import FunctionNode

async def ask_approval(ctx: Context, node_input: str):
  yield RequestInput(message="Approve?")

# Node won't rerun; user's response is passed as output to next node
approval_node = FunctionNode(ask_approval, rerun_on_resume=False)
```

### rerun_on_resume=True (default for LlmAgentWrapper)

The node is re-executed with the user's response available in `ctx.resume_inputs`:

```python
async def interactive_node(ctx: Context, node_input: str):
  if ctx.resume_inputs:
    # Second run: user responded
    user_answer = list(ctx.resume_inputs.values())[0]
    yield Event(output=f"User said: {user_answer}")
  else:
    # First run: ask the user
    yield RequestInput(message="What should I do?")
```

## HITL with LLM Agents

LLM agents support HITL via `LongRunningFunctionTool`:

```python
from google.adk.tools.long_running_tool import LongRunningFunctionTool

def approval_tool(request: str) -> str:
  """Request human approval for an action."""
  return f"Approved: {request}"

llm_agent = LlmAgent(
    name="agent_with_approval",
    model="gemini-2.5-flash",
    instruction="When you need approval, use the approval_tool.",
    tools=[LongRunningFunctionTool(func=approval_tool)],
)

# LlmAgentWrapper has rerun_on_resume=True by default
agent = Workflow(
    name="hitl_workflow",
    edges=[
        ('START', llm_agent),
        (llm_agent, next_step),
    ],
)
```

## Multi-Step HITL

A node can request input multiple times by checking `ctx.resume_inputs`:

```python
async def multi_step_form(ctx: Context, node_input: str):
  if not ctx.resume_inputs:
    # Step 1: Ask for name
    yield RequestInput(
        interrupt_id="ask_name",
        message="What is your name?",
    )
    return

  if "ask_name" in ctx.resume_inputs and "ask_email" not in ctx.resume_inputs:
    # Step 2: Ask for email
    yield RequestInput(
        interrupt_id="ask_email",
        message="What is your email?",
    )
    return

  # All inputs collected
  name = ctx.resume_inputs["ask_name"]
  email = ctx.resume_inputs["ask_email"]
  yield Event(output={"name": name, "email": email})
```

## HITL in Loops (Unique interrupt_id)

When a HITL node can fire multiple times in a loop (e.g. reject → revise → re-approve), you **must use a unique `interrupt_id` per iteration**. Reusing the same ID causes event-based state reconstruction to confuse earlier responses with the current interrupt, resulting in an infinite restart loop.

```python
async def review(ctx: Context, node_input: Any):
  # Counter-based unique ID per review cycle
  review_count = ctx.state.get('review_count', 0)
  interrupt_id = f'review_{review_count}'

  response = ctx.resume_inputs.get(interrupt_id)
  if response:
    route = 'approved' if response.get('approved') else 'rejected'
    yield Event(
        output=response,
        route=route,
        state={'review_count': review_count + 1},
    )
    return

  yield RequestInput(
      interrupt_id=interrupt_id,
      message="Approve this plan?",
      response_schema=ApprovalSchema,
  )
```

Key points:
- Store a counter in `ctx.state` and increment on each response
- Use the counter in the `interrupt_id` (e.g. `review_0`, `review_1`, ...)
- Look up `ctx.resume_inputs` with the same counter-based ID
- This applies to both resumable and non-resumable modes

## Resumability Configuration

### Resumable mode (recommended for multi-step HITL)

```python
from google.adk.apps.app import App, ResumabilityConfig

# Export BOTH root_agent and app from agent.py
root_agent = Workflow(name="my_workflow", edges=[...])

app = App(
    name="my_app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

When `is_resumable=True`:
- Workflow state is checkpointed in session's `agent_states` map
- On resume, the workflow loads checkpointed state and resumes at the interrupted node
- Required for multi-step HITL, `LongRunningFunctionTool`, and complex workflows

### Non-resumable mode (simpler)

When `is_resumable=False` (default) or no `App` is exported:
- No state checkpointing — the workflow replays from START on each user response
- State is reconstructed from session events during replay
- Completed nodes are skipped; execution resumes at the interrupted node
- Works for simple single-interrupt HITL without needing `App` or `ResumabilityConfig`
- For multi-step HITL or complex workflows, use resumable mode instead

## Responding to HITL Requests

From the client side, respond to function calls:

```python
from google.genai import types

# Extract function_call_id from the interrupt event
function_call_id = interrupt_event.content.parts[0].function_call.id

# Create response
response = types.Content(
    role="user",
    parts=[types.Part(
        function_response=types.FunctionResponse(
            id=function_call_id,
            name="adk_request_input",
            response={"result": "User's answer here"},
        )
    )],
)

# Send response to resume the workflow
async for event in runner.run_async(
    user_id=user_id,
    session_id=session_id,
    new_message=response,
):
  # Process resumed workflow events
  pass
```
