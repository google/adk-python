# Human-in-the-Loop

Pause a workflow to ask the user something, then continue with their answer.

```python
from google.adk import Context, Event, Workflow
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import RequestInput
from google.adk.workflow import FunctionNode
```

## Asking

Yield or return a `RequestInput`. The node's output stream is normalized, so
either works — a plain function can return one directly without becoming a
generator.

```python
async def approval_gate(ctx: Context, node_input: str):
  yield RequestInput(message='Please approve this action:')


def evaluate_request(request: TimeOffRequest):
  if request.days <= 1:
    return TimeOffDecision(approved=True)  # no interrupt at all
  return RequestInput(
      interrupt_id='manager_approval',
      message='Please review this time off request.',
      payload=request,
      response_schema=TimeOffDecision,
  )
```

The workflow emits an `adk_request_input` function call and stops. Responding to
that function call resumes it.

| Field | Type | Notes |
|---|---|---|
| `interrupt_id` | `str` | Auto-generated UUID when omitted |
| `message` | `str \| None` | Shown to the user |
| `payload` | `Any` | Arbitrary data carried along with the request |
| `response_schema` | Pydantic class, Python type, or JSON-schema dict | Expected response shape |

## Resuming: `rerun_on_resume`

The flag on the interrupted node decides what happens when the answer arrives.

**`rerun_on_resume=False`** (the default for `FunctionNode`) — the node is *not*
re-executed; the user's response becomes its output and flows downstream.

```python
approval_node = FunctionNode(func=ask_approval, rerun_on_resume=False)
```

**`rerun_on_resume=True`** (the default for an `LlmAgent` used as a node) — the
node runs again from the top, with answers available in `ctx.resume_inputs`,
keyed by `interrupt_id`.

```python
async def interactive_node(ctx: Context, node_input: str):
  if ctx.resume_inputs:
    answer = list(ctx.resume_inputs.values())[0]
    yield Event(output=f'User said: {answer}')
  else:
    yield RequestInput(message='What should I do?')
```

## Several questions from one node

Because a re-run node sees every answer so far, one node can walk a form:

```python
async def multi_step_form(ctx: Context, node_input: str):
  if not ctx.resume_inputs:
    yield RequestInput(interrupt_id='ask_name', message='What is your name?')
    return

  if 'ask_email' not in ctx.resume_inputs:
    yield RequestInput(interrupt_id='ask_email', message='What is your email?')
    return

  yield Event(output={
      'name': ctx.resume_inputs['ask_name'],
      'email': ctx.resume_inputs['ask_email'],
  })
```

## Feeding the answer back into a loop

A review-and-revise cycle turns the answer into a route. Vary the
`interrupt_id` per iteration (see the best-practices reference for why) and read
`ctx.resume_inputs` with the same id:

```python
async def review(ctx: Context, node_input: Any):
  review_count = ctx.state.get('review_count', 0)
  interrupt_id = f'review_{review_count}'

  response = ctx.resume_inputs.get(interrupt_id)
  if response:
    yield Event(
        output=response,
        route='approved' if response.get('approved') else 'rejected',
        state={'review_count': review_count + 1},
    )
    return

  yield RequestInput(
      interrupt_id=interrupt_id,
      message='Approve this plan?',
      response_schema=ApprovalSchema,
  )
```

Wire the routes back with `(review, {'rejected': revise, 'approved': publish})`.

## Resumable versus replayed

**Replay (the default).** With no `App`, or with `is_resumable=False`, each
response replays the workflow from `START`. Completed nodes are skipped and
state is rebuilt from event history, so only the interrupted node actually runs
again. Fine for a single interrupt; the replay cost grows with the graph.

**Resumable.** Export an `App` with `is_resumable=True` and the workflow
checkpoints its progress into `event.actions.agent_state` and resumes at the
interrupted node instead of replaying.

```python
root_agent = Workflow(name='my_workflow', edges=[...])

app = App(
    name='my_app',
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

Export both `root_agent` and `app` from `agent.py` — the loader prefers `app`
but other tooling looks for `root_agent`. Use resumable mode for multi-step
interrupts, for `LongRunningFunctionTool`, and for any graph large enough that
replaying it is wasteful.

## Human-in-the-loop from an LLM agent

An `LlmAgent` pauses through a `LongRunningFunctionTool` rather than
`RequestInput`:

```python
from google.adk.tools import LongRunningFunctionTool


def approval_tool(request: str) -> str:
  """Request human approval for an action."""
  return f'Approved: {request}'


llm_agent = LlmAgent(
    name='agent_with_approval',
    model='gemini-2.5-flash',
    instruction='When you need approval, use approval_tool.',
    tools=[LongRunningFunctionTool(func=approval_tool)],
)
```

The agent node already defaults to `rerun_on_resume=True`, so it picks the
conversation back up on its own.

## Answering from client code

```python
from google.genai import types

function_call_id = interrupt_event.content.parts[0].function_call.id

response = types.Content(
    role='user',
    parts=[types.Part(
        function_response=types.FunctionResponse(
            id=function_call_id,
            name='adk_request_input',
            response={'result': "User's answer here"},
        )
    )],
)

async for event in runner.run_async(
    user_id=user_id, session_id=session_id, new_message=response
):
  ...
```

A non-dict answer is carried under the `result` key as shown; a dict response
matching `response_schema` is passed through as-is.
