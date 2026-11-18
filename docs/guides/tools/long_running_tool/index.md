# LongRunningFunctionTool

The `LongRunningFunctionTool` class wraps a Python function whose real result
arrives later, such as after a batch job finishes, a manager approves a request,
or a webhook fires. The function starts the work and returns right away, and
your application sends the final result to the agent when it is ready.

If what you need is a person's approval before a tool does its work,
[tool confirmation](../tool_confirmation/index.md) fits that case more closely.
ADK pauses the call before the tool runs, your application sends back a yes or
a no, and ADK then runs the tool itself, so you do not relay the tool's result.
The `LongRunningFunctionTool` class is for a tool that starts the work right
away and whose result your application sends later, such as the approval
ticket that the example on this page opens in another system.

## Introduction

An ordinary tool call finishes inside a single turn. The model calls the tool,
the function returns, and the model reads the return value and continues. That
works well until the work takes minutes or needs a person to act. If the
function waits for the work to finish, the whole turn waits with it. If an
ordinary tool returns a status such as "started" instead, the model treats that
status as the answer, and it may call the tool again to find out more.

The `LongRunningFunctionTool` class lets you split the call in two. The function
runs once, starts the work, and returns a status or a ticket id. ADK marks the
call as long-running on the event it emits, which is how your application finds
the call and keeps its id. When the outcome is known, your application sends a
function response with that id, and the agent continues the conversation with
the final result in front of it. ADK also adds a note to the tool's description
that asks the model not to call the tool again while the work is pending.

The class itself is small. Most of the work happens in your application, because
only your application knows when the outside work has finished.

## Get started

The following agent handles expense reimbursements. It pays small amounts
straight away with an ordinary tool, and it asks a manager to approve larger
amounts, which is work that finishes outside the agent.

```python
from google.adk.agents import Agent
from google.adk.tools import LongRunningFunctionTool

def reimburse(purpose: str, amount: float) -> dict:
  """Reimburses the employee for an approved expense."""
  return {"status": "ok"}

def ask_for_approval(purpose: str, amount: float) -> dict:
  """Asks a manager to approve a reimbursement and returns a ticket id."""
  # A real application would open a ticket in its approval system here.
  return {
      "status": "pending",
      "amount": amount,
      "ticket_id": "reimbursement-ticket-001",
  }

root_agent = Agent(
    name="reimbursement_agent",
    instruction=(
        "Handle reimbursements. Approve amounts under $100 yourself with"
        " reimburse(). For larger amounts, call ask_for_approval() and wait"
        " for the manager's decision."
    ),
    tools=[reimburse, LongRunningFunctionTool(func=ask_for_approval)],
)
```

When the model calls `ask_for_approval`, ADK runs the function and records its
return value as the tool's response. The model reads the pending status and
usually tells the user that the request is waiting for approval.

You can try the whole exchange with `adk run` before you write any client code.
When the agent makes a long-running call, the command line prints a line that
starts with `[HITL] Waiting for input for ask_for_approval` and waits for you.
What you type becomes the function's final response. The command sends a JSON
object as it is, and it sends any other text as `{"result": "<your text>"}`.

In your own application, you write that part. While you read the events from
`runner.run_async`, look for a function call whose id appears in the event's
`long_running_tool_ids`, and keep it:

```python
# runner, user_id, session_id, and user_message are your own.
pending_call = None
async for event in runner.run_async(
    user_id=user_id, session_id=session_id, new_message=user_message
):
  for call in event.get_function_calls():
    if call.id in (event.long_running_tool_ids or set()):
      pending_call = call
```

When the manager has decided, build a function response with the same id and
name as the call you kept, and send it as the `new_message` of another
`run_async` call on the same session:

```python
from google.genai import types

# pending_call is the call you kept. runner and the ids are your own.
decision = types.Content(
    role="user",
    parts=[
        types.Part(
            function_response=types.FunctionResponse(
                id=pending_call.id,
                name=pending_call.name,
                response={"status": "approved", "approver": "alice"},
            )
        )
    ],
)

async for event in runner.run_async(
    user_id=user_id, session_id=session_id, new_message=decision
):
  ...
```

ADK uses the id to find the call you are answering, and the model continues with
the manager's decision. In this example, the instruction tells the model to call
`reimburse` next. If the id does not match any function call in the session, the
`run_async` method raises a `ValueError` that says the function call was not
found.

## How it works

A long-running call spans at least two runs of the agent. From your
application's side, it goes through the following steps:

1. The model calls the tool. ADK puts the call's id in the
   `long_running_tool_ids` of the event that carries the call. The
   `Event.is_final_response()` method returns `True` for that event, but the run
   does not stop there, so keep reading events until the stream ends.
2. ADK runs your function once and records what it returns as the tool's
   response. ADK uses a dict as it is and wraps any other value as
   `{"result": value}`.
3. The model reads that response and replies, usually by telling the user that
   the work is pending. The run ends.
4. Later, your application sends a function response with the same id and name.
   You can send more than one, such as a few progress updates followed by the
   final result. Each one continues the conversation and gives the model a new
   turn.
5. Each time a response arrives, the model sees the original call followed by
   the newest response for it. ADK replaces earlier responses for the same call
   in what the model sees, although they stay in the session. The model
   therefore reasons about the current status, and old updates do not accumulate
   in its context.

ADK does not call your function again at any point in these steps. The model
can call it again, though, as described under [Limitations](#limitations).

### Return nothing to hold the call open

If your function returns `None`, or another value that Python treats as false
such as an empty dict, ADK records no response for the call. The run stops right
after the model's call, and the model gets no turn until your application sends
the response. Use this when the tool has nothing useful to report yet and you do
not want the model to say anything to the user in the meantime. If you want the
model to tell the user that the work has started, return a status instead.

### Long-running tools that are not functions

The `is_long_running` attribute exists on every `BaseTool`, and
`LongRunningFunctionTool` sets it to `True`. You can pass `is_long_running=True`
to the constructor of your own `BaseTool` subclass to get the same handling. The
`LongRunningFunctionTool` class does the same for a plain function, and it also
adds the note to the description.

## Configuration options

The `LongRunningFunctionTool` constructor takes the function to wrap as its
`func` argument. ADK builds the tool's name, description, and parameter schema
from the function's name, docstring, and type hints, in the same way as
[`FunctionTool`](../function_tool/index.md). The tool also passes a `Context` to
the function when the function declares one, as `FunctionTool` does. Unlike
`FunctionTool`, the constructor has no `require_confirmation` argument.

### The note added to the description

ADK appends the following text to the description that the model sees:

```
NOTE: This is a long-running operation. Do not call this tool again if it has
already returned some intermediate or pending status.
```

If the function has no docstring, this note becomes the whole description, and
the model has nothing to tell it when to choose the tool. Write a docstring that
says what the tool starts and what its pending result means.

## Advanced applications

The exchange of a pending call and a later response also supports progress
reports and work that finishes after your application restarts.

### Report progress

When a job takes a long time, you can keep the user informed by sending several
function responses for the same call. Give each one the same id and name, and
put the current status in `response`, such as
`{"status": "running", "progress": "40%"}`. Each update replaces the previous
one in what the model sees, and each gives the model a turn in which it can tell
the user how far along the job is. Send the final result the same way when the
job finishes.

ADK stores each update in the session as an event, and each update triggers a
model request, so it adds storage and a model call. Send updates at the rate
your users need rather than every time your job reports progress.

### Keep pending work across restarts

ADK stores the pending call in the session, so the final response can arrive in
a later request, as long as your session service keeps the session. If the
answer might arrive after your application restarts, use a session service that
stores sessions persistently, as described in the
[session guide](../../sessions/session/index.md). Store the call's id and name
together with the user id and session id in whatever system tracks the outside
work, so that the code that finishes the job knows where to send the result.

If you also turn on resumability with `resumability_config` on your `App`, ADK
treats the invocation as paused at the long-running call and resumes it when the
response arrives. The [App guide](../../apps/app/index.md) describes that
option.

## Limitations

A long-running call depends on your application to finish it. Consider the
following limits when you design that part:

- ADK does not poll for the result, listen for a webhook, or time the call out.
  If your application never sends the matching function response, the call
  stays unanswered.
- The note in the description is a request to the model, and the model can
  still call the tool again. Your function runs again when it does, so write it
  to handle a repeated call, for example by returning the existing ticket when a
  request is already open.
- ADK has no built-in way to cancel a pending call or mark it as failed. If the
  outside work fails, send a function response that says so, such as
  `{"status": "failed", "reason": "manager rejected the request"}`.
- A message that carries a function response cannot also carry text. If it
  does, `run_async` raises a `ValueError`, so send the response and any new user
  text as separate messages.
- All the function responses in one message must answer calls from the same
  invocation. Otherwise `run_async` raises a `ValueError`.

## Related samples

- [Long-running functions](../../../../contributing/samples/tools/long_running_functions/agent.py)
  is a minimal example, with one tool that returns an in-progress status.
- [Human in the loop](../../../../contributing/samples/hitl/human_in_loop/agent.py)
  is the approval workflow on this page. Its
  [`main.py`](../../../../contributing/samples/hitl/human_in_loop/main.py) shows
  the client code that keeps the call and sends the manager's decision.
