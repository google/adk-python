# ToolConfirmation

Tool confirmation pauses a tool call until a person approves it, so that an
agent cannot close an account or transfer money on the model's decision alone.
The `ToolConfirmation` class defines the object that carries the request to your
application and the person's answer back to the tool, including an optional
payload for answers that need more than a yes or a no.

## Introduction

ADK gates a tool call with a pause and a resume. When the model calls a gated
tool, ADK stops before the tool does its work and emits a confirmation request,
which is a function call named `adk_request_confirmation`. Your application
shows the request to a person in whatever interface it has, such as a web page,
a chat message, or a terminal prompt, and sends their decision back as a
function response. ADK then runs the original call again with the decision
attached, and the tool either does its work or reports that the call was
rejected.

You can gate a tool in the following ways:

- Pass `require_confirmation` to `FunctionTool` or `McpToolset`. ADK asks for
  approval before your function runs, and the function itself contains no
  confirmation code. Use this form when a yes or no on the whole call is enough.
- Call `request_confirmation()` on your function's `Context`. Use this form
  when the decision depends on something the function works out first, when you
  want to write the text the person sees, or when you want the person to send
  data back along with their answer.

Both forms produce the same request and accept the same response, so your
application handles them with the same code.

## Get started

To gate a tool without changing its code, wrap it in a `FunctionTool` with
`require_confirmation=True`. The function below has no confirmation code in it,
because ADK only calls it after a person approves the call.

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

def close_account(account_id: str) -> dict:
  """Closes a user account. This is a destructive action."""
  return {"result": f"Account {account_id} closed successfully."}

root_agent = Agent(
    name="account_assistant",
    instruction="Help the user manage their account.",
    tools=[FunctionTool(close_account, require_confirmation=True)],
)
```

You can try it with `adk run`. When the model calls `close_account`, the command
line prints the confirmation request and asks you to type "yes" to confirm. It
treats `y`, `yes`, `true`, and `confirm` as approval, in any letter case, and
anything else as a rejection.

In your own application, you handle the request yourself. While you read the
events from `runner.run_async`, look for a function call named
`adk_request_confirmation`:

```python
# runner, user_id, session_id, and user_message are your own.
pending = None
async for event in runner.run_async(
    user_id=user_id, session_id=session_id, new_message=user_message
):
  for call in event.get_function_calls():
    if call.name == "adk_request_confirmation":
      pending = call
```

The call's arguments tell you what is waiting for approval. The value at
`pending.args["originalFunctionCall"]` holds the name and arguments of the gated
call, and the value at `pending.args["toolConfirmation"]["hint"]` holds a
message for the person. When the person decides, send a function response with
the same id and the name `adk_request_confirmation`, as the `new_message` of
another `run_async` call on the same session:

```python
from google.genai import types

# pending is the call you kept. runner and the ids are your own.
decision = types.Content(
    role="user",
    parts=[
        types.Part(
            function_response=types.FunctionResponse(
                id=pending.id,
                name="adk_request_confirmation",
                response={"confirmed": True},
            )
        )
    ],
)

async for event in runner.run_async(
    user_id=user_id, session_id=session_id, new_message=decision
):
  ...
```

ADK then runs `close_account` with the arguments the model originally passed, and
the agent continues from there. If you send `{"confirmed": False}` instead, the
function body does not run, and the model receives
`{"error": "This tool call is rejected."}` as the tool's result.

## How it works

A confirmed call spans two runs of the agent: one that asks and one that acts.
From your application's side, it goes through the following steps:

1. The model calls the tool.
2. The tool asks for confirmation instead of doing its work. With
   `require_confirmation`, the tool records the request and returns
   `{"error": "This tool call requires confirmation, please approve or reject."}`
   without calling your function. With `request_confirmation`, your function
   records the request and returns whatever you choose.
3. ADK emits the confirmation request as a separate event that holds one
   function call named `adk_request_confirmation`. Its `args` contain
   `originalFunctionCall`, the call being gated, and `toolConfirmation`, which
   holds the hint and any starting payload. With `require_confirmation`, the run
   ends here.
4. Your application collects a decision and sends it back as a function
   response with the same id, the name `adk_request_confirmation`, and a
   response built from the `ToolConfirmation` fields.
5. ADK finds the original call in the session and runs the tool again with the
   `tool_confirmation` property of its `Context` set to the decision. This time
   the tool either does its work or returns the rejection.

In step 5, ADK runs the call with the arguments recorded in the session, so the
call that runs is the call your application showed the person. Before it runs
anything, ADK checks that the agent still has the tool, that the tool still
requires confirmation or asked for it earlier, and that the call in the
confirmation request matches the call in the session history. If any check
fails, the run raises a `ValueError` and the tool does not run.

ADK gates each call separately. Approving one call to `close_account` does not
approve the next one, so a person sees a new request every time the model calls
the tool.

With `require_confirmation`, ADK writes the hint for you. The hint names the
tool and tells a developer to reply with a function response, so the text is
meant for developers, not end users. In a user interface, show the tool name and
arguments from `originalFunctionCall` instead, or use `request_confirmation` to
write your own hint.

## Configuration options

The options cover the argument that turns confirmation on for a tool and the
fields of the `ToolConfirmation` object that your application sends back.

### `require_confirmation`

The `FunctionTool` and `McpToolset` classes both accept `require_confirmation`,
and it defaults to `False`. Set it to `True` to gate every call to the tool. On
`McpToolset`, the setting applies to every tool the server lists, including
tools the server adds later. The
[McpToolset guide](../mcp_tool/mcp_toolset/index.md) describes the argument
alongside the toolset's other options.

You can also pass a callable that decides for each call, which lets you approve
small requests automatically and gate large ones. On `FunctionTool`, ADK calls
it with the same keyword arguments that your function is about to receive,
including the `Context` when your function declares one. The callable can be a
regular function or an `async` one. In the following example, the predicate has
the same parameters as the function it guards:

```python
from google.adk import Context
from google.adk.tools import FunctionTool

def reimburse(amount: float, ctx: Context) -> dict:
  """Reimburses the employee for the given amount."""
  return {"status": "ok"}

async def over_threshold(amount: float, ctx: Context) -> bool:
  return amount > 1000

reimburse_tool = FunctionTool(reimburse, require_confirmation=over_threshold)
```

On `FunctionTool`, the callable has to accept every argument the function
receives. If it leaves one out, the call fails with a `TypeError` about an
unexpected keyword argument. Add `**kwargs` to the callable if you only care
about some of the arguments. On `McpToolset`, ADK passes only the arguments the
callable declares, so the callable can name only the ones it needs.

ADK can call the callable more than once for the same tool call, because it
checks again when the decision comes back. Keep it free of side effects.

### `ToolConfirmation` fields

Your application sends the decision as a dict built from the following fields
of `ToolConfirmation`:

- The `hint` field is a string that defaults to an empty string. It is the text
  to show the person. You set it when you call `request_confirmation`, and ADK
  sets its own text when you use `require_confirmation`.
- The `confirmed` field is a boolean that defaults to `False`. It is the
  decision. Because the default is `False`, a response that leaves it out counts
  as a rejection.
- The `payload` field holds any JSON-serializable value and defaults to `None`.
  It carries data that goes beyond a yes or a no. When you call
  `request_confirmation` with a starting payload, the person can send back a
  changed one.

A response can contain only these keys. The `ToolConfirmation` class rejects
fields it does not know, so an extra key makes the run fail with a validation
error. Put any extra data inside `payload`.

## Advanced applications

The same request and response also let a person send data back with their
answer, and let the decision arrive long after the request.

### Ask for data along with approval

A manager who is asked to approve five days of leave might want to approve
three. To let them, call `request_confirmation` yourself with a starting
`payload`, and read the returned payload when the call runs again:

```python
from google.adk import Context

def request_time_off(days: int, ctx: Context) -> dict:
  """Requests days off for the employee."""
  if days <= 2:
    return {"status": "ok", "approved_days": days}

  confirmation = ctx.tool_confirmation
  if not confirmation:
    ctx.request_confirmation(
        hint=f"Approve {days} days off?",
        payload={"approved_days": days},
    )
    return {"status": "Manager approval is required."}

  if not confirmation.confirmed:
    return {"status": "The time off request is rejected.", "approved_days": 0}

  # A client can confirm without a payload, which approves the full request.
  approved = min((confirmation.payload or {}).get("approved_days", days), days)
  return {"status": "ok", "approved_days": approved}
```

ADK enters this function twice: once to ask and once to act. Everything before
the `request_confirmation` call runs on both passes, so keep side effects after
the point where the function checks the confirmation.

ADK records what the function returns on the first pass as its result. Unless
you also set `ctx.actions.skip_summarization = True` before you return,
as `require_confirmation` does, the model may read that result and reply to the
user before the manager has answered.

### Let the decision arrive later

A person may take minutes or days to decide. ADK stores the confirmation request
in the session, so the decision can arrive in a later `run_async` call, even
from another process, as long as your session service keeps the session. If you
turn on resumability with `resumability_config` on your `App`, ADK also treats
the confirmation request as a point where the invocation pauses, and the
decision resumes the same invocation. The [App guide](../../apps/app/index.md)
describes that option.

## Limitations

The following limits apply to tool confirmation today:

- ADK marks the `ToolConfirmation` class as experimental, so its interface may
  change. ADK emits a one-time `UserWarning` saying that the feature is enabled.
- The `request_confirmation` method does not stop your function. It records the
  request and returns, and the lines after it still run. Return right after you
  call it, and call it before anything with a side effect.
- The person cannot change the call's arguments through this mechanism, because
  ADK runs the call with the arguments recorded in the session. If you need them
  to adjust a request, ask for a `payload` and have the tool apply it.
- A message that carries a decision cannot also carry text. If it does,
  `run_async` raises a `ValueError`, so send the decision and any new user text
  as separate messages.

## Related samples

- [Tool confirmation](../../../../contributing/samples/hitl/tool_confirmation/agent.py)
  puts both forms in one agent, with `require_confirmation=True` on a
  destructive tool and a hand-written gate that only asks above an amount.
- [Human tool confirmation](../../../../contributing/samples/hitl/human_tool_confirmation/agent.py)
  shows a callable `require_confirmation`, a confirmation that carries a
  payload, and an app with resumability turned on.
