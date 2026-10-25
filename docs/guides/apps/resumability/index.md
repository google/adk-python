# ResumabilityConfig

The `ResumabilityConfig` class is the `App` setting that lets an invocation
continue from where it stopped. When you turn it on, ADK records in the session
how far each agent got, and it uses that record when you resume an invocation
that paused on a long-running tool call or stopped partway through.

## Introduction

Some tools cannot finish inside the turn that calls them. A reimbursement waits
for a manager's approval, and a data export can run for an hour. ADK models
these with `LongRunningFunctionTool`. The function starts the work and returns
at once, usually with a status such as `{"status": "pending"}`. ADK gives that
status to the model as the call's function response, so the model still answers
the user before the turn ends, and the call stays open. Your application sends
the real result back later as another function response for the same call,
possibly from a different process and days later. A function that returns
nothing answers nothing, so the turn stops at the call instead.

The call stays open for that later response whether or not you set
`ResumabilityConfig`. What the config adds is a record of progress. While the
invocation runs, agents write checkpoint events into the session that say where
they are and whether they have finished. When you resume, ADK reads those
checkpoints back. The record makes a difference in the following situations:

*   The long-running call was made by a sub-agent. With the config on, ADK
    sends the function response back to the sub-agent that made the call.
    Without it, ADK routes the next turn by its usual rules, and the response
    can reach an agent that has no call waiting for it.
*   The process stopped partway through a turn, for example after the model
    asked for a tool but before the tool ran. With the config on, resuming runs
    that tool call before the agent asks the model for anything new.
*   The root agent is neither an `LlmAgent` nor a `Workflow`, for example a
    `SequentialAgent`. For those root agents, resuming requires the config.
    Without it, a message with a function response starts a new invocation, and
    a call with only an `invocation_id` raises `ValueError`.

A single `LlmAgent` or a `Workflow` at the root can match a function response to
the invocation that is waiting for it without the config, so a simple approval
flow with one agent works either way. The config costs extra events in the
session, one each time an agent records a checkpoint, and it applies to every
agent in the app.

## Get started

The following agent asks a manager to approve large reimbursements. The example
wraps the `ask_for_approval` function in `LongRunningFunctionTool`, so when the
model calls it, the function returns a pending status with a ticket id, the
model tells the employee that the request is waiting, and the call stays open
for the manager's answer. A real version of the function also sends the request
to a manager before it returns.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.tools import LongRunningFunctionTool

def ask_for_approval(purpose: str, amount: float) -> dict[str, str]:
  """Asks a manager to approve a reimbursement."""
  return {"status": "pending", "ticket_id": "reimbursement-001"}

root_agent = LlmAgent(
    name="reimbursement_agent",
    instruction=(
        "Handle employee reimbursements. Amounts over $100 need manager"
        " approval, so call ask_for_approval and wait for its result before"
        " you tell the employee anything."
    ),
    tools=[LongRunningFunctionTool(func=ask_for_approval)],
)

app = App(
    name="reimbursement_app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

To try it without writing a driver, export the `App` as `app` in your agent
module and start `adk web`. When the agent pauses, you can send the final
result back from the web UI as a function response for the pending call.

### Detect the pause

In your own application, you run the turn with `run_async` and watch for the
pause. The event that carries the long-running call lists the call's id in
`event.long_running_tool_ids`. The following loop records that call.

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

runner = Runner(
    app=app,  # the App from the previous example
    session_service=InMemorySessionService(),
    auto_create_session=True,
)

# Inside an async function:
message = types.Content(
    role="user",
    parts=[types.Part(text="Reimburse $200 for conference travel.")],
)
pending_call = None
async for event in runner.run_async(
    user_id="user-1", session_id="session-1", new_message=message
):
  for call in event.get_function_calls():
    if call.id in (event.long_running_tool_ids or set()):
      pending_call = call
```

Let the loop run to the end instead of breaking out when you find the call. The
tool function runs after ADK yields the event that carries its call, and
stopping the iteration early cancels whatever the invocation still has to do,
which can include running the tool.

### Send the result back

When the approval arrives, call `run_async` again with the same user and
session, and with a message whose only part is a function response. Give the
response the same `id` and `name` as the call it answers.

```python
# Inside the same async function as the previous example:
approval = types.Content(
    role="user",
    parts=[
        types.Part(
            function_response=types.FunctionResponse(
                id=pending_call.id,
                name=pending_call.name,
                response={"status": "approved", "approver": "manager-7"},
            )
        )
    ],
)
async for event in runner.run_async(
    user_id="user-1", session_id="session-1", new_message=approval
):
  if event.content and event.content.parts and event.content.parts[0].text:
    print(event.author, event.content.parts[0].text)
```

ADK matches the response's id to the call in the session, continues the
invocation that made the call, and delivers the response to it. The model does
not see a new question from the user. It sees the result of the tool call it
made, and it writes its reply to the employee.

## What your application stores

Between the pause and the resume, your application is the only part of the
system that knows which conversation is waiting for which result. Store the
following values when you detect the pause:

*   The `user_id` and `session_id`, so that you can find the session again.
*   The `id` of the long-running call, which ADK uses to find the paused
    invocation.
*   The `name` of the call, which goes on the function response.
*   The `invocation_id` of the paused invocation, which every event of that
    turn carries. You need it only to resume without a function response, as
    described in the next section.

Everything else that ADK needs, including the user's original message and each
agent's checkpoints, is already in the session. The session therefore has to
last as long as the pause does. The `InMemorySessionService` class loses
everything when the process exits, so an approval that can take longer than the
life of one process needs a persistent session service such as
`DatabaseSessionService`.
The [session guide](../../sessions/session/index.md) compares the services.

If you pass `invocation_id` together with a function response, ADK checks it
against the invocation that made the call. When the two differ, ADK logs a
warning and resumes the invocation that made the call.

## Resume an invocation that stopped partway

An invocation can also stop because the process running it crashed or was shut
down. To continue it, call `run_async` with its `invocation_id` and no
`new_message`.

```python
# Inside an async function, with the runner from the earlier example:
async for event in runner.run_async(
    user_id="user-1",
    session_id="session-1",
    invocation_id=invocation_id,  # the invocation id you stored
):
  print(event.author, event.content)
```

ADK finds the user message that started the invocation and runs the agent on it
again, with the events the invocation had already produced as its history.
With the config on, the agent checks those events before it calls the model. It
first runs any tool call that the model asked for but that never ran. If a
long-running call is still waiting for its result, the invocation stays paused.
The agent calls the model only when nothing is outstanding.

## How it works

The config changes what ADK writes while an invocation runs and what it reads
back when you resume. The sequence spans two calls to `run_async`:

1.  While the first call runs, agents write checkpoint events into the session.
    These events carry `actions.agent_state` or `actions.end_of_agent` and have
    no content, so they do not appear as turns in the conversation. A workflow
    agent records which sub-agent it has reached, and an agent records the end
    of its own run only when it finished with no call left waiting, so a turn
    that ends on a long-running call writes no end-of-agent checkpoint.
2.  When the model calls a long-running tool, the tool function runs. If it
    returns a value, ADK gives that value to the model as the function
    response, and the model replies before the turn ends. The event that
    carries the call lists the call's id in `long_running_tool_ids`, and the
    call waits there for its real result.
3.  The second call to `run_async` brings the function response. ADK finds the
    invocation that owns the call by matching ids, reuses that invocation's id,
    and recovers the user message that started it.
4.  In an app with several agents, ADK routes the response to the agent that
    made the call.
5.  That agent receives the response for its waiting call. It checks its events
    before it calls the model again, as described in the previous section.

Step 2 works with or without the config. The checkpoints in step 1, the routing
in step 4, and the checks in step 5 are what the config adds.

## Configuration options

The `ResumabilityConfig` class has an `is_resumable` field, which defaults to
`False`. Setting it to `True` turns resumability on for every agent in the app.
You cannot turn it on for one agent only, so sub-agents that you did not intend
to make resumable also write checkpoints and add events to the session.

## Advanced applications

The same pause and resume works when you serve the app over HTTP instead of
calling `Runner` yourself.

### Resume through the API server

When you serve the app with `adk api_server`, a client resumes by posting to
`/run` or `/run_sse` with the same `appName`, `userId`, and `sessionId` as the
paused turn. The request's `newMessage` holds the function response, with the
call's `id` and `name`. To continue an invocation that stopped partway, the
client sends `invocationId` and leaves out `newMessage`. The request body uses
camelCase field names, like the rest of the API.

## Limitations

*   A resumed tool may run more than once. ADK runs a resumed tool call at
    least once and possibly more often, so running a resumable tool twice must
    have the same effect as running it once. A tool that charges a card or
    sends an email needs its own check, so that a second run does nothing.
*   In-memory state does not survive the pause. Anything a tool or callback kept
    in a local variable, a module global, or an open connection is gone when the
    invocation resumes. Keep what you need in session state, in artifacts, or in
    your own storage.
*   A resuming message carries function responses and nothing else. With an
    `LlmAgent` or a `Workflow` at the root, a message that mixes text with a
    function response raises `ValueError`. Send the tool result first, and send
    the user's next question as its own turn.
*   All function responses in one message must answer calls from the same
    invocation, and each must carry the id of a call that exists in the
    session. Otherwise `run_async` raises `ValueError`.
*   The `ResumabilityConfig` class is experimental. Constructing it emits a
    `UserWarning` once per process, and setting the environment variable
    `ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS=true` turns the warning off. Its
    shape may change in a later release.

## Related samples

*   [Human tool confirmation](../../../../contributing/samples/hitl/human_tool_confirmation/agent.py)
    sets `ResumabilityConfig` on an agent whose tools pause for a manager's
    confirmation before they run.
*   [A2A human in the loop](../../../../contributing/samples/a2a/a2a_human_in_loop/agent.py)
    pauses in a remote sub-agent and sets the config so that ADK routes the
    approval back to that sub-agent.
*   [Human in the loop](../../../../contributing/samples/hitl/human_in_loop/main.py)
    is a single reimbursement agent without the config, whose driver captures
    the long-running call and sends the approval back.
