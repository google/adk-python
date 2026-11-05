# EventActions

The `EventActions` class holds the part of an `Event` that ADK acts on. A tool
or callback sets its fields to end the turn at a tool result or to hand the
conversation to another agent, and your application reads the same fields on
the events it receives to see what each step changed.

## Introduction

Most of what an agent produces is content, meaning the text and function calls
that the model reads on the next turn. Some results are instructions for ADK
instead. A tool might want the turn to end with its own result, shown to the
user as it is, without another model call. A tool inside a loop might want the
loop to stop. Neither of those belongs in the text the model reads.

An `EventActions` object is where those instructions go. Every `Event` has one
in `event.actions`, and it is never `None`. Inside a tool or a callback you do
not create one yourself. The context you receive, a `ToolContext` in a tool or
a `CallbackContext` in a callback, exposes the actions of the event your code
is producing as `ctx.actions`. In a tool you usually name the parameter
`tool_context`, so the same object is `tool_context.actions`. You set fields on
that object, and ADK attaches it to the event.

The fields fall into groups according to who sets them. You set a few of them
directly as signals. The context fills in several others when you call a method
on it, such as writing to `ctx.state`. ADK writes the rest for its own
features, and you read them without setting them.

Import the `EventActions` class from `google.adk.events`.

## Get started

The following tool looks up an order and sets `skip_summarization`, so that the
turn ends with the tool's result. Without the flag, ADK sends the result back to
the model so that it can describe the order in a sentence. With the flag, the
tool result is the final response of the turn, and ADK does not call the model
again.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

def get_order_status(
    order_id: str, tool_context: ToolContext
) -> dict[str, str]:
  """Looks up the shipping status of an order."""
  tool_context.actions.skip_summarization = True
  return {"order_id": order_id, "status": "shipped", "eta": "2026-09-24"}

root_agent = LlmAgent(
    name="order_agent",
    instruction="Use get_order_status to answer questions about orders.",
    tools=[get_order_status],
)
```

Your application then reads the result from the final event of the turn. The
`event.is_final_response()` method returns `True` for the function response
event, because the tool set `skip_summarization` on it, and the result is in
the event's function responses.

```python
# Inside an async function, where runner is a Runner built for root_agent and
# message is the user's types.Content:
async for event in runner.run_async(
    user_id="user-1", session_id="session-1", new_message=message
):
  if event.is_final_response():
    for response in event.get_function_responses():
      print(response.name, response.response)
```

The result reaches your code as structured data, which suits an application
that renders an order card or a table itself. A chat interface that displays
only text parts shows nothing for this turn, because the model wrote no reply.
For a plain function tool like this one, ADK does not add a text part for you.

## How it works

The object you change through `ctx.actions` is the object that ends up on the
event. The names `ToolContext` and `CallbackContext` both refer to the same
class, `Context`, whose `actions` property returns the actions of the event
that ADK is building. Whatever you set before your tool returns is on the
function response event when ADK yields it.

That event then passes up through the agents that enclose the agent that
produced it, and each of them reacts to the fields it handles. ADK does not
check that some agent handled a field. A field set where no agent reads it has
no effect and produces no warning.

The following details decide whether an action reaches the agent that should
react to it:

*   A callback can lose its actions. In a `before_agent_callback` or an
    `after_agent_callback`, ADK emits an event only when the callback returns
    content or writes to `ctx.state`. ADK drops an action set by a callback
    that does neither, because no event carries it, so set actions from a tool
    when you can.
*   When the model calls several tools in one step, ADK merges their responses
    into one event and merges their actions as well. A field set by one tool
    then applies to the whole merged event. If one of three tools sets
    `skip_summarization`, the turn ends after all three results, and the model
    does not describe any of them.

## Configuration options

The `EventActions` class has many fields, and of those you set only
`skip_summarization`, `escalate`, and `transfer_to_agent` yourself. A context
method or ADK fills in the others.

### End the turn at a tool result: `skip_summarization`

A tool call normally takes two model calls. The model asks for the tool, ADK
runs it, and ADK sends the result back to the model so that it can write a
reply. Setting `ctx.actions.skip_summarization = True` skips the second model
call. The `Event.is_final_response()` method returns `True` for any event with
the flag set, and the agent's turn ends at that event.

Set it when the tool's output is already what the user should get, and a
sentence written by the model would only add latency or change the wording.
Note the following behavior of the flag:

*   The user may see no text. ADK adds the result as a visible text part only
    when the tool is an `AgentTool`. For any other tool, the result exists only
    as a function response, and an interface that does not display function
    responses shows an empty turn.
*   Even for an `AgentTool`, ADK adds no text part when the result contains an
    `error` key, or when the result is `None`, an empty string, or
    `{"result": None}`.
*   Several built-in tools set the flag for you, which can end a turn earlier
    than you expect. The `exit_loop` and `get_user_choice` tools always set
    it. A `FunctionTool`, an MCP tool, the bash tool, and the computer use tool
    set it while they wait for the user to confirm a call. A wrapped LangChain
    tool sets it when the LangChain tool has `return_direct` enabled.

For an `AgentTool`, pass `skip_summarization=True` to its constructor, and the
sub-agent's answer becomes the reply as the sub-agent wrote it.

### Leave a loop early: `escalate`

Setting `ctx.actions.escalate = True` asks the enclosing loop to stop. The
following agents read the field:

*   A `LoopAgent` checks every event that its sub-agents yield. When it sees
    `escalate`, it lets the current sub-agent finish and then stops, without
    starting another pass. The `max_iterations` argument still applies, so
    `escalate` is an early exit below that limit. A `LoopAgent` does not check
    which agent escalated, so an escalation inside a nested `LoopAgent` also
    ends every `LoopAgent` above it.
*   A `ParallelAgent` stops when one of its direct sub-agents escalates. It
    cancels the branches that are still running, so they do not finish.

No other agent reads the field. In a `SequentialAgent` the remaining sub-agents
still run, an `LlmAgent` with no loop above it ends its turn as usual, and a
`Workflow` ignores the field.

The following review loop stops as soon as the reviewer approves the draft. The
built-in `exit_loop` tool, exported from `google.adk.tools`, sets `escalate` and
`skip_summarization`, so you can give it to the reviewer instead of writing your
own tool.

```python
from google.adk.agents import LlmAgent
from google.adk.agents import LoopAgent
from google.adk.tools import exit_loop

reviewer = LlmAgent(
    name="reviewer",
    instruction=(
        "Critique the latest draft. If it needs no further changes, call"
        " exit_loop. Otherwise, write your critique."
    ),
    tools=[exit_loop],
)

review_loop = LoopAgent(
    name="review_loop",
    sub_agents=[reviewer],
    max_iterations=5,
)
```

The `LoopAgent` and `ParallelAgent` classes are deprecated in favor of
`Workflow`, and constructing either one emits a `DeprecationWarning`. In a
`Workflow`, a loop is a conditional edge that leads back to an earlier node. A
node leaves the loop by setting `ctx.route` or by yielding `Event(route=...)`
with a route that matches a different edge. The
[workflow graph guide](../../workflow/graph/index.md) explains routes and
conditional loops. The deprecation notice also states that you cannot yet use a
`Workflow` as a sub-agent of an `LlmAgent`, so a loop inside an `LlmAgent`'s
sub-agents still needs `LoopAgent`.

### Hand the conversation to another agent: `transfer_to_agent`

Setting `ctx.actions.transfer_to_agent` to the name of another agent in the same
agent tree hands the conversation to that agent. When an `LlmAgent` has agents
it can transfer to, such as its sub-agents, ADK adds a `transfer_to_agent` tool
to each model request, together with instructions that describe each target.
When the model calls that tool, the tool sets `transfer_to_agent` to the
target's name and `transfer_reason` to the reason the model gave, and the
target agent takes over.

A tool of your own can also set the field, to hand off from code, for example
after it checks the user's account type. If your app uses resumability, note
that when ADK resumes an invocation, it recognizes only transfers made by the
built-in `transfer_to_agent` tool when it decides which sub-agent to continue.

The [LlmAgent guide](../../agents/llm_agent/index.md) covers the settings that
restrict transfers, `disallow_transfer_to_parent` and
`disallow_transfer_to_peers`.

### Fields that context methods fill in

The following fields record what a step changed. The context fills in each one
when you call the matching method, so call the method instead of assigning the
field:

*   The `state_delta` field holds the session state changes. Writing
    `ctx.state["key"] = value` records the change here. The
    [State guide](../../sessions/state/index.md) explains the `app:`, `user:`,
    and `temp:` prefixes.
*   The `artifact_delta` field maps each artifact filename saved in the step to
    its new version. Calling `ctx.save_artifact(...)` fills it in, and the
    [BaseArtifactService guide](../../artifacts/artifact_service/index.md)
    covers versions.
*   The `requested_auth_configs` field holds credential requests, keyed by
    function call id. Calling `ctx.request_credential(auth_config)` adds one,
    and the [authenticated tools guide](../../auth/tool_auth/index.md) explains
    the pause and resume that follow.
*   The `requested_tool_confirmations` field holds confirmation requests, keyed
    by function call id. Calling `ctx.request_confirmation(...)` adds one, and
    the [Context guide](../../agents/context/index.md) shows an example.
*   The `render_ui_widgets` field lists widgets for the UI host to display.
    The `ctx.render_ui_widget(widget)` method adds one, and it raises
    `ValueError` when a widget with the same id is already in the list.
*   The `route` field tells a `Workflow` which outgoing edge to follow. Set
    `ctx.route` in a node, or yield `Event(route=...)`.

You read these fields on the events you receive to find out what changed. The
`event.actions.state_delta` field shows the state keys a step wrote, and
`event.actions.artifact_delta` shows the files it saved.

### Fields that ADK writes

ADK sets the remaining fields for its own features. You see them when you
inspect a session, and you read them without setting them:

*   The `end_of_agent` and `agent_state` fields are checkpoints that agents
    write when [resumability](../../apps/resumability/index.md) is on. ADK uses
    them to decide where a resumed invocation continues, and their
    documentation states that only ADK sets them.
*   The `compaction` field marks a summary written by
    [event compaction](../../apps/events_compaction/index.md).
*   The `rewind_before_invocation_id` field marks a rewind written by
    [Runner.rewind_async](../../runners/runner/rewind.md).
*   The `set_model_response` field carries the structured output from the
    built-in `set_model_response` tool, which ADK uses when an agent has both an
    `output_schema` and tools.

## Limitations

*   ADK stores a value that cannot be serialized as a string. If `state_delta`
    or `agent_state` holds something that cannot be converted to JSON, such as
    a function or a client object, ADK logs a warning and stores the value's
    `repr()`, so that it can still save the rest of the event. The value comes
    back from the session service as a string like `<function f at 0x...>`, and
    ADK raises no exception at the point where you wrote it.
*   Serialized field names use camelCase. The API server sends events with
    camelCase names, so JSON from the server reads `skipSummarization`,
    `stateDelta`, and `transferToAgent`. When you construct `EventActions`, it
    accepts either spelling.
*   The `EventActions` class rejects unknown field names. Passing a field that
    it does not define raises a `ValidationError`, including a misspelled key
    in a payload you validate from outside.
*   The `escalate` field has an effect only in `LoopAgent` and `ParallelAgent`,
    and both are deprecated. A `Workflow` leaves a loop through routes.

## Related guides and samples

*   [Event and NodeInfo](../event/index.md) covers the event that carries these
    actions.
*   [Multi-agent loop config](../../../../contributing/samples/multi_agent/multi_agent_loop_config/README.md)
    is a writer and refiner loop, defined in YAML, whose refiner calls
    `exit_loop` when the critique says the document is done.
*   [GEPA integration](../../../../contributing/samples/integrations/gepa/adk_agent.py)
    has a tool that sets `state_delta`, `skip_summarization`, and `escalate` in
    one environment step.
