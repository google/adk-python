# AgentTool

The `AgentTool` class lets one agent call another agent in the same way that it
calls a function tool. The parent agent's model sends the wrapped agent a
request, the wrapped agent works on it until it has an answer, and that answer
comes back to the parent as the result of the tool call.

ADK has a second way to call an agent as a tool, and it recommends that way
whenever the agent supports it. If you set `mode="single_turn"` on an `LlmAgent`
and list it in the parent's `sub_agents`, ADK wraps the agent as a tool for you
and runs it inside the parent's session. The `AgentTool` docstring discourages
using `AgentTool` directly for that reason. The next section compares the two
forms so that you can choose one before you read further, and
[LlmAgent single-turn mode](../../agents/llm_agent/single_turn.md) describes the
single-turn form in full.

## AgentTool or a single-turn sub-agent

A single-turn sub-agent and an `AgentTool` both appear to the parent's model as
a tool, but they behave differently in the following ways:

- A single-turn sub-agent runs inside the parent's session, and ADK records its
  events there in a branch of their own. An agent wrapped in `AgentTool` runs in
  a separate session that ADK discards after each call.
- A single-turn sub-agent sees only its tool input by default, but you can let
  it read the parent's conversation by setting `include_contents="default"` on
  it. An agent wrapped in `AgentTool` never sees the conversation.
- Single-turn mode works only for agent classes that accept
  `mode="single_turn"`, such as `LlmAgent` and `ManagedAgent`. A custom
  `BaseAgent` subclass has no `mode` field, so `AgentTool` is the way to call it
  as a tool.
- The `skip_summarization`, `include_plugins` and `propagate_grounding_metadata`
  options exist only on `AgentTool`.

If the specialist needs to hold a conversation with the user until its job is
done, neither form fits. A task-mode sub-agent does that, and
[LlmAgent task mode](../../agents/llm_agent/task.md) describes it.

## Introduction

As an application grows, you often want a specialist, an agent with its own
instruction and tools that does one job well. The `AgentTool` class lets you
give that specialist to a parent agent as one more entry in the parent's `tools`
list. The parent's model decides when to call it, reads its answer, and then
continues the conversation. The user talks to the parent the whole time, and
control never passes to the specialist.

The wrapped agent runs in a separate session that ADK creates for the call and
discards when the call ends. It can read and change the parent's session state,
and it can load and save the parent's artifacts, but it does not see the
conversation. Everything it needs to know has to arrive in the request or in
session state. Only its final answer comes back to the parent, so the parent's
model and the parent's session record one function call and one function
response, however many steps the wrapped agent took to get there.

Calling a specialist as a tool, in either form, keeps the specialist's tools out
of the parent's tool list, and it lets the specialist start from a clean
context, with only the request in front of it, even when the parent's
conversation is long.

## Get started

The following example gives a customer support agent an order specialist. The
specialist has a function tool for looking up orders, and the support agent
calls the specialist whenever the customer asks about an existing order.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

def lookup_order(order_id: str) -> dict:
  """Returns the status and items of an order."""
  return {"order_id": order_id, "status": "shipped", "items": ["kettle"]}

order_agent = LlmAgent(
    name="order_agent",
    description="Answers questions about a customer's existing orders.",
    instruction="Use lookup_order to answer the question you are given.",
    tools=[lookup_order],
)

root_agent = LlmAgent(
    name="support_agent",
    instruction=(
        "Help the customer. Call order_agent for anything about an existing"
        " order."
    ),
    tools=[AgentTool(agent=order_agent)],
)
```

You run the parent the same way as any other agent, for example with `adk run`
or `adk web`. When the support agent calls `order_agent`, the session records
the function call and the function response. The order agent's own call to
`lookup_order` happens inside its separate session and does not appear in the
parent's history.

The tool takes its name from the wrapped agent's `name` and its description from
the agent's `description`. The parent's model reads that description when it
decides whether to call the tool, so write it the way you would write a
function's docstring, saying what the agent does and when to use it. The
`description` field defaults to an empty string, and an agent without one
reaches the model as a tool that says nothing about itself.

Put the `AgentTool` in `tools`, not the agent itself. If you place an agent
directly in another agent's `tools` list, ADK raises a `ValueError` explaining
that agents cannot be used directly as tools.

## How it works

Each call to the tool starts a new, short run of the wrapped agent, and the call
returns when that run produces its final response.

### What the parent's model sees

If the wrapped agent has no `input_schema`, the tool takes a single required
string argument named `request`, whatever the agent's instruction says. The
parent's model writes its request into that string, and the wrapped agent
receives the string as its user message.

If you give the wrapped agent an `input_schema`, the tool's parameters come from
that Pydantic model instead. ADK validates the arguments that the parent's model
sends against the schema and passes them to the wrapped agent as a JSON
document. Use an input schema when the specialist needs several separate values
and you do not want the parent's model to pack them into one sentence.

### What the wrapped agent sees

The wrapped agent runs with its own session and a mix of new and shared
services:

- The session is new, belongs to the same user as the parent's session, and
  starts with a copy of the parent's session state. ADK leaves out keys that
  begin with `_adk` or `temp:_adk`, because it uses them for its own
  bookkeeping. Placeholders in the wrapped agent's instruction, such as
  `{customer_name}`, resolve against that copy.
- The parent's `temp:` keys are passed to the wrapped agent's invocation too, so
  its instruction, callbacks and tools can read them.
- The only message in the session is the request from the parent's model. ADK
  includes none of the parent's earlier turns.
- The wrapped agent shares the parent's artifacts. When it loads or saves an
  artifact, it reads from or writes to the parent's session.
- The credential service is the parent's, so a tool that needs a user
  credential finds the one the parent already obtained.
- The memory service is new and empty, so memory tools such as `load_memory`
  find nothing inside the wrapped agent.
- The parent's plugins run for the wrapped agent too, unless you set
  `include_plugins=False`.

The wrapped agent also inherits the parent's `RunConfig`, with a few
adjustments. It always runs without streaming, so its answer arrives in one
piece even when the parent streams. The `max_llm_calls` limit applies to the
wrapped agent's run on its own, counted separately from the parent's calls.

### What comes back

When the wrapped agent finishes, the parent receives the text of its final
response, with any thought parts removed. If the wrapped agent has an
`output_schema`, ADK validates that text against the schema and returns the
parsed value, which is a dictionary when the schema is a Pydantic model. If the
wrapped agent produced no response at all, the parent receives the last error
message from the run, or an empty string.

State changes travel back as well. ADK copies every change that the wrapped
agent makes to session state into the parent's state while the wrapped agent
runs. Setting `output_key` on the wrapped agent therefore stores its answer in
the parent's session state, where later instructions and tools can read it.
`temp:` state only travels one way: the parent's `temp:` keys reach the wrapped
agent, but `temp:` writes made inside the wrapped agent are trimmed when its
events are recorded and are not copied back to the parent's state.

## Configuration options

The `agent` argument is the only one you must pass. The other arguments change
what happens to the wrapped agent's answer and what runs alongside the wrapped
agent.

### Return the answer without a second model call

By default, the parent's model reads the tool result and writes the reply to
the user in its own words. When you set `skip_summarization=True`, the parent's
turn ends with the tool result instead. The function response becomes the final
response of the turn, and ADK adds the result to it as a text part so that a
chat interface displays it.

Use this option when the wrapped agent already produces the final text the user
should see, such as a formatted report, and a second pass through the parent's
model would only rephrase it. The trade-off is that the parent's model does not
see the result before the turn ends, so it cannot comment on it or act on it in
that turn.

### Keep the parent's plugins away from the wrapped agent

By default, every plugin that you register on the parent's `Runner` or `App`
also runs for the wrapped agent's model calls and tool calls.

Set `include_plugins=False` to run the wrapped agent with no plugins. Choose
this when a logging or metrics plugin should count only the parent's activity,
or when a plugin that enforces a policy on the parent should not apply to the
specialist.

### Carry grounding metadata back to the parent

When you set `propagate_grounding_metadata=True`, ADK keeps the grounding
metadata from the wrapped agent's last response in the parent's temporary
session state. In the current release, the parent attaches that metadata to its
own response only when one of the parent's tools is named
`google_search_agent`. For any other wrapped agent, the option has no visible
effect.

## Advanced applications

The basic setup passes one request string to an `LlmAgent` and gets text back.
You can also pass structured values or move a large toolset into the
specialist.

### Pass structured input and output

When the specialist needs several separate values, or when the parent needs a
predictable result, give the wrapped agent an `input_schema` and an
`output_schema`. In the following example, the parent's model calls
`translator` with two named arguments and receives a dictionary back.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from pydantic import BaseModel

class TranslationRequest(BaseModel):
  text: str
  target_language: str

class Translation(BaseModel):
  translated_text: str

translator = LlmAgent(
    name="translator",
    description="Translates text into a target language.",
    instruction="Translate the text into the requested language.",
    input_schema=TranslationRequest,
    output_schema=Translation,
)

root_agent = LlmAgent(
    name="assistant",
    instruction="Use the translator when the user asks for a translation.",
    tools=[AgentTool(agent=translator)],
)
```

The tool declaration lists `text` and `target_language` as its parameters, and
the result that reaches the parent is a dictionary with a `translated_text` key.

### Keep a large toolset out of the parent

ADK declares every tool in an agent's tool list to the model on every model
call, so each one costs prompt space. If one job needs a large set of tools,
such as everything an MCP server publishes, you can give those tools to a
specialist and give the specialist to the parent through `AgentTool`. The
parent's model then sees one tool in place of the whole set.

Because ADK closes the wrapped agent's toolsets after each call, an MCP toolset
on the wrapped agent reconnects to its server every time the parent calls the
specialist. Expect every call to include that connection time.

## Limitations

- Each call starts with an empty session, because ADK discards the wrapped
  agent's session afterwards. To carry information from one call to the next,
  have the wrapped agent write it to session state, for example with
  `output_key`.
- The parent's session records only the call and the response, so you cannot
  inspect the wrapped agent's steps there. Plugins still run for the wrapped
  agent by default, so a logging plugin can show you what it did.

## Related samples

- [AgentTool with grounding metadata](../../../../contributing/samples/tools/agent_tool_with_grounding_metadata/agent.py)
  wraps a Vertex AI Search agent as a tool, alongside an ordinary function
  tool.
- [MCP server as an AgentTool](../../../../contributing/samples/mcp/mcp_in_agent_tool_stdio/agent.py)
  gives an MCP toolset to a specialist so that its tools stay out of the
  parent's tool list.
