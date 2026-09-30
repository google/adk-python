# AgentCardBuilder

`AgentCardBuilder` writes the agent card a client reads before it sends your
agent any work, deriving it from an ADK agent or a `Workflow` you already have.
If [`to_a2a`](../agent_to_a2a/index.md) is how you put your agent on the
network, this is how you control what everyone out there sees of it.

## Introduction

The card is a small public document: the agent's name and description, the URL
to send tasks to, the protocol capabilities the server supports, and a list of
skills. An A2A client reads it and picks an agent on that basis.
`AgentCardBuilder` derives all of it from the agent object itself, so you never
write the document by hand and it cannot drift away from the tools the agent
actually has.

[`to_a2a`](../agent_to_a2a/index.md) calls this class on your behalf, so a
deployment that is content with the derived card never has to construct one. You
come here directly when you need a field `to_a2a` cannot set. It only ever passes the
agent and the RPC URL, so the provider, the version, the capabilities, the
security schemes, and the documentation URL all stay at their defaults. To set
any of them, build the card here and hand it back through
`to_a2a(agent, agent_card=...)`.

The import path is nested and not re-exported:

```python
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
```

## Get started

This example builds an agent card by hand for an agent that will be served by
`to_a2a`, setting the provider and version that the automatic card leaves empty.
Note that `build()` is a coroutine, so it has to be awaited before you can pass
the card anywhere.

```python
import asyncio

from a2a.types import AgentProvider
from google.adk import Agent
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.a2a.utils.agent_to_a2a import to_a2a


def roll_die(sides: int) -> int:
  """Roll a die with the given number of sides and return the result."""
  ...


root_agent = Agent(
    name="dice_agent",
    description="Rolls dice with any number of sides.",
    instruction="Use the roll_die tool to roll the dice the user asks for.",
    tools=[roll_die],
)

card = asyncio.run(
    AgentCardBuilder(
        agent=root_agent,
        rpc_url="https://agents.example.com/dice/",
        provider=AgentProvider(
            organization="Example Corp", url="https://example.com"
        ),
        agent_version="2.1.0",
    ).build()
)

a2a_app = to_a2a(root_agent, agent_card=card, rpc_path="dice")
```

Every argument is keyword-only, `agent` included. `AgentCardBuilder(root_agent)`
raises `TypeError`.

## How it works

The constructor only stores the values you gave it and fills in defaults. All
the work happens in `build()`, and the card it returns is assembled in three
stages.

1.  **Primary skills.** An `LlmAgent` gets one skill named `model` for the agent
    itself, then one skill per tool, then a `planning` skill if the agent has a
    planner and a `code-execution` skill if it has a code executor. Anything
    that is not an `LlmAgent`, such as a `SequentialAgent`, a `Workflow`, or an
    agent class of your own, gets one skill for the agent plus a `sub-agents`
    skill summarizing what it orchestrates.
2.  **Child skills.** The same derivation runs over each immediate child, which
    means the `sub_agents` of an agent, or the graph nodes of a `Workflow` with
    `START` excluded. If a child's skills fail to build, the failure is logged
    and that child is skipped, rather than taking the whole card down with it.
3.  **Assembly.** The two lists are concatenated into the card's `skills`, and
    everything the constructor was given fills in the rest of it.

Anything that raises during those steps is re-raised as
`RuntimeError("Failed to build agent card for {name}: ...")`, with the original
exception as its `__cause__`.

### Where the RPC URL lands

`rpc_url` goes into a different field depending on which a2a-sdk is installed,
which matters the moment you inspect a served card. On a2a-sdk 1.x the card has
no top-level `url`; the value appears as `supportedInterfaces[0].url` with a
`protocolBinding` of `JSONRPC`. On 0.3.x it is the top-level `url` field, with
`preferredTransport` beside it. In both cases a trailing slash is stripped, so
`rpc_url="https://agents.example.com/dice/"` is published as
`https://agents.example.com/dice`.

### What the skills look like

For the `dice_agent` above, the card carries two skills:

| `id` | `name` | `tags` | Description comes from |
| :--- | :--- | :--- | :--- |
| `dice_agent` | `model` | `["llm"]` | the agent's `description` |
| `dice_agent-roll_die` | `roll_die` | `["llm", "tools"]` | the tool's description |

The agent's `instruction` is never used, and that is on purpose. The card is a
discovery document served without authentication, so the description comes from
the agent's own public `description` field and from nothing else. An agent with
no description at all falls back to a generic string such as
`"An LLM-based agent"`, which tells a prospective caller very little, so it is
worth writing a real one.

Skill examples come only from a declared `ExampleTool` in the agent's tools.
They are never mined out of the instruction.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `agent` | `BaseAgent \| Workflow` | *required* | The unit to describe. Keyword-only. |
| `rpc_url` | `str \| None` | `'http://localhost:80/a2a'` | Where clients should send tasks. |
| `capabilities` | `AgentCapabilities \| None` | `AgentCapabilities()` | Protocol features the server supports. |
| `doc_url` | `str \| None` | `None` | Human documentation, published as `documentationUrl`. |
| `provider` | `AgentProvider \| None` | `None` | Who publishes the agent. Omitted from the card when unset. |
| `agent_version` | `str \| None` | `'0.0.1'` | The version string in the card. |
| `security_schemes` | `dict[str, SecurityScheme] \| None` | `None` | Named auth schemes a client must satisfy. |

**`rpc_url`.** The default is a placeholder, not a working address. Leaving it
unset produces a card that tells clients to call `http://localhost:80/a2a`,
which is almost never right. `to_a2a` always passes one, so this only bites when
you build a card yourself.

**`capabilities`.** The default is an empty `AgentCapabilities()`, which the
card renders as `streaming: false` and `pushNotifications: false`. That default
does real work rather than sitting there as documentation, because an A2A server
refuses a `message/stream` request with `UnsupportedOperationError` when the
card does not advertise streaming. If you want clients to be able to stream,
pass `AgentCapabilities(streaming=True)`. Building the card here is the only way
to do that, since `to_a2a` has no argument for it.

**`agent_version`.** Defaults to `0.0.1` for every agent, so a card that has
never been configured is indistinguishable from a first release. Set it from
whatever your deployment already uses as a version.

**`security_schemes`.** Published so a client knows what credentials to present.
Declaring a scheme here does not enforce anything; the server does not check
incoming requests against it.

## Advanced applications

A `Workflow` is the case where the derived card differs most from what you might
expect, because the card is assembled out of the graph rather than out of a tool
list.

### Describe a Workflow

*   **Problem solved**: you are serving a graph `Workflow` over A2A and want the
    card to say what the graph does.
*   **Implementation**: pass the workflow. Each node becomes a child skill, and
    an orchestration skill lists them, so node docstrings become the public
    description of your pipeline.

```python
from google.adk import Workflow
from google.adk.workflow import START


def normalize(node_input: str) -> str:
  """Normalizes the incoming text."""
  ...


def summarize(node_input: str) -> str:
  """Summarizes the normalized text."""
  ...


pipeline = Workflow(
    name="pipeline",
    description="Cleans and summarizes a document.",
    edges=[(START, normalize, summarize)],
)

card = asyncio.run(
    AgentCardBuilder(agent=pipeline, rpc_url="https://agents.example.com/pipeline/").build()
)
```

That card carries four skills. The `pipeline` skill's description is the
workflow's own description with a node listing appended: "Cleans and summarizes
a document. This workflow orchestrates the following nodes: normalize:
Normalizes the incoming text; summarize: Summarizes the normalized text."
A `pipeline-sub-agents` skill repeats the listing, and each node gets a skill
carrying its docstring. Your node docstrings are therefore published to anyone
who fetches the card, so write them as though a stranger will read them, because
one will.

## Limitations

*   **The card is a snapshot.** `build()` reads the agent as it is at that
    moment. An agent whose tool list changes at runtime still advertises the
    list it had when the card was built, and `to_a2a` builds it once per
    process, at server startup.
*   **Child skill ids repeat the child's name.** A sub-agent's skill id is
    formed as `{child_name}_{skill_id}`, and the child's own agent skill already
    uses the child's name as its id, so a sub-agent called `child_a` produces
    the id `child_a_child_a`. It is stable and unique, if ugly in a published
    card. Workflow nodes double the same way: a node called `normalize` gets
    the skill id `normalize_normalize`.
*   **Only immediate children are walked.** The recursion is one level deep, so
    a grandchild agent contributes no skills.
*   **`security_schemes` is descriptive only.** Nothing validates requests
    against it.
*   **Experimental.** The class is decorated `@a2a_experimental`, so
    constructing one emits a `UserWarning`. Set
    `ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS` to silence it.

## Related samples

*   [A2A root agent](../../../../../contributing/samples/a2a/a2a_root) serves an
    agent with `to_a2a`, using the card this class builds by default.
*   [A2A basic](../../../../../contributing/samples/a2a/a2a_basic) takes the
    other route entirely, with a hand-written `agent.json` card instead of a
    derived one.

## Related guides

*   [to_a2a](../agent_to_a2a/index.md) is the caller that builds a card for you,
    and it explains how to hand it one you built here instead.
*   [A2aAgentExecutor](../../executor/a2a_agent_executor/index.md) handles the
    requests that arrive from clients your card persuaded.
*   `RemoteA2aAgent` is on the other
    side of the exchange, reading the card you publish.
