# to_mcp_server

Exposes an ADK agent as an MCP server so any MCP host (Claude Code, OpenAI
Codex, an IDE, or any MCP client) can drive it as a single tool. It is the MCP
counterpart of `to_a2a`.

## Introduction

`to_mcp_server` turns a whole ADK agent into a standard
[Model Context Protocol](https://modelcontextprotocol.io/) server. The agent —
its model loop and all of its tools — is registered as a *single* MCP tool named
after the agent. A host that speaks MCP sends a request string and receives the
agent's final response; it never imports ADK and does not see the agent's
individual tools.

This solves the problem of making an ADK agent consumable by harnesses that are
not ADK. Where `to_a2a` publishes an agent over A2A, `to_mcp_server` publishes it
over MCP, so coding agents and IDEs that already speak MCP can delegate a task to
an ADK agent as if it were any other tool. It builds on `Runner` to execute the
agent and returns a `FastMCP` server, leaving the choice of transport (stdio for
local hosts, streamable-http for networked ones) to the caller.

## Get started

Define an agent and expose it. Running the file starts the MCP server on stdio;
an MCP host can also launch it as a subprocess.

```python
import random

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import to_mcp_server


def roll_die(sides: int) -> int:
  """Roll a die with the given number of sides and return the result."""
  return random.randint(1, sides)


dice_agent = LlmAgent(
    name="dice_agent",
    description="Rolls dice with any number of sides and reports the outcome.",
    instruction="Use the roll_die tool to roll the dice the user asks for.",
    tools=[roll_die],
)

# The whole agent becomes one MCP tool named "dice_agent".
server = to_mcp_server(dice_agent)

if __name__ == "__main__":
  server.run(transport="stdio")
```

A host configured to launch this file sees one tool, `dice_agent`, and calls it
with a `request` string; the ADK agent runs its own model and `roll_die` loop and
returns the answer.

## How it works

`to_mcp_server` creates a `FastMCP` server and registers one tool whose handler
runs the agent through a `Runner`. If no `runner` is supplied, one is built with
in-memory session, artifact, memory, and credential services.

On each tool call the handler:

1.  Resolves an ADK session (see below), then wraps the incoming `request` string
    as a user `Content`.
2.  Drives `Runner.run_async` and iterates the event stream.
3.  Forwards intermediate (non-final) text events to the host as MCP **progress
    notifications**, so the host can show the agent working in real time.
4.  Maps the parts of the final response to MCP content blocks and returns them:
    text becomes `TextContent`, inline image data becomes `ImageContent`, audio
    becomes `AudioContent`, and any other inline data becomes an
    `EmbeddedResource`. This is why a multimodal agent's output is preserved
    rather than flattened to text.

### Session continuity

`to_mcp_server` keeps one ADK session per MCP connection, so successive tool
calls on the same connection form a single multi-turn conversation. The mapping
from connection to session is held in a `weakref.WeakKeyDictionary`, so a
session's entry is dropped when its connection is garbage-collected. Over stdio
there is one connection per process, so all calls share one conversation; over
streamable-http each client connection gets its own session.

`to_mcp_server` depends on `Runner`, the agent (`BaseAgent`/`LlmAgent`),
`google.genai.types`, and `mcp.server.fastmcp.FastMCP`; it returns a `FastMCP`
that the caller runs on a transport of their choice.

## Configuration options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *required* | The agent to serve. Its model loop and all of its tools are exposed together as one MCP tool. |
| `name` | `str \| None` | `None` | The MCP server and tool name. Defaults to the agent's name (or `"adk_agent"`). Set it when you want the tool to appear under a name other than the agent's. |
| `instructions` | `str \| None` | `None` | Optional server instructions an MCP host may surface to its model to describe how to use the tool. |
| `runner` | `Runner \| None` | `None` | A pre-built `Runner`. If omitted, one is created with in-memory services. Supply your own to use persistent or custom session, artifact, memory, or credential services — this is the recommended path for a long-lived networked server. |

## Advanced applications

### Serving over the network

*   **Problem solved**: a host on another machine needs to reach the agent.
*   **Implementation**: run the same server with the networked transport:
    `server.run(transport="streamable-http")`. Nothing about the agent changes;
    only the transport differs.

### Bringing your own services

*   **Problem solved**: the default in-memory services do not persist across
    process restarts and are not suited to multi-client production serving.
*   **Implementation**: build a `Runner` with your chosen services and pass it in:
    `to_mcp_server(agent, runner=my_runner)`. The tool then uses those services
    for every call.

### Multimodal responses

*   **Problem solved**: the agent produces images or audio, not just text.
*   **Implementation**: no extra work — non-text parts of the final response are
    returned as `ImageContent`, `AudioContent`, or `EmbeddedResource`, so the
    host receives them alongside any text.

## Limitations

*   **Text input only**: the tool accepts a single `request` string. Passing
    media *into* the agent is not supported through the tool call, because MCP
    tool arguments are JSON that the host's model fills in and hosts do not place
    media in tool arguments. For media input, use MCP resources or elicitation
    instead.
*   **Default services are in-memory**: for a long-lived streamable-http server,
    sessions accumulate with no eviction; inject a `runner` with a persistent or
    cleaning session service. Tool calls on a single connection are expected to
    be sequential, since they share one session.
*   **Experimental**: `to_mcp_server` is `@experimental` and lives behind the
    `mcp` extra; its behavior may change in future releases.

## Related samples

*   [MCP: serve an ADK agent](../../../../../contributing/samples/mcp/mcp_serve_agent)
