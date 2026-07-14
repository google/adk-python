# ManagedAgent

## Introduction

`ManagedAgent` allows you to leverage managed agents backed by the Managed
Agents API (`interactions.create`) via either the
[Gemini Enterprise Agents Platform (GEAP, formerly Vertex)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/managed-agents)
or the [Gemini API](https://ai.google.dev/gemini-api/docs/agents) from within
your ADK flows. It is particularly useful when you want to utilize Google's
powerful first-party, out-of-the-box agents (like the Antigravity agent) that
have specialized server-side execution environments built-in without requiring
client-side function declarations.

This solves the developer problem of needing a robust, server-hosted environment
for agents that require specialized built-in capabilities, rather than managing
sandbox environments and Python code execution locally. `ManagedAgent` can be
used as a standalone agent, integrated directly into a workflow, or encapsulated
as a tool via `AgentTool` so that a coordinating `LlmAgent` can delegate
specialized tasks to it.

## Prerequisites

The `ManagedAgent` supports two distinct backends: the Gemini API backend and
the Gemini Enterprise Agents Platform (GEAP) backend. Depending on which backend
you intend to use, you must satisfy the corresponding prerequisites for
authentication and obtaining an Agent ID.

### Option 1: Gemini API Backend

*   **Authentication**: You must obtain a Gemini API key. Set this as the
    `GEMINI_API_KEY` environment variable.
*   **Agent ID**: You need an `agent_id` to connect to. You can either:
    *   Create a new agent by following the
        [Gemini API Agents documentation](https://ai.google.dev/gemini-api/docs/agents).
    *   Use an out-of-the-box agent ID, such as `antigravity-preview-05-2026`,
        which is commonly used in our examples.

### Option 2: Gemini Enterprise Agents Platform (GEAP) Backend

*   **Authentication**: GEAP (formerly Vertex) requires Google Cloud
    credentials. Follow the
    [GEAP setup instructions](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/managed-agents/create-manage#before-you-begin)
    to authenticate your local environment (e.g., using `gcloud auth
    application-default login`).
*   **Agent ID**: Similar to the Gemini API, you need an `agent_id`. You can
    either:
    *   Create a new agent via the
        [GEAP Managed Agents guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/managed-agents).
    *   Use an out-of-the-box agent ID if available to your project.

## Get started

Here is a minimal implementation of `ManagedAgent` demonstrating its use.

```python
import os
from google.adk.agents import ManagedAgent
from google.adk.tools import google_search
from google.genai import types

# Ensure you have the MANAGED_AGENT_ID and the proper environment config
_AGENT_ID = os.environ.get('MANAGED_AGENT_ID', 'antigravity-preview-05-2026')

managed_search_agent = ManagedAgent(
    name='managed_search_agent',
    description='Answers questions that need fresh, grounded information from the web.',
    agent_id=_AGENT_ID,
    environment={'type': 'remote'},
    tools=[google_search],
)

# A managed code execution agent using raw types.Tool
managed_code_execution_agent = ManagedAgent(
    name='managed_code_execution_agent',
    description='Solves computational questions by running code server-side.',
    agent_id=_AGENT_ID,
    environment={'type': 'remote'},
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)
```

To see an orchestrator pattern using this code, you could wrap them using
`AgentTool`:

```python
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

# The local coordinator delegates tasks to the server-backed agents
root_agent = LlmAgent(
    name='managed_tool_coordinator',
    description='Calls managed specialists as tools and composes the answer.',
    tools=[
        AgentTool(agent=managed_search_agent),
        AgentTool(agent=managed_code_execution_agent),
    ],
)
```

## How it works

The `ManagedAgent` implements the `BaseAgent` contract but bypasses standard
`generate_content` calls, instead sending interactions via
`_create_interactions` with `background=True`. It natively streams partial
events and terminal events in real-time back to the ADK `Runner` or parent flow.

When using the GEAP backend, it enforces a connection to the `global` location
since the Managed Agents API is solely available globally. Because it runs
remotely, tools are translated into standard `ToolParam` formats for
interactions; any raw `google.genai.types.Tool` configs are passed through to
the backend, enabling server-side code execution or remote google search
seamlessly.

### State: local session vs. remote

`ManagedAgent` keeps almost no state locally. The ADK session only persists two
values on the events it emits: the `previous_interaction_id` and the sandbox
`environment_id`. On each new turn the agent recovers both by scanning prior
session events, then reuses them so the conversation and its sandbox continue.

Everything else lives server-side. The Managed Agents API owns the sandbox
environment and the full interaction history, and that remote interaction — not
the local session — is the source of truth for continuing a conversation.
Response text appears in both places (the local ADK events and the remote
interaction history), but ADK stores only the ids it needs to recover and reuse
the remote state; it never re-sends prior turns.

## Advanced applications

### Tool encapsulation for orchestration

*   **Problem solved**: Sometimes a single LLM request needs to compose results
    from multiple independent, robust specialists without losing control of the
    execution turn.
*   **Implementation**: Encapsulate each `ManagedAgent` instance within its own
    separate `AgentTool` and provide them as a list of tools to an `LlmAgent`
    coordinator. The coordinator will invoke the managed agents (which run their
    sandboxed logic server-side), collect the results, and then compose the
    final synthesized response natively.

## Limitations

*   **Location pinned (GEAP only)**: For the GEAP backend, the Managed Agents
    API is currently only served from the `global` location. Enterprise clients
    using regional endpoints will raise an error.
*   **Server-side tools only**: Client-executed tools (Python functions,
    callables) and MCP tools are not supported. Providing these will raise a
    `NotImplementedError`.
*   **Streaming only**: The agent only supports streaming interactions.
    Background-polling execution or strictly non-streaming connections are not
    yet fully supported (it natively uses `stream=True` and yields events).

## Related samples

*   [Managed Agent Basic](../../../../contributing/samples/managed_agent/basic)
*   [Managed Agent Code Execution](../../../../contributing/samples/managed_agent/code_execution)
