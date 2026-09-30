# ADK Import Paths

Prefer the canonical short import. The verbose module path is listed only where
there is no short form, or where the short form does not exist yet.

## Canonical imports

These cover most agent code:

```python
from google.adk import Agent, Context, Event, Runner, Workflow
from google.adk.events import Event, EventActions, RequestInput
from google.adk.workflow import BaseNode, Edge, FunctionNode, JoinNode, node
from google.adk.workflow import DEFAULT_ROUTE, RetryConfig, START
```

`google.adk` and `google.adk.workflow` load their members lazily, so importing
from them does not pull in the whole package.

## Agents

| Component | Import |
|---|---|
| `Agent` (alias of `LlmAgent`) | `from google.adk import Agent` |
| `LlmAgent` | `from google.adk.agents import LlmAgent` |
| `BaseAgent` | `from google.adk.agents import BaseAgent` |
| `SequentialAgent` (deprecated — use `Workflow`) | `from google.adk.agents import SequentialAgent` |
| `ParallelAgent` (deprecated — use `Workflow`) | `from google.adk.agents import ParallelAgent` |
| `LoopAgent` (deprecated — use `Workflow`) | `from google.adk.agents import LoopAgent` |
| `RunConfig` | `from google.adk.agents import RunConfig` |

## Workflow graph

| Component | Import |
|---|---|
| `Workflow` | `from google.adk.workflow import Workflow` |
| `Edge` | `from google.adk.workflow import Edge` |
| `DEFAULT_ROUTE` (`"__DEFAULT__"`) | `from google.adk.workflow import DEFAULT_ROUTE` |
| `START` | `from google.adk.workflow import START` |
| `Graph` | `from google.adk.workflow._graph import Graph` |

`Graph` is private; construct workflows from `edges=[...]` unless you need
`Graph.from_edge_items(...)` directly.

## Nodes

| Component | Import |
|---|---|
| `BaseNode` | `from google.adk.workflow import BaseNode` |
| `FunctionNode` | `from google.adk.workflow import FunctionNode` |
| `Node` | `from google.adk.workflow import Node` |
| `@node` decorator | `from google.adk.workflow import node` |
| `JoinNode` | `from google.adk.workflow import JoinNode` |
| `RetryConfig` | `from google.adk.workflow import RetryConfig` |
| `NodeTimeoutError` | `from google.adk.workflow import NodeTimeoutError` |
| `_ToolNode` (private) | `from google.adk.workflow._tool_node import _ToolNode` |

Parallel-worker behavior has no importable class. Set `parallel_worker=True` on
`@node` or on an `LlmAgent`; the framework wraps it with an internal
`_ParallelWorker`.

## Events and context

| Component | Import |
|---|---|
| `Event` | `from google.adk import Event` |
| `EventActions` | `from google.adk.events import EventActions` |
| `RequestInput` | `from google.adk.events import RequestInput` |
| `Context` | `from google.adk import Context` |
| `CallbackContext` (alias of `Context`) | `from google.adk.agents.callback_context import CallbackContext` |
| `ReadonlyContext` | `from google.adk.agents.readonly_context import ReadonlyContext` |
| `ToolContext` (alias of `Context`) | `from google.adk.tools import ToolContext` |

## Task delegation

| Component | Import |
|---|---|
| `FinishTaskTool` | `from google.adk.agents.llm.task._finish_task_tool import FinishTaskTool` |
| `TaskRequest`, `TaskResult` | `from google.adk.agents.llm.task._task_models import TaskRequest, TaskResult` |

All three are private. Setting `mode='task'` attaches `FinishTaskTool`
automatically — there is no reason to import it in agent code.

## Tools

| Component | Import |
|---|---|
| `FunctionTool` | `from google.adk.tools import FunctionTool` |
| `BaseTool` | `from google.adk.tools import BaseTool` |
| `BaseToolset` | `from google.adk.tools.base_toolset import BaseToolset` |
| `LongRunningFunctionTool` | `from google.adk.tools import LongRunningFunctionTool` |
| `AgentTool` | `from google.adk.tools import AgentTool` |
| `McpToolset` | `from google.adk.tools.mcp_tool import McpToolset` |
| `StdioConnectionParams` | `from google.adk.tools.mcp_tool import StdioConnectionParams` |
| `SseConnectionParams` | `from google.adk.tools.mcp_tool import SseConnectionParams` |
| `StreamableHTTPConnectionParams` | `from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams` |
| `OpenAPIToolset`, `RestApiTool` | `from google.adk.tools.openapi_tool import OpenAPIToolset, RestApiTool` |

`MCPToolset` (all caps) still resolves but raises a deprecation warning — use
`McpToolset`.

## Runner, sessions, app

| Component | Import |
|---|---|
| `Runner` | `from google.adk import Runner` |
| `InMemoryRunner` | `from google.adk.runners import InMemoryRunner` |
| `InMemorySessionService` | `from google.adk.sessions import InMemorySessionService` |
| `DatabaseSessionService` | `from google.adk.sessions import DatabaseSessionService` |
| `VertexAiSessionService` | `from google.adk.sessions import VertexAiSessionService` |
| `App`, `ResumabilityConfig` | `from google.adk.apps import App, ResumabilityConfig` |
| `BasePlugin` | `from google.adk.plugins.base_plugin import BasePlugin` |

`DatabaseSessionService` needs the `db` extra; importing it without
`sqlalchemy` installed raises a "missing extra" error rather than
`ImportError`.

## Live streaming

| Component | Import |
|---|---|
| `LiveRequestQueue` | `from google.adk.live import LiveRequestQueue` |
| `LiveRequest` | `from google.adk.live import LiveRequest` |

## Models

| Component | Import |
|---|---|
| `BaseLlm` | `from google.adk.models.base_llm import BaseLlm` |
| `LiteLlm` | `from google.adk.models.lite_llm import LiteLlm` |
| `LlmRequest` | `from google.adk.models.llm_request import LlmRequest` |
| `LlmResponse` | `from google.adk.models.llm_response import LlmResponse` |

## Code executors

| Component | Import |
|---|---|
| `BuiltInCodeExecutor` | `from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor` |

## google-genai types

| Component | Import |
|---|---|
| `types` | `from google.genai import types` |
| `Content`, `ModelContent`, `Part` | `from google.genai.types import Content, ModelContent, Part` |
| `GenerateContentConfig` | `from google.genai.types import GenerateContentConfig` |
