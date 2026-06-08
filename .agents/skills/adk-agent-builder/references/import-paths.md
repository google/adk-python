# ADK Import Paths Quick Reference

## 📋 Agent Verification Checklist (Imports)
Use this checklist to ensure you are using the most idiomatic import paths:
- [ ] **Canonical Imports**: Did you use the short canonical imports where available (e.g., `from google.adk import Agent`) instead of the verbose ones?
- [ ] **Avoid Deprecated**: Are you avoiding deprecated paths (e.g., use `McpToolset` instead of `MCPToolset`)?

## Canonical Imports (preferred, used by all samples)

```python
from google.adk import Agent, Context, Event, Workflow
from google.adk.events import RequestInput
from google.adk.workflow import node, RetryConfig, Edge, JoinNode
```

## Core Agents

| Component | Import |
|-----------|--------|
| `Agent` (canonical) | `from google.adk import Agent` |
| `Agent` (verbose) | `from google.adk.agents.llm_agent import Agent` |
| `LlmAgent` | `from google.adk.agents.llm_agent import LlmAgent` |
| `SequentialAgent` | `from google.adk.agents.sequential_agent import SequentialAgent` |
| `ParallelAgent` | `from google.adk.agents.parallel_agent import ParallelAgent` |
| `LoopAgent` | `from google.adk.agents.loop_agent import LoopAgent` |

## Workflow Agents (Experimental)

| Component | Import |
|-----------|--------|
| `Workflow` | `from google.adk.workflow import Workflow` |
| `Edge` | `from google.adk.workflow import Edge` |
| `Agent` (supports task/single_turn mode) | `from google.adk import Agent` |

## Workflow Nodes

| Component | Import |
|-----------|--------|
| `FunctionNode` | `from google.adk.workflow import FunctionNode` |
| `_LlmAgentWrapper` (private, auto-used) | `from google.adk.workflow._llm_agent_wrapper import _LlmAgentWrapper` |
| `AgentNode` | `from google.adk.workflow._agent_node import AgentNode` |
| `_ToolNode` (private) | `from google.adk.workflow._tool_node import _ToolNode` |
| `JoinNode` | `from google.adk.workflow import JoinNode` |
| `ParallelWorker` | `from google.adk.workflow._parallel_worker import ParallelWorker` |
| `BaseNode`, `START` | `from google.adk.workflow import BaseNode, START` |
| `@node` decorator | `from google.adk.workflow import node` |

## Workflow Events and Context

| Component | Import |
|-----------|--------|
| `Event` | `from google.adk.events.event import Event` |
| `RequestInput` | `from google.adk.events.request_input import RequestInput` |
| `Context` | `from google.adk.agents.context import Context` |
| `WorkflowGraph` | `from google.adk.workflow._workflow_graph import WorkflowGraph` |
| `RetryConfig` | `from google.adk.workflow import RetryConfig` |

## Task Mode

| Component | Import |
|-----------|--------|
| `RequestTaskTool` | `from google.adk.agents.llm.task._request_task_tool import RequestTaskTool` |
| `FinishTaskTool` | `from google.adk.agents.llm.task._finish_task_tool import FinishTaskTool` |
| `TaskRequest`, `TaskResult` | `from google.adk.agents.llm.task._task_models import TaskRequest, TaskResult` |

## Tools

| Component | Import |
|-----------|--------|
| `FunctionTool` | `from google.adk.tools.function_tool import FunctionTool` |
| `BaseTool` | `from google.adk.tools.base_tool import BaseTool` |
| `BaseToolset` | `from google.adk.tools.base_toolset import BaseToolset` |
| `ToolContext` | `from google.adk.tools.tool_context import ToolContext` |
| `LongRunningFunctionTool` | `from google.adk.tools.long_running_tool import LongRunningFunctionTool` |
| `McpToolset` | `from google.adk.tools.mcp_tool.mcp_toolset import McpToolset` |
| `StdioConnectionParams` | `from google.adk.tools.mcp_tool import StdioConnectionParams` |
| `SseConnectionParams` | `from google.adk.tools.mcp_tool import SseConnectionParams` |
| `OpenAPIToolset` | `from google.adk.tools.openapi_tool import OpenAPIToolset` |

## Built-in Tools

| Tool | Import |
|------|--------|
| `google_search` | `from google.adk.tools import google_search` |
| `load_artifacts` | `from google.adk.tools import load_artifacts` |
| `load_memory` | `from google.adk.tools import load_memory` |
| `exit_loop` | `from google.adk.tools import exit_loop` |
| `transfer_to_agent` | `from google.adk.tools import transfer_to_agent` |
| `get_user_choice` | `from google.adk.tools import get_user_choice` |

## Runner and Session

| Component | Import |
|-----------|--------|
| `Runner` | `from google.adk.runners import Runner` |
| `InMemoryRunner` | `from google.adk.runners import InMemoryRunner` |
| `InMemorySessionService` | `from google.adk.sessions import InMemorySessionService` |
| `DatabaseSessionService` | `from google.adk.sessions import DatabaseSessionService` |

## App and Plugins

| Component | Import |
|-----------|--------|
| `App` | `from google.adk.apps import App` |
| `ResumabilityConfig` | `from google.adk.apps.app import ResumabilityConfig` |
| `BasePlugin` | `from google.adk.plugins.base_plugin import BasePlugin` |
| `ContextFilterPlugin` | `from google.adk.plugins.context_filter_plugin import ContextFilterPlugin` |

## Models

| Component | Import |
|-----------|--------|
| `LiteLlm` | `from google.adk.models.lite_llm import LiteLlm` |
| `LlmRequest` | `from google.adk.models.llm_request import LlmRequest` |
| `LlmResponse` | `from google.adk.models.llm_response import LlmResponse` |

## Callbacks

| Component | Import |
|-----------|--------|
| `CallbackContext` | `from google.adk.agents.callback_context import CallbackContext` |
| `ReadonlyContext` | `from google.adk.agents.readonly_context import ReadonlyContext` |

## Code Executors

| Component | Import |
|-----------|--------|
| `BuiltInCodeExecutor` | `from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor` |

## Google GenAI Types

| Component | Import |
|-----------|--------|
| `types` | `from google.genai import types` |
| `Content` | `from google.genai.types import Content` |
| `ModelContent` | `from google.genai.types import ModelContent` |
| `Part` | `from google.genai.types import Part` |
| `GenerateContentConfig` | `from google.genai.types import GenerateContentConfig` |
