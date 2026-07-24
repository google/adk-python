# ADK Developer Guides

This directory contains specific developer guides for the ADK Python implementation. For the official ADK documentation, visit [adk.dev](https://adk.dev/).

## Index

### Agents
* [LlmAgent Single-Turn Mode](agents/llm_agent/single_turn.md) - Guide on using LlmAgent in single-turn mode.
* [LlmAgent Task Mode](agents/llm_agent/task.md) - Guide on using LlmAgent in task mode.
* [ManagedAgent](agents/managed_agent/index.md) - Guide on using ManagedAgent with server-side tools.

### Events
* [Event and NodeInfo](events/event/index.md) - Understanding Event and NodeInfo in workflows.
* [RequestInput](events/request_input/index.md) - How to use RequestInput for human-in-the-loop interactions.

### Tools
* [to_mcp_server](tools/mcp_tool/agent_to_mcp/index.md) - Expose an ADK agent as an MCP server so any MCP host can drive it as a single tool (the MCP counterpart of to_a2a).

### Workflows
* [Workflow](workflow/workflow/index.md) - Graph-based orchestration of complex, multi-step agent interactions.
* [Workflow Graphs](workflow/graph/index.md) - Understanding nodes, edges, and graph structures in workflows.
* [Function Nodes](workflow/function_node/index.md) - Wrapping Python functions and generators as workflow nodes.
* [JoinNode](workflow/join_node/index.md) - Synchronizing parallel execution paths in workflows.
* [RetryConfig](workflow/retry_config/index.md) - Configuring retry policies for resilient workflow nodes.
* [ParallelWorker](workflow/parallel_worker/index.md) - Processing lists of items concurrently in workflows.
* [Dynamic Nodes](workflow/dynamic_nodes/index.md) - Scheduling and executing nodes dynamically at runtime.
