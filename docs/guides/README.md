# ADK Developer Guides

This directory contains specific developer guides for the ADK Python implementation. For the official ADK documentation, visit [adk.dev](https://adk.dev/).

## Index

### Agents
* [LiveRequestQueue](agents/live_request_queue/index.md) - Streaming content, realtime audio, and stream control signals to live agents.
* [LlmAgent Single-Turn Mode](agents/llm_agent/single_turn.md) - Guide on using LlmAgent in single-turn mode.
* [LlmAgent Task Mode](agents/llm_agent/task.md) - Guide on using LlmAgent in task mode.
* [ManagedAgent](agents/managed_agent/index.md) - Guide on using ManagedAgent with server-side tools.
* [RemoteA2aAgent Task Mode](agents/remote_a2a_agent/task.md) - Guide on using RemoteA2aAgent in task mode.

### Apps
* [App](apps/app/index.md) - The top-level container binding a root agent to app-wide plugins and configuration.

### Artifacts
* [BaseArtifactService](artifacts/artifact_service/index.md) - Storing binary payloads outside the conversation history, with versioning and user-scoped filenames.

### Auth
* [AuthConfig and authenticated tools](auth/tool_auth/index.md) - Declaring the credentials a tool needs, and the pause-for-consent handshake.

### Code Executors
* [BaseCodeExecutor](code_executors/code_executor/index.md) - Executing model-generated code safely across local, container, GKE, and managed sandbox backends.

### Events
* [Event and NodeInfo](events/event/index.md) - Understanding Event and NodeInfo in workflows.
* [RequestInput](events/request_input/index.md) - How to use RequestInput for human-in-the-loop interactions.

### Flows
* [Live model callbacks](flows/llm_flows/base_llm_flow/live_model_callbacks.md) - Inspecting or blocking content on a live bidirectional session.

### Integrations
* [Model Armor](integrations/model_armor/index.md) - Screening user input and model output with Google Cloud Model Armor.

### Labs
* [AntigravityAgent](labs/antigravity/index.md) - Runs a Google Antigravity SDK agent as an ADK agent node.

### Memory
* [BaseMemoryService](memory/memory_service/index.md) - Storing finished sessions and recalling them from later conversations.

### Models
* [BaseLlm and LLMRegistry](models/llm_registry/index.md) - The model interface, how a model name resolves to an implementation, and how to plug in your own.

### Planners
* [BasePlanner](planners/planner/index.md) - Guiding model execution with structured planning instructions, thinking configurations, and Plan-Re-Act thought tagging.

### Plugins
* [ReflectAndRetryModelPlugin](plugins/reflect_retry_model_plugin/index.md) - Self-healing, concurrent-safe error recovery for model failures.
* [ReflectAndRetryToolPlugin](plugins/reflect_retry_tool_plugin/index.md) - Self-healing, concurrent-safe error recovery for tool failures.

### Runners
* [Runner and InMemoryRunner](runners/runner/index.md) - Managing session lifecycles, state resolution, and streaming agent execution events.
* [Runner Live Streaming](runners/runner/live.md) - Real-time bidirectional audio/text streaming and non-blocking background tool execution with Gemini Multimodal Live API.

### Sessions
* [Session and BaseSessionService](sessions/session/index.md) - The session lifecycle, state scoping, and choosing a session service.
* [State](sessions/state/index.md) - Session state and the app:, user:, and temp: prefixes that decide what is shared and what is stored.

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
