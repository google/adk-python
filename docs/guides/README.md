# ADK Developer Guides

This directory contains specific developer guides for the ADK Python implementation. For the official ADK documentation, visit [adk.dev](https://adk.dev/).

## Index

### A2A
* [A2aAgentExecutor](a2a/executor/a2a_agent_executor/index.md) - Serving an ADK agent behind an A2A server, translating incoming requests into runs and ADK events into task updates.
* [A2aRemoteAgentConfig](a2a/agent/config/index.md) - Controlling the outbound call a RemoteA2aAgent makes to another agent.
* [AgentCardBuilder](a2a/utils/agent_card_builder/index.md) - Deriving the agent card a client reads before sending work, from an agent or workflow you already have.
* [to_a2a](a2a/utils/agent_to_a2a/index.md) - Putting an ADK agent on the network as a Starlette application that speaks the Agent2Agent protocol.

### Agents
* [BaseAgent](agents/base_agent/index.md) - The foundational base class for custom agents, container orchestrators, and lifecycle callbacks.
* [Context](agents/context/index.md) - The runtime interface for state, artifacts, memory, credentials, and dynamic execution.
* [Creating Agents with Configurations](agents/config/index.md) - Building and wiring multi-agent graphs from external YAML configuration files.
* [InvocationContext](agents/invocation_context/index.md) - The runtime dependency container and execution state for a single invocation turn.
* [LlmAgent](agents/llm_agent/index.md) - The primary conversational reasoning agent orchestrating models, tools, and workflows.
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

### CLI
* [ServiceRegistry](cli/service_registry/index.md) - Mapping a URI scheme to a factory, so your own session, memory, or artifact store can be selected by URI.
* [get_fast_api_app](cli/fast_api/index.md) - Serving every agent in a directory over ADK's HTTP API, with room for your own routes, middleware, and lifespan.

### Code Executors
* [BaseCodeExecutor](code_executors/code_executor/index.md) - Executing model-generated code safely across local, container, GKE, and managed sandbox backends.

### Environment
* [BaseEnvironment and LocalEnvironment](environment/base_environment/index.md) - The interface for a place where an agent runs shell commands and keeps files, and the implementation that runs them as local subprocesses.

### Errors
* [ADK exceptions](errors/index.md) - The exception types ADK raises on its own behalf, and which part of the framework each one comes from.

### Evaluation
* [AgentEvaluator](evaluation/agent_evaluator/index.md) - Measuring agent quality from inside a pytest suite by replaying recorded conversations and scoring each answer and tool call.
* [BaseEvalService and LocalEvalService](evaluation/eval_service/index.md) - Running evaluations that return results as data rather than as a test that passed or failed.
* [Efficiency metrics](evaluation/efficiency_evaluators/index.md) - Reference-free metrics reporting what a run consumed: tool calls, model calls and tokens.
* [EvalConfig and the eval config file](evaluation/eval_config/index.md) - The schema of the file that says which metrics score a run and how strict each one is.
* [Evaluator](evaluation/evaluator/index.md) - The interface behind the built-in metrics, and how to score a rule that is specific to your agent.

### Events
* [Event and NodeInfo](events/event/index.md) - Understanding Event and NodeInfo in workflows.
* [RequestInput](events/request_input/index.md) - How to use RequestInput for human-in-the-loop interactions.

### Examples
* [Example and ExampleTool](examples/example/index.md) - Showing the model worked input and output pairs so that it gets the shape of its own answers right.

### Features
* [Feature flags](features/feature_registry/index.md) - Turning behavior that is not yet stable on or off with the ADK_ENABLE and ADK_DISABLE environment variables.

### Flows
* [Live model callbacks](flows/llm_flows/base_llm_flow/live_model_callbacks.md) - Inspecting or blocking content on a live bidirectional session.

### Integrations
* [BigQueryToolset](integrations/bigquery/bigquery_toolset/index.md) - Exploring and querying BigQuery, and the write_mode setting that decides what the agent may change.
* [CrewaiTool](integrations/crewai/crewai_tool/index.md) - Wrapping a CrewAI tool so an ADK agent can call it.
* [DaytonaEnvironment](integrations/daytona/daytona_environment/index.md) - Running agent code in a Daytona hosted sandbox instead of on your machine.
* [E2BEnvironment](integrations/e2b/e2b_environment/index.md) - Running agent code in an E2B hosted sandbox, and what happens when the sandbox expires.
* [FirestoreSessionService](integrations/firestore/firestore_session_service/index.md) - A durable multi-process session store built on Firestore documents and transactions.
* [GCSToolset and GCSAdminToolset](integrations/gcs/gcs_toolset/index.md) - Giving an agent access to Cloud Storage objects and buckets, read-only until you say otherwise.
* [LangchainTool](integrations/langchain/langchain_tool/index.md) - Wrapping a LangChain tool so an ADK agent can call it.
* [Model Armor](integrations/model_armor/index.md) - Screening user input and model output with Google Cloud Model Armor.
* [MongoDbToolset](integrations/mongodb/mongodb_toolset/index.md) - Vector and hybrid search over a MongoDB database, with the query text embedded on the way through.
* [RedisSessionService](integrations/redis/redis_session_service/index.md) - Sharing sessions across processes through Redis, including the expiry every other backend lacks.

### Labs
* [AntigravityAgent](labs/antigravity/index.md) - Runs a Google Antigravity SDK agent as an ADK agent node.

### Live
* [Live tools](live/tools/index.md) - Asynchronous background execution and response scheduling for Gemini Live agents.
* [LiveRequestQueue](live/live_request_queue/index.md) - Streaming content, realtime audio, and stream control signals to live agents.

### Memory
* [BaseMemoryService](memory/memory_service/index.md) - Storing finished sessions and recalling them from later conversations.

### Models
* [BaseLlm and LLMRegistry](models/llm_registry/index.md) - The model interface, how a model name resolves to an implementation, and how to plug in your own.
* [FallbackModel](models/fallback_model/index.md) - Wrapping an ordered list of models and moving to the next one when a call fails.
* [ServiceTier](models/service_tier/index.md) - Choosing serving tiers for Interactions API calls, including deferred execution on off-peak capacity.

### Optimization
* [AgentOptimizer and Sampler](optimization/agent_optimizer/index.md) - Rewriting an agent's instruction automatically, scoring candidate prompts against an evaluation set and keeping the better one.

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

### Skills
* [Skill, Frontmatter, and Resources](skills/skill/index.md) - The SKILL.md file format, and the folder of instructions, reference documents, and scripts an agent pulls in only when it is relevant.
* [SkillRegistry](skills/skill_registry/index.md) - The interface behind a searchable catalog of skills that an agent discovers at runtime.

### Telemetry
* [TelemetryConfig](telemetry/telemetry_config/index.md) - What ADK puts in its OpenTelemetry traces, and whether the text of prompts and replies is copied onto exported spans.

### Tools
* [Node as tool](tools/node_tool/index.md) - Exposing workflows and deterministic nodes as agent tools with isolated runtime branching and resume support.
* [to_mcp_server](tools/mcp_tool/agent_to_mcp/index.md) - Expose an ADK agent as an MCP server so any MCP host can drive it as a single tool (the MCP counterpart of to_a2a).

### Utils
* [inject_session_state](utils/instructions_utils/index.md) - Substituting session state values and artifact contents into an instruction string.

### Workflows
* [BaseNode](workflow/base_node/index.md) - The foundational base class and configuration settings for all workflow nodes.
* [Workflow](workflow/workflow/index.md) - Graph-based orchestration of complex, multi-step agent interactions.
* [Workflow Graphs](workflow/graph/index.md) - Understanding nodes, edges, and graph structures in workflows.
* [Function Nodes](workflow/function_node/index.md) - Wrapping Python functions and generators as workflow nodes.
* [JoinNode](workflow/join_node/index.md) - Synchronizing parallel execution paths in workflows.
* [RetryConfig](workflow/retry_config/index.md) - Configuring retry policies for resilient workflow nodes.
* [ParallelWorker](workflow/parallel_worker/index.md) - Processing lists of items concurrently in workflows.
* [Dynamic Nodes](workflow/dynamic_nodes/index.md) - Scheduling and executing nodes dynamically at runtime.
