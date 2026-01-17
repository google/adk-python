# Plan: CodingAgent Implementation for ADK-Python

Create a production-ready `CodingAgent` class that generates Python code to execute tools via ReAct loop, with HTTP-based tool injection, dual-layer security (allowlist + container), configurable statefulness with history re-execution, and full ADK telemetry integration.

## Steps

1. **Create `CodingAgentConfig`** in [src/google/adk/agents/coding_agent_config.py](src/google/adk/agents/coding_agent_config.py) - Pydantic config extending `BaseAgentConfig` with fields: `model`, `instruction`, `tools`, `code_executor`, `authorized_imports` (frozenset), `max_iterations` (default 10), `error_retry_attempts` (default 2), `stateful` (default False), `tool_server_host` (default `host.docker.internal`, fallback `172.17.0.1`), `tool_server_port` (default 8765).

2. **Create `CodingAgentState`** in [src/google/adk/agents/coding_agent.py](src/google/adk/agents/coding_agent.py) - State extending `BaseAgentState` with: `iteration_count`, `error_count`, `execution_history` (list of `ExecutionStep` with `code`, `result`, `tool_traces`, `success` fields for re-execution optimization).

3. **Create `ToolCodeGenerator`** in [src/google/adk/code_executors/tool_code_generator.py](src/google/adk/code_executors/tool_code_generator.py) - Functions: `generate_runtime_header()` (HTTP client + `_call_adk_tool()` + trace collection), `generate_tool_stubs()` (typed function stubs from `BaseTool._get_declaration()`), `generate_final_answer_stub()`, `generate_system_prompt()` (tool docs + 1-2 few-shot examples showing `tool_code` format + `final_answer()` usage).

4. **Create `AllowlistValidator`** in [src/google/adk/code_executors/allowlist_validator.py](src/google/adk/code_executors/allowlist_validator.py) - `validate_imports()` using AST extraction, `DEFAULT_SAFE_IMPORTS` frozenset, `ImportValidationError` with violation details, `is_import_allowed()` helper supporting wildcards (e.g., `collections.*`).

5. **Create `ToolExecutionServer`** in [src/google/adk/code_executors/tool_execution_server.py](src/google/adk/code_executors/tool_execution_server.py) - FastAPI server with `POST /tool_call` routing to `BaseTool.run_async()` with full `ToolContext`, `GET /tool_trace`, lifecycle `start()/stop()`, configurable host detection (`host.docker.internal` → `172.17.0.1` fallback).

6. **Create `CodingAgentCodeExecutor`** in [src/google/adk/code_executors/coding_agent_code_executor.py](src/google/adk/code_executors/coding_agent_code_executor.py) - Composable wrapper with: tool stub prepending, allowlist pre-validation, server lifecycle, history re-execution (skip successful steps via hash comparison), trace extraction (`__TOOL_TRACE__:`), final answer detection (`__FINAL_ANSWER__:`).

7. **Create `CodingAgent` class** in [src/google/adk/agents/coding_agent.py](src/google/adk/agents/coding_agent.py) - `_run_async_impl()` ReAct loop: build system prompt with few-shot examples, call `canonical_model`, parse code blocks, validate imports, execute via `CodingAgentCodeExecutor`, detect `final_answer()` OR no-code fallback, yield events with `state_delta`, retry errors with LLM feedback up to `error_retry_attempts`.

8. **Add telemetry** in [src/google/adk/telemetry/tracing.py](src/google/adk/telemetry/tracing.py) - Add `trace_code_generation()`, `trace_code_execution()`, `trace_import_validation()`, `trace_tool_ipc()` following existing patterns with code content, duration, and error attributes.

9. **Update exports** in [src/google/adk/agents/__init__.py](src/google/adk/agents/__init__.py) and [src/google/adk/code_executors/__init__.py](src/google/adk/code_executors/__init__.py) - Add all new classes to `__all__` with lazy loading for executor components.

10. **Create comprehensive tests** in `tests/unittests/agents/test_coding_agent.py` and `tests/unittests/code_executors/test_coding_agent_*.py` - Cover: ReAct loop, final answer detection + fallback, allowlist validation, error retry, stateful history re-execution with skip optimization, tool traces, host fallback logic.

11. **Create sample agent** in `contributing/samples/coding_agent/` - Example with `web_search`, `calculator`, `read_file` tools demonstrating multi-step code generation with `ContainerCodeExecutor`.

## Key Implementation Details

### Few-shot examples in system prompt

```python
SYSTEM_PROMPT_EXAMPLES = '''
Example 1 - Using tools:
```tool_code
result = web_search(query="Python async best practices")
print(result["snippets"][0])
```

Example 2 - Final answer:
```tool_code
data = read_file(path="data.csv")
total = sum(float(row["amount"]) for row in data["rows"])
final_answer(f"The total amount is ${total:.2f}")
```
'''
```

### History re-execution optimization

```python
def _should_skip_step(self, step: ExecutionStep, code_hash: str) -> bool:
    """Skip if code unchanged and previously succeeded."""
    return step.success and step.code_hash == code_hash
```

### Host detection with fallback

```python
def _resolve_tool_server_host(self) -> str:
    if self.tool_server_host:
        return self.tool_server_host
    # Try host.docker.internal first (Docker Desktop)
    # Fallback to 172.17.0.1 (Linux bridge network)
    return detect_docker_host_address()
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CodingAgent._run_async_impl()               │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ Build prompt│──▶│ Call LLM     │──▶│ Parse code blocks   │  │
│  │ + tool docs │   │ (canonical_  │   │ (delimiters)        │  │
│  └─────────────┘   │  model)      │   └─────────┬───────────┘  │
│                    └──────────────┘             │               │
│                                                 ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          CodingAgentCodeExecutor.execute_code()         │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │   │
│  │  │ Validate    │─▶│ Prepend tool │─▶│ Execute in    │   │   │
│  │  │ imports     │  │ stubs +      │  │ container     │   │   │
│  │  │ (allowlist) │  │ runtime      │  │               │   │   │
│  │  └─────────────┘  └──────────────┘  └───────┬───────┘   │   │
│  └─────────────────────────────────────────────┼───────────┘   │
│                                                │               │
│         ┌──────────────────────────────────────┘               │
│         │  HTTP IPC (host.docker.internal)                     │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ToolExecutionServer (FastAPI)               │   │
│  │  POST /tool_call ──▶ BaseTool.run_async(ToolContext)    │   │
│  │  GET /tool_trace ──▶ call_traces[]                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐   ┌─────────────────────────────────────┐ │
│  │ Check final_    │◀──│ Extract traces + clean stdout       │ │
│  │ answer() OR     │   │ (__TOOL_TRACE__, __FINAL_ANSWER__)  │ │
│  │ fallback        │   └─────────────────────────────────────┘ │
│  └────────┬────────┘                                           │
│           │ if done: yield final Event                         │
│           │ else: feed result back to LLM (loop)               │
└───────────┴─────────────────────────────────────────────────────┘
```

## File Structure

```
src/google/adk/
├── agents/
│   ├── coding_agent.py           # CodingAgent + CodingAgentState
│   ├── coding_agent_config.py    # CodingAgentConfig
│   └── __init__.py               # Updated exports
├── code_executors/
│   ├── tool_code_generator.py    # Stub generation + system prompt
│   ├── allowlist_validator.py    # Import validation
│   ├── tool_execution_server.py  # FastAPI IPC server
│   ├── coding_agent_code_executor.py  # Main executor wrapper
│   └── __init__.py               # Updated exports
└── telemetry/
    └── tracing.py                # New trace functions

tests/unittests/
├── agents/
│   └── test_coding_agent.py
└── code_executors/
    ├── test_tool_code_generator.py
    ├── test_allowlist_validator.py
    └── test_coding_agent_code_executor.py

contributing/samples/
└── coding_agent/
    ├── __init__.py
    └── agent.py
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tool injection | HTTP IPC (Code Prepending) | Native ADK integration, full `ToolContext` access, async support |
| Security | Allowlist + Container | Defense-in-depth: import validation before container isolation |
| Final answer | Explicit `final_answer()` + fallback | Reliability with graceful degradation |
| Stateful mode | Re-execute history | Safer than pickle, with skip optimization for speed |
| Async tools | Sync wrapper via host server | Host handles async natively, container code stays simple |
| Docker host | Configurable with fallback | `host.docker.internal` → `172.17.0.1` for cross-platform |
| Retries | Default 2 with LLM feedback | Matches `BaseCodeExecutor.error_retry_attempts` pattern |
