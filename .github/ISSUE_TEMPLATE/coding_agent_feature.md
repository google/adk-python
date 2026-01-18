---
name: "Feature: CodingAgent - Code-generating Agent with Sandboxed Execution"
about: "New experimental agent type that generates and executes Python code"
title: "feat(agents): Add CodingAgent for code generation and sandboxed execution"
labels: "enhancement, agents, new-feature"
assignees: ''
---

## Summary

Add a new experimental agent type called **CodingAgent** that generates Python code to solve tasks, executes it in a sandboxed Docker container, and iterates using a ReAct-style loop. This mirrors the popular "Code Interpreter" pattern seen in other AI platforms.

## Is your feature request related to a problem?

Currently, ADK agents can only interact with the world through pre-defined tools. While powerful, this approach has limitations:

1. **Limited flexibility**: Users must anticipate all possible operations and create tools for each
2. **No computational capability**: Agents cannot perform complex calculations, data analysis, or create visualizations without custom tools
3. **No iteration**: Standard tool-calling doesn't easily support multi-step reasoning with intermediate computations
4. **Competitive gap**: Other platforms (OpenAI Code Interpreter, Anthropic's computer use) offer code execution capabilities

**User pain points:**
- "I want my agent to analyze a CSV file and create a chart" - requires building custom tools
- "I need multi-step calculations with intermediate results" - awkward with standard tools
- "I want the agent to figure out HOW to solve a problem, not just call predefined functions"

## Describe the solution

### CodingAgent Overview

A new agent type that:
1. Receives a task from the user
2. Generates Python code to accomplish the task
3. Executes the code in a sandboxed Docker container
4. Processes results and either provides an answer or continues iterating
5. Can call ADK tools from within generated code via HTTP IPC

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│   CodingAgent    │────▶│ Docker Container│
│                 │     │  (Gemini LLM)    │     │ (Python 3.11)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               │                         │ Executes
                               ▼                         │ generated code
                        ┌──────────────┐                 │
                        │ Tool Server  │◀────────────────┘
                        │ (HTTP IPC)   │  Tool calls via HTTP
                        └──────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| CodingAgent | `src/google/adk/agents/coding_agent.py` | Main agent class with ReAct loop |
| CodingAgentConfig | `src/google/adk/agents/coding_agent_config.py` | Pydantic configuration |
| CodingAgentCodeExecutor | `src/google/adk/code_executors/coding_agent_code_executor.py` | Wrapper that injects tools |
| ToolCodeGenerator | `src/google/adk/code_executors/tool_code_generator.py` | Generates Python stubs for tools |
| ToolExecutionServer | `src/google/adk/code_executors/tool_execution_server.py` | HTTP server for tool IPC |
| AllowlistValidator | `src/google/adk/code_executors/allowlist_validator.py` | Import security validation |

### API Design

```python
from google.adk.agents import CodingAgent
from google.adk.code_executors import ContainerCodeExecutor

def fetch_data(url: str) -> dict:
    """Fetch data from a URL."""
    # Implementation...

root_agent = CodingAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction="You are a data analyst. Analyze data and provide insights.",
    tools=[fetch_data],  # Tools available to generated code
    code_executor=ContainerCodeExecutor(image="python:3.11-slim"),
    authorized_imports=DEFAULT_SAFE_IMPORTS | {"pandas", "matplotlib"},
    max_iterations=10,
    error_retry_attempts=2,
)
```

### Security Features

1. **Sandboxed execution**: All code runs in isolated Docker containers
2. **Import allowlisting**: Only authorized imports are permitted (configurable)
3. **Tool isolation**: Tools execute on host via HTTP, not in container
4. **No filesystem access**: Container has no access to host filesystem
5. **Network isolation**: Container can only reach tool server

### Sample Agent

A complete Data Analysis Agent sample is included:
- Fetches datasets from URLs (Titanic, Iris, Tips)
- Analyzes data with pandas
- Creates visualizations with matplotlib
- Saves charts to host system via `save_chart` tool

## Describe alternatives you've considered

### Alternative 1: Extend LlmAgent with code execution
- **Pros**: Simpler architecture, reuses existing agent
- **Cons**: Conflates two distinct patterns, harder to configure

### Alternative 2: Code execution as a tool only
- **Pros**: Minimal changes, fits existing model
- **Cons**: No ReAct loop, no iteration, limited capability

### Alternative 3: Use external code execution service
- **Pros**: Offloads security concerns
- **Cons**: Adds external dependency, latency, cost

**Chosen approach**: Dedicated CodingAgent provides cleanest separation of concerns, explicit configuration, and full control over the execution environment.

## Implementation Details

### Files Added/Modified

**New files (agents):**
- `src/google/adk/agents/coding_agent.py` (~550 lines)
- `src/google/adk/agents/coding_agent_config.py` (~225 lines)

**New files (code_executors):**
- `src/google/adk/code_executors/coding_agent_code_executor.py` (~500 lines)
- `src/google/adk/code_executors/tool_code_generator.py` (~475 lines)
- `src/google/adk/code_executors/tool_execution_server.py` (~365 lines)
- `src/google/adk/code_executors/allowlist_validator.py` (~350 lines)

**Modified files:**
- `src/google/adk/agents/__init__.py` - Export CodingAgent
- `src/google/adk/code_executors/__init__.py` - Export new components

**Sample agent:**
- `contributing/samples/coding_agent/agent.py` (~360 lines)
- `contributing/samples/coding_agent/README.md` (~290 lines)

**Tests:**
- `tests/unittests/agents/test_coding_agent.py` (~310 lines)
- `tests/unittests/code_executors/test_allowlist_validator.py` (~320 lines)
- `tests/unittests/code_executors/test_tool_code_generator.py` (~320 lines)

### How Tool IPC Works

1. When CodingAgent starts, it launches a ToolExecutionServer on the host
2. Generated code includes tool stubs that make HTTP POST requests
3. Tool server receives requests, executes actual tool functions
4. Results are returned to container via HTTP response
5. On macOS/Windows: uses `host.docker.internal`
6. On Linux: uses Docker bridge network gateway

### Experimental Status

This feature is marked as **@experimental** because:
- API may change based on user feedback
- Security model is being refined
- Performance optimizations are ongoing

## Testing Plan

### Unit Tests

```bash
pytest tests/unittests/agents/test_coding_agent.py -v
pytest tests/unittests/code_executors/test_allowlist_validator.py -v
pytest tests/unittests/code_executors/test_tool_code_generator.py -v
```

### Manual E2E Tests

**Test 1: Basic Query**
```
Query: "What is 25 times 17?"
Expected: Agent generates code, calculates, returns "425"
```

**Test 2: Data Analysis**
```
Query: "What is the survival rate on the Titanic?"
Expected: Agent fetches data, analyzes with pandas, returns "38.38%"
```

**Test 3: Visualization**
```
Query: "Create a bar chart of Titanic survival by class"
Expected: Agent creates chart, saves to /tmp/adk_charts/, reports filepath
```

**Test 4: Multi-step Analysis**
```
Query: "Analyze Titanic data: show stats, survival by sex/class, and provide 3 insights"
Expected: Agent performs multiple steps, creates visualization, provides comprehensive answer
```

## Additional Context

### Related Work
- OpenAI Code Interpreter
- Anthropic Computer Use
- Google AI Studio code execution

### Dependencies
- Docker (required for sandboxed execution)
- ContainerCodeExecutor (existing ADK component)

### Future Enhancements
- [ ] Support for stateful execution (persist variables across turns)
- [ ] Custom container images with pre-installed packages
- [ ] Integration with VertexAI code execution
- [ ] Support for additional languages (JavaScript, etc.)

### Screenshots

**Data Analysis Agent in action:**
```
User: Create a bar chart showing survival rate by passenger class from Titanic

Agent: [Installs packages, fetches data, creates visualization]
Response: The bar chart has been saved to /tmp/adk_charts/survival_by_class.png
- 1st Class: 63% survival rate
- 2nd Class: 47% survival rate
- 3rd Class: 24% survival rate
```
