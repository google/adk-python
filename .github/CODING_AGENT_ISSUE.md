# GitHub Issue: CodingAgent Feature Request

**Use this content to create an issue at: https://github.com/google/adk-python/issues/new?template=feature_request.md**

---

## Title

`feat(agents): Add CodingAgent for code generation and sandboxed execution`

---

## Is your feature request related to a problem? Please describe.

Currently, ADK agents can only interact with the world through pre-defined tools. While powerful, this approach has limitations:

1. **Limited flexibility**: Users must anticipate all possible operations and create tools for each
2. **No computational capability**: Agents cannot perform complex calculations, data analysis, or create visualizations without custom tools
3. **No iteration**: Standard tool-calling doesn't easily support multi-step reasoning with intermediate computations
4. **Competitive gap**: Other platforms (OpenAI Code Interpreter, Anthropic's computer use) offer code execution capabilities

**User pain points:**
- "I want my agent to analyze a CSV file and create a chart" - requires building custom tools
- "I need multi-step calculations with intermediate results" - awkward with standard tools
- "I want the agent to figure out HOW to solve a problem, not just call predefined functions"

---

## Describe the solution you'd like

A new experimental agent type called **CodingAgent** that:

1. Receives a task from the user
2. Generates Python code to accomplish the task (using `tool_code` blocks)
3. Executes the code in a sandboxed Docker container
4. Processes results and either provides an answer or continues iterating (ReAct loop)
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

### Key Components

| Component | Description |
|-----------|-------------|
| CodingAgent | Main agent class with ReAct loop |
| CodingAgentCodeExecutor | Wrapper that injects tool stubs into code |
| ToolCodeGenerator | Generates Python function stubs for tools |
| ToolExecutionServer | HTTP server for tool IPC from container |
| AllowlistValidator | Import security validation |

### Security Features

1. **Sandboxed execution**: All code runs in isolated Docker containers
2. **Import allowlisting**: Only authorized imports are permitted (configurable)
3. **Tool isolation**: Tools execute on host via HTTP, not in container
4. **No filesystem access**: Container has no access to host filesystem

---

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

---

## Additional context

### Implementation Status

I have a working implementation ready for PR submission:

**New files (~2,500 lines of production code):**
- `src/google/adk/agents/coding_agent.py` - Main agent class
- `src/google/adk/agents/coding_agent_config.py` - Configuration
- `src/google/adk/code_executors/coding_agent_code_executor.py` - Executor wrapper
- `src/google/adk/code_executors/tool_code_generator.py` - Code generation
- `src/google/adk/code_executors/tool_execution_server.py` - HTTP IPC server
- `src/google/adk/code_executors/allowlist_validator.py` - Security validation

**Sample agent:**
- `contributing/samples/coding_agent/` - Data Analysis Agent demo

**Unit tests (~950 lines):**
- `tests/unittests/agents/test_coding_agent.py`
- `tests/unittests/code_executors/test_allowlist_validator.py`
- `tests/unittests/code_executors/test_tool_code_generator.py`

### Tested Scenarios

| Test | Status |
|------|--------|
| Basic math queries | ✅ Passed |
| Data analysis with pandas | ✅ Passed |
| Visualization with matplotlib | ✅ Passed |
| Multi-step analysis | ✅ Passed |
| Tool calling via HTTP IPC | ✅ Passed |
| Chart saving to host system | ✅ Passed |
| Error handling and retries | ✅ Passed |

### Related Work
- OpenAI Code Interpreter
- Anthropic Computer Use
- Google AI Studio code execution

### Future Enhancements (out of scope for initial PR)
- Stateful execution (persist variables across turns)
- Custom container images with pre-installed packages
- Integration with VertexAI code execution
- Support for additional languages

---

## Labels to add

- `enhancement`
- `agents`
- `new-feature`
